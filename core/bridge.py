"""EVA: per-layer semantic bridge (in-pipeline, active in train AND inference).

Mirrors the Intent Bridge streaming pattern: a single shared probe head emits a
semantic vector ``s_l = probe(h_l)`` at every layer ``l`` (B, L, bridge_dim).
These vectors form a cross-layer stream that

  * flows through DEPTH  : each layer injects its own (carried from previous
    step) plus bottom-up (fresh, from already-processed lower layers) and
    top-down (carried) neighbour semantics back into its hidden state;
  * flows through TIME   : a persistent ``bridge_stream`` buffer (EMA of the
    per-layer semantic vectors) is carried across forward calls, giving the
    model memory of its own recent semantic structure;
  * is SELF-SUPERVISED   : at every layer the probe is trained to predict the
    next token's embedding (cosine loss), so the bridge head receives a dense,
    well-distributed gradient at each depth rather than only from a single
    external head.

The probe/injection run unconditionally in forward (train and inference). Only
the auxiliary loss is training-only. Because every parameter lives inside the
model, ``model.named_parameters()`` is complete (no ``StopIteration`` when
saving checkpoints) and the probe inherits the model's LR groups.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticBridge(nn.Module):
    def __init__(self, D: int, n_layers: int, bridge_dim: int = 256, depth: bool = True, cfg=None):
        super().__init__()
        self.D = D
        self.n_layers = n_layers
        self.bridge_dim = bridge_dim
        self.depth = depth
        # ─── Readiness по компетентности bridge (замена слепой time-рампе) ───
        # Bridge самообучается предсказывать ЭТАЛОННЫЙ next-token embedding
        # (косинус-лосс) независимо от LM-лосса ствола, поэтому его
        # компетентность растёт даже когда ствол у случайного базиса. Это даёт
        # сигнал готовности, который НЕ зацикливается (в отличие от pred_err
        # зеркала, который при масштабе не падает). maturity = max(time_ramp,
        # bridge_readiness): ветви открываются, как только bridge стал
        # компетентным, а не по слепым часам, и при этом у init закрыты
        # (bridge случаен => readiness=0 => стабильность сохранена).
        self._br_r0 = float(getattr(cfg, 'matur_bridge_r0', 0.3))
        self._br_rs = float(getattr(cfg, 'matur_bridge_rs', 0.2))
        # baseline случайного режима (running max косинус-лосса) и EMA лосса
        self.register_buffer('bridge_loss_init', torch.tensor(1.0), persistent=False)
        self.register_buffer('bridge_loss_ema', torch.tensor(1.0), persistent=False)

        # Shared per-layer probe head (one set of weights applied at every layer
        # to keep parameter count small and force a common semantic readout).
        self.probe = nn.Sequential(
            nn.Linear(D, bridge_dim),
            nn.GELU(),
            nn.Linear(bridge_dim, bridge_dim),
        )
        # Project the next-token embedding target into bridge space for the loss.
        self.emb_proj = nn.Linear(D, bridge_dim)
        # Project the (carried) stream back into hidden space for injection.
        self.stream_proj = nn.Linear(bridge_dim, D)
        # Injection strength. Initialised to 0 so the bridge starts as a no-op
        # (no disruption of an already-training run) and grows only if it helps.
        self.stream_log_scale = nn.Parameter(torch.zeros(1))
        # U4: τ-coupled injection: α, β learnable; injection = α·τ_norm + β·(1-τ_norm)
        self._inj_alpha = nn.Parameter(torch.tensor(1.0))
        self._inj_beta = nn.Parameter(torch.tensor(0.5))
        # Per-neighbour сигмоид-веса (i-1, i, i+1): нормированное среднее
        # соседей вместо сырой суммы (режим Б — выпуклая комбинация, лакуна).
        self.stream_log_weights = nn.Parameter(torch.zeros(3))

        # Persistent cross-layer stream: (n_layers, bridge_dim). EMA-updated,
        # not part of the autograd graph (detached when written).
        self.register_buffer(
            "bridge_stream", torch.zeros(n_layers, bridge_dim), persistent=True
        )
        self._preds: list[torch.Tensor] | None = None

    @torch.no_grad()
    def readiness(self) -> torch.Tensor:
        """Скалярная готовность в [0,1] по компетентности bridge.

        sat = 1 - ema_loss / init_loss  (насколько косинус-лосс bridge упал
        относительно случайного базиса); readiness = sigmoid((sat - r0)/rs)
        минус базовое значение при sat=0, чтобы ровно 0 при отсутствии обучения
        (bridge случаен => ствол не возмущается => стабильность обучения).
        Возвращает detached scalar-тензор (буферы вне графа)."""
        init = self.bridge_loss_init.clamp(min=1e-3)
        sat = (1.0 - self.bridge_loss_ema / init).clamp(0.0, 1.0)
        base = torch.sigmoid(torch.tensor(-self._br_r0 / self._br_rs))
        return (torch.sigmoid((sat - self._br_r0) / self._br_rs) - base).clamp(0.0, 1.0)

    # ------------------------------------------------------------------ #
    def start_forward(self) -> None:
        self._preds = []

    def probe_layer(self, h_l: torch.Tensor) -> torch.Tensor:
        """Emit the semantic vector for a layer's hidden state -> (B, L, bridge_dim)."""
        return self.probe(h_l)

    def inject_layer(self, i: int, h_l: torch.Tensor, maturity: torch.Tensor = None,
                     tau_norm: torch.Tensor = None) -> torch.Tensor:
        """Add the cross-layer semantic stream signal to a layer's hidden state.

        `maturity` (scalar tensor, optional) scales the injection by the layer's
        unified maturation gate: the bridge stays a near-no-op until the layer's
        experts have ripened, so its gradient cannot perturb the trunk early.
        `tau_norm` (scalar tensor, optional) — U4: τ-normalized value for the
        layer, injection_strength = α·τ_norm + (1-α)·τ_norm_mod.
        """
        if not self.depth or self.n_layers == 0:
            return h_l
        neigh = [self.bridge_stream[i]]
        if i - 1 >= 0:
            neigh.append(self.bridge_stream[i - 1])          # bottom-up (fresh)
        if i + 1 < self.n_layers:
            neigh.append(self.bridge_stream[i + 1])          # top-down (carried)
        stack = torch.stack(neigh, 0)                        # (n, bridge_dim)
        # Режим Б: нормированное сигмоид-среднее соседей (выпуклая комбинация,
        # сумма весов = 1) вместо сырой суммы -> сохраняет лакуну между слоями.
        sw = torch.sigmoid(self.stream_log_weights[:stack.shape[0]])
        w = sw / sw.sum().clamp(min=1e-6)
        combined = (stack * w.unsqueeze(-1)).sum(0)          # (bridge_dim,)
        # U4: τ-coupled injection strength
        if tau_norm is not None:
            inj_strength = torch.sigmoid(self._inj_alpha) * tau_norm + \
                           torch.sigmoid(self._inj_beta) * (1.0 - tau_norm)
        else:
            inj_strength = torch.ones(1, device=h_l.device, dtype=h_l.dtype)
        scale = torch.tanh(self.stream_log_scale)           # bounded ∈ (-1, 1)
        if maturity is not None:
            scale = scale * maturity
        scale = scale * inj_strength
        if maturity is not None:
            scale = scale * maturity
        inj = scale * self.stream_proj(combined)             # (D,) — bounded сигнал
        return h_l + inj.view(1, 1, self.D)

    @torch.no_grad()
    def update_stream(self, i: int, s_l: torch.Tensor) -> None:
        """EMA-update the persistent stream from this layer's semantic vector."""
        m = s_l.detach().float().mean(dim=(0, 1))            # (bridge_dim,)
        self.bridge_stream[i].mul_(0.9).add_(m, alpha=0.1)

    def record(self, s_l: torch.Tensor) -> None:
        if self._preds is not None:
            self._preds.append(s_l)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def reset_stream(self) -> None:
        self.bridge_stream.zero_()

    def loss(self, y: torch.Tensor, embed_fn) -> torch.Tensor | None:
        """Self-supervised bridge loss: each layer predicts the next token embedding.

        Returns the mean over layers of ``1 - cos(s_l[:, :-1], emb_proj(embed(y[:,1:])))``.
        The caller multiplies by ``cfg.bridge_conn``. Returns ``None`` if no
        predictions were recorded this forward (e.g. inference without targets).
        """
        if self._preds is None or len(self._preds) == 0:
            return None
        emb = embed_fn(y[:, 1:])                              # (B, L-1, D)
        tgt = self.emb_proj(emb)                              # (B, L-1, bridge_dim)
        total = torch.zeros((), device=tgt.device, dtype=tgt.dtype)
        n = 0
        layer_means = []
        for s_l in self._preds:
            pred = s_l[:, :-1]                               # align to positions 0..L-2
            if pred.shape[1] != tgt.shape[1]:
                m = min(pred.shape[1], tgt.shape[1])
                pred = pred[:, :m]
                tgt_ = tgt[:, :m]
            else:
                tgt_ = tgt
            total = total + (1.0 - F.cosine_similarity(pred, tgt_, dim=-1, eps=1e-8).mean())
            n += 1
            layer_means.append(pred.mean(dim=(0, 1)))       # (bridge_dim,)
        loss_val = total / max(n, 1)
        # Inter-layer diversity: penalize collapsed probes (all layers same prediction)
        if len(layer_means) >= 2:
            stacked = F.normalize(torch.stack(layer_means), dim=-1)  # (n_layers, bridge_dim)
            sim = stacked @ stacked.T                                # (n_layers, n_layers)
            mask = ~torch.eye(len(layer_means), dtype=torch.bool, device=sim.device)
            diversity_penalty = sim[mask].mean()                     # mean off-diagonal similarity
            loss_val = loss_val + 0.1 * diversity_penalty
        # EMA косинус-лосса + baseline случайного режима для readiness().
        # Под no_grad: буферы вне графа, градиент по лоссу (probe/stream_proj)
        # сохраняется — он течёт через возвращаемый loss_val.
        with torch.no_grad():
            lv = loss_val.detach().float()
            self.bridge_loss_init.copy_(torch.maximum(self.bridge_loss_init, lv))
            self.bridge_loss_ema.lerp_(lv, 0.01)
        return loss_val
