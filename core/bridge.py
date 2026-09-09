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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticBridge(nn.Module):
    """Per-layer semantic bridge for cross-layer communication.

    Emits semantic vectors at each layer, maintains a persistent cross-layer
    stream (EMA), and injects neighbour semantics back into hidden states.
    Trained via self-supervised next-token embedding prediction.

    Args:
        D: Hidden dimension.
        n_layers: Number of transformer layers.
        bridge_dim: Dimension of the semantic projection space.
        depth: Whether to use cross-layer depth flow.
        cfg: Configuration object (optional, for maturation bridge params).
    """

    def __init__(
        self,
        D: int,
        n_layers: int,
        bridge_dim: int = 256,
        depth: bool = True,
        cfg: Optional[object] = None,
    ) -> None:
        super().__init__()
        self.D: int = D
        self.n_layers: int = n_layers
        self.bridge_dim: int = bridge_dim
        self.depth: bool = depth
        # ─── Readiness по компетентности bridge (замена слепой time-рампе) ───
        self._br_r0: float = float(getattr(cfg, 'matur_bridge_r0', 0.3))
        self._br_rs: float = float(getattr(cfg, 'matur_bridge_rs', 0.2))
        self.register_buffer('bridge_loss_init', torch.tensor(1.0), persistent=False)
        self.register_buffer('bridge_loss_ema', torch.tensor(1.0), persistent=False)

        self.probe: nn.Sequential = nn.Sequential(
            nn.Linear(D, bridge_dim),
            nn.GELU(),
            nn.Linear(bridge_dim, bridge_dim),
        )
        self.emb_proj: nn.Linear = nn.Linear(D, bridge_dim)
        self.stream_proj: nn.Linear = nn.Linear(bridge_dim, D)
        self.stream_log_scale: nn.Parameter = nn.Parameter(torch.zeros(1))
        self._inj_alpha: nn.Parameter = nn.Parameter(torch.tensor(1.0))
        self._inj_beta: nn.Parameter = nn.Parameter(torch.tensor(0.5))
        self.stream_log_weights: nn.Parameter = nn.Parameter(torch.zeros(3))

        self.register_buffer(
            "bridge_stream", torch.zeros(n_layers, bridge_dim), persistent=True
        )
        # Per-forward injection share ‖inj‖/‖h‖ — LIVE feature 3 of the
        # LayerBridgeGate diagnostics (was pinned to a 0.5 constant in the
        # stack's inline duplicate, audit M2).
        self.register_buffer("inj_ratio", torch.zeros(n_layers), persistent=False)
        self._preds: Optional[list[torch.Tensor]] = None

    @torch.no_grad()
    def readiness(self) -> torch.Tensor:
        """Скалярная готовность в [0,1] по компетентности bridge.

        sat = 1 - ema_loss / init_loss  (насколько косинус-лосс bridge упал
        относительно случайного базиса); readiness = sigmoid((sat - r0)/rs)
        минус базовое значение при sat=0, чтобы ровно 0 при отсутствии обучения.
        Возвращает detached scalar-тензор (буферы вне графа)."""
        init: torch.Tensor = self.bridge_loss_init.clamp(min=1e-3)
        sat: torch.Tensor = (1.0 - self.bridge_loss_ema / init).clamp(0.0, 1.0)
        base: torch.Tensor = torch.sigmoid(torch.tensor(-self._br_r0 / self._br_rs))
        return (torch.sigmoid((sat - self._br_r0) / self._br_rs) - base).clamp(0.0, 1.0)

    def start_forward(self) -> None:
        """Reset per-forward prediction list."""
        self._preds = []

    def probe_layer(self, h_l: torch.Tensor) -> torch.Tensor:
        """Emit the semantic vector for a layer's hidden state -> (B, L, bridge_dim)."""
        return self.probe(h_l)

    def inject_layer(
        self,
        i: int,
        h_l: torch.Tensor,
        maturity: Optional[torch.Tensor] = None,
        tau_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add the cross-layer semantic stream signal to a layer's hidden state.

        Args:
            i: Layer index.
            h_l: Hidden state tensor (B, L, D).
            maturity: Scalar tensor scaling injection by layer maturation.
            tau_norm: Scalar tensor for τ-normalized injection coupling (U4).

        Returns:
            Updated hidden state (B, L, D).
        """
        if not self.depth or self.n_layers == 0:
            return h_l
        neigh: list[torch.Tensor] = [self.bridge_stream[i]]
        if i - 1 >= 0:
            neigh.append(self.bridge_stream[i - 1])
        if i + 1 < self.n_layers:
            neigh.append(self.bridge_stream[i + 1])
        stack: torch.Tensor = torch.stack(neigh, 0)
        sw: torch.Tensor = torch.sigmoid(self.stream_log_weights[:stack.shape[0]])
        w: torch.Tensor = sw / sw.sum().clamp(min=1e-6)
        combined: torch.Tensor = (stack * w.unsqueeze(-1)).sum(0)
        if tau_norm is not None:
            inj_strength: torch.Tensor = (
                torch.sigmoid(self._inj_alpha) * tau_norm
                + torch.sigmoid(self._inj_beta) * (1.0 - tau_norm)
            )
        else:
            inj_strength = torch.ones(1, device=h_l.device, dtype=h_l.dtype)
        scale: torch.Tensor = torch.tanh(self.stream_log_scale)
        # Audit M2: maturity was applied TWICE here (before and after
        # inj_strength), so injection scaled as M² instead of the designed
        # linear M-coupling (U4 / README §18) — at mat≈0.3 the branch was
        # injected ~3.3x weaker than intended through the whole wake-up.
        if maturity is not None:
            scale = scale * maturity
        scale = scale * inj_strength
        inj: torch.Tensor = scale * self.stream_proj(combined)
        with torch.no_grad():
            r = inj.detach().norm() / (h_l.detach().norm() + 1e-8)
            self.inj_ratio[i] = r.clamp(0.0, 1.0)
        return h_l + inj.view(1, 1, self.D)

    @torch.no_grad()
    def update_stream(self, i: int, s_l: torch.Tensor) -> None:
        """EMA-update the persistent stream from this layer's semantic vector."""
        m: torch.Tensor = s_l.detach().float().mean(dim=(0, 1))
        self.bridge_stream[i].mul_(0.9).add_(m, alpha=0.1)

    def record(self, s_l: torch.Tensor) -> None:
        """Record a layer's semantic vector for the auxiliary loss."""
        if self._preds is not None:
            self._preds.append(s_l)

    @torch.no_grad()
    def reset_stream(self) -> None:
        """Zero out the persistent bridge stream."""
        self.bridge_stream.zero_()

    def loss(
        self, y: torch.Tensor, embed_fn: callable
    ) -> Optional[torch.Tensor]:
        """Self-supervised bridge loss: each layer predicts the next token embedding.

        Returns the mean over layers of ``1 - cos(s_l[:, :-1], emb_proj(embed(y[:,1:])))``.
        Returns ``None`` if no predictions were recorded this forward.
        """
        if self._preds is None or len(self._preds) == 0:
            return None
        emb: torch.Tensor = embed_fn(y[:, 1:])
        tgt: torch.Tensor = self.emb_proj(emb)
        total: torch.Tensor = torch.zeros((), device=tgt.device, dtype=tgt.dtype)
        n: int = 0
        layer_means: list[torch.Tensor] = []
        for s_l in self._preds:
            pred: torch.Tensor = s_l[:, :-1]
            if pred.shape[1] != tgt.shape[1]:
                m: int = min(pred.shape[1], tgt.shape[1])
                pred = pred[:, :m]
                tgt_ = tgt[:, :m]
            else:
                tgt_ = tgt
            total = total + (1.0 - F.cosine_similarity(pred, tgt_, dim=-1, eps=1e-8).mean())
            n += 1
            layer_means.append(pred.mean(dim=(0, 1)))
        loss_val: torch.Tensor = total / max(n, 1)
        if len(layer_means) >= 2:
            stacked: torch.Tensor = F.normalize(torch.stack(layer_means), dim=-1)
            sim: torch.Tensor = stacked @ stacked.T
            mask: torch.Tensor = ~torch.eye(len(layer_means), dtype=torch.bool, device=sim.device)
            diversity_penalty: torch.Tensor = sim[mask].mean()
            loss_val = loss_val + 0.1 * diversity_penalty
        with torch.no_grad():
            lv: torch.Tensor = loss_val.detach().float()
            self.bridge_loss_init.copy_(torch.maximum(self.bridge_loss_init, lv))
            self.bridge_loss_ema.lerp_(lv, 0.01)
        return loss_val
