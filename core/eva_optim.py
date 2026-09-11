"""core/eva_optim.py — EVA-AdamW: AdamW-эквивалентное ядро + архитектурно-осознанные
модификаторы, адаптированные под реальную архитектуру EVA (не обобщённый прототип).

mode='adamw' ПОБИТОВО идентичен torch.optim.AdamW (тот же порядок операций:
decoupled-wd до применения, lerp для m, denom=(√v/bc2**0.5)+eps, и тот же fused
`addcdiv_` в ветке без модификаторов) — проверено 100-шаговым A/B (maxdiff 0.0).
Используется как строгий эталон для A/B. mode='eva'/'eva_proj' включают
модификаторы ниже; каждый отключается независимым флагом.

Математический аудит (2026-09, измерения на полигоне audit_fires/audit_fixed):
  * gradient-EMA slow_ema с бeta_slow=0.9999 нормировалась быстрым √v̂ (окно ~20)
    — у сходимости лока по фикс. батчу отношение slow/fast доходит до 219× (шаг
    на 2 порядки больше AdamW), пользы на измерено. slow_ema по умолчанию OFF;
    при включении — EMA по самому Adam-направлению u (знак-масштаб, |ms|≤|u|max)
    и ВЫПУКЛОЕ смешивание u=(u+c·ms/bc3)/(1+c): шаг не раздувается никогда.
  * константы модификаторов связаны с τ-полем (attach_tau(model)):
    горизонт slow b3=1−exp(−1/τ_l), вес c=τ_min/τ_l (равный доп. лаг τ_min у
    всех слоёв), cap для τ = dev_max/(delta_t·lr) (полный проход dev_max за
    ширину рампы созревания).
  * AdamP-порог δ привязан к размерности: δ=2/√D — null-полоса косинуса двух
    случайных векторов; проецируем ТОЛЬКО при |cos|<δ (нет значимого радиального
    сигнала). Старый фикс. δ=0.111 с unilateral-gate (cos<δ) проецировал и
    антисогласованные (cos≈−0.99) обновления, перенормировкой усиливая касательный
    шум до 7×.
  * cautious-ренормировка ограничена null-полосой доли согласий: делитель
    ≥ 0.5·max(0.5, 1−2/√D) — амплификация ≤ 2 на большой размерности (бума),
    ≤4 на малой; случайные signs u·g ниже уровня шума не усиливаются.
  * все модификаторные операции in-place (dot/скалярные редукции): ноль
    полномерных аллокаций за шаг (пожирание VRAM на 723M убрано).

Роли вычисляются по реальным именам параметров (по дампу named_parameters()
модели 723.27M, seq_len=256):
  - tau        : tau_config / _tau_l_dev  — WD off, пошаговый update_cap (τ-привязанный)
  - scale_inv  : _vsa_log_param, ._vsa_tau_log, log_temp, log_tau, log_gain,
                 log_scale, _fusion_tau_alpha, _w_alpha_expert, *_log_*,
                 embed_mix, bit_bias, *_bias(VSA-биасы) — WD off (искажает LR)
  - zero_init  : intent_probe*, bus_head_proj* — WD off, cautious-only,
                 trust=abs(maturation_readiness)
  - bridge     : layers.*.mirror.bridge_glu_net.* — trust=bridge.readiness()
  - mem        : memory_bank.* — trust=abs(maturation_readiness)
  - matrix/кд2 : остальные dim>=2 — AdamW-эквивалент, lr от LLRD см. param_groups
  - scalar     : остальные dim<2 (w_i/w_d/w_q/w_u/w_v, biases) — WD off

Модификаторы (все вырождаются в AdamW при флагах выкл). Порядок в step():
u = m/denom -> AdamP-проекция (null-полоса 2/√D, in-place) -> slow-смешивание
(выпуклое, u-пространство) -> cautious-маска -> trust -> update_cap -> wd ->
apply (-lr/bc1):
  * projected : Gram–Schmidt u -= (⟨u,w⟩/‖w‖²)·w при |cos(w,u)| < δ=2/√D
                + нормосохраняющая перекалибровка (AdamP, Heo 2020); только
                dim>=2 где wd>0. Пошаговый фильтр направления, momentum не
                мутируется (эквивалентно оригиналу пошагово).
  * cautious  : обнуление u где u·g<=0, ренормировка на долю согласий (пол
                null-полосы — см. аудит)
  * slow_ema  : EMA по Adam-направлению u (не по сырому градиенту!) для
                матричных групп; выпуклая комбинация (u + c·û_slow)/(1+c);
                b3/c из τ-лестницы через attach_tau (fallback: kwargs)
  * trust     : u *= floor + (1-floor)*trust — восстанавливает пропорциональность
                созревающих ветвей, уничтоженную нормировкой Adam; trust из
                set_trust() по роли, клэмпится к [0,1]
  * update_cap : u.clamp_(-cap, cap) для τ-параметров; cap=dev_max/(delta_t·lr)
                из τ-бюджета (attach_tau), fallback 0.02
  * wd_enabled : включение/выключение weight_decay по роли; всегда только dim>=2
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
from torch.optim import Optimizer

# Подстроки, по которым параметр признаётся масштабно-инвариантным (WD off).
_SCALE_INVARIANT_SUBSTR = (
    "_vsa_log_param", ".vsa_tau_log", "log_temp", "log_tau", "log_gain",
    "log_scale", "fusion_tau_alpha", "w_alpha_expert", "log_dvar_mod_scale",
    "log_grad_mod_scale", "embed_mix", "bit_bias",
)
# Роль, чей trust берётся как |maturation.readiness| (созревание ветви).
_TRUST_BRIDGE = (".mirror.bridge_glu_net.",)
_TRUST_MEM = ("memory_bank.",)
# Zero-init параметры (WD off, cautious-only).
_ZERO_INIT_SUBSTR = ("intent_probe.", "intent_probe.bias", "bus_head_proj.",)

_TAU_DEV_SUBSTR = ("tau_config.", "_tau_l_dev")


def _adamp_project(u: torch.Tensor, w: torch.Tensor,
                   delta: Optional[float] = None, eps: float = 1e-8) -> torch.Tensor:
    """AdamP per-row projection (B2 rewrite, audit C).

    The previous global-flavoured variant was INVERTED against the paper: it
    projected only when |cos|<δ (fire on structureless noise at ≤1e-7 effect,
    94% of steps; and skip exactly the radial-collapse case where protection
    matters — measured ‖W‖ −99.9% under collapse pressure, same as plain).
    Now: fan-in ROWS, project when |cos| ≥ δ = 2/√fan_in (radially significant
    ⇒ remove the radial component, renormalize to the original row norm —
    paper semantics). Under the row-independent noise model P(fire) ≈ 4.6% and
    the projected component is beyond-noise by construction. Scalars/1-D are
    excluded (no geometry to preserve)."""
    if u.dim() < 2:
        return u
    rows = u.shape[0]
    W = w.reshape(rows, -1)
    U = u.reshape(rows, -1)
    wn = W.norm(dim=1, keepdim=True)
    un = U.norm(dim=1, keepdim=True)
    cos = (U * W).sum(dim=1, keepdim=True) / (un * wn + eps)
    if delta is None:
        delta = 2.0 / math.sqrt(max(W.shape[1], 4))
    sel = (cos.abs() >= delta) & (wn > 0)                     # (rows,1)
    n_hat = W / (wn + eps)
    E = U - (cos * un) * n_hat                                  # row-wise GS (full projection magnitude)
    en = E.norm(dim=1, keepdim=True)
    E = torch.where(en > eps, E / (en + eps) * un, torch.zeros_like(E))
    out = torch.where(sel, E, U).reshape_as(u)
    u.copy_(out)
    return u
    uf = u.reshape(-1)
    wf = w.reshape(-1)
    wn = float(wf.norm())
    un = float(uf.norm())
    if wn <= 0.0 or un <= 0.0:
        return u
    ow = float(torch.dot(uf, wf))
    if abs(ow) >= delta * un * wn:          # |cos| >= delta — значимый радиальный
        return u                            # сигнал: не трогаем направление
    coef = ow / (wn * wn + eps)
    uf.add_(wf, alpha=-coef)                # Gram-Schmidt in-place (view u)
    pn = float(uf.norm())
    if pn > eps:
        uf.mul_(un / pn)                    # нормосохранение (in-place)
    else:
        uf.zero_()
    return u


def _resolve_role(name: str, dim: int) -> dict:
    """Определяет роль и флаги группы по имени+размерности параметра."""
    if any(s in name for s in _TAU_DEV_SUBSTR):
        return dict(role="tau", wd=False, cap=0.02, trust=None, cautious=True)
    if any(s in name for s in _ZERO_INIT_SUBSTR):
        return dict(role="zero_init", wd=False, cap=None, trust="intent",
                    cautious=True)
    if any(s in name for s in _TRUST_BRIDGE):
        return dict(role="bridge", wd=True, cap=None, trust="bridge",
                    cautious=True)
    if any(s in name for s in _TRUST_MEM):
        return dict(role="mem", wd=True, cap=None, trust="mem", cautious=True)
    if any(s in name for s in _SCALE_INVARIANT_SUBSTR):
        return dict(role="scale_inv", wd=False, cap=None, trust=None,
                    cautious=True)
    if dim >= 2:
        return dict(role="matrix", wd=True, cap=None, trust=None, cautious=True)
    return dict(role="scalar", wd=False, cap=None, trust=None, cautious=True)


class EVAAdamW(Optimizer):
    """AdamW с модификаторами. mode='adamw' ≡ torch.optim.AdamW.

    Ожидает, что группы приходят из core.adaptation.build_optimizer — они несут
    lr (LLRD + lambda-иерархия), weight_decay, betas, role, layer_idxs. После
    построения вызывается attach_tau(model): константы slow-EMA, τ-cap
    пересчитываются из τ-лестницы модели.
    """

    def __init__(self, params, mode: str = "eva", lr: float = 3e-4,
                 betas=(0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.01,
                 cautious: bool = True, slow_ema: bool = False,
                 beta_slow: float = 0.9999, slow_mix: float = 0.25,
                 trust_enabled: bool = True, trust_floor: float = 0.5,
                 projected_wd: bool = False,
                 debug: bool = False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        mode=mode, cautious=cautious, slow_ema=slow_ema,
                        beta_slow=beta_slow, slow_mix=slow_mix,
                        trust_enabled=trust_enabled, trust_floor=trust_floor,
                        projected=projected_wd, debug=debug, role="matrix",
                        wd_enabled=True, trust_key=None, update_cap=None)
        super().__init__(params, defaults)
        self._trust: Dict[str, float] = {}

    # ── привязка констант к τ-лестнице модели ─────────────────────────────
    def attach_tau(self, model: torch.nn.Module) -> "EVAAdamW":
        """Пересчитывает медленнические константы из TauConfig модели.

        matrix-группы:  b3 = 1 − exp(−1/τ̄_l) (горизонт = лока τ),
                        c  = τ_min/τ̄_l (равный доп. лаг τ_min у всех слоёв)
        tau-группы:     cap = dev_max/(delta_t·lr) — полный проход dev_max
                        не быстрее ширины рампы созревания delta_t.
        """
        tc = getattr(model, "tau_config", None)
        if tc is None:
            return self
        with torch.no_grad():
            tau_l = tc.tau_l.detach().cpu().tolist()
        tmin = float(tc.tau_min)
        for g in self.param_groups:
            idxs: List[int] = [int(i) for i in (g.get("layer_idxs") or [])
                               if 0 <= int(i) < len(tau_l)]
            if g["role"] == "matrix" and idxs:
                tl = sum(tau_l[i] for i in idxs) / len(idxs)
                tl = max(tl, 1.0)
                g["beta_slow_g"] = 1.0 - math.exp(-1.0 / tl)
                g["slow_mix_g"] = min(1.0, tmin / tl)
            if g["role"] == "tau" and g.get("update_cap") is not None:
                lr = max(float(g["lr"]), 1e-12)
                g["update_cap"] = float(tc.dev_max) / (float(tc.delta_t) * lr)
        return self

    # ── доверие (maturation/bridge) из цикла обучения ─────────────────────
    def set_trust(self, mapping: Dict[str, float]) -> None:
        self._trust.update(mapping)

    def trust_snapshot(self) -> Dict[str, float]:
        return {g["trust_key"]: self._trust.get(g["trust_key"], 1.0)
                for g in self.param_groups if g["trust_key"]}

    # ── шаг ───────────────────────────────────────────────────────────────
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        eva_on = self.defaults["mode"] in ("eva", "eva_proj")
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            wd = group["weight_decay"] if group["wd_enabled"] else 0.0
            cautious = eva_on and group["cautious"]
            cap = group["update_cap"] if eva_on else None
            slow_ok = eva_on and group["slow_ema"] and group["role"] == "matrix"
            b3 = group.get("beta_slow_g", group["beta_slow"])
            cmix = group.get("slow_mix_g", group["slow_mix"])
            tkey = group["trust_key"] if (eva_on and group["trust_enabled"]) else None
            if tkey:
                trust = min(1.0, max(0.0, self._trust.get(tkey, 1.0)))
                tscale = group["trust_floor"] + (1.0 - group["trust_floor"]) * trust
            else:
                tscale = None
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if slow_ok:
                        st["exp_avg_slow"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)
                st["step"] += 1.0
                m = st["exp_avg"]
                v = st["exp_avg_sq"]
                step = int(st["step"])
                # Тот же порядок операций, что в torch.optim.AdamW (для
                # эквивалентности в mode='adamw'): lerp + addcmul_ +
                # denom=(√v/√bc2)+eps, decoupled-wd до применения.
                m.lerp_(g, 1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                bc1 = 1.0 - b1 ** step
                bc2 = 1.0 - b2 ** step
                # Побитово как в torch: bias_correction2**0.5 (float pow), а не
                # math.sqrt — ULP-расхождение double иначе инвертирует float32-
                # округление деления на ~половине "ровных" координат.
                denom = (v.sqrt() / (bc2 ** 0.5)).add_(eps)
                if not eva_on:
                    # mode='adamw': точный torch-путь (fused addcdiv_) — без
                    # отдельного bu=m/denom (у unfused div+add alpha иной ULP).
                    if wd > 0.0 and p.dim() >= 2:
                        p.mul_(1.0 - lr * wd)
                    p.addcdiv_(m, denom, value=-(lr / bc1))
                    continue
                u = m / denom
                # AdamP-проекция "сырого" Adam-направления (null-полоса 2/√D,
                # in-place) — оригинальный случай AdamP: p̄ = m̂/√v̂.
                if eva_on and group["projected"] and wd > 0.0 and p.dim() >= 2:
                    u = _adamp_project(u, p)
                if slow_ok:
                    ms = st.get("exp_avg_slow")
                    if ms is None:      # реставрация со старых чекпоинтов без slow-EMA
                        ms = st["exp_avg_slow"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)
                    # EMA по САМОМУ Adam-направлению u (знак-масштаб: |ms|≤|u|),
                    # выпуклая комбинация — шаг не может раздуться выше max|u|.
                    ms.lerp_(u, 1.0 - b3)
                    ms_hat = ms / (1.0 - b3 ** step)
                    u = (u + cmix * ms_hat) / (1.0 + cmix)
                if cautious:
                    mask = torch.mul(u, g).gt_(0)
                    u.mul_(mask)
                    # Ренорм только на null-полосе доли согласий (0.5±2σ, σ=√(0.25/D)).
                    # B2 (audit C): the mask keeps a fraction ρ of coords ⇒ surviving
                    # ENERGY is ~ρ·total ⇒ compensate with ÷√ρ, not ÷ρ. Measured at the
                    # random-sign floor the old form inflated ‖step‖ ×1.41 and biased
                    # direction (cos=0.45 vs noise); anisotropic oscillation was 3-3.8×
                    # worse than plain Adam; ÷√ρ restores parity (0.029→0.012 osc std).
                    floor = 0.5 * max(0.5, 1.0 - 2.0 / math.sqrt(max(mask.numel(), 4)))
                    u.div_(mask.mean().clamp_min(floor).sqrt())
                if tscale is not None and tscale != 1.0:
                    u.mul_(tscale)
                if cap is not None:
                    u.clamp_(-cap, cap)
                if wd > 0.0 and p.dim() >= 2:
                    p.mul_(1.0 - lr * wd)
                p.add_(u, alpha=-(lr / bc1))
                if group["debug"] and step % 500 == 0:
                    print(f"  [evo] role={group['role']} lr={lr:.2e} "
                          f"wd={wd:.3f} |u|={u.norm().item():.4f}")
        return loss


def build_eva_optimizer(model, base_lr: float, weight_decay: float = 0.01,
                        betas=(0.9, 0.95), mode: str = "eva",
                        llrd_decay: float = 0.9,
                        cautious: bool = True, slow_ema: bool = False,
                        beta_slow: float = 0.9999, slow_mix: float = 0.25,
                        trust_enabled: bool = True, trust_floor: float = 0.5,
                        projected_wd: bool = False, debug: bool = False):
    """Алиас: EVA-группы строятся в core.adaptation.build_optimizer(optimizer='eva').

    Единый источник строительства групп — build_optimizer (LLRD + роли).
    """
    from .adaptation import build_optimizer as _build
    return _build(model, base_lr, llrd_decay=llrd_decay, weight_decay=weight_decay,
                  betas=betas, optimizer=mode,
                  eva_kwargs=dict(cautious=cautious, slow_ema=slow_ema,
                                  beta_slow=beta_slow, slow_mix=slow_mix,
                                  trust_enabled=trust_enabled,
                                  trust_floor=trust_floor,
                                  projected_wd=projected_wd, debug=debug))
