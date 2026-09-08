"""core/eva_optim.py — EVA-AdamW: AdamW-эквивалентное ядро + архитектурно-осознанные
модификаторы, адаптированные под реальную архитектуру EVA (не обобщённый прототип).

mode='adamw' по-операторно эквивалентен torch.optim.AdamW (те же уравнения и
порядок операций; расхождения ≤ ~1e-7 ULP из-за foreach-ядра torch) —
используется как строгий эталон для A/B. mode='eva' включает модификаторы
ниже; каждый отключается независимым флагом.

Роли вычисляются по реальным именам параметров (по дампу named_parameters()
модели 723.27M, seq_len=256):
  - tau        : tau_config / _tau_l_dev  — WD off, пошаговый update_cap
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
u = m/denom -> AdamP-проекция (проектирование "сырого" Adam-направления,
как в оригинальном AdamP: p̄ = m̂/√v̂ до коррекции bc1) -> slow-смешивание ->
cautious-маска -> trust -> update_cap -> wd -> apply (-lr/bc1):
  * projected : Gram–Schmidt: u' = u - (⟨u,w⟩/‖w‖²)·w при cos(w,u) < δ (0.111)
                + нормосохраняющая перекалибровка ‖u'‖=‖u‖ (AdamP, Heo 2020);
                только для dim>=2 где wd>0. Не мутирует momentum (пер-шаговый
                фильтр; эквивалентно оригиналу пошагово).
  * cautious  : обнуление u там, где u*g<=0, с нормировкой на среднюю маску
  * slow_ema  : третья EMA (beta_slow=0.9999) только для матричных параметров;
                терм ms/bc3, Ada-нормирован на v̂^0.5 и смешан slow_mix; к-т
                net = lr·slow_mix (коррекция bc1 вынесена в self.step)
  * trust     : u *= floor + (1-floor)*trust — восстанавливает пропорциональность
                созревающих ветвей, уничтоженную нормировкой Adam; trust из
                set_trust() по роли, клэмпится к [0,1]
  * update_cap : u.clamp_(-cap, cap) для τ-параметров (trust-region геометрии)
  * wd_enabled : включение/выключение weight_decay по роли; всегда только dim>=2
"""
from __future__ import annotations

import math
from typing import Dict, Optional

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

# AdamP порог косинуса: проецируем только если u почти ортогонален w
# (cos < 0.111 => угол > ~83.6°, и ‖u_⊥‖ ≈ ‖u‖·sinθ ≥ 0.99·‖u‖ — нет вырожденности).
_ADAMP_DELTA = 0.111


def _adamp_project(u: torch.Tensor, w: torch.Tensor,
                   delta: float = _ADAMP_DELTA, eps: float = 1e-8) -> torch.Tensor:
    """AdamP-фильтр на шаге Adam.

    u — "сырое" Adam-направление m/denom (как в оригинале: до коррекции bc1);
    w — текущий вес. При cos(w,u) < delta вычитается компонента вдоль w
    (Gram–Schmidt), затем нормосохраняющая перекалибровка ‖u'‖=‖u‖.
    Проекция нормо-сохраняющая, поэтому bc1 при применении не влияет на геометрию.
    """
    uf = u.flatten()
    wf = w.detach().flatten()
    wn = wf.norm()
    un = uf.norm()
    if wn == 0.0 or un == 0.0:
        return u
    ow = (uf * wf).sum().clamp(-wn * un, wn * un)
    cos = float(ow / (wn * un + eps))
    if cos >= delta:
        return u
    proj = uf - (float(ow) / (wn * wn + eps)) * wf
    pn = proj.norm()
    if pn > eps:
        proj = proj * (un / pn)
    else:
        proj = uf.new_zeros(uf.shape)
    return proj.view_as(u)


def _resolve_role(name: str, dim: int) -> dict:
    """Определяет роль и флаги группы по имени+дородности параметра."""
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

    Ожидает, что группы приходят из stack.EVAStack.param_groups() — они уже
    несут lr (с LLRD и lambda-иерархией), weight_decay, betas. Здесь добавляются
    поля роли и per-group флаги модификаторов.
    """

    def __init__(self, params, mode: str = "eva", lr: float = 3e-4,
                 betas=(0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.01,
                 cautious: bool = True, slow_ema: bool = True,
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
            slow_mix = group["slow_mix"]
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
                # Тот же порядок операций, что в torch.optim.AdamW (для побитовой
                # идентичности в mode='adamw'): lerp + addcmul_ + denom-разложение.
                m.lerp_(g, 1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                bc1 = 1.0 - b1 ** step
                bc2 = 1.0 - b2 ** step
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                u = m / denom
                # AdamP-проекция "сырого" Adam-направления (до коррекции bc1 и
                # до slow/cautious) — оригинальный случай AdamP: p̄ = m̂/√v̂.
                if eva_on and group["projected"] and wd > 0.0 and p.dim() >= 2:
                    u = _adamp_project(u, p)
                if slow_ok:
                    ms = st.get("exp_avg_slow")
                    if ms is None:      # реставрация со старых чекпоинтов без slow-EMA
                        ms = st["exp_avg_slow"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)
                    b3 = group["beta_slow"]
                    ms.mul_(b3).add_(g, alpha=1.0 - b3)
                    # Ada-нормированный на v̂^0.5 slow-терм; bc1 вынесен наружу,
                    # чтобы net-коэффициент был ровно lr*slow_mix (см. apply ниже).
                    slow_n = (ms / (1.0 - b3 ** step)) / (v / bc2).sqrt().add_(eps)
                    u = u + slow_mix * bc1 * slow_n
                if cautious:
                    mask = (u * g > 0).to(u.dtype)
                    u.mul_(mask).div_(mask.mean().clamp_min(1e-3))
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
                        cautious: bool = True, slow_ema: bool = True,
                        beta_slow: float = 0.9999, slow_mix: float = 0.25,
                        trust_enabled: bool = True, trust_floor: float = 0.5,
                        projected_wd: bool = False, debug: bool = False):
    """Алиас: EVA-группы строятся в core.adaptation.build_optimizer(optimizer='eva').

    Единый источник строительства групп — build_optimizer (LLRD + роли).
    """
    from .adaptation import build_optimizer as _build
    return _build(model, base_lr, llrd_decay=llrd_decay, weight_decay=weight_decay,
                  betas=betas, optimizer="eva",
                  eva_kwargs=dict(cautious=cautious, slow_ema=slow_ema,
                                  beta_slow=beta_slow, slow_mix=slow_mix,
                                  trust_enabled=trust_enabled,
                                  trust_floor=trust_floor,
                                  projected_wd=projected_wd, debug=debug))
