"""WideBind: Hybrid Gate — unified sigmoid-softmax continuum.

Единая формула для ВСЕХ гейтов в архитектуре:

  gate = sigmoid(logits) * (1 + softmax(logits / tau))

Варианты использования:
  hybrid_gate(logits, tau)              — raw output (AdaptiveGate, SpectrumGate)
  hybrid_gate(logits, tau, log=True)    — log space (SigmoidCodedHead._su)
  hybrid_gate(scores, tau, normalize=True) — with L1 normalization (memory attention)

tau (temperature) управляет sharpness softmax:
  tau→0: winner-take-all (одна фича доминирует)
  tau→∞: равномерное распределение (softmax → 1/n для всех)
  tau=1: default balance
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def hybrid_gate(
    logits: torch.Tensor,
    tau: torch.Tensor | float,
    dim: int = -1,
    log: bool = False,
    normalize: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Unified sigmoid-softmax hybrid gate.

    Args:
        logits: (*, n_features) — raw feature scores
        tau: scalar or broadcastable tensor — temperature
        dim: dimension to softmax over (default: -1)
        log: if True, return (log_odds, log_base) instead of gate
        normalize: if True, normalize gate to sum to 1 (for attention)
        eps: clamp min/max for log space

    Returns:
        gate: (*, n_features) — combined gate values (if log=False)
        or (u, base) — log-odds and log-base (if log=True)
    """
    tau_t = torch.as_tensor(tau, device=logits.device, dtype=logits.dtype)
    tau_t = tau_t.clamp(min=0.1, max=10.0)

    # 1. Independent activation (sigmoid)
    independent = torch.sigmoid(logits)

    # 2. Relative emphasis (softmax with temperature)
    relative = F.softmax(logits / tau_t, dim=dim)

    # 3. Combined: sigmoid gates participation, softmax adds relative boost
    gate = independent * (1.0 + relative)

    if normalize:
        gate_sum = gate.sum(dim=dim, keepdim=True).clamp(min=eps)
        gate = gate / gate_sum

    if log:
        gate = gate.clamp(eps, 1 - eps)
        ls = torch.log(gate)
        lms = torch.log(1 - gate)
        u = ls - lms
        base = lms.sum(dim=dim)
        return u, base

    return gate


class AdaptiveGate(nn.Module):
    """Sigmoid-Softmax hybrid gate.

    Логика:
      independent = sigmoid(logits)        # [0,1] per feature, no competition
      relative = softmax(logits / tau)     # sum=1, relative ranking
      gate = independent * (1 + relative)  # combined effect

    Градиенты:
      sigmoid path: прямой gradient для включения/выключения фичей
      softmax path: relative gradient для перераспределения веса
      multiplicative: strong features get stronger (без подавления слабых)
    """

    def __init__(self, n_features: int, tau_init: float = 1.0, learnable_tau: bool = True):
        super().__init__()
        self.n_features = n_features

        if learnable_tau:
            self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        else:
            self.register_buffer('log_tau', torch.tensor(math.log(tau_init)))

        # Diagnostics
        self.register_buffer('_last_indep_mean', torch.tensor(0.0))
        self.register_buffer('_last_relative_entropy', torch.tensor(0.0))
        self.register_buffer('_last_gate_std', torch.tensor(0.0))

    def forward(self, logits: torch.Tensor, tau_prior: torch.Tensor | None = None) -> torch.Tensor:
        _DEV_CLAMP = 2.0  # unified deviation multiplier clamp (0.5..2.0)
        if tau_prior is not None:
            tau = tau_prior.clamp(min=0.1, max=10.0) * torch.exp(self.log_tau).clamp(min=1.0/_DEV_CLAMP, max=_DEV_CLAMP)
            tau = tau.clamp(min=0.1, max=10.0)
        else:
            tau = torch.exp(self.log_tau).clamp(min=0.1, max=10.0)
        gate = hybrid_gate(logits, tau)

        # Update diagnostics
        with torch.no_grad():
            independent = torch.sigmoid(logits)
            relative = F.softmax(logits / tau, dim=-1)
            self._last_indep_mean.copy_(independent.mean().detach())
            self._last_gate_std.copy_(gate.std().detach())
            self._last_relative_entropy.copy_(
                -(relative * (relative + 1e-8).log()).sum(dim=-1).mean().detach()
            )

        return gate

    def get_diagnostics(self) -> dict:
        return {
            'indep_mean': self._last_indep_mean.item(),
            'relative_entropy': self._last_relative_entropy.item(),
            'gate_std': self._last_gate_std.item(),
            'tau': torch.exp(self.log_tau).item(),
        }
