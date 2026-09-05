"""WideBind: SpectrumGate — sigmoid-softmax continuum.

Три режима на одном спектре, управляемом tau:

  tau→∞  : pure sigmoid  → maximum diversity (все фичи независимы)
  tau=1  : balanced      → diversity with precision (сигмоида + softmax boost)
  tau→0  : pure softmax  → final precision (winner-take-all, один выбор)

Формула (единая для всей архитектуры):
  gate = sigmoid(logits) * (1 + softmax(logits / tau))

Применение в WideBind:
  - Diagnostics routing: tau=∞ для всех diagnostics (diversity)
  - Bridge routing: tau≈1 для bridge contribution (balanced)
  - Intent selection: tau≈0 для top-1 intent (final precision)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from core.adaptive_gate import hybrid_gate


class SpectrumGate(nn.Module):
    """Sigmoid-Softmax spectrum gate.

    Modes:
      diversity:          tau → ∞, gate ≈ sigmoid(logits) * 1.0
      diversity+precision: tau ≈ 1, gate = sigmoid * (1 + softmax)
      final_precision:   tau → 0, gate ≈ sigmoid * (1 + one-hot)

    Learnable tau allows the model to transition between modes during training.
    """

    def __init__(
        self,
        n_features: int,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        tau_max: float = 10.0,
        learnable_tau: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.tau_min = tau_min
        self.tau_max = tau_max

        if learnable_tau:
            self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        else:
            self.register_buffer('log_tau', torch.tensor(math.log(tau_init)))

        # Diagnostics
        self.register_buffer('_last_indep_mean', torch.tensor(0.0))
        self.register_buffer('_last_relative_entropy', torch.tensor(0.0))
        self.register_buffer('_last_gate_std', torch.tensor(0.0))
        self.register_buffer('_last_dominance', torch.tensor(0.0))

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau).clamp(self.tau_min, self.tau_max)

    @property
    def mode(self) -> str:
        t = self.tau.item()
        if t > 5.0:
            return 'diversity'
        elif t > 0.5:
            return 'diversity+precision'
        else:
            return 'final_precision'

    def forward(self, logits: torch.Tensor, tau_external: torch.Tensor | None = None) -> torch.Tensor:
        _DEV_CLAMP = 2.0  # unified deviation multiplier clamp (0.5..2.0)
        if tau_external is not None:
            tau = tau_external.clamp(self.tau_min, self.tau_max) * torch.exp(self.log_tau).clamp(1.0/_DEV_CLAMP, _DEV_CLAMP)
            tau = tau.clamp(self.tau_min, self.tau_max)
        else:
            tau = self.tau
        gate = hybrid_gate(logits, tau)

        # Update diagnostics
        with torch.no_grad():
            independent = torch.sigmoid(logits)
            relative = torch.softmax(logits / tau, dim=-1)
            self._last_indep_mean.copy_(independent.mean().detach())
            self._last_gate_std.copy_(gate.std().detach())
            self._last_relative_entropy.copy_(
                -(relative * (relative + 1e-8).log()).sum(dim=-1).mean().detach()
            )
            max_vals = relative.max(dim=-1).values
            mean_vals = relative.mean(dim=-1)
            self._last_dominance.copy_((max_vals / (mean_vals + 1e-8)).mean().detach())

        return gate

    def set_tau(self, tau: float):
        with torch.no_grad():
            self.log_tau.fill_(math.log(max(tau, self.tau_min)))

    def get_diagnostics(self) -> dict:
        return {
            'tau': self.tau.item(),
            'mode': self.mode,
            'indep_mean': self._last_indep_mean.item(),
            'relative_entropy': self._last_relative_entropy.item(),
            'gate_std': self._last_gate_std.item(),
            'dominance': self._last_dominance.item(),
        }
