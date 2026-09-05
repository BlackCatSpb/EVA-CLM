"""WideBind: per-layer bridge gate with SpectrumGate (sigmoid-softmax hybrid).

Каждый слой имеет SpectrumGate, который агрегирует diagnostics в gate value.
Gate = SpectrumGate(diagnostics) * tau_maturation — связь с maturation.
SpectrumGate = sigmoid(logits) * (1 + softmax(logits/tau)) — оба преимущества.

GLOBAL READINESS: LayerBridgeGate активен только когда maturation.global_ready
=True (все слои проснулись). До этого — uniform weights (простое per-layer
maturation gating без сложного SpectrumGate).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from core.adaptive_gate import hybrid_gate


class SpectrumGate(nn.Module):
    """Sigmoid-Softmax hybrid gate.

    gate = sigmoid(logits) * (1 + softmax(logits / tau))

    - sigmoid: independent activation per feature (no zero-sum)
    - softmax: relative emphasis among features
    - tau controls the blend (high=diversity, low=precision)

    tau can be:
    - Learnable (log_tau parameter) — model learns the blend
    - External (from maturation) — self-regulation through system tau
    """

    def __init__(self, n_features: int, tau_init: float = 1.0):
        super().__init__()
        self.n_features = n_features
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))

    def forward(self, logits: torch.Tensor, tau_external: torch.Tensor | None = None) -> torch.Tensor:
        _DEV_CLAMP = 2.0  # unified deviation multiplier clamp (0.5..2.0)
        if tau_external is not None:
            tau = tau_external.clamp(0.1, 10.0) * torch.exp(self.log_tau).clamp(1.0/_DEV_CLAMP, _DEV_CLAMP)
            tau = tau.clamp(0.1, 10.0)
        else:
            tau = torch.exp(self.log_tau).clamp(0.1, 10.0)
        return hybrid_gate(logits, tau)

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau).clamp(0.1, 10.0)


class LayerBridgeGate(nn.Module):
    """Per-layer intelligent gate to SemanticBridge with SpectrumGate.
    
    Self-regulation through tau:
    - Immature layers (mat≈0) → tau→∞ → diversity (all diagnostics active)
    - Mature layers (mat≈1) → tau→0 → precision (top diagnostics dominate)
    
    Formula: effective_tau = tau_max * (1 - maturation) + tau_min * maturation
    
    Diagnostic features (per layer):
    0. pred_error_norm: предсказание зеркала (низкая = хорошо)
    1. gate_l1: стабильность гейтов экспертов (низкая = стабильно)
    2. mirror_norm: активность зеркала (умеренная = хорошо)
    3. bridge_contribution: вклад bridge (высокая = помогает)
    4. expert_entropy: разнообразие экспертов (умеренная = хорошо)
    5. diversity: разнообразие представлений (умеренная = хорошо)
    """
    
    def __init__(self, n_layers: int, health_features: int = 6,
                 tau_min: float = 0.3, tau_max: float = 5.0):
        super().__init__()
        self.n_layers = n_layers
        self.health_features = health_features
        self.tau_min = tau_min
        self.tau_max = tau_max
        
        # Per-layer SpectrumGate: each layer decides its own sigmoid/softmax blend
        self.gates = nn.ModuleList([
            SpectrumGate(health_features, tau_init=1.0)
            for _ in range(n_layers)
        ])
        
        # NaN/explosion control
        self._nan_count = 0
        self._max_nan = 10
    
    def _effective_tau(self, maturation: torch.Tensor) -> torch.Tensor:
        """Compute effective tau from maturation.
        
        Args:
            maturation: (n_layers,) or scalar — maturation gate values in [0, 1]
            
        Returns:
            effective_tau: same shape — tau for each layer
        """
        return self.tau_max * (1.0 - maturation) + self.tau_min * maturation
    
    def forward(
        self,
        layer_outputs: torch.Tensor,  # (n_layers, B, D)
        diagnostics: torch.Tensor,    # (n_layers, health_features)
        tau_maturation: torch.Tensor, # (n_layers,) — maturation gate values
        global_ready: bool = False,   # True when ALL layers are mature enough
    ):
        """Compute weighted bridge input from layer outputs.
        
        When global_ready=False: return uniform weights (simple maturation gating).
        When global_ready=True: full SpectrumGate with per-layer tau-driven diversity.
        
        Args:
            layer_outputs: (n_layers, B, D) — per-layer hidden states
            diagnostics: (n_layers, health_features) — per-layer diagnostics
            tau_maturation: (n_layers,) — maturation gate values
            global_ready: bool — True when all layers are mature enough
            
        Returns:
            bridge_input: (B, D) — weighted sum of layer outputs
            gate_weights: (n_layers, 1) — per-layer gate weights (for logging)
            gate_info: dict — diagnostic info for logging
        """
        n_layers = self.n_layers
        
        # ─── Global readiness gate ───
        # Before all layers are mature: uniform weights (no SpectrumGate).
        # This prevents the complex per-layer bridge routing from killing
        # immature layers. Bridge injection is still scaled by per-layer
        # maturation (in inject_layer), so immature layers get less injection.
        if not global_ready:
            normalized_gates = torch.ones(n_layers, device=layer_outputs.device, dtype=layer_outputs.dtype) / n_layers
            gate_info = {
                'lbg_tau': [self.tau_max] * n_layers,
                'lbg_raw_mean': 1.0 / n_layers,
                'lbg_tau_min': self.tau_max,
                'lbg_tau_max': self.tau_max,
                'lbg_global_ready': False,
            }
            weighted = normalized_gates.unsqueeze(-1).unsqueeze(-1) * layer_outputs
            bridge_input = weighted.sum(dim=0)
            return bridge_input, normalized_gates.unsqueeze(-1), gate_info
        
        # ─── Full SpectrumGate per-layer (global_ready=True) ───
        # 1. Compute effective tau from maturation (self-regulation)
        effective_tau = self._effective_tau(tau_maturation)  # (n_layers,)
        
        # 2. Per-layer SpectrumGate with maturation-driven tau
        raw_gates = []
        gate_taus = []
        for l in range(n_layers):
            # SpectrumGate: maturation tau overrides learnable tau
            gated_features = self.gates[l](diagnostics[l], tau_external=effective_tau[l])
            # Reduce to scalar: mean of gated features
            scalar_gate = gated_features.mean()
            raw_gates.append(scalar_gate)
            gate_taus.append(effective_tau[l].item())
        
        raw_gates = torch.stack(raw_gates)  # (n_layers,)
        
        # 3. Gate = raw_gate * maturation (conservative for immature layers)
        gates = raw_gates * tau_maturation  # (n_layers,)
        
        # 4. NaN/explosion control
        gates = torch.where(torch.isnan(gates), torch.zeros_like(gates), gates)
        gates = torch.clamp(gates, min=0.0, max=2.0)
        
        # 5. Weighted average normalization
        gate_sum = gates.sum()
        gate_sum = torch.clamp(gate_sum, min=1e-6)
        normalized_gates = gates / gate_sum  # (n_layers,)
        
        # 6. Fallback: if all gates are zero, use equal weights
        if gate_sum < 1e-6:
            normalized_gates = torch.ones_like(gates) / n_layers
            self._nan_count += 1
            if self._nan_count > self._max_nan:
                with torch.no_grad():
                    for g in self.gates:
                        g.log_tau.fill_(0.0)  # tau=1.0
                self._nan_count = 0
        else:
            self._nan_count = 0
        
        # 7. Weighted sum of layer outputs
        weighted = normalized_gates.unsqueeze(-1).unsqueeze(-1) * layer_outputs
        bridge_input = weighted.sum(dim=0)  # (B, D)
        
        # 8. Diagnostic info for logging
        gate_info = {
            'lbg_tau': gate_taus,
            'lbg_raw_mean': raw_gates.mean().item(),
            'lbg_tau_min': min(gate_taus),
            'lbg_tau_max': max(gate_taus),
            'lbg_global_ready': True,
        }
        
        return bridge_input, normalized_gates.unsqueeze(-1), gate_info
    
    def get_diagnostics(
        self,
        layers: nn.ModuleList,
        bridge_contribution: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract per-layer diagnostics from model state."""
        n = self.n_layers
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        diagnostics = torch.zeros(n, self.health_features, device=device, dtype=dtype)
        
        for l in range(n):
            layer = layers[l]
            mir = layer.mirror
            
            pe = getattr(mir, '_cached_pred_error_norm', None)
            if pe is not None:
                diagnostics[l, 0] = pe.detach().mean().clamp(0.0, 1.0)
            
            gl = getattr(mir, '_cached_gate_l1', None)
            if gl is not None:
                diagnostics[l, 1] = gl.detach().clamp(0.0, 1.0)
            
            mp = getattr(mir, '_cached_pred_k', None)
            if mp is not None:
                mn = mp.detach().norm()
                diagnostics[l, 2] = (mn / 1000.0).clamp(0.0, 1.0)
            
            if bridge_contribution is not None:
                diagnostics[l, 3] = bridge_contribution[l].clamp(0.0, 1.0)
            
            hp = getattr(mir, '_cached_hp', None)
            if hp is not None:
                hp_det = hp.detach()
                hp_norm = torch.sigmoid(hp_det)
                hp_norm = hp_norm / hp_norm.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                entropy = -(hp_norm * hp_norm.clamp_min(1e-9).log()).sum()
                max_entropy = math.log(hp_det.shape[-1])
                diagnostics[l, 4] = (entropy / max_entropy).clamp(0.0, 1.0)
            
            gate_l1 = getattr(mir, '_cached_gate_l1', None)
            if gate_l1 is not None:
                diagnostics[l, 5] = (1.0 - gate_l1).clamp(0.0, 1.0)
        
        return diagnostics
