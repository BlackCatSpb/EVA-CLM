from __future__ import annotations

import math
import torch
from typing import List, Optional, Tuple

from .training_control import mirror_lstats


class AdaptiveController:
    """
    Computes ALL adaptive hyperparameters from cognitive mirror state.

    Two fundamental signals drive every parameter:
    ──────────────────────────────────────────────────────────
    exploration = min(1, |mirror| / λ⁻²)
        How much correction is the mirror applying.
        High → model is actively adjusting, needs aggressive config.
        Low → model is stable, needs conservative config.

    differentiation = min(1, var(log_scale) / λ⁻⁴)
        How specialized has the mirror become (per-dim scaling).
        High → mirror has learned which dims to trust/suppress.
        Low → mirror hasn't differentiated, still exploring.

    λ_d hierarchy (d=3): λ₃ ≈ 1.839, λ⁻² ≈ 0.296, λ⁻⁴ ≈ 0.087
    All range defaults below are λ_d d=3 derived.

    Key design: ALL methods work at per-layer AND global resolution.
    ``layer_stats(layer)`` → per-layer (expl, diff)
    ``stats(blocks)`` → global average   (backward compat)

    New intelligent adaptivity:
    ──────────────────────────
    - ``pred_weight(blocks)`` — alpha loss weight scales with diff
      (more temporal learning when mirror has specialized)
    - ``tanh_bias_modulation(layer)`` — tanh_bias amplified by exploration
      (more asymmetric correction when actively exploring)
    - ``spectral_modulation(layer)`` — lambda_k amplified by differentiation
      (more aggressive freq shaping when experts are specialized)
    - ``pred_scale_mod(layer)`` — per-expert modulation from delta_var
      (experts with volatile dynamics get more temporal teaching signal)

    Mathematically derived ranges (λ_d d=3):
    ────────────────────────────────────────
    b_d ∈ [b_d_min, b_d_max] per layer, where b_d_min = 2.0 + 3.0*layer_frac
         expl=1 → b_d = b_d_min (shortest memory)
         expl=0 → b_d = b_d_max (longest memory, configurable vsa_b_d_max)
         L0: τ≈[7, 150] (default b_d_max=5.0), up to τ≈160K (b_d_max=12.0)
         Per-channel via gradient: b_d is (D,) with lerp-slow push to controller target
    b_i  ∈ [-3.0, -1.5] → i_gate ≈ [0.047, 0.18] (write rate via softplus)
    w_mem2v_scale ∈ [0.544, 1.0]  (memory contribution, λ⁻¹ to 1)
    ema_alpha ∈ [0.974, 0.992]  (cross-layer memory, 1-λ⁻⁶ to 1-λ⁻⁸)
    noise_scale ∈ [0.0076, 0.026]  (parameter noise, λ⁻⁸ to λ⁻⁶)
    pred_weight ∈ [0.026, 0.296]  (alpha loss weight, λ⁻⁶ to λ⁻²)
    tanh_bias_mod ∈ [1.0, 1.5]  (exploration amplification)
    spectral_mod ∈ [0.913, 1.087]  (differentiation, 1±λ⁻⁴)
    """
    @staticmethod
    def layer_stats(layer, expl_thresh: float = 0.296, diff_thresh: float = 0.087) -> Tuple[float, float]:
        """Per-layer (exploration, differentiation) from a single block.

        Live τ-aware signals (core.training_control.mirror_lstats):
          exploration      = min(1, |mirror| / expl_thresh)   [λ⁻² of the
                             hierarchy — the passed cfg value is USED; the
                             0.296 default is λ₃⁻²]
          differentiation  = behavioural divergence / its own running mean
                             (self-referenced, saturating) — replaces the old
                             var(log_scale)/λ⁻⁴, which froze at 0 whenever
                             log_scale stopped moving, pinning all per-layer
                             gains to their conservative bound forever.
        ``diff_thresh`` is a legacy signature knob: the self-referenced diff
        signal needs no absolute threshold and does not use it.
        """
        return mirror_lstats(layer, expl_thresh=expl_thresh)

    @staticmethod
    def stats(blocks, expl_thresh: float = 0.296, diff_thresh: float = 0.087) -> Tuple[float, float]:
        """Global average (exploration, differentiation) across all layers."""
        expl_sum = diff_sum = 0.0
        for layer in blocks:
            e, d = AdaptiveController.layer_stats(layer, expl_thresh, diff_thresh)
            expl_sum += e
            diff_sum += d
        n = len(blocks)
        return expl_sum / n, diff_sum / n

    # ─── Per-layer methods ────────────────────────────────────────

    @staticmethod
    def layer_b_d(layer, expl: Optional[float] = None, b_d_max: float = 5.0) -> float:
        """Per-layer decay bias. Layer uses its own exploration."""
        if expl is None:
            expl, _ = AdaptiveController.layer_stats(layer)
        lf = getattr(layer, 'layer_idx', 0) / max(getattr(layer, 'total_layers', 32) - 1, 1)
        b_d_min = 2.0 + 3.0 * lf
        b_d_val = b_d_max - expl * (b_d_max - b_d_min)
        return max(2.0, min(b_d_max, b_d_val))

    @staticmethod
    def layer_b_i(layer, expl: Optional[float] = None, tau_l: Optional[float] = None) -> float:
        """Per-layer write gate bias. Нормировка: i_gate ∝ 1/τ.
        
        i_gate = softplus(b_i_l). Равновесная норма памяти:
            ‖M_l‖ = i_gate · ‖h‖ · τ_l
        Для ‖M_l‖ = const по слоям: i_gate ∝ 1/τ_l.
        
        Базовое значение: i_gate_ref = 0.182 при τ_ref ≈ 32.
        c = 0.182 · 32 ≈ 5.83.
        i_gate_l = c / τ_l  →  b_i_l = softplus⁻¹(c / τ_l)
        """
        if expl is None:
            expl, _ = AdaptiveController.layer_stats(layer)
        b_i_base = -3.0 + expl * 1.5
        c = 5.83
        if tau_l is not None:
            i_target = min(1.0, c / tau_l)
        else:
            lf = getattr(layer, 'layer_idx', 0) / max(getattr(layer, 'total_layers', 32) - 1, 1)
            tau_l = 8.0 + 141.0 * lf
            i_target = min(1.0, c / tau_l)
        b_i_tau = math.log(max(i_target, 1e-6))
        b_i = b_i_base + b_i_tau
        return max(b_i, -6.0)  # floor: i_gate >= softplus(-6.0) ≈ 0.0025

    @staticmethod
    def layer_w_mem2v_scale(layer, min_val: float = 0.544, max_val: float = 1.0, diff: Optional[float] = None) -> float:
        """Per-layer memory contribution."""
        if diff is None:
            _, diff = AdaptiveController.layer_stats(layer)
        return max_val - diff * (max_val - min_val)

    @staticmethod
    def layer_noise_scale(layer, min_val: float = 0.0076, max_val: float = 0.026, diff: Optional[float] = None) -> float:
        """Per-layer parameter noise."""
        if diff is None:
            _, diff = AdaptiveController.layer_stats(layer)
        return max_val - diff * (max_val - min_val)

    @staticmethod
    def layer_ema_alpha(layer, min_val: float = 0.974, max_val: float = 0.992, diff: Optional[float] = None) -> float:
        """Per-layer EMA rate (for per-layer global_state aggregation)."""
        if diff is None:
            _, diff = AdaptiveController.layer_stats(layer)
        return min_val + diff * (max_val - min_val)

    # ─── New intelligent adaptivity ───────────────────────────────

    @staticmethod
    def pred_weight(blocks, min_val: float = 0.026, max_val: float = 0.296) -> float:
        """Adaptive alpha auxiliary loss weight.

        When mirror has differentiated (high diff), temporal prediction
        is more meaningful → increase pred_weight to drive alpha learning.
        When mirror hasn't specialized, pred would be noise → keep low.
        """
        _, diff = AdaptiveController.stats(blocks)
        return min_val + diff * (max_val - min_val)

    @staticmethod
    def tanh_bias_modulation(layer, expl: Optional[float] = None) -> float:
        """Scale tanh_bias by exploration.

        High exploration → more asymmetric correction needed → amplify.
        Range: [1.0, 1.296] (at most 1+λ⁻² boost).
        """
        if expl is None:
            expl, _ = AdaptiveController.layer_stats(layer)
        return 1.0 + 0.296 * expl

    @staticmethod
    def spectral_modulation(layer, diff: Optional[float] = None) -> float:
        """Modulate spectral lambda_k by differentiation.

        High diff → mirror has learned structure → amplify spectral
        contrast (more aggressive frequency shaping).
        Low diff → flatten spectral response (conservative).
        Range: [0.913, 1.087] = 1 ± λ⁻⁴.
        """
        if diff is None:
            _, diff = AdaptiveController.layer_stats(layer)
        return 1.0 + 0.087 * (diff - 0.5) * 2.0  # 0.913 at diff=0, 1.087 at diff=1

    @staticmethod
    def pred_scale_mod(layer) -> torch.Tensor:
        """Per-expert prediction-error modulation from delta_var.
        
        Experts with volatile K-space dynamics (high delta_var relative
        to layer average) get more temporal teaching signal.
        Uses tanh-based soft normalization instead of division to avoid NaN.
        Range: [0.5, 2.0] centered at 1.0.
        """
        dv = layer.mirror._delta_var
        dv_centered = dv - dv.mean()
        return (1.0 + 0.5 * torch.tanh(dv_centered)).clamp(0.1, 3.0)

    # ─── Global (backward-compat) wrappers ────────────────────────

    @staticmethod
    def b_d(blocks, b_d_max: float = 5.0) -> float:
        expl, _ = AdaptiveController.stats(blocks)
        return b_d_max - expl * 2.0

    @staticmethod
    def b_i(blocks) -> float:
        expl, _ = AdaptiveController.stats(blocks)
        return -3.0 + expl * 1.5

    @staticmethod
    def w_mem2v_scale(blocks, min_val: float = 0.544, max_val: float = 1.0) -> float:
        _, diff = AdaptiveController.stats(blocks)
        return max_val - diff * (max_val - min_val)

    @staticmethod
    def ema_alpha(blocks, min_val: float = 0.974, max_val: float = 0.992) -> float:
        _, diff = AdaptiveController.stats(blocks)
        return min_val + diff * (max_val - min_val)

    @staticmethod
    def noise_scale(blocks, min_val: float = 0.0076, max_val: float = 0.026) -> float:
        _, diff = AdaptiveController.stats(blocks)
        return max_val - diff * (max_val - min_val)
