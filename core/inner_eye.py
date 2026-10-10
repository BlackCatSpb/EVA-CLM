"""core/inner_eye.py — P4-1: обучаемая интроспекция (shared inner eye).

Один модуль на всю модель: слой входит признаком τ̃_l (τ-поле как система
координат, не индекс слоя), эксперт — признаками своего состояния; выход —
добавка к gate_logits зеркала через паттерн w_intent-канала (running-RMS +
τ-авторитет intent_alpha — предохранители, спасшие w_intent от runaway 62k,
run A2).

Zero-init выходного слоя ⇒ identity на init (forward побитово прежний).
Канал ограничен по построению: признаки — нормы/RMS/сигмоиды/клампы, выход
нормирован собственной RMS-EMA и зажат ±10 (последний предохранитель).

G-лестница зеркал (k_l ∈ {8,16,32} по третям глубины) поддержана нарезкой
expert_bias[:G] — модуль общий, но bias каждого слоя живёт на своём префиксе.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_FEAT = 13   # 12 признаков состояния + τ̃_l


class InnerEye(nn.Module):
    def __init__(self, G: int, hidden: int = 32):
        super().__init__()
        self.n_experts = int(G)
        self.net = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)      # identity at init
        nn.init.zeros_(self.net[-1].bias)
        self.expert_bias = nn.Parameter(torch.zeros(self.n_experts))
        self.register_buffer('_o_rms_ema', torch.ones(1), persistent=False)

    def forward(self, feats: torch.Tensor, tau_norm: float) -> torch.Tensor:
        """feats: (B,L,G,N_FEAT-1) — уже detached (см. mirror: градиент внутрь
        статистик зеркала не нужен, учится только сам eye); tau_norm — координата
        слоя. Возвращает (B,L,G) — RMS-нормированный выход; умножение на
        intent_alpha делает ВЫЗЫВАЮЩИЙ (зеркало) — как у ig/ctr."""
        B, L, G, _ = feats.shape
        tn = torch.full((B, L, G, 1), float(tau_norm),
                        device=feats.device, dtype=feats.dtype)
        o = self.net(torch.cat([feats, tn], dim=-1)).squeeze(-1)
        o = o + self.expert_bias[:G].view(1, 1, G)
        if self.training:
            # M8-доктрина: eval статистику не двигает — на eval авторитет
            # ограничен тренировочной EMA.
            # Блок 2 (корневой recompute-фикс): буфер _o_rms_ema общий для всех
            # слоёв; recompute слоёв идёт в обратном порядке, поэтому зеркало
            # временно ставит pre-call значение слоя (S_i) и возвращает S_final
            # после. Чтение update-then-read: локально считаем post-значение
            # первого прохода БЕЗ записи буфера — forward бит-идентичен.
            with torch.no_grad():
                _rms = o.detach().pow(2).mean().sqrt()
                if getattr(self, '_recomp', False):
                    _den = 0.99 * self._o_rms_ema + 0.01 * _rms
                else:
                    self._o_rms_ema.mul_(0.99).add_(_rms, alpha=0.01)
                    _den = self._o_rms_ema
        else:
            _den = self._o_rms_ema
        return (o / (_den + 1e-8)).clamp(-10.0, 10.0)
