"""T9: ковариационная память (порт EVA-Ai/FCP) — 2-й порядок корреляций.

Состояние — матрица ковариации на голову:  M_t = d_t·M_{t−1} + i_t·k_t k_tᵀ
(в отличие от VSA-суперпозиции векторов: M хранит парные корреляции).
Чтение — q_t·M_t·W_read. Затухание d_t выводится из τ слоя (единый язык),
запись i_t — контентный гейт. Обучение — chunked log-space скан
(стабильная форма, как в VSA-скане), инференс — одношаговая рекуррентность;
эквивалентность скана и рекуррентности закреплена тестами.

Интеграция в EVABlock — отдельным шагом (флаг cfg.cov_memory, default off).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CovarianceMemory(nn.Module):
    """Multi-head ковариационная память с τ-затуханием.

    Args:
        D: ширина входа/выхода.
        n_heads: число голов.
        head_dim: размерность головы (k/q пространство).
        tau: постоянная времени слоя (затухание d = exp(−1/τ)).
        chunk: длина чанка для параллельного скана.
    """

    def __init__(self, D: int, n_heads: int = 8, head_dim: int = 160,
                 tau: float = 64.0, chunk: int = 64) -> None:
        super().__init__()
        self.D = int(D)
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.tau = float(max(tau, 2.0))
        self.chunk = int(chunk)
        Hd = self.n_heads * self.head_dim
        self.k_proj = nn.Linear(D, Hd)
        self.q_proj = nn.Linear(D, Hd)
        # контентные гейты (как в VSA-регистрах): d_mod укорачивает память,
        # i_gate задаёт силу записи; покой = полный горизонт τ
        self.w_d = nn.Parameter(torch.zeros(D))
        self.b_d = nn.Parameter(torch.tensor(1.0))
        self.w_i = nn.Parameter(torch.zeros(D))
        self.b_i = nn.Parameter(torch.tensor(0.0))
        self.W_read = nn.Linear(Hd, D, bias=False)
        self.W_out = nn.Linear(D, D, bias=False)
        nn.init.zeros_(self.W_out.weight)   # zero-init ⇒ forward стартует бит-в-бит как residual

    # ─────────────────────────── decay / gates ───────────────────────────

    def _log_decay(self, x: torch.Tensor) -> torch.Tensor:
        """log d_t: τ-горизонт, укороченный контентом (d_mod ∈ (0,1])."""
        d_mod = torch.sigmoid(x @ self.w_d + self.b_d) / torch.sigmoid(self.b_d)
        d_mod = d_mod.clamp(max=1.0)
        return (-1.0 / self.tau) + torch.log(d_mod.clamp_min(1e-6))

    def _write_gate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x @ self.w_i + self.b_i)

    # ─────────────────────────── chunked scan ───────────────────────────

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        """x: (B, L, D). state: (B, H, Dh, Dh) или None.
        Возвращает (y (B, L, D), new_state (B, H, Dh, Dh))."""
        B, L, _ = x.shape
        H, Dh = self.n_heads, self.head_dim
        k = self.k_proj(x).reshape(B, L, H, Dh)
        q = self.q_proj(x).reshape(B, L, H, Dh)
        log_d = self._log_decay(x)          # (B, L)
        i_g = self._write_gate(x)           # (B, L)
        M = state if state is not None else x.new_zeros(B, H, Dh, Dh)

        C = min(self.chunk, L)
        outs = []
        for c0 in range(0, L, C):
            c1 = min(c0 + C, L)
            A = torch.cumsum(log_d[:, c0:c1], dim=1)          # (B, c), A ≤ 0, убывает
            A_last = A[:, -1].view(B, 1)                      # (B, 1)
            kc = k[:, c0:c1]                                  # (B, c, H, Dh)
            qc = q[:, c0:c1]
            ic = i_g[:, c0:c1]                                # (B, c)
            # стабильная форма: P'_t = Σ_{s≤t} exp(A_last − A_s)·i_s·k_s k_sᵀ
            # (вес ≤ 1), локальный вклад = exp(A_t − A_last)·P'_t
            w = (A_last - A).exp() * ic                       # (B, c)
            kk = kc.unsqueeze(-1) * kc.unsqueeze(-2)          # (B, c, H, Dh, Dh)
            P = torch.cumsum(w.view(B, c1 - c0, 1, 1, 1) * kk, dim=1)  # (B, c, H, Dh, Dh)
            M_t = (M.unsqueeze(1) * A.exp().view(B, c1 - c0, 1, 1, 1)
                   + P * (A - A_last).exp().view(B, c1 - c0, 1, 1, 1))
            y = torch.einsum('bthd,bthde->bthe', qc, M_t)      # (B, c, H, Dh)
            outs.append(y.reshape(B, c1 - c0, H * Dh))
            # состояние на конец чанка: M ← exp(A_last)·M + P'_last
            M = (M * A_last.view(B, 1, 1, 1).exp()) + P[:, -1]
        y = torch.cat(outs, dim=1)
        return self.W_out(self.W_read(y)), M.detach()

    # ─────────────────────────── streaming ───────────────────────────

    def step(self, x_t: torch.Tensor, state: torch.Tensor | None = None):
        """Одношаговая рекуррентность (инференс): x_t (B, 1, D)."""
        B = x_t.shape[0]
        H, Dh = self.n_heads, self.head_dim
        k = self.k_proj(x_t).reshape(B, 1, H, Dh)
        q = self.q_proj(x_t).reshape(B, 1, H, Dh)
        d = self._log_decay(x_t).exp().view(B, 1, 1, 1)        # (B,1,1,1)
        i_g = self._write_gate(x_t).view(B, 1, 1, 1)
        M = state if state is not None else x_t.new_zeros(B, H, Dh, Dh)
        kk = k.unsqueeze(-1) * k.unsqueeze(-2)                 # (B,1,H,Dh,Dh)
        M = M * d + kk * i_g
        y = torch.einsum('bthd,bthde->bthe', q, M).reshape(B, 1, H * Dh)
        return self.W_out(self.W_read(y)), M.detach()
