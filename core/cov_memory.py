"""T9: ковариационная память (порт EVA-Ai/FCP) — 2-й порядок корреляций.

Состояние — матрица ковариации на голову:  M_t = d_t·M_{t−1} + i_t·k_t k_tᵀ
(в отличие от VSA-суперпозиции векторов: M хранит парные корреляции k-пространства).
Чтение — y = (q_tᵀ M_t)/√Dh → W_read. Затухание d_t выводится из τ слоя
(единый язык, τ_api), запись i_t — контентный гейт; гейты и затухание — per-head
(каждая голова учит свою модуляцию поверх общей τ-базы слоя).

Обучение — chunked log-space скан (стабильная форма, как в VSA-скане),
инференс — одношаговая рекуррентность; эквивалентность скана и рекуррентности
закреплена тестами (включая B>1 и handoff forward↔step).

Интеграция: EVABlock, ветвь за cfg.cov_memory (default off; state — 6-й
элемент per-layer state, W_out zero-init ⇒ включение бит-в-бит residual).
Осторожно: при zero-init выхода градиент в k/q/W_read на ПЕРВОМ шаге равен 0
(учится только выходная проекция) — класс phantom_mix (M64.10); со второго
шага путь открывается. Это ожидаемо и покрыто тестом liveness.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from .tau_api import TAU_MIN


class CovarianceMemory(nn.Module):
    """Multi-head ковариационная память с τ-затуханием (per-head модуляция).

    Args:
        D: ширина входа/выхода.
        n_heads: число голов.
        head_dim: размерность головы (k/q пространство).
        tau: постоянная времени слоя (затухание d = exp(−1/τ)); живой τ_l
            обновляется блоком в forward.
        chunk: длина чанка для параллельного скана.
        rank: ранг низкорангового выхода (0 = полный D×D; r>0 экономит
            ~D²−2·D·r параметров на слой).
    """

    def __init__(self, D: int, n_heads: int = 4, head_dim: int = 32,
                 tau: float = 64.0, chunk: int = 64, rank: int = 128) -> None:
        super().__init__()
        self.D = int(D)
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.tau = float(max(tau, TAU_MIN))
        self.chunk = int(chunk)
        Hd = self.n_heads * self.head_dim
        self.k_proj = nn.Linear(D, Hd)
        self.q_proj = nn.Linear(D, Hd)
        # контентные per-head гейты (как в VSA-регистрах): d_mod укорачивает
        # память, i_gate задаёт силу записи; покой = полный горизонт τ
        self.w_d = nn.Parameter(torch.zeros(D, self.n_heads))
        self.b_d = nn.Parameter(torch.zeros(self.n_heads))
        self.w_i = nn.Parameter(torch.zeros(D, self.n_heads))
        self.b_i = nn.Parameter(torch.zeros(self.n_heads))
        self.W_read = nn.Linear(Hd, D, bias=False)
        # Выход: zero-init ⇒ forward стартует бит-в-бит как residual; при
        # rank>0 ноль стоит на W_out_b, а W_out_a — случайный ⇒ градиент
        # в выходную проекцию жив с первого шага.
        self.W_out_a = self.W_out_b = None
        if int(rank) > 0:
            self.W_out_a = nn.Linear(int(rank), D, bias=False)
            self.W_out_b = nn.Linear(D, int(rank), bias=False)
            nn.init.zeros_(self.W_out_b.weight)
        else:
            self.W_out = nn.Linear(D, D, bias=False)
            nn.init.zeros_(self.W_out.weight)

    def _out(self, y: torch.Tensor) -> torch.Tensor:
        if self.W_out_b is not None:
            return self.W_out_a(self.W_out_b(y))
        return self.W_out(y)

    # ─────────────────────────── decay / gates ───────────────────────────

    def _log_decay(self, x: torch.Tensor) -> torch.Tensor:
        """log d_t (B, L, H): τ-горизонт, укороченный контентом (d_mod ∈ (0,1])."""
        d_mod = torch.sigmoid(x @ self.w_d + self.b_d) / torch.sigmoid(self.b_d)
        d_mod = d_mod.clamp(max=1.0)
        return (-1.0 / self.tau) + torch.log(d_mod.clamp_min(1e-6))

    def _write_gate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x @ self.w_i + self.b_i)

    @staticmethod
    def _norm_state(state: torch.Tensor | None, B: int, H: int, Dh: int,
                    like: torch.Tensor) -> torch.Tensor:
        """Приводит state к (B, H, Dh, Dh): None → нули, 5-D → squeeze(1)."""
        if state is None:
            return like.new_zeros(B, H, Dh, Dh)
        if state.dim() == 5:
            state = state.squeeze(1)
        return state

    # ─────────────────────────── chunked scan ───────────────────────────

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        """x: (B, L, D). state: (B, H, Dh, Dh) или None.
        Возвращает (y (B, L, D), new_state (B, H, Dh, Dh))."""
        B, L, _ = x.shape
        H, Dh = self.n_heads, self.head_dim
        k = self.k_proj(x).reshape(B, L, H, Dh)
        q = self.q_proj(x).reshape(B, L, H, Dh)
        log_d = self._log_decay(x)          # (B, L, H)
        i_g = self._write_gate(x)           # (B, L, H)
        M = self._norm_state(state, B, H, Dh, x)
        scale = 1.0 / math.sqrt(Dh)

        C = min(self.chunk, L)
        outs = []
        for c0 in range(0, L, C):
            c1 = min(c0 + C, L)
            A = torch.cumsum(log_d[:, c0:c1], dim=1)          # (B, c, H), A ≤ 0
            A_last = A[:, -1].view(B, 1, H)                   # (B, 1, H)
            kc = k[:, c0:c1]                                  # (B, c, H, Dh)
            qc = q[:, c0:c1]
            ic = i_g[:, c0:c1]                                # (B, c, H)
            # стабильная форма: P'_t = Σ_{s≤t} exp(A_last − A_s)·i_s·k_s k_sᵀ
            # (вес ≤ 1), локальный вклад = exp(A_t − A_last)·P'_t
            w = (A_last - A).exp() * ic                       # (B, c, H)
            kk = kc.unsqueeze(-1) * kc.unsqueeze(-2)          # (B, c, H, Dh, Dh)
            P = torch.cumsum(w.view(B, c1 - c0, H, 1, 1) * kk, dim=1)
            M_t = (M.unsqueeze(1) * A.exp().view(B, c1 - c0, H, 1, 1)
                   + P * (A - A_last).exp().view(B, c1 - c0, H, 1, 1))
            y = torch.einsum('bchd,bchde->bche', qc, M_t) * scale
            outs.append(y.reshape(B, c1 - c0, H * Dh))
            # состояние на конец чанка: M ← exp(A_last)·M + P'_last
            M = (M * A_last.squeeze(1).view(B, H, 1, 1).exp()) + P[:, -1]
        y = torch.cat(outs, dim=1)
        return self._out(self.W_read(y)), M.detach()

    # ─────────────────────────── streaming ───────────────────────────

    def step(self, x_t: torch.Tensor, state: torch.Tensor | None = None):
        """Одношаговая рекуррентность (инференс): x_t (B, 1, D).
        state: (B, H, Dh, Dh) или None; возвращает (y (B, 1, D), state)."""
        B = x_t.shape[0]
        H, Dh = self.n_heads, self.head_dim
        k = self.k_proj(x_t).reshape(B, 1, H, Dh)
        q = self.q_proj(x_t).reshape(B, 1, H, Dh)
        d = self._log_decay(x_t).exp().view(B, H, 1, 1)        # (B, H, 1, 1)
        i_g = self._write_gate(x_t).view(B, H, 1, 1)           # (B, H, 1, 1)
        M = self._norm_state(state, B, H, Dh, x_t)
        kk = k.squeeze(1).unsqueeze(-1) * k.squeeze(1).unsqueeze(-2)  # (B,H,Dh,Dh)
        M = M * d + kk * i_g
        y = torch.einsum('bhd,bhde->bhe', q.squeeze(1), M).reshape(B, 1, H * Dh)
        y = y * (1.0 / math.sqrt(Dh))
        return self._out(self.W_read(y)), M.detach()
