"""τ-api — единый язык временных констант EVA (T8).

Один ТИП языка на класс константы:
  - τ       — горизонты: EMA-α = 1 − 1/τ (horizon), decay по лестнице;
  - λ       — масштабы (амплитуды/скорости записи) — задаются вне τ-api;
  - каденция — 1 − 1/период (period(n)): горизонты статистик, не τ-релевантные;
  - calibrated/design — оправданные нерутинные значения (реестр test_tau_lint).

Все новые временные константы обязаны приходить отсюда (τlint следит за этим).
Числа канонических дефолтов живут ТОЛЬКО здесь: TauConfig, VSA-база,
fallback-лестница adaptive_controller, config-дефолты EMA — ссылаются сюда.
"""
from __future__ import annotations

import math

# ─── Канонические границы τ-лестницы (семейство 2³…2⁹, середина = mem_tau_ref) ───
TAU_MIN: float = 8.0
TAU_MAX: float = 512.0
MEM_TAU_REF: float = 64.0

# ─── Каноническое расписание зрелости (design: ramp/safety-floor) ───
T0: float = 8000.0
T_DELAY: float = 8000.0
DELTA_T: float = 4000.0

# ─── Канонические температуры гейтов (геом. интерполяция) ───
GATE_TAU_MIN: float = 0.3
GATE_TAU_MAX: float = 5.0

# ─── Канонический LLRD ───
LLRD_GAMMA: float = 0.65

# ─── VSA-база (единый источник; block-fallback и stack-init ссылаются сюда) ───
VSA_LADDER: tuple = (8.0, 32.0, 128.0, 512.0)


def horizon(tau: float) -> float:
    """EMA-горизонт: α = 1 − 1/τ (v3, единственная формула «α»)."""
    return 1.0 - 1.0 / max(float(tau), 2.0)


def period(n: float) -> float:
    """Каденционный горизонт статистики: 1 − 1/n (n = период в шагах/наблюдениях)."""
    return 1.0 - 1.0 / max(float(n), 1.0 + 1e-9)


def temp(mat_gate: float, tau_min: float = GATE_TAU_MIN,
         tau_max: float = GATE_TAU_MAX) -> float:
    """Гейт-температура: геометрическая интерполяция tau_max→tau_min по mat_gate."""
    log_min = math.log(tau_min)
    log_max = math.log(tau_max)
    return math.exp(log_max + (log_min - log_max) * float(mat_gate))
