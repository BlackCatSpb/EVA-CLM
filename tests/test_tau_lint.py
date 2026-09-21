"""T8: τlint — реестр τ-конвенций и сторож временных констант.

Принцип «один язык» (T8): один ТИП языка на класс константы:
  - τ  — горизонты (α = 1−1/τ_l, decay по лестнице 8..512);
  - λ  — масштабы (амплитуды/скорости записи);
  - каденция — 1−1/период (горизонты статистик, не τ-релевантные);
  - calibrated — откалибровано по прогону/каденции (оправдание обязательно);
  - design — осознанный дизайн-выбор (оправдание обязательно);
  - dead — контракт без живого потребителя (документировано).

Тесты:
  1) реестр свежий — каждый пункт всё ещё в коде (починка = обновление реестра);
  2) файлы с «жёсткой α» (0.9/0.1, 0.99, 0.999) в живом коде — обязаны быть в
     реестре (новый файл с магической α → FAIL);
  3) файлы с clamp-диапазоном [0.1,10] — обязаны быть в реестре;
  4) канонический докстринг τ_config: формула α — v3, v2-ложь отсутствует.
"""
import re
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent / 'core'

# ─── Реестр: (файл, маркер) → (класс, оправдание) ────────────────────────────
REGISTRY = {
    ('bridge.py', '0.9'): ('cadence', 'bridge-stream EMA 0.9/0.1 — T8: кандидат на A/B к 1−1/τ_l (CARRY-проба)'),
    ('block.py', '0.9'): ('cadence', 'traj-state carry 0.9/0.1 — T8: второй нарушитель (A/B)'),
    ('mirror.py', '0.85'): ('lambda', 'expert_asymmetry α-оверрайд — A/B (τ_k-init); в проде config=True'),
    ('mirror.py', '0.999'): ('cadence', 'private_mem decay — τ-горизонт (A/B)'),
    ('mirror.py', '0.99'): ('cadence', 'EMA-массы зеркала (класс горизонтов статистик, T8)'),
    ('memory_bank.py', '0.01'): ('cadence', 'L1/L2 age-decay 0.01·age — перцентильный τ (A/B)'),
    ('phantom.py', '0.99'): ('calibrated', 'phantom decay/EMA — калибровано по прогону, не трогать без A/B'),
    ('embedding.py', '0.99'): ('calibrated', 'phantom EMA — калибровано'),
    ('embedding.py', '0.05'): ('calibrated', 'phantom obs-EMA — калибровано'),
    ('losses.py', '1.0 - torch.exp'): ('tau', 'diversity-α v2 1−exp(−τ_l/τ_min) — ЖИВАЯ вторая формула «α» (A/B v2→v3)'),
    ('eva_optim.py', 'math.exp(-1.0 / tl)'): ('tau', 'b3 slow-mix 1−exp(−1/τ̄) — третья формула (A/B)'),
    ('tau_config.py', '1.0 - 1.0 / tau_l'): ('tau', 'v3 — ЕДИНЫЙ EMA-горизонт (эталон, tau_config:180)'),
    ('adaptive_controller.py', '0.166'): ('lambda', 'i_target c = λ⁻⁶·τ_ref=32 — двухъязычное произведение (A/B: τ_ref 32/64)'),
    ('adaptive_controller.py', 'tau_api.TAU_MIN'): ('tau', 'fallback-лестница — ТЕПЕРЬ из tau_api (единый источник, T8)'),
    ('tau_api.py', 'VSA_LADDER'): ('tau', 'АВТОРИТЕТ: канонические границы/лестницы/периоды (T8)'),
    ('memory_bank.py', '0.1, 10'): ('calibrated', 'clamp [0.1,10] лог-температур — калиброванный диапазон, НЕ τ-выведен'),
    ('concept_layer.py', '0.1'): ('calibrated', 'UCL clamp [0.1,10] лог-температур — тот же класс'),
    ('logit_cache.py', '0.1'): ('calibrated', 'кэш clamp [0.1,10] лог-температур — тот же класс'),
    ('tau_config.py', 'gate_tau'): ('dead', 'gate_tau-лестница — без живого потребителя с M64.5 (документировано в property)'),
    ('adaptive_gate.py', '0.1, max=10.0'): ('calibrated', 'clamp [0.1,10] обучаемого log_tau — калиброванный диапазон'),
    ('bind.py', '0.1, max=10.0'): ('calibrated', 'clamp [0.1,10] спиральной τ — калиброванный диапазон'),
    ('config.py', '0.99'): ('design', 'config-объявленные EMA-decay (head_lacuna/phantom/matur/ls_ema/…) — декларированные ноблевые значения'),
    ('lr_scheduler.py', '0.99'): ('cadence', 'ls_ema — горизонт статистики LR (конфиг-управляемый)'),
    ('maturation.py', 'matur_ema'): ('cadence', 'matur_ema — pred-error EMA (гладкость)'),
    ('stack.py', '0.999'): ('cadence', 'vsa_b_d_smooth + _bus_rms — горизонты статистик'),
    ('training_control.py', 'scale_ema_decay'): ('cadence', 'scale_ema — горизонт балансера'),
    ('param_writers.py', '0.999'): ('registry', 'P3-3: сам реестр двойных писателей — '
                                    'τ_ctrl и цитаты настроек писателей (источник истины, не магия)'),
}

HARD_ALPHA = re.compile(r'(0\.9\s*[,/]|0\.99\b|0\.999\b)')
CLAMP_010 = re.compile(r'clamp\s*\([^)]*0\.1[^)]*10|clamp\s*\([^)]*10[^)]*0\.1')
OPTIMIZER_PARAMS = ('betas', 'weight_decay', 'llrd_decay', 'eps', 'lr=')


def _is_comment(line):
    s = line.lstrip()
    return s.startswith('#')


def _load(fname):
    return (CORE / fname).read_text(encoding='utf-8', errors='replace')


def _live_lines(fname):
    for i, line in enumerate(_load(fname).splitlines(), 1):
        if _is_comment(line):
            continue
        if any(p in line for p in OPTIMIZER_PARAMS):
            continue
        yield i, line


def test_registry_is_fresh():
    """Каждый пункт реестра всё ещё присутствует в коде."""
    missing = []
    for (fname, marker), (cls, why) in REGISTRY.items():
        if marker not in _load(fname):
            missing.append(f'{fname}:{marker!r} ({cls} — {why[:40]}…)')
    assert not missing, 'Реестр устарел (пункты исчезли из кода):\n' + '\n'.join(missing)


def test_hard_alpha_files_registered():
    """Файлы с жёсткой α в живом коде обязаны быть в реестре."""
    violations = []
    for f in sorted(CORE.glob('*.py')):
        if any(HARD_ALPHA.search(line) for _, line in _live_lines(f.name)):
            if not any(fn == f.name for fn, _ in REGISTRY):
                violations.append(f'{f.name} — жёсткая α вне реестра (T8: зарегистрировать или τ-api)')
    assert not violations, '\n'.join(violations)


def test_clamp_010_files_registered():
    """Файлы с clamp [0.1,10] в живом коде обязаны быть в реестре."""
    violations = []
    for f in sorted(CORE.glob('*.py')):
        if any(CLAMP_010.search(line) for _, line in _live_lines(f.name)):
            if not any(fn == f.name for fn, _ in REGISTRY):
                violations.append(f'{f.name} — clamp [0.1,10] вне реестра (T8)')
    assert not violations, '\n'.join(violations)


def test_tau_config_docstring_v3():
    """Канонический докстринг: формула «α» — v3; v2-ложь отсутствует."""
    text = _load('tau_config.py')
    assert '1 − 1/τ_l' in text, 'докстринг tau_config не содержит v3-формулу'
    assert '1 − exp(−τ_l / τ_min)' not in text, 'докстринг tau_config всё ещё утверждает v2'
