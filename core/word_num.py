"""WORD ARITHMETIC — буквенно-числовое кодирование слов (см. docs/WORD_ARITHMETIC.md).

Буква → простое число (φ), слово → произведение простых (N — состав, порядок
букв теряется) или полином (V — порядок сохраняется). Операции:
  - делимость     : включение букв (морфологическое родство)
  - НОД/НОК       : пересечение/объединение состава
  - morph_sim(M)  : близость состава в [0, 1] (лог-пространство)

Кодирование не требует обучения и работает на нормализованных леммах
(ё → е), поэтому не затрагивает чекпоинты WideBind и формат трафарета.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

ALPHABET = 'абвгдежзийклмнопрстуфхцчшщъыьэюя'

_PRIMES: List[int] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137,
]

PHI: Dict[str, int] = dict(zip(ALPHABET, _PRIMES))
INV: Dict[int, str] = {p: c for c, p in PHI.items()}

# Порядок букв теряется в N; V кодирует порядок полиномом (B > max φ)
_BASE = max(_PRIMES) + 1
_CODE = {c: i + 1 for i, c in enumerate(ALPHABET)}

_LETTERS_ONLY = set(ALPHABET) | {'ё'}


def _norm(w: str) -> str:
    return ''.join(c if c != 'ё' else 'е' for c in w.lower() if c in _LETTERS_ONLY)


def n_of(word: str) -> int:
    """Состав слова: произведение φ(букв). Анаграммы равны. Порядок теряется."""
    n = 1
    for c in _norm(word):
        n *= PHI[c]
    return n


def v_of(word: str) -> int:
    """Порядок слова: полином V = Σ code(c)·Bⁱ. Уникален для упорядоченного слова."""
    v = 0
    for i, c in enumerate(_norm(word)):
        v += _CODE[c] * (_BASE ** i)
    return v


def factors(n: int) -> str:
    """Восстановление состава из числа (факторизация)."""
    out = []
    for p in _PRIMES:
        while n % p == 0:
            out.append(INV[p])
            n //= p
        if n == 1:
            break
    return ''.join(out)


def gcd_of(a: str, b: str) -> int:
    return math.gcd(n_of(a), n_of(b))


def lcm_of(a: str, b: str) -> int:
    return math.lcm(n_of(a), n_of(b))


def morph_sim(a: str, b: str) -> float:
    """Морфологическая близость состава в [0, 1]:
    M = 2·log НОД / (log N₁ + log N₂). 1 = одинаковый состав, 0 = нет общих букв."""
    na, nb = n_of(a), n_of(b)
    if na <= 1 or nb <= 1:
        return 0.0
    g = math.gcd(na, nb)
    if g <= 1:
        return 0.0
    return min(1.0, 2.0 * math.log(g) / (math.log(na) + math.log(nb)))


def log_size(word: str) -> float:
    """Логарифмический «размер состава» (аддитивен по буквам)."""
    n = n_of(word)
    return math.log(n) if n > 1 else 0.0