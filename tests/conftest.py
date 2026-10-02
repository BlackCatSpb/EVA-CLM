"""Общие фикстуры тестов (M65-opt).

Аудит: в tests/ не было conftest.py, и 4 файла (test_math_audit, test_model,
test_m50_stream_fuse, test_t9_ckpt_io) использовали случайные тензоры БЕЗ
сида — падения/прохождения могли зависеть от розыгрыша. Автосид на каждый
тест делает весь набор детерминированным.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _deterministic_seed():
    torch.manual_seed(0)
    yield
