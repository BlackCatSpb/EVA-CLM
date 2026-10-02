"""Projector: сегментация слов по событиям записи концептов (M65-opt).

Ранее core/projector.py не имел ни одного теста (аудит ошибочно счёл модуль
мёртвым — его использует scripts/proj_read.py). Замки: границы сегментов,
декодирование спанов, id концепта на конце слова.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.projector import Projector      # noqa: E402


class _Tok:
    """Заглушка токенизатора: decode(ids) -> 't<ids>'."""

    def decode(self, ids):
        return 't' + '_'.join(str(int(i)) for i in ids)


def test_segment_boundaries():
    p = Projector(_Tok())
    we = torch.zeros(1, 6, dtype=torch.bool)
    we[0, 2] = True                      # слово [0..2], затем хвост [3..5]
    spans = p.segment(we)
    assert spans == [[(0, 3), (3, 6)]], spans
    # последняя позиция всегда закрывает сегмент, даже без события
    we2 = torch.zeros(1, 4, dtype=torch.bool)
    assert p.segment(we2) == [[(0, 4)]]
    # батч из двух строк
    we3 = torch.zeros(2, 3, dtype=torch.bool)
    we3[0, 0] = True
    we3[1, 1] = True
    assert p.segment(we3) == [[(0, 1), (1, 3)], [(0, 2), (2, 3)]]


def test_read_words_and_concept_spans():
    p = Projector(_Tok())
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    we = torch.zeros(1, 5, dtype=torch.bool)
    we[0, 1] = True
    words = p.read_words(ids, we)
    assert words == [['t1_2', 't3_4_5']], words
    cid = torch.tensor([[9, 9, 7, 7, 7]])
    assert p.concept_spans(cid, we) == [[9, 7]]
