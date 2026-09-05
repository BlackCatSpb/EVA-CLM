"""Projector — дисциплинированное считывание слов из скрытого состояния WideBind.

Прожектор не добавляет новых потерь и не вводит магических чисел: он
опирается на уже заложенную функциональность архитектуры.

Источники сигналов (существующие механизмы):
  - границы слов  = события записи концептов CollectiveConceptLayer._write_event.
    Эксперты сами выбирают, что и когда запоминать в коллективную память;
    запись концепта и есть естественная граница разметки (слова/фразы).
  - id концепта    = слот наилучшего совпадения CollectiveConceptLayer._concept_id
    (best match среди выученных концептов на K-пространстве экспертов).
  - сборка токенов = штатный токенизатор (decode спанов по границам).

Таким образом прожектор лишь ЧИТАЕТ то, что архитектура уже выучила:
структуру слов фиксируют эксперты/коллективная память/слой концептов,
а прожектор превращает скрытое состояние в связные слова.
"""

from __future__ import annotations

from tokenizers import Tokenizer


class Projector:
    """Readout слов из потока токенов по сигналам слоя концептов."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def segment(write_event):
        """write_event: (B, L) bool -> список[(start, end)] на батч (end исключительно).

        Граница слова = позиция события записи концепта; слово = спан
        от предыдущей границы (exclusive) до текущей (inclusive).
        """
        spans = []
        for b in range(write_event.shape[0]):
            we = write_event[b]
            row = []
            start = 0
            L = write_event.shape[1]
            for t in range(L):
                if we[t] or t == L - 1:
                    row.append((start, t + 1))
                    start = t + 1
            spans.append(row)
        return spans

    def read_words(self, ids, write_event):
        """ids: (B, L) long -> список[список[слово]] (декодированные спаны)."""
        spans = self.segment(write_event)
        out = []
        for b in range(ids.shape[0]):
            words = []
            for (s, e) in spans[b]:
                words.append(self.tokenizer.decode(ids[b, s:e].tolist()))
            out.append(words)
        return out

    def concept_spans(self, concept_id, write_event):
        """concept_id: (B, L) long -> id концепта на конце каждого слова."""
        spans = self.segment(write_event)
        out = []
        for b in range(concept_id.shape[0]):
            row = []
            for (s, e) in spans[b]:
                row.append(int(concept_id[b, e - 1].item()))
            out.append(row)
        return out
