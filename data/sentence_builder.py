"""SentenceBuilder — интеллектуальная сборка предложений из слов (русский язык).

Модель важности слов построена по λ_d-иерархии (см. docs/LANGUAGE_LAMBDA.md
проекта FCF): семантическое поле имеет ядро (k=0) и периферию (вес λ_d⁻ᵏ).

  k=0  — предикативное ядро: сказуемое, подлежащее, прямое дополнение
  k=1  — косвенные актанты: адресат, инструмент, локатив, время
  k=2  — модификаторы: определения (прилагательные), обстоятельства (наречия)
  k=3  — служебные: предлоги, союзы, частицы (соединители, без лексического ядра)

Сборка идёт от ядра наружу:
  1) анализ слов (pymorphy3) → леммы, части речи, падежи
  2) ранжирование важности и назначение ролей
  3) каркас: предикат → подлежащее → актанты по падежу/предлогу
  4) прикрепление модификаторов к их "хозяевам"
  5) морфологическое согласование (падеж, род, число)
  6) порядок слов: тема → сказуемое → объекты → обстоятельства
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pymorphy3

from core.word_num import morph_sim

_MORPH = pymorphy3.MorphAnalyzer()

# Обобщённое золотое сечение (d=3): язык как λ_d-машина
LAMBDA = 1.46557

# Вес глубины в семантическом поле (LANGUAGE_LAMBDA.md: ядро → периферия)
IMPORTANCE_WEIGHTS = {0: 1.0, 1: LAMBDA ** -1, 2: LAMBDA ** -2, 3: LAMBDA ** -3}

# Служебные части речи — соединители, не несущие лексического ядра
FUNCTION_POS = frozenset({'PREP', 'CONJ', 'PRCL', 'INTJ'})

# Базовая роль по падежу (без предлога)
CASE_ROLES = {
    'nomn': 'subject',
    'accs': 'direct_object',
    'gent': 'possessor',
    'datv': 'recipient',
    'ablt': 'instrument',
    'loct': 'location',
}

# Семантические роли предложных групп: (предлог, падеж) → роль
PREP_ROLES = {
    ('в', 'loct'): 'location',
    ('на', 'loct'): 'location',
    ('в', 'accs'): 'direction',
    ('на', 'accs'): 'direction',
    ('по', 'datv'): 'path',
    ('к', 'datv'): 'direction',
    ('с', 'ablt'): 'companion',
    ('у', 'gent'): 'place_at',
    ('о', 'loct'): 'topic',
    ('для', 'gent'): 'purpose',
    ('из', 'gent'): 'source',
    ('от', 'gent'): 'source',
    ('за', 'ablt'): 'place_behind',
    ('под', 'ablt'): 'place_under',
    ('над', 'ablt'): 'place_over',
    ('между', 'ablt'): 'place_between',
    ('перед', 'ablt'): 'place_front',
    ('после', 'gent'): 'time_after',
    ('до', 'gent'): 'time_before',
    ('во', 'loct'): 'time_in',
}

# Роли-актанты ядра (важность 0..1)
ROLE_IMPORTANCE = {
    'predicate': 0,
    'subject': 0,
    'direct_object': 0,
    'instrument': 1,
    'recipient': 1,
    'location': 1,
    'direction': 1,
    'path': 1,
    'possessor': 1,
    'companion': 1,
    'place_at': 1,
    'topic': 1,
    'purpose': 1,
    'source': 1,
    'place_behind': 1,
    'place_under': 1,
    'place_over': 1,
    'place_between': 1,
    'place_front': 1,
    'time_after': 1,
    'time_before': 1,
    'time_in': 1,
    'attribute': 2,
    'adverb': 2,
    'numeral': 2,
}

# Валентностные схемы частых глаголов: лемма → допустимые роли актантов.
# Используется как предпочтение при выборе, какой актант куда привязать.
VALENCY = {
    'идти': {'direction', 'path', 'location'},
    'ехать': {'direction', 'location'},
    'бежать': {'path', 'direction'},
    'лететь': {'direction', 'location'},
    'жить': {'location'},
    'находиться': {'location'},
    'спать': {'location'},
    'лежать': {'location'},
    'сидеть': {'location'},
    'стоять': {'location'},
    'класть': {'direct_object', 'location'},
    'ставить': {'direct_object', 'location'},
}


@dataclass
class WordInfo:
    text: str
    lemma: str
    pos: str
    case: Optional[str]
    number: Optional[str]
    gender: Optional[str]
    animacy: Optional[str]
    is_function: bool
    depth: int = 3
    role: str = 'function'
    importance: float = IMPORTANCE_WEIGHTS[3]
    head: Optional[int] = None
    form: str = ''


@dataclass
class SentenceResult:
    text: str
    core: List[str] = field(default_factory=list)
    periphery: List[str] = field(default_factory=list)
    roles: List[WordInfo] = field(default_factory=list)
    explanation: List[str] = field(default_factory=list)


class SentenceBuilder:
    """Собирает грамматически корректное русское предложение из набора слов.

    Если передан stencil (трафарет порядка слов из корпуса), сборка идёт
    по его статистике: выбор ядра по salience, прикрепление модификаторов
    по PMI, порядок — по реальным частеречным цепочкам корпуса.
    """

    def __init__(self, stencil=None):
        self._morph = _MORPH
        self.stencil = stencil

    def parse(self, words: List[str]) -> List[WordInfo]:
        out = []
        for w in words:
            w = w.strip().lower()
            if not w:
                continue
            p = self._morph.parse(w)[0]
            tag = p.tag
            out.append(WordInfo(
                text=w,
                lemma=p.normal_form,
                pos=str(tag.POS or 'UNKN'),
                case=getattr(tag, 'case', None),
                number=getattr(tag, 'number', None),
                gender=getattr(tag, 'gender', None),
                animacy=getattr(tag, 'animacy', None),
                is_function=bool(tag.POS in FUNCTION_POS),
            ))
        return out

    def _inflect(self, wi: WordInfo, gr: List[str]) -> str:
        try:
            p = self._morph.parse(wi.lemma)[0]
            infl = p.inflect(set(gr))
            if infl is not None and infl.word:
                return infl.word
        except Exception:
            pass
        return wi.text

    def build(self, words: List[str]) -> SentenceResult:
        parsed = self.parse(words)
        if not parsed:
            return SentenceResult(text='', core=[], periphery=[], roles=[])
        if self.stencil is not None:
            return self._build_with_stencil(parsed)

        # ── 1. Ранжирование важности и назначение ролей ──
        for wi in parsed:
            self._assign_role(wi, parsed)

        # ── 2. Каркас: предикат, подлежащее, актанты ──
        predicate = next((w for w in parsed if w.role == 'predicate'), None)
        subjects = [w for w in parsed if w.role == 'subject']
        subject = subjects[0] if subjects else None

        if predicate is not None and subject is not None:
            self._agree_verb(predicate, subject)
        if subject is not None:
            self._agree_subject_phrase(parsed, subject)

        # ── 3. Порядок слов ──
        ordered = self._order(parsed, predicate, subject)

        # ── 4. Сборка строки ──
        text = self._compose(ordered)

        core = [w.lemma for w in parsed if w.depth == 0]
        periphery = [w.lemma for w in parsed if w.depth > 0]
        explanation = self._explain(parsed, predicate, subject)

        return SentenceResult(
            text=text,
            core=core,
            periphery=periphery,
            roles=parsed,
            explanation=explanation,
        )

    # ────────────────────────────────────────────────────────────────

    # ── сборка по трафарету (stencil) ──────────────────────────────

    def _attach_score(self, wi: WordInfo, host: WordInfo, st: Stencil) -> float:
        pmi = st.pmi(wi.lemma, host.lemma)
        m = max(0.0, morph_sim(wi.lemma, host.lemma) - 0.5)
        return pmi + 0.6 * m

    def _frame_has_accs(self, frames) -> bool:
        if not frames:
            return False
        accs = sum(c for k, c in frames.items()
                   if k == 'accs' or (isinstance(k, tuple) and k[1] == 'accs'))
        total = sum(c for k, c in frames.items()
                    if k in CASE_ROLES or (isinstance(k, tuple) and k[1] in CASE_ROLES))
        return total > 0 and accs / total > 0.25

    def _build_with_stencil(self, parsed: List[WordInfo]) -> SentenceResult:
        st = self.stencil
        for wi in parsed:
            wgt, _ = st.salience(wi.lemma, morph=self._morph)
            wi.importance = wgt
            wi.head = None
            wi.role = ''
            wi.depth = 3

        # ── ядро: предикат → подлежащее → прямое дополнение ──
        verbs = [w for w in parsed if w.pos in ('VERB', 'INFN')]
        predicate = max(verbs, key=lambda w: w.importance) if verbs else None
        if predicate is not None:
            predicate.role = 'predicate'
            predicate.depth = 0
            for v in verbs:
                if v is not predicate:
                    v.role = 'aux_verb'
                    v.depth = 3

        nouns = [w for w in parsed if w.pos in ('NOUN', 'NPRO')]
        nomn = [w for w in nouns if w.case == 'nomn']
        subject = max(nomn, key=lambda w: w.importance) if nomn else None
        if subject is not None:
            subject.role = 'subject'
            subject.depth = 0
        obj = None
        if predicate is not None:
            frames = st.verb_frames.get(predicate.lemma)
            frame_accs = self._frame_has_accs(frames)
            accs = [w for w in nouns if w.case == 'accs' and w is not subject]
            if accs and (frame_accs or not frames):
                obj = max(accs, key=lambda w: w.importance)
                obj.role = 'direct_object'
                obj.depth = 0

        # ── роли остальных слов ──
        for wi in parsed:
            if wi.role:
                continue
            pos = wi.pos
            if pos in ('NOUN', 'NPRO'):
                wi.role = CASE_ROLES.get(wi.case, 'related')
                if wi.case == 'nomn':
                    wi.role = 'related'
                    wi.depth = 1
                else:
                    wi.depth = ROLE_IMPORTANCE.get(wi.role, 1)
                    if wi.role == 'direct_object':
                        if wi is not obj:
                            wi.depth = 1
                        elif predicate is None:
                            wi.depth = 1
                        else:
                            vframes = st.verb_frames.get(predicate.lemma)
                            v_accs = self._frame_has_accs(vframes)
                            if vframes and not v_accs:
                                wi.depth = 1
            elif pos in ('ADJF', 'PRTF', 'PRTS'):
                wi.role = 'attribute'
                wi.depth = 2
            elif pos == 'NUMR':
                wi.role = 'numeral'
                wi.depth = 2
            elif pos in ('ADVB', 'GRND'):
                wi.role = 'adverb'
                wi.depth = 2
            elif pos == 'PREP':
                wi.role = 'preposition'
                wi.depth = 3
            else:
                wi.role = 'function'
                wi.depth = 3

        # ── валентность предиката: привязать актанты по рамкам трафарета ──
        bound_preps = set()
        if predicate is not None:
            frames = st.verb_frames.get(predicate.lemma)
            if frames:
                preps = {w.lemma: w for w in parsed if w.pos == 'PREP'}
                noun_by_case = defaultdict(list)
                for w in parsed:
                    if w.pos in ('NOUN', 'NPRO') and w.case:
                        noun_by_case[w.case].append(w)
                for key, cnt in frames.most_common(30):
                    if isinstance(key, tuple):
                        prep_l, case = key
                        if prep_l in preps and case in noun_by_case and prep_l not in bound_preps:
                            prep = preps[prep_l]
                            target = max(noun_by_case[case], key=lambda w: w.importance)
                            prep.head = id(target)
                            target.role = 'object_of_prep'
                            target.depth = 1
                            bound_preps.add(prep_l)
                            break
                    else:
                        case = key
                        if case in noun_by_case:
                            target = max(noun_by_case[case], key=lambda w: w.importance)
                            if target.role not in ('subject', 'direct_object'):
                                target.role = 'object_of_prep'
                                target.depth = 1
                            break

        # ── предлоги: прикрепить к существительному по PMI ──
        hosts_n = [w for w in parsed if w.pos in ('NOUN', 'NPRO')]
        for prep in [w for w in parsed if w.pos == 'PREP']:
            if prep.lemma in bound_preps:
                continue
            if hosts_n:
                target = max(hosts_n, key=lambda w: st.pmi(prep.lemma, w.lemma))
                prep.head = id(target)
                if target.role in ('subject',) or target.role in CASE_ROLES.values() \
                        or target.role == 'related':
                    target.role = 'object_of_prep'
                    target.depth = 1

        # ── модификаторы: прикрепить к хозяину по max PMI ──
        hosts = [w for w in parsed if w.pos in ('NOUN', 'NPRO', 'VERB', 'INFN', 'GRND')]
        for wi in parsed:
            if wi.role in ('attribute', 'adverb', 'numeral') and hosts:
                wi.head = id(max(hosts, key=lambda h: self._attach_score(wi, h, st)))

        # ── порядок ──
        ordered = self._order_by_stencil(parsed, predicate, subject)

        # ── морфосогласование ──
        if predicate is not None and subject is not None:
            self._agree_verb(predicate, subject)
        if subject is not None:
            self._agree_subject_phrase(parsed, subject)
        self._agree_attributes(parsed)

        text = self._compose(ordered)
        core = [w.lemma for w in parsed if w.depth == 0]
        periphery = [w.lemma for w in parsed if w.depth > 0]
        explanation = self._explain(parsed, predicate, subject)
        return SentenceResult(text=text, core=core, periphery=periphery,
                              roles=parsed, explanation=explanation)

    def _order_by_stencil(self, parsed: List[WordInfo], predicate: Optional[WordInfo],
                          subject: Optional[WordInfo]) -> List[WordInfo]:
        order: List[WordInfo] = []
        used = set()

        def take(w: Optional[WordInfo]):
            if w is None or id(w) in used:
                return
            order.append(w)
            used.add(id(w))

        def emit(w: Optional[WordInfo]):
            if w is None or id(w) in used:
                return
            prep = next((p for p in parsed if p.pos == 'PREP' and p.head == id(w)), None)
            if prep is not None:
                take(prep)
            for m in parsed:
                if m is not w and m.head == id(w) and m.role in ('attribute', 'numeral'):
                    take(m)
            take(w)

        topic = subject if subject is not None else (
            max((w for w in parsed if w.pos in ('NOUN', 'NPRO')),
                key=lambda w: w.importance, default=None))
        emit(topic)
        if predicate is not None:
            take(predicate)
            for w in parsed:
                if w.role == 'adverb' and w.head == id(predicate):
                    take(w)
        for w in parsed:
            if w.role == 'direct_object':
                emit(w)
        for w in sorted((w for w in parsed
                         if id(w) not in used
                         and w.role not in ('function', 'aux_verb')),
                        key=lambda w: (-w.importance, w.depth)):
            emit(w)
        for w in parsed:
            if id(w) not in used:
                take(w)
        return order

    def _agree_attributes(self, parsed: List[WordInfo]):
        for wi in parsed:
            if wi.role not in ('attribute', 'numeral') or wi.head is None:
                continue
            if wi.role == 'numeral' and wi.lemma not in ('один', 'одна', 'одно', 'одни'):
                continue
            target = next((p for p in parsed if id(p) == wi.head), None)
            if target is None:
                continue
            gr = [target.case or 'nomn', target.number or 'sing',
                  target.gender or '']
            wi.form = self._inflect(wi, gr)

    # ────────────────────────────────────────────────────────────────

    def _assign_role(self, wi: WordInfo, all_words: List[WordInfo]):
        pos = wi.pos
        if pos == 'VERB' or (pos == 'INFN' and not any(
                w.pos == 'VERB' for w in all_words if w is not wi)):
            wi.role = 'predicate'
            wi.depth = 0
        elif pos in ('NOUN', 'NPRO', 'ADJS', 'PRTF', 'NUMR'):
            role = CASE_ROLES.get(wi.case, 'related')
            if role == 'direct_object' and wi.case == 'accs' and self._has_prep(wi, all_words):
                role = 'object_of_prep'
            wi.role = role
            wi.depth = ROLE_IMPORTANCE.get(role, 1)
        elif pos == 'ADJF':
            wi.role = 'attribute'
            wi.depth = 2
        elif pos == 'ADVB':
            wi.role = 'adverb'
            wi.depth = 2
        elif pos == 'PREP':
            wi.role = 'preposition'
            wi.depth = 3
        else:
            wi.role = 'function'
            wi.depth = 3
        wi.importance = IMPORTANCE_WEIGHTS[wi.depth]

    def _has_prep(self, wi: WordInfo, all_words: List[WordInfo]) -> bool:
        for w in all_words:
            if w.pos == 'PREP' and w.head == self._index(wi):
                return True
        return False

    @staticmethod
    def _index(wi: WordInfo) -> int:
        return id(wi)

    def _prep_group_role(self, prep: WordInfo, noun: WordInfo) -> str:
        return PREP_ROLES.get((prep.lemma, noun.case), 'location')

    def _agree_verb(self, verb: WordInfo, subject: WordInfo):
        number = subject.number
        if verb.pos == 'INFN':
            gr = ['3per', number or 'sing', 'pres']
            verb.form = self._inflect(verb, gr)
        elif verb.pos == 'VERB':
            gr = ['3per', number or 'sing']
            if subject.gender and subject.number == 'sing':
                gr += [subject.gender]
            verb.form = self._inflect(verb, gr)

    def _agree_subject_phrase(self, parsed: List[WordInfo], subject: WordInfo):
        for w in parsed:
            if w.role == 'attribute' and w.head is None:
                w.form = self._inflect(w, ['nomn', subject.number or 'sing',
                                           subject.gender or ''])
            elif w.role == 'attribute' and w.head is not None:
                target = next((p for p in parsed if p.head == w.head), None)
                if target is not None:
                    gr = [target.case or 'nomn', target.number or 'sing',
                          target.gender or '']
                    w.form = self._inflect(w, gr)

    def _order(self, parsed: List[WordInfo], predicate: WordInfo,
               subject: Optional[WordInfo]) -> List[WordInfo]:
        order: List[WordInfo] = []
        prep_groups = {w.head: w for w in parsed if w.pos == 'PREP'}

        def group_for(wi: WordInfo) -> List[WordInfo]:
            grp = []
            prep = prep_groups.get(id(wi))
            if prep is not None:
                grp.append(prep)
            for w in parsed:
                if w.role in ('attribute', 'numeral') and w.head == id(wi):
                    grp.append(w)
            grp.append(wi)
            return grp

        used = set()
        if subject is not None:
            for w in group_for(subject):
                order.append(w)
                used.add(id(w))
        if predicate is not None:
            order.append(predicate)
            used.add(id(predicate))
            for w in parsed:
                if w.role == 'adverb' and w.head == id(predicate) and id(w) not in used:
                    order.append(w)
                    used.add(id(w))
        for w in parsed:
            if id(w) in used or w.role == 'function' or w.pos in ('PREP',):
                continue
            order.append(w)
            used.add(id(w))
        for w in parsed:
            if id(w) in used:
                continue
            order.append(w)
        return order

    def _compose(self, ordered: List[WordInfo]) -> str:
        parts = []
        for wi in ordered:
            form = wi.form or wi.text
            parts.append(form)
        text = ' '.join(parts).strip()
        if text:
            text = text[0].upper() + text[1:]
            if text[-1] not in '.!?…':
                text += '.'
        return text

    def _explain(self, parsed: List[WordInfo], predicate: Optional[WordInfo],
                 subject: Optional[WordInfo]) -> List[str]:
        lines = []
        for wi in sorted(parsed, key=lambda w: (w.depth, parsed.index(w))):
            label = wi.role
            lines.append(f'{wi.text} → {label} (глубина {wi.depth}, '
                         f'важность {wi.importance:.3f})')
        if subject is not None and predicate is not None:
            lines.append(f'ядро: {subject.text} + {predicate.form or predicate.text}')
        return lines