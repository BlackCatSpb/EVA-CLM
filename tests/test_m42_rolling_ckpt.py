"""M42 locks: rolling latest.pt + cache flush + val_history, shared envelope.

Батч 6 (source-локи -> AST): проверки идут по AST-узлам (has_call/has_compare/
has_str/has_dict_entry), а не по сырым подстрокам; отсутствие имени файла
latest.pt остаётся raw-поиском (любое вхождение, включая комментарий, ловится).
"""
import json
import os

import _srclock as srclock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, 'notebooks', 'eva_colab.ipynb')


def _cell10():
    nb = json.load(open(NB, encoding='utf-8'))
    return ''.join(''.join(c.get('source', [])) for c in nb['cells']
                   if 'TRAINING LOOP' in ''.join(c.get('source', [])))


def _cell8():
    nb = json.load(open(NB, encoding='utf-8'))
    return ''.join(nb['cells'][8]['source'])


def test_cell10_periodic_flush_no_rolling_ckpt():
    s = _cell10()
    assert srclock.has_compare(s, 'step % 495', '==', '0'), 'the 495-step flush vanished'
    assert srclock.has_call(s, 'model.reset_cache') and \
        srclock.has_call(s, 'torch.cuda.empty_cache'), 'the flush body vanished'
    assert srclock.has_str(s, 'val_history.jsonl')
    # operator: the checkpoint name is best, as before M42 — no latest.pt
    assert 'latest.pt' not in s
    # the best save must go through the shared builder now (no duplicated dict):
    # exactly ONE codebook_fingerprint call in the envelope cell = one envelope.
    assert len(srclock.call_sites(s, 'codebook_fingerprint')) == 1, 'two envelope copies?'


def test_resume_prefers_best_only():
    s = _cell8()
    assert srclock.has_str(s, 'best.pt') and 'latest.pt' not in s
    t = os.path.join(ROOT, 'scripts', 'train.py')
    # строковые литералы, не подстроки: комментарии про историю latest.pt
    # (M42 post-mortem) легальны, живой путь резюма — только best.pt
    assert srclock.has_str(t, 'best.pt') and not srclock.has_str(t, 'latest.pt')
    assert srclock.has_compare(t, 'step % 495', '==', '0')   # the flush stays
