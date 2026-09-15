"""M42 locks: rolling latest.pt + cache flush + val_history, shared envelope."""
import json
import os

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
    assert 'step % 495 == 0' in s
    assert 'model.reset_cache()' in s and 'torch.cuda.empty_cache()' in s
    assert 'val_history.jsonl' in s
    # operator: the checkpoint name is best, as before M42 — no latest.pt
    assert 'latest.pt' not in s
    # the best save must go through the shared builder now (no duplicated dict)
    assert s.count("'code_fp': codebook_fingerprint(model)") == 1, 'two envelope copies?'


def test_resume_prefers_best_only():
    s = _cell8()
    assert 'best.pt' in s and 'latest.pt' not in s
    t = open(os.path.join(ROOT, 'scripts', 'train.py'),
             encoding='utf-8', errors='replace').read()
    assert "'best.pt'" in t and "'latest.pt'" not in t
    assert 'step % 495 == 0' in t   # the flush stays
