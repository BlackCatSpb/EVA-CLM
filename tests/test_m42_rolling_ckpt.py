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


def test_cell10_periodic_rolling_checkpoint_and_flush():
    s = _cell10()
    assert 'def _envelope(_step):' in s and 'def _atomic_save(env, name):' in s
    assert 'step % 495 == 0' in s
    assert "_atomic_save(_envelope(step), 'latest.pt')" in s
    assert 'model.reset_cache()' in s and 'torch.cuda.empty_cache()' in s
    assert 'val_history.jsonl' in s
    # the best save must go through the shared builder now (no duplicated dict)
    assert s.count("'code_fp': codebook_fingerprint(model)") == 1, 'two envelope copies?'


def test_resume_prefers_latest():
    s = _cell8()
    assert "latest.pt" in s and 'resume source: latest.pt' in s
    t = open(os.path.join(ROOT, 'scripts', 'train.py'),
             encoding='utf-8', errors='replace').read()
    assert "'latest.pt'" in t and '_full_env' in t and '_atomic42' in t
    assert 'step % 495 == 0' in t
