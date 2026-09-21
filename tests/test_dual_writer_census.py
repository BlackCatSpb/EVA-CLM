"""P3-3: every `.data` writer of a parameter-class leaf must be registered.

The AST scanner mirrors test_tau_lint's discipline: a new control-law write
without a registry entry turns the test red. `BUFFER_WRITERS` holds the
non-parameter `.data` writes (the tie value mirrors) — they are reported
knowingly, not failed.
"""
import ast
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from core.param_writers import DUAL_WRITERS, BUFFER_WRITERS, TAU_ADAM, SEPARATION  # noqa: E402

MUTATORS = {'copy_', 'lerp_', 'add_', 'mul_', 'fill_', 'zero_', 'sub_', 'clamp_'}
PARAM_LEAVES = set(DUAL_WRITERS) | set(BUFFER_WRITERS) | {
    'readout', 'basis', 'embed_mix', 'log_scale', 'log_temp',
}


def _chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return '.'.join(reversed(parts))


def _scan():
    hits = []
    root = pathlib.Path(__file__).resolve().parent.parent / 'core'
    for p in sorted(root.glob('*.py')):
        tree = ast.parse(p.read_text(encoding='utf-8-sig'))
        for n in ast.walk(tree):
            tgt = None
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in MUTATORS):
                tgt = n.func.value
            elif isinstance(n, ast.Assign):
                tgt = next((t for t in n.targets if isinstance(t, ast.Subscript)), None)
            if tgt is None:
                continue
            ch = _chain(tgt)
            if '.data' not in ch:
                continue
            leaf = ch.split('.data')[0].split('.')[-1]
            if leaf in PARAM_LEAVES:
                hits.append((p.name, n.lineno, ch, leaf))
    return hits


def test_every_dual_writer_registered():
    unreg = [(f, ln, ch) for f, ln, ch, leaf in _scan()
             if leaf not in DUAL_WRITERS and leaf not in BUFFER_WRITERS]
    assert not unreg, (f'unregistered parameter writers (register them in '
                       f'core/param_writers.py or remove): {unreg}')


def test_timescale_separation():
    for leaf, (writer, tau, note) in DUAL_WRITERS.items():
        if tau is None:
            assert note, f'{leaf}: quasi-static without a justification'
            continue
        assert tau >= SEPARATION * TAU_ADAM, \
            f'{leaf}: tau_ctrl={tau} < {SEPARATION * TAU_ADAM} (the two-timescale scheme is not separable)'


def test_control_writers_alive():
    """Runtime canary: the slow loop really writes (while the optimizer is alive)."""
    import sys as _s
    _s.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from core.config import EVAConfig
    from core.stack import EVAStack
    torch.manual_seed(0)
    cfg = EVAConfig(D=64, vocab=64, n_layers=2, gradient_checkpointing=False,
                    logit_cache_enabled=False)
    m = EVAStack(cfg).train()
    b0 = m.layers[0].b_d.detach().clone()
    x = torch.randint(3, cfg.vocab, (1, 16))
    h = m.embed_tokens(x)
    m(h, None, step=1, tokens=x, adaptive=True)
    m(h, None, step=2, tokens=x, adaptive=True)
    assert not torch.equal(b0, m.layers[0].b_d.detach()), 'the b_d lerp is dead'
