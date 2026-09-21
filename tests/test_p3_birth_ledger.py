"""P3-2: the BirthLedger — the MDL arithmetic, the blacklist, the excise/retire."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.birth_ledger import BirthLedger       # noqa: E402
from core.config import EVAConfig               # noqa: E402
from core.stack import EVAStack                 # noqa: E402


def _model():
    torch.manual_seed(0)
    cfg = EVAConfig(D=64, vocab=64, n_layers=2, gradient_checkpointing=False,
                    logit_cache_enabled=False)
    m = EVAStack(cfg)
    m.eval()
    return m, cfg


def test_mdl_break_even_arithmetic():
    led = BirthLedger()
    r = 2560 + 64                     # a phantom bit: D + K
    n = 2000 * 384                    # 2000 steps x 384 tokens
    be = led.lam * r * math.log(n) / n
    assert abs(be - 0.0463) < 0.002, f'break-even {be:.4f} nat/token'
    assert abs(led.mdl(be, n, r)) < 1.0, 'at the break-even the verdict is ~0'
    assert led.mdl(0.0, n, r) < 0, 'a zero-effect newborn must not pay'
    # a UCL concept: D + bridge_dim
    be_ucl = led.lam * (2560 + 256) * math.log(n) / n
    assert abs(be_ucl - 0.0497) < 0.003, f'UCL break-even {be_ucl:.4f}'


def test_blacklist_blocks_rebirth():
    led = BirthLedger()
    d = torch.randn(64)
    led.blacklist.append(dict(d=torch.nn.functional.normalize(d, dim=-1), until=1000))
    assert led.allow_birth(d, step=500) is False
    assert led.allow_birth(d, step=1001) is True          # cooldown expired
    other = torch.randn(64)
    assert led.allow_birth(other, step=500) is True       # an orthogonal direction is free


def test_excise_and_retire_are_exact():
    m, cfg = _model()
    head = m.lm_head
    assert head.Kp > 0
    led = BirthLedger()
    j = 3
    d = head.phantom_basis.data[j].clone()
    led.record('phantom_bit', d, step=0, r=64 + 32, slot=j)
    with torch.no_grad():
        head.phantom_mix.data[:, j].normal_(0.0, 0.1)
        mix_before = head.phantom_mix.data[:, j].clone()
        basis_before = head.phantom_basis.data[j].clone()
    undo = led._excise(m, led.entries[0])
    assert undo is not None
    assert float(head.phantom_mix.data[:, j].abs().max()) == 0.0
    undo()
    assert torch.equal(head.phantom_mix.data[:, j], mix_before)
    led._retire(m, led.entries[0])
    led.entries[0]['retired'] = True      # measure() sets this before _retire
    assert float(head.phantom_mix.data[:, j].abs().max()) == 0.0
    assert float(head.phantom_basis.data[j].abs().max()) == 0.0
    assert led.free_rows('phantom_bit') == [j]


def test_measure_gives_a_verdict_and_restores():
    m, cfg = _model()
    head = m.lm_head
    j = 2
    d = head.phantom_basis.data[j].clone()
    led = BirthLedger(horizon_steps=1, dwell=99)          # never retires here
    led.record('phantom_bit', d, step=0, r=64 + 32, slot=j)
    x = torch.randint(3, cfg.vocab, (1, 16))

    def probe(xx, yy):
        with torch.no_grad():
            h = m.embed_tokens(xx)
            out, *_ = m(h, None, global_state=None, adaptive=False, step=1, tokens=xx)
            ce, _ = m.compute_losses(out[:, :-1], yy[:, 1:])
        return float(ce)

    before = {k: v.clone() for k, v in m.state_dict().items()}
    out = led.measure(m, probe, (x, x.clone()), step=10, tokens_per_step=384)
    assert out.get('phantom_bit') == 1
    assert len(led.entries[0]['verdicts']) == 1
    after = m.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), f'{k} changed across the birth measurement'
    assert led.stats()['births'] == 1
    sd = led.state_dict()
    led2 = BirthLedger()
    led2.load_state_dict(sd)
    assert led2.stats() == led.stats()
