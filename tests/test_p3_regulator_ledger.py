"""P3-1: the RegulatorLedger — dead/alive classification and bit-exact reversibility."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig                  # noqa: E402
from core.regulator_ledger import RegulatorLedger  # noqa: E402
from core.stack import EVAStack                    # noqa: E402


def _model():
    torch.manual_seed(0)
    cfg = EVAConfig(D=64, vocab=64, n_layers=2, gradient_checkpointing=False,
                    logit_cache_enabled=True)
    m = EVAStack(cfg)
    m.eval()
    return m, cfg


def _probe_factory(m):
    def probe(x, y):
        with torch.no_grad():
            h = m.embed_tokens(x)
            out, *_ = m(h, None, global_state=None, adaptive=False, step=1, tokens=x)
            ce, _ = m.compute_losses(out[:, :-1], y[:, 1:])
        return float(ce)
    return probe


def _batches(cfg):
    torch.manual_seed(3)
    xa = torch.randint(3, cfg.vocab, (1, 16))
    xb = torch.randint(3, cfg.vocab, (1, 16))
    return [(xa, xa.clone()), (xb, xb.clone())]


def test_ledger_identity_is_reversible():
    m, cfg = _model()
    led = RegulatorLedger(m, cfg, probe=_probe_factory(m),
                          per_round=10_000, dwell=1)
    led.batches = _batches(cfg)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    cfg_before = (cfg.w_mem2v_scale_min, cfg.w_mem2v_scale_max)
    led.measure_round(m)                      # all regulators in one round
    after = m.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), f'{k} changed across the ledger round'
    assert (cfg.w_mem2v_scale_min, cfg.w_mem2v_scale_max) == cfg_before
    assert not any(hasattr(l, '_damp_on') for l in m.layers)
    assert not any(hasattr(l, '_pen_decay_on') for l in m.layers)


def test_ledger_classifies_dead_and_alive():
    m, cfg = _model()
    batches = _batches(cfg)
    # alive by construction: the token_bias is set to the batch's log-unigram
    # (the optimal bias), so zeroing it must RAISE the CE (delta > 0 -> ACTIVE)
    tb = m.lm_head.token_bias
    cnt = torch.bincount(batches[0][0].reshape(-1), minlength=cfg.vocab).float()
    with torch.no_grad():
        tb.copy_((cnt / cnt.sum().clamp_min(1.0)).clamp_min(1e-6).log())
    box = {}

    def _alive_enter():
        box['v'] = tb.detach().clone()
        tb.data.zero_()

    def _alive_leave():
        tb.data.copy_(box['v'])

    from core.regulator_ledger import Reg
    regs = [Reg('dead_noop', lambda: None, lambda: None),
            Reg('alive_unigram', _alive_enter, _alive_leave)]
    led = RegulatorLedger(m, cfg, probe=_probe_factory(m), regs=regs,
                          per_round=10_000, dwell=2)
    led.batches = batches
    for _ in range(3):
        led.measure_round(m)
    st = led.state
    assert st['dead_noop']['status'] == 'DORMANT', st['dead_noop']
    assert st['alive_unigram']['status'] == 'ACTIVE', st['alive_unigram']
    assert 'dead_noop' in led.suggestions()
    assert torch.equal(tb, box['v']), 'the alive clamp must restore'


def test_default_registry_is_reversible_and_builds():
    from core.regulator_ledger import build_registry
    m, cfg = _model()
    regs = build_registry(m, cfg)
    names = {r.name for r in regs}
    assert {'logit_cache', 'vpm', 'spec_damp', 'pen_decay', 'mem2v_adapt'} <= names
    for r in regs:
        r.identity()
        r.restore()


def test_ledger_state_roundtrip():
    m, cfg = _model()
    led = RegulatorLedger(m, cfg, probe=_probe_factory(m), per_round=3)
    led.batches = _batches(cfg)
    led.measure_round(m)
    sd = led.state_dict()
    led2 = RegulatorLedger(m, cfg, probe=_probe_factory(m), per_round=3)
    led2.batches = _batches(cfg)
    led2.load_state_dict(sd)
    assert led2.ptr == led.ptr
    assert led2.sigma_re == led.sigma_re
    assert set(led2.state) == set(led.state)
