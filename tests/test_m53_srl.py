"""M53 locks: the State Resolution Loop (annealed EM over the codes).

Prototype-verified semantics: a clean code -> conf 1.0 / expl ~0.03 nats/bit;
a mix of two codes -> conf 0.50 (a contradiction); random noise -> expl > thr
(a lacuna). The loop is off by default (bit-identical forward).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EVAConfig  # noqa: E402
from core import EVAStack  # noqa: E402

SMALL = dict(n_layers=2, D=512, mlp_groups=4, code_dim=16, code_sparsity=4, vocab=1820)


def _model(**kw):
    torch.manual_seed(0)
    return EVAStack(EVAConfig(**{**SMALL, **kw}))


def _argmax_code(m, u):
    C = m.lm_head.codes.float()
    return int((u.reshape(-1, u.shape[-1]) @ (2.0 * C - 1.0).T).argmax(-1)[0])


def test_srl_recovers_a_clean_code():
    m = _model().eval()
    true_idx = 7
    c = m.lm_head.codes[true_idx].float()
    torch.manual_seed(1)
    u0 = (2.0 * c - 1.0) * 4.0 + 1.5 * torch.randn(1, 1, m.lm_head.K)
    u, info = m.lm_head.srl(u0)
    assert _argmax_code(m, u) == true_idx, 'the loop did not commit to the true code'
    assert float(info['conf']) > 0.9
    assert float(info['expl']) < m.lm_head.srl_expl_thr


def test_srl_split_is_a_contradiction():
    m = _model().eval()
    c1 = m.lm_head.codes[3].float()
    c2 = m.lm_head.codes[900].float()
    u0 = ((2.0 * c1 - 1.0) + (2.0 * c2 - 1.0)) * 2.0
    _, info = m.lm_head.srl(u0.reshape(1, 1, -1))
    assert float(info['conf']) < 0.9, f"split posterior not detected: {float(info['conf'])}"


def test_srl_random_is_a_lacuna():
    # scale-free form: the threshold (0.7 nats/bit) is calibrated for the
    # production K=64; in the small stack (K=16, dense codebook) the contrast
    # is the honest test: random must cost many times a clean code.
    m = _model().eval()
    c = m.lm_head.codes[5].float()
    torch.manual_seed(4)
    u_clean = (2.0 * c - 1.0) * 4.0 + 1.0 * torch.randn(1, 1, m.lm_head.K)
    _, info_clean = m.lm_head.srl(u_clean)
    torch.manual_seed(2)
    _, info_rand = m.lm_head.srl(3.0 * torch.randn(1, 1, m.lm_head.K))
    assert float(info_rand['expl']) > 3.0 * float(info_clean['expl']), \
        f"lacuna not separated: clean={float(info_clean['expl'])} rand={float(info_rand['expl'])}"
    assert float(info_clean['expl']) < m.lm_head.srl_expl_thr


def test_srl_steps_zero_is_identity():
    m = _model().eval()
    u0 = torch.randn(1, 1, m.lm_head.K)
    u, info = m.lm_head.srl(u0, steps=0)
    assert torch.allclose(u, u0)
    assert float(info['conf']) == 1.0


def test_srl_shortlist_agrees_with_full():
    m = _model().eval()
    c = m.lm_head.codes[11].float()
    torch.manual_seed(3)
    u0 = (2.0 * c - 1.0) * 4.0 + 1.0 * torch.randn(1, 1, m.lm_head.K)
    u_full, _ = m.lm_head.srl(u0, shortlist=10 ** 6)
    u_sl, _ = m.lm_head.srl(u0, shortlist=64)
    assert torch.allclose(u_full, u_sl, atol=1e-4), 'shortlist changed the outcome'


def test_forward_runs_srl_and_reports_telemetry():
    m = _model(head_srl=True, head_srl_after=0).train()
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    h = m.embed_tokens(x)
    out, st, gs, _ = m(h, None, step=1, tokens=x)
    assert hasattr(m.lm_head, '_last_srl'), 'SRL telemetry missing'
    for k in ('conf', 'ent', 'expl'):
        assert k in m.lm_head._last_srl


def test_srl_off_by_default():
    m = _model()
    assert m.lm_head.srl_on is False


def test_srl_waits_for_the_warmup():
    # M53c: at init the SRL commits to random codes (measured: the CE doubles),
    # so it activates only from cfg.head_srl_after.
    m = _model(head_srl=True, head_srl_after=10).train()
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    h = m.embed_tokens(x)
    m(h, None, step=0, tokens=x)
    assert not hasattr(m.lm_head, '_last_srl'), 'SRL ran before the warmup'
    m(h, None, step=10, tokens=x)
    assert hasattr(m.lm_head, '_last_srl'), 'SRL did not activate at the warmup step'


def test_phantom_bank_waits_for_the_warmup():
    m = _model(head_phantom_every=1, head_phantom_after=10).train()
    x = torch.randint(1, SMALL['vocab'], (1, 8))
    h = m.embed_tokens(x)
    m(h, None, step=0, tokens=x)
    assert int(m.lm_head.phantom_bank._obs) == 0, 'the bank observed before the warmup'
    m(h, None, step=10, tokens=x)
    assert int(m.lm_head.phantom_bank._obs) >= 1, 'the bank did not observe after the warmup'
