"""B7: locks for audit-01 HIGH findings (docs/audit/01_data_path.md)."""
import importlib.util
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import EVAConfig, EVAStack                      # noqa: E402
from core.training_control import hard_veto_ceiling       # noqa: E402


def _mini(**kw):
    cfg = EVAConfig(n_layers=2, D=256, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=600, save_dir='.', **kw)
    torch.manual_seed(0)
    return EVAStack(cfg).train()


def test_f07_ceiling_is_geometry_free_and_history_consistent():
    # The M14-era calibration was K*ln2 = 32*0.693 = 22.17 at code_dim=32;
    # 2*ln(V) at the real vocab must reproduce it (and stay put when the
    # code geometry moves to 64 — the exact bug F-07 describes).
    assert abs(hard_veto_ceiling(65536) - 2 * math.log(65536)) < 1e-9
    assert abs(hard_veto_ceiling(65536) - 22.18) < 0.05
    assert hard_veto_ceiling(65536) < 34.0        # the live garbage class is caught
    assert hard_veto_ceiling(50000) < 21.7


def test_f01_tokenstream_mismatch_is_loud(tmp_path):
    spec = importlib.util.spec_from_file_location(
        '_tr_mod', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'scripts', 'train.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                        # __main__-guarded, no side effects
    f = tmp_path / 'stream.bin'
    ids = np.arange(200, dtype=np.uint16)
    ids[123] = 65535
    ids.tofile(f)
    st = mod.TokenStream(str(f))
    x, y, off, wrapped = st.get_batch(16, 1, 0, vocab=65536)   # fine: full uint16 vocab
    assert int(x.max()) <= 65535
    with pytest.raises(ValueError, match='vocab'):             # B7: no silent clip-fold
        st.get_batch(16, 1, 110, vocab=60000)


def test_f02_f03_l2_bank_content_is_state_and_eval_is_read_only():
    m = _mini(memory_bank=True, logit_cache_enabled=False, intent_bridge=False)
    buf_names = {n for n, _ in m.named_buffers()}
    par_names = {n for n, _ in m.named_parameters()}
    assert 'memory_bank.l2.keys' in buf_names
    assert 'memory_bank.l2.keys' not in par_names      # F-02: content is not a weight
    l2 = m.memory_bank.l2
    # force maturity so the write path is open (mirrors mid/late training)
    m.maturation.gate.data.fill_(1.0)
    tokens = torch.randint(3, 600, (1, 24))
    tokens[:, ::4] = 2                                  # SEPs -> sentence boundaries
    h = m.embed_tokens(tokens)

    m.eval()
    k0 = l2.keys.clone(); w0 = int(l2._write_idx.item())
    with torch.no_grad():
        m(h, None, step=None, adaptive=False, tokens=tokens)
    assert torch.equal(l2.keys, k0)                      # F-03: eval wrote nothing
    assert int(l2._write_idx.item()) == w0

    m.train()
    with torch.no_grad():
        m(h, None, step=None, adaptive=False, tokens=tokens)
    assert not torch.equal(l2.keys, k0)                  # training still writes
    assert int(l2._write_idx.item()) > w0
