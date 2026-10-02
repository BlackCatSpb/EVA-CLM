"""M65-opt батч 2b: поведенческие замки (mirror/concept/block/logit).

  1. `_alpha_override_py` публикуется писателем (MirrorLRScheduler.step) и
     совпадает с буфером;
  2. `_mlp_cnt_py` + reset_mlp_observer сбрасывают буфер и счётчик вместе;
  3. `_kp()` кэш среза: ленивая подтяжка и инвалидация;
  4. `_cached_ig_eff` — property (None до forward, float после);
  5. `_mature_py`: снимок зрелости согласован с буфером (ленивая подтяжка).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.config import EVAConfig               # noqa: E402
from core.lr_scheduler import MirrorLRScheduler  # noqa: E402
from core.stack import EVAStack                 # noqa: E402


def _model():
    cfg = EVAConfig(n_layers=2, D=128, mlp_groups=4, code_dim=16,
                    code_sparsity=4, vocab=256, logit_cache_enabled=False,
                    gradient_checkpointing=False, save_dir='.')
    torch.manual_seed(0)
    return EVAStack(cfg).train()


def test_alpha_override_py_is_published_by_scheduler():
    m = _model()
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    sched = MirrorLRScheduler(m, opt, warmup=10)
    sched.step()
    for l in m.layers:
        assert abs(l.mirror._alpha_override_py
                   - float(l.mirror._alpha_override)) < 1e-6, \
            'python-двойник _alpha_override рассинхронизирован с буфером'


def test_mlp_counter_and_reset_are_synced():
    m = _model()
    x = torch.randint(1, 256, (1, 16))
    m(m.embed_tokens(x), None, step=5, tokens=x)
    blk = m.layers[0]
    assert blk._mlp_cnt_py > 0
    assert int(blk._mlp_cnt.item()) == blk._mlp_cnt_py
    blk.reset_mlp_observer()
    assert blk._mlp_cnt_py == 0 and int(blk._mlp_cnt.item()) == 0
    assert float(blk._mlp_now_ema) == 0.0 and float(blk._mlp_base_ema) == 0.0


def test_kp_cache_lazy_and_invalidated():
    m = _model()
    head = m.lm_head
    assert head._kp_active_py is None
    v0 = head._kp()
    assert v0 == int(head._kp_active.item())
    with torch.no_grad():
        head._kp_active.fill_(head.Kp + 3)
    assert head._kp() == v0, 'кэш обязан не перечитывать буфер без инвалидации'
    head._kp_active_py = None
    assert head._kp() == int(head._kp_active.item()) == head.Kp + 3


def test_ig_eff_is_property_and_mature_snapshot():
    m = _model()
    mir = m.layers[0].mirror
    # property-механика (не зависит от того, активен ли intent-путь в мини-конфиге)
    mir._cached_ig_eff_t = None
    assert mir._cached_ig_eff is None
    with torch.no_grad():
        mir._cached_ig_eff_t = torch.tensor(2.0)
    assert mir._cached_ig_eff == 2.0
    x = torch.randint(1, 256, (1, 16))
    m(m.embed_tokens(x), None, step=5, tokens=x)
    assert mir._cached_ig_eff is None or isinstance(mir._cached_ig_eff, float)

    cl = getattr(m, 'concept_layer', None)
    if cl is not None:
        cl._mature_py = None                   # как после резюма
        m(m.embed_tokens(x), None, step=6, tokens=x)
        assert abs(cl._mature_py - float(cl._mature.item())) < 1e-6, \
            'снимок зрелости рассинхронизирован с буфером'
