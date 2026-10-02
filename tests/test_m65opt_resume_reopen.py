"""M65-opt батч 7: замок восстановленной проводки mlp_mod_scale_reopen.

Поле конфига документировалось («на резюме выставить log(3) — гейт открыт»),
но нигде не читалось (потерянная проводка). Контракт: значение <= 0 — no-op
(обычное резюме не меняется молча), > 0 — гейт mod_scale_mlp всех слоёв
выставляется в это значение.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.config import EVAConfig      # noqa: E402
from core.stack import EVAStack        # noqa: E402


def _model():
    cfg = EVAConfig(n_layers=2, D=128, mlp_groups=4, code_dim=16,
                    code_sparsity=4, vocab=256, logit_cache_enabled=False,
                    gradient_checkpointing=False, save_dir='.')
    torch.manual_seed(0)
    return EVAStack(cfg)


def test_zero_is_noop_and_value_reopens_all_layers():
    m = _model()
    mir = m.layers[0].mirror
    with torch.no_grad():
        mir.mod_scale_mlp.data.fill_(math.log(2.0))   # «спящий» инит
    m.cfg.mlp_mod_scale_reopen = 0.0
    assert m.apply_resume_reopen() == 0
    assert abs(float(mir.mod_scale_mlp.mean()) - math.log(2.0)) < 1e-6, \
        '0 обязан НЕ менять гейт (обычное резюме не трогаем)'

    m.cfg.mlp_mod_scale_reopen = math.log(3.0)
    n = m.apply_resume_reopen()
    assert n == len(m.layers)
    for l in m.layers:
        assert abs(float(l.mirror.mod_scale_mlp.mean()) - math.log(3.0)) < 1e-6, \
            'гейт не переоткрыт'


def test_default_is_off():
    cfg = EVAConfig(n_layers=1, D=64, mlp_groups=2, code_dim=16,
                    code_sparsity=4, vocab=64)
    assert cfg.mlp_mod_scale_reopen == 0.0, 'дефолт обязан быть инертным'
