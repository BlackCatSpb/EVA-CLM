"""FCF_CPR: сжатый инференс-чекпоинт — реальные контракты (M65-opt).

Было: ручной бенчмарк без единого теста и с путём к чужому чекпоинту
(WideBind/checkpoints/step_15000_infer.pt). Стало: roundtrip-контракты
компрессора — до этого core/compression.py не имел покрытия:
  (1) декомпрессия восстанавливает веса в пределах 8-битной квантизации;
  (2) набор ключей сохраняется, реестр удаляемых буферов пересоздаётся
      (decompress_sd бросает при разрыве реестра);
  (3) сжатый файл меньше обычного сохранения;
  (4) мини-модель после roundtrip делает forward и генерацию без ошибок.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.compression import FCF_CPR, is_removable   # noqa: E402
from core.config import EVAConfig             # noqa: E402
from core.stack import EVAStack               # noqa: E402


def _mini():
    cfg = EVAConfig(n_layers=2, D=128, mlp_groups=4, code_dim=16,
                    code_sparsity=4, vocab=256, logit_cache_enabled=False,
                    gradient_checkpointing=False, save_dir='.')
    torch.manual_seed(0)
    return EVAStack(cfg)


def test_compressed_roundtrip_is_within_quantization_error(tmp_path):
    m = _mini()
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    cpr = FCF_CPR()
    path = str(tmp_path / 'mini_fcf.pt')
    size = cpr.save_compressed({'step': 5, 'model': sd, 'cfg': m.cfg}, path,
                               log=None)
    assert size > 0 and os.path.exists(path)
    ck2 = cpr.load_compressed(path, log=None)
    sd2 = ck2['model']
    # контракт: все исходные ключи на месте; дополнительно decompress_sd
    # ПЕРЕСОЗДАЁТ удаляемые буферы (V_dct/codes) — это и есть реестр-замок.
    assert set(sd) <= set(sd2), 'исходные ключи потерялись'
    extra = set(sd2) - set(sd)
    assert all(is_removable(k) for k in extra), f'неожиданные лишние ключи: {extra}'
    for k, v in sd.items():
        w = sd2[k]
        assert w.shape == v.shape, k
        if v.numel() == 0 or not v.dtype.is_floating_point:
            assert torch.equal(w, v), k
            continue
        rng = float(v.max() - v.min())
        err = float((w - v).abs().max())
        # 8-битная равномерная квантизация: err <= range/512 (с запасом x2.5)
        assert err <= rng / 200.0 + 1e-6, f'{k}: err={err:.3e} range={rng:.3e}'
    plain = str(tmp_path / 'plain.pt')
    torch.save({'step': 5, 'model': sd, 'cfg': m.cfg}, plain)
    assert os.path.getsize(path) < os.path.getsize(plain), 'компрессор не сжал'


def test_decompressed_model_forwards_and_generates(tmp_path):
    m = _mini().eval()
    x = torch.randint(1, 256, (1, 16))
    with torch.no_grad():
        h = m.embed_tokens(x)
        out, *_ = m(h, None, step=5, tokens=x)
        logits = m.lm_head(out)
    cpr = FCF_CPR()
    path = str(tmp_path / 'mini_fcf2.pt')
    cpr.save_compressed({'step': 5, 'model': m.state_dict(), 'cfg': m.cfg}, path,
                        log=None)
    ck2 = cpr.load_compressed(path, log=None)
    m2 = _mini().eval()
    _miss, unexp = m2.load_state_dict(ck2['model'], strict=False)
    # пересозданные removable-буферы для не-persistent моделей приходят
    # «лишними» (модель считает их сама) — они не ошибка; остальное — ошибка.
    unexp = [k for k in unexp if not is_removable(k)]
    assert not unexp, f'неожиданные ключи: {unexp[:5]}'
    with torch.no_grad():
        h2 = m2.embed_tokens(x)
        out2, *_ = m2(h2, None, step=5, tokens=x)
        logits2 = m2.lm_head(out2)
        nxt = int(logits2[0, -1].argmax())
    assert logits2.shape == logits.shape
    assert torch.isfinite(logits2).all()
    assert 0 <= nxt < 256
    # квантизация не меняет масштаб логитов (loose sanity, 8 бит = грубо)
    s1, s2 = float(logits.std()), float(logits2.std())
    assert abs(s2 - s1) < 0.5 * s1 + 1e-6, f'масштаб логитов уехал: {s1:.4f} -> {s2:.4f}'
