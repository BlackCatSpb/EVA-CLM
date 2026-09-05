"""
Тест Intent Bridge (нисходяще-восходящая передача «намерения» экспертам).

Проверяет:
 1. Checkpoint-совместимость: модель с intent_bridge=True и zero-init
    параметрами даёт БАЙТ-ИДЕНТИЧНЫЙ выход базовой модели
    (=> можно доучивать, НЕ с нуля).
 2. Механизм работает: симулированный intent_state + ненулевые w_intent
    меняет выход экспертов и по w_intent течёт градиент.

Запуск: py -3.12 scripts/test_intent_bridge.py
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import WideBindStack
from scripts.generate import load_inference_checkpoint

CKPT = os.path.join('checkpoints', 'step_11844_fcf.pt')


def build(bridge: bool):
    state = load_inference_checkpoint(CKPT, skip_compression=True, device='cpu')
    cfg = state.get('cfg')
    cfg.intent_bridge = bridge
    m = WideBindStack(cfg)
    miss, unexp = m.load_state_dict(state['model'], strict=False)
    m.eval()
    return m, cfg


def main():
    torch.manual_seed(0)
    base, cfg = build(False)
    D, V, nL = cfg.D, cfg.vocab, cfg.n_layers

    x = torch.randint(0, V, (1, 16))
    emb = base.embed_tokens(x)

    with torch.no_grad():
        o0, *_ = base(emb, None, adaptive=False)

    # (1) intent_bridge в zero-init => идентичный выход
    br, _ = build(True)
    with torch.no_grad():
        o1, *_ = br(emb, None, adaptive=False, intent_state=None)
    assert o0.shape == o1.shape
    maxdiff = (o0 - o1).abs().max().item()
    assert torch.allclose(o0, o1, atol=1e-6), f'bridge not no-op at init, maxdiff={maxdiff:.2e}'
    print(f'[1] CHECKPOINT-SAFE: zero-init bridge == baseline (maxdiff={maxdiff:.2e})')

    # число новых параметров (w_intent/b_intent + probe)
    n_new = sum(p.numel() for n, p in br.named_parameters()
                if 'w_intent' in n or 'b_intent' in n or 'intent_probe' in n)
    print(f'    new params (bridge on): {n_new:,}')

    # (2) симуляция сигнала: ненулевые w_intent + synthetic intent_state
    with torch.no_grad():
        for lay in br.layers:
            lay.mirror.w_intent.copy_(0.3 * torch.randn_like(lay.mirror.w_intent))
            lay.mirror.b_intent.copy_(0.1 * torch.randn_like(lay.mirror.b_intent))
    G = br.layers[0].mirror.G
    synth = [torch.randn(1, 1, G, l.mirror.k) * 0.1 for l in br.layers]  # per-layer stream
    with torch.no_grad():
        o2, *_ = br(emb, None, adaptive=False, intent_state=synth)
    assert not torch.isnan(o2).any(), 'NaN in intent-modulated output'
    diff_signal = (o0 - o2).abs().max().item()
    assert diff_signal > 1e-5, 'synthetic intent did not change output'
    print(f'[2] SIGNAL WORKS: synthetic intent_state changes output (maxdiff={diff_signal:.2e})')

    # (3) градиент течёт в w_intent (эксперты «ловят» intent)
    br.train()
    o3, *_ = br(emb, None, adaptive=False, intent_state=synth)
    o3.sum().backward()
    grads = [lay.mirror.w_intent.grad for lay in br.layers
             if hasattr(lay.mirror, 'w_intent')]
    any_grad = any(g is not None and g.abs().sum().item() > 0 for g in grads)
    assert any_grad, 'no gradient reached w_intent'
    print(f'[3] GRADIENT: w_intent receives gradient (experts catch intent)')

    print('\nALL PASS — Intent Bridge реализован и checkpoint-совместим.')


if __name__ == '__main__':
    main()
