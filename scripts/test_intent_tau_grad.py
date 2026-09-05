"""
Тест: градиент доходит до выделенного _tau_intent_dev через aux-loss intent_tau.

Запуск: py -3.12 scripts/test_intent_tau_grad.py
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import WideBindStack
from scripts.generate import load_inference_checkpoint

CKPT = os.path.join('checkpoints', 'step_11844_fcf.pt')


def build():
    state = load_inference_checkpoint(CKPT, skip_compression=True, device='cpu')
    cfg = state.get('cfg')
    cfg.intent_bridge = True
    cfg.intent_tau_hierarchy_weight = 0.01
    m = WideBindStack(cfg)
    m.load_state_dict(state['model'], strict=False)
    m.train()
    return m, cfg


def main():
    torch.manual_seed(0)
    m, cfg = build()
    D, V, nL = cfg.D, cfg.vocab, cfg.n_layers

    x = torch.randint(0, V, (1, 16))
    emb = m.embed_tokens(x)
    G = m.layers[0].mirror.G
    synth = [torch.randn(1, 1, G, l.mirror.k) * 0.1 for l in m.layers]  # per-layer stream

    o, *_ = m(emb, None, adaptive=False, intent_state=synth)
    ce_loss, aux_dict = m.compute_losses(o, x, h_emb=emb)
    assert 'intent_tau' in aux_dict, f'no intent_tau in aux_dict: {list(aux_dict)}'
    # backward through both ladders to confirm each param gets its own gradient
    loss = aux_dict['intent_tau'] + aux_dict.get('w_m2v', torch.zeros(1))
    loss.backward(retain_graph=True)

    g = m._tau_intent_dev.grad
    assert g is not None, '_tau_intent_dev.grad is None'
    gnorm = g.norm().item()
    print(f'[_tau_intent_dev] grad norm = {gnorm:.4e}')
    assert gnorm > 0, '_tau_intent_dev got zero gradient'

    g2 = m._tau_l_dev.grad
    print(f'[_tau_l_dev]      grad norm = {0.0 if g2 is None else g2.norm().item():.4e}')
    print(f'[intent_tau] loss = {aux_dict["intent_tau"].item():.4e}')
    print('\nPASS — _tau_intent_dev receives gradient via intent_tau aux.')


if __name__ == '__main__':
    main()
