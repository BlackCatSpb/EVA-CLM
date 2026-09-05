"""Test τ-reform v2: monotonic ladder, intent_alpha, gradient flow."""
import torch
from core.config import WideBindConfig
from core.stack import WideBindStack


def test_forward_backward():
    cfg = WideBindConfig(
        D=2560, n_layers=24, vocab=65536, bind_K=32,
        bridge_dim=256, bridge_conn=0.1, memory_bank=True,
        seq_len=128, batch_size=1, use_amp=False,
        mem_l1_slots=3, mem_l2_slots=8, mem_l3_concepts=4,
        collective_S=4, collective_layer_idx=None,
        mlp_groups=32,
    )
    model = WideBindStack(cfg).float()

    print(f'Params: {model.param_count()/1e6:.2f}M')

    # ─── Tau system ───
    tau = model.tau_config
    tau_l = tau.tau_l
    diffs = tau_l[1:] - tau_l[:-1]
    assert (diffs > 0).all(), f'NOT MONOTONIC: {diffs.tolist()}'
    print(f'Monotonic: True')

    r = tau_l[-1].item() / tau_l[0].item()
    assert 30 < r < 200, f'Range {r:.1f}x outside [30, 200]'
    print(f'Range: {tau_l[0].item():.1f} -> {tau_l[-1].item():.1f} ({r:.1f}x)')

    alpha = tau.intent_alpha
    assert alpha[0].item() < 0.8, f'alpha[0]={alpha[0].item():.4f} too high (should be ~0.70)'
    assert alpha[-1].item() > 0.99, f'alpha[-1]={alpha[-1].item():.4f} too low (should be ~1.0)'
    print(f'intent_alpha: [{alpha[0].item():.4f}, ..., {alpha[-1].item():.4f}]')

    lr_spread = tau.lr_mult.max().item() / tau.lr_mult.min().item()
    assert 5 < lr_spread < 100, f'LR spread {lr_spread:.1f}x outside [5, 100]'
    print(f'lr_mult spread: {lr_spread:.1f}x')

    # ─── Forward + backward ───
    tokens = torch.randint(0, cfg.vocab, (1, 128))
    h = model.embed_tokens(tokens)
    out = model(h, step=100, adaptive=True)
    loss = out[0].sum()
    loss.backward()
    print(f'Forward+backward OK')

    # ─── Gradient checks ───
    g_tau = model.tau_config._tau_dev.grad
    assert g_tau is not None, '_tau_dev has no gradient'
    print(f'_tau_dev grad norm: {g_tau.norm().item():.6e}')

    # ─── Diagnostics ───
    diag = tau.get_diagnostics()
    assert 'tau_dev_utilization' in diag
    assert 'intent_alpha_min' in diag
    assert 'mem_tau_L1' in diag
    print(f'Diagnostics: {len(diag)} metrics')

    print('\n=== ALL TESTS PASSED ===')


if __name__ == '__main__':
    test_forward_backward()
