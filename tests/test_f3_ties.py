"""F3 (math audit) + P0-2 A/B-ручка `tie_grad`: ties IN-GRAPH vs чтение зеркала.

История: до F3 `BottleneckBind._tie_hook` копировал W_proj.weight.data в W_out
под no_grad на каждом forward — градиент выходного пути (измерено ‖dL/dW_out‖≈51.5)
писался в W_out и стирался следующей копией; в зеркале (tie_mirror_proj=True,
production) W_out — буфер, синхронизированный из W_proj, — «K-space автоэнкодер»
был верен по значению и мёртв по градиенту.

F3 сделал tie в графе, но опыт резюма 7040 показал: на 11k-шаговом чекпойнте
это ВЗРЫВООПАСНАЯ смена тропы (h_norm ×15, u-насыщение 0.66 за 55 шагов —
g_wo ×18, branch_r_mirror ×10). Поэтому in-graph tie теперь за флагом
`cfg.tie_grad` (default False = прежнее чтение зеркала/буфера: forward побитово
тот же, градиент выходного пути мёртв). Эти тесты лок'ят оба режима.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.bind import BottleneckBind            # noqa: E402
from core.config import EVAConfig               # noqa: E402
from core.mirror import GroupedCognitiveMirror  # noqa: E402


def _bind(tie: bool, tie_grad: bool = True):
    cfg = EVAConfig(D=64, vocab=64)
    cfg.bind_twist_mode = 'off'      # the tied path of the legacy diagonal mode
    cfg.tie_bind = tie
    cfg.tie_grad = tie_grad
    torch.manual_seed(0)
    return BottleneckBind(D=64, K=32, cfg=cfg)


def _sync_untied_bind(m):
    with torch.no_grad():
        m.W_out.copy_(m.W_proj.weight)


def test_f3_bind_tie_matches_the_untied_gradient():
    tied, untied = _bind(True), _bind(False)
    untied.load_state_dict(tied.state_dict(), strict=False)   # same values
    _sync_untied_bind(untied)                                 # W_out = W_proj^T
    h = torch.randn(1, 5, 64, requires_grad=True)
    out_t = tied(h)
    out_t.sum().backward()
    out_u = untied(h)
    out_u.sum().backward()
    assert torch.allclose(out_t.detach(), out_u.detach(), rtol=1e-5, atol=1e-6)
    g_expected = untied.W_proj.weight.grad + untied.W_out.grad
    assert torch.allclose(tied.W_proj.weight.grad, g_expected, rtol=1e-4, atol=1e-6), \
        'the tied W_proj gradient must include the output path'


def test_f3_bind_forward_uses_w_proj_not_w_out():
    m = _bind(True)
    m._hook.remove()                 # stop the value mirror; W_out is now unused
    h = torch.randn(1, 5, 64)
    with torch.no_grad():
        m.W_out.fill_(123.0)
    out = m(h)
    with torch.no_grad():
        m.W_out.zero_()
    assert torch.allclose(out, m(h), rtol=1e-6, atol=1e-6), \
        'the tied forward must not read W_out'


def test_tie_grad_off_reads_the_mirror_and_keeps_values_identical():
    """Default (tie_grad=False): forward читает W_out (значения те же — хук
    синхронизирует с W_proj^T), но выходной градиент НЕ течёт в W_proj."""
    on = _bind(True, tie_grad=True)
    off = _bind(True, tie_grad=False)
    off.load_state_dict(on.state_dict(), strict=False)
    h = torch.randn(1, 5, 64)
    with torch.no_grad():
        out_on = on(h)
        out_off = off(h)
    assert torch.allclose(out_on, out_off, rtol=1e-6, atol=1e-7), \
        'значения окуляра обязаны совпадать (буфер = W_proj^T по хуку)'
    h1 = torch.randn(1, 5, 64, requires_grad=True)
    off(h1).sum().backward()
    # в off-режиме выходной путь пишет в W_out (Parameter), не в W_proj
    assert off.W_out.grad is not None and float(off.W_out.grad.abs().sum()) > 0.0
    # входной путь (hp = h @ W_proj) даёт W_proj свой градиент — он ненулевой,
    # но БЕЗ вклада выходного пути: у on-режима он строго больше по норме
    assert float(off.W_proj.weight.grad.norm()) > 0.0
    h2 = torch.randn(1, 5, 64, requires_grad=True)
    on(h2).sum().backward()
    assert float(on.W_proj.weight.grad.norm()) > float(off.W_proj.weight.grad.norm()), \
        'in-graph tie обязан добавлять в W_proj градиент выходного пути'


def _mirror(tie: bool, tie_grad: bool = True):
    torch.manual_seed(0)
    return GroupedCognitiveMirror(D=64, G=2, k=16, n_layers=2,
                                  tie_mirror_proj=tie, tie_grad=tie_grad,
                                  seq_len=8)


def test_f3_mirror_tie_matches_the_untied_gradient():
    tied, untied = _mirror(True), _mirror(False)
    untied.load_state_dict(tied.state_dict(), strict=False)   # W_out = W_proj^T
    h = torch.randn(1, 4, 64, requires_grad=True)
    mem = torch.randn(1, 4, 64)
    torch.manual_seed(1)
    out_t = tied(h, mem)[0]
    out_t.sum().backward()
    torch.manual_seed(1)
    out_u = untied(h, mem)[0]
    out_u.sum().backward()
    assert torch.allclose(out_t.detach(), out_u.detach(), rtol=1e-5, atol=1e-6)
    # W_out is (G,k,d) while W_proj is (G,d,k): the tie is W_out = W_proj^T,
    # so the untied output-path gradient must be permuted back.
    g_expected = untied.W_proj.grad + untied.W_out.grad.permute(0, 2, 1)
    assert torch.allclose(tied.W_proj.grad, g_expected, rtol=1e-4, atol=1e-6), \
        'the tied mirror W_proj gradient must include the reconstruction path'


def test_mirror_tie_grad_off_is_value_identical_and_gradient_dead():
    on = _mirror(True, tie_grad=True)
    off = _mirror(True, tie_grad=False)
    off.load_state_dict(on.state_dict(), strict=False)
    h = torch.randn(1, 4, 64)
    mem = torch.randn(1, 4, 64)
    torch.manual_seed(2)
    out_on = on(h, mem)[0]
    torch.manual_seed(2)
    out_off = off(h, mem)[0]
    assert torch.allclose(out_on, out_off, rtol=1e-5, atol=1e-6), \
        'значения зеркала обязаны совпадать (буфер синхронизирован)'
    h1 = torch.randn(1, 4, 64, requires_grad=True)
    torch.manual_seed(2)
    off(h1, mem)[0].sum().backward()
    h2 = torch.randn(1, 4, 64, requires_grad=True)
    torch.manual_seed(2)
    on(h2, mem)[0].sum().backward()
    assert float(on.W_proj.grad.norm()) > float(off.W_proj.grad.norm()), \
        'in-graph tie обязан добавлять в W_proj градиент реконструкции'
