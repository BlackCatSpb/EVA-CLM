# -*- coding: utf-8 -*-
"""T9.11 лок: генерация использует технологии модели (паритет с тренировкой).

- state/gs/intent_state ведутся между шагами (кросс-слойный self-model EMA и
  intent-поток не рвутся, как в тренировке);
- step = base_step + step (гейты temper/lacuna/SRL/phantom читают реальный шаг,
  а не 0-based шаг генерации);
- промпт завершается SEP (id=2) — sent_* эмбеддинги, sentence-ring и записи
  банка включаются.

Run: python -m pytest tests/test_t9_generate_parity.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from core.config import EVAConfig
from core.stack import EVAStack


def _model():
    torch.manual_seed(0)
    cfg = EVAConfig(n_layers=2, D=256, mlp_groups=4, code_dim=16, code_sparsity=4,
                    vocab=400, save_dir='.', logit_cache_enabled=True,
                    memory_bank=True, intent_bridge=True, vsa_decay_floor_k=2.0,
                    gradient_checkpointing=False)
    return cfg, EVAStack(cfg).eval()


def test_generation_loop_threads_gs_and_intent():
    """Реплика петли генерации: gs/intent_state/state обязаны вестись между
    шагами (иначе контекст зеркала и intent-поток сбрасываются каждый токен)."""
    cfg, m = _model()
    torch.manual_seed(7)
    ctx = torch.randint(3, cfg.vocab, (1, 32)); ctx[:, ::11] = 2
    state = gs = intent_state = None
    gs_norms, intent_lens = [], []
    with torch.no_grad():
        for step in range(5):
            h = m.embed_tokens(ctx)
            out, state, gs, rb = m(h, state, global_state=gs, adaptive=False,
                                   step=1000 + step, intent_state=intent_state,
                                   tokens=ctx)
            intent_state = getattr(m, '_last_intent_state', None)
            assert gs is not None, 'gs не вернулся'
            gs_norms.append(float(gs.norm()))
            if intent_state is not None:
                intent_lens.append(len(intent_state) if hasattr(intent_state, '__len__') else -1)
    assert gs_norms[1] != gs_norms[0], 'gs не обновляется между шагами'
    assert all(n > 0 for n in gs_norms)


def test_step_gates_match_training_regime():
    """step=1000+ (а не 0-based) — temper/SRL/phantom-гейты включаются, как в
    тренировке: с step<after они бы молчали (режимное расхождение)."""
    cfg, m = _model()
    head = m.lm_head
    assert int(getattr(head, 'temper_after', 0)) > 0
    torch.manual_seed(7)
    ctx = torch.randint(3, cfg.vocab, (1, 32)); ctx[:, ::11] = 2
    with torch.no_grad():
        h = m.embed_tokens(ctx)
        m(h, None, global_state=None, adaptive=False, step=2000, tokens=ctx)
        assert head._temper_active is True, 'temper выключен на step=2000'
        h2 = m.embed_tokens(ctx)
        m(h2, None, global_state=None, adaptive=False, step=0, tokens=ctx)
        assert head._temper_active is False, 'temper включён на step=0 (0-based)'


def test_source_locks():
    """Статические локи (R1-класс: правка wiring не должна откатиться)."""
    base = os.path.join(os.path.dirname(__file__), '..', 'scripts')
    for fname in ('generate.py', 'smart_controller.py'):
        src = open(os.path.join(base, fname), encoding='utf-8').read()
        assert 'global_state=gs' in src, f'{fname}: gs не ведётся'
        assert 'intent_state=intent_state' in src, f'{fname}: intent не ведётся'
        assert 'base_step + step' in src, f'{fname}: step не сдвинут к шагу чекпойнта'
    g = open(os.path.join(base, 'generate.py'), encoding='utf-8').read()
    assert 'prompt_tokens + [2]' in g, 'generate.py: нет SEP на конце промпта'
    sc = open(os.path.join(base, 'smart_controller.py'), encoding='utf-8').read()
    assert 'ids + [2]' in sc, 'smart_controller.py: нет SEP на конце промпта'
