"""
smart_infer.py — CLI для эксперимента «умный инференс» для EVA-CLM.

Примеры:
  py scripts/smart_infer.py --prompt "Привет, как дела?" --tokens 60 --compare
  py scripts/smart_infer.py --prompt "Москва — это" --tokens 80 --compress
  py scripts/smart_infer.py --prompt "Привет" --tokens 40 --compress --checkpoint "checkponts/best 19.pt"
"""
import os, sys, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate import load_inference_checkpoint, load_russian_tokenizer, generate
from scripts.smart_controller import SmartController, smart_generate


def run_with_compression(model, prompt, ctrl, max_tokens, device, no_trunc=False):
    """Run generation with tau-adaptive compression + print stats."""
    from core.tau_compression import TauAdaptiveLogitsCache

    # Get tau_norm and maturation from model
    if hasattr(model, 'tau_config') and model.tau_config is not None:
        tc = model.tau_config
        tau_norm = tc.tau_norm.detach().cpu()
    else:
        tau_norm = torch.linspace(0, 1, len(model.layers))

    if hasattr(model, 'maturation') and model.maturation is not None:
        mat = model.maturation
        mat_gate = (mat.gate.detach().cpu() if hasattr(mat, 'gate')
                    else torch.ones(len(model.layers), device='cpu'))
    else:
        mat_gate = torch.ones(len(model.layers), device='cpu')

    # Create cache
    cache = TauAdaptiveLogitsCache(
        B=1, V=model.cfg.vocab, n_layers=len(model.layers),
        device=device, tau_norm=tau_norm, maturation=mat_gate
    )

    print(f'\n--- tau-adaptive compression ---')
    print(cache.schedule_str())
    print()

    # Run generation with compression
    from scripts.generate import load_russian_tokenizer
    ctrl.no_trunc = no_trunc
    model.eval()
    tok = load_russian_tokenizer()
    det = lambda ids: tok.decode(ids, skip_special_tokens=True)
    ids = tok.encode(prompt).ids
    tokens = torch.tensor(ids, dtype=torch.long, device=device)
    L = model.cfg.seq_len
    state = None
    rb = None
    head = model.lm_head
    tb = getattr(head, 'token_bias', None)

    out_ids = list(ids)
    n = len(model.layers)

    for step in range(max_tokens):
        ctx = tokens[-L:].unsqueeze(0)
        h = model.embed_tokens(ctx)
        out, state, _, rb = model(h, state, adaptive=False,
                                   step=step,
                                   reasoning_buffer=rb[0] if rb is not None else None,
                                   reasoning_count=rb[1] if rb is not None else None,
                                   tokens=ctx)
        model.observe_output(out)

        # Compress logits per-layer (simulated — we only have final logits)
        logits = head(out[:, -1:, :])[0, 0]

        # Compress the final logits through cache
        decompressed = cache.step(n - 1, logits)

        if not torch.isfinite(decompressed).all():
            decompressed = torch.nan_to_num(decompressed, nan=0.0, posinf=1e4, neginf=-1e4)
            state = None
            rb = None

        # Use decompressed logits for sampling (tests quality)
        trust_ts = []
        gate_ts = []
        for layer in model.layers:
            t, g = layer.mirror.meta_signals()
            trust_ts.append(t)
            gate_ts.append(g)
        trust_stack = torch.stack(trust_ts)
        gate_stack = torch.stack(gate_ts)
        wvec = torch.tensor(ctrl.tau_l_vec, dtype=trust_stack.dtype, device=trust_stack.device)
        wsum = wvec.sum()
        trust_val = ((trust_stack * wvec).sum() / wsum).item() if wsum > 0 else 0.5
        if not __import__('math').isfinite(trust_val):
            trust_val = 0.5
        mind = {'trust_max': trust_val, 'gate_ema_mean': gate_stack[-1].item()}

        temp, top_p, top_k, rep_pen, alpha = ctrl.decide(decompressed, mind, step)
        model.reasoning_scale_override = ctrl.model_reason_override
        if tb is not None:
            decompressed = (decompressed - tb) + alpha * tb
        nt = ctrl.sample(decompressed, temp, top_p, top_k, rep_pen)
        ctrl.recent.append(nt)
        max_recent = max(ctrl.rep_window + ctrl.rep_ngram, ctrl.alarm_window)
        if len(ctrl.recent) > max_recent:
            ctrl.recent = ctrl.recent[-max_recent:]
        out_ids.append(nt)
        tokens = torch.cat([tokens, torch.tensor([nt], dtype=torch.long, device=device)])

    # Print stats
    stats = cache.stats()
    print(f'\n--- compression stats ---')
    print(f'  Ratio: {stats["compression_ratio"]:.1f}x')
    print(f'  Per step: {stats["per_step_original_kb"]:.1f} KB -> {stats["per_step_compressed_kb"]:.1f} KB')
    print(f'  Strategies: {stats["strategies"]}')
    print(f'  tau_norm range: {stats["tau_norm_range"]}')

    return det(out_ids), ctrl.decisions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkponts/best 19.pt')
    ap.add_argument('--prompt', default='Привет, как дела?')
    ap.add_argument('--tokens', type=int, default=60)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--compare', action='store_true', help='also run baseline')
    ap.add_argument('--no-top', action='store_true', help='no top-p/top-k (pure temperature sampling)')
    ap.add_argument('--no-reasoning', action='store_true')
    ap.add_argument('--compress', action='store_true', help='enable tau-adaptive logits compression')
    args = ap.parse_args()

    device = ('cuda' if (args.device == 'auto' and torch.cuda.is_available())
              else (args.device if args.device != 'auto' else 'cpu'))

    state = load_inference_checkpoint(args.checkpoint, skip_compression=True, device='cpu')
    cfg = state['cfg']
    model = __import__('core').EVAStack(cfg).to(device)
    model.load_state_dict(state['model'], strict=False)
    model.reasoning_scale_override = 0.0
    vocab = cfg.vocab

    print(f'[model] step={state.get("step", "?")}, val_loss={state.get("best_val_loss", "?")}')
    print(f'[config] D={cfg.D}, n_layers={cfg.n_layers}, vocab={cfg.vocab}')

    if args.compare:
        model.reasoning_scale_override = 0.0
        base = generate(model, args.prompt, args.tokens, 0.9, 0, rep_penalty=2.0,
                        rep_window=5, reset_reasoning=False, bias_alpha=0.0)
        print(f'\nBASELINE: {base}')

    if args.compress:
        ctrl = SmartController(model, vocab, reasoning_on=not args.no_reasoning, no_trunc=args.no_top)
        text, dec = run_with_compression(model, args.prompt, ctrl, args.tokens, device, no_trunc=args.no_top)
        print(f'\nCOMPRESSED: {text}')
        modes = {}
        for d in dec:
            modes[d[1]] = modes.get(d[1], 0) + 1
        print(f'MODES: {modes}')
    else:
        ctrl = SmartController(model, vocab, reasoning_on=not args.no_reasoning, no_trunc=args.no_top)
        print(f'\n[tau] personality={ctrl.tau_personality:.1f} norm={ctrl.tau_norm:.2f} '
              f'temp=({ctrl.temp_lo:.2f},{ctrl.temp_hi:.2f}) trust_thr={ctrl.trust_thr:.2f}')
        text, dec = smart_generate(model, args.prompt, ctrl, args.tokens, no_trunc=args.no_top)
        print(f'\nSMART: {text}')
        modes = {}
        for d in dec:
            modes[d[1]] = modes.get(d[1], 0) + 1
        print(f'MODES: {modes}')


if __name__ == '__main__':
    main()
