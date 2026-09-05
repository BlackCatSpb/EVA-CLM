"""
smart_infer.py — CLI для эксперимента «умный инференс».
Обёртка над scripts/smart_controller.py: прогоняет baseline (статическая генерация
из generate.py) и tau-умный SmartController, печатает оба текста и сводку решений.

Пример: py -3.12 scripts/smart_infer.py --prompt "Привет, как дела?" --tokens 60 --compare
"""
import os, sys, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate import (load_inference_checkpoint, load_russian_tokenizer, generate)
from scripts.smart_controller import SmartController, smart_generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/step_11844_fcf.pt')
    ap.add_argument('--prompt', default='Привет, как дела?')
    ap.add_argument('--tokens', type=int, default=60)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--compare', action='store_true', help='также прогнать baseline')
    ap.add_argument('--no-top', action='store_true', help='без top-p/top-k (чистый temperature-семплинг)')
    ap.add_argument('--no-reasoning', action='store_true')
    args = ap.parse_args()

    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else (args.device if args.device != 'auto' else 'cpu')
    state = load_inference_checkpoint(args.checkpoint, skip_compression=False, device='cpu')
    cfg = state['cfg']
    model = __import__('core').WideBindStack(cfg).to(device)
    model.load_state_dict(state['model'], strict=False)
    model.reasoning_scale_override = 0.0
    vocab = cfg.vocab

    if args.compare:
        model.reasoning_scale_override = 0.0
        base = generate(model, args.prompt, args.tokens, 0.9, 0, rep_penalty=2.0,
                        rep_window=5, reset_reasoning=False, bias_alpha=0.0)
        print('BASELINE:', base)
        print()

    ctrl = SmartController(model, vocab, reasoning_on=not args.no_reasoning, no_trunc=args.no_top)
    print(f'[tau] personality={ctrl.tau_personality:.1f} norm={ctrl.tau_norm:.2f} '
          f'temp=({ctrl.temp_lo:.2f},{ctrl.temp_hi:.2f}) trust_thr={ctrl.trust_thr:.2f}'
          f'{" no_trunc" if args.no_top else ""}')
    text, dec = smart_generate(model, args.prompt, ctrl, args.tokens, no_trunc=args.no_top)
    print('SMART:', text)
    print()
    print('DECISIONS (step, mode, H, trust, temp, top_p, top_k, rep, reason, win, ngr, alpha, alarm):')
    for d in dec[::6]:
        print('  ', d)
    modes = {}
    for d in dec:
        modes[d[1]] = modes.get(d[1], 0) + 1
    print('MODE COUNTS:', modes)


if __name__ == '__main__':
    main()
