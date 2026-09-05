"""
WideBind text generation.
Uses HuggingFace tokenizer from the training data directory.
"""

import os, sys, math, torch, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch.nn.functional as F
from tokenizers import Tokenizer

from core import WideBindConfig, WideBindStack
from core.compression import FCF_CPR


class AdaptiveSampler:
    """Self-governing sampling in the spirit of the project's adaptive controllers.

    No static knobs — every parameter follows the model's own state:
      - softmax entropy H of the *raw* head logits is the genuine uncertainty signal;
        H above its own EMA -> distribution is flat -> EXPLORE (warm, wide nucleus),
        H below EMA -> sharp -> EXPLOIT (cool, narrow nucleus). Mirrors the
        exploration_threshold semantics of AdaptiveController (stack.py:717).
      - the head already flattens by exp(log_temp) inside (embedding.py:197); the
        sampler normalizes generation temperature against it so base_temp means
        the same thing across checkpoints.
      - repetition escalator: repeated n-grams raise an alarm state E; penalty
        strength grows with E and temperature is nudged up to break the loop
        (the model's own restarts heal themselves the same way — no interventions).
    """

    def __init__(self, base_temp=0.8, top_k=40, base_top_p=0.90,
                 rep_penalty=2.0, rep_window=5, rep_ngram=3,
                 alarm_window=16, temp_range=(0.40, 1.60), p_range=(0.80, 0.99),
                 ema_alpha=0.92, esc_growth=0.5, esc_decay=0.25,
                 log_temp_ref=0.0, norm_log_temp=True, verbose=False):
        self.base_temp = base_temp
        self.top_k = top_k
        self.base_top_p = base_top_p
        self.rep_penalty = rep_penalty
        self.rep_window = rep_window
        self.rep_ngram = rep_ngram
        self.alarm_window = alarm_window
        self.temp_range = temp_range
        self.p_range = p_range
        self.ema_alpha = ema_alpha
        self.esc_growth = esc_growth
        self.esc_decay = esc_decay
        self.log_temp_ref = log_temp_ref
        self.norm_log_temp = norm_log_temp
        self.verbose = verbose
        self.mode = 'exploit'
        self._H_ema = None
        self._E = 0.0
        self._last = []
        self._log = []

    def _run_entropy(self, logits):
        probs = torch.softmax(logits.double(), dim=-1)
        return -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()

    def _stuck_state(self):
        hits = 0
        n = self.rep_ngram
        w = self._last[-self.alarm_window:]
        if len(w) >= n + 1:
            for i in range(len(w) - n):
                for j in range(i + 1, len(w) - n + 1):
                    if w[i:i + n] == w[j:j + n]:
                        hits += 2
        counts = {}
        for t in w:
            counts[t] = counts.get(t, 0) + 1
        top = max(counts.values()) if counts else 0
        if top >= 3:
            hits += 1
        return hits

    def sample(self, logits, temp_factor_ext=None):
        """Mature the raw head logits into the next token + decision record."""
        H = self._run_entropy(logits)
        if self._H_ema is None:
            self._H_ema = H
        self._H_ema = self.ema_alpha * self._H_ema + (1 - self.ema_alpha) * H
        d = (H - self._H_ema) / max(self._H_ema, 1e-9)

        # exploration / exploitation: flat -> explore, sharp -> exploit
        f_t = 1.0 + 0.5 * math.tanh(2.0 * d)
        p_eff = max(self.p_range[0], min(self.p_range[1],
                                         self.base_top_p + 0.1 * math.tanh(3.0 * d)))

        # repetition escalator
        hits = self._stuck_state()
        if hits >= 2:
            self._E = min(self._E + self.esc_growth, 3.0)
        else:
            self._E = max(self._E - self.esc_decay, 0.0)
        penalty = self.rep_penalty * (1.0 + 0.6 * self._E)
        if self._E >= 1.0:
            f_t = max(f_t, 1.0 + 0.2 * min(self._E, 2.0))

        # temperature normalization against the head's learned log_temp
        t = self.base_temp
        if temp_factor_ext is not None:
            t = t * temp_factor_ext
        if self.norm_log_temp:
            t = t / math.exp(max(self.log_temp_ref, -10.0))
        t_eff = max(self.temp_range[0], min(self.temp_range[1], t * f_t))

        z = logits / t_eff
        for rid in list(self._last)[-self.rep_window:]:
            z[rid] -= penalty
        if self.top_k > 0:
            vals, _ = torch.topk(z, self.top_k)
            z[z < vals[-1:]] = -float('inf')
        if p_eff < 1.0:
            probs = torch.softmax(z, dim=-1)
            sp, si = torch.sort(probs, dim=-1, descending=True)
            cum = torch.cumsum(sp, dim=-1)
            keep = cum <= p_eff
            keep[..., 0] = True
            mask = torch.zeros_like(z, dtype=torch.bool)
            mask.scatter_(-1, si, keep)
            z = z.masked_fill(~mask, -float('inf'))
        probs = torch.softmax(z, dim=-1)
        token = int(torch.multinomial(probs, 1).item())

        self.mode = 'stuck' if self._E >= 1.0 else ('explore' if f_t > 1.02 else 'exploit')
        self._last.append(token)
        rec = dict(step=len(self._last), H=H, H_ema=self._H_ema, d=d,
                   mode=self.mode, t_eff=t_eff, top_p=p_eff, penal=penalty, E=self._E)
        self._log.append(rec)
        if self.verbose and len(self._log) % 10 == 0:
            print(f'  [{rec["mode"]:7s}] step={len(self._last):3d} '
                  f'H={H:.2f} (ema {self._H_ema:.2f}) t={t_eff:.2f} p={p_eff:.2f} '
                  f'pen={penalty:.2f} E={self._E:.2f}')
        return token


def load_russian_tokenizer(path=None):
    """Load BPE tokenizer from russian_tokenizer/tokenizer.json.

    Resolution order (first match wins):
      1. explicit path
      2. <repo>/wb/russian_tokenizer/tokenizer.json (65k vocab, current)
      3. <repo>/wb/russian_tokenizer/tokenizer_v65536.json (65k, fallback)
      4. <repo>/fcp/russian_tokenizer/tokenizer.json
      5. ../fcp/russian_tokenizer/tokenizer.json
    """
    if path is not None:
        tok_file = os.path.join(path, 'russian_tokenizer', 'tokenizer.json')
    else:
        repo = os.path.join(os.path.dirname(__file__), '..')
        candidates = [
            os.path.join(repo, 'wb', 'russian_tokenizer', 'tokenizer.json'),
            os.path.join(repo, 'wb', 'russian_tokenizer', 'tokenizer_v65536.json'),
            os.path.join(repo, 'fcp', 'russian_tokenizer', 'tokenizer.json'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'fcp', 'russian_tokenizer', 'tokenizer.json'),
        ]
        tok_file = next((p for p in candidates if os.path.exists(p)), None)
    if tok_file and os.path.exists(tok_file):
        tok = Tokenizer.from_file(tok_file)
        tok.enable_padding(pad_id=0, pad_token='<|pad|>')
        tok.enable_truncation(max_length=512)
        return tok
    return None


@torch.no_grad()
def generate(model, prompt, max_new_tokens=128, temperature=1.0, top_k=50,
             show_mind=False, continuous_learn=False, context_mem=None,
             sampler=None, rep_penalty=2.0, rep_window=5, reset_reasoning=False,
             bias_alpha=0.0):
    """Generate tokens from prompt string."""
    model.eval()
    # Reset cross-call recurrent state so each prompt is independent.
    if hasattr(model, '_last_salience'):
        model._last_salience = None
    device = next(model.parameters()).device
    L = model.cfg.seq_len
    
    # Load tokenizer
    tok = load_russian_tokenizer()
    if tok is None:
        raise FileNotFoundError('russian_tokenizer/tokenizer.json not found')
    
    # Encode prompt
    encoded = tok.encode(prompt)
    prompt_tokens = encoded.ids
    detokenize = lambda ids: tok.decode(ids, skip_special_tokens=True)
    
    tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    
    # Generate
    state = None
    allow_write = continuous_learn or None
    rb = None
    
    mind_log = []
    
    recent = []
    head = model.lm_head
    tb = getattr(head, 'token_bias', None)
    for step in range(max_new_tokens):
        ctx = tokens[-L:].unsqueeze(0)
        
        if reset_reasoning:
            model.reset_reasoning()
            rb = None
        h = model.embed_tokens(ctx)
        out, state, _, rb = model(h, state, adaptive=False,
                                  context_mem=context_mem, allow_write=allow_write,
                                  step=step,
                                  reasoning_buffer=rb[0] if rb is not None else None,
                                  reasoning_count=rb[1] if rb is not None else None,
                                  tokens=ctx)
        model.observe_output(out)  # salience of THIS step -> next step's intent
        
        if show_mind and step % 10 == 0:
            info = model.layers[0].mirror.debug_mind()
            info['step'] = step
            mind_log.append(info)
            if step % 50 == 0:
                print(f'  step {step}: mem_norm={info.get("private_mem_norm",0):.4f} '
                      f'w_help={info.get("w_help",0):.4f} '
                      f'trust_diag={info.get("trust_diag_mean",0):.4f}')
        
        logits = head(out[:, -1:, :])[0, 0]
        if tb is not None and bias_alpha != 1.0:
            logits = (logits - tb) + bias_alpha * tb
        if not torch.isfinite(logits).all():
            # model diverged to NaN/Inf (hot output regime); reset recurrent
            # state and sample from flat logits instead of crashing.
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            state = None
        if sampler is not None:
            next_token = torch.tensor([sampler.sample(logits)], device=device)
        else:
            logits = logits / temperature
            # Repetition penalty: subtract fixed penalty (sign-safe)
            for rid in list(recent)[-rep_window:]:
                logits[rid] -= rep_penalty
            if top_k > 0:
                vals, _ = torch.topk(logits, top_k)
                logits[logits < vals[-1:]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
        
        recent.append(next_token.item())
        if len(recent) > rep_window:
            recent.pop(0)
        tokens = torch.cat([tokens, next_token], dim=0)
    
    if show_mind and mind_log:
        import json
        log_path = f'mind_log_{hash(prompt) & 0xFFFFFFFF:08x}.json'
        with open(log_path, 'w') as f:
            json.dump(mind_log, f, indent=2, default=float)
        print(f'  Mind log saved to {log_path}')
    
    return detokenize(tokens.tolist())


def load_inference_checkpoint(path, skip_compression=False, device='cpu'):
    """Load a checkpoint for inference, transparently handling FCF_CPR-compressed
    files. By default a non-compressed checkpoint is used directly (skip_compression
    is True unless --compress is passed); this avoids the CPU-heavy on-the-fly
    FCF_CPR compression that makes the GPU appear idle. Pass skip_compression=False
    (via --compress) to write/use a <name>_fcf.pt artifact instead."""
    from torch.serialization import add_safe_globals
    add_safe_globals([WideBindConfig])
    cpr = FCF_CPR()
    state = torch.load(path, map_location='cpu', weights_only=False)

    if 'model_compressed' in state:
        # Already compressed: decompress in-memory (no second disk read).
        cfg = state.get('cfg') or WideBindConfig()
        state['model'] = cpr.decompress_sd(state['model_compressed'], state['meta'], cfg)
        del state['model_compressed']
        del state['meta']
        return state

    # Non-compressed checkpoint.
    if skip_compression:
        return state

    # Compress on the fly into an artifact, then load the compressed version.
    fcf_path = (path[:-3] + '_fcf.pt') if path.endswith('.pt') else (path + '_fcf.pt')
    if not os.path.exists(fcf_path):
        print(f'FCF_CPR: compressing {path} -> {fcf_path}')
        cpr.save_compressed(state, fcf_path)
    st = torch.load(fcf_path, map_location='cpu', weights_only=False)
    cfg = st.get('cfg') or WideBindConfig()
    st['model'] = cpr.decompress_sd(st['model_compressed'], st['meta'], cfg)
    del st['model_compressed']
    del st['meta']
    return st


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='Path to .pt checkpoint')
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--skip-compression', action='store_true',
                        help='Use a non-compressed checkpoint directly, without running FCF_CPR (default behavior)')
    parser.add_argument('--compress', action='store_true',
                        help='Compress checkpoint to <name>_fcf.pt on load (slower first load, saves disk on reuse)')
    parser.add_argument('--smart', action='store_true',
                        help='Use tau-aware adaptive SmartController instead of static/adaptive sampling')
    parser.add_argument('--smart-no-reasoning', action='store_true',
                        help='Disable the per-token reasoning toggle inside smart mode')
    parser.add_argument('--no-top', action='store_true',
                        help='Disable top-p/top-k truncation (pure temperature sampling); applies to --smart')
    parser.add_argument('--show-mind', action='store_true', help='Log meta-cognitive mirror stats')
    parser.add_argument('--continuous-learn', action='store_true', help='Allow memory writes during generation')
    parser.add_argument('--context-mem', type=str, default='', help='Path to .pt file with context memory tensor (G, k)')
    parser.add_argument('--static', action='store_true', help='Static sampling (old behavior); default is adaptive')
    parser.add_argument('--top-p', type=float, default=0.90, help='Nucleus base (adaptive) or fixed width (--static)')
    parser.add_argument('--rep-penalty', type=float, default=2.0, help='Base repetition penalty')
    parser.add_argument('--rep-window', type=int, default=5, help='Repetition penalty window')
    parser.add_argument('--rep-ngram', type=int, default=3, help='Repetition n-gram for escalation')
    parser.add_argument('--alarm-window', type=int, default=16, help='Escalator observation window (wider than penalty window)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (0 = no seeding)')
    parser.add_argument('--reset-reasoning', action='store_true', help='Ablation: reset reasoning buffer before every forward step')
    parser.add_argument('--reasoning', choices=['natural', 'off', 'full'], default='off',
                        help='Reasoning scale: natural (learned ramp), off (0.0), full (1.0)')
    parser.add_argument('--bias-alpha', type=float, default=0.0,
                        help='Head token_bias weight: logits = (logits - tb) + alpha*tb (0.0 = no bias, best)')
    parser.add_argument('--no-log-temp-norm', action='store_true', help='Disable log_temp normalization in adaptive mode')
    parser.add_argument('--adaptive-verbose', action='store_true', help='Print adaptive sampler decisions every 10 steps')
    parser.add_argument('--threads', type=int, default=0,
                        help='CPU threads for torch (0 = torch default; 12 best on i5-12450H)')
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint: by default use the file directly (no on-the-fly FCF_CPR
    # compression, which is CPU-heavy and makes the GPU look idle). Pass --compress
    # to create the <name>_fcf.pt artifact, or point at an existing _fcf.pt.
    skip_compression = args.skip_compression or not args.compress
    state = load_inference_checkpoint(args.checkpoint, skip_compression=skip_compression, device=device)
    cfg = state.get('cfg', WideBindConfig())
    model = WideBindStack(cfg).to(device)
    model.load_state_dict(state['model'], strict=False)
    if model.explicit_reasoning:
        model.reasoning_enabled_step = int(state.get('reasoning_enabled_step', 0))
        print(f'Reasoning: enabled, ramp t={model.reasoning_enabled_step} '
              f'scale={model.reasoning_scale:.4f}')
    model.reasoning_scale_override = {'natural': None, 'off': 0.0, 'full': 1.0}[args.reasoning]
    
    print(f'Loaded checkpoint: step={state.get("step", "?")}  params={model.param_count():,}')
    print(f'[device] model on {next(model.parameters()).device} '
          f'(cuda_available={torch.cuda.is_available()}, requested={device})')

    # Smart mode: tau-aware adaptive controller (optional replacement for normal sampling)
    if args.smart:
        from scripts.smart_controller import smart_generate, SmartController
        vocab = cfg.vocab
        ctrl = SmartController(model, vocab, reasoning_on=not args.smart_no_reasoning,
                               no_trunc=args.no_top)
        print(f'[smart][tau] personality={ctrl.tau_personality:.1f} norm={ctrl.tau_norm:.2f} '
              f'temp=({ctrl.temp_lo:.2f},{ctrl.temp_hi:.2f}) trust_thr={ctrl.trust_thr:.2f}'
              f'{" no_trunc" if args.no_top else ""}')
        prompts = [args.prompt] if args.prompt else [
            'Привет, как дела?', 'Москва — столица',
            'В начале было Слово', 'Искусственный интеллект']
        for p in prompts:
            text, dec = smart_generate(model, p, ctrl, max_new_tokens=args.tokens,
                                       rep_window=ctrl.rep_window, no_trunc=args.no_top)
            print(f'> {p}')
            print(text)
            print()
            print('  decisions (step, mode, H, trust, temp, top_p, top_k, rep, reason, win, ngr, alpha, alarm):')
            for d in dec[::6]:
                print('   ', d)
        sys.exit(0)

    # Prompts
    context_mem = None
    if args.context_mem:
        cm = torch.load(args.context_mem, map_location=device, weights_only=True)
        context_mem = cm.to(device)

    if args.seed:
        torch.manual_seed(args.seed)

    sampler = None
    if not args.static:
        sampler = AdaptiveSampler(
            base_temp=args.temperature, top_k=args.top_k, base_top_p=args.top_p,
            rep_penalty=args.rep_penalty, rep_window=args.rep_window, rep_ngram=args.rep_ngram,
            alarm_window=args.alarm_window,
            norm_log_temp=not args.no_log_temp_norm, verbose=args.adaptive_verbose)
        log_temp_val = None
        for name, p in model.lm_head.named_parameters():
            if 'log_temp' in name:
                log_temp_val = p.data.mean().item()
        if log_temp_val is not None:
            sampler.log_temp_ref = log_temp_val
            print(f'Adaptive sampler: base_temp={args.temperature} top_p={args.top_p} '
                  f'rep_pen={args.rep_penalty} w={args.rep_window} ngram={args.rep_ngram} '
                  f'log_temp_norm={sampler.norm_log_temp} ref={log_temp_val:.4f}')
        else:
            print(f'Adaptive sampler: base_temp={args.temperature} top_p={args.top_p} '
                  f'rep_pen={args.rep_penalty} w={args.rep_window} ngram={args.rep_ngram} '
                  f'log_temp_norm={sampler.norm_log_temp} (no log_temp in head)')
    elif args.seed:
        print(f'Static sampling, seeded: {args.seed}')

    if args.prompt:
        text = generate(model, args.prompt, args.tokens, args.temperature, args.top_k,
                        show_mind=args.show_mind, continuous_learn=args.continuous_learn,
                        context_mem=context_mem, sampler=sampler,
                        rep_penalty=args.rep_penalty, rep_window=args.rep_window,
                        reset_reasoning=args.reset_reasoning, bias_alpha=args.bias_alpha)
        print(f'Prompt: {args.prompt}')
        print(f'Generated: {text}')
    else:
        prompts = [
            'Привет, как дела?',
            'Москва — столица',
            'В начале было Слово',
            'Искусственный интеллект',
        ]
        for p in prompts:
            text = generate(model, p, 100, 0.8, 0,
                            show_mind=args.show_mind, continuous_learn=args.continuous_learn,
                            context_mem=context_mem, sampler=sampler,
                            rep_penalty=args.rep_penalty, rep_window=args.rep_window,
                            reset_reasoning=args.reset_reasoning, bias_alpha=args.bias_alpha)
            print(f'> {p}')
            print(text)
            print()
