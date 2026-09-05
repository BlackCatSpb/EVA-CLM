"""
Эксперимент: гибридная голова sigmoid + softmax (SpectrumGate) в LM head.

Tau берётся из модели — из её же обучаемого log_temp. Никаких магических чисел.

Формула:
  sig = sigmoid(zt)           # базовый independent gate
  tau = exp(log_temp)         # из модели, K параметров
  rel = softmax(zt / tau)     # relative emphasis
  gate = sig * (1 + rel)      # SpectrumGate

Запуск:
  python scripts/hybrid_head_experiment.py "checkpoints\best 5.pt" --prompt "Привет, как дела?" --tokens 30
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from core import WideBindConfig, WideBindStack
from core.embedding import SigmoidCodedHead
from scripts.generate import load_inference_checkpoint, load_russian_tokenizer, generate


class HybridSigmoidSoftmaxHead(nn.Module):
    """SigmoidCodedHead + SpectrumGate: sigmoid * (1 + softmax).

    Tau = exp(log_temp) из модели. Новых параметров нет.
    """

    def __init__(self, base_head: SigmoidCodedHead):
        super().__init__()
        self.base = base_head

    def _gates(self, h, temp_factor=None, bus_bias=None):
        return self.base._gates(h, temp_factor=temp_factor, bus_bias=bus_bias)

    def _su(self, zt):
        sig = torch.sigmoid(zt)
        tau = torch.exp(self.base.log_temp).clamp(0.1, 10.0)
        rel = torch.softmax(zt / tau, dim=-1)
        gate = sig * (1.0 + rel)
        gate = gate.clamp(1e-7, 1 - 1e-7)
        ls = torch.log(gate)
        lms = torch.log(1 - gate)
        u = ls - lms
        base = lms.sum(-1)
        return u, base

    def forward(self, h, bus_bias=None):
        if h.dim() == 2:
            h = h.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False
        u, base = self._su(self._gates(h, bus_bias=bus_bias))
        logits = u @ self.base.codes.T + base[..., None] + self.base.token_bias
        if self.base.normalize:
            logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        if squeeze:
            logits = logits.squeeze(1)
        return logits

    @property
    def token_bias(self):
        return self.base.token_bias

    @property
    def codes(self):
        return self.base.codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', type=str)
    ap.add_argument('--prompt', default='Привет, как дела?')
    ap.add_argument('--tokens', type=int, default=30)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    device = args.device
    state = load_inference_checkpoint(args.checkpoint, skip_compression=True, device='cpu')
    cfg = state.get('cfg', WideBindConfig())
    tok = load_russian_tokenizer()

    print(f'Step: {state.get("step", "?")}')
    print(f'Prompt: {args.prompt}')
    print()

    # --- Original ---
    model_orig = WideBindStack(cfg).to(device)
    model_orig.load_state_dict(state['model'], strict=False)
    model_orig.reasoning_scale_override = 0.0

    result_orig = generate(model_orig, args.prompt, args.tokens, 0.9, 0,
                           rep_penalty=2.0, rep_window=5, reset_reasoning=False,
                           bias_alpha=0.0)
    print(f'[ORIGINAL] sigmoid:')
    print(f'  {result_orig}')
    print()

    # --- Hybrid ---
    model_hybrid = WideBindStack(cfg).to(device)
    model_hybrid.load_state_dict(state['model'], strict=False)
    model_hybrid.reasoning_scale_override = 0.0
    model_hybrid.lm_head = HybridSigmoidSoftmaxHead(model_hybrid.lm_head)

    result_hybrid = generate(model_hybrid, args.prompt, args.tokens, 0.9, 0,
                             rep_penalty=2.0, rep_window=5, reset_reasoning=False,
                             bias_alpha=0.0)
    print(f'[HYBRID] sigmoid * (1 + softmax), tau=log_temp from model:')
    print(f'  {result_hybrid}')
    print()

    # --- Compare logits ---
    ctx = tok.encode(args.prompt)
    tokens = torch.tensor(ctx.ids, dtype=torch.long, device=device).unsqueeze(0)
    h = model_orig.embed_tokens(tokens)
    with torch.no_grad():
        out_orig, _, _, _ = model_orig(h, None, adaptive=False, step=0)
        logits_orig = model_orig.lm_head(out_orig[:, -1:, :])[0, 0]

        out_hyb, _, _, _ = model_hybrid(h, None, adaptive=False, step=0)
        logits_hyb = model_hybrid.lm_head(out_hyb[:, -1:, :])[0, 0]

    topk_orig = torch.topk(logits_orig, 10)
    topk_hyb = torch.topk(logits_hyb, 10)

    print('=== Top-10 next token ===')
    print(f'{"Token":<30} {"Orig":>10} {"Hybrid":>10}')
    print('-' * 52)
    seen = set()
    for v1, v2 in zip(topk_orig.indices, topk_hyb.indices):
        t1 = tok.decode([v1.item()], skip_special_tokens=True)
        t2 = tok.decode([v2.item()], skip_special_tokens=True)
        l1 = logits_orig[v1].item()
        l2 = logits_hyb[v2].item()
        if t1 not in seen:
            print(f'{t1:<30} {l1:>10.3f}')
            seen.add(t1)
        if t2 not in seen:
            print(f'{t2:<30} {"":>10} {l2:>10.3f}')
            seen.add(t2)

    H_orig = -(torch.softmax(logits_orig, -1) * F.log_softmax(logits_orig, -1)).sum()
    H_hyb = -(torch.softmax(logits_hyb, -1) * F.log_softmax(logits_hyb, -1)).sum()
    print(f'\nEntropy: orig={H_orig:.3f}  hybrid={H_hyb:.3f}')


if __name__ == '__main__':
    main()
