"""
Test inference on compressed WideBind checkpoint.
Measures: VRAM, tok/s, generation quality.
"""
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from core.compression import FCF_CPR
from core import WideBindConfig, WideBindStack


def test_inference(ckpt_path, prompt_len=16, gen_len=128, device='cuda'):
    print(f'Loading: {ckpt_path} ({os.path.getsize(ckpt_path)/1e6:.0f} MB)')
    
    # Load compressed
    cpr = FCF_CPR()
    if ckpt_path.endswith('_fcf.pt') or ckpt_path.endswith('_infer.pt'):
        ckpt = cpr.load_compressed(ckpt_path)
    else:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']
    
    # Model (fp16: half VRAM, ~2x speed on MX550)
    model = WideBindStack(cfg).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    if device == 'cuda' and next(model.parameters()).dtype == torch.float32:
        model.half()  # convert params to fp16
    print(f'  Params: {model.param_count():,}')
    
    def mem_usage():
        if device == 'cuda':
            return torch.cuda.max_memory_allocated() / 1e9
        return 0
    
    torch.cuda.reset_peak_memory_stats()
    mem_before = mem_usage()
    
    # Prefill
    prompt = torch.randint(0, min(cfg.vocab, 100), (1, prompt_len), device=device)
    h = model.embed_tokens(prompt)
    state = None
    t0 = time.time()
    with torch.no_grad():
        out, state, _, _ = model(h, state)
    t_prefill = time.time() - t0
    logits = model.compute_loss(out, prompt)  # just to compute
    
    mem_prefill = mem_usage()
    
    # Generate token-by-token
    x = out[:, -1:, :]  # (1, 1, D) — last hidden state
    tokens = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(gen_len):
            out, state, _, _ = model(x, state)
            logits = model.lm_head(out)  # (1, 1, vocab)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens.append(next_token.item())
            x = model.embed_tokens(next_token)
    
    t_gen = time.time() - t0
    mem_gen = mem_usage()
    
    tok_s = gen_len / max(t_gen, 1e-10)
    print(f'  VRAM: prefill={mem_prefill-mem_before:.2f} GB gen={mem_gen-mem_before:.2f} GB peak={mem_gen:.2f} GB')
    print(f'  Prefill: {t_prefill*1000:.0f}ms ({prompt_len} tok)')
    print(f'  Generation: {t_gen*1000:.0f}ms total, {t_gen/gen_len*1000:.0f}ms/tok, {tok_s:.0f} tok/s')
    print(f'  Generated tokens: {tokens[:20]}...')
    
    return model, tokens


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"})')
    print()
    
    test_inference(
        r'C:\Users\black\OneDrive\Desktop\WideBind\checkpoints\step_15000_infer.pt',
        prompt_len=16, gen_len=128, device=device
    )
