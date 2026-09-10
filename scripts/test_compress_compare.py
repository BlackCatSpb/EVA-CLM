"""Compare generation with and without tau-adaptive compression."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from core.config import WideBindConfig as EVAConfig
from core.stack import EVAStack
from core.tau_compression import TauAdaptiveLogitsCache
from scripts.generate import load_inference_checkpoint, load_russian_tokenizer

# Load model
state = load_inference_checkpoint('checkponts/best 19.pt', skip_compression=False, device='cpu')
cfg = state['cfg']
model = EVAStack(cfg).to('cpu')
model.load_state_dict(state['model'], strict=False)
model.eval()
tok = load_russian_tokenizer()

prompt = 'Moscow is'
ids = tok.encode(prompt).ids
tokens = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

# WITHOUT compression
print('=== WITHOUT compression ===')
with torch.no_grad():
    h = model.embed_tokens(tokens)
    out, _, _, _ = model(h, None, adaptive=False, step=0, tokens=tokens)
    logits = model.lm_head(out[:, -1:, :])[0, 0]
    probs = torch.softmax(logits / 0.9, -1)
    token = torch.multinomial(probs, 1).item()
    print(f'Next token: {tok.decode([token])}')
    top5 = torch.topk(probs, 5)
    for v, i in zip(top5.values, top5.indices):
        print(f'  {tok.decode([i.item()]):>20s}: {v.item():.4f}')

# WITH compression
print()
print('=== WITH tau-adaptive compression ===')
tau_norm = torch.linspace(0, 1, len(model.layers))
mat_gate = torch.ones(len(model.layers))
cache = TauAdaptiveLogitsCache(
    B=1, V=cfg.vocab, n_layers=len(model.layers),
    device='cpu', tau_norm=tau_norm, maturation=mat_gate
)

with torch.no_grad():
    h = model.embed_tokens(tokens)
    out, _, _, _ = model(h, None, adaptive=False, step=0, tokens=tokens)
    logits = model.lm_head(out[:, -1:, :])[0, 0]
    decompressed = cache.step(len(model.layers) - 1, logits)
    probs = torch.softmax(decompressed / 0.9, -1)
    token_c = torch.multinomial(probs, 1).item()
    print(f'Next token: {tok.decode([token_c])}')
    top5 = torch.topk(probs, 5)
    for v, i in zip(top5.values, top5.indices):
        print(f'  {tok.decode([i.item()]):>20s}: {v.item():.4f}')

# Compare
print()
acc = (logits.argmax(-1) == decompressed.argmax(-1)).float().item()
print(f'Top-1 accuracy: {acc:.4f}')
print(f'Same token sampled: {token == token_c}')
stats = cache.stats()
print(f'Compression ratio: {stats["compression_ratio"]:.1f}x')
print(f'Per step: {stats["per_step_original_kb"]:.1f} KB -> {stats["per_step_compressed_kb"]:.2f} KB')
