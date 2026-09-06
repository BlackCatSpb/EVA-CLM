"""
Benchmark: LogitCache generation quality and speed.
Compares generation with/without cache.
"""
import sys, os, time, torch, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.serialization import add_safe_globals
from core import EVAConfig, EVAStack
from core.config import WideBindConfig
from core.logit_cache import LogitCacheAttention
from scripts.generate import load_russian_tokenizer
add_safe_globals([EVAConfig, WideBindConfig])


def load_model():
    ckpt = torch.load(r"C:\Users\black\OneDrive\Desktop\EVA CLM\checkponts\best 19.pt",
                      map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']
    model = EVAStack(cfg)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval().float()
    return model, cfg


def bench_generation(model, prompt_ids, n_tokens, use_cache):
    """Benchmark generation with/without cache."""
    gc.collect()
    L = 128
    tok = load_russian_tokenizer()
    tokens = list(prompt_ids[0])
    state = None

    # Warmup
    with torch.no_grad():
        ctx = torch.tensor([tokens[-L:]], dtype=torch.long)
        h = model.embed_tokens(ctx)
        out, state, _, _ = model(h, state, adaptive=False, step=0, tokens=ctx)
        model.reset_cache()

    # Benchmark
    t_start = time.time()
    for i in range(n_tokens):
        ctx = torch.tensor([tokens[-L:]], dtype=torch.long)
        h = model.embed_tokens(ctx)
        with torch.no_grad():
            out, state, _, _ = model(h, state, adaptive=False, step=i, tokens=ctx)

        logits = model.lm_head(out)

        if use_cache:
            h_aug, logits = model.process_with_cache(out, logits, use_cache=True)

        next_token = logits[:, -1, :].argmax(dim=-1).item()
        tokens.append(next_token)

    elapsed = time.time() - t_start
    text = tok.decode(tokens[len(prompt_ids[0]):])
    cache_mb = model.cache_size_mb() if use_cache else 0

    return {
        'elapsed': elapsed,
        'ms_per_token': (elapsed / n_tokens) * 1000,
        'tokens_per_sec': n_tokens / elapsed,
        'text': text,
        'cache_mb': cache_mb,
    }


def main():
    print("=" * 60)
    print("LOGITCACHE GENERATION BENCHMARK")
    print("=" * 60)

    model, cfg = load_model()
    tok = load_russian_tokenizer()
    prompt = "Привет, меня зовут"
    enc = tok.encode(prompt)
    prompt_ids = enc.ids
    print(f"\nPrompt: '{prompt}' ({len(prompt_ids)} tokens)")

    N = 20

    # Test 1: Without cache (baseline)
    print(f"\n--- Test 1: WITHOUT cache ({N} tokens) ---")
    model.logit_cache = None
    r1 = bench_generation(model, [prompt_ids], N, use_cache=False)

    # Test 2: With cache
    print(f"\n--- Test 2: WITH cache ({N} tokens) ---")
    model.logit_cache = LogitCacheAttention(
        D=cfg.D, V=cfg.vocab, n_layers=cfg.n_layers,
        max_tokens=1000, n_heads=8,
    )
    r2 = bench_generation(model, [prompt_ids], N, use_cache=True)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'No Cache':<20} {'With Cache':<20}")
    print("-" * 65)
    print(f"{'Time (sec)':<25} {r1['elapsed']:<20.3f} {r2['elapsed']:<20.3f}")
    print(f"{'Tokens/sec':<25} {r1['tokens_per_sec']:<20.1f} {r2['tokens_per_sec']:<20.1f}")
    print(f"{'ms/token':<25} {r1['ms_per_token']:<20.1f} {r2['ms_per_token']:<20.1f}")
    print(f"{'Cache size (MB)':<25} {'N/A':<20} {r2['cache_mb']:<20.4f}")

    # Quality comparison
    print(f"\n--- Quality ---")
    texts_match = r1['text'] == r2['text']
    print(f"Texts identical: {'YES' if texts_match else 'NO'}")
    print(f"\nWithout cache:")
    print(f"  {r1['text']}")
    print(f"\nWith cache:")
    print(f"  {r2['text']}")

    # Cache projection
    print(f"\n--- Cache Projection ---")
    avg_bytes = r2['cache_mb'] * 1024 * 1024 / N
    print(f"Avg entry: {avg_bytes:.0f} bytes ({avg_bytes/1024:.2f} KB)")
    print(f"Projected 1M tokens: {avg_bytes * 1_000_000 / (1024*1024):.2f} MB")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
