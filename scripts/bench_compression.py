"""
Benchmark: Memory Bank state compression speed test
Compares generation with/without tau-adaptive compression.
"""
import sys, time, torch, gc, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.serialization import add_safe_globals
from core import EVAConfig, EVAStack
from core.config import WideBindConfig
add_safe_globals([EVAConfig, WideBindConfig])
from scripts.generate import load_russian_tokenizer


def load_model():
    ckpt_path = r"C:\Users\black\OneDrive\Desktop\EVA CLM\checkponts\best 19.pt"
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']
    model = EVAStack(cfg)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval().float()
    return model, cfg


def bench_generation(model, prompt_tokens, n_new_tokens, use_compression):
    """Benchmark one generation run."""
    gc.collect()

    tok = load_russian_tokenizer()
    L = 128

    tokens = list(prompt_tokens[0])
    state = None

    # Warmup
    with torch.no_grad():
        ctx = torch.tensor([tokens[-L:]], dtype=torch.long)
        h = model.embed_tokens(ctx)
        out, state, _, _ = model(h, state, adaptive=False, step=0, tokens=ctx)
        if hasattr(model, 'memory_bank') and model.memory_bank is not None:
            model.memory_bank.reset()

    # Benchmark
    start_time = time.time()

    for i in range(n_new_tokens):
        ctx = torch.tensor([tokens[-L:]], dtype=torch.long)
        h = model.embed_tokens(ctx)
        with torch.no_grad():
            out, state, _, _ = model(h, state, adaptive=False, step=i, tokens=ctx)

        logits = out[:, -1, :]
        next_token = logits.argmax(dim=-1).item()
        tokens.append(next_token)

        # Compress/decompress between steps
        if use_compression and hasattr(model, 'memory_bank') and model.memory_bank is not None:
            model.memory_bank.compress_state()
            model.memory_bank.decompress_state()

    elapsed = time.time() - start_time
    text = tok.decode(tokens[len(prompt_tokens[0]):])

    return {
        'elapsed': elapsed,
        'tokens_per_sec': n_new_tokens / elapsed,
        'ms_per_token': (elapsed / n_new_tokens) * 1000,
        'text': text,
        'n_tokens': n_new_tokens,
    }


def main():
    print("=" * 60)
    print("MEMORY BANK COMPRESSION BENCHMARK")
    print("=" * 60)

    model, cfg = load_model()

    tok = load_russian_tokenizer()
    prompt = "Привет, меня зовут"
    enc = tok.encode(prompt)
    prompt_ids = enc.ids
    print(f"\nPrompt: '{prompt}' ({len(prompt_ids)} tokens)")

    N_NEW = 20

    # Test 1: Without compression
    print(f"\n--- Test 1: WITHOUT compression ({N_NEW} tokens) ---")
    r1 = bench_generation(model, [prompt_ids], N_NEW, use_compression=False)

    # Reset memory bank
    if hasattr(model, 'memory_bank') and model.memory_bank is not None:
        model.memory_bank.reset()

    # Test 2: With compression
    print(f"\n--- Test 2: WITH compression ({N_NEW} tokens) ---")
    r2 = bench_generation(model, [prompt_ids], N_NEW, use_compression=True)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'No Compression':<20} {'With Compression':<20}")
    print("-" * 65)
    print(f"{'Time (sec)':<25} {r1['elapsed']:<20.3f} {r2['elapsed']:<20.3f}")
    print(f"{'Tokens/sec':<25} {r1['tokens_per_sec']:<20.1f} {r2['tokens_per_sec']:<20.1f}")
    print(f"{'ms/token':<25} {r1['ms_per_token']:<20.1f} {r2['ms_per_token']:<20.1f}")

    # Compression ratio
    if hasattr(model, 'memory_bank') and model.memory_bank is not None:
        stats = model.memory_bank.compression_stats()
        print(f"\n--- Compression Stats ---")
        print(f"Original size:   {stats['original_kb']:.1f} KB")
        print(f"Compressed size: {stats['compressed_kb']:.1f} KB")
        print(f"Compression ratio: {stats['ratio']:.1f}x")

    # Quality check
    texts_match = r1['text'] == r2['text']
    print(f"\n--- Quality ---")
    print(f"Texts identical: {'YES' if texts_match else 'NO'}")
    if not texts_match:
        print(f"\nWithout compression:\n  {r1['text']}")
        print(f"\nWith compression:\n  {r2['text']}")

    print("\n" + "=" * 60)
    if texts_match:
        print("BENCHMARK PASSED: compression preserves quality")
    else:
        print("BENCHMARK FAILED: compression changed output")
    print("=" * 60)


if __name__ == '__main__':
    main()
