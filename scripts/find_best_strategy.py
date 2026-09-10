"""
Zeckendorf-based compression for Memory Bank.
Deterministic, sparse, learnable by the model.
"""
import sys, os, time, torch, gc, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.serialization import add_safe_globals
from core import EVAConfig, EVAStack
from core.config import WideBindConfig
add_safe_globals([EVAConfig, WideBindConfig])
from scripts.generate import load_russian_tokenizer


def load_model():
    ckpt = torch.load(r"C:\Users\black\OneDrive\Desktop\EVA CLM\checkponts\best 19.pt",
                      map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']
    model = EVAStack(cfg)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval().float()
    return model, cfg


def encode(prompt):
    tok = load_russian_tokenizer()
    return tok.encode(prompt).ids


def decode(ids):
    tok = load_russian_tokenizer()
    return tok.decode(ids)


# ─── Zeckendorf compression ──────────────────────────────────

def _fib_sequence(n):
    """Generate Fibonacci sequence up to n."""
    fib = [1, 2]
    while fib[-1] <= n:
        fib.append(fib[-1] + fib[-2])
    return fib[:-1]


def zeckendorf_encode(value, fib_sequence):
    """
    Encode a single value using Zeckendorf representation.
    Returns list of Fibonacci indices that sum to value.
    """
    if value <= 0:
        return []
    
    remaining = int(value)
    indices = []
    prev_idx = -2  # Track previous index to ensure non-consecutive
    
    for i in range(len(fib_sequence) - 1, -1, -1):
        if fib_sequence[i] <= remaining and i > prev_idx + 1:
            indices.append(i)
            remaining -= fib_sequence[i]
            prev_idx = i
    
    return indices


def zeckendorf_decode(indices, fib_sequence):
    """Decode Zeckendorf indices back to value."""
    return sum(fib_sequence[i] for i in indices)


def compress_zeckendorf(tensor, n_levels=8, scale_factor=1000):
    """
    Compress tensor using Zeckendorf representation.
    
    1. Quantize to integers: int_val = round(value * scale_factor)
    2. Encode each integer using Zeckendorf (sparse Fibonacci representation)
    3. Store only active bit positions
    
    Args:
        tensor: Input tensor
        n_levels: Number of Fibonacci levels to use
        scale_factor: Scale factor for quantization
    
    Returns:
        compressed: List of (indices_per_element)
        metadata: min_val, scale_factor for reconstruction
    """
    flat = tensor.reshape(-1).float()
    t_min, t_max = flat.min(), flat.max()
    
    # Normalize to [0, 1] then scale to integer range
    if t_max - t_min < 1e-8:
        normalized = torch.zeros_like(flat)
    else:
        normalized = (flat - t_min) / (t_max - t_min)
    
    # Scale to integer range for Zeckendorf encoding
    int_values = (normalized * (2**n_levels - 1)).clamp(0, 2**n_levels - 1).long()
    
    # Generate Fibonacci sequence for this range
    fib = _fib_sequence(2**n_levels)
    
    # Encode each value
    compressed = []
    for v in int_values.tolist():
        indices = zeckendorf_encode(v, fib)
        compressed.append(indices)
    
    return compressed, {
        'min': t_min.item(),
        'max': t_max.item(),
        'shape': tensor.shape,
        'dtype': tensor.dtype,
        'n_levels': n_levels,
        'scale_factor': scale_factor,
        'fib': fib,
    }


def decompress_zeckendorf(compressed, metadata):
    """Decompress Zeckendorf-encoded tensor."""
    fib = metadata['fib']
    n_levels = metadata['n_levels']
    
    # Decode each value
    int_values = []
    for indices in compressed:
        val = zeckendorf_decode(indices, fib)
        int_values.append(val)
    
    # Convert back to float
    int_tensor = torch.tensor(int_values, dtype=torch.float32)
    normalized = int_tensor / (2**n_levels - 1)
    
    # Denormalize
    t_min = metadata['min']
    t_max = metadata['max']
    flat = normalized * (t_max - t_min) + t_min
    
    return flat.reshape(metadata['shape']).to(metadata['dtype'])


def zeckendorf_size_bytes(compressed, metadata):
    """Estimate compressed size in bytes."""
    # Each element: list of indices (variable length)
    # Store as: count (1 byte) + indices (1 byte each, max 8 levels)
    total_bytes = 0
    for indices in compressed:
        total_bytes += 1 + len(indices)  # count + indices
    return total_bytes


# ─── Strategies ──────────────────────────────────────────────

class StrategyZeckendorf:
    """Zeckendorf compression for all levels."""
    name = "zeckendorf"
    def __init__(self, model, n_levels=8):
        self.model = model
        self.n_levels = n_levels
        self._cache = {}
    
    def compress(self):
        mb = self.model.memory_bank
        self._cache = {
            'l1': compress_zeckendorf(mb.l1.buf.data, self.n_levels),
            'l2_k': compress_zeckendorf(mb.l2.keys.data, self.n_levels),
            'l2_v': compress_zeckendorf(mb.l2.vals.data, self.n_levels),
            'l3_k': compress_zeckendorf(mb.l3.concept_keys.data, self.n_levels),
            'l3_v': compress_zeckendorf(mb.l3.concept_vals.data, self.n_levels),
        }
    
    def decompress(self):
        mb = self.model.memory_bank
        c = self._cache
        mb.l1.buf.data.copy_(decompress_zeckendorf(*c['l1']))
        mb.l2.keys.data.copy_(decompress_zeckendorf(*c['l2_k']))
        mb.l2.vals.data.copy_(decompress_zeckendorf(*c['l2_v']))
        mb.l3.concept_keys.data.copy_(decompress_zeckendorf(*c['l3_k']))
        mb.l3.concept_vals.data.copy_(decompress_zeckendorf(*c['l3_v']))
    
    def size_bytes(self):
        total = 0
        for key, (compressed, meta) in self._cache.items():
            total += zeckendorf_size_bytes(compressed, meta)
        return total


class StrategyZeckendorfSelective:
    """Zeckendorf for stable levels (L2/L3), uniform8 for volatile (L1)."""
    name = "zeckendorf_selective"
    def __init__(self, model, n_levels_l2=8, n_levels_l3=6):
        self.model = model
        self.n_levels_l2 = n_levels_l2
        self.n_levels_l3 = n_levels_l3
        self._cache = {}
    
    def compress(self):
        mb = self.model.memory_bank
        
        # L1: uniform8 (volatile, fast)
        l1 = mb.l1.buf.data
        l1_flat = l1.reshape(-1).float()
        l1_min, l1_max = l1_flat.min(), l1_flat.max()
        l1_scale = (l1_max - l1_min) / 255.0
        if l1_scale < 1e-8:
            l1_idx = torch.zeros(l1_flat.shape, dtype=torch.uint8)
        else:
            l1_idx = ((l1_flat - l1_min) / l1_scale).clamp(0, 255).to(torch.uint8)
        
        self._cache = {
            'l1': (l1_idx, l1_min, l1_scale, l1.shape),
            'l2_k': compress_zeckendorf(mb.l2.keys.data, self.n_levels_l2),
            'l2_v': compress_zeckendorf(mb.l2.vals.data, self.n_levels_l2),
            'l3_k': compress_zeckendorf(mb.l3.concept_keys.data, self.n_levels_l3),
            'l3_v': compress_zeckendorf(mb.l3.concept_vals.data, self.n_levels_l3),
        }
    
    def decompress(self):
        mb = self.model.memory_bank
        c = self._cache
        
        # L1: uniform8
        idx, mn, sc, shape = c['l1']
        mb.l1.buf.data.copy_((idx.float() * sc + mn).reshape(shape))
        
        # L2/L3: Zeckendorf
        mb.l2.keys.data.copy_(decompress_zeckendorf(*c['l2_k']))
        mb.l2.vals.data.copy_(decompress_zeckendorf(*c['l2_v']))
        mb.l3.concept_keys.data.copy_(decompress_zeckendorf(*c['l3_k']))
        mb.l3.concept_vals.data.copy_(decompress_zeckendorf(*c['l3_v']))
    
    def size_bytes(self):
        total = 0
        # L1: uniform8
        idx, mn, sc, shape = self._cache['l1']
        total += idx.numel() + 8
        # L2/L3: Zeckendorf
        for key in ['l2_k', 'l2_v', 'l3_k', 'l3_v']:
            compressed, meta = self._cache[key]
            total += zeckendorf_size_bytes(compressed, meta)
        return total


class StrategyZeckendorfAdaptive:
    """Adaptive Zeckendorf: different n_levels based on tau."""
    name = "zeckendorf_adaptive"
    def __init__(self, model, base_levels=8):
        self.model = model
        self.base_levels = base_levels
        self._cache = {}
    
    def compress(self):
        mb = self.model.memory_bank
        
        # Get tau values if available
        tau_config = getattr(mb, 'tau_config', None)
        if tau_config is not None and hasattr(tau_config, 'tau_norm'):
            tau_norm = tau_config.tau_norm.mean().item()
        else:
            tau_norm = 0.5
        
        # Adaptive levels: higher tau → more levels (slower, more precise)
        # Lower tau → fewer levels (faster, less precise)
        n_l1 = max(4, int(self.base_levels * (1 - tau_norm)))  # volatile: fewer levels
        n_l2 = self.base_levels  # medium
        n_l3 = min(12, int(self.base_levels * (1 + tau_norm)))  # stable: more levels
        
        self._cache = {
            'l1': compress_zeckendorf(mb.l1.buf.data, n_l1),
            'l2_k': compress_zeckendorf(mb.l2.keys.data, n_l2),
            'l2_v': compress_zeckendorf(mb.l2.vals.data, n_l2),
            'l3_k': compress_zeckendorf(mb.l3.concept_keys.data, n_l3),
            'l3_v': compress_zeckendorf(mb.l3.concept_vals.data, n_l3),
        }
        self._cache['meta'] = {'n_l1': n_l1, 'n_l2': n_l2, 'n_l3': n_l3}
    
    def decompress(self):
        mb = self.model.memory_bank
        c = self._cache
        mb.l1.buf.data.copy_(decompress_zeckendorf(*c['l1']))
        mb.l2.keys.data.copy_(decompress_zeckendorf(*c['l2_k']))
        mb.l2.vals.data.copy_(decompress_zeckendorf(*c['l2_v']))
        mb.l3.concept_keys.data.copy_(decompress_zeckendorf(*c['l3_k']))
        mb.l3.concept_vals.data.copy_(decompress_zeckendorf(*c['l3_v']))
    
    def size_bytes(self):
        total = 0
        for key in ['l1', 'l2_k', 'l2_v', 'l3_k', 'l3_v']:
            compressed, meta = self._cache[key]
            total += zeckendorf_size_bytes(compressed, meta)
        return total


# ─── Existing strategies for comparison ──────────────────────

def compress_uniform8(tensor):
    flat = tensor.reshape(-1).float()
    t_min, t_max = flat.min(), flat.max()
    scale = (t_max - t_min) / 255.0
    if scale < 1e-8:
        idx = torch.zeros(flat.shape, dtype=torch.uint8)
    else:
        idx = ((flat - t_min) / scale).clamp(0, 255).to(torch.uint8)
    return idx, t_min, scale


def decompress_uniform8(idx, t_min, scale, shape):
    return (idx.float() * scale + t_min).reshape(shape)


class StrategyUniform8:
    name = "uniform8"
    def __init__(self, model):
        self.model = model
        self._cache = {}
    
    def compress(self):
        mb = self.model.memory_bank
        self._cache = {
            'l1': compress_uniform8(mb.l1.buf.data),
            'l2_k': compress_uniform8(mb.l2.keys.data),
            'l2_v': compress_uniform8(mb.l2.vals.data),
            'l3_k': compress_uniform8(mb.l3.concept_keys.data),
            'l3_v': compress_uniform8(mb.l3.concept_vals.data),
        }
    
    def decompress(self):
        mb = self.model.memory_bank
        c = self._cache
        mb.l1.buf.data.copy_(decompress_uniform8(*c['l1'], mb.l1.buf.shape))
        mb.l2.keys.data.copy_(decompress_uniform8(*c['l2_k'], mb.l2.keys.shape))
        mb.l2.vals.data.copy_(decompress_uniform8(*c['l2_v'], mb.l2.vals.shape))
        mb.l3.concept_keys.data.copy_(decompress_uniform8(*c['l3_k'], mb.l3.concept_keys.shape))
        mb.l3.concept_vals.data.copy_(decompress_uniform8(*c['l3_v'], mb.l3.concept_vals.shape))
    
    def size_bytes(self):
        mb = self.model.memory_bank
        return sum(v[0].numel() + 8 for v in self._cache.values())


# ─── Benchmark ───────────────────────────────────────────────

def cosine_sim(a, b):
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    return torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def bench_strategy(model, prompt_ids, strategy, n_tokens=20, label=""):
    """Run generation with given strategy, return metrics."""
    gc.collect()
    L = 128
    tokens = list(prompt_ids)
    state = None

    # Warmup
    with torch.no_grad():
        ctx = torch.tensor([tokens[-L:]], dtype=torch.long)
        h = model.embed_tokens(ctx)
        out, state, _, _ = model(h, state, adaptive=False, step=0, tokens=ctx)
        model.memory_bank.reset()

    # Save original state
    orig_state = {
        'l1': model.memory_bank.l1.buf.data.clone(),
        'l2_k': model.memory_bank.l2.keys.data.clone(),
        'l2_v': model.memory_bank.l2.vals.data.clone(),
        'l3_k': model.memory_bank.l3.concept_keys.data.clone(),
        'l3_v': model.memory_bank.l3.concept_vals.data.clone(),
    }

    # Benchmark
    t_start = time.time()
    for i in range(n_tokens):
        ctx = torch.tensor([tokens[-L:]], dtype=torch.long)
        h = model.embed_tokens(ctx)
        with torch.no_grad():
            out, state, _, _ = model(h, state, adaptive=False, step=i, tokens=ctx)
        next_token = out[:, -1, :].argmax(dim=-1).item()
        tokens.append(next_token)

        strategy.compress()
        strategy.decompress()

    elapsed = time.time() - t_start

    # Quality: cosine similarity of final state
    cos_l1 = cosine_sim(orig_state['l1'], model.memory_bank.l1.buf.data)
    cos_l2k = cosine_sim(orig_state['l2_k'], model.memory_bank.l2.keys.data)
    cos_l2v = cosine_sim(orig_state['l2_v'], model.memory_bank.l2.vals.data)
    cos_l3k = cosine_sim(orig_state['l3_k'], model.memory_bank.l3.concept_keys.data)
    cos_l3v = cosine_sim(orig_state['l3_v'], model.memory_bank.l3.concept_vals.data)

    text = decode(tokens[len(prompt_ids):])
    comp_bytes = strategy.size_bytes()
    orig_bytes = sum(v.numel() * 4 for v in orig_state.values())

    return {
        'label': label,
        'elapsed': elapsed,
        'ms_per_token': (elapsed / n_tokens) * 1000,
        'tokens_per_sec': n_tokens / elapsed,
        'text': text,
        'orig_kb': orig_bytes / 1024,
        'comp_kb': comp_bytes / 1024,
        'ratio': orig_bytes / max(comp_bytes, 1),
        'cos_l1': cos_l1,
        'cos_l2k': cos_l2k,
        'cos_l2v': cos_l2v,
        'cos_l3k': cos_l3k,
        'cos_l3v': cos_l3v,
        'cos_mean': (cos_l1 + cos_l2k + cos_l2v + cos_l3k + cos_l3v) / 5,
    }


def main():
    print("=" * 100)
    print("ZECKENDORF COMPRESSION STRATEGY FINDER")
    print("=" * 100)

    model, cfg = load_model()
    prompt = "Привет, меня зовут"
    prompt_ids = encode(prompt)
    print(f"Prompt: '{prompt}' ({len(prompt_ids)} tokens)")

    N = 20
    results = []

    # Strategy 1: Uniform8 (baseline)
    print("\n[1/7] uniform8...")
    s = StrategyUniform8(model)
    results.append(bench_strategy(model, prompt_ids, s, N, "uniform8"))
    model.memory_bank.reset()

    # Strategy 2: Zeckendorf 6 levels
    print("[2/7] zeckendorf_6...")
    s = StrategyZeckendorf(model, n_levels=6)
    results.append(bench_strategy(model, prompt_ids, s, N, "zeckendorf_6"))
    model.memory_bank.reset()

    # Strategy 3: Zeckendorf 8 levels
    print("[3/7] zeckendorf_8...")
    s = StrategyZeckendorf(model, n_levels=8)
    results.append(bench_strategy(model, prompt_ids, s, N, "zeckendorf_8"))
    model.memory_bank.reset()

    # Strategy 4: Zeckendorf 10 levels
    print("[4/7] zeckendorf_10...")
    s = StrategyZeckendorf(model, n_levels=10)
    results.append(bench_strategy(model, prompt_ids, s, N, "zeckendorf_10"))
    model.memory_bank.reset()

    # Strategy 5: Zeckendorf selective (L1=uniform8, L2/L3=zeckendorf)
    print("[5/7] zeckendorf_selective...")
    s = StrategyZeckendorfSelective(model, n_levels_l2=8, n_levels_l3=6)
    results.append(bench_strategy(model, prompt_ids, s, N, "zeckendorf_sel"))
    model.memory_bank.reset()

    # Strategy 6: Zeckendorf adaptive
    print("[6/7] zeckendorf_adaptive...")
    s = StrategyZeckendorfAdaptive(model, base_levels=8)
    results.append(bench_strategy(model, prompt_ids, s, N, "zeckendorf_adp"))
    model.memory_bank.reset()

    # Strategy 7: No compression (baseline)
    print("[7/7] no compression...")
    gc.collect()
    tokens = list(prompt_ids)
    state = None
    with torch.no_grad():
        ctx = torch.tensor([tokens[-L:]], dtype=torch.long)
        h = model.embed_tokens(ctx)
        out, state, _, _ = model(h, state, adaptive=False, step=0, tokens=ctx)
        model.memory_bank.reset()
    t0 = time.time()
    for i in range(N):
        ctx = torch.tensor([tokens[-L:]], dtype=torch.long)
        h = model.embed_tokens(ctx)
        with torch.no_grad():
            out, state, _, _ = model(h, state, adaptive=False, step=i, tokens=ctx)
        tokens.append(out[:, -1, :].argmax(dim=-1).item())
    elapsed = time.time() - t0
    text = decode(tokens[len(prompt_ids):])
    results.append({
        'label': 'NONE',
        'elapsed': elapsed,
        'ms_per_token': (elapsed / N) * 1000,
        'tokens_per_sec': N / elapsed,
        'text': text,
        'orig_kb': 110.0,
        'comp_kb': 110.0,
        'ratio': 1.0,
        'cos_l1': 1.0, 'cos_l2k': 1.0, 'cos_l2v': 1.0,
        'cos_l3k': 1.0, 'cos_l3v': 1.0, 'cos_mean': 1.0,
    })

    # Results table
    print("\n" + "=" * 120)
    print(f"{'Strategy':<22} {'ms/tok':<10} {'tok/s':<8} {'KB_orig':<10} {'KB_comp':<10} "
          f"{'Ratio':<8} {'Cos(L1)':<9} {'Cos(L2k)':<9} {'Cos(L3v)':<9} {'CosMean':<9}")
    print("-" * 120)
    for r in results:
        print(f"{r['label']:<22} {r['ms_per_token']:<10.0f} {r['tokens_per_sec']:<8.2f} "
              f"{r['orig_kb']:<10.1f} {r['comp_kb']:<10.1f} "
              f"{r['ratio']:<8.1f} {r['cos_l1']:<9.4f} {r['cos_l2k']:<9.4f} "
              f"{r['cos_l3v']:<9.4f} {r['cos_mean']:<9.4f}")

    # Text comparison
    print("\n" + "=" * 120)
    print("TEXT COMPARISON")
    print("=" * 120)
    base_text = results[-1]['text']
    for r in results:
        match = "SAME" if r['text'] == base_text else "DIFF"
        print(f"\n[{r['label']}] {match}:")
        print(f"  {r['text']}")

    # Recommendation
    print("\n" + "=" * 120)
    print("RECOMMENDATION")
    print("=" * 120)
    # Best: highest ratio with cos_mean > 0.99
    good = [r for r in results[:-1] if r['cos_mean'] > 0.99]
    if good:
        best = max(good, key=lambda x: x['ratio'])
        print(f"Best strategy (quality preserved): {best['label']}")
        print(f"  Ratio: {best['ratio']:.1f}x | CosMean: {best['cos_mean']:.4f} | "
              f"Speed: {best['ms_per_token']:.0f} ms/tok")
    else:
        # Find best trade-off
        best = max(results[:-1], key=lambda x: x['ratio'] * x['cos_mean'])
        print(f"Best trade-off: {best['label']}")
        print(f"  Ratio: {best['ratio']:.1f}x | CosMean: {best['cos_mean']:.4f} | "
              f"Speed: {best['ms_per_token']:.0f} ms/tok")


if __name__ == '__main__':
    L = 128
    main()
