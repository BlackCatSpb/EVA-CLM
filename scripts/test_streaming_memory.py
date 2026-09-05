"""Streaming memory bank v2: CONCAT approach.

Instead of attention-based read, concatenate memory summary directly.
Memory = running average of previous sentence embeddings.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, time

D = 48
N_LAYERS = 2
N_HEADS = 2
VOCAB = 32
BATCH = 64
STEPS = 1500
LR = 5e-4
ATTN_WINDOW = 16
GAP = 40
S1_LEN = 6
S2_LEN = 6
SEQ_LEN = S1_LEN + 1 + GAP + S2_LEN + 1 + 8  # = 62
N_MEMORY = 4  # number of memory slots to keep


class RunningMemory(nn.Module):
    """Keep top-N sentence embeddings as memory."""
    def __init__(self, n_mem=N_MEMORY):
        super().__init__()
        self.n_mem = n_mem
        self.mem = nn.Parameter(torch.randn(n_mem, D) * 0.02)
        self.write_idx = 0
        self.writes = 0
        self.proj = nn.Linear(D, D)

    def write(self, embedding):
        """Write to slot (round-robin)."""
        s = self.write_idx % self.n_mem
        self.mem.data[s] = embedding.detach()
        self.write_idx += 1
        self.writes += 1

    def read(self):
        """Return projected memory summary: (D,)"""
        return self.proj(self.mem.mean(0))  # mean over slots -> (D,)

    def reset(self):
        self.write_idx = 0
        self.writes = 0


class WindowedAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.window = window_size

    def forward(self, x):
        B, S, D = x.shape
        mask = torch.ones(S, S, device=x.device, dtype=torch.bool)
        for i in range(S):
            start = max(0, i - self.window)
            mask[i, start:i+1] = False
        out, _ = self.attn(x, x, x, attn_mask=mask)
        return out


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = nn.LayerNorm(D)
        self.a = WindowedAttention(D, N_HEADS, ATTN_WINDOW)
        self.m = nn.Sequential(nn.Linear(D, D*2), nn.GELU(), nn.Linear(D*2, D))
    def forward(self, x):
        h = self.n(x)
        return x + self.a(h) + self.m(self.n(x + self.a(h)))


class Model(nn.Module):
    def __init__(self, use_mem=False, n_mem=N_MEMORY):
        super().__init__()
        self.use_mem = use_mem
        self.emb = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(SEQ_LEN, D)
        self.layers = nn.ModuleList([Layer() for _ in range(N_LAYERS)])
        self.head = nn.Linear(D, VOCAB)
        if use_mem:
            self.mem = RunningMemory(n_mem)
            self.mem_proj = nn.Linear(D, D)

    def forward(self, tok, tgt=None, reset_mem=False):
        B, S = tok.shape
        if self.use_mem and reset_mem:
            self.mem.reset()
        x = self.emb(tok) + self.pos(torch.arange(S, device=tok.device).unsqueeze(0))
        if self.use_mem:
            is_sep = (tok == 2)
            for b in range(B):
                sent_start = 0
                for t in range(S):
                    if is_sep[b, t]:
                        summary = x[b, sent_start:t+1].mean(0)
                        self.mem.write(summary)
                        sent_start = t + 1
            if self.mem.writes > 0:
                mem_read = self.mem.read()  # (n_mem * D,)
                mem_read = mem_read.unsqueeze(0).unsqueeze(0).expand(B, S, -1)
                x = x + self.mem_proj(mem_read)
        h = x
        for layer in self.layers:
            h = layer(h)
        logits = self.head(h)
        if tgt is not None:
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1))
            acc = (logits.argmax(-1) == tgt).float().mean()
            return loss, acc
        return logits


def gen_count_task(n=600):
    gap_start = S1_LEN + 1 + GAP
    data = []
    for _ in range(n):
        s1 = torch.randint(0, VOCAB, (S1_LEN,))
        n_unique = len(torch.unique(s1))
        s2 = torch.full((S2_LEN,), n_unique, dtype=torch.long)

        full = torch.zeros(SEQ_LEN, dtype=torch.long)
        full[:S1_LEN] = s1
        full[S1_LEN] = 2
        full[gap_start:gap_start+S2_LEN] = s2
        full[gap_start+S2_LEN] = 2
        tgt = torch.cat([full[1:], torch.tensor([0])])
        data.append((full, tgt))
    return data


def run_test(n=600):
    print(f"\n{'='*60}")
    print(f"COUNT TASK v2 (concat memory, window={ATTN_WINDOW}, gap={GAP})")
    print(f"{'='*60}")
    data = gen_count_task(n)
    split = int(n * 0.8)
    train_d, test_d = data[:split], data[split:]

    results = {}
    for label, use_mem in [("no-mem", False), ("with-mem", True)]:
        torch.manual_seed(42)
        model = Model(use_mem=use_mem, n_mem=N_MEMORY)
        np_ = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        t0 = time.time()

        model.train()
        for step in range(STEPS):
            idx = torch.randint(0, split, (BATCH,))
            tok = torch.stack([train_d[i][0] for i in idx])
            tgt = torch.stack([train_d[i][1] for i in idx])
            loss, acc = model(tok, tgt, reset_mem=(step % 20 == 0))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if (step+1) % 300 == 0:
                print(f"  [{label}] {step+1}: loss={loss.item():.3f} acc={acc.item():.4f}")

        elapsed = time.time() - t0

        model.eval()
        for split_name, dataset in [("train", train_d), ("test", test_d)]:
            tl, ta, ntok = 0, 0, 0
            with torch.no_grad():
                for tok, tgt in dataset:
                    loss, acc = model(tok.unsqueeze(0), tgt.unsqueeze(0), reset_mem=True)
                    tl += loss.item()
                    ta += acc.item() * SEQ_LEN
                    ntok += SEQ_LEN
            results[f'{label}_{split_name}'] = {
                'acc': ta/ntok, 'ppl': math.exp(tl/len(dataset)),
            }

        results[label] = {
            **results[f'{label}_train'],
            'test_acc': results[f'{label}_test']['acc'],
            'test_ppl': results[f'{label}_test']['ppl'],
            'params': np_, 'time': elapsed,
            'writes': model.mem.writes if use_mem else 0,
        }

    no = results['no-mem']
    wi = results['with-mem']
    print(f"\n  {'':12s}  {'Train':>16s}  {'Test':>16s}  {'Params':>8s}")
    print(f"  {'No mem':12s}  acc={no['acc']:.4f} ppl={no['ppl']:.1f}  acc={no['test_acc']:.4f} ppl={no['test_ppl']:.1f}  {no['params']:,}")
    print(f"  {'With mem':12s}  acc={wi['acc']:.4f} ppl={wi['ppl']:.1f}  acc={wi['test_acc']:.4f} ppl={wi['test_ppl']:.1f}  {wi['params']:,}")
    da = wi['test_acc'] - no['test_acc']
    dp = no['test_ppl'] - wi['test_ppl']
    print(f"  delta test: acc={da:+.4f}  ppl={dp:+.2f}  overhead {(wi['params']-no['params'])/no['params']*100:.0f}%")
    return da, dp


if __name__ == '__main__':
    print("=" * 60)
    print("STREAMING MEMORY BANK v2 - CONCAT")
    print("=" * 60)
    d, p = run_test()
    print(f"\n{'='*60}")
    if d > 0.05:
        print(f"  >> VALIDATED: memory helps (+{d:.4f})")
    elif d > 0.01:
        print(f"  >> Good: +{d:.4f}")
    elif d > 0:
        print(f"  >> Marginal: +{d:.4f}")
    else:
        print(f"  >> No benefit: {d:+.4f}")
    print("=" * 60)
