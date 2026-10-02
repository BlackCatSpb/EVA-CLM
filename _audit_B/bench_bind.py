"""Isolated cost of the bind circular-index gather + FFT alternative."""
import sys, time, statistics
sys.path.insert(0, r'C:\EVA_CLM_OPT')
import torch
torch.set_num_threads(8)

K = 32
N = 384 * 4 * 3   # B*L*S*nd for mini: 4608
idx = torch.tensor([[(t + n) % K for n in range(K)] for t in range(K)], dtype=torch.long)


def bench(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), statistics.median(ts)


def cur():
    a = torch.randn(N, K, requires_grad=True)
    b = torch.randn(N, K, requires_grad=True)
    bg = b[..., idx]
    hrr = torch.einsum('bt,btn->bn', a, bg)
    hrr.sum().backward()


def fft_version():
    a = torch.randn(N, K, requires_grad=True)
    b = torch.randn(N, K, requires_grad=True)
    b2 = torch.roll(b.flip(-1), 1, dims=-1)
    out = torch.fft.irfft(torch.fft.rfft(a, n=K) * torch.fft.rfft(b2, n=K), n=K)
    out.sum().backward()


# correctness
a = torch.randn(5, K)
b = torch.randn(5, K)
ref = torch.einsum('bt,btn->bn', a, b[..., idx])
b2 = torch.roll(b.flip(-1), 1, dims=-1)
alt = torch.fft.irfft(torch.fft.rfft(a, n=K) * torch.fft.rfft(b2, n=K), n=K)
print('fft max abs err:', float((ref - alt).abs().max()))

m1 = bench(cur)
m2 = bench(fft_version)
print(f'current gather+einsum fwd+bwd: min={m1[0]*1000:.2f}ms med={m1[1]*1000:.2f}ms')
print(f'fft version        fwd+bwd: min={m2[0]*1000:.2f}ms med={m2[1]*1000:.2f}ms  speedup={m1[0]/m2[0]:.2f}x')

# forward only
def fwd_cur():
    a = torch.randn(N, K); b = torch.randn(N, K)
    return torch.einsum('bt,btn->bn', a, b[..., idx])
def fwd_fft():
    a = torch.randn(N, K); b = torch.randn(N, K)
    b2 = torch.roll(b.flip(-1), 1, dims=-1)
    return torch.fft.irfft(torch.fft.rfft(a, n=K) * torch.fft.rfft(b2, n=K), n=K)
print('fwd only current:', [f'{t*1000:.2f}ms' for t in bench(fwd_cur)])
print('fwd only fft    :', [f'{t*1000:.2f}ms' for t in bench(fwd_fft)])

# memory of bg
print('bg tensor elems:', N * K * K, '= %.1f MB fp32' % (N * K * K * 4 / 1e6))
