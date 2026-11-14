"""K = 2 (and K = 1) cumulant propagation against Monte Carlo along depth proportional to width, He-ReLU, sigma = 0.

usage: k2_scaling.py N_WIDTH LMAX SEED NTRUTH_LOG2 [THREADS]
Writes results/k2s_n{n}_L{Lmax}_seed{seed}.json with per-layer nMSE (ARC metric, debiased by 1/N_truth) for every
depth l <= Lmax (truncation convention: depth l is the first l layers of one net), the K = 2 FLOP count (ARC's counter),
and the plain-MC gain at equal cost. Uses ARC's code via work/arc-noisy/kprop_noisy.py (read-only import).
"""
import json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ARCN = os.path.abspath(os.path.join(HERE, '..', '..', 'arc-noisy'))
sys.path.insert(0, ARCN)
import anet                      # noqa: E402
import kprop_noisy as KN         # noqa: E402

n, Lmax, seed, nlog2 = (int(a) for a in sys.argv[1:5])
threads = int(sys.argv[5]) if len(sys.argv) > 5 else 4
torch.set_num_threads(threads)
out = os.path.join(HERE, 'results', f'k2s_n{n}_L{Lmax}_seed{seed}.json')
os.makedirs(os.path.dirname(out), exist_ok=True)
if os.path.exists(out):
    sys.exit(0)

net = anet.Net(seed, n=n, L=Lmax, sigma=0.0)

# Monte Carlo truth for every layer, float32 matmuls, float64 accumulation
t0 = time.time()
N = 1 << nlog2
B = 4096 if n <= 512 else 2048
g = torch.Generator().manual_seed(4_242_001 + 13 * seed + 7 * n)
Wf = [w.float() for w in net.W]
S1 = torch.zeros(Lmax, n, dtype=torch.float64)
S2 = torch.zeros(Lmax, n, dtype=torch.float64)
done = 0
while done < N:
    b = min(B, N - done)
    h = torch.randn(b, n, generator=g, dtype=torch.float32)
    for l, w in enumerate(Wf):
        h = torch.relu(h @ w.T)
        hd = h.double()
        S1[l] += hd.sum(0)
        S2[l] += (hd * hd).sum(0)
    done += b
mean = (S1 / N).numpy()
var = (S2 / N).numpy() - mean ** 2
t_truth = time.time() - t0

res = dict(n=n, Lmax=Lmax, seed=seed, N_truth=N, t_truth=t_truth, var_mean=[float(v.mean()) for v in var],
           mean_sq=[float((m ** 2).mean()) for m in mean])
for name, k in (('k1', 1), ('k2', 2)):
    t0 = time.time()
    means, fl, sec = KN.kprop_noisy(net, Lmax, k, factor=False, kind='simple', keep=True, avg_metric=True)
    nm = []
    for l in range(Lmax):
        est = means[l].numpy()
        nm.append(float(((est - mean[l]) ** 2).mean() / var[l].mean() - 1.0 / N))
    res[name] = dict(nmse=nm, flops_total=float(fl), secs=sec)
# plain MC cost of one path to depth l (flopscope-style): l * (n (2n - 1) + n); K2's ARC count grows ~ linearly in l
cost_path_per_layer = n * (2 * n - 1) + n
res['cost_path_per_layer'] = cost_path_per_layer
res['N_equiv_k2'] = res['k2']['flops_total'] / (Lmax * cost_path_per_layer)
json.dump(res, open(out, 'w'))
print(f"n={n} L={Lmax} seed={seed} truth {t_truth:.0f}s k2 {res['k2']['secs']:.0f}s "
      f"nmse_k2(L)={res['k2']['nmse'][-1]:.3e} Neq={res['N_equiv_k2']:.1f}", flush=True)
