"""ARC's factorized K = 3 cumulant propagation against Monte Carlo along depth = alpha * width (alpha = 1/8, 1/4),
He-ReLU, sigma = 0. Separate runs to depth n/8 and n/4 give exact FLOP counts (ARC's counter) for each depth.
usage: k3f_scaling.py N_WIDTH SEED NTRUTH_LOG2 [THREADS]
Writes results/k3f_n{n}_seed{seed}.json."""
import json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', 'arc-noisy')))
import anet                      # noqa: E402
import kprop_noisy as KN         # noqa: E402

n, seed, nlog2 = (int(a) for a in sys.argv[1:4])
threads = int(sys.argv[4]) if len(sys.argv) > 4 else 4
torch.set_num_threads(threads)
out = os.path.join(HERE, 'results', f'k3f_n{n}_seed{seed}.json')
if os.path.exists(out):
    sys.exit(0)
Lmax = n // 4
net = anet.Net(seed, n=n, L=Lmax, sigma=0.0)
N = 1 << nlog2
g = torch.Generator().manual_seed(5_151_001 + 13 * seed + 7 * n)
Wf = [w.float() for w in net.W]
S1 = torch.zeros(Lmax, n, dtype=torch.float64); S2 = torch.zeros(Lmax, n, dtype=torch.float64)
done, B = 0, 4096
while done < N:
    b = min(B, N - done)
    h = torch.randn(b, n, generator=g, dtype=torch.float32)
    for l, w in enumerate(Wf):
        h = torch.relu(h @ w.T); hd = h.double()
        S1[l] += hd.sum(0); S2[l] += (hd * hd).sum(0)
    done += b
mean = (S1 / N).numpy(); var = (S2 / N).numpy() - mean ** 2
res = dict(n=n, seed=seed, N_truth=N, runs={})
cost_layer = n * (2 * n - 1) + n
for depth in (n // 8, n // 4):
    if depth < 1:
        continue
    sub = net.truncate(depth)
    for name, k, fac in (('k3f', 3, True), ('k2', 2, False)):
        est, fl, sec = KN.kprop_noisy(sub, depth, k, factor=fac, kind='simple', keep=False, avg_metric=True)
        e = est.numpy()
        nm = float(((e - mean[depth - 1]) ** 2).mean() / var[depth - 1].mean() - 1.0 / N)
        res['runs'][f'{name}_d{depth}'] = dict(depth=depth, nmse=nm, flops=float(fl), secs=sec,
                                               N_eq=float(fl) / (depth * cost_layer))
json.dump(res, open(out, 'w'))
print(f"n={n} seed={seed} " + " ".join(f"{k}: nmse {v['nmse']:.2e} Neq {v['N_eq']:.0f}" for k, v in res['runs'].items()),
      flush=True)
