"""Run E (PREREG): small-ball structure of the state distribution. NS noisy states z_c at layers 4, 8, 12, 16, 24 of one
net (states at layer c do not depend on the total depth). For every state, the number of other states within angle
theta on a log grid (float64 distances, theta = 2 arcsin(|u - v| / 2) for unit vectors), giving the correlation
integral C_c(theta) (share of pairs within theta) and per-state ball masses.
nice -n 19 ionice -c3 python r5_states.py --n 64 --seed 400 [--NS 65536]"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import kf

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=64); ap.add_argument('--seed', type=int, default=400)
ap.add_argument('--NS', type=int, default=65536); ap.add_argument('--layers', default='4,8,12,16,24')
ap.add_argument('--threads', type=int, default=3); ap.add_argument('--block', type=int, default=256)
ap.add_argument('--runs', default=os.path.join(HERE, '..', 'runs'))
A = ap.parse_args()
torch.set_num_threads(A.threads)
n, seed, NS = A.n, A.seed, A.NS
layers = [int(c) for c in A.layers.split(',')]
net = kf.make_net(seed, n, max(layers))
tag = f"n{n}_s{seed}"
GRID = np.logspace(-4, math.log10(math.pi), 49)
t0 = time.time()
g = kf.gen(80_000 + 1000 * seed + n)
Z = {c: torch.zeros(NS, n, dtype=kf.DT, device=kf.DEV) for c in layers}
bsz = min(4096, NS)
for b in range(0, NS, bsz):
    zs, _ = kf.forward(net, kf.draw_x(bsz, n, g), g)
    for c in layers:
        Z[c][b:b + bsz] = zs[c - 1]
res = {}
for c in layers:
    U = Z[c] / Z[c].norm(dim=-1, keepdim=True)
    d2grid = torch.tensor((2 * np.sin(GRID / 2)) ** 2, dtype=kf.DT, device=kf.DEV)        # |u - v|^2 at angle theta
    counts = torch.zeros(NS, len(GRID), dtype=torch.int64, device=kf.DEV)
    for i0 in range(0, NS, A.block):
        Ub = U[i0:i0 + A.block]
        d2 = (2.0 - 2.0 * (Ub @ U.T)).clamp_min(0.0)                       # (b, NS)
        d2[torch.arange(Ub.shape[0], device=kf.DEV), torch.arange(i0, i0 + Ub.shape[0], device=kf.DEV)] = 1e9   # drop self
        idx = torch.bucketize(d2, d2grid)                                   # bin j: d2grid[j-1] < d2 <= d2grid[j]
        h = torch.zeros(Ub.shape[0], len(GRID) + 1, dtype=torch.int64, device=kf.DEV)
        h.scatter_add_(1, idx, torch.ones_like(idx))
        counts[i0:i0 + Ub.shape[0]] = h.cumsum(1)[:, :len(GRID)]            # number within each theta
    C = counts.double().sum(0) / (NS * (NS - 1.0))                          # share of ordered pairs within theta
    lc = np.log(np.maximum(C.cpu().numpy(), 1e-300))
    slope = np.gradient(lc, np.log(GRID))
    q = {str(qq): (counts.double().quantile(qq, dim=0) / (NS - 1)).tolist() for qq in (0.01, 0.1, 0.5, 0.9)}
    npairs = counts.double().sum(0).cpu().numpy() / 2
    res[c] = {'grid': GRID.tolist(), 'C': C.tolist(), 'slope': slope.tolist(), 'pairs_within': npairs.tolist(),
              'mass_quantiles': q, 'median_angle_random': None}
    j = int(np.argmax(npairs >= 100))
    print(f"[{tag}] c={c}: C at theta = " + " ".join(f"{GRID[i]:.2g}:{C[i]:.2e}" for i in range(0, 49, 6)) +
          f" | smallest theta with >=100 pairs {GRID[j]:.3g}, local slope there {slope[j]:.2f} "
          f"(slopes at 0.03/0.1/0.3 rad: " + "/".join(f"{np.interp(math.log(t), np.log(GRID), slope):.2f}"
                                                         for t in (0.03, 0.1, 0.3)) + f")  {time.time()-t0:.0f}s",
          flush=True)
    np.save(os.path.join(A.runs, f'E_{tag}_c{c}_counts.npy'), counts.cpu().numpy().astype(np.int32))
fj = os.path.join(A.runs, f'E_{tag}.json')
prev = json.load(open(fj)) if os.path.exists(fj) else {}
prev.update({str(c): v for c, v in res.items()})
json.dump(prev, open(fj, 'w'), indent=1)
print(f"[{tag}] done {time.time() - t0:.0f}s", flush=True)
