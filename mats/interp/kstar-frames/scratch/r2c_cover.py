"""Run B covering statistics, recomputed with float64 pair scores (lib/cover.py v2; the v1 statistics in
runs/Bres_*.json used a float32 Gram matrix, which the GPT-6 review showed can give CS > 1). Same statistics as
r2b_analyze.py's covering block, at c = L/2: ball masses p(x) (share of the references whose frame, projected on x's
active units, captures >= 1 - delta of x's top-k sensitivity), random-centre R99 (Poisson form, and the exact
fixed-R form 1 - mean (1 - p)^R), greedy centres chosen on half of B and scored on the other half, union of all
references. Also the raw (unprojected) CS of the reference frames as a check. Writes runs/Bcover2_<tag>.json and
runs/Bcov2_<tag>_c<c>.npy.
nice -n 19 ionice -c3 python r2c_cover.py --n 64 --seed 400 --L 32"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import kf, cover

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=64); ap.add_argument('--seed', type=int, default=400)
ap.add_argument('--L', type=int, default=32); ap.add_argument('--threads', type=int, default=3)
ap.add_argument('--runs', default=os.path.join(HERE, '..', 'runs'))
A = ap.parse_args()
torch.set_num_threads(A.threads)
n, L, seed = A.n, A.L, A.seed
tag = f"n{n}_s{seed}_L{L}"
c = L // 2
DELTAS = (0.3, 0.2, 0.1, 0.05)
t0 = time.time()
d = np.load(os.path.join(A.runs, f'B_{tag}_c{c}.npz'))
k = int(d['k'])
dev = kf.DEV
VB, sB = torch.from_numpy(d['VB']).double().to(dev), torch.from_numpy(d['sB']).double().to(dev)
DB = (torch.from_numpy(d['zB']) > 0).double().to(dev)
VA = torch.from_numpy(d['VA'][:, :, :k]).double().to(dev)
NB, NA = VB.shape[0], VA.shape[0]
CSm, Om = cover.pair_scores(VB, sB, DB, VA, k)
# raw CS (unprojected reference frames), float64, in blocks
CSraw = torch.zeros(NB, NA, dtype=torch.float32)
s2 = sB ** 2
top = s2[:, :k].sum(-1)
for i0 in range(0, NB, 32):
    Vt = (VB[i0:i0 + 32] * sB[i0:i0 + 32, None, :]).transpose(1, 2)      # (a, KK, n) = S V^T
    for j0 in range(0, NA, 512):
        P = torch.einsum("akn,bnj->abkj", Vt, VA[j0:j0 + 512])           # (a, b, KK, k)
        CSraw[i0:i0 + 32, j0:j0 + 512] = ((P ** 2).sum((-1, -2)) / top[i0:i0 + 32, None]).float().cpu()
res = {'n': n, 'seed': seed, 'L': L, 'c': c, 'k': k, 'NA': NA, 'NB': NB, 'max_CS': float(CSm.max()),
       'n_CS_above_1': int((CSm > 1 + 1e-6).sum()), 'max_CS_raw': float(CSraw.max()), 'cover': {}}
loss = 1 - CSm
half = NB // 2
for dl in DELTAS:
    covm = loss <= dl
    p = covm.double().mean(1).numpy()
    p_raw = ((1 - CSraw) <= dl).double().mean(1).numpy()
    ch, curve = cover.greedy_cover(covm[:half])
    sel = torch.tensor(ch, dtype=torch.long)
    run_cov = torch.zeros(NB - half, dtype=torch.bool)
    after = []
    for j in range(len(ch)):
        run_cov |= covm[half:, sel[j]]
        after.append(float(run_cov.double().mean()))
    reached = {}
    for fr in (0.5, 0.9, 0.99):
        js = [j + 1 for j, v in enumerate(after) if v >= fr]
        reached[str(fr)] = js[0] if js else None
    # exact fixed-R random-centre coverage: R with 1 - mean (1 - p)^R >= 0.99
    def r99_exact(pp):
        if np.mean(pp <= 0) >= 0.01:
            return float('inf')
        lo, hi = 0.0, 1.0
        f = lambda R: np.mean(np.exp(R * np.log1p(-np.minimum(pp, 1 - 1e-15))))
        while f(hi) > 0.01:
            hi *= 2
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) > 0.01 else (lo, mid)
        return hi
    res['cover'][str(dl)] = {
        'p_mean': float(p.mean()), 'p_median': float(np.median(p)), 'p_q10': float(np.quantile(p, 0.1)),
        'frac_p0': float((p == 0).mean()), 'R99_poisson': cover.r99(p), 'R99_exact': r99_exact(p),
        'greedy_centres_B1': len(ch), 'greedy_B1_final': curve[-1] if curve else 0.0,
        'greedy_B2_final': after[-1] if after else 0.0, 'greedy_centres_for_B2': reached,
        'union_all_A': float(covm.any(1).double().mean()),
        'raw': {'p_mean': float(p_raw.mean()), 'p_median': float(np.median(p_raw)), 'frac_p0': float((p_raw == 0).mean()),
                'union_all_A': float(((1 - CSraw) <= dl).any(1).double().mean())}}
    v = res['cover'][str(dl)]
    print(f"[{tag}] c={c} delta={dl}: p mean {v['p_mean']:.3e} median {v['p_median']:.2e} p=0 {v['frac_p0']:.3f} "
          f"R99 {v['R99_poisson']:.3g}/{v['R99_exact']:.3g} | greedy {len(ch)} -> B1 {v['greedy_B1_final']:.3f}, "
          f"B2 {v['greedy_B2_final']:.3f} | all A covers {v['union_all_A']:.3f} | raw: p mean {v['raw']['p_mean']:.3e}, "
          f"p=0 {v['raw']['frac_p0']:.3f}", flush=True)
np.save(os.path.join(A.runs, f'Bcov2_{tag}_c{c}.npy'), CSm.numpy().astype(np.float16))
json.dump(res, open(os.path.join(A.runs, f'Bcover2_{tag}.json'), 'w'), indent=1)
print(f"[{tag}] max CS {res['max_CS']:.6f} ({res['n_CS_above_1']} above 1), max raw CS {res['max_CS_raw']:.6f}; "
      f"done {time.time() - t0:.0f}s", flush=True)
