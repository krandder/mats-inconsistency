"""H2 (meet in the middle at the output): est = w mu_v + (1 - w) ybar_N, mu_v a sample-free kprop estimate (variant v,
ARC-counter cost C_v), ybar_N the Kerdock-MUB sampler mean on the rest of the budget (largest stored checkpoint whose
cost <= B - C_v), w = (V_j / N_j) / (b2_v + V_j / N_j). b2_v (median nMSE of v) and V_j (median Kerdock nMSE x N at each stored
checkpoint j) come from the DEVELOPMENT nets 100-107 only. At each budget the variant (or "no closure",
i.e. plain Kerdock) with the lowest predicted error is used. Version 2 (after the GPT-6 code review): measured
checkpoints, no unused budget beyond checkpoint rounding, plain Kerdock compared at the largest checkpoint with cost <= B
on the same replicates. Per-net ratio Kerdock / H2, median and mean with bootstrap 95% CIs.
usage: combine.py SIGMA L [SAMPDIR]
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import truthlib
from analyze import BUDGETS, boot

DEV = list(range(100, 108)); EVAL = list(range(24))
VARIANTS = ('k1', 'k2', 'k2a', 'k3f', 'k3fa')


def kprop(seed, sigma, L, v):
    p = os.path.join(HERE, 'results', f'kprop_{seed}_s{sigma:g}_L{L}_{v}.json')
    if not os.path.exists(p):
        return None
    d = json.load(open(p)); return np.array(d['means'][L - 1]), float(d['flops']) * KFAC


KFAC = float(os.environ.get('KPROP_COST_FACTOR', '1'))     # sensitivity: charge kprop KFAC x ARC's counter


def kerd(seed, sigma, L, sampdir):
    p = os.path.join(HERE, sampdir, f'samp_{seed}_s{sigma:g}_L{L}.npz')
    tp = os.path.join(HERE, 'truth', f'truth_256_32_s{sigma:g}_seed{seed}.npz')
    if not os.path.exists(p) or not os.path.exists(tp):
        return None
    d = np.load(p); return d['N_mub'], d['costs_mub'], d[f'sum_mub_{L}'], truthlib.load(seed, sigma)


def main(sigma, L, sampdir='samp_v2'):
    dsig = sigma
    has = lambda sg: os.path.exists(os.path.join(HERE, sampdir, f'samp_{DEV[0]}_s{sg:g}_L{L}.npz'))
    if not has(sigma):
        dsig = 0.001                                   # dev error models from the nearest noise level with dev runs
        print(f'(no development-net sampler runs at sigma={sigma}; error models taken from sigma={dsig} dev nets)')
    b2, cost = {}, {}
    for v in VARIANTS:
        e = []
        for s in DEV:
            k = kprop(s, dsig, L, v); tp = os.path.join(HERE, 'truth', f'truth_256_32_s{dsig:g}_seed{s}.npz')
            if k is not None and os.path.exists(tp):
                e.append(truthlib.nmse(k[0], truthlib.load(s, dsig), L)); cost[v] = k[1]
        if len(e) >= 3:
            b2[v] = float(np.median(e))
    VN = []                                            # per dev net: nMSE x N at every checkpoint (same grid for all nets)
    for s in DEV:
        kd = kerd(s, dsig, L, sampdir)
        if kd is None:
            continue
        N, C, S, T = kd
        VN.append([np.mean([truthlib.nmse(S[r, j] / N[j], T, L) for r in range(S.shape[0])]) * N[j] for j in range(len(N))])
    if not b2 or len(VN) < 3:
        sys.exit(f'missing dev data: b2 {list(b2)}, Kerdock dev nets {len(VN)}')
    Vj = np.median(np.array(VN), 0)                    # V by checkpoint (the design's efficiency changes with N); as measured
    print(f"H2, sigma={sigma}, L={L}. Dev error models (nets 100-107): Kerdock nMSE x N = {Vj.min():.3f}-{Vj.max():.3f} by N; " +
          ', '.join(f"{v}: b2 {b2[v]:.2e} at {cost[v]:.2e} FLOPs" for v in b2))
    rows = []
    for s in EVAL:
        kd = kerd(s, sigma, L, sampdir)
        if kd is None:
            continue
        N, C, S, T = kd; R = S.shape[0]
        res = []
        for B in BUDGETS:
            jK = int(np.searchsorted(C, B, side='right') - 1)
            # best sample-free entry affordable at B, chosen by the dev-net medians (no eval information)
            sf = [v for v in b2 if cost[v] <= B]
            vsf = min(sf, key=lambda v: b2[v]) if sf else None
            ksf = kprop(s, sigma, L, vsf) if vsf else None
            sfree = truthlib.nmse(ksf[0], T, L) if ksf is not None else np.nan
            if jK < 0:
                res.append((np.nan, np.nan, '', sfree)); continue
            plain = np.mean([truthlib.nmse(S[r, jK] / N[jK], T, L) for r in range(R)])
            best = (Vj[jK] / N[jK], None, jK)                  # option: no closure
            for v in b2:
                j = int(np.searchsorted(C, B - cost[v], side='right') - 1)
                if j < 0:
                    continue
                pred = b2[v] * (Vj[j] / N[j]) / (b2[v] + Vj[j] / N[j])
                if pred < best[0]:
                    best = (pred, v, j)
            _, v, j = best
            if v is None:
                res.append((plain, plain, 'kerdock', sfree)); continue
            k = kprop(s, sigma, L, v)
            if k is None:
                res.append((plain, np.nan, v, sfree)); continue
            w = (Vj[j] / N[j]) / (b2[v] + Vj[j] / N[j])
            comb = np.mean([truthlib.nmse(w * k[0] + (1 - w) * S[r, j] / N[j], T, L) for r in range(R)])
            res.append((plain, comb, v, sfree))
        rows.append(res)
    print(f"evaluation nets: {len(rows)}")
    print('| budget | variant used | Kerdock median | best kprop median | H2 median | gain over Kerdock, median [95% CI] | '
          'mean [95% CI] | gain over the better of Kerdock and kprop (per net), median [95% CI] |')
    print('|---|---|---|---|---|---|---|---|')
    for bi, B in enumerate(BUDGETS):
        vals = [(r[bi][0], r[bi][1], r[bi][3]) for r in rows if not np.isnan(r[bi][1])]
        if len(vals) < 3:
            continue
        a = np.array(vals); ratio = a[:, 0] / a[:, 1]
        better = np.where(np.isnan(a[:, 2]), a[:, 0], np.minimum(a[:, 0], a[:, 2])) / a[:, 1]
        used = sorted(set(r[bi][2] for r in rows))
        md, mn, mb = boot(ratio, np.median), boot(ratio, np.mean), boot(better, np.median)
        sfm = np.nanmedian(a[:, 2]) if np.any(~np.isnan(a[:, 2])) else float('nan')
        print(f"| {B:.2g} | {','.join(used)} | {np.median(a[:, 0]):.2e} | {sfm:.2e} | {np.median(a[:, 1]):.2e} | "
              f"{md[0]:.3f} [{md[1]:.3f}, {md[2]:.3f}] | {mn[0]:.3f} [{mn[1]:.3f}, {mn[2]:.3f}] | "
              f"{mb[0]:.3f} [{mb[1]:.3f}, {mb[2]:.3f}] |")


if __name__ == '__main__':
    main(float(sys.argv[1]), int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else 'samp_v2')
