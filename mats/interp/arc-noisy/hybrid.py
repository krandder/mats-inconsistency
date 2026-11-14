"""H1 hybrid: Kerdock-MUB carrier + linear control at layer m, scored on the stored Kerdock replicates (paired).
Version 2 (after the GPT-6 code review): measured equal-budget comparison, no rescaling.

est(N) = abar_L(N) - A_m (abar_m(N) - mu_m^c); A_m fitted on a separate 4,096-path plain-MC pilot (gpu_samplers.py);
mu_m^c = a kprop variant's mean at layer m (results/kprop_*.json) or, as an oracle showing the potential (never a
claim), the true layer-m mean. Hybrid cost at checkpoint j = Kerdock cost of j (costs_mub, incl. per-block setup) + pilot
paths + one least-squares fit + the closure's ARC-counter cost to depth m (results/kprop_costs.json). At budget B: plain
Kerdock uses the largest checkpoint with cost <= B, the hybrid the largest with hybrid cost <= B; both measured on the same
replicates. Per-net ratio (Kerdock nMSE / hybrid nMSE), median and mean over nets with bootstrap 95% CIs.
usage: hybrid.py SEEDS SIGMA L [SAMPDIR]
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import truthlib
from analyze import BUDGETS, boot, fmt

COSTS = json.load(open(os.path.join(HERE, 'results', 'kprop_costs.json')))


def closure_mean(seed, sigma, m, v):
    for Lrun in (32, 8):
        p = os.path.join(HERE, 'results', f'kprop_{seed}_s{sigma:g}_L{Lrun}_{v}.json')
        if os.path.exists(p):
            d = json.load(open(p))
            if len(d['means']) >= m:
                return np.array(d['means'][m - 1])
    return None


def run(seeds, sigma, L, sampdir='samp_v2'):
    per = {}                                          # variant -> list over nets of (budgets, 2) [kerdock, hybrid]
    for seed in seeds:
        p = os.path.join(HERE, sampdir, f'samp_{seed}_s{sigma:g}_L{L}.npz')
        tp = os.path.join(HERE, 'truth', f'truth_256_32_s{sigma:g}_seed{seed}.npz')
        if not os.path.exists(p) or not os.path.exists(tp):
            continue
        d = np.load(p); T = truthlib.load(seed, sigma)
        N = d['N_mub']; C = d['costs_mub']; SL = d[f'sum_mub_{L}']; R = SL.shape[0]
        over0 = float(d['pilot_cost']) + float(d['fit_cost'])
        base = np.array([[truthlib.nmse(SL[r, j] / N[j], T, L) for j in range(len(N))] for r in range(R)])
        jK = [int(np.searchsorted(C, B, side='right') - 1) for B in BUDGETS]
        ctrls = [int(k[2:]) for k in d.files if k.startswith('A_')]
        for m in ctrls:
            A = d[f'A_{m}'].astype(np.float64); Sm = d[f'sum_mub_{m}']
            for v in ('oracle', 'k3f', 'k2a', 'k2'):
                if v == 'oracle':
                    mu_c = T['mean'][m - 1]; cc = 0.0
                else:
                    mu_c = closure_mean(seed, sigma, m, v); cc = COSTS.get(v, {}).get(str(m))
                    if mu_c is None or cc is None:
                        continue
                rows = []
                for bi, B in enumerate(BUDGETS):
                    jH = int(np.searchsorted(C + over0 + cc, B, side='right') - 1)
                    if jH < 0 or jK[bi] < 0:
                        rows.append((np.nan, np.nan)); continue
                    h = np.mean([truthlib.nmse(SL[r, jH] / N[jH] - A @ (Sm[r, jH] / N[jH] - mu_c), T, L) for r in range(R)])
                    rows.append((base[:, jK[bi]].mean(), h))
                per.setdefault(f'm{m}-{v}', []).append(rows)
    return per


def report(per, title):
    print(title)
    for k, rows in per.items():
        rows = np.array(rows)                         # (nets, budgets, 2)
        line = []
        for bi, B in enumerate(BUDGETS):
            ok = ~np.isnan(rows[:, bi, 1])
            if ok.sum() < 3:
                line.append(f'{B:.2g}: -'); continue
            r = rows[ok, bi, 0] / rows[ok, bi, 1]
            md, mn = boot(r, np.median), boot(r, np.mean)
            line.append(f'{B:.2g}: {md[0]:.2f} [{md[1]:.2f},{md[2]:.2f}] / {mn[0]:.2f} [{mn[1]:.2f},{mn[2]:.2f}]')
        print(f'  {k:12s} (n={rows.shape[0]}) gain = Kerdock/hybrid, median [CI] / mean [CI]:  ' + ' | '.join(line))


if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')]; sigma = float(sys.argv[2]); L = int(sys.argv[3])
    sampdir = sys.argv[4] if len(sys.argv) > 4 else 'samp_v2'
    report(run(seeds, sigma, L, sampdir), f'H1, L={L}, sigma={sigma}, {sampdir}')
