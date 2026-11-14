"""Kill tests (PREREG-TAILS section 5) on development nets 102-105, sigma = 0.001.
python tails/killtests.py RUNDIR KPDIR"""
import glob, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deduce as D
from common import c_path, c_move, run_evals, load_truth, gs_b50, logmeanexp

RD, KP = sys.argv[1], sys.argv[2]
SEEDS = (102, 103, 104, 105)
TR = {s: load_truth(os.path.join(RD, 'truth', f'truth_{s}_s0.001.json')) for s in SEEDS
      if os.path.exists(os.path.join(RD, 'truth', f'truth_{s}_s0.001.json'))}
print('truth for', sorted(TR))


def ki(k):
    return (4, 6, 8, 10, 12).index(k)


# ---- U2 vs C-K3f (L = 4, 8; k = 8)
print('\n== U2 (saddlepoint on the K3f description) vs C-K3f (heaviside), fraction of events within 50%')
for L in (4, 8, 32):
    rows = {'C-K3f': [], 'U2 sp4': [], 'U2r rad4': []}
    for s in TR:
        f = os.path.join(KP, f'kp_{s}_L{L}_s0.001_k3f.json')
        if not os.path.exists(f):
            continue
        kp = json.load(open(f)); T = TR[s][L]
        for b in range(8):
            for k in range(5):
                p = math.exp(T['lp'][b, k]); t = T['thr'][b, k]
                mm = np.array(kp['moments'])[:, b]; kk = D.cumulants_from_moments(mm)
                rows['C-K3f'].append((k, abs(kp['heavi'][b][k] / p - 1)))
                v = D.saddle(t, kk, 4)
                if not np.isfinite(v):
                    v = D.saddle(t, kk, 3)
                if not np.isfinite(v):
                    v = D.gauss(t, kk)
                rows['U2 sp4'].append((k, abs(v / p - 1)))
                v = D.radial(t, mm, order=4)
                rows['U2r rad4'].append((k, abs(v / p - 1) if np.isfinite(v) else np.inf))
    for name, r in rows.items():
        if not r:
            continue
        a = np.array(r)
        print(f'  L={L:2d} {name:>9}: within50 ' + ' '.join('%.2f' % np.mean(a[a[:, 0] == k, 1] <= 0.5) for k in range(5))
              + ' | median rel err ' + ' '.join('%.2f' % np.median(a[a[:, 0] == k, 1]) for k in range(5)) + f'  (n={int(np.sum(a[:, 0] == 0))})')

# ---- U3: Monte Carlo moments (2^22) + saddlepoint, L = 4, k = 8
print('\n== U3 (MC moments 2^22 + saddlepoint)')
for L in (4, 8, 32):
    r3 = []
    for s in TR:
        f = os.path.join(RD, 'x3k22', f'x3_{s}_s0.001.json')
        if not os.path.exists(f):
            continue
        M = np.array(json.load(open(f))['moments'][str(L)]); T = TR[s][L]
        for b in range(8):
            for k in range(5):
                p = math.exp(T['lp'][b, k]); t = T['thr'][b, k]; kk = D.cumulants_from_moments(M[:, b])
                v = D.saddle(t, kk, 4)
                if not np.isfinite(v):
                    v = D.gauss(t, kk)
                r3.append((k, abs(v / p - 1)))
    if r3:
        a = np.array(r3)
        print(f'  L={L:2d}: within50 ' + ' '.join('%.2f' % np.mean(a[a[:, 0] == k, 1] <= 0.5) for k in range(5)))

# ---- U1: mode splitting at L = 32, k = 8, amortised B50 vs C-GS per-event B50
print('\n== U1 (mode splitting), L = 32')
for L in (32,):
    wins = {k: [] for k in range(5)}; rm = {k: [] for k in range(5)}; wins_s = {k: [] for k in range(5)}; wins_m = {k: [] for k in range(5)}
    for s in TR:
        f = os.path.join(RD, 'mode', f'mode_{s}_L{L}_s0.001.json')
        if not os.path.exists(f):
            continue
        mo = json.load(open(f)); T = TR[s][L]; cm = c_move(L)
        setup = mo['M0'] * c_path(L) + 2 * mo['M0'] * 256 * 256 + 10 * 256 ** 3      # pilot paths, Gram matrix, eigendecomposition
        runs = {}
        for name in ('up', 'down'):
            ru = mo['runs'][name]
            K = len(ru['levels'])
            per_run = (run_evals(2000, K + 1) * cm + (K + 1) * 2000 * ((L - 1) * (256 * (2 * 256 - 1) + 3 * 256))
                       + (K + 1) * 2000 * (256 * (2 * 256 - 1) + 256 * 5))   # moves + readout (recomputes h, as implemented) + all outputs
            runs[name] = (np.array(ru['est']), per_run, ru['pilot_n_eval'] * cm)
        both = sum(r[1] for r in runs.values())                   # both directions are run (same cost model as analyze_tails)
        tot_fixed = setup + sum(r[2] for r in runs.values())
        for b in range(8):
            name = 'up' if mo['c'][b] > 0 else 'down'
            E, per_run, _ = runs[name]
            for k in range(5):
                p = math.exp(T['lp'][b, k])
                rmse = float(np.mean((E[:, b, k] / p - 1) ** 2))           # normalised residuals: all-zero gives exactly 1
                rm[k].append(rmse)
                b50_am = (tot_fixed + both * rmse / 0.25) / (5 * 256) if rmse <= 1 else np.inf
                b50_strict = (tot_fixed + both * rmse / 0.25) / (5 * 256) if rmse < 1 else np.inf
                b50_m = (tot_fixed + both * rmse / 0.25) / (5 * 256) if rmse <= 0.81 else np.inf
                wins_m[k].append(b50_m < gs_b50(L, T["nlev"][b, k], T["rv1"][b, k]))
                wins_s[k].append(b50_strict < gs_b50(L, T["nlev"][b, k], T["rv1"][b, k]))
                b50_gs = gs_b50(L, T['nlev'][b, k], T['rv1'][b, k])
                wins[k].append(b50_am < b50_gs)
    for k in range(5):
        if wins[k]:
            print(f'  k={(4, 6, 8, 10, 12)[k]:2d}: frac events amortised U1 B50 < C-GS B50: {np.mean(wins[k]):.2f} (rMSE <= 1), {np.mean(wins_s[k]):.2f} (rMSE < 1, amendment 2-3), {np.mean(wins_m[k]):.2f} (rel RMSE <= 0.9); median rMSE {np.median(rm[k]):.2f}; frac rMSE<=1 {np.mean(np.array(rm[k]) <= 1):.2f}')

# ---- U4: Gaussian importance sampling, L = 4, k = 8
print('\n== U4 (Gaussian IS fitted to splitting particles), L = 4, k = 8')
rv = {}
for f in sorted(glob.glob(os.path.join(RD, 'x4', 'x4_*.json'))):
    for e in json.load(open(f))['events']:
        for kk, o in e['by_k'].items():
            rv.setdefault(kk, []).append(o['relvar_vs_ref'])
for kk, v in rv.items():
    print(f'  top-{kk} directions: median relative variance per sample {np.median(v):.3g} (n={len(v)})')

# ---- U5: twisted SMC leverage r
print('\n== U5 (layer-wise twisted SMC): r = P(A and A\') / P(A) at k = 8')
for f in sorted(glob.glob(os.path.join(RD, 'x2k', 'x2_*.json'))):
    r = json.load(open(f))
    for sg, o in r['by_sigma'].items():
        S = np.array(o['single']); C = np.array(o['coupled'])
        ratio = np.exp(np.array([logmeanexp(c) for c in C]) - np.array([logmeanexp(x) for x in S]))
        print(f"  {os.path.basename(f)} sigma {sg}: median r {np.median(ratio):.3f} [min {ratio.min():.3f}]")
