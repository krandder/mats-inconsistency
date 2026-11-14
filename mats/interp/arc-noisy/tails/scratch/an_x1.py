"""Analyse X1 (+ X3 oracle cumulants): per-neuron GS efficiency, amortised mode splitting, cumulant readouts.
python an_x1.py RUNDIR"""
import glob, json, math, os, sys
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/tails')
import deduce as D

RD = sys.argv[1]
TGT = ['1e-4', '1e-6', '1e-8', '1e-10', '1e-12']
for f in sorted(glob.glob(os.path.join(RD, 'x1', 'x1_*.json'))):
    r = json.load(open(f)); NN = len(r['neurons']); N = r['frozen_N']; R = r['R']
    print(f"=== {os.path.basename(f)}  mode share {r['mode_share']:.3f}  R2(z,S) median {np.median(r['r2']):.2f}  frac c>0 {np.mean(np.array(r['c']) > 0):.2f}")
    ref = np.zeros((NN, 5)); wn_gs = np.zeros((NN, 5)); rv_gs = np.zeros((NN, 5))
    for b, pn in enumerate(r['per_neuron']):
        for k in range(5):
            v = np.array(pn['logp'][k]); m = v.max(); w = np.exp(v - m)
            ref[b, k] = math.log(w.mean()) + m
            rv = w.var(ddof=1) / w.mean() ** 2
            ev = N + N * 5 * (pn['nlev'][k] - 1)
            rv_gs[b, k] = rv; wn_gs[b, k] = rv * ev
    print('  per-neuron GS: median rv1 (N=1000) by level', np.round(np.median(rv_gs, 0), 2), ' median WNRV (evals)', ['%.1e' % v for v in np.median(wn_gs, 0)])
    print('  reference log10 p: median by level', np.round(np.median(ref, 0) / math.log(10), 2))
    for sg in ('+1', '-1'):
        mo = r.get(f'mode{sg}')
        if mo is None:
            continue
        E = np.array(mo['est'])                                      # (Rm, NN, 5)
        evr = mo['n_eval_per_run']
        P = np.exp(ref)[None]
        rmse = ((E - P) ** 2).mean(0) / P[0] ** 2                    # relative MSE of one run
        wn = rmse * evr / 256
        gain = wn_gs / wn
        sel = (np.array(r['c']) > 0) if sg == '+1' else (np.array(r['c']) < 0)
        print(f'  mode {sg}: {len(mo["levels"])} levels, evals/run {evr:.0f}; neurons with c {"> 0" if sg == "+1" else "< 0"}: {sel.sum()}')
        for k in range(5):
            g = gain[sel, k]
            print(f'     {TGT[k]:>6}: rMSE median {np.median(rmse[sel, k]):.2f}; gain over per-neuron GS (WNRV, amortised /256): median {np.median(g):.2g}, '
                  f'frac >1 {np.mean(g > 1):.2f}, >10 {np.mean(g > 10):.2f}; frac rMSE<=0.25 {np.mean(rmse[sel, k] <= 0.25):.2f}')
    # oracle cumulant readouts
    x3f = os.path.join(RD, 'x3', f"x3_{r['seed']}_s{r['sigma']:g}.json")
    if os.path.exists(x3f):
        mom = np.array(json.load(open(x3f))['moments'][str(r['L'])])   # (4, 256)
        rows = {m: np.zeros((NN, 5)) for m in ('gauss', 'edge4', 'sp3', 'sp4', 'rad4')}
        for b in range(NN):
            mm = mom[:, b]; kk = D.cumulants_from_moments(mm)
            for k in range(5):
                t = r['thr'][b][k]
                vals = dict(gauss=D.gauss(t, kk), edge4=D.edgeworth(t, kk), sp3=D.saddle(t, kk, 3), sp4=D.saddle(t, kk, 4),
                            rad4=D.radial(t, mm, order=4))
                for m, v in vals.items():
                    rows[m][b, k] = (v / math.exp(ref[b, k]) - 1) if (v is not None and np.isfinite(v) and v > 0) else np.nan
        print('  oracle cumulants (MC 2^26): median |rel err| by level; frac within 50%; frac nan')
        for m, a in rows.items():
            med = np.nanmedian(np.abs(a), 0)
            w50 = np.mean(np.abs(np.nan_to_num(a, nan=9e9)) <= 0.5, 0)
            print(f'     {m:>6}: ' + ' '.join('%.2f' % v for v in med) + '  | within50 ' + ' '.join('%.2f' % v for v in w50) + '  | nan ' + ' '.join('%.2f' % v for v in np.mean(np.isnan(a), 0)))

# kprop-description moments (scratch/km_dev.json) with the same readouts
KM = json.load(open('/home/agent/workspaces/mats/work/arc-noisy/tails/scratch/km_dev.json'))
print('\n##### readouts from kprop description moments (K3f) vs per-neuron GS references')
for f in sorted(glob.glob(os.path.join(RD, 'x1', 'x1_*.json'))):
    r = json.load(open(f)); NN = len(r['neurons'])
    key = f"{r['seed']}_{r['L']}_k3f"
    if key not in KM:
        continue
    M = np.array(KM[key]['M'])
    ref = np.zeros((NN, 5))
    for b, pn in enumerate(r['per_neuron']):
        for k in range(5):
            v = np.array(pn['logp'][k]); m = v.max(); ref[b, k] = math.log(np.exp(v - m).mean()) + m
    rows = {m: np.zeros((NN, 5)) for m in ('gauss', 'edge4', 'sp3', 'sp4', 'rad4')}
    for b in range(NN):
        mm = M[:, b]; kk = D.cumulants_from_moments(mm)
        for k in range(5):
            t = r['thr'][b][k]
            vals = dict(gauss=D.gauss(t, kk), edge4=D.edgeworth(t, kk), sp3=D.saddle(t, kk, 3), sp4=D.saddle(t, kk, 4),
                        rad4=D.radial(t, mm, order=4))
            for m, v in vals.items():
                rows[m][b, k] = (v / math.exp(ref[b, k]) - 1) if (v is not None and np.isfinite(v) and v > 0) else np.nan
    print(f"=== {os.path.basename(f)} (K3f flops {KM[key]['flops']:.2e})")
    for m, a in rows.items():
        med = np.nanmedian(np.abs(a), 0)
        w50 = np.mean(np.abs(np.nan_to_num(a, nan=9e9)) <= 0.5, 0)
        lg = np.nanmedian(np.log10(1 + a), 0)
        print(f'     {m:>6}: med|relerr| ' + ' '.join('%.2f' % v for v in med) + ' | within50 ' + ' '.join('%.2f' % v for v in w50)
              + ' | med log10 ratio ' + ' '.join('%+.2f' % v for v in lg) + ' | nan ' + ' '.join('%.2f' % v for v in np.mean(np.isnan(a), 0)))
