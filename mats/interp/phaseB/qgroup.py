"""E7.2 (PREREG-B amendment 14): does the post-hoc 4-vs-4 grouping hold on fresh nets?
predict NET: per-source DAG test (D1, real weights, K = 64, first 2,000 test episodes) against the net's own K = 64 deviation
             (fresh, seed 1) -> predicted group (A if R^2(mean) > 0.3) appended with a timestamp to results/e7_group_predictions.json.
             Must run before any cross-net correlation of that net exists.
test: mean-deviation correlations (E6 estimator, 8,000 test episodes) of each new net with the four A and four B nets.
usage: python qgroup.py predict RUN_DIR | python qgroup.py test
"""
import json, math, os, sys, time
import numpy as np
import torch
import nnoise as N
import devpred as DP
import dag as DG
import qw
import qshare as SH

torch.set_num_threads(1)
A_NETS = ['L2_s0.05', 'L2seed1_s0.05', 'L2mx0_s0.05', 'L2mx1_s0.05']; B_NETS = ['L2seed2_s0.05', 'L2ws0_s0.05', 'L2ws3_s0.05', 'L2mx4_s0.05']
PRED = 'results/e7_group_predictions.json'


def predict(run_dir):
    nm = os.path.basename(run_dir.rstrip('/'))
    assert not os.path.exists(os.path.join(run_dir, 'share_test.npz')), 'correlation data already exists'
    model, meta = N.load_run(run_dir); model.requires_grad_(False); sg = meta['sigma']
    X, v, Tt = SH.test_set(); n = 2000; T2 = {k: (a[:n] if k != 'comp_vars' else a) for k, a in Tt.items()}
    h, _, _ = DP.marginal(model, X[:n], v[:n], sg, {}, 64, 1); dev = qw.deviations_t(h, T2)
    h, _, _ = DP.marginal(model, X[:n], v[:n], sg, DG.make_sur(model, DG.masks('D1', meta['depth'])), 64, 11); dd = qw.deviations_t(h, T2)
    r2 = {k: DP.r2(dev[k], dd[k]) for k in SH.FEATS}
    gap = meta['best_val'] - meta['val_bayes']
    rec = dict(net=nm, time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), dag_R2=r2, val_gap=gap,
               excluded=bool(gap > 0.03), predicted_group=('A' if r2['mean'] > 0.3 else 'B'))
    allp = json.load(open(PRED)) if os.path.exists(PRED) else []
    allp.append(rec); json.dump(allp, open(PRED, 'w'), indent=1)
    print(rec, flush=True)


def test():
    preds = json.load(open(PRED)); out = []
    def load(nm):
        z = np.load(os.path.join('runs', nm, 'share_test.npz')); return z['r1_mean'], z['r2_mean']
    def cov(a, b): return float(np.mean((a - a.mean()) * (b - b.mean())))
    def rho(a, b):
        a1, a2 = load(a); b1, b2 = load(b)
        return 0.5 * (cov(a1, b2) + cov(a2, b1)) / math.sqrt(cov(a1, a2) * cov(b1, b2))
    for p in preds:
        if p['excluded']: out.append(dict(p, result='excluded (validation gap > 0.03)')); continue
        ra = [rho(p['net'], o) for o in A_NETS]; rb = [rho(p['net'], o) for o in B_NETS]
        mA, mB = float(np.mean(ra)), float(np.mean(rb))
        ok = (mA >= 0.6 and mA - mB >= 0.1) if p['predicted_group'] == 'A' else (mA <= 0.55)
        out.append(dict(p, corr_with_A=ra, corr_with_B=rb, mean_A=mA, mean_B=mB, confirmed=bool(ok)))
        print(p['net'], p['predicted_group'], round(mA, 3), round(mB, 3), 'confirmed' if ok else 'NOT confirmed', flush=True)
    news = [o['net'] for o in out if 'mean_A' in o]
    pairs = [(a, b) for i, a in enumerate(news) for b in news[i + 1:]]
    json.dump(dict(rows=out, new_pairs={f'{a}~{b}': rho(a, b) for a, b in pairs}), open('results/e7_group_test.json', 'w'), indent=1)


if __name__ == '__main__':
    predict(sys.argv[2]) if sys.argv[1] == 'predict' else test()
