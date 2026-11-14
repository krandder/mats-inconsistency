# E3 hawk v2: headroom/ceiling probe on the real ShipHunt env (harness/env/shiphunt.py).
# For each preset: #placements, #ordered plans, and over random episodes the exact Bayes P(success)
# of (a) a uniformly random plan, (b) the best of n random plans (gold of BoN under an oracle),
# (c) the best plan overall; plus the share of states whose best plan has P(success) >= 0.95 (ceiling).
# Also: the BoN optimism gap of an unbiased-noise forecaster (logit noise sd 1) vs a recalibrated one.
import sys, json, itertools, numpy as np
sys.path.insert(0, '/home/agent/workspaces/eurisko-study/lir-agent/harness')
from env.shiphunt import PRESETS, ShipHunt, placements
rng = np.random.default_rng(1)
lg = lambda p: np.log(np.clip(p,1e-4,1-1e-4)/(1-np.clip(p,1e-4,1-1e-4))); sg = lambda z: 1/(1+np.exp(-z))
out = {}
for name in ['tiny','easy','medium','hard']:
    cfg = PRESETS[name]; P = placements(cfg); nc = cfg.n_cells; L = cfg.plan_len
    # success depends only on the SET of cells (order matters only for steps)
    sets = list(itertools.combinations(range(nc), L))
    E = 300 if name != 'hard' else 120
    ps_all = []
    S = np.array(sets)
    succ = np.concatenate([(P[:, S[i:i+500]].sum(2) >= cfg.need_hits).astype(np.float32)
                           for i in range(0, len(S), 500)], 1)   # [placements, plan sets], fixed per preset
    for e in range(E):
        env = ShipHunt(cfg, seed=10_000 + e)
        w = env.posterior().astype(np.float32)
        ps_all.append(w @ succ)                   # exact P(success) per plan set
    ps_all = np.array(ps_all)                     # [E, n_sets]
    best = ps_all.max(1)
    row = dict(n_placements=int(len(P)), n_ordered_plans=int(np.prod(range(nc-L+1, nc+1))), n_plan_sets=len(sets),
               episodes=E, mean_p_random=round(float(ps_all.mean()),4), mean_p_best=round(float(best.mean()),4),
               share_best_ge_0p95=round(float((best>=0.95).mean()),4))
    # BoN over random candidate sets (64 per state), oracle gold and gaps for two forecasters
    n_max = 64
    idx = rng.integers(0, len(sets), (E, n_max)); v = np.take_along_axis(ps_all, idx, 1)
    x = lg(v) + rng.standard_normal(v.shape)
    f_noisy = sg(x)
    # recalibrate: isotonic-ish binning fit on a separate draw of the same distribution
    idx2 = rng.integers(0, len(sets), (E, n_max)); v2 = np.take_along_axis(ps_all, idx2, 1)
    x2 = (lg(v2) + rng.standard_normal(v2.shape)).ravel(); v2 = v2.ravel()
    bins = np.quantile(x2, np.linspace(0,1,51)); k2 = np.clip(np.digitize(x2,bins)-1,0,49)
    cal = np.array([v2[k2==k].mean() for k in range(50)])
    f_cal = cal[np.clip(np.digitize(x,bins)-1,0,49)]
    bon = {}
    for n in [1,4,16,64]:
        r = np.arange(E)
        jn = f_noisy[:, :n].argmax(1); jc = f_cal[:, :n].argmax(1); jo = v[:, :n].argmax(1)
        bon[n] = dict(gold_oracle=round(float(v[r,jo].mean()),4), gold_noisy=round(float(v[r,jn].mean()),4),
                      gap_noisy=round(float((f_noisy[r,jn]-v[r,jn]).mean()),4),
                      gap_cal=round(float((f_cal[r,jc]-v[r,jc]).mean()),4),
                      se_gap_noisy=round(float((f_noisy[r,jn]-v[r,jn]).std()/np.sqrt(E)),4))
    row['bon'] = bon
    out[name] = row
    print(name, json.dumps(row))
json.dump(out, open(sys.argv[1], 'w'), indent=1)
