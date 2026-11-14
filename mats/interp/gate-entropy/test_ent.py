"""Estimator checks on synthetic patterns with known entropy (gate-entropy)."""
import sys, time, math, json
import numpy as np, torch
torch.set_num_threads(1)
import ent
rng = np.random.default_rng(0)
res = {}
# 1. iid Bernoulli bits: exact H = m * h2(p)
for m, p in ((12, 0.5), (24, 0.3), (64, 0.5)):
    Hx = m * float(ent.h2(p))
    for N in (1 << 12, 1 << 16):
        G = rng.random((N, m)) < p
        t = time.time(); e = ent.all_estimates(G, made=True, seed=1)
        res[f'iid_m{m}_p{p}_N{N}'] = dict(exact=Hx, plugin=e['plugin'], mm=e['mm'], nsb=e['nsb'].get('H'), nsb_sd=e['nsb'].get('sd_beta'),
                                         H2=e['H2'], H2zero=e['H2_zero'], indep=e['indep_upper'], fvsbn=e['fvsbn']['H_up'], made=e['made']['H_up'], s=time.time()-t)
        print(f'iid_m{m}_p{p}_N{N}', json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[f"iid_m{m}_p{p}_N{N}"].items()}), flush=True)
# 2. sign patterns of correlated Gaussian (orthant), m = 16; reference from 2^22 samples
m = 16
A = rng.standard_normal((m, 4)) ; C = A @ A.T + 0.3 * np.eye(m)
Lc = np.linalg.cholesky(C)
def draw(N): return (rng.standard_normal((N, m)) @ Lc.T + 0.2 * rng.standard_normal(m)) > 0
Gref = draw(1 << 22); cref, _ = ent.pattern_ids(Gref); Href = ent.plugin(cref); Hmm = ent.miller_madow(cref)
print('orthant ref plugin', Href, 'mm', Hmm, 'Kobs', len(cref))
for N in (1 << 10, 1 << 12, 1 << 14, 1 << 16):
    G = draw(N); e = ent.all_estimates(G, made=True, seed=1)
    print(f'orthant N={N}', dict(ref=round(Hmm, 3), plugin=round(e['plugin'], 3), mm=round(e['mm'], 3), nsb=round(e['nsb']['H'], 3), nsb_sd=round(e['nsb']['sd_beta'], 3), H2=round(e['H2'], 3), indep=round(e['indep_upper'], 3), fvsbn=round(e['fvsbn']['H_up'], 3), made=round(e['made']['H_up'], 3)), flush=True)
