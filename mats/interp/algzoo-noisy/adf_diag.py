"""Saved diagnostics for the ADF on T16 (review 1, finding 9): cell weights and h-moments at t = 5 vs simulation,
error estimates at several budgets including a gate-pattern-keyed run; ADF on noiseless M_{2,2} and M_{4,3}.
usage: python adf_diag.py OUT.json"""
import sys, json, time
import numpy as np
import az, adf, alg
out = sys.argv[1]; res = {}
W, r, s = az.load_npz('runs/ft_16_10_ex_s0.001.npz')
A = adf.ADF(W, r, s, k_split=2); comps = A.run(upto=5)
res['t5_cell_weights'] = sorted(float(v[0][0]) for v in comps.values())
rng = np.random.default_rng(0); x = rng.standard_normal((200000, 10))
hs, zs, lo = alg.sample_chain(W, r, s, x, rng); st = alg.alg_states(x)
res['t5_cells'] = {}
for key in [(4, 3), (0, 1), (2, 4)]:
    sel = (st['a'][:, 5] == key[0]) & (st['b'][:, 5] == key[1])
    c = [v for k, v in comps.items() if k[:2] == key][0][0]
    res['t5_cells'][str(key)] = dict(mc_mean=hs[sel, 5].mean(0).tolist(), adf_mean=c[1][:16].tolist(),
                                    mc_sd=hs[sel, 5].std(0).tolist(), adf_sd=np.sqrt(np.diag(c[2]))[:16].tolist())
res['T16_error_truth_2^24'] = 0.0406
res['T16_runs'] = []
for k, gk in [(0, ()), (2, ()), (4, ()), (4, (4, 9))]:
    t0 = time.time(); A = adf.ADF(W, r, s, k_split=k, gate_key=gk); c = A.run(); e, _ = A.error_rate(c, 'seq')
    res['T16_runs'].append(dict(k_split=k, gate_key=list(gk), err=e, flops=A.fl.n, components=sum(len(v) for v in c.values()), seconds=time.time() - t0))
for (d, n, sd) in [(2, 2, 2), (4, 3, 0)]:
    Wo = az.load_arc(d, n, seed=sd); m = az.model_from(Wo); acc = az.accuracy(m, 2 ** 22, noise=False)
    A = adf.ADF(Wo, np.zeros((n, d)), 0.0, k_split=4); c = A.run(); e, _ = A.error_rate(c, 'seq')
    res[f'noiseless_M{d}{n}'] = dict(adf_err=e, true_err=1 - acc['acc'], flops=A.fl.n)
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 't5_cells'}, indent=0))
