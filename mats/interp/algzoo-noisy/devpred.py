"""Deviation prediction and scenario tests (PREREG Part 2: V1, S1-S8).
usage: python devpred.py MODEL.npz DAG.json TAU OUT.json
The DAG surrogate (greedy row with the given tau) predicts per-input error probabilities with its own noise draws.
"""
import sys, json, math, time
import numpy as np
import az, dag, alg

t0 = time.time()
path, dpath, tau, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
W, r, sigma = az.load_npz(path)
net = dag.Net(W, r, sigma); d, n = net.d, net.n
D = json.load(open(dpath))
row = [g for g in D['greedy'] if abs(g['tau'] - tau) < 1e-12][0]
keep = dict(rec=np.zeros((n, d, d), bool), inp=np.zeros((n, d), bool), out=np.zeros(d, bool))
for (k, t, i, j) in row['edges']:
    if k == 'rec': keep['rec'][t, i, j] = True
    elif k == 'inp': keep['inp'][t, i] = True
    else: keep['out'][i] = True
rng = np.random.default_rng(777001)
xm = rng.standard_normal((2 ** 16, n)); em = rng.standard_normal((2 ** 16, n, d))
_, hsm, _ = net.run(xm, em); mu = [h.mean(0) for h in hsm]


def eps_of(x, K, surrogate, seed):
    g = np.random.default_rng(seed); B = len(x); y = np.argsort(x, -1)[:, -2]; e = np.zeros(B)
    for i in range(0, B, 2048):
        xc = np.repeat(x[i:i + 2048], K, 0); ec = g.standard_normal((len(xc), n, d))
        lo, _, _ = net.run(xc, ec, keep=keep if surrogate else None, mu=mu)
        e[i:i + 2048] = (lo.argmax(1) != np.repeat(y[i:i + 2048], K)).reshape(-1, K).mean(1)
    return e


def r2(a, b): return float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())


K = 32
xv = rng.standard_normal((2 ** 15, n))
e1 = eps_of(xv, K, False, 1); e2 = eps_of(xv, K, False, 2); es = eps_of(xv, K, True, 3)
res = dict(model=path, sigma=sigma, tau=tau, K=K, n_inputs=len(xv), dag_edges=row['K'],
           R2_dag=r2(e1, es), R2_noise_floor=r2(e1, e2), p_net=float(e1.mean()), p_dag=float(es.mean()),
           frac_noise_sensitive=float(((e1 > 0.05) & (e1 < 0.95)).mean()))
print({k: v for k, v in res.items()}, flush=True)

# scenarios on a large fresh set, one noise draw per input (network and DAG with independent noise)
N = 2 ** 20
xs = rng.standard_normal((N, n)); y = np.argsort(xs, -1)[:, -2]
err_n = np.zeros(N, bool); err_s = np.zeros(N, bool)
for i in range(0, N, 8192):
    lo, _, _ = net.run(xs[i:i + 8192], rng.standard_normal((min(8192, N - i), n, d)))
    err_n[i:i + 8192] = lo.argmax(1) != y[i:i + 8192]
    lo, _, _ = net.run(xs[i:i + 8192], rng.standard_normal((min(8192, N - i), n, d)), keep=keep, mu=mu)
    err_s[i:i + 8192] = lo.argmax(1) != y[i:i + 8192]
srt = np.sort(xs, -1); Mx, Sx, Tx = srt[:, -1], srt[:, -2], srt[:, -3]
a = np.argmax(xs, -1); b = y
p = err_n.mean()


def rate(mask, e=err_n):
    k = int(mask.sum()); q = float(e[mask].mean()) if k else float('nan')
    return dict(n=k, rate=q, ci=[q - 1.96 * math.sqrt(q * (1 - q) / max(k, 1)), q + 1.96 * math.sqrt(q * (1 - q) / max(k, 1))])


npos = (xs > 0).sum(1)
sc = {}
sc['S1'] = dict(net=rate(npos == 1), dag=rate(npos == 1, err_s), target='>= 5p')
sc['S1']['hit'] = bool(sc['S1']['net']['ci'][0] >= 5 * p)
sc['S2'] = dict(net=rate(npos == 0), dag=rate(npos == 0, err_s), target='>= 0.5')
sc['S2']['hit'] = bool(sc['S2']['net']['ci'][0] >= 0.5)
g1 = Mx - Sx; m3 = g1 <= np.quantile(g1, 0.1)
sc['S3'] = dict(net=rate(m3), dag=rate(m3, err_s), target='>= 2p'); sc['S3']['hit'] = bool(sc['S3']['net']['ci'][0] >= 2 * p)
g2 = Sx - Tx; m4 = g2 <= np.quantile(g2, 0.1)
sc['S4'] = dict(net=rate(m4), dag=rate(m4, err_s), target='>= 2p'); sc['S4']['hit'] = bool(sc['S4']['net']['ci'][0] >= 2 * p)
ra, rb = rate(a > b), rate(a < b)
sc['S5'] = dict(net_a_gt_b=ra, net_a_lt_b=rb, dag_a_gt_b=rate(a > b, err_s), dag_a_lt_b=rate(a < b, err_s), target='rate(a>b) >= 1.3 rate(a<b)')
sc['S5']['hit'] = bool(ra['ci'][0] >= 1.3 * rb['ci'][1])
m6 = (a == n - 1) | (b == n - 1)
sc['S6'] = dict(net=rate(m6), dag=rate(m6, err_s), target='<= 0.5p'); sc['S6']['hit'] = bool(sc['S6']['net']['ci'][1] <= 0.5 * p)
# S7 on the eps set
srt_v = np.sort(xv, -1); g1v = srt_v[:, -1] - srt_v[:, -2]; g2v = srt_v[:, -2] - srt_v[:, -3]
ns = (e1 > 0.05) & (e1 < 0.95)
near = (g1v <= np.quantile(g1v, 0.1)) | (g2v <= np.quantile(g2v, 0.1))
sc['S7'] = dict(share_overall=float(ns.mean()), share_near=float(ns[near].mean()), n_near=int(near.sum()), target='share_near >= 3 share_overall')
sc['S7']['hit'] = bool(ns[near].mean() >= 3 * ns.mean())
hits8 = []
for k in ['S1', 'S2', 'S3', 'S4', 'S6']:
    hits8.append(abs(sc[k]['dag']['rate'] / max(sc[k]['net']['rate'], 1e-12) - 1) <= 0.25)
hits8.append(abs(sc['S5']['dag_a_gt_b']['rate'] / sc['S5']['net_a_gt_b']['rate'] - 1) <= 0.25 and abs(sc['S5']['dag_a_lt_b']['rate'] / sc['S5']['net_a_lt_b']['rate'] - 1) <= 0.25)
sc['S8'] = dict(per_scenario_within_25pct=hits8, hit=bool(all(hits8)))
res['p_net_scen'] = float(p); res['p_dag_scen'] = float(err_s.mean()); res['scenarios'] = sc
# position table (a, b) for the record (computed after the predictions above were written)
tab = np.zeros((n, n)); tabs = np.zeros((n, n)); cnt = np.zeros((n, n))
np.add.at(tab, (a, b), err_n); np.add.at(tabs, (a, b), err_s); np.add.at(cnt, (a, b), 1)
res['by_ab_net'] = (tab / np.maximum(cnt, 1)).tolist(); res['by_ab_dag'] = (tabs / np.maximum(cnt, 1)).tolist()
res['seconds'] = time.time() - t0
json.dump(res, open(out, 'w'), indent=1)
for k, v in sc.items(): print(k, json.dumps(v)[:400], flush=True)
print('done', res['seconds'])
