"""ARC's bar in three columns (PREREG Amendment 5): (a) deductive only, (b) deductive plus a little sampling, (c) plain
sampling, MSE against FLOPs over a family of weight-perturbed copies of one trained model.

usage: python bar3.py MODEL.npz ESTIM.json OUT.json [LDAG.json]
The family and its truths are those of ESTIM.json (estim.py, same perturbation seeds). Advice found once on the base
model and never on the variants: the dp-targeted surrogate's edge set and its mean biases mu_t (LDAG.json + the base
model), used in (a). Per variant, only the estimators run.
(a) a1: ADF on the full network (k = 0, 2);  a2: ADF on the surrogate (base mu), k = 0.
(b) b1: ADF per-cell predictions -> Neyman allocation over the n(n-1) algorithmic cells, then stratified sampling;
    b2: control variate Y = ADF per-cell prediction of the input's cell (mean known deductively), beta fitted;
    b3: two-level estimator with the surrogate: N samples of the surrogate + N/4 coupled (network - surrogate).
    The ADF cost (and, for b3, nothing else) is added to the sampling cost.
(c) Monte Carlo (from ESTIM.json; same per-sequence FLOP count).
"""
import sys, json, math, time
import numpy as np
import az, adf, dag

t0 = time.time()
path, epath, out = sys.argv[1], sys.argv[2], sys.argv[3]
lpath = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != '-' else None
SUR_ONLY = 'surrogate_only' in sys.argv
W0, r0, sigma = az.load_npz(path)
d = W0[0].shape[0]; n = W0[2].shape[0]
E = json.load(open(epath)); P = np.array(E['truths']); NM = len(P); DELTA = E['DELTA']; FL = E['flops_per_sample']
cells = [(a, b) for a in range(n) for b in range(n) if a != b]
rng0 = np.random.default_rng(31)
keep = None; mu_base = None
if lpath:
    L = json.load(open(lpath))
    row = [x for x in L['runs'] if x['target'] == 'dp' and x.get('edges') and x['Nref_log2'] == 14][0]
    keep = dict(rec=np.zeros((n, d, d), bool), inp=np.zeros((n, d), bool), out=np.zeros(d, bool))
    for (k, t, i, j) in row['edges']:
        if k == 'rec': keep['rec'][t, i, j] = True
        elif k == 'inp': keep['inp'][t, i] = True
        else: keep['out'][i] = True
    netb = dag.Net(W0, r0, sigma)
    _, hs, _ = netb.run(rng0.standard_normal((2 ** 16, n)), rng0.standard_normal((2 ** 16, n, d))); mu_base = [h.mean(0) for h in hs]
    FL_SUR = 2 * (keep['rec'].sum() + keep['inp'].sum() + keep['out'].sum()) + n * 3 * d + 2 * n * d + n + n * math.log2(n)


def perturbed(k):
    if k == 0: return [w.copy() for w in W0]
    g = np.random.default_rng(1000 + k)
    return [w + DELTA * np.sqrt((w ** 2).mean()) * g.standard_normal(w.shape) for w in W0]


def cell_sample(a, b, m, rng):
    v = np.sort(rng.standard_normal((m, n)), 1)
    X = np.empty((m, n)); others = [j for j in range(n) if j not in (a, b)]
    perm = np.argsort(rng.random((m, n - 2)), 1)
    X[:, a] = v[:, -1]; X[:, b] = v[:, -2]; X[:, others] = np.take_along_axis(v[:, :-2], perm, 1)
    return X


Ns = [2 ** k for k in (10, 12, 14, 16)]
R = 4
col = dict(a1_k0=[], a1_k2=[], a2_k0=[], b1={N: [] for N in Ns}, b2={N: [] for N in Ns}, b3={N: [] for N in Ns})
fl = dict(a1_k0=[], a1_k2=[], a2_k0=[])
for k in range(NM):
    W = perturbed(k)
    m = az.model_from(W, r=r0, sigma=sigma); m.calibrate_r(2 ** 18, iters=2); r = m.r.double().numpy()
    net = dag.Net(W, r, sigma)
    rng = np.random.default_rng(5000 + k)
    # (a1) ADF full network, also gives per-cell predictions for (b)
    if SUR_ONLY:
        As = adf.ADF(W, r, sigma, k_split=0, keep=keep, mu=mu_base); cs = As.run(); es_, _ = As.error_rate(cs, 'seq')
        col['a2_k0'].append(es_ - P[k]); fl['a2_k0'].append(As.fl.n)
        y_of = lambda X: np.argsort(X, -1)[:, -2]
        for N in Ns:
            for rep in range(R):
                X1 = rng.standard_normal((N, n)); lo1, _, _ = net.run(X1, rng.standard_normal((N, n, d)), keep=keep, mu=mu_base)
                m1 = (lo1.argmax(1) != y_of(X1)).mean()
                N2 = max(N // 4, 1); X2 = rng.standard_normal((N2, n)); eps2 = rng.standard_normal((N2, n, d))
                lo2n, _, _ = net.run(X2, eps2); lo2s, _, _ = net.run(X2, eps2, keep=keep, mu=mu_base)
                d2 = ((lo2n.argmax(1) != y_of(X2)).astype(float) - (lo2s.argmax(1) != y_of(X2)).astype(float)).mean()
                col['b3'][N].append((m1 + d2 - P[k]) ** 2)
        print('model', k, 'truth %.4f adf_sur %.4f' % (P[k], es_), flush=True)
        continue
    A0 = adf.ADF(W, r, sigma, k_split=0); c0 = A0.run(); e0, per0 = A0.error_rate(c0, 'seq')
    wts = {};
    for key, lst in c0.items(): wts[key[:2]] = wts.get(key[:2], 0.0) + sum(w for w, _, _ in lst)
    pc = {}
    for key, v in per0.items(): pc[key[:2]] = pc.get(key[:2], 0.0) + v
    phat = np.array([pc.get(c, 0.0) / max(wts.get(c, 1e-300), 1e-300) for c in cells]); phat = np.clip(phat, 1e-4, 1 - 1e-4)
    col['a1_k0'].append(e0 - P[k]); fl['a1_k0'].append(A0.fl.n)
    A2 = adf.ADF(W, r, sigma, k_split=2); c2 = A2.run(); e2, _ = A2.error_rate(c2, 'seq')
    col['a1_k2'].append(e2 - P[k]); fl['a1_k2'].append(A2.fl.n)
    if keep is not None:
        As = adf.ADF(W, r, sigma, k_split=0, keep=keep, mu=mu_base); cs = As.run(); es_, _ = As.error_rate(cs, 'seq')
        col['a2_k0'].append(es_ - P[k]); fl['a2_k0'].append(As.fl.n)
    y_of = lambda X: np.argsort(X, -1)[:, -2]
    p_adf = float(np.mean(phat))           # cells are equiprobable
    for N in Ns:
        for rep in range(R):
            # b1: Neyman allocation over cells from the deductive per-cell predictions
            alloc = np.sqrt(phat * (1 - phat)); alloc = np.maximum(2, np.floor(N * alloc / alloc.sum())).astype(int)
            est = 0.0
            for ci, (a, b) in enumerate(cells):
                X = cell_sample(a, b, alloc[ci], rng)
                lo, _, _ = net.run(X, rng.standard_normal((len(X), n, d)))
                est += (lo.argmax(1) != b).mean() / len(cells)
            col['b1'][N].append((est - P[k]) ** 2)
            # b2: control variate with the ADF per-cell prediction of each input's cell
            X = rng.standard_normal((N, n)); lo, _, _ = net.run(X, rng.standard_normal((N, n, d)))
            e = (lo.argmax(1) != y_of(X)).astype(float)
            a_ = np.argmax(X, 1); b_ = y_of(X)
            Y = phat[np.array([cells.index((ai, bi)) for ai, bi in zip(a_, b_)])]
            cv = np.cov(e, Y); beta = cv[0, 1] / max(cv[1, 1], 1e-300)
            col['b2'][N].append((e.mean() - beta * (Y.mean() - p_adf) - P[k]) ** 2)
            # b3: two-level with the surrogate (base mu)
            if keep is not None:
                X1 = rng.standard_normal((N, n)); lo1, _, _ = net.run(X1, rng.standard_normal((N, n, d)), keep=keep, mu=mu_base)
                m1 = (lo1.argmax(1) != y_of(X1)).mean()
                N2 = max(N // 4, 1); X2 = rng.standard_normal((N2, n)); eps2 = rng.standard_normal((N2, n, d))
                lo2n, _, _ = net.run(X2, eps2); lo2s, _, _ = net.run(X2, eps2, keep=keep, mu=mu_base)
                d2 = ((lo2n.argmax(1) != y_of(X2)).astype(float) - (lo2s.argmax(1) != y_of(X2)).astype(float)).mean()
                col['b3'][N].append((m1 + d2 - P[k]) ** 2)
    print('model', k, 'truth %.4f adf %.4f adf_sur %s' % (P[k], e0, ('%.4f' % es_) if keep is not None else '-'), flush=True)
if SUR_ONLY:
    res = dict(model=path, surrogate_edges=int(keep['rec'].sum() + keep['inp'].sum() + keep['out'].sum()),
               a=dict(adf_on_surrogate_k0=dict(flops=float(np.mean(fl['a2_k0'])), mse=float(np.mean(np.square(col['a2_k0']))), bias=float(np.mean(col['a2_k0'])))),
               b=dict(two_level_surrogate=[dict(N=N, flops=N * FL_SUR + (N // 4) * (FL + FL_SUR), mse=float(np.mean(col['b3'][N]))) for N in Ns]),
               flops_per_sample_surrogate=float(FL_SUR), seconds=time.time() - t0)
    json.dump(res, open(out, 'w'), indent=1); print(json.dumps(res, indent=0)); sys.exit(0)
adf_fl = float(np.mean(fl['a1_k0']))
res = dict(model=path, sigma=sigma, NM=NM, truths=P.tolist(), flops_per_sample=FL,
           a=dict(adf_k0=dict(flops=adf_fl, mse=float(np.mean(np.square(col['a1_k0']))), bias=float(np.mean(col['a1_k0']))),
                  adf_k2=dict(flops=float(np.mean(fl['a1_k2'])), mse=float(np.mean(np.square(col['a1_k2']))), bias=float(np.mean(col['a1_k2'])))),
           b=dict(neyman_adf=[dict(N=N, flops=adf_fl + N * FL, mse=float(np.mean(col['b1'][N]))) for N in Ns],
                  cv_adf_cells=[dict(N=N, flops=adf_fl + N * FL, mse=float(np.mean(col['b2'][N]))) for N in Ns]),
           c=dict(mc=E['est']['mc']))
if keep is not None:
    res['a']['adf_on_surrogate_k0'] = dict(flops=float(np.mean(fl['a2_k0'])), mse=float(np.mean(np.square(col['a2_k0']))), bias=float(np.mean(col['a2_k0'])))
    res['b']['two_level_surrogate'] = [dict(N=N, flops=N * FL_SUR + (N // 4) * (FL + FL_SUR), mse=float(np.mean(col['b3'][N]))) for N in Ns]
    res['surrogate_edges'] = int(keep['rec'].sum() + keep['inp'].sum() + keep['out'].sum())
res['seconds'] = time.time() - t0
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ('truths',)}, indent=0)[:3000])
