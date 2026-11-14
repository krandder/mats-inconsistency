"""Amendment 7, step 2: TOP3 under ARC's protocol, three columns, on an existing estim.py family.
usage: python bar_top3.py MODEL.npz ESTIM.json OUT.json
(a) TOP3 on each model's own weights and recalibrated r, budgets (KS, KG); nothing fitted.
(b) hybrids: (b1) control variate with the TOP3 box prediction g(x) (E g exact); (b2) stratified sampling over 8
    strata of (cell, box) units grouped by predicted error, Neyman allocation with predicted s.d. floored at 0.02.
(c) from the estim file (Monte Carlo, stratified over 90 cells, RQMC).
Diagnostics on the base model: per (pos1, pos2) cell and reliability by predicted error; per-input predictions
against eps(x) (noise-corrected R^2, as in sysid.py).
"""
import sys, json, time, math
import numpy as np
import torch
import az, top3

torch.set_num_threads(1)
t0 = time.time()
path, epath, out = sys.argv[1], sys.argv[2], sys.argv[3]
W0, r0, sigma = az.load_npz(path)
E = json.load(open(epath)); P = np.array(E['truths']); NM = len(P); DELTA = E['DELTA']; FL = E['flops_per_sample']
n = W0[2].shape[0]
BUD = [(1, 2), (2, 2), (2, 3), (3, 4), (4, 6), (6, 10), (8, 16)]   # (6, 10) and (8, 16) added after the smoke test (convergence)
B_SETUP = [(1, 2), (2, 3)]
rng = np.random.default_rng(97531)


def perturbed(k):
    if k == 0: return [w.copy() for w in W0]
    g = np.random.default_rng(1000 + k)
    return [w + DELTA * np.sqrt((w ** 2).mean()) * g.standard_normal(w.shape) for w in W0]


@torch.no_grad()
def batch_err(m, x, gen):
    lo = m(torch.as_tensor(x, dtype=torch.float32), noise=True, gen=gen)
    return (lo.argmax(-1).numpy() != np.argsort(x, -1)[:, -2]).astype(float)


res = dict(model=path, sigma=sigma, truths=P.tolist(), flops_per_sample=FL, budgets=BUD, a=[], b1={}, b2={}, diag={})
models = []
for k in range(NM):
    Wk = perturbed(k); m = az.model_from(Wk, r=r0, sigma=sigma); m.calibrate_r(2 ** 18, iters=2)
    rk = m.r.double().numpy(); T3 = top3.Top3(Wk, rk, sigma)
    tabs = {}
    ests = []
    for (KS, KG) in BUD:
        t1 = time.time(); p, fl, T, G = T3.estimate(KS, KG); tabs[(KS, KG)] = (p, fl, T, G); ests.append((p, fl, time.time() - t1))
    models.append(dict(k=k, m=m, T3=T3, tabs=tabs))
    print('model', k, 'truth %.5f' % P[k], ' '.join('%s:%.5f' % (str(b), e[0]) for b, e in zip(BUD, ests)),
          '(%.0fs)' % sum(e[2] for e in ests), flush=True)
for bi, b in enumerate(BUD):
    est = np.array([mm['tabs'][b][0] for mm in models]); err = est - P
    res['a'].append(dict(KS=b[0], KG=b[1], nodes=int(len(models[0]['T3'].cells) * b[0] * b[1] ** 2),
                         flops=float(models[0]['tabs'][b][1]), mse=float((err ** 2).mean()), bias=float(err.mean()),
                         mse_without_base=float((err[1:] ** 2).mean()), est=est.tolist()))
    print('(a)', b, 'flops %.3g mse %.3g bias %+.4f' % (res['a'][-1]['flops'], res['a'][-1]['mse'], res['a'][-1]['bias']), flush=True)
json.dump(res, open(out, 'w'), indent=1)

# ---------------- diagnostics on the base model ----------------
mm = models[0]; T3 = mm['T3']; m0 = mm['m']; g0 = torch.Generator().manual_seed(4242)
pF, flF, TF, GF = mm['tabs'][BUD[-1]]
xd = rng.standard_normal((2 ** 20, n)); ed = np.concatenate([batch_err(m0, xd[i:i + 2 ** 16], g0) for i in range(0, 2 ** 20, 2 ** 16)])
cell, box = T3.box_of(xd, GF); gd = TF[cell, box]
idx = np.argsort(-xd, 1); c90 = idx[:, 0] * n + idx[:, 1]
pred90 = np.zeros(n * n); w90 = np.zeros(n * n)
for ci, (i, j, kk) in enumerate(T3.cells):
    pred90[i * n + j] += (TF[ci] * GF['pb']).sum(); w90[i * n + j] += 1
pred90 = pred90 / np.maximum(w90, 1)
emp90 = np.array([ed[c90 == c].mean() if (c90 == c).any() else np.nan for c in range(n * n)])
ok = w90 > 0
res['diag']['base_truth'] = float(P[0]); res['diag']['base_sampled_2^20'] = float(ed.mean())
res['diag']['cells90'] = dict(pred=pred90[ok].tolist(), emp=emp90[ok].tolist(),
                              corr=float(np.corrcoef(pred90[ok], emp90[ok])[0, 1]))
edges = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1.01]
rel = []
for lo_, hi_ in zip(edges[:-1], edges[1:]):
    s = (gd >= lo_) & (gd < hi_)
    if s.any(): rel.append(dict(lo=lo_, hi=hi_, mass=float(s.mean()), pred=float(gd[s].mean()), emp=float(ed[s].mean())))
res['diag']['reliability_finest'] = rel
# scenario breakdown: negative 2nd max, near-tie of 2nd/3rd (lowest decile of g23), rest
srt = np.take_along_axis(xd, idx, 1); g23 = srt[:, 1] - srt[:, 2]; q10 = np.quantile(g23, 0.1)
for nm, s in (('neg2nd', srt[:, 1] < 0), ('tie23_decile', (g23 < q10) & (srt[:, 1] >= 0)), ('rest', (g23 >= q10) & (srt[:, 1] >= 0))):
    res['diag']['scen_' + nm] = dict(mass=float(s.mean()), emp=float(ed[s].mean()), pred_box=float(gd[s].mean()))
# per-input predictions against eps(x)
xi = rng.standard_normal((2 ** 14, n))
eA = np.zeros(len(xi)); eB = np.zeros(len(xi))
for rep in range(16):
    eA += batch_err(m0, xi, g0); eB += batch_err(m0, xi, g0)
eA /= 16; eB /= 16
pi = T3.predict_inputs(xi)
ve = float(np.cov(eA, eB)[0, 1])
r2 = 0.5 * sum(1 - (np.mean((pi - a) ** 2) - np.mean(a * (1 - a)) / 15) / ve for a in (eA, eB))
from sklearn.metrics import roc_auc_score
e1 = batch_err(m0, xi, g0)
res['diag']['per_input'] = dict(n=len(xi), mean_pred=float(pi.mean()), mean_eps=float(0.5 * (eA.mean() + eB.mean())),
                               share_R2=float(r2), auc=float(roc_auc_score(e1, pi)), corr=float(np.corrcoef(pi, 0.5 * (eA + eB))[0, 1]))
print('diag', json.dumps({k: v for k, v in res['diag'].items() if k not in ('cells90',)})[:1500], flush=True)
json.dump(res, open(out, 'w'), indent=1)

# ---------------- column (b) ----------------
Ns = [2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16]; R = 4
se1 = {(b, N): [] for b in BUD for N in Ns}; se2 = {(b, N): [] for b in B_SETUP for N in Ns}
for mm in models:
    k = mm['k']; T3 = mm['T3']
    for N in Ns:
        for rep in range(R):
            gen = torch.Generator().manual_seed(7 * N + 31 * rep + 1000 * k + 5)
            rr = np.random.default_rng(13 * N + 17 * rep + 1000 * k + 5)
            x = rr.standard_normal((N, n)); e = batch_err(mm['m'], x, gen)
            for b in BUD:
                p_b, fl_b, T, G = mm['tabs'][b]; c_, bx = T3.box_of(x, G); g = T[c_, bx]
                cv = np.cov(e, g); beta = cv[0, 1] / max(cv[1, 1], 1e-300)
                se1[(b, N)].append((e.mean() - beta * (g.mean() - p_b) - P[k]) ** 2)
            for b in B_SETUP:
                p_b, fl_b, T, G = mm['tabs'][b]; nb = T.shape[1]; nC = T.shape[0]
                pu = T.ravel(); wu = np.tile(G['pb'], nC) / nC
                st = np.clip(np.searchsorted(edges, pu, 'right') - 1, 0, len(edges) - 2)
                hs = [h for h in range(len(edges) - 1) if (st == h).any()]
                Wh = np.array([wu[st == h].sum() for h in hs]); ph = np.array([(wu[st == h] * pu[st == h]).sum() for h in hs]) / Wh
                Sh = np.maximum(np.sqrt(ph * (1 - ph)), 0.02); al = Wh * Sh / (Wh * Sh).sum()
                Nh = np.maximum(2, np.round(N * al).astype(int)); est = 0.0
                for h, Wv, nh in zip(hs, Wh, Nh):
                    units = np.flatnonzero(st == h); pr = wu[units] / wu[units].sum()
                    pick = units[rr.choice(len(units), size=nh, p=pr)]
                    xs = T3.sample_box(pick // nb, pick % nb, G, rr)
                    est += Wv * batch_err(mm['m'], xs, gen).mean()
                se2[(b, N)].append((est - P[k]) ** 2)
    print('(b) model', k, 'done', '%.0fs' % (time.time() - t0), flush=True)
for b in BUD:
    fl_b = models[0]['tabs'][b][1]
    res['b1'][str(b)] = [dict(N=N, flops=float(fl_b + N * (FL + 300)), mse=float(np.mean(se1[(b, N)])),
                              mse_se=float(np.std(se1[(b, N)]) / math.sqrt(len(se1[(b, N)])))) for N in Ns]
for b in B_SETUP:
    fl_b = models[0]['tabs'][b][1]
    res['b2'][str(b)] = [dict(N=N, flops=float(fl_b + N * (FL + 300)), mse=float(np.mean(se2[(b, N)])),
                              mse_se=float(np.std(se2[(b, N)]) / math.sqrt(len(se2[(b, N)])))) for N in Ns]
res['c'] = E['est']
res['seconds'] = time.time() - t0
json.dump(res, open(out, 'w'), indent=1)
print('b1', json.dumps(res['b1'])[:1500]); print('b2', json.dumps(res['b2'])[:1500]); print('done %.0fs' % res['seconds'])
