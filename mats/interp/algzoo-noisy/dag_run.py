"""Build question-driven DAGs for a noisy RNN and compare with forward-only pruning.
usage: python dag_run.py MODEL.npz OUT.json [N_FIT_LOG2] [N_TEST_LOG2]
"""
import sys, json, time
import numpy as np
import az, dag

t0 = time.time()
path, out = sys.argv[1], sys.argv[2]
nfit = 2 ** int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 2 ** 14
ntest = 2 ** int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 2 ** 16
W, r, sigma = az.load_npz(path)
net = dag.Net(W, r, sigma); d, n = net.d, net.n
rng = np.random.default_rng(2024)
xm = rng.standard_normal((2 ** 16, n)); em = rng.standard_normal((2 ** 16, n, d))
_, hsm, _ = net.run(xm, em); mu = [h.mean(0) for h in hsm]
xf = rng.standard_normal((nfit, n)); ef = rng.standard_normal((nfit, n, d))
xt = rng.standard_normal((ntest, n)); et = rng.standard_normal((ntest, n, d))
full = net.full_keep()
_, base_f = dag.question_metrics(net, xf, ef, full, mu)
_, base_t = dag.question_metrics(net, xt, et, full, mu)
# noise floor of the question: flip rate between two independent noise draws on the same inputs
lo2, _, _ = net.run(xt, rng.standard_normal((ntest, n, d)))
y_t = np.argsort(xt, -1)[:, -2]
noise_flip = float(((lo2.argmax(1) != y_t) != base_t['err']).mean())
sal, m, s = net.saliency(xf, ef, mu)
# forward-only saliency: contribution to the chain-rule path KL of each edge alone (first order, gate-on only)
fsal = dict(rec=np.zeros((n, d, d)), inp=np.zeros((n, d)), out=np.full(d, np.inf))
hs = base_f['hs']
for t in range(n):
    zt = hs[t] @ net.W_hh.T + xf[:, t:t + 1] * net.W_hi
    on = (zt > 0).astype(float); den = 2 * (sigma * r[t]) ** 2 + 1e-300
    fsal['inp'][t] = (on * (xf[:, t:t + 1] * net.W_hi) ** 2).mean(0) / den
    if t >= 1:
        dev = hs[t] - mu[t]
        fsal['rec'][t] = (np.einsum('bi,bj->ij', on, dev ** 2) / nfit) * net.W_hh ** 2 / den[:, None]
edges = []
for t in range(1, n):
    for i in range(d):
        for j in range(d): edges.append(('rec', t, i, j))
for t in range(n):
    for i in range(d): edges.append(('inp', t, i, -1))
for i in range(d): edges.append(('out', n, i, -1))


def val(S, e):
    k, t, i, j = e
    return S['rec'][t, i, j] if k == 'rec' else (S['inp'][t, i] if k == 'inp' else S['out'][i])


def keep_from(sel):
    K = dict(rec=np.zeros((n, d, d), bool), inp=np.zeros((n, d), bool), out=np.zeros(d, bool))
    for (k, t, i, j) in sel:
        if k == 'rec': K['rec'][t, i, j] = True
        elif k == 'inp': K['inp'][t, i] = True
        else: K['out'][i] = True
    return K


res = dict(model=path, sigma=sigma, nfit=nfit, ntest=ntest, p_net_test=float(base_t['err'].mean()), noise_flip=noise_flip,
           n_edges_total=len(edges), sweep_question=[], sweep_forward=[], greedy=[])
order_q = sorted(edges, key=lambda e: -val(sal, e))
order_f = sorted(edges, key=lambda e: -val(fsal, e))
for name, order in (('sweep_question', order_q), ('sweep_forward', order_f)):
    for K in sorted(set([max(1, int(round(f * len(edges)))) for f in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0)])):
        keep = keep_from(order[:K])
        qm, _ = dag.question_metrics(net, xt, et, keep, mu, base_t)
        res[name].append(dict(K=K, **{k: v for k, v in qm.items() if k != 'pathKL_steps'}))
    print(name, [(x['K'], round(x['flip'], 4), round(x['pathKL'], 1)) for x in res[name]], flush=True)
json.dump(res, open(out, 'w'), indent=1)

# greedy refinement at several flip targets: start from the smallest top-K prefix meeting the target, then try to
# remove kept edges (lowest saliency first), accept if flip on the fit set stays <= target
TAUS = [] if 'nogreedy' in sys.argv else [0.02, 0.01, 0.005]
for tau in TAUS:
    lo, hi = 1, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        qm, _ = dag.question_metrics(net, xf, ef, keep_from(order_q[:mid]), mu, base_f)
        if qm['flip'] <= tau: hi = mid
        else: lo = mid + 1
    sel = list(order_q[:lo]); start = lo
    for e in ([] if 'prefixonly' in sys.argv else sorted(sel, key=lambda e: val(sal, e))):
        trial = [x for x in sel if x != e]
        qm, _ = dag.question_metrics(net, xf, ef, keep_from(trial), mu, base_f)
        if qm['flip'] <= tau: sel = trial
    keep = keep_from(sel)
    qt, _ = dag.question_metrics(net, xt, et, keep, mu, base_t)
    qf, _ = dag.question_metrics(net, xf, ef, keep, mu, base_f)
    nodes_used = sorted({(t, j) for (k, t, i, j) in sel if k == 'rec'} | {(n, i) for (k, t, i, j) in sel if k == 'out'})
    row = dict(tau=tau, start_K=start, K=len(sel), fit=qf, test=qt, edges=[list(map(lambda v: v if isinstance(v, str) else int(v), e)) for e in sel],
               in_deg=keep['rec'].sum(2).tolist(), out_deg=keep['rec'].sum(1).tolist(), inp=keep['inp'].astype(int).tolist(),
               out=keep['out'].astype(int).tolist())
    res['greedy'].append(row)
    print('tau', tau, 'start', start, 'K', len(sel), 'test flip %.4f dp %+.4f pathKL %.3g' % (qt['flip'], qt['dp'], qt['pathKL']), flush=True)
    json.dump(res, open(out, 'w'), indent=1)
res['seconds'] = time.time() - t0
json.dump(res, open(out, 'w'), indent=1)
print('done', res['seconds'])
