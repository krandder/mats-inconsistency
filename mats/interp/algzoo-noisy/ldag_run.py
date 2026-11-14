"""Loss-targeted DAG (PREREG Amendment 4): the smallest mean-ablated edge set whose AVERAGE loss (cross-entropy, and
separately the error rate) stays within sampling's standard error, overall and in 13 fixed strata.

usage: python ldag_run.py MODEL.npz OUT.json [greedy_targets e.g. "dL:14"]
Strata (fixed in PREREG Amendment 4): quartiles of the top-two gap (4) and max-position group x 2nd-max-position group
with groups {0-2, 3-6, 7-9} (9). Tolerance for N_ref network samples: overall sd/sqrt(N_ref) (fit-set bias must be
<= 0.5 of it), stratum k: sd_k / sqrt(N_ref pi_k).
"""
import sys, json, time, math
import numpy as np
import az, dag

t0 = time.time()
path, out = sys.argv[1], sys.argv[2]
GREEDY = sys.argv[3].split(',') if len(sys.argv) > 3 else ['dL:14']
W, r, sigma = az.load_npz(path)
net = dag.Net(W, r, sigma); d, n = net.d, net.n
rng = np.random.default_rng(777)
xm = rng.standard_normal((2 ** 16, n)); em = rng.standard_normal((2 ** 16, n, d))
_, hsm, _ = net.run(xm, em); mu = [h.mean(0) for h in hsm]
NF, NT = 2 ** 15, 2 ** 16
xf = rng.standard_normal((NF, n)); ef = rng.standard_normal((NF, n, d))
xt = rng.standard_normal((NT, n)); et = rng.standard_normal((NT, n, d))
# gap quartile thresholds fixed from a large independent sample of the input law
xq = np.sort(np.random.default_rng(1).standard_normal((2 ** 20, n)), 1); GQ = np.quantile(xq[:, -1] - xq[:, -2], [0.25, 0.5, 0.75])


def strata(x):
    s = np.sort(x, 1); gap = s[:, -1] - s[:, -2]; a = np.argmax(x, 1); b = np.argsort(x, 1)[:, -2]
    grp = lambda p: np.where(p <= 2, 0, np.where(p <= 6, 1, 2))
    S = [np.digitize(gap, GQ) == q for q in range(4)] + [(grp(a) == i) & (grp(b) == j) for i in range(3) for j in range(3)]
    return S


def ce_err(lo, x):
    y = np.argsort(x, -1)[:, -2]
    m = lo.max(1, keepdims=True); lse = m[:, 0] + np.log(np.exp(lo - m).sum(1))
    return lse - lo[np.arange(len(x)), y], (lo.argmax(1) != y).astype(float)


lo_f, hs_f, _ = net.run(xf, ef); CEf, ERf = ce_err(lo_f, xf); Sf = strata(xf)
lo_t, hs_t, _ = net.run(xt, et); CEt, ERt = ce_err(lo_t, xt); St = strata(xt)
sd = dict(dL=CEt.std(), dp=ERt.std()); sd_k = dict(dL=[CEt[s].std() for s in St], dp=[ERt[s].std() for s in St])
pi_k = [s.mean() for s in St]


def tol(target, Nref):
    return sd[target] / math.sqrt(Nref), [sd_k[target][k] / math.sqrt(Nref * pi_k[k]) for k in range(len(St))]


def biases(keep, x, e, S, base_ce, base_er):
    lo, _, _ = net.run(x, e, keep=keep, mu=mu)
    ce, er = ce_err(lo, x)
    dce, der = ce - base_ce, er - base_er
    return dict(dL=float(dce.mean()), dp=float(der.mean()), dL_k=[float(dce[s].mean()) for s in S], dp_k=[float(der[s].mean()) for s in S],
                dL_se=float(dce.std() / math.sqrt(len(x))), flip=float((er != base_er).mean()))


def passes(b, target, Nref, slack=0.5):
    T, Tk = tol(target, Nref)
    ok = abs(b[target]) <= slack * T and all(abs(b[target + '_k'][k]) <= Tk[k] for k in range(len(Tk)))
    worst = max([abs(b[target]) / T] + [abs(b[target + '_k'][k]) / Tk[k] for k in range(len(Tk))])
    return ok, worst


# loss saliency (first order, squared): E[(dCE/dz_{t+1,i} * W_ij * (h_{t,j} - mu_{t,j}))^2]
y = np.argsort(xf, -1)[:, -2]
P = np.exp(lo_f - lo_f.max(1, keepdims=True)); P /= P.sum(1, keepdims=True); P[np.arange(NF), y] -= 1
g_h = P @ net.W_oh
sal = dict(rec=np.zeros((n, d, d)), inp=np.zeros((n, d)), out=np.zeros(d))
sal['out'] = (P[:, :, None] * net.W_oh[None] * (hs_f[n] - mu[n])[:, None, :]).sum(1).__pow__(2).mean(0)
for t in range(n - 1, -1, -1):
    zt = hs_f[t] @ net.W_hh.T + xf[:, t:t + 1] * net.W_hi
    g_z = g_h * (zt > 0)
    sal['inp'][t] = ((g_z * xf[:, t:t + 1] * net.W_hi) ** 2).mean(0)
    if t >= 1:
        dev = hs_f[t] - mu[t]
        sal['rec'][t] = np.einsum('bi,bj->ij', g_z ** 2, dev ** 2) / NF * net.W_hh ** 2
    g_h = g_z @ net.W_hh
edges = [('rec', t, i, j) for t in range(1, n) for i in range(d) for j in range(d)] + [('inp', t, i, -1) for t in range(n) for i in range(d)] + [('out', n, i, -1) for i in range(d)]
val = lambda e: sal['rec'][e[1], e[2], e[3]] if e[0] == 'rec' else (sal['inp'][e[1], e[2]] if e[0] == 'inp' else sal['out'][e[2]])
order = sorted(edges, key=lambda e: -val(e))


def keep_from(sel):
    K = dict(rec=np.zeros((n, d, d), bool), inp=np.zeros((n, d), bool), out=np.zeros(d, bool))
    for (k, t, i, j) in sel:
        if k == 'rec': K['rec'][t, i, j] = True
        elif k == 'inp': K['inp'][t, i] = True
        else: K['out'][i] = True
    return K


res = dict(model=path, sigma=sigma, n_edges=len(edges), sd=dict(dL=float(sd['dL']), dp=float(sd['dp'])), pi_k=[float(v) for v in pi_k],
           p_net=float(ERt.mean()), L_net=float(CEt.mean()), runs=[])
for target in ('dL', 'dp'):
    for NrefL in (10, 14, 18):
        Nref = 2 ** NrefL
        lo_, hi_ = 1, len(edges)
        while lo_ < hi_:
            mid = (lo_ + hi_) // 2
            ok, _ = passes(biases(keep_from(order[:mid]), xf, ef, Sf, CEf, ERf), target, Nref)
            if ok: hi_ = mid
            else: lo_ = mid + 1
        sel = list(order[:lo_]); kp = lo_
        if f'{target}:{NrefL}' in GREEDY:
            # block-greedy removal (blocks of 32 edges in increasing saliency, split on failure)
            cand = sorted(sel, key=val)
            def try_remove(block):
                global sel
                trial = [e for e in sel if e not in set(block)]
                ok, _ = passes(biases(keep_from(trial), xf, ef, Sf, CEf, ERf), target, Nref)
                if ok: sel = trial; return
                if len(block) > 1:
                    try_remove(block[:len(block) // 2]); try_remove(block[len(block) // 2:])
            for i in range(0, len(cand), 32):
                try_remove(cand[i:i + 32])
        keep = keep_from(sel)
        bt = biases(keep, xt, et, St, CEt, ERt); okt, worst = passes(bt, target, Nref, slack=1.0)
        T, Tk = tol(target, Nref)
        row = dict(target=target, Nref_log2=NrefL, tol=T, K_prefix=kp, K=len(sel), frac=len(sel) / len(edges), greedy=f'{target}:{NrefL}' in GREEDY,
                   test=bt, test_pass=bool(okt), test_worst_ratio=float(worst),
                   per_step_rec=[int(keep['rec'][t].sum()) for t in range(1, n)], inp=int(keep['inp'].sum()), out=int(keep['out'].sum()))
        if row['greedy'] or target == 'dp': row['edges'] = [[e[0], int(e[1]), int(e[2]), int(e[3])] for e in sel]
        res['runs'].append(row)
        print(target, NrefL, 'tol %.2e' % T, 'K_prefix', kp, 'K', len(sel), '(%.0f%%)' % (100 * len(sel) / len(edges)), 'test bias %.2e (dL) %.2e (dp)' % (bt['dL'], bt['dp']),
              'flip %.3f' % bt['flip'], 'test pass', okt, 'worst %.2f' % worst, '%.0fs' % (time.time() - t0), flush=True)
        json.dump(res, open(out, 'w'), indent=1)
res['seconds'] = time.time() - t0
json.dump(res, open(out, 'w'), indent=1)
