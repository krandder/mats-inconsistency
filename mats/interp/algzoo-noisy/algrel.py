"""Descriptions relative to the algorithm (PREREG Amendment 6).

Advice, fitted once on the base model's own samples: per step t and algorithmic cell c = (a_t, b_t), the CL regression
h_t ~ N(B_{t,c} f(u), Sigma_{t,c}), u = (M, S, x_last), f = [1, M, S, x_last, relu(M), relu(S), relu(x_last)].
 1. Algorithm-relative ablation DAG: an ablated edge (t, j -> i) carries W_ij * (B_{t,c} f(u))_j for the input's own
    cell and u (the conditional mean given the algorithm state) instead of W_ij * E h_tj. Flip and dp targets as before.
 2. Algorithm-indexed description at step n with the network's noise (linear response to every competitor): per cell
    and u, logits ~ N(W_oh B f(u), W_oh Sigma W_oh^T); P(error | c, u) by K fixed Gaussian nodes (deterministic).
    Deductive average: quadrature over the exact law of u given the cell (midpoint quantile rules); per-input
    prediction eps_hat(x) at the input's own (c, u).
 3. ARC's bar in three columns over the family of ESTIM.json: (a) deductive average; for a variant, the last step is
    recomputed with the variant's weights from the base description of h_{n-1} (statistical linearisation of the
    ReLU, exact noise); (b) control variate e(x) - beta (eps_hat_base(x) - E eps_hat_base); (c) Monte Carlo.
usage: python algrel.py MODEL.npz ESTIM.json OUT.json [skipdag]
"""
import sys, json, math, time
import numpy as np
from scipy.stats import norm, qmc
from scipy.special import ndtr
import az, alg, dag

t0 = time.time()
path, epath, out = sys.argv[1], sys.argv[2], sys.argv[3]
SKIPDAG = 'skipdag' in sys.argv
W, r, sigma = az.load_npz(path)
W_hi, W_hh, W_oh = W; d = W_hi.shape[0]; n = W_oh.shape[0]
rng = np.random.default_rng(424242)
F = 7


def feats(M, S, xl):
    S = np.where(np.isfinite(S), S, 0.0)
    return np.stack([np.ones_like(M), M, S, xl, np.maximum(M, 0), np.maximum(S, 0), np.maximum(xl, 0)], -1)


def cell_index(a, b):
    return (a + 1) * 11 + (b + 1)


# ---------------- advice: CL fit on the base model ----------------
NFIT = 2 ** 17
xf = rng.standard_normal((NFIT, n)); hsf, _, _ = alg.sample_chain(W, r, sigma, xf, rng); Af = alg.alg_states(xf)
CL = {}
for t in range(1, n + 1):
    key = cell_index(Af['a'][:, t], Af['b'][:, t]); Fe = feats(Af['M'][:, t], Af['S'][:, t], xf[:, t - 1])
    CL[t] = {}
    for k in np.unique(key):
        s = key == k
        beta, *_ = np.linalg.lstsq(Fe[s], hsf[s, t], rcond=None)
        R_ = hsf[s, t] - Fe[s] @ beta
        CL[t][int(k)] = (beta, np.cov(R_.T) + 1e-12 * np.eye(d))
n_params = sum(b.size + d * (d + 1) // 2 for b, _ in CL[n].values())
res = dict(model=path, sigma=sigma, d=d, cells_at_n=len(CL[n]), params_at_n=int(n_params),
           params_all_steps=int(sum(b.size + d * (d + 1) // 2 for t in CL for b, _ in CL[t].values())))


def cond_means(x, A):
    """Per-input conditional means hhat_t (B, n+1, d) from the advice."""
    B = x.shape[0]; H = np.zeros((B, n + 1, d))
    for t in range(1, n + 1):
        key = cell_index(A['a'][:, t], A['b'][:, t]); Fe = feats(A['M'][:, t], A['S'][:, t], x[:, t - 1])
        for k in np.unique(key):
            s = key == k
            beta = CL[t][int(k)][0] if int(k) in CL[t] else np.zeros((F, d))
            H[s, t] = Fe[s] @ beta
    return H


# ---------------- 1. algorithm-relative ablation DAG ----------------
def run_rel(x, eps, keep, Hc):
    B = x.shape[0]; h = np.zeros((B, d))
    for t in range(n):
        if keep is None:
            z = h @ W_hh.T + x[:, t:t + 1] * W_hi
        else:
            Km = keep['rec'][t]
            z = h @ (W_hh * Km).T + Hc[:, t] @ (W_hh * (~Km)).T + x[:, t:t + 1] * W_hi
        h = np.maximum(z, 0) + sigma * r[t] * eps[:, t]
    if keep is None: return h @ W_oh.T
    return h @ (W_oh * keep['out']).T + Hc[:, n] @ (W_oh * (~keep['out'])).T


if not SKIPDAG:
    NF, NT = 2 ** 14, 2 ** 16
    x1 = rng.standard_normal((NF, n)); e1 = rng.standard_normal((NF, n, d)); A1 = alg.alg_states(x1); H1 = cond_means(x1, A1)
    x2 = rng.standard_normal((NT, n)); e2 = rng.standard_normal((NT, n, d)); A2 = alg.alg_states(x2); H2 = cond_means(x2, A2)
    y1 = A1['b'][:, n]; y2 = A2['b'][:, n]
    base1 = run_rel(x1, e1, None, H1).argmax(1) != y1; base2 = run_rel(x2, e2, None, H2).argmax(1) != y2
    # saliency with deviations from the conditional mean
    net = dag.Net(W, r, sigma); lo, hs, _ = net.run(x1, e1)
    lo2_ = lo.copy(); lo2_[np.arange(NF), y1] = -np.inf; jst = lo2_.argmax(1)
    m = lo[np.arange(NF), y1] - lo[np.arange(NF), jst]; sk = 0.1 * m.std(); kk = np.exp(-0.5 * (m / sk) ** 2)
    g_h = W_oh[y1] - W_oh[jst]
    sal = dict(rec=np.zeros((n, d, d)), out=(kk[:, None] * (g_h * (hs[n] - H1[:, n])) ** 2).mean(0))
    for t in range(n - 1, -1, -1):
        zt = hs[t] @ W_hh.T + x1[:, t:t + 1] * W_hi; g_z = g_h * (zt > 0)
        if t >= 1:
            dev = hs[t] - H1[:, t]
            sal['rec'][t] = np.einsum('b,bi,bj->ij', kk, g_z ** 2, dev ** 2) / NF * W_hh ** 2
        g_h = g_z @ W_hh
    edges = [('rec', t, i, j) for t in range(1, n) for i in range(d) for j in range(d)] + [('out', n, i, -1) for i in range(d)]
    val = lambda e: sal['rec'][e[1], e[2], e[3]] if e[0] == 'rec' else sal['out'][e[2]]
    order = sorted(edges, key=lambda e: -val(e))

    def keep_from(sel):
        K = dict(rec=np.zeros((n, d, d), bool), out=np.zeros(d, bool))
        for (kind, t, i, j) in sel:
            if kind == 'rec': K['rec'][t, i, j] = True
            else: K['out'][i] = True
        return K

    def flip_dp(keep, x, e, H, base, y):
        es = run_rel(x, e, keep, H).argmax(1) != y
        return float((es != base).mean()), float(es.mean() - base.mean())

    sweep = []
    for f in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        K = int(round(f * len(edges))); fl_, dp_ = flip_dp(keep_from(order[:K]), x2, e2, H2, base2, y2)
        sweep.append(dict(frac=f, K=K, flip=fl_, dp=dp_)); print('rel-ablation sweep', f, K, round(fl_, 4), round(dp_, 4), flush=True)
    res['dag_rel'] = dict(n_edges=len(edges), note='input edges always kept', sweep=sweep, targets=[])
    for tau in (0.02, 0.01):
        lo_, hi_ = 0, len(edges)
        while lo_ < hi_:
            mid = (lo_ + hi_) // 2
            if flip_dp(keep_from(order[:mid]), x1, e1, H1, base1, y1)[0] <= tau: hi_ = mid
            else: lo_ = mid + 1
        fl_, dp_ = flip_dp(keep_from(order[:lo_]), x2, e2, H2, base2, y2)
        res['dag_rel']['targets'].append(dict(tau=tau, K=lo_, frac=lo_ / len(edges), test_flip=fl_, test_dp=dp_))
        print('rel-ablation tau', tau, 'K', lo_, '(%.1f%%)' % (100 * lo_ / len(edges)), 'test flip %.4f dp %+.4f' % (fl_, dp_), flush=True)
    # noise floor of flips for this question
    eb = run_rel(x2, rng.standard_normal((NT, n, d)), None, H2).argmax(1) != y2
    res['dag_rel']['noise_flip'] = float((eb != base2).mean())
    json.dump(res, open(out, 'w'), indent=1)

# ---------------- 2. algorithm-indexed description, deductive average ----------------
KZ = 128
ZN = norm.ppf(qmc.Sobol(d=n, scramble=True, seed=7).random(KZ)) # fixed deterministic Gaussian nodes for the logit noise


def p_err_gauss(mu_l, L_l, b):
    """P(argmax != b) for logits N(mu_l, L L^T), batched over leading dims of mu_l (..., n); L_l (..., n, n)."""
    ell = mu_l[..., None, :] + np.einsum('...ij,kj->...ki', L_l, ZN)
    return (ell.argmax(-1) != b[..., None]).mean(-1)


def os2_nodes(tn, KS=24, KM=8, KL=6):
    """Quadrature nodes for (M, S, x_last-other) of the top two of tn iid N(0,1): midpoint quantile rules.
    Returns S (KS,), M (KS, KM), XL (KS, KL), equal weights."""
    grid = np.linspace(-6, 6, 20001); P = norm.cdf(grid)
    FS = P ** tn + tn * P ** (tn - 1) * (1 - P)                      # CDF of the 2nd largest
    u = (np.arange(KS) + 0.5) / KS; S = np.interp(u, FS, grid)
    PS = norm.cdf(S)
    v = (np.arange(KM) + 0.5) / KM; M = norm.ppf(PS[:, None] + (1 - PS[:, None]) * v[None, :])
    w = (np.arange(KL) + 0.5) / KL; XL = norm.ppf(PS[:, None] * w[None, :])
    return S, M, XL


def desc_average(Wv, rv, CLn, at_n=True, fl=None):
    """Deductive average error from the step-n description (at_n) for weights Wv (uses W_oh of Wv)."""
    Whi, Whh, Woh = Wv; tot = 0.0
    S, M, XL = os2_nodes(n); KS, KM = M.shape; KL = XL.shape[1]
    cells = [(a, b) for a in range(n) for b in range(n) if a != b]
    for (a, b) in cells:
        beta, Sig = CLn[cell_index(np.array(a), np.array(b)).item()]
        Ll = np.linalg.cholesky(Woh @ Sig @ Woh.T + 1e-12 * np.eye(n))
        if a == n - 1: xl = np.repeat(M[:, :, None], 1, 2)
        elif b == n - 1: xl = np.repeat(np.repeat(S[:, None, None], KM, 1), 1, 2)
        else: xl = np.repeat(XL[:, None, :], KM, 1)
        Mm = np.repeat(M[:, :, None], xl.shape[2], 2); Ss = np.repeat(np.repeat(S[:, None, None], KM, 1), xl.shape[2], 2)
        fe = feats(Mm, Ss, xl)                                       # (KS, KM, KL', F)
        mu_l = (fe @ beta) @ Woh.T
        pe = p_err_gauss(mu_l, np.broadcast_to(Ll, mu_l.shape + (n,)), np.full(mu_l.shape[:-1], b))
        tot += pe.mean() / len(cells)
        if fl is not None: fl[0] += mu_l.size * (2 * F * d / n + 2 * d) + pe.size * KZ * (2 * n * n + n)
    return float(tot)


def desc_average_last_step(Wv, rv, sg, CLm, fl=None, KX=16):
    """Variant estimate: base description of h_{n-1} (CLm = CL[n-1]), the last step with the variant's weights:
    z = Whh h + Whi x, statistical linearisation of ReLU, the variant's noise, logits through Woh."""
    Whi, Whh, Woh = Wv; tot = 0.0; tn = n - 1
    S, M, XL = os2_nodes(tn, KS=16, KM=6, KL=4); KS, KM = M.shape
    xq = norm.ppf((np.arange(KX) + 0.5) / KX)
    cells = [(a, b) for a in range(tn) for b in range(tn) if a != b]
    for (a, b) in cells:
        beta, Sig = CLm[cell_index(np.array(a), np.array(b)).item()]
        if a == tn - 1: xl = M[:, :, None]
        elif b == tn - 1: xl = np.repeat(S[:, None, None], KM, 1)
        else: xl = np.repeat(XL[:, None, :], KM, 1)
        Mm = np.repeat(M[:, :, None], xl.shape[2], 2); Ss = np.repeat(np.repeat(S[:, None, None], KM, 1), xl.shape[2], 2)
        fe = feats(Mm, Ss, xl); mh = fe @ beta                       # (KS, KM, L, d)
        Vz = np.einsum('ij,jk,ik->i', Whh, Sig, Whh)
        for xv in xq:
            mz = mh @ Whh.T + xv * Whi; sz = np.sqrt(np.maximum(Vz, 1e-18))
            uz = mz / sz; Pz = ndtr(uz); E1 = mz * Pz + sz * np.exp(-0.5 * uz ** 2) / math.sqrt(2 * math.pi)
            # covariance of h_n: P (W Sig W^T) P (statistical linearisation) + residual diag + noise
            Cz = Whh @ Sig @ Whh.T
            E2 = (mz ** 2 + sz ** 2) * Pz + mz * sz * np.exp(-0.5 * uz ** 2) / math.sqrt(2 * math.pi)
            Vr = np.maximum(E2 - E1 ** 2 - Pz ** 2 * Vz, 0)
            mu_l = E1 @ Woh.T
            # new cell at n: branch by x vs (M, S)
            Mx = Mm; Sx = Ss
            newb = np.where(xv > Mx, a, np.where(xv > Sx, tn, b))
            # per-node covariance: Woh (P Cz P + diag(Vr) + noise) Woh^T; P varies per node -> use node-wise Cholesky
            Pm = Pz.reshape(-1, d); Vm = Vr.reshape(-1, d)
            C = np.einsum('ni,ij,nj->nij', Pm, Cz, Pm) + np.einsum('ni,ij->nij', Vm, np.eye(d)) + np.diag((sg * rv[n - 1]) ** 2)[None]
            Cl = np.einsum('ki,nij,lj->nkl', Woh, C, Woh) + 1e-12 * np.eye(n)[None]
            Ll = np.linalg.cholesky(Cl)
            pe = p_err_gauss(mu_l.reshape(-1, n), Ll, newb.reshape(-1))
            tot += pe.mean() / len(cells) / KX
            if fl is not None: fl[0] += pe.size * (2 * d * d * 3 + 2 * n * d * d + n ** 3 + KZ * (2 * n * n + n))
    return float(tot)


def eps_hat(x, A, CLn, Woh):
    B = x.shape[0]; key = cell_index(A['a'][:, n], A['b'][:, n]); Fe = feats(A['M'][:, n], A['S'][:, n], x[:, n - 1])
    out_ = np.zeros(B)
    for k in np.unique(key):
        s = key == k; beta, Sig = CLn[int(k)]
        Ll = np.linalg.cholesky(Woh @ Sig @ Woh.T + 1e-12 * np.eye(n))
        mu_l = (Fe[s] @ beta) @ Woh.T
        out_[s] = p_err_gauss(mu_l, np.broadcast_to(Ll, mu_l.shape + (n,)), A['b'][s, n])
    return out_


flc = [0.0]
p_desc_base = desc_average(W, r, CL[n], fl=flc)
res['desc_average_base'] = p_desc_base; res['desc_average_flops'] = flc[0]
fl2 = [0.0]
p_last_base = desc_average_last_step(W, r, sigma, CL[n - 1], fl=fl2)
res['desc_last_step_base'] = p_last_base; res['desc_last_step_flops'] = fl2[0]
# per-input predictions vs the network's per-input error probability (deviation R^2)
xv = rng.standard_normal((2 ** 13, n)); Av = alg.alg_states(xv)
epsn, _, _ = alg.err_prob(W, r, sigma, xv, 32, rng)
eh = eps_hat(xv, Av, CL[n], W_oh)
res['eps_hat_R2'] = float(1 - ((epsn - eh) ** 2).sum() / ((epsn - epsn.mean()) ** 2).sum())
res['eps_hat_mean_sample'] = float(eh.mean())
print('description: base average %.4f (flops %.3g), last-step version %.4f (flops %.3g), eps_hat R2 %.3f' % (p_desc_base, flc[0], p_last_base, fl2[0], res['eps_hat_R2']), flush=True)
json.dump(res, open(out, 'w'), indent=1)

# ---------------- 3. ARC's bar, three columns, over the estim.py family ----------------
E = json.load(open(epath)); P = np.array(E['truths']); NM = len(P); DELTA = E['DELTA']; FL = E['flops_per_sample']


def perturbed(k):
    if k == 0: return [w.copy() for w in W]
    g = np.random.default_rng(1000 + k)
    return [w + DELTA * np.sqrt((w ** 2).mean()) * g.standard_normal(w.shape) for w in W]


Ns = [2 ** k for k in (10, 12, 14)]; R = 4
colA = []; colB = {N: [] for N in Ns}
for k in range(NM):
    Wv = perturbed(k)
    mv = az.model_from(Wv, r=r, sigma=sigma); mv.calibrate_r(2 ** 18, iters=2); rv = mv.r.double().numpy()
    pa = desc_average_last_step(Wv, rv, sigma, CL[n - 1])
    colA.append(pa - P[k])
    for N in Ns:
        for rep in range(R):
            x = rng.standard_normal((N, n)); A = alg.alg_states(x)
            _, _, lo = alg.sample_chain(Wv, rv, sigma, x, rng)
            e = (lo.argmax(1) != A['b'][:, n]).astype(float)
            Y = eps_hat(x, A, CL[n], W_oh)
            cv = np.cov(e, Y); beta = cv[0, 1] / max(cv[1, 1], 1e-300)
            colB[N].append((e.mean() - beta * (Y.mean() - p_desc_base) - P[k]) ** 2)
    print('variant', k, 'truth %.4f  (a) %.4f' % (P[k], pa), flush=True)
FL_EH = 2 * F * d * n / n + KZ * (2 * n * n + n)                    # per-input eps_hat cost (approx.)
res['bar'] = dict(a=dict(flops=fl2[0], mse=float(np.mean(np.square(colA))), bias=float(np.mean(colA)), est=(P + np.array(colA)).tolist()),
                  b=[dict(N=N, flops=flc[0] + N * (FL + FL_EH), mse=float(np.mean(colB[N]))) for N in Ns],
                  c=E['est']['mc'], truths=P.tolist())
res['seconds'] = time.time() - t0
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ('dag_rel',)}, indent=0)[:2500])
