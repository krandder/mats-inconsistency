"""P4 rules on a trained net at one level: exact coalition tables (n <= 12),
Aumann-Shapley / integrated gradients, expected gradients, sampled Shapley ladders,
LOO / MARG / mean-ablation, GRADxACT, MI-1BIT, VINFO-LOGISTIC, WNORM, random.

Game (resample interchange, post-GELU h_l, later layers read the base x):
coalition S keeps h(x) on S's channels and takes h(x') elsewhere.
  flip : mean_j 1[sign z_j = sign z0_j]          (z0 = intact logits)
  soft : mean_j sigmoid(z_j * sign z0_j)          (probability kept on intact argmax)
  pay  : mean_j (1-eps_j) log p_j(y_j) + eps_j log(1-p_j(y_j)), p clipped to [0.01, 0.99]
Costs are forward-equivalents per (x, x') pair, forwards from level l, backward = 2.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
import gen
from nets import xin

GAMES = ('flip', 'soft', 'pay')
CLIP = (0.01, 0.99)


def game_vals(z, s0, ypm, eps):
    """Per-row values of the three games. z, s0, ypm: (R, nout); eps (nout,)."""
    flip = ((torch.sign(z) == s0).float()).mean(1)
    soft = torch.sigmoid(z * s0).mean(1)
    p = torch.sigmoid(z * ypm).clamp(*CLIP)
    pay = ((1 - eps) * torch.log(p) + eps * torch.log(1 - p)).mean(1)
    return {'flip': flip, 'soft': soft, 'pay': pay}


def gmat(groups, width):
    G = torch.zeros(len(groups), width)
    for k, g in enumerate(groups):
        G[k, torch.as_tensor(g)] = 1
    return G


class LevelGame:
    """Holds the pairs and activations for one (net, level)."""

    def __init__(self, net, l, Xbits, Y, groups, npair=2048, seed=0, eps=None):
        rng = np.random.default_rng(seed)
        self.net, self.l = net, l
        self.xall = xin(Xbits)
        with torch.no_grad():
            zall, hs = net(self.xall, want=True)
        self.hall = hs[l - 1]
        self.zall = zall
        self.xi = rng.integers(0, len(Xbits), npair)
        self.xp = rng.integers(0, len(Xbits), npair)
        self.H, self.Hp = self.hall[self.xi], self.hall[self.xp]
        self.D = self.H - self.Hp
        self.xb = self.xall[self.xi]
        self.s0 = torch.sign(zall[self.xi])
        self.s0[self.s0 == 0] = 1
        self.ypm = torch.tensor(2.0 * Y[self.xi] - 1.0, dtype=torch.float32)
        nout = Y.shape[1]
        self.eps = torch.zeros(nout) if eps is None else torch.tensor(eps, dtype=torch.float32)
        self.groups = groups
        self.n = len(groups)
        self.width = self.H.shape[1]
        self.G = gmat(groups, self.width)
        self.P = npair

    # ---- forward helpers
    @torch.no_grad()
    def eval_masks(self, M, pidx=None, chunk_rows=200_000):
        """M: (C, n) 0/1 group masks (1 = keep h(x)); returns dict game -> (C, P') per-pair values."""
        pidx = np.arange(self.P) if pidx is None else pidx
        H, Hp, xb = self.H[pidx], self.Hp[pidx], self.xb[pidx]
        s0, ypm = self.s0[pidx], self.ypm[pidx]
        P = len(pidx)
        cm = (torch.as_tensor(M, dtype=torch.float32) @ self.G).clamp(max=1)  # (C, width)
        out = {g: torch.zeros(len(M), P) for g in GAMES}
        step = max(1, chunk_rows // P)
        for i in range(0, len(M), step):
            c = cm[i:i + step]
            h = Hp[None] + c[:, None, :] * (H - Hp)[None]
            C = h.shape[0]
            z = self.net.from_level(self.l, h.reshape(C * P, -1), xb.repeat(C, 1))
            v = game_vals(z, s0.repeat(C, 1), ypm.repeat(C, 1), self.eps)
            for g in GAMES:
                out[g][i:i + C] = v[g].reshape(C, P)
        return out

    def path_grads(self, ts, games=('soft', 'pay'), pidx=None):
        """Per-pair group attributions (P, n) = sum_c D_c * dv/dh_c at h_t, for each t
        in ts (scalar or per-pair tensor). Returns dict game -> list over ts, and v(h_t)."""
        pidx = np.arange(self.P) if pidx is None else pidx
        H, Hp, xb = self.H[pidx], self.Hp[pidx], self.xb[pidx]
        D = H - Hp
        out = {g: [] for g in games}
        vt = {g: [] for g in games}
        for t in ts:
            tt = t if torch.is_tensor(t) else torch.full((len(pidx), 1), float(t))
            h = (Hp + tt * D).detach().requires_grad_(True)
            z = self.net.from_level(self.l, h, xb)
            v = game_vals(z, self.s0[pidx], self.ypm[pidx], self.eps)
            for gi, g in enumerate(games):
                gr, = torch.autograd.grad(v[g].sum(), h, retain_graph=gi < len(games) - 1)
                out[g].append(((gr * D) @ self.G.t()).detach())
                vt[g].append(v[g].mean().item())
        return out, vt


# ---------------------------------------------------------------- aggregation helpers
def popcounts(n):
    return np.array([bin(s).count('1') for s in range(1 << n)])


def shapley_table(vals, n):
    return gen.shapley_from_table(np.asarray(vals, float), n)


def loo_marg(vals, n):
    return gen.loo_marg(np.asarray(vals, float), n)


def merged_shapley(vals, n, merge):
    """Exact Shapley when the players in `merge` act as one joint player. Returns
    (phi over the remaining players + joint player last, index list)."""
    others = [k for k in range(n) if k not in merge]
    m = len(others) + 1
    tab = np.zeros(1 << m)
    jm = sum(1 << k for k in merge)
    for s in range(1 << m):
        full = 0
        for i, k in enumerate(others):
            if (s >> i) & 1:
                full |= 1 << k
        if (s >> (m - 1)) & 1:
            full |= jm
        tab[s] = vals[full]
    return gen.shapley_from_table(tab, m), others


def sampled_shapley_from_table(T, n, K, pidx, rng):
    """Emulate K random orders per pair using the per-pair table T (2^n, P)."""
    P = len(pidx)
    phi = np.zeros((P, n))
    for _ in range(K):
        perms = np.argsort(rng.random((P, n)), axis=1)
        mask = np.zeros(P, np.int64)
        prev = T[0, pidx]
        for j in range(n):
            k = perms[:, j]
            mask = mask | (1 << k)
            cur = T[mask, pidx]
            phi[np.arange(P), k] += cur - prev
            prev = cur
    return phi / K


@torch.no_grad()
def sampled_shapley_forward(game, K, pidx, rng, games=GAMES):
    """K random orders per pair with real forwards (n > 12). Returns game -> (P, n)."""
    n, P = game.n, len(pidx)
    phi = {g: np.zeros((P, n)) for g in games}
    for _ in range(K):
        perms = np.argsort(rng.random((P, n)), axis=1)
        # coalition masks along each pair's order: (n+1, P, n)
        M = np.zeros((n + 1, P, n), np.float32)
        for j in range(n):
            M[j + 1] = M[j]
            M[j + 1, np.arange(P), perms[:, j]] = 1
        # evaluate row by row: rows = (step j, pair p)
        vals = {g: np.zeros((n + 1, P)) for g in games}
        H, Hp, xb = game.H[pidx], game.Hp[pidx], game.xb[pidx]
        for j in range(n + 1):
            cm = (torch.as_tensor(M[j]) @ game.G).clamp(max=1)
            z = game.net.from_level(game.l, Hp + cm * (H - Hp), xb)
            v = game_vals(z, game.s0[pidx], game.ypm[pidx], game.eps)
            for g in games:
                vals[g][j] = v[g].numpy()
        for g in games:
            for j in range(n):
                phi[g][np.arange(P), perms[:, j]] += vals[g][j + 1] - vals[g][j]
    return {g: phi[g] / K for g in games}


def mi_1bit(hall, groups, Yhat, nthr=33):
    """Exact I(best single-threshold readout of the group; Yhat) over all inputs.
    Directions: each channel and the group's top PCA direction."""
    out = []
    H = hall.numpy()
    for g in groups:
        A = H[:, g]
        dirs = [A[:, j] for j in range(A.shape[1])]
        Ac = A - A.mean(0)
        if A.shape[1] > 1 and np.abs(Ac).sum() > 0:
            _, _, vt = np.linalg.svd(Ac, full_matrices=False)
            dirs.append(Ac @ vt[0])
        best = 0.0
        for d in dirs:
            for thr in np.unique(np.quantile(d, np.linspace(0.02, 0.98, nthr))):
                b = (d > thr).astype(np.int8)
                if b.min() == b.max():
                    continue
                best = max(best, gen.mi_bits(b[None], Yhat))
        out.append(best)
    return np.array(out)


def vinfo_logistic(hall, groups, Y, seed=0, iters=60):
    """Usable information (Xu et al. 2020): held-out log-score gain (nats, summed over
    outputs) of a logistic reader on the group alone over the prior."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Y))
    tr, te = idx[: 3 * len(Y) // 4], idx[3 * len(Y) // 4:]
    Yt = torch.tensor(Y, dtype=torch.float32)
    prior = Yt[tr].mean(0).clamp(1e-4, 1 - 1e-4)
    base = (Yt[te] * torch.log(prior) + (1 - Yt[te]) * torch.log(1 - prior)).sum(1).mean()
    out = []
    for g in groups:
        A = hall[:, g]
        A = (A - A[tr].mean(0)) / (A[tr].std(0) + 1e-6)
        lin = torch.nn.Linear(A.shape[1], Y.shape[1])
        opt = torch.optim.LBFGS(lin.parameters(), max_iter=iters, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(lin(A[tr]), Yt[tr]) + 1e-3 * (lin.weight ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            ll = -F.binary_cross_entropy_with_logits(lin(A[te]), Yt[te], reduction='none').sum(1).mean()
        out.append(float(ll - base))
    return np.array(out)


def wnorm(net, l, groups):
    if l >= net.nlev:
        return np.zeros(len(groups))
    Wn = net.lins[l].weight.detach()[:, :net.widths[l - 1]]
    return np.array([float(Wn[:, g].norm()) for g in groups])


# ---------------------------------------------------------------- metrics
def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float('nan')
    return float(spearmanr(a, b).correlation)


def ktau(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float('nan')
    return float(kendalltau(a, b).statistic)


def top1(rule, ref, tol=0.01):
    rule, ref = np.asarray(rule, float), np.asarray(ref, float)
    return float(int(np.argmax(rule)) in set(np.flatnonzero(ref >= ref.max() - tol)))


def bern_expect(vals, n, t):
    pc = popcounts(n)
    w = t ** pc * (1 - t) ** (n - pc)
    return float((w * vals).sum())


# ---------------------------------------------------------------- one level, exact reference
AS_M = (5, 10, 20, 50)
SHAP_K = (1, 2, 3, 4, 6, 8, 16, 32, 64)
PSUB = (128, 512, 2048)


def level_analysis(net, l, Xbits, Y, groups, seed=0, eps=None, ladder=True, extra=True, prog=None):
    """All rules at one level with the exact coalition table as reference (n <= 12)."""
    game = LevelGame(net, l, Xbits, Y, groups, seed=seed, eps=eps)
    n, P = game.n, game.P
    rng = np.random.default_rng(seed + 7)
    M = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.float32)
    T = {g: v.numpy().astype(np.float64) for g, v in game.eval_masks(M).items()}  # (2^n, P)
    vbar = {g: T[g].mean(1) for g in GAMES}
    full = (1 << n) - 1
    R = {}  # rule name -> scores (n,)
    ref = {}
    for g in GAMES:
        ref[g] = shapley_table(vbar[g], n)
        loo, marg = loo_marg(vbar[g], n)
        R[f'{g.upper()}-LOO'] = loo
        R[f'{g.upper()}-MARG'] = marg
        R[f'{g.upper()}-SHAP-exact'] = ref[g]
    effgap = {g: float(ref[g].sum() - (vbar[g][full] - vbar[g][0])) for g in GAMES}
    # mean ablation LOO (flip game)
    hmean = game.hall.mean(0)
    with torch.no_grad():
        rows = []
        for k in range(n):
            h = game.H.clone()
            h[:, torch.as_tensor(groups[k])] = hmean[torch.as_tensor(groups[k])]
            z = net.from_level(l, h, game.xb)
            rows.append(float(game_vals(z, game.s0, game.ypm, game.eps)['flip'].mean()))
    R['FLIP-LOO-mean'] = vbar['flip'][full] - np.array(rows)
    # AS / IG, midpoint rule, all m
    as_pp = {}
    comp = {}
    vpath = {}
    for m in AS_M:
        ts = [(k + 0.5) / m for k in range(m)]
        gr, _ = game.path_grads(ts)
        for g in ('soft', 'pay'):
            A = torch.stack(gr[g]).mean(0).numpy().astype(np.float64)  # (P, n)
            as_pp[(g, m)] = A
            R[f'AS-{g.upper()}-m{m}'] = A.mean(0)
            dv = T[g][full] - T[g][0]
            comp[f'{g}-m{m}'] = float(np.abs(A.sum(1) - dv).sum() / max(np.abs(dv).sum(), 1e-12))
    # expected gradients: 1 or 8 random t per pair
    for kk in (1, 8):
        ts = [torch.rand(P, 1, generator=torch.Generator().manual_seed(seed * 100 + j)) for j in range(kk)]
        gr, _ = game.path_grads(ts)
        for g in ('soft', 'pay'):
            A = torch.stack(gr[g]).mean(0).numpy().astype(np.float64)
            as_pp[(g, f'eg{kk}')] = A
            R[f'EG-{g.upper()}-{kk}'] = A.mean(0)
    # GRADxACT (first-order LOO at h(x))
    gr, _ = game.path_grads([1.0])
    for g in ('soft', 'pay'):
        A = gr[g][0].numpy().astype(np.float64)
        as_pp[(g, 'gxa')] = A
        R[f'GRADxACT-{g.upper()}'] = A.mean(0)
    # path diagnostic D(t)
    _, vt = game.path_grads([0.25, 0.5, 0.75])
    Dt = {g: [abs(vt[g][i] - bern_expect(vbar[g], n, t)) / max(abs(vbar[g][full] - vbar[g][0]), 1e-12)
              for i, t in enumerate((0.25, 0.5, 0.75))] for g in ('soft', 'pay')}
    # sampled Shapley ladders (emulated from the per-pair table)
    ladder_rows = []
    if ladder:
        for g in GAMES:
            for K in SHAP_K:
                phi = sampled_shapley_from_table(T[g], n, K, np.arange(P), rng)
                R_K = phi.mean(0)
                R[f'{g.upper()}-SHAP-K{K}'] = R_K
                for Ps in PSUB:
                    est = phi[:Ps].mean(0)
                    ladder_rows.append(dict(method=f'{g.upper()}-SHAP', param=K, P=Ps,
                                            cost_pair=K * (n - 1) + 2, est=est.tolist()))
        for (g, m), A in as_pp.items():
            cp = 3 * m if isinstance(m, int) else (3 if m in ('gxa', 'eg1') else 24)
            name = {'gxa': 'GRADxACT', 'eg1': 'EG', 'eg8': 'EG'}.get(m, 'AS') + '-' + g.upper()
            par = m if isinstance(m, int) else (1 if m in ('gxa', 'eg1') else 8)
            for Ps in PSUB:
                ladder_rows.append(dict(method=name, param=par, P=Ps, cost_pair=cp, est=A[:Ps].mean(0).tolist()))
        for g in GAMES:
            for Ps in PSUB:
                loo = T[g][full, :Ps].mean() - T[g][[full ^ (1 << k) for k in range(n)], :Ps].mean(1)
                ladder_rows.append(dict(method=f'{g.upper()}-LOO', param=0, P=Ps, cost_pair=n + 1, est=loo.tolist()))
    if extra:
        Yhat = (game.zall > 0).numpy().astype(np.int8).T
        R['MI-1BIT'] = mi_1bit(game.hall, groups, Yhat)
        R['VINFO-LOGISTIC'] = vinfo_logistic(game.hall, groups, Y)
        R['WNORM'] = wnorm(net, l, groups)
    R['RANDOM'] = rng.random(n)
    out = dict(n=n, P=P, eff_gap=effgap, completeness=comp, Dt=Dt,
               vfull={g: float(vbar[g][full]) for g in GAMES}, vempty={g: float(vbar[g][0]) for g in GAMES},
               rules={k: np.asarray(v, float).tolist() for k, v in R.items()},
               ref={g: ref[g].tolist() for g in GAMES}, ladder=ladder_rows)
    out['_T'] = T  # kept in memory for plant analysis (not serialized)
    out['_pairs'] = (game.xi, game.xp)
    return out


def plant_analysis(res, plants):
    """Triplicate copies summed vs joint, with the exact table."""
    T = res['_T']
    n = res['n']
    copies = plants.get('copies') or []
    if len(copies) != 3:
        return None
    out = {}
    for g in GAMES:
        vbar = T[g].mean(1)
        phi_j, others = merged_shapley(vbar, n, copies)
        full = (1 << n) - 1
        jm = sum(1 << k for k in copies)
        loo_joint = vbar[full] - vbar[full ^ jm]
        out[g] = dict(joint_shap=float(phi_j[-1]), others_shap=phi_j[:-1].tolist(), others=others,
                      summed_shap=float(sum(res['ref'][g][k] for k in copies)),
                      copies_shap=[res['ref'][g][k] for k in copies],
                      joint_loo=float(loo_joint))
    return out


# ---------------------------------------------------------------- stage 2 (natural nets, n = 24 sign codes)
def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    po = (a == b).mean()
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def code_alignment(net, prog, X, thr=0.95):
    """Cohen's kappa of each sign code (h_2 > 0) with the balanced program gates (up to negation)."""
    V = gen.values(prog, X)
    with torch.no_grad():
        _, hs = net(xin(X), want=True)
    codes = (hs[1] > 0).numpy().T.astype(np.int8)          # (24, N)
    gates = []
    for l in range(1, len(prog['levels']) - 1):
        for k in range(V[l].shape[0]):
            p = V[l][k].mean()
            if 0.1 < p < 0.9:
                gates.append((l, k, V[l][k]))
    K = np.zeros((len(codes), len(gates)))
    for i, c in enumerate(codes):
        if c.min() == c.max():
            continue
        for j, (_, _, g) in enumerate(gates):
            K[i, j] = max(kappa(c, g), kappa(c, 1 - g))
    Kx = np.array([[max(kappa(c, X[:, j]), kappa(c, 1 - X[:, j])) if c.min() != c.max() else 0
                    for j in range(X.shape[1])] for c in codes])
    best = K.max(1) if len(gates) else np.zeros(len(codes))
    return dict(n_gates=len(gates), code_best_kappa=best.round(4).tolist(),
                share_codes_aligned=float((best >= thr).mean()),
                share_gates_recovered=float((K.max(0) >= thr).mean()) if len(gates) else 0.0,
                share_codes_input=float((Kx.max(1) >= thr).mean()),
                code_balance=codes.mean(1).round(3).tolist())


def curves(game, order, metric):
    """Insertion: keep top-k from h(x); deletion: resample top-k. Returns per-k values."""
    n = game.n
    ins = np.zeros((n + 1, n), np.float32)
    dele = np.ones((n + 1, n), np.float32)
    for k in range(1, n + 1):
        ins[k] = ins[k - 1]; ins[k, order[k - 1]] = 1
        dele[k] = dele[k - 1]; dele[k, order[k - 1]] = 0
    vi = game.eval_masks(ins)[metric].mean(1).numpy()
    vd = game.eval_masks(dele)[metric].mean(1).numpy()
    return vi, vd


def greedy(game, metric, mode):
    n = game.n
    S = np.zeros(n, np.float32) if mode == 'ins' else np.ones(n, np.float32)
    vals = [float(game.eval_masks(S[None])[metric].mean())]
    left = list(range(n))
    while left:
        M = np.repeat(S[None], len(left), 0)
        for i, k in enumerate(left):
            M[i, k] = 1 if mode == 'ins' else 0
        v = game.eval_masks(M)[metric].mean(1).numpy()
        i = int(np.argmax(v)) if mode == 'ins' else int(np.argmin(v))
        S = M[i]
        vals.append(float(v[i]))
        left.pop(i)
    return np.array(vals)


CURVE_RULES = ['AS-SOFT-m5', 'AS-SOFT-m20', 'AS-SOFT-m50', 'AS-PAY-m20', 'AS-PAY-m50', 'EG-SOFT-8',
               'GRADxACT-SOFT', 'GRADxACT-PAY', 'FLIP-LOO', 'FLIP-LOO-mean', 'FLIP-MARG', 'PAY-LOO',
               'FLIP-SHAP-ref', 'SOFT-SHAP-ref', 'PAY-SHAP-ref', 'FLIP-SHAP-K2', 'FLIP-SHAP-K8',
               'MI-1BIT', 'VINFO-LOGISTIC', 'WNORM', 'RANDOM']


def r2_analysis(net, prog, X, Y, seed=0, P=1024, Klad=64, Kref=256, nrand=16):
    width = net.widths[1]
    groups = [np.array([k]) for k in range(width)]
    game = LevelGame(net, 2, X, Y, groups, npair=P, seed=seed)
    n = game.n
    rng = np.random.default_rng(seed + 11)
    R = {}
    M = np.concatenate([np.ones((1, n)), np.zeros((1, n)), 1 - np.eye(n), np.eye(n)]).astype(np.float32)
    v = {g: x.numpy().mean(1) for g, x in game.eval_masks(M).items()}
    for g in GAMES:
        R[f'{g.upper()}-LOO'] = v[g][0] - v[g][2:2 + n]
        R[f'{g.upper()}-MARG'] = v[g][2 + n:] - v[g][1]
    hmean = game.hall.mean(0)
    with torch.no_grad():
        rows = []
        for k in range(n):
            h = game.H.clone()
            h[:, k] = hmean[k]
            z = net.from_level(2, h, game.xb)
            rows.append(float(game_vals(z, game.s0, game.ypm, game.eps)['flip'].mean()))
    R['FLIP-LOO-mean'] = v['flip'][0] - np.array(rows)
    comp = {}
    for m in AS_M:
        gr, _ = game.path_grads([(k + 0.5) / m for k in range(m)])
        for g in ('soft', 'pay'):
            A = torch.stack(gr[g]).mean(0).numpy().astype(np.float64)
            R[f'AS-{g.upper()}-m{m}'] = A.mean(0)
            dv = (v[g][0] - v[g][1])
            comp[f'{g}-m{m}'] = float(abs(A.mean(0).sum() - dv) / max(abs(dv), 1e-12))
    for kk in (1, 8):
        ts = [torch.rand(P, 1, generator=torch.Generator().manual_seed(seed * 100 + j)) for j in range(kk)]
        gr, _ = game.path_grads(ts)
        for g in ('soft', 'pay'):
            R[f'EG-{g.upper()}-{kk}'] = torch.stack(gr[g]).mean(0).numpy().mean(0)
    gr, _ = game.path_grads([1.0])
    for g in ('soft', 'pay'):
        R[f'GRADxACT-{g.upper()}'] = gr[g][0].numpy().mean(0)
    # sampled Shapley with real forwards: ladder orders then independent reference orders
    lad = {g: np.zeros((P, n)) for g in GAMES}
    ref = {g: np.zeros((P, n)) for g in GAMES}
    ladder = {g: [] for g in GAMES}
    Ks = (1, 2, 4, 8, 16, 32, 64)
    for k in range(Klad + Kref):
        phi = sampled_shapley_forward(game, 1, np.arange(P), rng)
        for g in GAMES:
            (lad if k < Klad else ref)[g] += phi[g]
        if k + 1 in Ks and k < Klad:
            for g in GAMES:
                ladder[g].append(dict(K=k + 1, cost_pair=(k + 1) * (n - 1) + 2, est=(lad[g] / (k + 1)).mean(0).tolist()))
    for g in GAMES:
        R[f'{g.upper()}-SHAP-ref'] = (ref[g] / Kref).mean(0)
        for d in ladder[g]:
            if d['K'] in (2, 8):
                R[f'{g.upper()}-SHAP-K{d["K"]}'] = np.array(d['est'])
    Yhat = (game.zall > 0).numpy().astype(np.int8).T
    R['MI-1BIT'] = mi_1bit(game.hall, groups, Yhat)
    R['VINFO-LOGISTIC'] = vinfo_logistic(game.hall, groups, Y)
    R['WNORM'] = wnorm(net, 2, groups)
    R['RANDOM'] = rng.random(n)
    # curves on fresh pairs
    cg = LevelGame(net, 2, X, Y, groups, npair=P, seed=seed + 1000)
    cur = {}
    base = {}
    for metric in ('flip', 'pay'):
        gi, gd = greedy(cg, metric, 'ins'), greedy(cg, metric, 'del')
        ri, rd = [], []
        for _ in range(nrand):
            a, b = curves(cg, rng.permutation(n), metric)
            ri.append(a); rd.append(b)
        ri, rd = np.mean(ri, 0), np.mean(rd, 0)
        vfull = gi[-1] if False else float(cg.eval_masks(np.ones((1, n), np.float32))[metric].mean())
        auc = lambda c: float(np.mean(c))
        base[metric] = dict(greedy_ins=gi.tolist(), greedy_del=gd.tolist(), rand_ins=ri.tolist(), rand_del=rd.tolist())
        for rule in CURVE_RULES:
            if rule not in R:
                continue
            order = np.argsort(-np.asarray(R[rule]))
            ci, cd = curves(cg, order, metric)
            ni = (auc(ci) - auc(ri)) / max(auc(gi) - auc(ri), 1e-12)
            nd = (auc(rd) - auc(cd)) / max(auc(rd) - auc(gd), 1e-12)  # deletion: lower values are better
            cur.setdefault(rule, {})[f'ins_{metric}'] = float(min(ni, 1.0))
            cur[rule][f'del_{metric}'] = float(min(nd, 1.0))
            cur[rule][f'raw_ins_{metric}'] = ci.tolist()
            cur[rule][f'raw_del_{metric}'] = cd.tolist()
    for rule in cur:
        cur[rule]['mean4'] = float(np.mean([cur[rule][k] for k in ('ins_flip', 'del_flip', 'ins_pay', 'del_pay')]))
    return dict(n=n, P=P, completeness=comp, v_full={g: float(v[g][0]) for g in GAMES},
                v_empty={g: float(v[g][1]) for g in GAMES}, rules={k: np.asarray(x, float).tolist() for k, x in R.items()},
                shap_ladder=ladder, curves=cur, curve_base=base)
