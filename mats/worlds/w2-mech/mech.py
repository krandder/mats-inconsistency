"""W2 mechanism study: shared library (see PREREG.md).

bayes_ext(ep)     exact Bayes extended: latent mixture moments, P(high) per source, beta per token
load_net(name)    trained (best.pt) or untrained (seed-0 init) SetPFN
forward(...)      re-implemented SetPFN forward with full attention, site capture and residual patching
"""
import itertools, math, os, sys
import numpy as np
import torch

WD = '/home/agent/workspaces/mats/work/worlds'
RUNS = os.path.join(WD, 'runs_w2b_1028', 'combined')
sys.path.insert(0, WD)
import world2_market as W   # noqa: E402
import harness as Hn         # noqa: E402

torch.set_num_threads(1)
S2LEV = np.array([W.SIG_L ** 2, W.SIG_H ** 2])


# ------------------------------------------------------------------------------------------------ Bayes
def bayes_ext(ep):
    """Exact Bayes with everything the probes need. Returns dict."""
    H, y, c0, uj, p0, pj, U, q = W._system(ep, 'bayes')
    n, D = H.shape
    K, S = ep.K, ep.S
    unk_list = list(np.flatnonzero(ep.unk))
    used = sorted(set(uj[uj >= 0]) | set(pj[pj >= 0]))
    remap = {u: i for i, u in enumerate(used)}
    Uu = len(used)
    lev = np.array(list(itertools.product((0, 1), repeat=Uu)), np.int64).reshape(2 ** Uu, Uu)
    nh = len(lev)
    nv = np.tile(c0, (nh, 1))                                   # (nh, n) row noise var
    for r in np.flatnonzero(uj >= 0):
        nv[:, r] = c0[r] + S2LEV[lev[:, remap[uj[r]]]]
    P = np.tile(p0, (nh, 1))
    for i in np.flatnonzero(pj >= 0):
        P[:, i] = p0[i] + S2LEV[lev[:, remap[pj[i]]]]
    Wt = 1.0 / nv
    Lam = np.einsum('nd,hn,nf->hdf', H, Wt, H) + np.einsum('hd,de->hde', 1.0 / P, np.eye(D))
    Cov = np.linalg.inv(Lam)
    G = np.einsum('hde,ne,hn->hdn', Cov, H, Wt)                 # (nh, D, n) linear map obs -> mean
    mu = np.einsum('hdn,n->hd', G, y)
    eta = np.einsum('nd,hn,n->hd', H, Wt, y)
    sign, logdet = np.linalg.slogdet(Lam)
    logml = -0.5 * (np.log(nv).sum(1) + np.log(P).sum(1) + logdet + (Wt * y ** 2).sum(1) - (eta * mu).sum(1))
    w = np.exp(logml - logml.max()); w /= w.sum()
    mean = (w[:, None] * mu).sum(0)
    var = (w[:, None] * (np.einsum('hdd->hd', Cov) + mu ** 2)).sum(0) - mean ** 2
    # per-source noise posterior
    pH = np.full(S, np.nan)
    for s in unk_list:
        u = unk_list.index(s)
        if u in remap:
            pH[s] = float((w * lev[:, remap[u]]).sum())
        else:
            pH[s] = 0.5
    Es2 = np.where(ep.unk, np.where(np.isnan(pH), 0.5, pH) * S2LEV[1] + (1 - np.where(np.isnan(pH), 0.5, pH)) * S2LEV[0],
                   ep.sig ** 2)
    bmean, bvar = mean[K:K + S], var[K:K + S]
    # Bayes linear weights on rows for v_q, mapped back to token values
    grow = (w[:, None] * G[:, q, :]).sum(0)                    # (n,)
    beta = grow.copy()
    orig_row = {int(ep.q_ev[t]): t for t in np.flatnonzero(~ep.q_relay)}
    for t in range(n):
        e = int(ep.q_ev[t])
        if ep.q_relay[t] and ep.ev_vis[e]:
            beta[orig_row[e]] -= grow[t]
    vm = mean[:K]
    it = ep.ev_items[ep.q_ev]
    afun = (ep.ev_coef[ep.q_ev] * np.where(it >= 0, vm[np.maximum(it, 0)], 0.0)).sum(1)    # E[a.v] per token
    out = dict(w=w, mean=mean, var=var, pH=pH, Es2=Es2, bmean=bmean, bvar=bvar,
               logprec=-np.log(Es2 + bvar), beta=beta, vfun=afun, resid=ep.y - afun,
               qmean=float((w * mu[:, q]).sum()),
               qvar=float((w * (Cov[:, q, q] + mu[:, q] ** 2)).sum() - (w * mu[:, q]).sum() ** 2))
    return out


# ------------------------------------------------------------------------------------------------ nets
def load_net(name, trained=True, ckpt='best.pt'):
    run = os.path.join(RUNS, name)
    if trained:
        model, world, meta = Hn.load_run(run, ckpt)
        return model.cpu().float().eval()
    import json
    meta = json.load(open(os.path.join(run, 'meta.json')))
    a = meta['args']
    torch.manual_seed(a.get('seed', 0))
    model = Hn.SetPFN(W.FEAT, a['depth'], a['width'], a['heads'], 'real', a['mix'])
    return model.eval()


def site_names(depth):
    s = ['L0']
    for l in range(1, depth + 1):
        s += [f'L{l}.a', f'L{l}']
    return s


@torch.no_grad()
def forward(model, X, M, keep_sites=True, keep_attn=False, patch=None, attn_rows=None, attn_edit=None):
    """X (B, T, F), M (B, T) torch. patch: dict l -> (pos (B, T) bool, states (B, T, d)) applied to the residual after
    block l (l = 0: embedding). attn_rows: if given, (B, T) bool of attention-query rows to keep (saves memory).
    Returns (mean, logsd, pred tuple), sites dict, attn list (B, h, T, T) or None."""
    B, T, _ = X.shape
    ctx = M.clone(); ctx[:, -1] = False
    empty = ~ctx.any(1); ctx[empty, -1] = True
    km = ctx[:, None, None, :]
    x = model.inp(X)
    sites, attn = {}, []

    def pt(l, x):
        if patch is not None and l in patch:
            pos, st = patch[l]
            x = torch.where(pos[..., None], st, x)
        return x
    x = pt(0, x)
    if keep_sites: sites['L0'] = x
    for li, blk in enumerate(model.blocks):
        d = x.shape[-1]; dh = d // blk.h
        q, k, v = blk.qkv(blk.ln1(x)).split(d, -1)
        q, k, v = (t.view(B, T, blk.h, dh).transpose(1, 2) for t in (q, k, v))
        s = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        a = s.masked_fill(~km, float('-inf')).softmax(-1)
        if attn_edit is not None:
            a = attn_edit(li, a)
        if keep_attn: attn.append(a)
        x = x + blk.o((a @ v).transpose(1, 2).reshape(B, T, d))
        if keep_sites: sites[f'L{li + 1}.a'] = x
        x = x + blk.mlp(blk.ln2(x))
        x = pt(li + 1, x)
        if keep_sites: sites[f'L{li + 1}'] = x
    o = model.out(model.ln(x[:, -1]))
    lg, mu, ls = o.split(model.mix, -1)
    lw = lg.log_softmax(-1); ls = ls.clamp(-7, 4)
    w = lw.exp()
    mean = (w * mu).sum(-1)
    var = (w * (torch.exp(2 * ls) + mu ** 2)).sum(-1) - mean ** 2
    return (mean, 0.5 * torch.log(var), (w, mu, torch.exp(2 * ls))), sites, (attn if keep_attn else None)


# ------------------------------------------------------------------------------------------------ data
def c4_set():
    import json
    es = os.path.join(RUNS, 'es_c4_96')
    idx = json.load(open(os.path.join(es, 'index.json')))
    c = [c for c in idx['cells'] if c['policy'] == 'c4' and c['N'] == 96][0]
    d = dict(np.load(os.path.join(es, c['file']), allow_pickle=True))
    world = W.make_world()
    ep = world.sample(np.random.default_rng(c['seed']), c['E'], 96, 'c4')
    assert np.allclose(ep['target'], d['target'])
    return ep, d


def probe_set(E=2000, N=96, seed=4242):
    world = W.make_world()
    return world.sample(np.random.default_rng(seed), E, N, 'dir')


# ------------------------------------------------------------------------------------------------ ridge
LAMS = np.logspace(-3, 4, 15)


def _fit(X, Y, lams):
    G = X.T @ X; b = X.T @ Y
    ev, U = np.linalg.eigh(G)
    Ub = U.T @ b
    return [U @ (Ub / (ev + l * len(X))[:, None]) for l in lams]


def r2(yt, yp):
    sst = ((yt - yt.mean(0)) ** 2).sum(0)
    return 1 - ((yt - yp) ** 2).sum(0) / np.maximum(sst, 1e-300)


def ridge_probe(Xtr, Ytr, Xte, Yte, gtr, nfold=5):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xs = (Xtr - mu) / sd; Xt = (Xte - mu) / sd
    folds = gtr % nfold
    cv = np.zeros((len(LAMS), Ytr.shape[1]))
    for f in range(nfold):
        tr, va = folds != f, folds == f
        ym = Ytr[tr].mean(0)
        Ws = _fit(Xs[tr], Ytr[tr] - ym, LAMS)
        for i, Wl in enumerate(Ws):
            cv[i] += r2(Ytr[va], Xs[va] @ Wl + ym) / nfold
    best = cv.argmax(0)
    ym = Ytr.mean(0)
    Ws = _fit(Xs, Ytr - ym, LAMS)
    pred = np.stack([Xt @ Ws[best[j]][:, j] + ym[j] for j in range(Ytr.shape[1])], 1)
    return r2(Yte, pred)


def ridge_fit(Xtr, Ytr, gtr, nfold=5):
    """Same pipeline as ridge_probe; returns a predictor function."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xs = (Xtr - mu) / sd
    folds = gtr % nfold
    cv = np.zeros((len(LAMS), Ytr.shape[1]))
    for f in range(nfold):
        tr, va = folds != f, folds == f
        ym = Ytr[tr].mean(0)
        Ws = _fit(Xs[tr], Ytr[tr] - ym, LAMS)
        for i, Wl in enumerate(Ws):
            cv[i] += r2(Ytr[va], Xs[va] @ Wl + ym) / nfold
    best = cv.argmax(0)
    ym = Ytr.mean(0)
    Ws = _fit(Xs, Ytr - ym, LAMS)
    Wm = np.stack([Ws[best[j]][:, j] for j in range(Ytr.shape[1])], 1)
    return lambda X: ((X - mu) / sd) @ Wm + ym
