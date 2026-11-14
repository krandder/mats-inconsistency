"""Interp step 1 (Locate): ridge probes, attention statistics, symmetry checks. See PREREG.md.

usage: python locate.py RUN_DIR TAG OUT_JSON      (RUN_DIR relative to ../tiny-pfn)
"""
import json, math, os, sys
import numpy as np
import torch

TP = os.environ.get('TINYPFN_DIR', '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, TP)
import world as W
import pfn as P

torch.set_num_threads(1)
S, Q, QL = W.S, W.Q, W.QL
T = S * Q + 1
QI = T - 1
NTR = 12000
SITES = ('emb', 'attn1', 'mlp1', 'attn2', 'mlp2')
LAMS = np.logspace(-3, 4, 15)


# ---------------------------------------------------------------- forward with caching
def forward_cache(model, X, m, bs=2000):
    """Returns sites dict name -> (n, T, d) float32, attn list per layer (n, H, T, T), head outputs."""
    model.eval()
    acts = {k: [] for k in SITES}; attns = [[] for _ in model.blocks]; heads = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            Xb, mb = torch.from_numpy(X[i:i + bs]), torch.from_numpy(m[i:i + bs])
            B = len(Xb)
            ctx = mb.clone(); ctx[:, QI] = False
            allow = ctx[:, None, :].expand(B, T, T).clone(); allow[:, QI, QI] = True
            eye = torch.eye(T, dtype=torch.bool)[None]; allow = allow | (eye & ~mb[:, :, None])
            x = model.inp(Xb); acts['emb'].append(x.numpy())
            for li, blk in enumerate(model.blocks):
                d = x.shape[-1]
                q, k, v = blk.qkv(blk.ln1(x)).split(d, -1)
                sp = lambda t: t.view(B, T, blk.h, d // blk.h).transpose(1, 2)
                q, k, v = sp(q), sp(k), sp(v)
                s = (q @ k.transpose(-1, -2) / math.sqrt(d // blk.h)).masked_fill(~allow[:, None], float('-inf'))
                a = s.softmax(-1)
                x = x + blk.o((a @ v).transpose(1, 2).reshape(B, T, d))
                acts[f'attn{li + 1}'].append(x.numpy()); attns[li].append(a.numpy())
                if not blk.attn_only:
                    x = x + blk.mlp(blk.ln2(x))
                acts[f'mlp{li + 1}'].append(x.numpy())
            y = model.out(model.ln(x[:, QI]))
            heads.append(y.numpy())
            ref = model(Xb, mb)
            K = model.mixk
            got = (y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K])
            assert max(float((g - r).abs().max()) for g, r in zip(got, ref)) < 1e-5
    acts = {k: np.concatenate(v) for k, v in acts.items()}
    attns = [np.concatenate(a) for a in attns]
    y = np.concatenate(heads); K = model.mixk
    head = (y[:, K:2 * K], np.clip(y[:, 2 * K:], -6, 3), y[:, :K])
    return acts, attns, head


# ---------------------------------------------------------------- targets
def targets(D, model_mean):
    o, y = D['o'], D['y']
    n = len(y)
    R = o[:, :, :QL] - y[:, None, :QL]                          # (n, S, QL)
    ss = D['ss_resid']                                          # (n, S)
    pl = D['src_post_lab']                                      # (n, S, 3)
    vlev = W.VLEV
    wb = D['w_bayes']                                           # (n, 27)
    comb = W.COMB
    pb = np.stack([np.stack([wb[:, comb[:, s] == j].sum(1) for j in range(3)], 1) for s in range(S)], 1)  # (n,S,3)
    El = (pl / vlev).sum(2); Eb = (pb / vlev).sum(2)
    VC = vlev[comb]; PREC = 1 + (1 / VC).sum(1)
    beta = np.einsum('nc,cs->ns', wb, (1 / VC) / PREC[:, None])  # (n, S)
    assert np.allclose((beta * o[:, :, QL]).sum(1), D['mean_bayes'], atol=1e-8)
    H = -(wb * np.log(np.maximum(wb, 1e-300))).sum(1)
    src = dict(SS=ss, logSS=np.log(ss), p_lab=pl, E_lab=El, p_bayes=pb, E_bayes=Eb, beta=beta)
    lab_tok = dict(r=R, r2=R ** 2)                              # (n, S, QL)
    qt = dict(mean_bayes=D['mean_bayes'], logvar_bayes=np.log(D['var_bayes']), var_bayes=D['var_bayes'],
              entropy_w27=H, model_mean=model_mean, mean_lab=D['mean_lab'])
    qq_extra = dict(mean_bayes=D['mean_bayes'], logvar_bayes=np.log(D['var_bayes']))
    return src, lab_tok, qt, qq_extra, beta, El


def lab_idx():
    return np.array([s * Q + q for s in range(S) for q in range(QL)])   # order (s, q)


def qq_idx():
    return np.array([s * Q + QL for s in range(S)])


def stack_feats(A, idx):
    """A (n, T, d) -> (n, len(idx), d)."""
    return A[:, idx]


# ---------------------------------------------------------------- ridge with grouped CV
def _fit(X, Y, lam):
    """X standardised (n,d), Y centred (n,k); returns W (d,k) for each lam: list."""
    G = X.T @ X; b = X.T @ Y
    ev, U = np.linalg.eigh(G)
    Ub = U.T @ b
    return [U @ (Ub / (ev + l * len(X))[:, None]) for l in lam]


def r2(yt, yp):
    sst = ((yt - yt.mean(0)) ** 2).sum(0)
    return 1 - ((yt - yp) ** 2).sum(0) / np.maximum(sst, 1e-300)


def ridge_probe(Xtr, Ytr, Xte, Yte, gtr, nfold=5):
    """Xtr (ntr_samples, d), Ytr (ntr_samples, k); gtr episode index per sample. Returns held-out R^2 per column."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xs = (Xtr - mu) / sd; Xt = (Xte - mu) / sd
    folds = gtr % nfold
    cv = np.zeros((len(LAMS), Ytr.shape[1]))
    for f in range(nfold):
        tr, va = folds != f, folds == f
        ym = Ytr[tr].mean(0)
        Ws = _fit(Xs[tr], Ytr[tr] - ym, LAMS)
        for i, Wl in enumerate(Ws):
            cv[i] += r2(Ytr[va], Xs[va] @ Wl + ym) / nfold
    best = cv.argmax(0)                                          # per column
    ym = Ytr.mean(0)
    Ws = _fit(Xs, Ytr - ym, LAMS)
    pred = np.stack([Xt @ Ws[best[j]][:, j] + ym[j] for j in range(Ytr.shape[1])], 1)
    return r2(Yte, pred), LAMS[best], (mu, sd, np.stack([Ws[best[j]][:, j] for j in range(Ytr.shape[1])], 1), ym)


def apply_probe(pr, X):
    mu, sd, Wm, ym = pr
    return ((X - mu) / sd) @ Wm + ym


def probe_set(F, Yfull, perm):
    """F (n, ntok, d) features, Yfull (n, ntok, k) targets. Returns R^2 (k,), R^2 shuffled (k,), probe."""
    n, nt, d = F.shape
    k = Yfull.shape[-1]
    tr = np.arange(NTR); te = np.arange(NTR, n)
    g = np.repeat(np.arange(NTR), nt)
    Xtr = F[tr].reshape(-1, d).astype(np.float64); Xte = F[te].reshape(-1, d).astype(np.float64)
    Ytr = Yfull[tr].reshape(-1, k); Yte = Yfull[te].reshape(-1, k)
    R, lam, pr = ridge_probe(Xtr, Ytr, Xte, Yte, g)
    Ysh = Yfull[tr][perm].reshape(-1, k)
    Rs, _, _ = ridge_probe(Xtr, Ysh, Xte, Yte, g)
    return R, Rs, pr


def per_source_to_tokens(v, kind):
    """v (n, S) or (n, S, k) -> (n, ntok, k) for LAB (12 tokens, order (s,q)) or QQ (3 tokens)."""
    if v.ndim == 2: v = v[:, :, None]
    if kind == 'LAB':
        return np.repeat(v, QL, axis=1)
    return v


def all_probes(acts, src, lab_tok, qt, qq_extra, perm):
    res = {}
    li, qi = lab_idx(), qq_idx()
    for site in SITES:
        A = acts[site]
        FL, FQ, FT = A[:, li], A[:, qi], A[:, QI:QI + 1]
        for name, v in src.items():
            for kind, F in (('LAB', FL), ('QQ', FQ)):
                Y = per_source_to_tokens(v, kind)
                R, Rs, _ = probe_set(F, Y, perm)
                res[f'{name}|{kind}|{site}'] = (R.tolist(), Rs.tolist())
        for name, v in lab_tok.items():
            Y = v.reshape(len(v), -1)[:, :, None]                # order (s, q) matches lab_idx
            R, Rs, _ = probe_set(FL, Y, perm)
            res[f'{name}|LAB|{site}'] = (R.tolist(), Rs.tolist())
        for name, v in qt.items():
            R, Rs, _ = probe_set(FT, v[:, None, None], perm)
            res[f'{name}|QT|{site}'] = (R.tolist(), Rs.tolist())
        for name, v in qq_extra.items():
            Y = np.repeat(v[:, None, None], S, 1)
            R, Rs, _ = probe_set(FQ, Y, perm)
            res[f'{name}|QQ|{site}'] = (R.tolist(), Rs.tolist())
    return res


# ---------------------------------------------------------------- attention statistics
def attn_stats(attns, D, beta, El):
    te = np.arange(NTR, len(beta))
    out = {}
    R2 = (D['o'][:, :, :QL] - D['y'][:, None, :QL]) ** 2
    for li, A in enumerate(attns):
        A = A[te]                                                # (n, H, T, T)
        H = A.shape[1]
        for h in range(H):
            a = A[:, h]
            key = f'L{li + 1}H{h}'
            st = {}
            for frm in ('LAB', 'QQ'):
                own_lab, own_qq, oth_lab, oth_qq, slf = [], [], [], [], []
                same_q_oth = []
                rcorr_x, rcorr_y = [], []
                for s in range(S):
                    qs = range(QL) if frm == 'LAB' else [QL]
                    for q in qs:
                        i = s * Q + q
                        row = a[:, i]
                        slf.append(row[:, i])
                        ol = [s * Q + qq for qq in range(QL) if s * Q + qq != i]
                        own_lab.append(row[:, ol].sum(1) / len(ol))
                        own_qq.append(row[:, s * Q + QL] if frm == 'LAB' else np.full(len(row), np.nan))
                        others = [t for t in range(S) if t != s]
                        oth_lab.append(row[:, [t * Q + qq for t in others for qq in range(QL)]].sum(1) / (len(others) * QL))
                        oth_qq.append(row[:, [t * Q + QL for t in others]].sum(1) / len(others))
                        if frm == 'LAB':
                            same_q_oth.append(row[:, [t * Q + q for t in others]].sum(1) / len(others))
                        for qq in range(QL):
                            j = s * Q + qq
                            if j != i:
                                rcorr_x.append(row[:, j]); rcorr_y.append(R2[te, s, qq])
                m = lambda L: float(np.nanmean(np.concatenate(L)))
                st[frm] = dict(self=m(slf), own_lab_per_tok=m(own_lab), own_qq=m(own_qq) if frm == 'LAB' else None,
                               oth_lab_per_tok=m(oth_lab), oth_qq_per_tok=m(oth_qq),
                               same_q_oth_lab_per_tok=m(same_q_oth) if frm == 'LAB' else None,
                               ratio_lab=m(own_lab) / m(oth_lab),
                               corr_attn_r2=float(np.corrcoef(np.concatenate(rcorr_x), np.concatenate(rcorr_y))[0, 1]))
                if frm == 'LAB':
                    st[frm]['ratio_qq'] = m(own_qq) / m(oth_qq)
            row = a[:, QI]
            lab_m = np.stack([row[:, [s * Q + q for q in range(QL)]].sum(1) for s in range(S)], 1)  # (n, S)
            qq_m = np.stack([row[:, s * Q + QL] for s in range(S)], 1)
            all_m = lab_m + qq_m
            b, e = beta[te], El[te]
            cr = lambda x, y: float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
            st['QT'] = dict(self=float(row[:, QI].mean()), lab=float(lab_m.sum(1).mean()), qq=float(qq_m.sum(1).mean()),
                            r_all_beta=cr(all_m, b), r_qq_beta=cr(qq_m, b), r_lab_beta=cr(lab_m, b),
                            r_all_E=cr(all_m, e), r_qq_E=cr(qq_m, e),
                            r_qqnorm_beta=cr(qq_m / qq_m.sum(1, keepdims=True), b / b.sum(1, keepdims=True)))
            # QQ -> other sources' QQ tokens: does a source's QQ token weight other QQ tokens by their beta?
            xs, ys = [], []
            for s in range(S):
                for t in range(S):
                    if t != s:
                        xs.append(a[:, s * Q + QL, t * Q + QL]); ys.append(beta[te, t])
            st['QQ']['r_qq_to_oth_qq_beta'] = cr(np.concatenate(xs), np.concatenate(ys))
            out[key] = st
    return out


# ---------------------------------------------------------------- symmetry
def relabel(X, ds, perm_src):
    """Move identity vectors between sources: token (s, q) gets the identity vector of source perm_src[s]."""
    X2 = X.copy()
    for s in range(S):
        for q in range(Q):
            X2[:, s * Q + q, :ds] = X[:, perm_src[s] * Q + q, :ds]
    return X2


def symmetry(model, X, m, ds, acts, head, probes, src, first_sites):
    out = {}
    # (i) token permutation equivariance
    perm_src = np.array([1, 2, 0])
    order = np.concatenate([np.arange(perm_src[s] * Q, perm_src[s] * Q + Q) for s in range(S)] + [[QI]])
    Xp = X[:, order]
    a2, _, h2 = forward_cache(model, Xp[:2000], m[:2000])
    out['equivariance_max_abs'] = float(max(np.abs(a2[k] - acts[k][:2000, order]).max() for k in SITES))
    out['equivariance_head_max_abs'] = float(max(np.abs(x - y[:2000]).max() for x, y in zip(h2, head)))
    # (ii) relabel identity vectors among sources (readings stay in place)
    Xr = relabel(X, ds, perm_src)
    ar, _, hr = forward_cache(model, Xr, m)
    mu0, _ = P.head_moments(head); mur, _ = P.head_moments(hr)
    te = np.arange(NTR, len(X))
    out['relabel_mean_abs_dmean'] = float(np.abs(mur - mu0)[te].mean())
    for name, kind, site in first_sites:
        idx = lab_idx() if kind == 'LAB' else qq_idx()
        pr = probes[(name, kind, site)]
        Y = per_source_to_tokens(src[name], kind)[te].reshape(-1)
        p0 = apply_probe(pr, acts[site][te][:, idx].reshape(-1, acts[site].shape[-1]).astype(np.float64))[:, 0]
        p1 = apply_probe(pr, ar[site][te][:, idx].reshape(-1, ar[site].shape[-1]).astype(np.float64))[:, 0]
        # what the token would decode if it followed its identity vector instead of its readings
        Yid = per_source_to_tokens(src[name][:, perm_src], kind)[te].reshape(-1)
        out[f'relabel_{name}_{kind}_{site}'] = dict(r2_orig=float(r2(Y, p0)), r2_relabelled_own=float(r2(Y, p1)),
                                                    r2_relabelled_vs_identity_source=float(r2(Yid, p1)))
    # identity-vector redraw (step-4 diagnostic)
    rng = np.random.default_rng(777)
    n = len(X)
    Xd = X.copy()
    semb = rng.standard_normal((n, S, ds)) / 4; qemb = rng.standard_normal((n, Q, ds)) / 4
    for s in range(S):
        for q in range(Q):
            Xd[:, s * Q + q, :ds] = semb[:, s]; Xd[:, s * Q + q, ds:2 * ds] = qemb[:, q]
    Xd[:, QI, ds:2 * ds] = qemb[:, QL]
    _, _, hd = forward_cache(model, Xd, m)
    mud, _ = P.head_moments(hd)
    return out, mud


def first_positive_site(name, kind, R, RU, RS):
    for site in SITES:
        k = f'{name}|{kind}|{site}'
        r, ru, rs = np.mean(R[k]), np.mean(RU[k]), np.mean(RS[k])
        if r >= 0.5 and r - max(ru, rs) >= 0.10:
            return site
    return None


def main():
    run, tag, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    rd = os.path.join(TP, run)
    meta = json.load(open(os.path.join(rd, 'meta.json')))
    ds = meta['ds']
    D = np.load(os.path.join(TP, 'runs', 'testset.npz' if ds == W.DS else f'testset_ds{ds}.npz'))
    X, m = D['X'], D['valid']
    trained = P.build(meta); trained.load_state_dict(torch.load(os.path.join(rd, 'model.pt')))
    torch.manual_seed(meta['seed']); untrained = P.build(meta)
    perm = np.random.default_rng(0).permutation(NTR)
    result = dict(run=run, tag=tag, seed=meta['seed'], ds=ds)
    store = {}
    for nm, model in (('trained', trained), ('untrained', untrained)):
        acts, attns, head = forward_cache(model, X, m)
        mu, var = P.head_moments(head)
        src, lab_tok, qt, qq_extra, beta, El = targets(D, mu)
        if nm == 'untrained':
            qt['model_mean'] = store['trained_mu']               # same target for both (trained model's mean)
        else:
            store['trained_mu'] = mu
            nll = P.head_nll_np(head, D['ystar'])
            result['nll_gap'] = float((nll - D['nll_bayes']).mean())
        result[f'probes_{nm}'] = all_probes(acts, src, lab_tok, qt, qq_extra, perm)
        result[f'attn_{nm}'] = attn_stats(attns, D, beta, El)
        store[nm] = (model, acts, head, src)
        print(tag, nm, 'done', flush=True)
    # symmetry on trained, using probes refit at the first positive site of this network
    model, acts, head, src = store['trained']
    R = {k: v[0] for k, v in result['probes_trained'].items()}
    RU = {k: v[0] for k, v in result['probes_untrained'].items()}
    RS = {k: v[1] for k, v in result['probes_trained'].items()}
    first = {}
    probes = {}
    li, qi = lab_idx(), qq_idx()
    for name in ('SS', 'logSS', 'E_lab', 'beta'):
        for kind in ('LAB', 'QQ'):
            site = first_positive_site(name, kind, R, RU, RS)
            first[f'{name}|{kind}'] = site
    fs = []
    for key, site in first.items():
        name, kind = key.split('|')
        if site is None: continue
        idx = li if kind == 'LAB' else qi
        _, _, pr = probe_set(acts[site][:, idx], per_source_to_tokens(src[name], kind), perm)
        probes[(name, kind, site)] = pr
        fs.append((name, kind, site))
    sym, mud = symmetry(model, X, m, ds, acts, head, probes, src, fs)
    mu0, _ = P.head_moments(head)
    te = np.arange(NTR, len(X))
    sym['redraw_sd_dmean'] = float(np.std((mud - mu0)[te]) / math.sqrt(2))   # per-draw sd of the identity noise
    sym['sd_model_minus_bayes'] = float(np.std((mu0 - D['mean_bayes'])[te]))
    result['first_sites'] = first
    result['symmetry'] = sym
    json.dump(result, open(outp, 'w'))
    print(tag, 'gap', result['nll_gap'], 'sym', json.dumps(sym)[:400], flush=True)


if __name__ == '__main__':
    main()
