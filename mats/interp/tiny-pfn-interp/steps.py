"""Steps 2-4 (PREREG Amendment 1): intervene, replace, identity-vector noise.
usage: python steps.py RUN_DIR TAG OUT_JSON
"""
import json, math, os, sys
import numpy as np
import torch
from locate import TP, W, P, NTR, QI, Q, QL, S, T, targets, lab_idx, qq_idx

torch.set_num_threads(1)
QQ = qq_idx()
LABS = [np.arange(s * Q, s * Q + QL) for s in range(S)]


# ---------------------------------------------------------------- hookable forward
def run(model, X, m, hk=None, cache=False, bs=2000):
    """hk: dict of (kind, layer) -> f(tensor, sl) -> tensor. kinds: 'attn' (B,H,T,T) pattern; 'heads' (B,H,T,d)
    per-head contribution after the output projection (no bias); 'attn_out' (B,T,d) total attention output (incl.
    bias); 'mlp_out' (B,T,d); 'resid' (B,T,d) after the block. Returns head tuple (numpy) and optional cache."""
    hk = hk or {}
    model.eval()
    heads, C = [], {}
    with torch.no_grad():
        for i in range(0, len(X), bs):
            sl = slice(i, min(i + bs, len(X)))
            Xb, mb = torch.from_numpy(X[sl]), torch.from_numpy(m[sl])
            B = len(Xb)
            ctx = mb.clone(); ctx[:, QI] = False
            allow = ctx[:, None, :].expand(B, T, T).clone(); allow[:, QI, QI] = True
            eye = torch.eye(T, dtype=torch.bool)[None]; allow = allow | (eye & ~mb[:, :, None])
            x = model.inp(Xb)
            for li, blk in enumerate(model.blocks):
                d = x.shape[-1]; H = blk.h; dh = getattr(blk, "dh", d // H)
                q, k, v = blk.qkv(blk.ln1(x)).split(H * dh, -1)
                sp = lambda t: t.view(B, T, H, dh).transpose(1, 2)
                q, k, v = sp(q), sp(k), sp(v)
                a = (q @ k.transpose(-1, -2) / math.sqrt(dh)).masked_fill(~allow[:, None], float('-inf')).softmax(-1)
                if ('attn', li) in hk: a = hk[('attn', li)](a, sl)
                z = a @ v                                                   # (B,H,T,dh)
                Wo = blk.o.weight                                           # (d, d)
                hc = torch.stack([z[:, h] @ Wo[:, h * dh:(h + 1) * dh].T for h in range(H)], 1)  # (B,H,T,d)
                if ('heads', li) in hk: hc = hk[('heads', li)](hc, sl)
                ao = hc.sum(1) + blk.o.bias
                if ('attn_out', li) in hk: ao = hk[('attn_out', li)](ao, sl)
                x = x + ao
                mo = blk.mlp(blk.ln2(x))
                if ('mlp_out', li) in hk: mo = hk[('mlp_out', li)](mo, sl)
                x = x + mo
                if ('resid', li) in hk: x = hk[('resid', li)](x, sl)
                if cache:
                    for nm, t in ((f'attn{li}', a), (f'heads{li}', hc), (f'attn_out{li}', ao), (f'mlp_out{li}', mo),
                                  (f'resid{li}', x)):
                        C.setdefault(nm, []).append(t.numpy().copy())
            heads.append(model.out(model.ln(x[:, QI])).numpy())
    y = np.concatenate(heads); K = model.mixk
    head = (y[:, K:2 * K], np.clip(y[:, 2 * K:], -6, 3), y[:, :K])
    if cache:
        C = {k: np.concatenate(v) for k, v in C.items()}
        return head, C
    return head


def moments(head):
    mu, var = P.head_moments(head)
    return mu, np.log(var)


def fitline(x, y):
    A = np.stack([x, np.ones_like(x)], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(slope=float(coef[0]), r2=float(r2))


def sub(D, idx):
    return {k: D[k][idx] for k in ('o', 'y', 'v', 'vidx')}


# ---------------------------------------------------------------- step 2
def step2(model, D, X, m, ds):
    te = np.arange(NTR, len(X)); n = len(te)
    rng = np.random.default_rng(1)
    s_i = rng.integers(0, S, n); j_i = rng.integers(0, 3, n)
    dA = sub(D, te)
    dB = {k: v.copy() for k, v in dA.items()}
    eps = rng.standard_normal((n, QL)) * np.sqrt(W.VLEV[j_i])[:, None]
    dB['o'][np.arange(n), s_i, :QL] = dA['y'][:, :QL] + eps
    XA = X[te]; mA = m[te]; XB = XA.copy()
    for q in range(QL):
        XB[np.arange(n), s_i * Q + q, 2 * ds] = dB['o'][np.arange(n), s_i, q]
    _, _, momA, _ = W.references(dA); _, _, momB, _ = W.references(dB)
    bm = momB['bayes'][0] - momA['bayes'][0]
    bv = np.log(momB['bayes'][1]) - np.log(momA['bayes'][1])
    hA, CA = run(model, XA, mA, cache=True); hB, CB = run(model, XB, mA, cache=True)
    mA_, vA_ = moments(hA)
    tok = s_i * Q + QL
    ar = np.arange(n)
    hdiff = CB['resid0'][ar, tok] - CA['resid0'][ar, tok]
    rdir = np.random.default_rng(2).standard_normal(hdiff.shape)
    rdir *= (np.linalg.norm(hdiff, axis=1) / np.linalg.norm(rdir, axis=1))[:, None]

    def patch_resid(src_rows, toks):
        def f(x, sl):
            x = x.clone(); idx = np.arange(sl.start, sl.stop)
            for t in toks(idx):
                x[np.arange(len(idx)), t] = torch.from_numpy(src_rows(idx, t))
            return x
        return f

    variants = {
        'full_swap': None,
        'a_QQ_mlp1': {('resid', 0): patch_resid(lambda idx, t: CB['resid0'][idx, t], lambda idx: [tok[idx]])},
        'b_QQ_attn1_pattern': {('attn', 0): (lambda a, sl: _patch_attn(a, sl, CB['attn0'], tok))},
        'c_LAB_mlp1': {('resid', 0): patch_resid(lambda idx, t: CB['resid0'][idx, t],
                                                 lambda idx: [s_i[idx] * Q + q for q in range(QL)])},
        'R_random_QQ_mlp1': {('resid', 0): patch_resid(lambda idx, t: CA['resid0'][idx, t] + rdir[idx].astype(np.float32),
                                                       lambda idx: [tok[idx]])},
    }
    out = {}
    full = None
    for nm, h in variants.items():
        hh = hB if h is None else run(model, XA, mA, hk=h)
        mm, vv = moments(hh)
        dm, dv = mm - mA_, vv - vA_
        if nm == 'full_swap': full = (dm, dv)
        out[nm] = dict(mean=fitline(bm, dm), logvar=fitline(bv, dv),
                       mean_vs_full=fitline(full[0], dm), logvar_vs_full=fitline(full[1], dv),
                       mean_abs_shift=float(np.abs(dm).mean()))
    out['bayes_mean_abs_shift'] = float(np.abs(bm).mean())
    return out


def _patch_attn(a, sl, srcA, tok):
    a = a.clone(); idx = np.arange(sl.start, sl.stop)
    a[np.arange(len(idx)), :, tok[idx]] = torch.from_numpy(srcA[idx, :, tok[idx]])
    return a


# ---------------------------------------------------------------- step 3
def step3(model, D, X, m, pool_heads):
    n = len(X); tr = np.arange(NTR); te = np.arange(NTR, n)
    head, C = run(model, X, m, cache=True)
    ys = D['ystar']
    src, _, _, _, beta, El = targets(D, P.head_moments(head)[0])
    nll = lambda h: float(P.head_nll_np(h, ys[te]).mean())
    base = nll(tuple(x[te] for x in head))
    Xt, mt = X[te], m[te]
    # means / fits on train
    ao_mean = C['attn_out0'][tr][:, QQ].reshape(-1, C['attn_out0'].shape[-1]).mean(0)
    hc_mean = C['heads0'][tr][:, :, QQ].mean((0, 2))                       # (H, d)
    mo = C['mlp_out0'][:, QQ]                                               # (n, 3, d)
    mo_mean = mo[tr].reshape(-1, mo.shape[-1]).mean(0)
    Z1 = np.stack([src['logSS'], src['E_lab']], 2)                         # (n, S, 2)
    Z2 = np.concatenate([Z1, src['p_lab']], 2)
    fits = {}
    for zn, Z in (('z1', Z1), ('z2', Z2)):
        A = np.concatenate([Z, np.ones(Z.shape[:2] + (1,))], 2)
        coef, *_ = np.linalg.lstsq(A[tr].reshape(-1, A.shape[-1]), mo[tr].reshape(-1, mo.shape[-1]), rcond=None)
        rec = A @ coef
        r2 = 1 - ((mo[te] - rec[te]) ** 2).sum() / ((mo[te] - mo[te].reshape(-1, mo.shape[-1]).mean(0)) ** 2).sum()
        fits[zn] = (rec[te].astype(np.float32), float(r2))
    bt = beta[te]

    def set_qq(val):                                                        # val: (d,) or (nte, 3, d)
        def f(x, sl):
            x = x.clone()
            v = torch.from_numpy(np.broadcast_to(val, (n - NTR, S, x.shape[-1]))[sl].astype(np.float32).copy())
            x[:, QQ] = v
            return x
        return f

    def uniform_own(heads_):
        def f(a, sl):
            a = a.clone()
            for s in range(S):
                for h in heads_:
                    a[:, h, QQ[s]] = 0
                    a[:, h, QQ[s], LABS[s]] = 0.25
            return a
        return f

    def heads_mean(heads_):
        def f(hc, sl):
            hc = hc.clone()
            for h in heads_:
                hc[:, h, QQ] = torch.from_numpy(hc_mean[h].astype(np.float32))
            return hc
        return f

    def qt_redistribute(w, lab_too):                                       # w (nte, 3) weights summing to 1
        def f(a, sl):
            a = a.clone(); ww = torch.from_numpy(w[sl].astype(np.float32))  # (B, 3)
            row = a[:, :, QI]                                               # (B, H, T)
            M = row[:, :, QQ].sum(-1, keepdim=True)
            row[:, :, QQ] = M * ww[:, None, :]
            if lab_too:
                Ls = torch.stack([row[:, :, L].sum(-1) for L in LABS], -1)  # (B, H, 3)
                Lt = Ls.sum(-1, keepdim=True)
                for s, L in enumerate(LABS):
                    shape = row[:, :, L] / Ls[:, :, s:s + 1].clamp_min(1e-12)
                    row[:, :, L] = shape * (Lt * ww[:, None, :])[:, :, s:s + 1]
            a[:, :, QI] = row
            return a
        return f

    wb = bt / bt.sum(1, keepdims=True); wu = np.full_like(wb, 1 / 3)
    H = model.blocks[0].h
    allh = list(range(H))
    E = {}
    E['3a_attn'] = ({('attn', 0): uniform_own(allh)}, {('attn_out', 0): set_qq(ao_mean)})
    if pool_heads and len(pool_heads) < H:
        E['3a_attn_poolheads'] = ({('attn', 0): uniform_own(pool_heads)}, {('heads', 0): heads_mean(pool_heads)})
    E['3a_mlp'] = ({('mlp_out', 0): set_qq(fits['z1'][0])}, {('mlp_out', 0): set_qq(mo_mean)})
    E['3a_mlp_z2'] = ({('mlp_out', 0): set_qq(fits['z2'][0])}, {('mlp_out', 0): set_qq(mo_mean)})
    E['3a_both'] = ({('attn', 0): uniform_own(allh), ('mlp_out', 0): set_qq(fits['z1'][0])},
                    {('attn_out', 0): set_qq(ao_mean), ('mlp_out', 0): set_qq(mo_mean)})
    E['3b'] = ({('attn', 1): qt_redistribute(wb, False)}, {('attn', 1): qt_redistribute(wu, False)})
    E['3b_lab_too'] = ({('attn', 1): qt_redistribute(wb, True)}, {('attn', 1): qt_redistribute(wu, True)})
    E['3_all'] = ({**E['3a_both'][0], **E['3b'][0]}, {**E['3a_both'][1], **E['3b'][1]})
    # ---- POST-HOC variants (PREREG Amendment 1b, after seeing small_s2's pre-registered step-3 scores)
    Rr = (D['o'][:, :, :QL] - D['y'][:, None, :QL])                       # (n, S, QL)
    A0 = C['attn0']                                                         # (n, H, T, T)
    tilt = {}
    for h in allh:
        lx, r1, r2_ = [], [], []
        for s_ in range(S):
            la = np.log(np.maximum(A0[tr][:, h, QQ[s_]][:, LABS[s_]], 1e-30))
            lx.append(la - la.mean(1, keepdims=True))
            rr = Rr[tr, s_]; r1.append(rr - rr.mean(1, keepdims=True)); q2 = rr ** 2; r2_.append(q2 - q2.mean(1, keepdims=True))
        Aa = np.stack([np.concatenate(r1).ravel(), np.concatenate(r2_).ravel()], 1)
        cf, *_ = np.linalg.lstsq(Aa, np.concatenate(lx).ravel(), rcond=None)
        tilt[h] = cf
    Rte = Rr[te]

    def within_own(mode):                                                   # keep own-LAB mass, reshape within it
        def f(a, sl):
            a = a.clone()
            for s_ in range(S):
                for h in allh:
                    Mown = a[:, h, QQ[s_]][:, LABS[s_]].sum(-1, keepdim=True)
                    if mode == 'flat':
                        w = torch.full((a.shape[0], QL), 0.25)
                    else:
                        rr = torch.from_numpy(Rte[sl, s_].astype(np.float32))
                        w = torch.softmax(float(tilt[h][0]) * rr + float(tilt[h][1]) * rr ** 2, -1)
                    rowv = a[:, h, QQ[s_]].clone(); rowv[:, LABS[s_]] = Mown * w
                    a[:, h, QQ[s_]] = rowv
            return a
        return f
    E['ph_attn_tilt_vs_flat'] = ({('attn', 0): within_own('tilt')}, {('attn', 0): within_own('flat')})
    Mz = None
    A1 = np.concatenate([Z1, np.ones(Z1.shape[:2] + (1,))], 2)
    coef, *_ = np.linalg.lstsq(A1[tr].reshape(-1, 3), mo[tr].reshape(-1, mo.shape[-1]), rcond=None)
    Mz, c0 = coef[:2], coef[2]                                              # (2, d), (d,)
    Pm = Mz.T @ np.linalg.pinv(Mz @ Mz.T) @ Mz                              # projector onto span(rows of Mz)
    zbar = Z1[tr].reshape(-1, 2).mean(0)

    def mlp_sub(zval):                                                      # zval (nte, S, 2)
        def f(x, sl):
            xn = x.numpy()
            zz = zval[sl]
            tgt = c0 + zz @ Mz                                              # (B, S, d)
            xq = xn[:, QQ]
            xq = xq - (xq - tgt) @ Pm.T
            x = x.clone(); x[:, QQ] = torch.from_numpy(xq.astype(np.float32))
            return x
        return f
    zt = Z1[te]
    E['ph_mlp_subspace'] = ({('mlp_out', 0): mlp_sub(zt)}, {('mlp_out', 0): mlp_sub(np.broadcast_to(zbar, zt.shape))})
    E['ph_tilt_subspace_3b'] = ({('attn', 0): within_own('tilt'), ('mlp_out', 0): mlp_sub(zt), ('attn', 1): qt_redistribute(wb, False)},
                                {('attn', 0): within_own('flat'), ('mlp_out', 0): mlp_sub(np.broadcast_to(zbar, zt.shape)),
                                 ('attn', 1): qt_redistribute(wu, False)})
    out = dict(nll_model=base, nll_bayes=float(D['nll_bayes'][te].mean()), mlp_fit_r2={k: v[1] for k, v in fits.items()},
               tilt={str(h): [float(c) for c in v] for h, v in tilt.items()})
    for k, (rep, abl) in E.items():
        nr, na = nll(run(model, Xt, mt, hk=rep)), nll(run(model, Xt, mt, hk=abl))
        out[k] = dict(nll_replaced=nr, nll_ablated=na, denom=na - base, score=(na - nr) / (na - base))
    return out


# ---------------------------------------------------------------- step 4
def simplex(k, dim):
    E = np.eye(k) - 1.0 / k
    U, _, _ = np.linalg.svd(E)
    V = E @ U[:, :k - 1]
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    assert V.shape[1] == dim
    return V


def id_set(k, ds, rng):
    R = np.linalg.qr(rng.standard_normal((ds, ds)))[0]
    base = np.eye(ds)[:k] if k <= ds else simplex(k, ds)
    return base @ R.T * math.sqrt(ds) / 4


def step4(model, D, X, m, ds):
    n = len(X)
    T_ = {k: D[k] for k in ('comp_means', 'comp_vars', 'w_bayes')}
    ys, nb, mb = D['ystar'], D['nll_bayes'], D['mean_bayes']

    def score(h):
        nll = P.head_nll_np(h, ys); mu, _ = P.head_moments(h)
        kl = P.kl_grid(T_, h, chunk=200)
        return dict(gap=float((nll - nb).mean()), kl=float(kl.mean()), sd_mean_err=float((mu - mb).std()),
                    var_mean_err=float(((mu - mb) ** 2).mean()))
    out = {'original': score(run(model, X, m))}
    for f in range(3):
        rng = np.random.default_rng(f)
        semb = np.broadcast_to(id_set(S, ds, rng), (n, S, ds)); qemb = np.broadcast_to(id_set(Q, ds, rng), (n, Q, ds))
        Xc, mc = W.assemble(semb, qemb, D['o'], D['y'])
        out[f'const_frame{f}'] = score(run(model, Xc, mc))
    rng = np.random.default_rng(99)
    mus, lss, lgs = [], [], []
    K = 16
    for _ in range(K):
        Xd, md = W.tokens_np(dict(o=D['o'], y=D['y']), rng, ds)
        h = run(model, Xd, md)
        mus.append(h[0]); lss.append(h[1])
        lg = h[2] - h[2].max(1, keepdims=True); lg = lg - np.log(np.exp(lg).sum(1, keepdims=True))
        lgs.append(lg - math.log(K))
    out['avg16'] = score((np.concatenate(mus, 1), np.concatenate(lss, 1), np.concatenate(lgs, 1)))
    o = out['original']
    for k in list(out):
        if k != 'original':
            out[k]['frac_kl_removed'] = 1 - out[k]['kl'] / o['kl']
            out[k]['frac_meanerr_var_removed'] = 1 - out[k]['var_mean_err'] / o['var_mean_err']
    return out


def main():
    run_dir, tag, outp = sys.argv[1:4]
    rd = os.path.join(TP, run_dir)
    meta = json.load(open(os.path.join(rd, 'meta.json')))
    ds = meta['ds']
    D = dict(np.load(os.path.join(TP, 'runs', 'testset.npz' if ds == W.DS else f'testset_ds{ds}.npz')))
    X, m = D['X'], D['valid']
    trained = P.build(meta); trained.load_state_dict(torch.load(os.path.join(rd, 'model.pt')))
    torch.manual_seed(meta['seed']); untrained = P.build(meta)
    s1 = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', f'{tag}.json')))
    pool = [h for h in range(meta['heads']) if s1['attn_trained'][f'L1H{h}']['QQ']['ratio_lab'] >= 2]
    res = dict(tag=tag, pool_heads=pool)
    res['step2_trained'] = step2(trained, D, X, m, ds); print(tag, 'step2 trained', flush=True)
    res['step2_untrained'] = step2(untrained, D, X, m, ds); print(tag, 'step2 untrained', flush=True)
    res['step3'] = step3(trained, D, X, m, pool); print(tag, 'step3', flush=True)
    res['step4'] = step4(trained, D, X, m, ds); print(tag, 'step4', flush=True)
    json.dump(res, open(outp, 'w'))


if __name__ == '__main__':
    main()
