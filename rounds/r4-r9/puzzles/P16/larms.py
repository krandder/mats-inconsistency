"""L-all (oracle ceiling) and L-train (library-matcher control).

Both compute the Neyman-Pearson log-likelihood ratio of the candidate's post-birth values,
    LR_k = log sum_{h in H1} pi_h L_h(k) - log sum_{h in H0} pi_h L_h(k),   L_h(k) = prod_{t=H+1}^{H+k} p(b_t | B_t, y_t; h),
knowing the episode's generative model for x -> y and the listed columns B (x is marginalised per row by enumeration).
Instances h are exactly the generator's parameter grids (gen.type_params / gen.build_col); prior: types equal, generator
weights within a type. H1 = instances with CMI >= 0.05 bits under the model, H0 = instances with CMI == 0 (analytically
zero types: T0, T3a, T3b, HN1, HN3, HN4; others: fp32 CMI < 1e-5); instances in between belong to neither.

L-all: all 20 types, the episode's true S family. L-train: the 8 in-distribution types and, for the y model, a posterior
mixture over the 4 training S families (weights from the likelihood of (B_t, y_t) on rows 1..H+k; each family's p(y|x)
floored with 1e-6 uniform mass). The instance partition for L-train uses its MAP family.
"""
import numpy as np
import torch
from gen import (ALL_TYPES, ID_TYPES, TRAIN_FAMS, H, bits_table, build_col, type_params, fam_mask, col_mi_y)
from oracle import add_col, cmi_from_joint

ZERO_TYPES = {'T0', 'T3a', 'T3b', 'HN1', 'HN3', 'HN4'}
THR_POS, THR_ZERO = 0.05, 1e-5


def py_for_family(ep, fam):
    X = 2 ** ep.nx; w = ep.w
    YB = bits_table(w)
    pbit = np.ones((X, 2 ** w))
    for j in range(w):
        pbit *= np.where(YB[None, :, j] == ep.gtab[j][:, None], 1 - ep.eta, ep.eta)
    if ep.branch == 'mix':
        S = fam_mask(fam, w).astype(np.float64)
        return ep.ztab[:, None] * (S / S.sum())[None] + (1 - ep.ztab[:, None]) * pbit
    return pbit


def hn1_k(ep):
    cands = [(col_mi_y(c, ep.py), k) for k, c in enumerate(ep.cols[:ep.m]) if not c.par and c.ydep is None and not c.markov]
    return max(cands)[1] if cands else 0


def library(ep, types):
    """[(type, weight, Col)] with weights summing to 1 over types present (types equal)."""
    saved = ep.info.get('hn1_k')
    ep.info['hn1_k'] = hn1_k(ep) if saved is None else saved
    per = []
    for t in types:
        ps = type_params(t, ep)
        if ps:
            per.append((t, ps))
    lib = []
    for t, ps in per:
        tot = sum(p[0] for p in ps)
        for wt, prm in ps:
            lib.append((t, wt / tot / len(per), build_col(t, prm, ep)))
    if saved is None:
        ep.info.pop('hn1_k')
    return lib


def listed_J(ep, dev, dt):
    X = 2 ** ep.nx
    J = torch.ones(1, X, 1, dtype=dt, device=dev)
    for col in ep.cols[:ep.m]:
        par = torch.full((1, 3), -1, dtype=torch.long, device=dev)
        par[0, :len(col.par)] = torch.tensor(col.par, dtype=torch.long)
        J = add_col(J, torch.as_tensor(col.P1, dtype=dt, device=dev)[None], par)
    return J[0]                                                     # (X, 2^m) = p(B | x)


def _parent_cfg(par, n, dev):
    s = torch.arange(n, device=dev)
    cfg = torch.zeros(n, dtype=torch.long, device=dev)
    for i, p in enumerate(par):
        cfg |= ((s >> p) & 1) << i
    return cfg


def instance_cmi(ep, lib, py, J, dev, dt=torch.float32, chunk=32):
    """CMI (bits) of every library instance under p(y|x) = py."""
    X, n = J.shape
    pyt = torch.as_tensor(py, dtype=dt, device=dev)
    Jd = J.to(dt)
    pBy = Jd.T @ pyt / X                                               # (n, Y)
    out = np.zeros(len(lib))
    gen_idx = []
    for i, (t, wt, col) in enumerate(lib):
        if t in ZERO_TYPES or col.markov:
            out[i] = 0.0
        elif col.ydep is not None:
            q = torch.as_tensor(col.ydep, dtype=dt, device=dev)[None]
            p = torch.stack([pBy * (1 - q), pBy * q], 1)[None]
            out[i] = float(cmi_from_joint(p)[0])
        else:
            gen_idx.append(i)
    for c0 in range(0, len(gen_idx), chunk):
        ids = gen_idx[c0:c0 + chunk]
        Ms = []
        for i in ids:
            col = lib[i][2]
            cfg = _parent_cfg(col.par, n, dev)
            Ms.append(torch.as_tensor(col.P1, dtype=dt, device=dev)[:, cfg])      # (X, n)
        M = torch.stack(Ms)                                              # (c, X, n)
        p1 = torch.matmul((Jd[None] * M).transpose(1, 2), pyt) / X        # (c, n, Y)
        p = torch.stack([pBy[None] - p1, p1], 2).clamp(min=0)
        out[ids] = cmi_from_joint(p).clamp(min=0).cpu().numpy()
    return out


def row_lik(ep, lib, rows, J, py, dev, dt=torch.float32, chunk=256):
    """p(b_t = observed | B_t, y_t; h) for rows (xi, y, V) restricted to post-birth rows -> (Nh, K) float64 log."""
    xi, y, V = rows
    Vp = torch.as_tensor(V[H:], dtype=torch.long, device=dev)
    yp = torch.as_tensor(y[H:], dtype=torch.long, device=dev)
    K = len(yp); m = ep.m; X = 2 ** ep.nx
    bidx = (Vp[:, :m] << torch.arange(m, device=dev)).sum(1)            # (K,)
    pyt = torch.as_tensor(py, dtype=dt, device=dev)
    post = J.to(dt)[:, bidx].T * pyt[:, yp].T                          # (K, X)
    post = post / post.sum(1, keepdim=True).clamp(min=1e-30)
    b = Vp[:, m]
    out = torch.zeros(len(lib), K, dtype=torch.float64, device=dev)
    gen_idx = []
    for i, (t, wt, col) in enumerate(lib):
        if col.ydep is not None:
            p1 = torch.as_tensor(col.ydep, dtype=torch.float64, device=dev)[yp]
            out[i] = torch.log(torch.where(b == 1, p1, 1 - p1).clamp(min=1e-300))
        elif col.markov:
            same = torch.cat([torch.zeros(1, dtype=torch.bool, device=dev), b[1:] == b[:-1]])
            stay = col.markov + (1 - col.markov) * .5
            pr = torch.where(same, torch.full_like(b, stay, dtype=torch.float64), torch.full_like(b, 1 - stay, dtype=torch.float64))
            pr[0] = .5
            out[i] = torch.log(pr)
        else:
            gen_idx.append(i)
    for c0 in range(0, len(gen_idx), chunk):
        ids = gen_idx[c0:c0 + chunk]
        P1 = torch.stack([torch.as_tensor(lib[i][2].P1, dtype=dt, device=dev) for i in ids])      # (c, X, 8)
        cfg = torch.zeros(len(ids), K, dtype=torch.long, device=dev)
        for r, i in enumerate(ids):
            for d, p in enumerate(lib[i][2].par):
                cfg[r] |= Vp[:, p] << d
        G = torch.gather(P1, 2, cfg[:, None, :].expand(-1, X, -1))       # (c, X, K)
        p1 = torch.einsum('cxk,kx->ck', G, post).double().clamp(1e-12, 1 - 1e-12)
        out[ids] = torch.log(torch.where(b[None] == 1, p1, 1 - p1))
    return out


def lr_curve(lik, logw, cmi):
    """lik (Nh,K) per-row log-lik, logw (Nh,) log prior, cmi (Nh,) -> (K,) LR after k rows (k = 1..K)."""
    cum = lik.cumsum(1)
    a = torch.as_tensor(logw, dtype=torch.float64, device=lik.device)[:, None] + cum
    h1 = torch.as_tensor(cmi >= THR_POS, device=lik.device)
    h0 = torch.as_tensor(cmi < THR_ZERO, device=lik.device)
    if h1.sum() == 0 or h0.sum() == 0:
        return torch.full((lik.shape[1],), float('nan'), dtype=torch.float64)
    l1 = torch.logsumexp(a[h1], 0) - torch.logsumexp(a[h1, 0] - cum[h1, 0], 0)
    l0 = torch.logsumexp(a[h0], 0) - torch.logsumexp(a[h0, 0] - cum[h0, 0], 0)
    return (l1 - l0).cpu()


def fam_rowlik(J, py, y, V, m, dev, dt=torch.float64):
    """per-row log p(B_t, y_t) under p(y|x) = py -> numpy (T,)."""
    Vt = torch.as_tensor(V[:, :m], dtype=torch.long, device=dev)
    bidx = (Vt << torch.arange(m, device=dev)).sum(1)
    pyt = torch.as_tensor(py, dtype=dt, device=dev)
    pr = (J.to(dt)[:, bidx] * pyt[:, torch.as_tensor(y, device=dev)]).mean(0)
    return torch.log(pr.clamp(min=1e-300)).cpu().numpy()


def l_scores(ep, rows, K, dev, which=('L-all', 'L-train')):
    """-> {arm: (K,) LR after k = 1..K post-birth rows}."""
    xi, y, V = rows
    rows_k = (xi[:H + K], y[:H + K], V[:H + K])
    J = listed_J(ep, dev, torch.float64)
    out = {}
    if 'L-all' in which:
        lib = library(ep, ALL_TYPES)
        cmi = instance_cmi(ep, lib, ep.py, J, dev)
        lik = row_lik(ep, lib, rows_k, J, ep.py, dev)
        out['L-all'] = lr_curve(lik, np.log([l[1] for l in lib]), cmi)
    if 'L-train' in which:
        lib = library(ep, ID_TYPES)
        logw = np.log([l[1] for l in lib])
        fams = TRAIN_FAMS if ep.branch == 'mix' else TRAIN_FAMS[:1]
        pys = []
        for f in fams:
            p = py_for_family(ep, f)
            pys.append((1 - 1e-6) * p + 1e-6 / p.shape[1])
        # family posterior at each k uses rows 1..H+k (B_t, y_t); per-row b-likelihoods mixed by that posterior
        m = ep.m
        fl = np.array([np.cumsum(fam_rowlik(J, p, y[:H + K], V[:H + K], m, dev))[H:H + K] for p in pys])
        mapf = int(np.argmax(fl[:, -1]))
        cmi = instance_cmi(ep, lib, pys[mapf], J, dev)
        curves = torch.stack([lr_curve(row_lik(ep, lib, rows_k, J, p, dev), logw, cmi) for p in pys])   # (F, K)
        # LR under the family mixture: the per-family LRs combine through the hypothesis-level likelihoods; with the
        # posterior concentrated (64+ rows of y) we use the posterior-weighted LR (exact when one family dominates)
        pw = torch.softmax(torch.as_tensor(fl, dtype=torch.float64), 0)
        out['L-train'] = (pw * torch.nan_to_num(curves, nan=0.0)).sum(0)
    return out
