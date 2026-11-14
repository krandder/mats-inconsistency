"""Prior-free and SGD valuers, torch, batched over E episodes that share (m, w, T). Works on CPU (tests) and GPU.

Episode rows: V (E, T, m+1) long (candidate last, -1 = absent), Yb (E, T, w) long, candidate born at row index H.
Every function returns per-row log-probabilities (nats), so V^preq_k for every k is a cumulative sum over rows H..H+k-1.

C1  running counts: per-output KT gain of p(y_j | b) against p(y_j), summed over j.
C4  sparse parent-set BMA, prequential, over parent sets of size <= 3 from B, b (3-valued: 0/1/absent) and y_<j.
    b-sets back off to their b-free pair set: p = (n1 + alpha q_pair) / (n + alpha); while b is absent a b-set predicts
    exactly like its pair and the with-prior splits each pair's prior mass (1-rho, rho), so the with and without
    mixtures agree before birth. Reverse direction: log p(b | B, y) - log p(b | B) on post-birth rows.
C5  sequential CI test by betting (Shaer et al. form): stake tanh(c * [log p(y|B,b) - log p(y|B,b~)]), b~ ~ p-hat(b|B)
    (C4 reverse B-only predictive), mean over twins, ONS bet size in [0, 1/2]; value = log wealth.
S1/S2 SGD readers: batched MADE MLPs (one per episode, with and without b), prequential online Adam.
"""
from itertools import combinations
from math import comb, log
import numpy as np
import torch

KS = (4, 8, 16, 32, 64, 128)
_SETS = {}


def set_table(nv, kmax, device):
    key = (nv, kmax, str(device))
    if key not in _SETS:
        sets = [s for k in range(kmax + 1) for s in combinations(range(nv), k)]
        idx = torch.full((len(sets), kmax), -1, dtype=torch.long)
        for i, s in enumerate(sets):
            idx[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        size = torch.tensor([len(s) for s in sets])
        _SETS[key] = (sets, idx.to(device), size.to(device))
    return _SETS[key]


def _gather_counts(cfg, y, ncfg, rows_at=None):
    """cfg (E,S,T) long, y (E,T) 0/1 -> exclusive counts n, n1 (E,S,T) of earlier rows with the same config.
    If rows_at = (F, rows): counts over rows < F evaluated at cfg[..., rows]."""
    E, S, T = cfg.shape
    oh = torch.zeros(E, S, T, ncfg, dtype=torch.int32, device=cfg.device)
    oh.scatter_(3, cfg[..., None], 1)
    oh1 = oh * y[:, None, :, None].to(torch.int32)
    C = oh.cumsum(2) - oh
    C1 = oh1.cumsum(2) - oh1
    n = C.gather(3, cfg[..., None])[..., 0]
    n1 = C1.gather(3, cfg[..., None])[..., 0]
    return n, n1, C, C1, oh, oh1


def _logmix(logpi, LW, lp):
    """logpi (S,), LW (E,S,T) exclusive cum log-lik, lp (E,S,T) -> (E,T) log predictive of the mixture."""
    a = logpi[None, :, None] + LW
    return torch.logsumexp(a + lp, 1) - torch.logsumexp(a, 1)


def bma_forward(V, Yb, a=.5, alpha=2., rho=.5, kmax=3, settle=None, flip=True):
    """C4 forward. Returns dict of (E,T) per-row log p(y_t): 'with', 'without', and 'with_flip' (b's value flipped at
    row t, weights from the real history). settle = (F, L): also 'set_with'/'set_without' (E,L): rows F..F+L-1
    predicted from counts and weights frozen at F."""
    E, T, ncol = V.shape
    m = ncol - 1; w = Yb.shape[2]
    dev = V.device; dt = torch.float64
    Vb = V.clone(); Vb[Vb < 0] = 2
    out = {k: torch.zeros(E, T, dtype=dt, device=dev) for k in ('with', 'without', 'with_flip')}
    if settle is not None:
        F, L = settle
        out['set_with'] = torch.zeros(E, L, dtype=dt, device=dev)
        out['set_without'] = torch.zeros(E, L, dtype=dt, device=dev)
    for j in range(w):
        Z = torch.cat([Vb, Yb[:, :, :j]], 2)                    # (E,T,nv); var m = b
        nv = Z.shape[2]
        sets, sidx, size = set_table(nv, kmax, dev)
        S = len(sets)
        pw = torch.tensor([3 ** d for d in range(kmax)], device=dev)
        vals = Z[:, :, sidx.clamp(min=0)]                       # (E,T,S,kmax)
        vals = vals * (sidx >= 0)
        cfg = (vals * pw).sum(3).transpose(1, 2).contiguous()   # (E,S,T)
        hasb = (sidx == m).any(1)
        bpos = (sidx == m).long().argmax(1)
        # pair of each b-set = the same set without b
        index = {s: i for i, s in enumerate(sets)}
        pair = torch.tensor([index[tuple(v for v in s if v != m)] if m in s else i for i, s in enumerate(sets)], device=dev)
        y = Yb[:, :, j]
        n, n1, C, C1, oh, oh1 = _gather_counts(cfg, y, 3 ** kmax)
        n = n.to(dt); n1 = n1.to(dt)
        pkt = (n1 + a) / (n + 2 * a)
        q = pkt[:, pair]
        bval = Vb[:, :, m]                                       # (E,T)
        absent = (bval == 2)[:, None, :]
        hb = hasb[None, :, None]
        p = torch.where(hb, torch.where(absent, q, (n1 + alpha * q) / (n + alpha)), pkt)
        lp = torch.where(y[:, None, :] == 1, torch.log(p), torch.log1p(-p))
        LW = lp.cumsum(2) - lp
        # priors: without = uniform over sizes then over sets of that size among the nv-1 non-b vars
        nob = ~hasb
        logpi0 = torch.tensor([-log(kmax + 1) - log(comb(nv - 1, len(s))) if m not in s else 0. for s in sets],
                              dtype=dt, device=dev)
        splittable = nob & (size < kmax)
        logpi_w = torch.where(hasb, logpi0[pair] + log(rho), torch.where(splittable, logpi0 + log(1 - rho), logpi0))
        out['with'] += _logmix(logpi_w, LW, lp)
        out['without'] += _logmix(logpi0[nob], LW[:, nob], lp[:, nob])
        if flip:
            fl = torch.where(bval == 2, bval, 1 - bval)          # (E,T)
            cfgf = cfg + (hb & ~absent) * (fl - bval)[:, None, :] * (3 ** bpos)[None, :, None]
            nf = C.gather(3, cfgf[..., None])[..., 0].to(dt)
            n1f = C1.gather(3, cfgf[..., None])[..., 0].to(dt)
            pf = torch.where(hb, torch.where(absent, q, (n1f + alpha * q) / (nf + alpha)), pkt)
            lpf = torch.where(y[:, None, :] == 1, torch.log(pf), torch.log1p(-pf))
            out['with_flip'] += _logmix(logpi_w, LW, lpf)
        if settle is not None:
            rows = torch.arange(F, F + L, device=dev)
            CF = C[:, :, F]                                       # exclusive counts at F = counts over rows < F
            C1F = C1[:, :, F]
            cr = cfg[:, :, rows]                                  # (E,S,L)
            ns = CF.gather(2, cr).to(dt); n1s = C1F.gather(2, cr).to(dt)
            pk = (n1s + a) / (ns + 2 * a)
            qs = pk[:, pair]
            ab = (bval[:, rows] == 2)[:, None, :]
            ps = torch.where(hb, torch.where(ab, qs, (n1s + alpha * qs) / (ns + alpha)), pk)
            ys = y[:, rows][:, None, :]
            ls = torch.where(ys == 1, torch.log(ps), torch.log1p(-ps))
            WF = LW[:, :, F]                                      # log-lik of rows < F
            aw = logpi_w[None] + WF
            out['set_with'] += torch.logsumexp(aw[..., None] + ls, 1) - torch.logsumexp(aw, 1)[:, None]
            a0 = logpi0[nob][None] + WF[:, nob]
            out['set_without'] += torch.logsumexp(a0[..., None] + ls[:, nob], 1) - torch.logsumexp(a0, 1)[:, None]
        del oh, oh1, C, C1
    return out


def bma_reverse(V, Yb, H, a=.5, kmax=3):
    """C4 reverse on post-birth rows (index H..T-1): (E,T-H) log p(b|B,y) - log p(b|B), and p-hat(b_t=1 | B, past)."""
    E, T, ncol = V.shape
    m = ncol - 1; w = Yb.shape[2]
    dev = V.device; dt = torch.float64
    Z = torch.cat([V[:, H:, :m], Yb[:, H:]], 2)                # binary
    b = V[:, H:, m]
    nv = m + w
    sets, sidx, size = set_table(nv, kmax, dev)
    pw = torch.tensor([2 ** d for d in range(kmax)], device=dev)
    vals = Z[:, :, sidx.clamp(min=0)] * (sidx >= 0)
    cfg = (vals * pw).sum(3).transpose(1, 2).contiguous()
    n, n1, *_ = _gather_counts(cfg, b, 2 ** kmax)
    n = n.to(dt); n1 = n1.to(dt)
    p = (n1 + a) / (n + 2 * a)
    lp = torch.where(b[:, None, :] == 1, torch.log(p), torch.log1p(-p))
    LW = lp.cumsum(2) - lp
    isB = (sidx < m).all(1)
    logpi = torch.tensor([-log(kmax + 1) - log(comb(nv, len(s))) for s in sets], dtype=dt, device=dev)
    logpiB = torch.tensor([-log(kmax + 1) - log(comb(m, len(s))) for s, f in zip(sets, isB.tolist()) if f],
                          dtype=dt, device=dev)
    lw = _logmix(logpi, LW, lp)
    lb = _logmix(logpiB, LW[:, isB], lp[:, isB])
    aB = logpiB[None, :, None] + LW[:, isB]
    wB = torch.softmax(aB, 1)
    phat = (wB * p[:, isB]).sum(1)
    return lw - lb, phat


def c1_counts(V, Yb, H, a=.5):
    """(E, T-H) per-row C1 gain."""
    b = V[:, H:, -1]; Y = Yb[:, H:]
    E, Tp = b.shape
    dt = torch.float64
    tri = torch.tril(torch.ones(Tp, Tp, device=V.device, dtype=dt), -1)   # tri[t, t'] = 1 if t' < t
    same = (b[:, :, None] == b[:, None, :]).to(dt) * tri                     # (E,t,t')
    g = torch.zeros(E, Tp, dtype=dt, device=V.device)
    n = torch.arange(Tp, device=V.device, dtype=dt)
    for j in range(Y.shape[2]):
        y = Y[:, :, j].to(dt)
        n1 = y.cumsum(1) - y
        p0 = (n1 + a) / (n + 2 * a)
        nb = same.sum(2); nb1 = (same * y[:, None, :]).sum(2)
        pb = (nb1 + a) / (nb + 2 * a)
        g += torch.where(y == 1, torch.log(pb) - torch.log(p0), torch.log1p(-pb) - torch.log1p(-p0))
    return g


def c5_betting(lp_with, lp_flip, b, phat, gen, n_twins=8, scale=1.0):
    """lp_with, lp_flip (E,K) forward log p(y_t | B, b_t) and with b flipped, on post-birth rows; b, phat (E,K).
    Returns (E,K) log wealth after each row."""
    E, K = b.shape
    u = torch.rand((n_twins, E, K), generator=gen, dtype=torch.float64).to(b.device)
    tw = (u < phat[None]).long()
    s = torch.where(tw == b[None], torch.zeros_like(u), (lp_with - lp_flip)[None].expand_as(u))
    g = torch.tanh(scale * s).mean(0)
    lam = torch.zeros(E, dtype=torch.float64, device=b.device)
    A = torch.ones(E, dtype=torch.float64, device=b.device)
    logW = torch.zeros(E, dtype=torch.float64, device=b.device)
    out = torch.zeros(E, K, dtype=torch.float64, device=b.device)
    c = 2 / (2 - np.log(3))
    for t in range(K):
        gt = g[:, t]
        logW = logW + torch.log1p(lam * gt)
        out[:, t] = logW
        z = gt / (1 + lam * gt)
        A = A + z * z
        lam = (lam + c * z / A).clamp(0, .5)
    return out


# ---------------------------------------------------------------- SGD readers (S1 from scratch, S2 warm start)
class MadeBank(torch.nn.Module):
    """E independent one-hidden-layer MADE readers. Input features f (E,N,D) = (2v-1 for present values, present flag)
    per input column; output y (E,N,w) autoregressive in index order j."""

    def __init__(self, E, D, w, width, device, gen):
        super().__init__()
        self.w = w
        deg = torch.arange(width, device=device) % w                        # hidden unit k sees y_<deg
        self.register_buffer('my', (torch.arange(w, device=device)[:, None] < deg[None]).float())    # (w, width)
        self.register_buffer('mo', (deg[:, None] <= torch.arange(w, device=device)[None]).float())   # (width, w)
        s1 = 1 / np.sqrt(D + w)
        self.Wf = torch.nn.Parameter(torch.randn(E, D, width, generator=gen).to(device) * s1)
        self.Wy = torch.nn.Parameter(torch.randn(E, w, width, generator=gen).to(device) * s1)
        self.b1 = torch.nn.Parameter(torch.zeros(E, 1, width, device=device))
        self.Wo = torch.nn.Parameter(torch.randn(E, width, w, generator=gen).to(device) / np.sqrt(width))
        self.bo = torch.nn.Parameter(torch.zeros(E, 1, w, device=device))

    def logp(self, f, y):
        h = torch.relu(torch.bmm(f, self.Wf) + torch.bmm(y, self.Wy * self.my) + self.b1)
        lg = torch.bmm(h, self.Wo * self.mo) + self.bo
        return -torch.nn.functional.binary_cross_entropy_with_logits(lg, y, reduction='none').sum(2)   # (E,N)


def row_feats(V):
    pres = (V >= 0).float()
    val = torch.where(V >= 0, 2 * V.float() - 1, torch.zeros_like(V, dtype=torch.float))
    return torch.cat([val, pres], 2)


def _fit(model, opt, f, y, mask, steps):
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        lp = model.logp(f, y)
        loss = -(lp * mask).sum() / mask.sum(1).clamp(min=1).mean()
        loss.backward()
        opt.step()


def sgd_reader(V, Yb, H, K, mode='S2', width=64, lr=3e-3, wd=1e-4, E_steps=20, pre_steps=300, seed=0):
    """Prequential gain per post-birth row (E,K) for the with-b vs without-b SGD readers.
    S1: online from row 1 (E_steps full-batch Adam steps on the replay buffer after each row).
    S2: pretrain on rows < H (pre_steps), then the with copy gets zero-initialised weights for b; online over K rows."""
    E, T, ncol = V.shape
    w = Yb.shape[2]
    dev = V.device
    gen = torch.Generator().manual_seed(seed)
    f_with = row_feats(V[:, :H + K])
    Vn = V[:, :H + K].clone(); Vn[:, :, -1] = -1
    f_wo = row_feats(Vn)
    y = Yb[:, :H + K].float()
    D = f_with.shape[2]
    base = MadeBank(E, D, w, width, dev, gen)
    opt = torch.optim.AdamW(base.parameters(), lr=lr, weight_decay=wd)
    rowi = torch.arange(H + K, device=dev)
    if mode == 'S2':
        _fit(base, opt, f_wo, y, (rowi < H).float()[None].expand(E, -1), pre_steps)
    else:
        for t in range(1, H + 1):
            _fit(base, opt, f_wo, y, (rowi < t).float()[None].expand(E, -1), E_steps)
    # zero-init b's input weights (feature columns of b: value index m, presence index 2m+1)
    m = ncol - 1
    with torch.no_grad():
        base.Wf[:, m].zero_(); base.Wf[:, 2 * m + 1].zero_()
    import copy
    mw, mo_ = copy.deepcopy(base), copy.deepcopy(base)
    ow = torch.optim.AdamW(mw.parameters(), lr=lr, weight_decay=wd); ow.load_state_dict(opt.state_dict())
    oo = torch.optim.AdamW(mo_.parameters(), lr=lr, weight_decay=wd); oo.load_state_dict(opt.state_dict())
    gain = torch.zeros(E, K, dtype=torch.float64, device=dev)
    for k in range(K):
        t = H + k
        with torch.no_grad():
            gain[:, k] = (mw.logp(f_with[:, t:t + 1], y[:, t:t + 1]) - mo_.logp(f_wo[:, t:t + 1], y[:, t:t + 1]))[:, 0].double()
        msk = (rowi <= t).float()[None].expand(E, -1)
        _fit(mw, ow, f_with, y, msk, E_steps)
        _fit(mo_, oo, f_wo, y, msk, E_steps)
    return gain


def consumer_gain(V, Yb, Vf, Ybf, width=64, lr=3e-3, wd=1e-4, steps=600, seed=0):
    """Realized value to the slow consumer: an MLP (S2 architecture) trained on the given rows with and without b,
    mean log-score gain per row on a fresh block (Vf, Ybf all rows with b present). Returns (E,) nats/row."""
    E = V.shape[0]; w = Yb.shape[2]
    dev = V.device
    res = []
    for use_b in (True, False):
        gen = torch.Generator().manual_seed(seed)
        Va = V.clone(); Vfa = Vf.clone()
        if not use_b:
            Va[:, :, -1] = -1; Vfa[:, :, -1] = -1
        f = row_feats(Va); ff = row_feats(Vfa)
        mdl = MadeBank(E, f.shape[2], w, width, dev, gen)
        opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=wd)
        _fit(mdl, opt, f, Yb.float(), torch.ones(E, f.shape[1], device=dev), steps)
        with torch.no_grad():
            res.append(mdl.logp(ff, Ybf.float()).double().mean(1))
    return res[0] - res[1]


def cum_at(gain, ks=KS):
    """gain (E,K) per post-birth row -> {k: (E,) V^preq_k}."""
    c = gain.cumsum(1)
    return {k: c[:, k - 1] for k in ks if k <= gain.shape[1]}
