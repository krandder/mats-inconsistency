"""ploop.py -- parity world: distillation data from the tree, training of the rollout policy, and scoring.

Search / training code paths (grow_tree, collect, train) read no exact number. Scoring functions (score_tree,
score_is, route_logprob users) receive the exact answers as arguments from the driver, after the search has finished.
"""
import numpy as np
import torch
import torch.nn.functional as F

import ptree
import ppol
from pw import UN, log_mix


def new_tree(inst, cap, root=None):
    T = ptree.Tree(inst, cap)
    A = np.full((1, inst.n), UN, dtype=np.int8) if root is None else root.reshape(1, -1).astype(np.int8)
    T.add(inst.ev(A), -1, 0.0)
    return T


def collect(T, S, rng, draws=None):
    """Distillation targets = the amplified next-step value distribution at the states the search opened.
    For an opened state s and every assumption j still open there:
        target_j(s) = [score with h_j = 1] / [score], over what the amplification accounted for below s:
        closed states (exact totals and exact splits) plus the specification's own audit draws below the open
        frontier (each draw is a completion with weight UF * y / n_draw: an unbiased estimate of unfound score).
    States are sampled with probability proportional to sqrt(share of the accounted score that lies below them)."""
    inst, N = T.inst, T.N
    off = T.logU[0]
    zf, za = np.zeros(N), np.zeros((N, inst.n))
    cl = T.status[:N] == 2
    zf[cl] = np.exp(T.logU[:N][cl] - off)
    za[cl] = zf[cl, None] * T.p1[:N][cl]
    if draws is not None:
        pick, w, Hp = draws
        np.add.at(zf, pick, w)
        np.add.at(za, pick, w[:, None] * Hp)
    par = T.parent
    for i in range(N - 1, 0, -1):
        if zf[i] > 0:
            zf[par[i]] += zf[i]
            za[par[i]] += za[i]
    cand = np.flatnonzero((T.status[:N] == 1) & (zf > 0))
    if len(cand) == 0:
        return None
    if len(cand) > S:
        p = np.sqrt(zf[cand] / zf[0])
        cand = rng.choice(cand, size=S, replace=False, p=p / p.sum())
    e = inst.ev(T.A[cand])
    fv, fc = ppol.features(inst, e['A'], e['tl'], e['codes'], e['nun'])
    tgt = np.clip(za[cand] / zf[cand, None], 0.0, 1.0).astype(np.float32)
    return fv, fc, tgt, e['op'].astype(np.float32)


def train(net, buf, steps, bs, lr, seed, inc, log=None):
    """buf: list of (fv, fc, tgt, open-mask) arrays. Cross-entropy to the soft targets on the open assumptions."""
    if len(buf) == 0:
        if log:
            log('    nothing to train on: the search found no score at this budget')
        return float('nan')
    fv, fc, tg, op = [torch.from_numpy(np.concatenate([b[i] for b in buf])) for i in range(4)]
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    net.train()
    run = None
    for s in range(steps):
        for pg in opt.param_groups:
            pg['lr'] = lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * s / steps)))
        idx = torch.randint(0, len(fv), (bs,), generator=g)
        lg = net(fv[idx], fc[idx], inc)
        l = (F.binary_cross_entropy_with_logits(lg, tg[idx], reduction='none') * op[idx]).sum() / op[idx].sum()
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run = l.item() if run is None else 0.98 * run + 0.02 * l.item()
    net.eval()
    if log:
        log('    trained %d steps on %d states: running loss %.4f' % (steps, len(fv), run))
    return run


def score_tree(inst, policy, marks, rng, pstar, lz, rho, eps, n_draw, root=None, delta=0.05):
    """Grow one tree to each mark; score its books and the audit against the exact answer (pstar, lz), AFTER the fact.
    Returns per mark: scalars, and the arrays lo, hi, pt (certain) and alo, ahi, apt (audit) over the open assumptions."""
    T = new_tree(inst, 2 * max(marks) + 8, root)
    op = T.A[0] == UN
    out = {}
    for mk in marks:
        ptree.grow(T, policy, mk, rng, rho=rho, eps=eps)
        bk = ptree.books(T)
        au = ptree.audit(T, bk, n_draw, rng, delta)
        apt = np.clip(au.get('pt', bk['pt']), au['lo'], au['hi'])
        r = dict(opened=T.opened, calls=T.calls, w_cert=float((bk['hi'] - bk['lo'])[op].mean()),
                 w_aud=float((au['hi'] - au['lo'])[op].mean()), unfound_ub=float(bk['unfound_ub']),
                 n_open=bk['n_open'], n_closed=bk['n_closed'], logZ_lo=bk['logZ_lo'], logZ_hi=bk['logZ_hi'],
                 lo=bk['lo'][op], hi=bk['hi'][op], pt=bk['pt'][op], alo=au['lo'][op], ahi=au['hi'][op], apt=apt[op])
        if pstar is not None:
            tol = 1e-9
            ps = pstar[op]
            r.update(viol_cert=int(((ps < r['lo'] - tol) | (ps > r['hi'] + tol)).sum()),
                     viol_Z=int(not (bk['logZ_lo'] - 1e-7 <= lz <= bk['logZ_hi'] + 1e-7)),
                     miss_aud=int(((ps < r['alo'] - tol) | (ps > r['ahi'] + tol)).sum()),
                     err_pt=float(np.abs(r['pt'] - ps).mean()), err_aud=float(np.abs(r['apt'] - ps).mean()),
                     found=float(np.exp(np.log(max(bk['Z'], 1e-300)) + T.logU[0] - lz)))
        out[mk] = r
    return out, T


def score_is(inst, policy, Kmarks, rng, pstar, eps):
    logw, bel, calls = ptree.run_is(inst, policy, max(Kmarks), rng, eps)
    sn = ptree.snis(logw, bel, Kmarks)
    return {K: dict(err=float(np.abs(sn[K]['b'] - pstar).mean()), ess=sn[K]['ess'], calls=calls * K / max(Kmarks))
            for K in Kmarks}


def route_logprob(inst, policy, H, rng, eps):
    """log of the rollout policy's probability of producing each complete assignment in H [B, n] (its own order, its
    value probabilities with the floor, the books' exact numbers once the state is closed)."""
    B, n = H.shape
    A = np.full((B, n), UN, dtype=np.int8)
    lp = np.zeros(B)
    active = np.ones(B, dtype=bool)
    while active.any():
        ia = np.flatnonzero(active)
        e = inst.ev(A[ia])
        d = e['closed']
        if d.any():
            r = ia[d]
            p1 = np.clip(e['p1'][d], 1e-300, 1.0)
            opn = e['op'][d]
            h = H[r]
            with np.errstate(divide='ignore'):
                lp[r] += (opn * np.where(h == 1, np.log(p1), np.log(np.clip(1 - e['p1'][d], 1e-300, 1.0)))).sum(1)
            active[r] = False
        if (~d).any():
            sub = {k: v[~d] for k, v in e.items()}
            q, j = policy(inst, sub, rng)
            r = ia[~d]
            ar = np.arange(len(r))
            qj = log_mix(np.clip(q[ar, j], 0.0, 1.0), eps)
            v = H[r, j]
            with np.errstate(divide='ignore'):
                lp[r] += np.where(v == 1, np.log(qj), np.log(1 - qj))
            A[r, j] = v
    return lp


def one_pass(inst, policy):
    """The rollout policy's own one-pass belief: its value head at the root (one network call)."""
    e = inst.ev(np.full((1, inst.n), UN, dtype=np.int8))
    return policy.q(inst, e)[0] if hasattr(policy, 'q') else e['p1'][0]
