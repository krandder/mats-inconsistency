"""World FA core: generator, tokens, exact Bayes and references, batched in torch (CPU or GPU, float64 maths).

EPISODE. N traders, T resolved history rounds, one query round.
  v_t ~ N(0, 1) per round; report r_{t,i} = v_t + e_{t,i}; e_t ~ N(0, Sigma) iid over rounds (history and query).
  History tokens show r_{t,i} and v_t; the query round shows r_i only; target v (the query round's value).
  Sigma is drawn once per episode from one of two priors (cfg['prior']):

  (a) source graphs (finite family). A random partition of the traders into blocks of size <= m, with
      p(partition) proportional to prod_B alpha_{|B|}. Each block B draws a marginal level s2_B from the grid A_M and,
      if |B| >= 2, a shared fraction f_B from the grid A_F. Every member has error variance s2_B; members of a block
      share one source with variance f_B s2_B (optional signs sigma_i = -1 w.p. a_psign) and keep idiosyncratic
      variance (1 - f_B) s2_B:  Sigma_B = s2_B [(1 - f_B) I + f_B sigma sigma^T].  f near 1 = near-copies of one
      estimate (the one-hop dedup case); a singleton is an independent trader with the same marginal grid.
      Exact Bayes: p(v | data) prop. to N(v; 0, 1) sum_P prod_B g_B(v), with g_B(v) = sum over the block's levels of
      prior x N(history errors of B) x N(r_B - v 1); the sum over partitions is a DP over subsets on a v grid
      (cost O(#subset pairs x grid), i.e. exponential in N: exact up to N ~ 14). Output: grid mixture (w_k on 384
      grid points spanning >= +-10 sd, kernel sd 0.8 x spacing, weights deconvolved to first order so the mixture
      equals the posterior density up to O(spacing^4); log-density error <= 2e-3 nats within +-3 sd in tests;
      typically 1e-5, largest near narrow mixture components).
  (b) inverse Wishart. Hyperparameter h ~ pi over a finite list (delta, sig2, rho); nu = N + 1 + delta,
      Psi = delta M, M = sig2 [(1 - rho) I + rho 11^T], Sigma ~ IW(nu, Psi) (so E Sigma = M, and every k x k marginal
      has a law that does not depend on N). Exact Bayes: conjugate posterior IW(nu + T, Psi + S) from the history
      errors; the query errors then follow a multivariate t, and p(v | data) prop. to N(v; 0, 1) (1 + Q(v))^(-(nu'+1)/2)
      with Q quadratic in v. Writing the power as a Gamma scale mixture gives p(v | data) = int p(lam | data)
      N(v; mu(lam), 1/tau(lam)) dlam, a 1-D integral done by the trapezoid rule in log lam (exponentially accurate):
      a Gaussian mixture. A mixture over h is weighted by the matrix-t history evidence times the query evidence.

TOKENS (one per (trader, round) report, plus one query token; no positions):
  [trader id (d_id) | round id (d_round) | report r | realised v (history; 0 otherwise) | hist flag | query-report
   flag | query flag]. Trader and round ids are random unit vectors drawn per episode. The query round has its own
  round id. The query token carries only the query flag. Layout: history (t, i) at t * Nm + i, query report i at
  Tm * Nm + i, query token last; invalid slots are masked.
"""
import functools
import itertools
import json
import math

import numpy as np
import torch

F64 = torch.float64
LOG2PI = math.log(2 * math.pi)
LOGPI = math.log(math.pi)

DEFAULTS = dict(                          # defaults = the regime chosen by the rewiring audit (PREREG-FA.md, 6)
    prior='a',
    d_id=32, d_round=32,
    T_train=(0, 32), p_T0=0.1,           # training: T = 0 w.p. p_T0, else log-uniform integer in [1, T_train[1]]
    # (a) source graphs
    a_m=(1.0, 3.0, 8.0), a_pm=None,       # marginal error-variance grid (Var v = 1) and its prior (None = uniform)
    a_pm_block=None,                      # prior of the marginal level for blocks of size >= 2 (None = a_pm)
    a_f=(0.25, 0.5, 0.75, 0.9, 0.97), a_pf=(1, 1, 1, 3, 8),   # shared fraction of blocks of size >= 2
    a_alpha=(1.0, 1.0, 2.0, 4.0),         # partition weights alpha_s for block sizes s = 1..m (m = len)
    a_psign=0.25, a_nmax=14, a_grid=384,  # a_psign: P(loading -1) per trader on its block's source
    # (b) inverse Wishart: list of (delta, sig2, rho, weight); rho > 0 = a common source
    b_hyper=((2.0, 4.0, 0.0, 0.5), (2.0, 4.0, 0.6, 0.5)), b_nodes=96,
    # (f) enumerated finite family: every relabelling of each template (exact Bayes = mixture over M hypotheses).
    # template: blocks [[size, rho, n_negative], ...] (other traders independent), marginal variance 'marg' of every
    # trader, optional common source variance 'common', template weight 'w', 'train' (False = held out of training:
    # fresh structures for the memorisation check). Default: GPT-6's isolating case (N = 4: one pair, rho .8: M = 6).
    f_templates=({'blocks': [[2, 0.8, 0]], 'marg': 1.0, 'common': 0.0, 'w': 1.0, 'train': True},),
    f_maxcomp=512,                        # above this many hypotheses the predictive is returned as a grid mixture
    f_templates_byN=None,                 # optional {str(N): templates} overriding f_templates for that N
    tok_col=64,                           # tok_mode 'qcol' / 'qcov': rounds held in each query token's error column
    # tokens: 'report' (one per (trader, round)) or 'column' (2N + 1, Gemini comparator); explicit error channel
    tok_mode='report', tok_err=True,
    id_ortho=False,                       # trader ids orthonormal within each episode (binding aid; else random unit)
    id_mode='random',                     # 'slot': trader i's id is the one-hot slot vector e_i (learned embedding)
    col_random_slots=False,               # qcol / qcov: the T rounds go to random column positions (shared by traders)
    # training curriculum over T (large -> small): list of [first_step, Tmin, Tmax, p_T0]; None = T_train / p_T0
    T_curr=None,
    a_train_alpha=None,                   # (a) partition weights used for TRAINING episodes only (None = a_alpha)
    # references
    d0=2.0,                               # pseudo-count of the plug-in variance references
)


def full_cfg(cfg=None):
    c = dict(DEFAULTS)
    c.update(cfg or {})
    for k in ('a_m', 'a_f', 'a_alpha'):
        c[k] = tuple(float(x) for x in c[k])
    c['b_hyper'] = tuple(tuple(float(x) for x in h) for h in c['b_hyper'])
    assert all(h[0] > 0 for h in c['b_hyper']), 'IW needs nu = N + 1 + delta > N + 1 (delta > 0)'
    c['f_templates'] = tuple(dict(t) for t in c['f_templates'])
    return c


def _probs(p, n):
    p = np.ones(n) if p is None else np.asarray(p, float)
    return p / p.sum()


# ================================================================== partition prior (a)
@functools.lru_cache(maxsize=None)
def part_tables(N, alpha):
    """logZ[n] = log sum over partitions of n labelled traders of prod alpha_|B| (block size <= m), and
    logP[n, s] = log p(block of a given trader has size s | n traders) for the sequential sampler."""
    m = len(alpha)
    la = np.log(np.asarray(alpha))
    logZ = np.full(N + 1, -np.inf); logZ[0] = 0.0
    logP = np.full((N + 1, m), -np.inf)
    for n in range(1, N + 1):
        t = np.array([math.lgamma(n) - math.lgamma(s) - math.lgamma(n - s + 1) + la[s - 1] + logZ[n - s]
                      if s <= n else -np.inf for s in range(1, m + 1)])
        mx = t.max(); logZ[n] = mx + math.log(np.exp(t - mx).sum())
        logP[n] = t - logZ[n]
    return logZ, logP


def pair_probs(N, alpha):
    """p(trader in a block of size >= 2), p(two given traders share a block)."""
    if N < 2:
        return 0.0, 0.0
    _, logP = part_tables(N, alpha)
    ps = np.exp(logP[N])
    s = np.arange(1, len(alpha) + 1)
    return float(ps[1:].sum()), float((ps * (s - 1)).sum() / (N - 1))


# ================================================================== generation
def _unit(shape, dev):
    x = torch.randn(shape, device=dev)
    return x / x.norm(dim=-1, keepdim=True)


def _ortho_ids(E, N, d, dev):
    """(E, N, d) trader ids with orthonormal rows within each episode (random rotation; needs N <= d)."""
    assert N <= d, 'orthonormal ids need N <= d_id'
    q, _ = torch.linalg.qr(torch.randn(E, d, N, device=dev))
    return q.transpose(1, 2).contiguous()


def _as_vec(x, E, dev):
    return torch.as_tensor(x, device=dev).long().expand(E).clone() if torch.as_tensor(x).dim() == 0 else \
        torch.as_tensor(x, device=dev).long()


def sample_T(E, cfg, dev='cpu'):
    lo, hi = cfg['T_train']
    t = torch.exp(math.log(max(lo, 1)) + torch.rand(E, device=dev, dtype=F64) * (math.log(hi + 1) - math.log(max(lo, 1))))
    t = t.long().clamp(max(lo, 1), hi)
    return torch.where(torch.rand(E, device=dev) < cfg['p_T0'], torch.zeros_like(t), t)


def sigma_a(E, Ne, cfg, dev):
    """Prior (a): Sigma (E, Nm, Nm) (identity on padding) and structure arrays."""
    Nm = int(Ne.max())
    alpha = cfg['a_alpha']; m = len(alpha)
    _, logP = part_tables(Nm, alpha)
    logP = torch.as_tensor(logP, device=dev)
    rem = Ne.clone(); pos = torch.zeros(E, dtype=torch.long, device=dev); nb = torch.zeros_like(pos)
    blk_pos = torch.full((E, Nm), -1, dtype=torch.long, device=dev)
    ar = torch.arange(Nm, device=dev)
    for _ in range(Nm):
        act = rem > 0
        if not act.any():
            break
        g = logP[rem.clamp(min=1)] - torch.log(-torch.log(torch.rand(E, m, device=dev, dtype=F64)))
        s = g.argmax(1) + 1
        s = torch.where(act, s, torch.zeros_like(s))
        inb = (ar[None] >= pos[:, None]) & (ar[None] < (pos + s)[:, None])
        blk_pos = torch.where(inb, nb[:, None].expand(E, Nm), blk_pos)
        pos = pos + s; rem = rem - s; nb = nb + act.long()
    valid = ar[None] < Ne[:, None]
    key = torch.where(valid, torch.rand(E, Nm, device=dev, dtype=F64), torch.full((E, Nm), 2.0, device=dev, dtype=F64))
    perm = key.argsort(1)
    blk = torch.gather(blk_pos, 1, perm)                           # block id of trader i (-1 = padding)
    bsize = torch.zeros(E, Nm + 1, dtype=torch.long, device=dev).scatter_add_(1, blk.clamp(min=0), valid.long())
    size = torch.where(valid, torch.gather(bsize, 1, blk.clamp(min=0)), torch.zeros_like(blk))
    pm = torch.as_tensor(_probs(cfg['a_pm'], len(cfg['a_m'])), device=dev)
    pmb = torch.as_tensor(_probs(_pmb(cfg), len(cfg['a_m'])), device=dev)
    pf = torch.as_tensor(_probs(cfg['a_pf'], len(cfg['a_f'])), device=dev)
    lm1 = torch.multinomial(pm.float().expand(E, -1), Nm, replacement=True)       # per block id (singletons)
    lmb = torch.multinomial(pmb.float().expand(E, -1), Nm, replacement=True)      # per block id (size >= 2)
    bsz = bsize[:, :Nm]
    lm = torch.where(bsz >= 2, lmb, lm1)
    lf = torch.multinomial(pf.float().expand(E, -1), Nm, replacement=True)
    Lm = torch.as_tensor(cfg['a_m'], device=dev, dtype=F64); Lf = torch.as_tensor(cfg['a_f'], device=dev, dtype=F64)
    bi = blk.clamp(min=0)
    s2 = Lm[torch.gather(lm, 1, bi)]
    f = torch.where(size >= 2, Lf[torch.gather(lf, 1, bi)], torch.zeros_like(s2))
    sign = torch.where(torch.rand(E, Nm, device=dev) < cfg['a_psign'], -1.0, 1.0).to(F64)
    same = (blk[:, :, None] == blk[:, None, :]) & valid[:, :, None] & valid[:, None, :]
    sh = torch.sqrt(f * s2) * sign
    Sig = same * sh[:, :, None] * sh[:, None, :]
    diag = torch.where(valid, s2, torch.ones_like(s2))
    Sig = Sig + torch.diag_embed(diag - torch.diagonal(Sig, dim1=1, dim2=2))
    return Sig, dict(blk=torch.where(valid, blk, -1), size=size, s2=s2, f=f, sign=sign)


def sigma_b(E, Ne, cfg, dev):
    """Prior (b): Sigma ~ IW(N + 1 + delta, delta M) via the Bartlett decomposition (identity on padding)."""
    Nm = int(Ne.max())
    hyp = torch.as_tensor(cfg['b_hyper'], device=dev, dtype=F64)
    h = torch.multinomial((hyp[:, 3] / hyp[:, 3].sum()).float().expand(E, -1), 1)[:, 0]
    delta, sig2, rho = hyp[h, 0], hyp[h, 1], hyp[h, 2]
    ar = torch.arange(Nm, device=dev)
    valid = ar[None] < Ne[:, None]
    vv = valid[:, :, None] & valid[:, None, :]
    eye = torch.eye(Nm, device=dev, dtype=F64).expand(E, Nm, Nm)
    M = sig2[:, None, None] * ((1 - rho)[:, None, None] * eye + rho[:, None, None])
    Psi = torch.where(vv, delta[:, None, None] * M, eye)
    L = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Psi)))     # chol(Psi^-1)
    nu = Ne.to(F64) + 1 + delta
    dof = nu[:, None] - ar[None].to(F64)                                              # chi2 dof nu - i, i = 0..
    chi = torch.sqrt(2 * torch._standard_gamma(torch.where(valid, dof / 2, torch.ones_like(dof))))
    A = torch.tril(torch.randn(E, Nm, Nm, device=dev, dtype=F64), -1) * vv
    A = A + torch.diag_embed(torch.where(valid, chi, torch.ones_like(chi)))
    Sig = torch.cholesky_inverse(L @ A)
    Sig = 0.5 * (Sig + Sig.transpose(1, 2))
    return Sig, dict(h=h, delta=delta)


# ------------------------------------------------------------------ prior (f): enumerated finite family
def templates_for(cfg, N):
    """The (f) templates used for N traders (f_templates_byN overrides f_templates for listed N)."""
    by = cfg.get('f_templates_byN') or {}
    return by.get(str(int(N)), cfg['f_templates'])


def _tkey(templates):
    return json.dumps([dict(t) for t in templates], sort_keys=True)


@functools.lru_cache(maxsize=None)
def family(N, tkey):
    """All relabellings of every template for N traders. Returns Sig (M, N, N), log prior (M,), template index (M,),
    train flag (M,) as numpy. Hypotheses are distinct labelled covariance matrices; a template's weight is split
    uniformly over its distinct relabellings."""
    templates = json.loads(tkey)
    Sigs, lps, tid, trn = [], [], [], []
    wtot = sum(t.get('w', 1.0) for t in templates)
    for ti, t in enumerate(templates):
        blocks = [tuple(bl) + (0,) * (3 - len(bl)) for bl in t['blocks']]
        assert sum(bl[0] for bl in blocks) <= N, f'template {ti} needs more than N = {N} traders'
        seen = set()
        mats = []

        def rec(j, remaining, acc):
            if j == len(blocks):
                key = frozenset(x[:3] for x in acc)          # a global sign flip of a block is the same Sigma
                if key not in seen:
                    seen.add(key); mats.append(acc)
                return
            k, rho, nneg = blocks[j]
            for mem in itertools.combinations(sorted(remaining), k):
                for neg in itertools.combinations(mem, int(nneg)):
                    pos = frozenset(mem) - frozenset(neg)
                    sk = frozenset([frozenset(neg), pos]) if nneg else None
                    rec(j + 1, remaining - set(mem), acc + [(frozenset(mem), float(rho), sk, frozenset(neg))])
        rec(0, set(range(N)), [])
        for acc in mats:
            S = np.eye(N) * t['marg']
            for mem, rho, _, neg in acc:
                sg = {i: (-1.0 if i in neg else 1.0) for i in mem}
                for i in mem:
                    for j2 in mem:
                        if i != j2:
                            S[i, j2] = t['marg'] * rho * sg[i] * sg[j2]
            S = S + t.get('common', 0.0)
            Sigs.append(S); tid.append(ti); trn.append(bool(t.get('train', True)))
            lps.append(math.log(t.get('w', 1.0) / wtot) - math.log(len(mats)))
    return np.stack(Sigs), np.array(lps), np.array(tid), np.array(trn)


def sigma_f(E, Ne, cfg, dev):
    """Prior (f): draw a hypothesis from the family; cfg['_part'] = 'train' / 'fresh' restricts the draw to the
    training / held-out templates."""
    train_only, fresh_only = cfg.get('_part') == 'train', cfg.get('_part') == 'fresh'
    Nm = int(Ne.max())
    Sig = torch.eye(Nm, device=dev, dtype=F64).repeat(E, 1, 1)
    hidx = torch.zeros(E, dtype=torch.long, device=dev); tidx = torch.zeros_like(hidx)
    for N in sorted(set(Ne.tolist())):
        S, lp, tid, trn = family(N, _tkey(templates_for(cfg, N)))
        keep = trn if train_only else (~trn if fresh_only else np.ones(len(lp), bool))
        assert keep.any(), 'no hypotheses in the requested part of the family'
        p = torch.as_tensor(np.where(keep, np.exp(lp), 0.0), device=dev)
        idx = torch.nonzero(Ne == N)[:, 0]
        h = torch.multinomial((p / p.sum()).float().expand(len(idx), -1), 1)[:, 0]
        Sig[idx, :N, :N] = torch.as_tensor(S, device=dev)[h]
        hidx[idx] = h; tidx[idx] = torch.as_tensor(tid, device=dev)[h]
    return Sig, dict(h=hidx, template=tidx)


SIGMA = {'a': sigma_a, 'b': sigma_b, 'f': sigma_f}


def assemble(Sig, e, v_all, Ne, Te, cfg, dev, extra=None):
    """Batch dict from Sigma (E, Nm, Nm), errors e (E, Tm + 1, Nm) (query round last), values v_all (E, Tm + 1)."""
    E, R, Nm = e.shape
    Tm = R - 1
    vt = torch.arange(Nm, device=dev)[None] < Ne[:, None]
    vr = torch.arange(Tm, device=dev)[None] < Te[:, None]
    e = e * vt[:, None, :]
    e[:, :Tm] = e[:, :Tm] * vr[:, :, None]
    r = (v_all[:, :, None] + e) * vt[:, None, :]
    r[:, :Tm] = r[:, :Tm] * vr[:, :, None]
    if cfg.get('id_mode', 'random') == 'slot':
        assert Nm <= cfg['d_id']
        uid = torch.eye(cfg['d_id'], device=dev)[:Nm].expand(E, Nm, cfg['d_id']).clone()
    else:
        uid = _ortho_ids(E, Nm, cfg['d_id'], dev) if cfg.get('id_ortho') else _unit((E, Nm, cfg['d_id']), dev)
    b = dict(N=Ne, T=Te, vt=vt, vr=vr, Sigma=Sig, v_hist=v_all[:, :Tm] * vr, r_hist=r[:, :Tm], v=v_all[:, Tm],
             r=r[:, Tm], uid=uid, rid=_unit((E, Tm + 1, cfg['d_round']), dev))
    if cfg.get('col_random_slots'):
        b['colperm'] = torch.argsort(torch.rand(E, cfg['tok_col'], device=dev), 1)      # slot of round t
    b.update(extra or {})
    return b


def gen(E, N, T, cfg, dev='cpu'):
    """E episodes. N, T: int or (E,) tensors. Uses the global torch RNG (seed it for determinism)."""
    Ne = _as_vec(N, E, dev); Te = _as_vec(T, E, dev)
    Nm, Tm = int(Ne.max()), int(Te.max())
    Sig, info = SIGMA[cfg['prior']](E, Ne, cfg, dev)
    Lc = torch.linalg.cholesky(Sig)
    e = torch.randn(E, Tm + 1, Nm, device=dev, dtype=F64) @ Lc.transpose(1, 2)
    v_all = torch.randn(E, Tm + 1, device=dev, dtype=F64)
    return assemble(Sig, e, v_all, Ne, Te, cfg, dev, info)


def shuffle_history(b):
    """Shuffled-history control: permute history columns among valid traders (query reports untouched)."""
    E, Nm = b['vt'].shape
    dev = b['vt'].device
    key = torch.where(b['vt'], torch.rand(E, Nm, device=dev), torch.full((E, Nm), 2.0, device=dev))
    perm = key.argsort(1)
    out = dict(b)
    out['r_hist'] = torch.gather(b['r_hist'], 2, perm[:, None, :].expand_as(b['r_hist']))
    out['r_hist'] = torch.where(b['vt'][:, None, :], out['r_hist'], torch.zeros_like(out['r_hist']))
    out['true'] = b
    out['perm'] = perm
    return out


def block_sets(G, N, T, k, fs, cfg, dev='cpu', gate=False, s2_idx=None):
    """One-hop dedup / history-gate sets. G groups x len(fs) arms (arms consecutive within a group).
    Each group: N - k background traders drawn from the prior; k special traders with a common marginal s2 (grid
    A_M) whose errors are sqrt(f s2) eta + sqrt((1 - f) s2) z with the SAME eta, z, background and values in every
    arm (common random numbers); f = 0 is k independent traders with the same marginal accuracy.
    gate=True: every arm's QUERY round is copied from arm 0 (identical query reports), so arms differ only in their
    history (history gate). Returns a batch with extra columns arm, group, fk (shared fraction), s2k."""
    Nb = N - k
    Te = torch.full((G,), T, dtype=torch.long, device=dev)
    if Nb > 0:
        Sb, _ = SIGMA[cfg['prior']](G, torch.full((G,), Nb, device=dev), cfg, dev)
        eb = torch.randn(G, T + 1, Nb, device=dev, dtype=F64) @ torch.linalg.cholesky(Sb).transpose(1, 2)
    Lm = torch.as_tensor(cfg['a_m'], device=dev, dtype=F64)
    li = torch.randint(len(cfg['a_m']), (G,), device=dev) if s2_idx is None else torch.full((G,), s2_idx, device=dev)
    s2 = Lm[li]
    eta = torch.randn(G, T + 1, 1, device=dev, dtype=F64); z = torch.randn(G, T + 1, k, device=dev, dtype=F64)
    v_all = torch.randn(G, T + 1, device=dev, dtype=F64)
    Sigs, es = [], []
    for f in fs:
        ek = torch.sqrt(f * s2)[:, None, None] * eta + torch.sqrt((1 - f) * s2)[:, None, None] * z
        Sk = s2[:, None, None] * ((1 - f) * torch.eye(k, device=dev, dtype=F64) + f)
        Sig = torch.zeros(G, N, N, device=dev, dtype=F64); Sig[:, :k, :k] = Sk
        e = torch.zeros(G, T + 1, N, device=dev, dtype=F64); e[:, :, :k] = ek
        if Nb > 0:
            Sig[:, k:, k:] = Sb; e[:, :, k:] = eb
        Sigs.append(Sig); es.append(e)
    if gate:
        for e in es[1:]:
            e[:, T] = es[0][:, T]
    A = len(fs)
    il = lambda xs: torch.stack(xs, 1).reshape((G * A,) + xs[0].shape[1:])
    Sig, e = il(Sigs), il(es)
    v_rep = v_all.repeat_interleave(A, 0)
    Ne = torch.full((G * A,), N, dtype=torch.long, device=dev)
    extra = dict(arm=torch.arange(A, device=dev).repeat(G), group=torch.arange(G, device=dev).repeat_interleave(A),
                 fk=torch.as_tensor(fs, device=dev, dtype=F64).repeat(G), s2k=s2.repeat_interleave(A), k=k)
    b = assemble(Sig, e, v_rep, Ne, Te.repeat_interleave(A), cfg, dev, extra)
    # the same ids across the arms of a group (only the information structure differs)
    b['uid'] = b['uid'][::A].repeat_interleave(A, 0); b['rid'] = b['rid'][::A].repeat_interleave(A, 0)
    return b


def pair_sets_f(G, N, T, blocks, cfg, dev='cpu', gate=False):
    """Paired episodes for prior (f) tests (A7). G groups x len(blocks) arms (arms consecutive within a group).
    Every trader has error variance 1; arm a puts the k x k correlation matrix blocks[a] on the first k traders, the
    rest independent. Common random numbers: the same standard normals, values and ids in every arm (errors are
    L_a z). gate=True: every arm's QUERY round is copied from arm 0 (identical query reports)."""
    A = len(blocks)
    k = blocks[0].shape[0]
    z = torch.randn(G, T + 1, N, device=dev, dtype=F64)
    v_all = torch.randn(G, T + 1, device=dev, dtype=F64)
    Sigs, es = [], []
    for B in blocks:
        Sig = torch.eye(N, device=dev, dtype=F64).repeat(G, 1, 1)
        Sig[:, :k, :k] = torch.as_tensor(B, device=dev, dtype=F64)
        e = z @ torch.linalg.cholesky(Sig).transpose(1, 2)
        Sigs.append(Sig); es.append(e)
    if gate:
        for e in es[1:]:
            e[:, T] = es[0][:, T]
    il = lambda xs: torch.stack(xs, 1).reshape((G * A,) + xs[0].shape[1:])
    Ne = torch.full((G * A,), N, dtype=torch.long, device=dev)
    extra = dict(arm=torch.arange(A, device=dev).repeat(G), group=torch.arange(G, device=dev).repeat_interleave(A))
    b = assemble(il(Sigs), il(es), v_all.repeat_interleave(A, 0), Ne, torch.full((G * A,), T, dtype=torch.long, device=dev),
                 cfg, dev, extra)
    for key in ('uid', 'rid', 'colperm'):
        if key in b:
            b[key] = b[key][::A].repeat_interleave(A, 0)
    return b


# ================================================================== tokens
# feature layout (offsets after the trader id (d_id) and the round slot (d_round)):
F_R, F_V, F_E, F_HIST, F_QREP, F_QUERY, F_LOGN, F_LOGT = range(8)


def feat_dim(cfg):
    base = cfg['d_id'] + cfg['d_round'] + 8
    mode = cfg.get('tok_mode', 'report')
    if mode == 'qcol':
        return base + cfg['tok_col']
    if mode == 'qcov':
        return base + cfg['tok_col'] + 1 + cfg['d_id']
    return base


def tokens_q(b, cfg):
    """tok_mode 'qcol' / 'qcov' (A5 arms 1-2, binding given): N query-report tokens + the query token, no history
    tokens. Query-report token i = report-mode query token + its own trader's history error column e_{t,i} / sqrt(T)
    (t < T, zero-padded to tok_col rounds; column position = round). 'qcov' adds the empirical covariance row attached
    by trader key: S_ii / T and c_i = sum_{j != i} (S_ij / T) uid_j (exact read-out <c_i, uid_j> with orthonormal ids)."""
    E, Tm, Nm = b['r_hist'].shape
    dev = b['r'].device
    di, dr = cfg['d_id'], cfg['d_round']
    o = di + dr
    base = o + 8
    K = cfg['tok_col']
    assert Tm <= K, f'T = {Tm} exceeds the {K}-round error column'
    X = torch.zeros(E, Nm + 1, feat_dim(cfg), device=dev)
    M = torch.zeros(E, Nm + 1, dtype=torch.bool, device=dev)
    logN = torch.log(b['N'].float())[:, None]; logT = torch.log1p(b['T'].float())[:, None]
    eh, S = hist_stats(b)
    Tc = b['T'].clamp(min=1).to(F64)
    q = X[:, :Nm]
    q[..., :di] = b['uid'].float()
    q[..., di:o] = b['rid'][:, Tm, None].float()
    q[..., o + F_R] = b['r'].float()
    q[..., o + F_QREP] = 1.0
    q[..., o + F_LOGN] = logN; q[..., o + F_LOGT] = logT
    if Tm > 0:
        colv = (eh / torch.sqrt(Tc)[:, None, None]).transpose(1, 2).float()                # (E, Nm, Tm)
        if 'colperm' in b:
            slots = b['colperm'][:, :Tm]                                                     # (E, Tm)
            q[..., base:base + K] = torch.zeros_like(q[..., base:base + K]).scatter(
                2, slots[:, None, :].expand(E, Nm, Tm), colv)
        else:
            q[..., base:base + Tm] = colv
    if cfg.get('tok_mode') == 'qcov':
        Sh = S / Tc[:, None, None] * (b['T'] > 0).to(F64)[:, None, None]
        q[..., base + K] = torch.diagonal(Sh, dim1=1, dim2=2).float()
        off = Sh - torch.diag_embed(torch.diagonal(Sh, dim1=1, dim2=2))
        q[..., base + K + 1:base + K + 1 + di] = (off @ b['uid'].to(F64)).float()
    q *= b['vt'][..., None]
    M[:, :Nm] = b['vt']
    X[:, Nm, o + F_QUERY] = 1.0
    X[:, Nm, o + F_LOGN] = logN[:, 0]; X[:, Nm, o + F_LOGT] = logT[:, 0]
    M[:, Nm] = True
    return X, M


def query_positions(cfg, N, T):
    """Positions of the N query-report tokens (one N and T for the batch)."""
    mode = cfg.get('tok_mode', 'report')
    if mode in ('qcol', 'qcov'):
        return torch.arange(N)
    if mode == 'column':
        return torch.arange(N) + N
    return torch.arange(N) + T * N


def tokens(b, cfg, L=None):
    """X (E, L, F) float32, mask (E, L) bool, query token last.
    tok_mode 'report' (main): one token per (trader, round) report: [uid | round id | r | v | e = r - v | hist |
      0 | 0 | log N | log(1 + T)]; query-round tokens [uid | query-round id | r | 0 | 0 | 0 | 1 | 0 | log N | log(1+T)].
    tok_mode 'column' (comparator, Gemini): 2N + 1 tokens. History token i: [uid_i | h_i | 0 | 0 | 0 | 1 | 0 | 0 | ..]
      with h_i = T^-1/2 sum_t e_{t,i} u_t, u_t = sqrt(d_round) x round id t (shared random per-round features, so
      <h_i, h_j> / d_round ~ S_ij / T); query token i as in report mode (round slot 0).
    tok_err = False drops the explicit error channel (the model must subtract)."""
    if cfg.get('tok_mode', 'report') in ('qcol', 'qcov'):
        return tokens_q(b, cfg)
    E, Tm, Nm = b['r_hist'].shape
    dev = b['r'].device
    di, dr = cfg['d_id'], cfg['d_round']
    o = di + dr
    F = feat_dim(cfg)
    logN = torch.log(b['N'].float())[:, None]
    logT = torch.log1p(b['T'].float())[:, None]
    column = cfg.get('tok_mode', 'report') == 'column'
    nh = Nm if column else Tm * Nm
    L = L or nh + Nm + 1
    X = torch.zeros(E, L, F, device=dev)
    M = torch.zeros(E, L, dtype=torch.bool, device=dev)
    eh, _ = hist_stats(b)
    if column:
        u = b['rid'][:, :Tm].to(F64) * math.sqrt(dr)                                        # (E, Tm, dr)
        hcol = torch.einsum('eti,etd->eid', eh, u) / torch.sqrt(b['T'].clamp(min=1).to(F64))[:, None, None]
        hm = b['vt'] & (b['T'] > 0)[:, None]
        h = X[:, :Nm]
        h[..., :di] = b['uid'].float()
        h[..., di:o] = hcol.float()
        h[..., o + F_HIST] = 1.0
        h[..., o + F_LOGN] = logN; h[..., o + F_LOGT] = logT
        h *= hm[..., None]
        M[:, :Nm] = hm
    elif Tm > 0:
        h = X[:, :Tm * Nm].view(E, Tm, Nm, F)
        hm = b['vr'][:, :, None] & b['vt'][:, None, :]
        h[..., :di] = b['uid'][:, None].float()
        h[..., di:o] = b['rid'][:, :Tm, None].float()
        h[..., o + F_R] = b['r_hist'].float()
        h[..., o + F_V] = b['v_hist'][:, :, None].float()
        if cfg.get('tok_err', True):
            h[..., o + F_E] = eh.float()
        h[..., o + F_HIST] = 1.0
        h[..., o + F_LOGN] = logN[:, :, None]; h[..., o + F_LOGT] = logT[:, :, None]
        h *= hm[..., None]
        M[:, :Tm * Nm] = hm.reshape(E, -1)
    q = X[:, nh:nh + Nm]
    q[..., :di] = b['uid'].float()
    if not column:
        q[..., di:o] = b['rid'][:, Tm, None].float()
    q[..., o + F_R] = b['r'].float()
    q[..., o + F_QREP] = 1.0
    q[..., o + F_LOGN] = logN; q[..., o + F_LOGT] = logT
    q *= b['vt'][..., None]
    M[:, nh:nh + Nm] = b['vt']
    X[:, L - 1, o + F_QUERY] = 1.0
    X[:, L - 1, o + F_LOGN] = logN[:, 0]; X[:, L - 1, o + F_LOGT] = logT[:, 0]
    M[:, L - 1] = True
    return X, M


# ================================================================== shared helpers
def hist_stats(b):
    """History errors (E, Tm, Nm) (zero on padding) and their scatter S (E, Nm, Nm)."""
    eh = (b['r_hist'] - b['v_hist'][:, :, None]) * b['vr'][:, :, None] * b['vt'][:, None, :]
    return eh, eh.transpose(1, 2) @ eh


def _lse(x, dim):
    return torch.logsumexp(x, dim)


def mixture_moments(w, m, v):
    mean = (w * m).sum(-1)
    return mean, (w * (v + m ** 2)).sum(-1) - mean ** 2


# ================================================================== exact Bayes (b)
def _mvlgamma(x, n):
    """log multivariate gamma Gamma_n(x) for per-episode n (E,) and x (E,)."""
    j = torch.arange(int(n.max()), device=x.device, dtype=F64)
    terms = torch.lgamma((x[:, None] - j[None] / 2).clamp(min=1e-300)) * (j[None] < n[:, None])
    return n.to(F64) * (n.to(F64) - 1) / 4 * LOGPI + terms.sum(1)


def bayes_b(b, cfg, use_hist=True, nodes=None):
    """Exact posterior under prior (b). Returns Gaussian mixture (w, m, v) (E, H * nodes) and log evidence (E,)."""
    nodes = nodes or cfg['b_nodes']
    dev = b['r'].device
    E, Nm = b['r'].shape
    Ne = b['N']; Nf = Ne.to(F64)
    vt = b['vt']; vv = vt[:, :, None] & vt[:, None, :]
    u = vt.to(F64)
    eye = torch.eye(Nm, device=dev, dtype=F64).expand(E, Nm, Nm)
    if use_hist:
        _, S = hist_stats(b); Tf = b['T'].to(F64)
    else:
        S = torch.zeros(E, Nm, Nm, device=dev, dtype=F64); Tf = torch.zeros(E, device=dev, dtype=F64)
    r = b['r'] * u
    W, Mu, V, LW = [], [], [], []
    for (delta, sig2, rho, pw) in cfg['b_hyper']:
        Psi = torch.where(vv, delta * sig2 * ((1 - rho) * eye + rho), eye)
        nu = Nf + 1 + delta
        Lp = torch.linalg.cholesky(Psi); Lq = torch.linalg.cholesky(Psi + S)
        ld_p = 2 * torch.log(torch.diagonal(Lp, dim1=1, dim2=2)).sum(1)
        ld_q = 2 * torch.log(torch.diagonal(Lq, dim1=1, dim2=2)).sum(1)
        nup = nu + Tf
        lhist = (-Nf * Tf / 2 * LOGPI + _mvlgamma(nup / 2, Ne) - _mvlgamma(nu / 2, Ne) + nu / 2 * ld_p - nup / 2 * ld_q)
        Pu = torch.cholesky_solve(u[..., None], Lq)[..., 0]
        Pr = torch.cholesky_solve(r[..., None], Lq)[..., 0]
        a = (u * Pu).sum(1); bb = (u * Pr).sum(1); c = (r * Pr).sum(1)
        k = (nup + 1) / 2
        ce = 1 + c - bb ** 2 / a
        c1 = 1 + c

        def logf(lu):                                  # lu (E, G) = log lambda
            lam = torch.exp(lu)
            tau = 1 + 2 * lam * a[:, None]
            return k[:, None] * lu - lam * c1[:, None] + 2 * lam ** 2 * bb[:, None] ** 2 / tau - 0.5 * torch.log(tau)
        lo = torch.log(k / c1) - 40 / k - 3
        hi = torch.log((k + 10 * torch.sqrt(k) + 45) / ce)
        g0 = torch.linspace(0, 1, 600, device=dev, dtype=F64)
        u0 = lo[:, None] + (hi - lo)[:, None] * g0[None]
        f0 = logf(u0)
        keep = f0 > f0.max(1, keepdim=True).values - 42
        i_lo = (keep.float().argmax(1) - 1).clamp(min=0)
        i_hi = (599 - keep.flip(1).float().argmax(1) + 1).clamp(max=599)
        a0 = torch.gather(u0, 1, i_lo[:, None])[:, 0]; a1 = torch.gather(u0, 1, i_hi[:, None])[:, 0]
        g1 = torch.linspace(0, 1, nodes, device=dev, dtype=F64)
        uu = a0[:, None] + (a1 - a0)[:, None] * g1[None]
        lf = logf(uu)
        du = (a1 - a0) / (nodes - 1)
        tw = torch.ones(nodes, device=dev, dtype=F64); tw[0] = tw[-1] = 0.5
        lw = lf + torch.log(du)[:, None] + torch.log(tw)[None]
        lint = _lse(lw, 1)
        lpr = -Nf / 2 * LOGPI - torch.lgamma(k - Nf / 2) - 0.5 * ld_q + lint
        lam = torch.exp(uu); tau = 1 + 2 * lam * a[:, None]
        W.append(lw - lint[:, None]); Mu.append(2 * lam * bb[:, None] / tau); V.append(1 / tau)
        LW.append(math.log(pw) + lhist + lpr)
    LW = torch.stack(LW, 1)
    lev = _lse(LW, 1) - math.log(sum(h[3] for h in cfg['b_hyper']))
    ph = torch.softmax(LW, 1)
    w = torch.cat([torch.exp(W[i]) * ph[:, i:i + 1] for i in range(len(W))], 1)
    return (w, torch.cat(Mu, 1), torch.cat(V, 1)), lev


# ================================================================== exact Bayes (a)
@functools.lru_cache(maxsize=None)
def subset_tables(N, m):
    """Subsets of size <= m (by size: members (nB, s), codes), and the DP plan: per popcount p, the codes S of size p
    (nS,), candidate first blocks B (containing the lowest member of S, |B| <= m) as compact subset indices (nS, c),
    rest codes S \\ B (nS, c), block sizes (nS, c)."""
    by_size, cidx, off = {}, {}, 0
    for s in range(1, min(m, N) + 1):
        mem = np.array(list(itertools.combinations(range(N), s)), dtype=np.int64).reshape(-1, s)
        codes = (1 << mem).sum(1)
        by_size[s] = (mem, codes)
        for j, cd in enumerate(codes):
            cidx[int(cd)] = off + j
        off += len(codes)
    plan = []
    for p in range(1, N + 1):
        Ss = [sum(1 << i for i in c) for c in itertools.combinations(range(N), p)]
        Bs, Rs, Zs = [], [], []
        for S in Ss:
            mem = [i for i in range(N) if S >> i & 1]
            low, rest = mem[0], mem[1:]
            bl, rl, zl = [], [], []
            for s in range(1, min(m, p) + 1):
                for c in itertools.combinations(rest, s - 1):
                    B = (1 << low) | sum(1 << i for i in c)
                    bl.append(cidx[B]); rl.append(S ^ B); zl.append(s)
            Bs.append(bl); Rs.append(rl); Zs.append(zl)
        plan.append((np.array(Ss), np.array(Bs), np.array(Rs), np.array(Zs)))
    return by_size, off, plan


@functools.lru_cache(maxsize=None)
def level_combos(s, cfg_key):
    """Level combinations of a block of size s: (d (C, s) idiosyncratic variances, sh (C,) shared variance,
    sig (C, s) signs, logprior (C,))."""
    a_m, a_pm, a_f, a_pf, psign, a_pmb = cfg_key
    pm = _probs(a_pm if s == 1 else a_pmb, len(a_m)); pf = _probs(a_pf, len(a_f))
    rows = []
    if s == 1:
        for i, m in enumerate(a_m):
            rows.append(([m], 0.0, [1.0], math.log(pm[i])))
    else:
        # Sigma_B depends on the signs only through sigma sigma^T: enumerate patterns with sigma_1 = +1 and give each
        # the prior mass of the pattern and of its global flip (halves the work)
        signs = [(1.0,) * s] if psign == 0 else [(1.0,) + t for t in itertools.product((1.0, -1.0), repeat=s - 1)]
        for i, m in enumerate(a_m):
            for j, f in enumerate(a_f):
                for sg in signs:
                    nneg = sum(x < 0 for x in sg)
                    lp = math.log(pm[i]) + math.log(pf[j])
                    if psign > 0:
                        lp += float(np.logaddexp(nneg * math.log(psign) + (s - nneg) * math.log(1 - psign),
                                                 (s - nneg) * math.log(psign) + nneg * math.log(1 - psign)))
                    rows.append(([(1 - f) * m] * s, f * m, list(sg), lp))
    d = np.array([r[0] for r in rows]); sh = np.array([r[1] for r in rows])
    sig = np.array([r[2] for r in rows]); lp = np.array([r[3] for r in rows])
    return d, sh, sig, lp


def _cfg_key(cfg):
    t = lambda x: None if x is None else tuple(float(y) for y in x)
    return (cfg['a_m'], t(cfg['a_pm']), cfg['a_f'], t(cfg['a_pf']), float(cfg['a_psign']), t(_pmb(cfg)))


def _pmb(cfg):
    return cfg['a_pm'] if cfg.get('a_pm_block') is None else cfg['a_pm_block']


def block_coefs(S, r, Tf, cfg, N, budget=1 << 24):
    """For every subset B (|B| <= m) and level combo c: log g_{B,c}(v) = const + lin v - 0.5 quad v^2.
    Returns per size s: (const (E, nB, C), lin (E, nB, C), quad (C,)) and the subset table."""
    dev = S.device
    m = len(cfg['a_alpha'])
    by_size, _, _ = subset_tables(N, m)
    out = {}
    E = S.shape[0]
    for s, (mem, _) in by_size.items():
        d, sh, sig, lp = (torch.as_tensor(x, device=dev, dtype=F64) for x in level_combos(s, _cfg_key(cfg)))
        invd = 1 / d; w = sig * invd
        kk = 1 + sh * invd.sum(1)
        logdet = torch.log(d).sum(1) + torch.log(kk)
        quad = invd.sum(1) - sh * w.sum(1) ** 2 / kk                                     # (C,)
        memt = torch.as_tensor(mem, device=dev)
        C = len(lp)
        consts, lins = [], []
        step = max(1, budget // max(1, E * C * s * s))
        for j in range(0, len(mem), step):
            mj = memt[j:j + step]
            SB = S[:, mj[:, :, None], mj[:, None, :]]                                    # (E, nb, s, s)
            rB = r[:, mj]                                                                # (E, nb, s)
            trS = torch.einsum('ebi,ci->ebc', torch.diagonal(SB, dim1=2, dim2=3), invd)
            wSw = torch.einsum('ebij,ci,cj->ebc', SB, w, w)
            hq = trS - sh * wSw / kk
            rw = torch.einsum('ebi,ci->ebc', rB, w)
            lin = torch.einsum('ebi,ci->ebc', rB, invd) - sh * rw * w.sum(1) / kk
            q0 = torch.einsum('ebi,ci->ebc', rB ** 2, invd) - sh * rw ** 2 / kk
            const = (lp - 0.5 * (Tf[:, None, None] + 1) * (s * LOG2PI + logdet) - 0.5 * hq - 0.5 * q0)
            consts.append(const); lins.append(lin)
        out[s] = (torch.cat(consts, 1), torch.cat(lins, 1), quad)
    return out


def _logg_grid(coefs, grid, budget=1 << 25):
    """log g_B on the grid: (E, nBtot, G) in compact subset order."""
    outs = []
    E, G = grid.shape
    for s in sorted(coefs):
        const, lin, quad = coefs[s]
        nB, C = const.shape[1], const.shape[2]
        step = max(1, budget // max(1, E * C * G))
        for j in range(0, nB, step):
            x = (const[:, j:j + step, :, None] + lin[:, j:j + step, :, None] * grid[:, None, None, :]
                 - 0.5 * quad[None, None, :, None] * grid[:, None, None, :] ** 2)
            outs.append(_lse(x, 2))
    return torch.cat(outs, 1)


def _dp(logg, N, cfg):
    """log sum_P prod_B alpha_|B| g_B on the grid for the full trader set: (E, G)."""
    m = len(cfg['a_alpha'])
    la = torch.as_tensor(np.log(np.r_[1.0, cfg['a_alpha']]), device=logg.device, dtype=F64)
    _, _, plan = subset_tables(N, m)
    E, _, G = logg.shape
    Z = torch.full((E, 1 << N, G), -math.inf, device=logg.device, dtype=F64)
    Z[:, 0] = 0.0
    for Ss, Bs, Rs, Zs in plan:
        Bt = torch.as_tensor(Bs, device=logg.device); Rt = torch.as_tensor(Rs, device=logg.device)
        x = logg[:, Bt] + Z[:, Rt] + la[torch.as_tensor(Zs, device=logg.device)][None, :, :, None]
        Z[:, torch.as_tensor(Ss, device=logg.device)] = _lse(x, 2)
    return Z[:, (1 << N) - 1]


def bayes_a(b, cfg, use_hist=True, G=None, return_logpost=False):
    """Exact posterior under prior (a) for a batch with ONE trader count N. Returns grid mixture (w, m, v) (E, G)
    and log evidence (E,)."""
    G = G or cfg['a_grid']
    N = int(b['N'][0])
    assert bool((b['N'] == N).all()), 'bayes_a needs one N per call (use bayes())'
    assert N <= cfg['a_nmax'], f'prior (a) exact Bayes capped at N <= {cfg["a_nmax"]}'
    dev = b['r'].device
    E = len(b['r'])
    r = b['r'][:, :N]
    if use_hist:
        _, S = hist_stats(b); Tf = b['T'].to(F64)
    else:
        S = torch.zeros(E, N, N, device=dev, dtype=F64); Tf = torch.zeros(E, device=dev, dtype=F64)
    S = S[:, :N, :N]
    coefs = block_coefs(S, r, Tf, cfg, N)
    logZN = part_tables(N, cfg['a_alpha'])[0][N]

    def post(grid):
        lg = _logg_grid(coefs, grid)
        return -0.5 * grid ** 2 - 0.5 * LOG2PI + _dp(lg, N, cfg) - logZN
    return grid_mixture(post, r.mean(1) / (1 + 1.0 / N), G, return_logpost)


def grid_mixture(post, c0, G, return_logpost=False):
    """Posterior on an adaptive grid from an unnormalised log density post(grid (E, n)) -> (E, n).
    Coarse passes (256 points) until the grid covers (edges 40 nats below the max) and resolves (sd > 4 spacings)
    the posterior; then one fine pass (G points) over the region within 40 nats of the max, and at least
    mean +- 10 sd. Returns the grid mixture (w, m, v) (E, G), log evidence (E,) [and (grid, log density)]."""
    dev = c0.device
    E = len(c0)
    lin = lambda n: torch.linspace(-1, 1, n, device=dev, dtype=F64)[None]
    cen, half = c0, torch.full_like(c0, 8.0)
    n0 = 256
    for _ in range(12):
        grid = cen[:, None] + half[:, None] * lin(n0)
        lp = post(grid)
        dx = grid[:, 1] - grid[:, 0]
        w = torch.softmax(lp, 1)
        mu = (w * grid).sum(1); sd = torch.sqrt((w * (grid - mu[:, None]) ** 2).sum(1))
        tails = torch.maximum(lp[:, 0], lp[:, -1]) < lp.max(1).values - 40
        res = sd > 4 * dx
        if bool((tails & res).all()):
            break
        # widen if the tails are cut (keeps resolution up to ~30 sd half-width); re-centre and shrink if unresolved
        new_half = torch.where(~tails, 1.5 * half, 10 * torch.maximum(sd, dx))
        cen = torch.where(tails & res, cen, mu); half = torch.where(tails & res, half, new_half)
    keep = lp > lp.max(1, keepdim=True).values - 40
    i_lo = (keep.float().argmax(1) - 1).clamp(min=0); i_hi = (n0 - 1 - keep.flip(1).float().argmax(1) + 1).clamp(max=n0 - 1)
    lo = torch.minimum(torch.gather(grid, 1, i_lo[:, None])[:, 0], mu - 10 * sd)
    hi = torch.maximum(torch.gather(grid, 1, i_hi[:, None])[:, 0], mu + 10 * sd)
    grid = (lo + hi)[:, None] / 2 + (hi - lo)[:, None] / 2 * lin(G)
    lp = post(grid)
    dx = grid[:, 1] - grid[:, 0]
    lev = _lse(lp, 1) + torch.log(dx)
    h2 = (0.8 * dx) ** 2
    # first-order deconvolution of the Gaussian kernel: weights prop. to p - (h2 / 2) p'' (p''/p from central
    # differences of the exact log density), so the mixture equals p up to O(h^4) instead of p * N(0, h2)
    d1 = torch.zeros_like(lp); d2 = torch.zeros_like(lp)
    d1[:, 1:-1] = (lp[:, 2:] - lp[:, :-2]) / (2 * dx[:, None])
    d2[:, 1:-1] = (lp[:, 2:] - 2 * lp[:, 1:-1] + lp[:, :-2]) / dx[:, None] ** 2
    corr = (1 - 0.5 * h2[:, None] * (d1 ** 2 + d2)).clamp(min=1e-3)
    w = torch.softmax(lp + torch.log(corr), 1)
    out = (w, grid, h2[:, None].expand(E, G).clone())
    if return_logpost:
        return out, lev, (grid, lp - lev[:, None])
    return out, lev


def diag_bayes(b, cfg, G=256):
    """Correlation-blind Bayes: the exact posterior under the same prior with every off-diagonal of Sigma set to 0
    (each trader keeps the prior's marginal law of its own variance and learns it from its own history).
    (a): the family restricted to singletons (marginal level ~ A_M). (b): per-trader IW 1x1 marginal,
    sigma_i^2 ~ IG((delta + 2) / 2, delta sig2 / 2), so each query error is Student-t; v by grid quadrature."""
    if cfg['prior'] == 'a':
        c = dict(cfg); c['a_alpha'] = (1.0,)
        return bayes(b, c, True)
    dev = b['r'].device
    u = b['vt'].to(F64); r = b['r'] * u
    _, S = hist_stats(b)
    Sd = torch.diagonal(S, dim1=1, dim2=2); Tf = b['T'].to(F64)
    hyp = cfg['b_hyper']
    lpi = [math.log(h[3] / sum(x[3] for x in hyp)) for h in hyp]
    terms = []
    for (delta, sig2, rho, _) in hyp:
        a0, b0 = (delta + 2) / 2, delta * sig2 / 2
        a1, b1 = a0 + Tf[:, None] / 2 + 0 * Sd, b0 + Sd / 2                                # (E, Nm)
        lh = (a0 * math.log(b0) - math.lgamma(a0) + torch.lgamma(a1) - a1 * torch.log(b1) - Tf[:, None] / 2 * LOG2PI)
        terms.append(((lh * u).sum(1), a1, b1))

    def post(grid):
        out = []
        for i, (lh, a1, b1) in enumerate(terms):
            x2 = (r[:, None, :] - grid[:, :, None]) ** 2                                   # (E, G, Nm)
            lt = ((torch.lgamma(a1 + 0.5) - torch.lgamma(a1) - 0.5 * torch.log(2 * math.pi * b1))[:, None, :]
                  - (a1 + 0.5)[:, None, :] * torch.log1p(x2 / (2 * b1[:, None, :])))
            out.append(lpi[i] + lh[:, None] + (lt * u[:, None, :]).sum(2))
        return -0.5 * grid ** 2 - 0.5 * LOG2PI + _lse(torch.stack(out, 0), 0)
    return grid_mixture(post, r.sum(1) / (b['N'].to(F64) + 1), G)[0]


@functools.lru_cache(maxsize=8)
def _family_mats(N, tkey, diag, dev):
    S, lp, tid, trn = family(N, tkey)
    S = torch.as_tensor(S, device=dev)
    if diag:
        S = torch.diag_embed(torch.diagonal(S, dim1=1, dim2=2))
    L = torch.linalg.cholesky(S)
    Pi = torch.cholesky_inverse(L)
    ld = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
    Lq = torch.linalg.cholesky(S + 1.0)                           # marginal of the query reports: Sigma_h + 11^T
    Pq = torch.cholesky_inverse(Lq)
    ldq = 2 * torch.log(torch.diagonal(Lq, dim1=1, dim2=2)).sum(1)
    u = Pi.sum(2)                                                 # Sigma_h^-1 1  (M, N)
    return (Pi, ld, Pq, ldq, u, torch.as_tensor(lp, device=dev), torch.as_tensor(trn, device=dev),
            torch.as_tensor(tid, device=dev))


def bayes_f(b, cfg, use_hist=True, diag=False, train_only=False, query_weights=True, return_post=False):
    """Exact posterior under prior (f) for a batch with ONE trader count N:
        p(h | history, r) prop. to p(h) N(history errors; 0, Sigma_h) N(r; 0, Sigma_h + 11^T)
        p(v | h, r) = N(m_h, V_h),  V_h = 1 / (1 + 1' Sigma_h^-1 1),  m_h = V_h 1' Sigma_h^-1 r.
    The query-marginal factor is part of the weights (query_weights=False drops it: the INCORRECT history-only
    weighting, kept for the test that shows the difference). diag: the family with correlations removed;
    train_only: the family restricted to training templates. Returns the mixture (E, M) (or a grid mixture if
    M > f_maxcomp), log evidence, and with return_post the hypothesis posterior (E, M) and beta* = E[Sigma^-1 1]."""
    N = int(b['N'][0])
    assert bool((b['N'] == N).all())
    dev = b['r'].device
    Pi, ld, Pq, ldq, u, lp, trn, _ = _family_mats(N, _tkey(templates_for(cfg, N)), diag, str(dev))
    r = b['r'][:, :N]
    if use_hist:
        _, S = hist_stats(b); S = S[:, :N, :N]; Tf = b['T'].to(F64)
    else:
        S = torch.zeros(len(r), N, N, device=dev, dtype=F64); Tf = torch.zeros(len(r), device=dev, dtype=F64)
    lw = lp[None] - 0.5 * Tf[:, None] * (N * LOG2PI + ld[None]) - 0.5 * torch.einsum('mij,eij->em', Pi, S)
    if query_weights:
        lw = lw - 0.5 * (N * LOG2PI + ldq[None]) - 0.5 * torch.einsum('ei,mij,ej->em', r, Pq, r)
    if train_only:
        lw = torch.where(trn[None], lw, torch.full_like(lw, -math.inf))
    lev = _lse(lw, 1)
    w = torch.softmax(lw, 1)
    a = u.sum(1)
    V = (1 / (1 + a))[None].expand_as(w)
    m = (r @ u.T) * V
    if w.shape[1] > cfg['f_maxcomp']:
        def post(grid):
            out = []
            for j in range(0, grid.shape[1], 64):
                g = grid[:, j:j + 64]
                out.append(_lse(torch.log(w.clamp(min=1e-300))[:, :, None] - 0.5 * torch.log(2 * math.pi * V)[:, :, None]
                                - 0.5 * (g[:, None, :] - m[:, :, None]) ** 2 / V[:, :, None], 1))
            return torch.cat(out, 1)
        mix = grid_mixture(post, (w * m).sum(1), 256)[0]
    else:
        mix = (w, m, V.clone())
    if return_post:
        return mix, lev, w, w @ u
    return mix, lev


def bayes(b, cfg, use_hist=True, **kw):
    """Exact Bayes predictive (w, m, v) for any batch (groups by N for priors (a) and (f); pads mixtures).
    kw (prior f only): diag, train_only."""
    if cfg['prior'] == 'b':
        return bayes_b(b, cfg, use_hist)[0]
    if cfg['prior'] == 'f':
        E = len(b['r']); dev = b['r'].device
        outs = {}
        for N in sorted(set(b['N'].tolist())):
            idx = torch.nonzero(b['N'] == N)[:, 0]
            M = len(family(N, _tkey(templates_for(cfg, N)))[1])
            ch = len(idx) if M <= cfg['f_maxcomp'] else max(1, (1 << 25) // (M * 64))
            parts = [bayes_f(take(b, idx[j:j + ch]), cfg, use_hist, **kw)[0] for j in range(0, len(idx), ch)]
            outs[N] = (idx, tuple(torch.cat([p[k] for p in parts]) for k in range(3)))
        K = max(o[1][0].shape[1] for o in outs.values())
        w = torch.zeros(E, K, device=dev, dtype=F64); m = torch.zeros_like(w); v = torch.ones_like(w)
        for idx, (ww, mm, vv) in outs.values():
            k = ww.shape[1]
            w[idx, :k], m[idx, :k], v[idx, :k] = ww, mm, vv
        return w, m, v
    E = len(b['r']); dev = b['r'].device
    G = cfg['a_grid']
    w = torch.zeros(E, G, device=dev, dtype=F64); m = torch.zeros_like(w); v = torch.ones_like(w)
    for N in sorted(set(b['N'].tolist())):
        idx = torch.nonzero(b['N'] == N)[:, 0]
        sub = take(b, idx)
        ch = a_chunk(N, cfg)
        for j in range(0, len(idx), ch):
            sj = take(sub, torch.arange(j, min(j + ch, len(idx)), device=dev))
            (ww, mm, vv), _ = bayes_a(sj, cfg, use_hist)
            ii = idx[j:j + ch]
            w[ii], m[ii], v[ii] = ww, mm, vv
    return w, m, v


def a_chunk(N, cfg, budget=1 << 25):
    """Episodes per bayes_a call so the DP's largest tensor stays under ~budget float64 entries."""
    _, _, plan = subset_tables(N, len(cfg['a_alpha']))
    big = max(max(Bs.size for _, Bs, _, _ in plan), 1 << N)
    return int(max(1, min(256, budget // (big * cfg['a_grid']))))


def take(b, idx):
    """Sub-batch (trims trader / round padding to the sub-batch maxima)."""
    out = {}
    Nm = int(b['N'][idx].max()); Tm = int(b['T'][idx].max())
    for k, x in b.items():
        if not torch.is_tensor(x) or x.dim() == 0 or x.shape[0] != len(b['N']):
            out[k] = x
            continue
        x = x[idx]
        if k in ('vt', 'r', 'uid'):
            x = x[:, :Nm]
        elif k in ('Sigma',):
            x = x[:, :Nm, :Nm]
        elif k in ('vr', 'v_hist'):
            x = x[:, :Tm]
        elif k == 'r_hist':
            x = x[:, :Tm, :Nm]
        elif k == 'rid':
            x = torch.cat([x[:, :Tm], x[:, -1:]], 1)
        elif k in ('blk', 'size', 's2', 'f', 'sign', 'perm'):
            x = x[:, :Nm]
        out[k] = x
    return out


# ================================================================== references
def prior_mean_sigma(N, cfg, dev='cpu'):
    """E Sigma under the prior for N traders (N x N)."""
    eye = torch.eye(N, device=dev, dtype=F64)
    if cfg['prior'] == 'f':
        S, lp, _, _ = family(N, _tkey(templates_for(cfg, N)))
        return torch.as_tensor((np.exp(lp)[:, None, None] * S).sum(0), device=dev)
    if cfg['prior'] == 'b':
        hyp = cfg['b_hyper']; tot = sum(h[3] for h in hyp)
        return sum(h[3] / tot * h[1] * ((1 - h[2]) * eye + h[2]) for h in hyp)
    pm = _probs(cfg['a_pm'], len(cfg['a_m'])); pmb = _probs(_pmb(cfg), len(cfg['a_m']))
    pf = _probs(cfg['a_pf'], len(cfg['a_f']))
    Em1 = float((pm * np.array(cfg['a_m'])).sum()); Emb = float((pmb * np.array(cfg['a_m'])).sum())
    Ef = float((pf * np.array(cfg['a_f'])).sum())
    pshared, psame = pair_probs(N, cfg['a_alpha'])
    off = psame * Emb * Ef * (1 - 2 * cfg['a_psign']) ** 2
    return ((1 - pshared) * Em1 + pshared * Emb) * eye + off * (1 - eye)


def _gauss(m, v):
    return torch.ones_like(m)[:, None], m[:, None], v[:, None]


def gls(Sig, r, u):
    """Gaussian posterior for v with prior N(0, 1) and r ~ N(v u, Sig) (u = valid mask, padding identity)."""
    L = torch.linalg.cholesky(Sig)
    Pu = torch.cholesky_solve(u[..., None], L)[..., 0]
    a = (u * Pu).sum(1); bb = (r * u * Pu).sum(1)
    return bb / (1 + a), 1 / (1 + a)


def references(b, cfg, with_bayes=True, with_diag=True):
    """dict name -> (w, m, v) torch mixtures. 'bayes' (full), 'bayes_T0' (history ignored: the best rule that uses
    the query reports and the prior only), 'diag_bayes' (exact Bayes with the correlations assumed away: the
    strongest correlation-blind rule), 'prior', 'equal', 'prec', 'gls_plugin', 'oracle'; 'bayes_train' (Bayes under
    the training part of the prior) when training holds structures out."""
    dev = b['r'].device
    E, Nm = b['r'].shape
    u = b['vt'].to(F64); Nf = b['N'].to(F64); Tf = b['T'].to(F64)
    r = b['r'] * u
    eh, S = hist_stats(b)
    d0 = cfg['d0']
    Sbar = torch.stack([_pad(prior_mean_sigma(int(n), cfg, dev), Nm) for n in b['N'].tolist()]) \
        if len(set(b['N'].tolist())) > 1 else _pad(prior_mean_sigma(int(b['N'][0]), cfg, dev), Nm).expand(E, Nm, Nm)
    vv = (u[:, :, None] * u[:, None, :]) > 0
    eye = torch.eye(Nm, device=dev, dtype=F64).expand(E, Nm, Nm)
    out = {}
    out['prior'] = _gauss(torch.zeros(E, device=dev, dtype=F64), torch.ones(E, device=dev, dtype=F64))
    # equal weights; variance of the average error estimated from history (pseudo-count d0 at its prior mean)
    ebar = eh.sum(2) / Nf[:, None]
    s2bar0 = (u[:, :, None] * Sbar * u[:, None, :]).sum((1, 2)) / Nf ** 2
    s2 = (d0 * s2bar0 + (ebar ** 2).sum(1)) / (d0 + Tf)
    rb = r.sum(1) / Nf
    out['equal'] = _gauss(rb / (1 + s2), s2 / (1 + s2))
    # precision weighting ignoring correlations (per-trader variance from history, pseudo-count d0)
    sd2 = (d0 * torch.diagonal(Sbar, dim1=1, dim2=2) + torch.diagonal(S, dim1=1, dim2=2)) / (d0 + Tf[:, None])
    wi = u / sd2
    out['prec'] = _gauss((wi * r).sum(1) / (1 + wi.sum(1)), 1 / (1 + wi.sum(1)))
    # GLS with the regularised sample covariance (pseudo-count N + 2 at the prior mean)
    k0 = Nf + 2
    Sh = (k0[:, None, None] * Sbar + S) / (k0 + Tf)[:, None, None]
    Sh = torch.where(vv, Sh, eye)
    out['gls_plugin'] = _gauss(*gls(Sh, r, u))
    out['oracle'] = _gauss(*gls(torch.where(vv, b['Sigma'], eye), r, u))
    if with_diag:
        out['diag_bayes'] = bayes(b, cfg, True, diag=True) if cfg['prior'] == 'f' else diag_bayes(b, cfg)
    if with_bayes:
        out['bayes'] = bayes(b, cfg, True)
        out['bayes_T0'] = bayes(b, cfg, False)
        if cfg['prior'] == 'f' and not all(t.get('train', True) for t in cfg['f_templates']):
            out['bayes_train'] = bayes(b, cfg, True, train_only=True)     # Bayes under the training family only
        if cfg['prior'] == 'a' and cfg.get('a_train_alpha') is not None:
            out['bayes_train'] = bayes(b, dict(cfg, a_alpha=tuple(float(x) for x in cfg['a_train_alpha'])), True)
    return out


def _pad(M, Nm):
    n = M.shape[0]
    if n == Nm:
        return M
    out = torch.eye(Nm, device=M.device, dtype=M.dtype)
    out[:n, :n] = M
    return out


def nll_t(pred, y):
    w, m, v = pred
    lp = torch.log(w.clamp(min=1e-300)) - 0.5 * torch.log(2 * math.pi * v) - 0.5 * (y[:, None] - m) ** 2 / v
    lp = torch.where(w > 0, lp, torch.full_like(lp, -math.inf))
    return -_lse(lp, 1)


def beta_star(b, cfg):
    """Dense auxiliary target (critique item 5): per-trader precision weights beta* = E[Sigma^-1 1 | data] (E, Nm).
    (f): exact, posterior over hypotheses given history AND query reports. (b): E[Sigma^-1 | history] 1 =
    nu' Psi'^-1 1 (history only). (a): not available (the subset DP does not give posterior moments of Sigma^-1)."""
    E, Nm = b['r'].shape
    dev = b['r'].device
    out = torch.zeros(E, Nm, device=dev, dtype=F64)
    if cfg['prior'] == 'f':
        for N in sorted(set(b['N'].tolist())):
            idx = torch.nonzero(b['N'] == N)[:, 0]
            out[idx, :N] = bayes_f(take(b, idx), cfg, return_post=True)[3]
        return out
    if cfg['prior'] == 'b':
        assert len(cfg['b_hyper']) == 1, 'beta* for (b) implemented for one hyperparameter'
        delta, sig2, rho, _ = cfg['b_hyper'][0]
        vt = b['vt']; vv = vt[:, :, None] & vt[:, None, :]
        eye = torch.eye(Nm, device=dev, dtype=F64).expand(E, Nm, Nm)
        _, S = hist_stats(b)
        Psi = torch.where(vv, delta * sig2 * ((1 - rho) * eye + rho), eye) + S
        nup = b['N'].to(F64) + 1 + delta + b['T'].to(F64)
        return nup[:, None] * torch.cholesky_solve(vt.to(F64)[..., None], torch.linalg.cholesky(Psi))[..., 0] * vt
    raise NotImplementedError('beta* is not available for prior (a)')
