"""Shrunken copy of pfn-dedup/world.py (that file is not modified or imported, so its run is untouched).

Same generative law: o_{s,q} = y_q + z_{g(s),q} + e_{s,q}; y ~ N(0,1); z ~ N(0,SG2) per group and question;
e ~ N(0,SE2); partition prior 0.7 Ewens(1) + 0.3 all-singletons. Shrunk to S=3 sources, Q=3 questions
(2 labelled + query), no duplicate rows, identity embeddings of dim DS=8. Tokens: one per (source, question)
plus one query token, T = S*Q + 1 = 10; features [source id (DS), question id (DS), o, label, missing, query flag].

Given the partition the whole input (tokens and the target y*) is exactly Gaussian, so the input distribution
is a finite Gaussian mixture; `component_gaussians` returns the exact mean/covariance per partition.
"""
import itertools, math
import numpy as np
import torch

S, Q, QL = 3, 3, 2
SG2, SE2 = 1.0, 0.25
DS = 8
T = S * Q + 1
F = 2 * DS + 4


def all_partitions(n):
    out = []
    def rec(i, labels, m):
        if i == n:
            out.append(tuple(labels)); return
        for g in range(m + 1):
            rec(i + 1, labels + [g], max(m, g + 1))
    rec(0, [], 0)
    return np.array(out, dtype=np.int64)


PARTS = all_partitions(S)


def ewens(labels, alpha=1.0):
    n = len(labels)
    sizes = np.bincount(labels)
    num = alpha ** len(sizes) * np.prod([math.factorial(m - 1) for m in sizes])
    den = np.prod([alpha + i for i in range(n)])
    return num / den


_E = np.array([ewens(p) for p in PARTS])
_SING = np.array([len(set(p)) == S for p in PARTS])
PRIOR = 0.7 * _E / _E.sum() + 0.3 * _SING
PRIOR = PRIOR / PRIOR.sum()


def sigma_of(labels):
    B = (labels[:, None] == labels[None, :]).astype(float)
    return SE2 * np.eye(len(labels)) + SG2 * B


SIG = np.stack([sigma_of(p) for p in PARTS])
SIGI = np.linalg.inv(SIG)
LOGDET = np.linalg.slogdet(SIG)[1]
SIGQ = SIG + 1.0
SIGQI = np.linalg.inv(SIGQ)
LOGDETQ = np.linalg.slogdet(SIGQ)[1]
ONE = np.ones(S)
PREC = 1.0 + np.einsum('i,pij,j->p', ONE, SIGI, ONE)
WVEC = np.einsum('pij,j->pi', SIGI, ONE)


# ---------------------------------------------------------------- sampling (torch, fresh every batch)
class Sampler:
    def __init__(self, seed, device='cpu'):
        self.g = torch.Generator(device=device); self.g.manual_seed(seed)
        self.dev = device
        self.prior = torch.tensor(PRIOR, dtype=torch.float32, device=device)
        self.parts = torch.tensor(PARTS, device=device)

    def sample(self, n):
        g, dev = self.g, self.dev
        pidx = torch.multinomial(self.prior, n, replacement=True, generator=g)
        lab = self.parts[pidx]                                              # (n,S)
        y = torch.randn(n, Q, generator=g, device=dev)
        z = torch.randn(n, S, Q, generator=g, device=dev) * math.sqrt(SG2)
        e = torch.randn(n, S, Q, generator=g, device=dev) * math.sqrt(SE2)
        o = y[:, None, :] + torch.gather(z, 1, lab[:, :, None].expand(n, S, Q)) + e
        semb = torch.randn(n, S, DS, generator=g, device=dev) / 4
        qemb = torch.randn(n, Q, DS, generator=g, device=dev) / 4
        X = tokens(semb, qemb, o, y)
        return dict(X=X, ystar=y[:, QL], pidx=pidx, lab=lab, y=y, o=o)


def tokens(semb, qemb, o, y):
    n = o.shape[0]
    X = torch.zeros(n, T, F, dtype=o.dtype, device=o.device)
    for s in range(S):
        for q in range(Q):
            t = s * Q + q
            X[:, t, :DS] = semb[:, s]
            X[:, t, DS:2 * DS] = qemb[:, q]
            X[:, t, 2 * DS] = o[:, s, q]
            if q < QL:
                X[:, t, 2 * DS + 1] = y[:, q]
            else:
                X[:, t, 2 * DS + 2] = 1.0
    X[:, T - 1, DS:2 * DS] = qemb[:, QL]
    X[:, T - 1, 2 * DS + 3] = 1.0
    return X


# ---------------------------------------------------------------- exact input law per partition
def component_gaussians():
    """For every partition p: mean (T*F+1,) and covariance of [flattened tokens; y*], exact.
    Built as an affine map of independent standard normal latents, so it matches the sampler by construction."""
    nlat = S * DS + Q * DS + Q + S * Q + S * Q          # semb, qemb, y, z (per group slot), e
    oS, oQ, oY, oZ, oE = 0, S * DS, S * DS + Q * DS, S * DS + Q * DS + Q, S * DS + Q * DS + Q + S * Q
    out_m, out_C = [], []
    for p in PARTS:
        M = np.zeros((T * F + 1, nlat)); b = np.zeros(T * F + 1)
        def row(t, f): return t * F + f
        for s in range(S):
            for q in range(Q):
                t = s * Q + q
                for i in range(DS):
                    M[row(t, i), oS + s * DS + i] = 0.25
                    M[row(t, DS + i), oQ + q * DS + i] = 0.25
                r = row(t, 2 * DS)
                M[r, oY + q] = 1.0
                M[r, oZ + p[s] * Q + q] = math.sqrt(SG2)
                M[r, oE + s * Q + q] = math.sqrt(SE2)
                if q < QL:
                    M[row(t, 2 * DS + 1), oY + q] = 1.0
                else:
                    b[row(t, 2 * DS + 2)] = 1.0
        for i in range(DS):
            M[row(T - 1, DS + i), oQ + QL * DS + i] = 0.25
        b[row(T - 1, 2 * DS + 3)] = 1.0
        M[T * F, oY + QL] = 1.0                           # y*
        out_m.append(b); out_C.append(M @ M.T)
    return np.stack(out_m), np.stack(out_C)


def shape_components():
    """Partitions grouped into orbits under source permutations (the model and the propagator are equivariant).
    Returns representative partition indices, orbit probabilities, and the list of source permutations."""
    reps, probs, seen = [], [], set()
    perms = list(itertools.permutations(range(S)))
    for i, p in enumerate(PARTS):
        if i in seen: continue
        orbit = set()
        for pm in perms:
            relab = np.array(p)[list(pm)]
            # canonicalise labels (first appearance order)
            mp, can = {}, []
            for l in relab:
                if l not in mp: mp[l] = len(mp)
                can.append(mp[l])
            j = [k for k, q in enumerate(PARTS) if tuple(q) == tuple(can)][0]
            orbit.add(j)
        seen |= orbit
        reps.append(i); probs.append(PRIOR[list(orbit)].sum())
    return np.array(reps), np.array(probs), perms


# ---------------------------------------------------------------- exact Bayes predictive NLL (numpy)
def bayes_nll(o, y):
    """o: (n,S,Q), y: (n,Q). Exact Bayes over partitions; returns NLL of y[:,QL] and predictive mean/var."""
    o = np.asarray(o, float); y = np.asarray(y, float)
    n = len(o)
    ostar = o[:, :, QL]; ystar = y[:, QL]
    R = o[:, :, :QL] - y[:, None, :QL]
    quad = np.einsum('nsq,pst,ntq->np', R, SIGI, R)
    loglab = -0.5 * quad - 0.5 * QL * LOGDET[None, :] - 0.5 * QL * S * math.log(2 * math.pi)
    quadq = np.einsum('ns,pst,nt->np', ostar, SIGQI, ostar)
    logq = -0.5 * quadq - 0.5 * LOGDETQ[None, :] - 0.5 * S * math.log(2 * math.pi)
    lw = np.log(PRIOR)[None, :] + loglab + logq
    lw -= lw.max(1, keepdims=True)
    w = np.exp(lw); w /= w.sum(1, keepdims=True)
    means = np.einsum('pi,ni->np', WVEC, ostar) / PREC[None, :]
    vars_ = np.broadcast_to((1 / PREC)[None, :], means.shape)
    comp = -0.5 * np.log(2 * np.pi * vars_) - 0.5 * (ystar[:, None] - means) ** 2 / vars_
    mx = comp.max(1, keepdims=True)
    nll = -(mx[:, 0] + np.log((w * np.exp(comp - mx)).sum(1)))
    md = (w * means).sum(1)
    vd = (w * (vars_ + means ** 2)).sum(1) - md ** 2
    # iid-distinct reference (ignores shared errors) for scale
    s2 = SG2 + SE2
    prec_b = 1 + S / s2; mean_b = ostar.sum(1) / s2 / prec_b
    nll_b = 0.5 * np.log(2 * np.pi / prec_b) + 0.5 * (ystar - mean_b) ** 2 * prec_b
    return nll, md, vd, nll_b
