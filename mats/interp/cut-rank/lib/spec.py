"""Spectral estimators for the cut matrix with split-sample noise control (cut-rank lane).

Ms: list of S >= 4 independent Monte Carlo estimates (F x C) of the same matrix M (independent inputs per split and per
row). Energies are per entry, (1/(F C)) x squared Frobenius quantities, so they compare with eps^2 Vbar.
- In-sample Gram: mean over pairs s < t of sym(M_s M_t^T)/(F C): unbiased for M M^T/(F C) (no noise-squared bias).
  Its eigen-tail is biased low (Ky Fan/Jensen), so ranks from it tend to be underestimates.
- Cross-fitted: subspace from the mean of one half of the splits, energy from the pair-Gram of the other half,
  halves swapped and averaged. Unbiased energy for a subspace chosen independently; the tail is biased high.
"""
import numpy as np


def pair_gram(Ms):
    """mean over pairs s<t of sym(M_s M_t^T) / (F C), computed as ((sum M)(sum M)^T - sum M M^T) / (S(S-1))."""
    F, C = Ms[0].shape
    S = len(Ms)
    T = np.zeros((F, C)); D = np.zeros((F, F))
    for m in Ms:
        m = m.astype(np.float64)
        T += m
        D += m @ m.T
    return (T @ T.T - D) / (S * (S - 1)) / (F * C)


def tails_in(Ms):
    G = pair_gram(Ms)
    lam = np.linalg.eigvalsh(G)[::-1]
    tot = lam.sum()
    return np.concatenate([[tot], tot - np.cumsum(lam)]), lam


def tails_cf(Ms):
    S = len(Ms)
    F, C = Ms[0].shape
    halves = (list(range(S // 2)), list(range(S // 2, S)))
    res = []
    for sel, ev in (halves, halves[::-1]):
        U, _, _ = np.linalg.svd(np.mean([Ms[i] for i in sel], axis=0).astype(np.float64), full_matrices=False)
        G = pair_gram([Ms[i] for i in ev])
        cap = np.einsum('fi,fg,gi->i', U, G, U)
        tot = np.trace(G)
        t = np.concatenate([[tot], tot - np.cumsum(cap)])
        if len(t) < F + 1:
            t = np.concatenate([t, np.full(F + 1 - len(t), t[-1])])
        res.append(t)
    return np.mean(res, axis=0)


def eps_rank(tail, thr):
    idx = np.nonzero(tail <= thr)[0]
    return int(idx[0]) if len(idx) else len(tail)  # len(tail) = F + 1: not reached with F rows ("saturated")


def ranks(tail, Vbar, eps_list=(0.3, 0.1, 0.03, 0.01)):
    return {e: eps_rank(tail, e * e * Vbar) for e in eps_list}


def _pair_mean_err(Ts, X):
    """mean over pairs s<t of (T_s - X)(T_t - X), elementwise mean: unbiased for (M - X)^2 if X is independent."""
    S = len(Ts)
    tot = np.zeros_like(X); sq = np.zeros_like(X)
    for t in Ts:
        d = t.astype(np.float64) - X
        tot += d; sq += d * d
    return (tot * tot - sq) / (S * (S - 1))


def probe_protocol(Ms, s_list, rng):
    """Held-out error of a rank-s one-way linear protocol built from probe fronts and probe columns, using split 0
    for the protocol and the pairs of the other splits for unbiased scoring. Fronts: first half train, second half
    test; columns: a random half train, the rest test. Returns {s: mean squared error per test entry}."""
    M0 = Ms[0].astype(np.float64)
    F, C = M0.shape
    tf = np.arange(F // 2); vf = np.arange(F // 2, F)
    perm = rng.permutation(C); tc = perm[:C // 2]; vc = perm[C // 2:]
    U, Sv, Vt = np.linalg.svd(M0[np.ix_(tf, tc)], full_matrices=False)
    Bm = M0[np.ix_(vf, tc)]; Cm = M0[np.ix_(tf, vc)]
    Ts = [m[np.ix_(vf, vc)] for m in Ms[1:]]
    out = {}
    for s in s_list:
        s = min(s, len(Sv))
        X = np.zeros((len(vf), len(vc))) if s == 0 else (Bm @ Vt[:s].T / Sv[:s]) @ (U[:, :s].T @ Cm)
        out[s] = float(np.mean(_pair_mean_err(Ts, X)))
    return out


def per_net_errors(Ms, s, ncols_per_net):
    """Error of the rank-s projection of split 0 (front subspace from split 0 alone), per (front, net) block of
    ncols_per_net consecutive columns, unbiased via pairs of the other splits."""
    M0 = Ms[0].astype(np.float64)
    U, _, _ = np.linalg.svd(M0, full_matrices=False)
    P = U[:, :s]
    X = P @ (P.T @ M0)
    e = _pair_mean_err(Ms[1:], X)
    F, C = e.shape
    return e.reshape(F, C // ncols_per_net, ncols_per_net).mean(2)


def tails_heldout(Ms, rng=None):
    """POST HOC (added 2026-10-30 ~23:49Z, not pre-registered): held-out-front tail, an upper estimate of the
    population tail for the sampled columns. Column subspace = top-s eigenvectors of the pair-averaged column Gram of
    the training fronts (first half); tail_ho(s) = mean over test fronts of the pair-averaged energy of their rows
    outside that subspace, per entry. Unbiased over input noise given the subspace; biased high (suboptimal
    subspace, chosen on other fronts). Returns tail[s] for s = 0..min(F/2, C)."""
    S = len(Ms)
    F, C = Ms[0].shape
    tr = np.arange(F // 2); te = np.arange(F // 2, F)
    Tt = np.zeros((len(tr), C)); D = np.zeros((C, C))
    for m in Ms:
        a = m[tr].astype(np.float64); Tt += a; D += a.T @ a
    Gc = (Tt.T @ Tt - D) / (S * (S - 1)) / len(tr)
    lam, V = np.linalg.eigh(Gc); V = V[:, ::-1]
    smax = min(len(tr), C)
    # energy of test rows along each training eigenvector, pair-averaged over splits
    P = [m[te].astype(np.float64) @ V[:, :smax] for m in Ms]     # (F_te, smax) projections
    Tp = sum(P); Dp = sum(p * p for p in P)
    cap = ((Tp * Tp - Dp) / (S * (S - 1))).mean(0) / C              # captured energy per direction, per entry
    Tm = sum(m[te].astype(np.float64) for m in Ms); Dm = sum(m[te].astype(np.float64) ** 2 for m in Ms)
    tot = float(((Tm * Tm - Dm) / (S * (S - 1))).mean())
    return np.concatenate([[tot], tot - np.cumsum(cap)])
