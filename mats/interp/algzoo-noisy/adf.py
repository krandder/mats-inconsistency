"""Deductive forward description of a noisy 2nd-argmax RNN: a Gaussian mixture over (h_t, M_t, S_t) indexed by the
algorithmic state (a_t, b_t) and, optionally, by the gate pattern of selected neurons ("alg cells x gate patterns").

No samples of the network are used. Each step:
  1. append x_t ~ N(0, 1) (independent of the state);
  2. split on the algorithmic event (x > M: new max; S < x < M: new 2nd max; x < S: no change) by exact one-constraint
     Gaussian truncation, moment-matched after each constraint (assumed-density filtering);
  3. split on up to `k_split` uncertain gates z_i = (W h + w x)_i (most uncertain first) the same way; the remaining
     uncertain gates are statistically linearised (Stein: slope P(z_i > 0), mean E ReLU(z_i), residual variance kept
     as independent noise); gates with P < eps or > 1 - eps are fixed;
  4. map linearly to (h', M', S'), add the noise sigma^2 r_t^2 on h';
  5. merge components with the same key (new alg cell [+ gate key]) by moment matching.
At t = n the error probability of each component is P(argmax W_oh h != b) under its Gaussian (sequential truncation
over the 9 pairwise constraints, or exact Genz via scipy for checking).
FLOPs are counted analytically per operation (dense matrix products 2mnk; truncation 3D^2 + 10D; erf/exp as 20).
"""
import math
from collections import defaultdict
import numpy as np
from scipy.special import ndtr, log_ndtr

SQ2PI = math.sqrt(2 * math.pi)


class Flops:
    def __init__(self): self.n = 0.0
    def add(self, k): self.n += k


def truncate(mu, S, a, c, sign, fl, eps=1e-300):
    """Condition N(mu, S) on sign*(a.y + c) > 0. Returns (prob, mu', S'). Exact first two moments of the truncation."""
    Sa = S @ a
    m = sign * (a @ mu + c)
    v = a @ Sa
    fl.add(2 * len(mu) ** 2 + 4 * len(mu) + 60)
    if v <= 1e-18:
        return (1.0 if m > 0 else 0.0), mu, S
    s = math.sqrt(v)
    al = -m / s
    P = float(ndtr(-al))
    if P < 1e-300:
        return 0.0, mu, S
    lam = math.exp(-0.5 * al * al - math.log(SQ2PI) - float(log_ndtr(-al)))
    mu2 = mu + sign * Sa * (lam / s)
    S2 = S - np.outer(Sa, Sa) * (lam * (lam - al) / v)
    fl.add(3 * len(mu) ** 2 + 2 * len(mu))
    return P, mu2, S2


def relu_moments(m, s):
    """Elementwise E ReLU(z), P(z > 0), Var ReLU(z) for z ~ N(m, s^2)."""
    s = np.maximum(s, 1e-12); u = m / s
    P = ndtr(u); ph = np.exp(-0.5 * u * u) / SQ2PI
    E1 = m * P + s * ph
    E2 = (m * m + s * s) * P + m * s * ph
    return E1, P, np.maximum(E2 - E1 * E1, 0.0)


class ADF:
    def __init__(self, W, r, sigma, k_split=4, eps=1e-6, gate_key=(), prune=1e-12, final='seq', keep=None, mu=None):
        self.W_hi, self.W_hh, self.W_oh = [np.asarray(w, float) for w in W]
        self.d = self.W_hi.shape[0]; self.n = self.W_oh.shape[0]
        self.r = np.asarray(r, float) if r is not None else np.zeros((self.n, self.d))
        self.sigma = float(sigma); self.k_split = k_split; self.eps = eps
        self.gate_key = tuple(gate_key); self.prune = prune; self.final = final
        self.fl = Flops()
        self.keep = keep; self.mu = mu
        d, n = self.d, self.n
        self.Wt = []; self.wt = []; self.bt = []
        for t in range(n):
            if keep is None:
                self.Wt.append(self.W_hh); self.wt.append(self.W_hi); self.bt.append(np.zeros(d))
            else:
                Wm = self.W_hh * keep['rec'][t]
                self.Wt.append(Wm); self.wt.append(self.W_hi * keep['inp'][t])
                self.bt.append((self.W_hh * (~keep['rec'][t])) @ mu[t])
        if keep is None:
            self.W_out = self.W_oh; self.b_out = np.zeros(n)
        else:
            self.W_out = self.W_oh * keep['out']; self.b_out = (self.W_oh * (~keep['out'])) @ mu[n]

    # state vector v = (h_0..h_{d-1}, M, S); y = (v, x)
    def run(self, upto=None, verbose=False):
        d = self.d; D = d + 2
        comps = {(-1, -1, ()): [(1.0, np.zeros(D), np.zeros((D, D)))]}
        upto = self.n if upto is None else upto
        self.history = []
        for t in range(upto):
            comps = self.step(comps, t)
            self.history.append({k: sum(w for w, _, _ in v) for k, v in comps.items()})
            if verbose: print('t', t + 1, 'components', sum(len(v) for v in comps.values()), 'flops %.3g' % self.fl.n, flush=True)
        return comps

    def step(self, comps, t):
        d = self.d; D = d + 2; fl = self.fl
        W, w = self.Wt[t], self.wt[t]
        G = np.zeros((d, D + 1)); G[:, :d] = W; G[:, D] = w          # z = G y + b
        self._b = self.bt[t]
        new = defaultdict(list)
        for key, lst in comps.items():
            a, b, _ = key
            for (pw, mu, S) in lst:
                if pw < self.prune: continue
                muy = np.concatenate([mu, [0.0]]); Sy = np.zeros((D + 1, D + 1)); Sy[:D, :D] = S; Sy[D, D] = 1.0
                iM, iS, ix = d, d + 1, D
                br = []
                if t == 0:
                    br.append(((0, -1), 1.0, muy, Sy, 'first'))
                elif t == 1:
                    e = np.zeros(D + 1); e[ix] = 1; e[iM] = -1
                    P, m1, S1 = self._trunc(muy, Sy, e, 0.0, +1)
                    br.append(((1, 0), P, m1, S1, 'A'))
                    P2, m2, S2 = self._trunc(muy, Sy, e, 0.0, -1)
                    br.append(((0, 1), P2, m2, S2, 'B'))
                else:
                    e = np.zeros(D + 1); e[ix] = 1; e[iM] = -1
                    P, m1, S1 = self._trunc(muy, Sy, e, 0.0, +1)
                    br.append(((t, a), P, m1, S1, 'A'))
                    Pn, mn, Sn = self._trunc(muy, Sy, e, 0.0, -1)
                    e2 = np.zeros(D + 1); e2[ix] = 1; e2[iS] = -1
                    P2, m2, S2 = self._trunc(mn, Sn, e2, 0.0, +1)
                    br.append(((a, t), Pn * P2, m2, S2, 'B'))
                    P3, m3, S3 = self._trunc(mn, Sn, e2, 0.0, -1)
                    br.append(((a, b), Pn * P3, m3, S3, 'C'))
                for (na, nb), P, my, Sy_, kind in br:
                    if pw * P < self.prune: continue
                    for (Pg, mg, Sg, gates) in self._split_gates(my, Sy_, G):
                        if pw * P * Pg < self.prune: continue
                        mu2, S2, gates = self._map(mg, Sg, G, gates, kind, t)
                        gk = tuple(int(gates[i]) if gates[i] >= 0 else 2 for i in self.gate_key)
                        new[(na, nb, gk)].append((pw * P * Pg, mu2, S2))
        return {k: [self._merge(v)] for k, v in new.items()}

    def _trunc(self, mu, S, a, c, sign):
        return truncate(mu, S, a, c, sign, self.fl)

    def _split_gates(self, my, Sy, G):
        """Split on up to k_split gates, most uncertain first (re-checked inside each sub-cell). Returns a list of
        (prob, mu, S, gates) with gates[i] = 1 / 0 for split gates and -1 for gates decided later in _map."""
        fl = self.fl
        mz = G @ my + self._b; vz = np.einsum('ij,jk,ik->i', G, Sy, G); fl.add(2 * G.size + 2 * G.shape[0] * Sy.shape[0] ** 2)
        P = ndtr(mz / np.sqrt(np.maximum(vz, 1e-18)))
        unc = np.where((P > self.eps) & (P < 1 - self.eps))[0]
        order = unc[np.argsort(np.abs(P[unc] - 0.5))][: self.k_split]
        cells = [(1.0, my, Sy, -np.ones(self.d))]
        for i in order:
            nxt = []
            for (pp, m, S, g) in cells:
                vi = G[i] @ S @ G[i]; fl.add(2 * S.size + 2 * len(m))
                pi = float(ndtr((G[i] @ m + self._b[i]) / math.sqrt(max(vi, 1e-18))))
                if pi <= self.eps or pi >= 1 - self.eps:
                    nxt.append((pp, m, S, g)); continue
                for sgn, gv in ((+1, 1.0), (-1, 0.0)):
                    p2, m2, S2 = self._trunc(m, S, G[i], self._b[i], sgn)
                    if pp * p2 <= 0: continue
                    g2 = g.copy(); g2[i] = gv
                    nxt.append((pp * p2, m2, S2, g2))
            cells = nxt
        return cells

    def _map(self, my, Sy, G, gates, kind, t):
        """y -> v' = (h', M', S') linear (+ statistical linearisation of unsplit uncertain gates) + noise."""
        d = self.d; D = d + 2; fl = self.fl
        A = np.zeros((D, D + 1)); c = np.zeros(D); extra = np.zeros(D)
        mz = G @ my; vz = np.einsum('ij,jk,ik->i', G, Sy, G); fl.add(2 * G.size + 2 * d * (D + 1) ** 2)
        b = self._b
        und = gates < 0
        Pz = ndtr((mz + b) / np.sqrt(np.maximum(vz, 1e-18)))
        gates = gates.copy()
        gates[und & (Pz >= 1 - self.eps)] = 1.0
        gates[und & (Pz <= self.eps)] = 0.0
        lin = gates < 0
        if lin.any():
            E1, P, V = relu_moments(mz[lin] + b[lin], np.sqrt(np.maximum(vz[lin], 0)))
            A[:d][lin] = P[:, None] * G[lin]
            c[:d][lin] = E1 - P * mz[lin]
            extra[:d][lin] = np.maximum(V - P * P * vz[lin], 0.0)
            fl.add(40 * lin.sum())
        on = gates == 1.0
        A[:d][on] = G[on]
        c[:d][on] = b[on]
        iM, iS, ix = d, d + 1, D
        if kind == 'first': A[iM, ix] = 1.0; c[iS] = 0.0
        elif kind == 'A': A[iM, ix] = 1.0; A[iS, iM] = 1.0
        elif kind == 'B': A[iM, iM] = 1.0; A[iS, ix] = 1.0
        else: A[iM, iM] = 1.0; A[iS, iS] = 1.0
        mu2 = A @ my + c
        S2 = A @ Sy @ A.T
        fl.add(2 * D * (D + 1) ** 2 + 2 * D * D * (D + 1))
        S2[np.arange(d), np.arange(d)] += extra[:d] + (self.sigma * self.r[t]) ** 2
        return mu2, S2, gates

    def _merge(self, lst):
        W_ = np.array([p for p, _, _ in lst]); tot = W_.sum()
        if len(lst) == 1: return (tot, lst[0][1], lst[0][2])
        mus = np.stack([m for _, m, _ in lst]); mu = (W_[:, None] * mus).sum(0) / tot
        S = sum(p * (Sg + np.outer(m - mu, m - mu)) for p, m, Sg in lst) / tot
        self.fl.add(len(lst) * 3 * mu.size ** 2)
        return (tot, mu, S)

    def error_rate(self, comps, method='seq'):
        """P(error) = sum over final components of weight x P(argmax W_oh h != b)."""
        d = self.d; n = self.n; tot = 0.0; per = {}
        for (a, b, gk), lst in comps.items():
            for (pw, mu, S) in lst:
                mh, Sh = mu[:d], S[:d, :d]
                L = self.W_out; ml = L @ mh + self.b_out; Sl = L @ Sh @ L.T; self.fl.add(2 * n * d * d + 2 * n * n * d)
                Pc = self.p_correct(ml, Sl, b, method)
                per[(a, b, gk)] = per.get((a, b, gk), 0.0) + pw * (1 - Pc)
                tot += pw * (1 - Pc)
        return tot, per

    def p_correct(self, ml, Sl, b, method='seq'):
        n = len(ml); others = [j for j in range(n) if j != b]
        Dm = np.zeros((n - 1, n)); Dm[:, b] = 1; Dm[np.arange(n - 1), others] = -1
        md = Dm @ ml; Sd = Dm @ Sl @ Dm.T
        if method == 'genz':
            from scipy.stats import multivariate_normal
            Sd = Sd + 1e-12 * np.eye(n - 1)
            return float(multivariate_normal(mean=np.zeros(n - 1), cov=Sd).cdf(md, abseps=1e-7, releps=1e-6))
        # sequential truncation (ADF), most likely-violated constraint first
        P = 1.0; m = md.copy(); S = Sd.copy()
        for _ in range(n - 1):
            s = np.sqrt(np.maximum(np.diag(S), 1e-30)); u = m / s
            i = int(np.argmin(np.where(np.isfinite(u), u, np.inf)))
            e = np.zeros(n - 1); e[i] = 1.0
            p, m, S = truncate(m, S, e, 0.0, +1, self.fl)
            P *= p
            if P <= 0: return 0.0
            S[i, :] = 0; S[:, i] = 0; m[i] = 1e30   # constraint satisfied; remove it
        return P
