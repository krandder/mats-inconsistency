"""Value-level queries P(x_i = 1 | C) answered by K rollouts masked to the evidence C (evidence = a hard rule on
edges, i.e. a factor of the score).  Exact conditionals of T and of reality are for evaluation only."""
import itertools
import numpy as np
from world import A, D, N, DX, XARR, traj_law

KEYS = list(itertools.product([-1, 0, 1], repeat=N))          # evidence: -1 = not given
DXF = np.tile(DX, (A, 1))                                     # (96, 3) values of each augmented trajectory
QUERIES = [(i, k) for k in KEYS for i in range(N) if k[i] == -1]


def match(key, X):
    return np.all([(X[:, j] == key[j]) | (key[j] == -1) for j in range(N)], axis=0)


def cond(law_x, i, key):
    """Exact P(x_i = 1 | key) under a law over the 8 value vectors."""
    m = match(key, XARR)
    return law_x[m & (XARR[:, i] == 1)].sum() / law_x[m].sum()


class Amp:
    """K fresh rollouts per query from the rollout policy masked to the evidence."""

    def __init__(self, Q, Qr, R, lo=0.02):
        self.R, self.lo = R, lo
        self.PC, self.wC = {}, {}
        for k in KEYS:
            C = {j: k[j] for j in range(N) if k[j] != -1}
            P = traj_law(Q, Qr, C).ravel()
            self.PC[k] = P / P.sum()
            with np.errstate(divide="ignore", invalid="ignore"):
                self.wC[k] = np.where(P > 0, 0.5 * np.tile(R, A) / self.PC[k], 0.0)

    def stats(self, qi, qk, K, rng):
        """qi: array of variables, qk: list of evidence keys.  Returns weighted fraction, unweighted fraction, ESS."""
        qi = np.asarray(qi)
        pw, pu, ess = np.zeros(len(qi)), np.zeros(len(qi)), np.zeros(len(qi))
        groups = {}
        for n, k in enumerate(qk):
            groups.setdefault(k, []).append(n)
        for k, g in groups.items():
            g = np.array(g)
            idx = rng.choice(A * D, size=(len(g), K), p=self.PC[k])
            Wt = self.wC[k][idx]
            ind = DXF[idx, qi[g][:, None]] == 1
            pw[g] = (Wt * ind).sum(1) / Wt.sum(1)
            pu[g] = ind.mean(1)
            ess[g] = Wt.sum(1) ** 2 / (Wt ** 2).sum(1)
        return pw, pu, ess

    def score_of_values(self, xkeys, K, rng):
        """Estimate of sum over the demonstration's orders of the score of the values x, from K rollouts masked to x."""
        out = np.zeros(len(xkeys))
        groups = {}
        for n, k in enumerate(xkeys):
            groups.setdefault(k, []).append(n)
        for k, g in groups.items():
            idx = rng.choice(A * D, size=(len(g), K), p=self.PC[k])
            out[np.array(g)] = self.wC[k][idx].mean(1)
        return out

    def lg(self, p):
        p = np.clip(p, self.lo, 1 - self.lo)
        return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))
