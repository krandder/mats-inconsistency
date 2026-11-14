"""The amplifications on top of rollouts of the rollout policy (all read the score only at terminal states).

snis_sets      : K rollouts, self-normalised importance weights against the score.
SoftTree       : soft tree search with sum (log-sum-exp) backups, MENTS form as adapted to GFlowNets by
                 Morozov et al. 2406.13655 (leaf statistics initialised from the policy's Q; option 1 reads the
                 score at terminal leaves, option 2 never does, as in that paper's experiments).
mh_kernel      : back-and-forth refinement (Local Search GFlowNets, 2310.02710): backtrack j components,
                 rebuild with the policy, accept by Metropolis-Hastings / as printed / only if the score rises.
"""
import numpy as np
from world import A, D, N, STEPS, PREFIXES, DESC, children, lse, softmax, traj_law


def weights(P, R, mode="uniform"):
    """Importance weight of each augmented trajectory.  uniform: R(d) * (1/2) / P_F(a, d) (fixed backward
    policy over the alien coordinate).  marg: R(d) / q(d), q(d) = sum_a P_F(a, d) (count each d once)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if mode == "uniform":
            w = 0.5 * R[None, :] / P
        else:
            w = np.repeat((R / P.sum(0))[None, :], A, 0)
    return np.where(P > 0, w, 0.0)


def snis_sets(P, w, K, reps, rng):
    """reps independent sets of K rollouts.  Returns the self-normalised weighted measure of each set over
    demonstrations (reps, D), its unweighted counterpart, and the effective sample size of each set."""
    idx = rng.choice(A * D, size=(reps, K), p=P.ravel() / P.sum())
    W = w.ravel()[idx]
    Wn = W / W.sum(1, keepdims=True)
    flat = (np.arange(reps)[:, None] * (A * D) + idx).ravel()
    meas = np.bincount(flat, weights=Wn.ravel(), minlength=reps * A * D).reshape(reps, A, D)
    cnt = np.bincount(flat, minlength=reps * A * D).reshape(reps, A, D) / K
    ess = W.sum(1) ** 2 / (W ** 2).sum(1)
    return meas, cnt, ess


class SoftTree:
    def __init__(self, Q, Qroot, R, pb=None, option=1, eps=0.1, rng=None):
        self.Q, self.R, self.option, self.eps = Q, R, option, eps
        self.pb = np.full((A, D), 0.5) if pb is None else pb
        self.rng = rng or np.random.default_rng(0)
        self.Qt = {"root": Qroot.copy()}
        self.Nv = {"root": 0}
        self.evals = 0

    def _kids(self, node):
        return list(range(A)) if node == "root" else children(node[1])

    def _child(self, node, c):
        return (c, ()) if node == "root" else (node[0], node[1] + (c,))

    def simulate(self):
        node, path = "root", []
        while True:
            kids, qt = self._kids(node), self.Qt[node]
            self.Nv[node] += 1
            ps = min(1.0, self.eps * len(kids) / np.log(self.Nv[node] + 2))
            sm = softmax(qt)
            if sm.sum() == 0:
                sm = np.full(len(kids), 1.0 / len(kids))
            pi = (1 - ps) * sm + ps / len(kids)
            ci = self.rng.choice(len(kids), p=pi)
            nxt = self._child(node, kids[ci])
            path.append((node, ci, nxt))
            if len(nxt[1]) == N:
                break
            if nxt not in self.Qt:
                self.Qt[nxt] = self.Q[nxt].copy()
                self.Nv[nxt] = 0
                break
            node = nxt
        for node, ci, nxt in reversed(path):
            if len(nxt[1]) == N:
                if self.option == 1:
                    d = DESC[nxt[1]][0]
                    self.Qt[node][ci] = np.log(self.R[d]) + np.log(self.pb[nxt[0], d])
                    self.evals += 1
            else:
                self.Qt[node][ci] = lse(self.Qt[nxt])

    def law(self):
        """Law over augmented trajectories of sampling down the tree (policy Q where the tree has no node)."""
        Qe = {k: self.Qt.get(k, v) for k, v in self.Q.items()}
        return traj_law(Qe, self.Qt["root"])

    def logZ(self):
        return lse(self.Qt["root"])


def mh_kernel(P, R, rule="mh"):
    """Exact transition matrix over the 96 augmented trajectories of one back-and-forth move: draw j in 1..4,
    drop the last j of the 4 components (a, step1, step2, step3), rebuild them with the policy, then filter.
    rule: 'mh' (Metropolis-Hastings, reverse over forward), 'printed' (the ratio as printed in 2310.02710
    section 4: forward over reverse), 'greedy' (accept only if the score rises: that paper's default)."""
    n = A * D
    Pf = P.ravel()
    Rf = np.tile(R, A)
    pi = 0.5 * Rf
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(Pf > 0, pi / Pf, 0.0)
    Kmat = np.zeros((n, n))
    for j in range(1, 5):
        keep = 4 - j
        key = [((t // D,) + STEPS[t % D][:keep - 1]) if keep >= 1 else () for t in range(n)]
        groups = {}
        for t, k in enumerate(key):
            groups.setdefault(k, []).append(t)
        for g in groups.values():
            g = np.array(g)
            prop = Pf[g] / max(Pf[g].sum(), 1e-300)
            for t in g:
                if Pf[t] == 0:
                    Kmat[t, t] += 0.25
                    continue
                if rule == "mh":
                    acc = np.minimum(1.0, w[g] / w[t])
                elif rule == "printed":
                    acc = np.minimum(1.0, (Rf[g] / Rf[t]) * (Pf[g] / Pf[t]))
                else:
                    acc = (Rf[g] > Rf[t]).astype(float)
                acc[g == t] = 1.0
                move = 0.25 * prop * acc
                Kmat[t, g] += move
                Kmat[t, t] += 0.25 - move.sum()
    return Kmat
