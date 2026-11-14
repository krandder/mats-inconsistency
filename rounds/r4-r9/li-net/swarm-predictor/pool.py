"""Pool the claims of many short reasoning chains about one ShipHunt board into one set of prices.

Reuses the lir-agent harness and never modifies it: env.shiphunt.placements() gives the legal
boards and ShipHunt.posterior() gives exact Bayes. The harness directory is put on sys.path
(override with LIR_HARNESS).

Worlds z. Labelled ship layouts (which cells each ship covers). The harness enumerates
occupied-cell patterns u. Some patterns come from more than one layout (ships 3 and 2 end to end
in one row), so the prior p0 is the env's own (uniform over patterns), split equally among the
layouts of a pattern. The exact posterior is ShipHunt.posterior(), split in the same way; this is
exact because evidence only depends on which cells are occupied. The house prior p0 knows the rules
only (ships contiguous, lengths fixed, no overlap). It never reads the evidence: evidence reaches
the pool only through the chains' claims.

Statements j. Yes/no facts about z: "occ:X" = cell X is occupied; "H{L}:r{r}" = a ship of length
L lies horizontally in row r. F[z, j] = 1 when statement j is true on board z.

Claims. Chain i says P(statement j) = q_ij, with stated confidence w_ij in [0, 1].

Pools. Each pool returns a price for every statement. A statement that no chain claimed gets its
rules-only prior p0_j.
  mean   average of q over the claims on j                                         baseline (a)
  vote   share of claims on j with q > 0.5 (self-consistency agreement rate)        baseline (b)
  stmt   log-linear pool, statement by statement: logit p_j = logit p0_j + theta_j          (c)
  world  the same pool over legal boards: mu(z) ∝ p0(z) exp(sum_j F[z, j] theta_j),
         prices = marginals of mu                                                     (d, naive)
  proj   the stmt pool's prices on the claimed statements, then projected onto legal boards
         (project(): the IDEAS §1 fair-price projection with the house prior p0)       (d, proj)
where theta_j = sum over claims on j of b_ij * (logit q_ij - logit p0_j), b_ij = beta_ij / beta_0.
Names in METHODS: LL-* = stmt, W-* = world, D-* = proj; the suffix avg / add / fit is the stake rule
below (fit = kappa, alpha learned by fit_stakes; D-* use the LL-* stakes, then project).
Stakes b:
  avg    b = w / (sum of w over the claims on j). The weights sum to 1, so the pool averages
         (the STUDY-LIR 3f finding: two independent 0.9 signals stay at 0.9).
  add    b = kappa * w / m_j ** alpha, with m_j = number of claims on j. kappa = 1, alpha = 0 adds
         every claim as independent evidence; fit_stakes() learns (kappa, alpha) on held-out
         episodes (alpha = 1 is averaging, alpha = 0 is adding).
Why "world" double counts. theta_j are log-odds shifts, which are the right natural parameters only
when statements are independent under p0. Cells and ship rows are correlated under the rules, so one
noiseless chain that states exact Bayes on every open statement already lands 0.0150 Brier from Bayes
through "world" with fixed stakes, against 0.00015 through "proj" (smoke_results.json:
floor_one_exact_chain_all_open_statements). "proj" matches the pooled prices instead of adding their
shifts, which removes that floor.
Why "world" is the PDG pool. Over distributions mu on legal boards, minimise
  beta_0 KL(mu || p0) + sum_ij beta_ij [KL(mu_j || q_ij) - KL(mu_j || p0_j)],
i.e. one edge per claim at +beta and a copy of the prior marginal at -beta (the evidence-adding
correction of STUDY-LIR 3f). The bracket is linear in mu, so the minimiser is exactly
  mu ∝ p0 * prod_ij (q_ij(f_j) / p0_j(f_j)) ** b_ij,
which is the formula above (STUDY-LIR 3c: mu ∝ p0 * prod_i p_i ** (beta_i / beta_0)). Its prices are
coherent by construction, because they are the marginals of one distribution over legal boards.
"""
from __future__ import annotations

import copy
import itertools
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

HARNESS = os.environ.get("LIR_HARNESS", "/home/agent/workspaces/eurisko-study/lir-agent/harness")
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)
from env.shiphunt import LETTERS, EnvConfig, ShipHunt, placements  # noqa: E402

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, float)))


def _layouts(cfg: EnvConfig):
    """Labelled layouts: tuples of ships (L, 'H'|'V', row|col, cells), same-length ships unordered."""
    W, H = cfg.W, cfg.H
    per = []
    for L in cfg.ships:
        opts = []
        for r in range(H):
            for c in range(W):
                if c + L <= W:
                    opts.append((L, "H", r, tuple(r * W + c + k for k in range(L))))
                if L > 1 and r + L <= H:
                    opts.append((L, "V", c, tuple((r + k) * W + c for k in range(L))))
        per.append(opts)
    seen = set()
    for combo in itertools.product(*per):
        cells = [x for s in combo for x in s[3]]
        if len(set(cells)) == len(cells):
            seen.add(tuple(sorted(combo)))
    return sorted(seen)


class Board:
    """Legal boards, statements, the rules-only prior and the exact Bayes oracle for one EnvConfig."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        P = placements(cfg)
        self.U = len(P)
        key = {frozenset(np.flatnonzero(r).tolist()): u for u, r in enumerate(P)}
        self.layouts = _layouts(cfg)
        self.Z = len(self.layouts)
        self.u_of = np.array([key[frozenset(x for s in lay for x in s[3])] for lay in self.layouts])
        self.mult = np.bincount(self.u_of, minlength=self.U)
        assert (self.mult > 0).all(), "every harness pattern must have a layout"
        names, cols = [], []
        for c in range(cfg.n_cells):
            names.append(f"occ:{LETTERS[c]}")
            cols.append([c in {x for s in lay for x in s[3]} for lay in self.layouts])
        for L in sorted(set(cfg.ships), reverse=True):
            if L < 2:
                continue
            for r in range(cfg.H):
                names.append(f"H{L}:r{r}")
                cols.append([any(s[0] == L and s[1] == "H" and s[2] == r for s in lay) for lay in self.layouts])
        self.names: List[str] = names
        self.J = len(names)
        self.F = np.array(cols, dtype=float).T                     # [Z, J]
        self.logp0 = -np.log(self.U * self.mult[self.u_of].astype(float))
        self.p0 = np.exp(self.logp0)
        self.p0_marg = self.p0 @ self.F                            # [J] rules-only prior prices

    # ---- exact Bayes (harness oracle) ---------------------------------------------------------
    def from_patterns(self, w_u: np.ndarray) -> np.ndarray:
        return w_u[self.u_of] / self.mult[self.u_of]

    def bayes(self, env: ShipHunt) -> np.ndarray:
        """Exact posterior over layouts given all of env's evidence (ShipHunt.posterior)."""
        return self.from_patterns(env.posterior())

    def bayes_subset(self, env: ShipHunt, probe_idx: Sequence[int], fired_idx: Sequence[int]) -> np.ndarray:
        """Exact posterior given only some of the evidence items (same oracle, on a shallow copy)."""
        e = copy.copy(env)
        e.probes = [env.probes[i] for i in probe_idx]
        e.fired = [env.fired[i] for i in fired_idx]
        return self.from_patterns(e.posterior())

    def truth(self, env: ShipHunt) -> int:
        """Index of the true labelled layout: uniform among the layouts of env.occ, seeded by env.seed."""
        P = placements(self.cfg)
        u = int(np.flatnonzero((P == env.occ[None]).all(1))[0])
        cand = np.flatnonzero(self.u_of == u)
        return int(cand[np.random.default_rng([env.seed, 7]).integers(len(cand))])

    def text(self, j: int) -> str:
        """Plain-language statement, for prompts."""
        n = self.names[j]
        if n.startswith("occ:"):
            c = LETTERS.index(n[4:])
            return f"cell {n[4:]} (row {c // self.cfg.W}, column {c % self.cfg.W}) is occupied by a ship"
        L, r = n[1:].split(":r")
        return f"the ship of length {L} lies horizontally in row {r}"


@dataclass
class Claims:
    """All claims of one episode: chain id, statement id, claimed probability q, stated confidence w."""
    chain: np.ndarray
    stmt: np.ndarray
    q: np.ndarray
    w: np.ndarray

    def select(self, mask: np.ndarray) -> "Claims":
        return Claims(self.chain[mask], self.stmt[mask], self.q[mask], self.w[mask])

    def first_chains(self, n: int) -> "Claims":
        return self.select(self.chain < n)

    @staticmethod
    def concat(parts: List["Claims"]) -> "Claims":
        return Claims(*(np.concatenate([getattr(p, f) for p in parts]) for f in ("chain", "stmt", "q", "w")))


# ---- stakes and pools ---------------------------------------------------------------------------
def stakes(board: Board, cl: Claims, mode: str, kappa: float = 1.0, alpha: float = 0.0) -> np.ndarray:
    if mode == "avg":
        tot = np.bincount(cl.stmt, weights=cl.w, minlength=board.J)
        return cl.w / tot[cl.stmt]
    m = np.bincount(cl.stmt, minlength=board.J).astype(float)
    return kappa * cl.w / m[cl.stmt] ** alpha


def shift(board: Board, cl: Claims) -> np.ndarray:
    """Per-claim log-odds shift over the prior, logit q - logit p0_j."""
    return logit(cl.q) - logit(board.p0_marg[cl.stmt])


def theta(board: Board, cl: Claims, b: np.ndarray) -> np.ndarray:
    return np.bincount(cl.stmt, weights=b * shift(board, cl), minlength=board.J)


def world(board: Board, th: np.ndarray) -> np.ndarray:
    lg = board.logp0 + board.F @ th
    lg = lg - lg.max()
    mu = np.exp(lg)
    return mu / mu.sum()


def price_mean(board: Board, cl: Claims) -> np.ndarray:
    n = np.bincount(cl.stmt, minlength=board.J)
    s = np.bincount(cl.stmt, weights=cl.q, minlength=board.J)
    return np.where(n > 0, s / np.maximum(n, 1), board.p0_marg)


def price_vote(board: Board, cl: Claims) -> np.ndarray:
    n = np.bincount(cl.stmt, minlength=board.J)
    v = np.where(cl.q > 0.5, 1.0, np.where(cl.q < 0.5, 0.0, 0.5))
    s = np.bincount(cl.stmt, weights=v, minlength=board.J)
    return np.where(n > 0, s / np.maximum(n, 1), board.p0_marg)


METHODS = ("prior", "mean", "vote", "LL-avg", "LL-add", "LL-fit", "W-avg", "W-add", "W-fit", "D-avg", "D-fit")


def method_stakes(board: Board, cl: Claims, method: str, params: Optional[Dict] = None) -> np.ndarray:
    kind = method.split("-")[1]
    if kind == "avg":
        return stakes(board, cl, "avg")
    if kind == "add":
        return stakes(board, cl, "add", 1.0, 0.0)
    return stakes(board, cl, "add", params["kappa"], params["alpha"])


def project(board: Board, t: np.ndarray, mask: np.ndarray, c: float = 20.0, iters: int = 60, tol: float = 1e-8):
    """PDG projection of a price table onto legal boards (IDEAS §1 fair-price projection, house p0):
    minimise over mu on legal boards  KL(mu || p0) + sum over masked j of c * KL(mu_j || t_j).
    Unmasked statements carry no edge. Stationarity: mu ∝ p0 exp(F lam) with
    lam_j = c * (logit t_j - logit mu_j) on masked j and 0 elsewhere; solved by damped Newton on lam.
    c = beta_table / beta_0: large c reproduces a coherent table exactly; an incoherent table goes to
    the nearest coherent one. Returns (mu [Z], max abs residual)."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return board.p0.copy(), 0.0
    F = board.F[:, idx]
    lt = logit(np.clip(t[idx], 1e-4, 1 - 1e-4))
    lam = np.zeros(len(idx))

    def state(l):
        lg = board.logp0 + F @ l
        lg = lg - lg.max()
        mu = np.exp(lg)
        mu /= mu.sum()
        m = np.clip(mu @ F, 1e-12, 1 - 1e-12)
        return mu, m, l / c - (lt - logit(m))

    mu, m, r = state(lam)
    for _ in range(iters):
        if np.abs(r).max() < tol:
            break
        Fc = F - m
        cov = (Fc * mu[:, None]).T @ Fc
        Jm = np.eye(len(idx)) / c + cov / (m * (1 - m))[:, None]
        step = np.linalg.solve(Jm, r)
        h, base = 1.0, np.abs(r).max()
        for _ in range(30):
            mu2, m2, r2 = state(lam - h * step)
            if np.abs(r2).max() < base:
                break
            h *= 0.5
        lam, mu, m, r = lam - h * step, mu2, m2, r2
    return mu, float(np.abs(r).max())


def price(board: Board, cl: Claims, method: str, params: Optional[Dict] = None):
    """Returns (prices [J], mu [Z] or None). D-* use the LL-* stakes, then project()."""
    if method == "prior" or len(cl.q) == 0:
        return board.p0_marg.copy(), (board.p0.copy() if method[0] in "WD" else None)
    if method == "mean":
        return price_mean(board, cl), None
    if method == "vote":
        return price_vote(board, cl), None
    th = theta(board, cl, method_stakes(board, cl, method, params))
    if method.startswith("LL"):
        return sigmoid(logit(board.p0_marg) + th), None
    if method.startswith("D"):
        claimed = np.bincount(cl.stmt, minlength=board.J) > 0
        mu, _ = project(board, sigmoid(logit(board.p0_marg) + th), claimed, params.get("c", 20.0) if params else 20.0)
        return mu @ board.F, mu
    mu = world(board, th)
    return mu @ board.F, mu


# ---- learned stakes ----------------------------------------------------------------------------
def _golden(f, lo, hi, iters=40):
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (c, fc) if fc < fd else (d, fd)


def fit_stakes(board: Board, episodes: List[Dict], kind: str, alphas: Sequence[float] = None) -> Dict:
    """Learn (kappa, alpha) for b = kappa * w / m_j ** alpha on training episodes (outcomes only,
    never the oracle). kind 'stmt': mean log loss of the per-statement pool over open statements;
    kind 'world': the pool's log loss on the realised board, -log mu(z*) (the STUDY-LIR 3c market
    loss whose stake gradient is minus the settlement). Each episode dict needs 'claims', 'y' [J],
    'zstar', 'open' [J] bool. Both losses are convex in kappa for fixed alpha; alpha is a grid."""
    J = board.J
    S = np.stack([np.bincount(e["claims"].stmt, weights=e["claims"].w * shift(board, e["claims"]), minlength=J)
                  for e in episodes])
    M = np.stack([np.bincount(e["claims"].stmt, minlength=J) for e in episodes]).astype(float)
    Mp = np.where(M > 0, M, 1.0)
    Y = np.stack([e["y"] for e in episodes])
    O = np.stack([e["open"] for e in episodes]).astype(float)
    zs = np.array([e["zstar"] for e in episodes])
    lp0 = logit(board.p0_marg)[None]
    if alphas is None:
        alphas = np.round(np.arange(0.0, 1.26, 0.125), 3) if M.max() > 1 else [0.0]

    def loss(lk, a):
        th = np.exp(lk) * S * Mp ** (-a)
        if kind == "stmt":
            x = (lp0 + th) * (2 * Y - 1)
            return float((np.logaddexp(0.0, -x) * O).sum() / O.sum())
        lg = board.logp0[None] + th @ board.F.T
        mx = lg.max(1, keepdims=True)
        lse = (mx[:, 0] + np.log(np.exp(lg - mx).sum(1)))
        return float(np.mean(lse - lg[np.arange(len(zs)), zs]))

    best = None
    for a in alphas:
        lk, f = _golden(lambda t: loss(t, a), -7.0, 3.0)
        if best is None or f < best[2]:
            best = (float(np.exp(lk)), float(a), f)
    return {"kappa": best[0], "alpha": best[1], "train_loss": best[2], "kind": kind}
