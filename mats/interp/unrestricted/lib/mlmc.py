"""Multilevel reference-trajectory ("shared-trajectory") sampler for the output mean of a bias-free ReLU net.

Origin: gpt-6-astra, round 3 of the unrestricted-estimators lane (coprover/astra_r3_answer.md, section 1); this module
is the reusable implementation (lane report: outbox/unrestricted-estimators-2026-10-31.md, section 3.7). Variance
measurements that motivated it: scratch/mlmc_check.py, logs/mlmc_check.log.

Idea. Inputs' directions merge as they go deeper (the cosine deficit 1 - kappa^(l)(0) falls like 9 pi^2 / (2 l^2)), so
after r layers an input mostly differs from any other by its amplitude |h_r|. Run ONE reference direction Q through
the whole net, keeping its normalised path q_l and per-layer norms nu_l, and let s_r = prod_{l > r} nu_l. The control
    G_r(U) = m_n * s_r * |h_r(U)| * q_L          (m_n = E|x| for x ~ N(0, I_n); U uniform on the sphere)
needs only an r-layer prefix of U. The telescoping identity
    mu = E_U Y(U),  Y(U) = m_n h_L(U)
       = G_0 + sum_{j=1}^{J} E_U[G_{r_j}(U) - G_{r_{j-1}}(U)] + E_U[Y(U) - G_{r_J}(U)],   r_0 = 0 < r_1 < ... < r_J < L,
with independent samples per level (both terms of a level on the SAME sample), gives an unbiased estimator. G_0 =
m_n f_W(Q) is a constant (no sampling). Radius is exact by positive homogeneity (sigma = 0).

Cost model (forward-pass units, one pass = one input through all L layers with schoolbook products, 2 n^2 L FLOPs,
lib/nets.py pass_flops): setup 1 pass (the reference path); a sample at level j < J+1 costs r_j / L passes (prefix
r_j; the norm |h_{r_{j-1}}| is read on the way); a sample at the last level costs 1 pass. Norms, products and
aggregation are O(n) per sample and ignored (lower order). With level variances V_j and costs C_j, the optimal
allocation for a budget of T passes is N_j proportional to sqrt(V_j / C_j), and the resulting risk is
(sum_j sqrt(V_j C_j))^2 / T (standard multilevel MC). Radius-exact MC (RB-MC) has risk V_sph / T.

Proven rate (astra, under layerwise batching costs): E|Y - G_r|^2 = O(1/r + n/L^2), so dyadic levels give risk
O(log^2 L / (L N)) at the cost of N passes: gain Omega(n / log^2 n) over RB-MC at L = alpha n. Measured (ideal,
setup excluded): about break-even at L <= n/4 (including ARC's 256 x 32), 1.5x at L = n = 128, 1.7-3x at L = 4n, 15x at
width 8, depth 256. The two-level version (one prefix depth r) is usually better than dyadic levels at these sizes.

Usage:
    from nets import Net
    from mlmc import ReferencePath, pilot_level_stats, allocate, mlmc_estimate
    net = Net(seed, n, L)                                   # sigma = 0
    ref = ReferencePath(net.W, rng)                         # 1 pass
    levels = [0, 16]                                        # two-level: prefix depth 16, then the full pass
    V, C = pilot_level_stats(net.W, ref, levels, rng, P=1024)   # pilot (charge it if used for allocation)
    Nj = allocate(V, C, budget_passes=2**14)
    est, cost = mlmc_estimate(net.W, ref, levels, Nj, rng)  # est in R^n, cost in passes (setup included)
Deterministic use: replace rng-drawn directions by any seed source; lane report 3.1 gives an exact symmetric seed
construction (first-layer QR) under idealised arithmetic.
Limits: sigma = 0 only (with activation noise the telescoping stays unbiased if each level's two terms share the
noise of the prefix, but the bound above is not proven); dead nets (V_sph = 0) give zero variance trivially.
Run tests: python3 tests/test_mlmc.py
"""
import math
import numpy as np

try:
    from .nets import mean_radius, sphere_from_gauss
except ImportError:
    from nets import mean_radius, sphere_from_gauss


class ReferencePath:
    """Normalised path of one reference direction: per-layer norms nu_l, final direction q_L, suffix scales s_r."""

    def __init__(self, W, rng=None, q0=None):
        n = W[0].shape[0]
        L = len(W)
        if q0 is None:
            q0 = sphere_from_gauss(rng.standard_normal((1, n)))[0]
        q = np.asarray(q0, dtype=float)
        q = q / np.linalg.norm(q)
        self.q0 = q.copy()
        nus = np.empty(L)
        for l, Wl in enumerate(W):
            z = np.maximum(Wl @ q, 0.0)
            nus[l] = np.linalg.norm(z)
            q = z / nus[l] if nus[l] > 0 else np.eye(n)[0]
        self.nu = nus
        self.qL = q
        self.suffix = np.ones(L + 1)
        for r in range(L - 1, -1, -1):
            self.suffix[r] = self.suffix[r + 1] * nus[r]
        self.n, self.L = n, L
        self.mn = mean_radius(n)

    def G0(self):
        """G_0 = m_n * f_W(Q): constant control (the reference input's radius-exact output)."""
        return self.mn * self.suffix[0] * self.qL


def _prefix_norms(W, U, depths):
    """Run unit directions U (P, n) through the first max(depths) layers; return {r: |h_r(U)|} and h at max depth."""
    out = {0: np.linalg.norm(U, axis=1)}
    h = U
    dmax = max(depths)
    for l in range(dmax):
        h = np.maximum(h @ W[l].T, 0.0)
        if (l + 1) in depths:
            out[l + 1] = np.linalg.norm(h, axis=1)
    return out, h


def level_samples(W, ref, levels, j, U):
    """Samples (P, n) of level j: G_{r_j} - G_{r_{j-1}} for 1 <= j < len(levels), or Y - G_{r_last} for
    j == len(levels) (the last level, a full pass). levels[0] must be 0."""
    L = len(W)
    mn, qL, suf = ref.mn, ref.qL, ref.suffix
    if j < len(levels):
        r1, r0 = levels[j], levels[j - 1]
        nr, _ = _prefix_norms(W, U, {r0, r1})
        return mn * (suf[r1] * nr[r1] - suf[r0] * nr[r0])[:, None] * qL[None, :]
    rl = levels[-1]
    nr, h = _prefix_norms(W, U, {rl, L})
    return mn * h - mn * (suf[rl] * nr[rl])[:, None] * qL[None, :]


def level_costs(levels, L):
    """Cost per sample in passes: r_j / L for the intermediate levels, 1 for the last level."""
    return [levels[j] / L for j in range(1, len(levels))] + [1.0]


def pilot_level_stats(W, ref, levels, rng, P=1024):
    """Pilot estimates of the per-level variances (trace over outputs) and costs. Pilot cost: P * sum(costs)."""
    n = W[0].shape[0]
    V = []
    for j in range(1, len(levels) + 1):
        U = sphere_from_gauss(rng.standard_normal((P, n)))
        d = level_samples(W, ref, levels, j, U)
        V.append(float(np.sum(d.var(0))))
    return V, level_costs(levels, len(W))


def allocate(V, C, budget_passes):
    """Integer sample counts N_j ~ sqrt(V_j / C_j) within a budget (passes, setup excluded), at least 1 per level.
    One sample per level is reserved first and the remainder is split in proportion to sqrt(V_j / C_j), rounding
    down, so sum_j N_j C_j <= budget_passes. Raises ValueError if the budget cannot buy one sample per level."""
    base = sum(C)
    if budget_passes < base:
        raise ValueError(f'budget {budget_passes} passes is below one sample per level ({base:.3g})')
    w = [math.sqrt(max(v, 0.0) / c) for v, c in zip(V, C)]
    tot = sum(wi * c for wi, c in zip(w, C))
    rem = budget_passes - base
    if tot <= 0:
        return [1 for _ in C]
    return [1 + int(math.floor(rem * wi / tot)) for wi in w]


def mlmc_estimate(W, ref, levels, Nj, rng, chunk=1 << 14):
    """Unbiased estimate of mu = E_x a_L (vector in R^n) and its cost in passes (setup of 1 pass included)."""
    n = W[0].shape[0]
    est = ref.G0().copy()
    C = level_costs(levels, len(W))
    cost = 1.0
    for j in range(1, len(levels) + 1):
        N = int(Nj[j - 1])
        acc = np.zeros(n)
        for c in range(0, N, chunk):
            U = sphere_from_gauss(rng.standard_normal((min(chunk, N - c), n)))
            acc += level_samples(W, ref, levels, j, U).sum(0)
        est += acc / N
        cost += N * C[j - 1]
    return est, cost
