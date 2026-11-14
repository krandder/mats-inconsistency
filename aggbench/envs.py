"""Environments with exact Bayes posteriors.

E1  linear-Gaussian. theta ~ N(0, I_d). Agent i observes
    s_i = A_i theta + B_i eta + eps_i,  eta ~ N(0, I_r) shared by all agents,
    eps_i ~ N(0, sigma_i^2 I). A_i covers a random subset of coordinates
    (overlapping coverage); shared factors eta make errors correlated.
    Agents report their local posterior (mean, precision) of theta.

E2  discrete. Y in {0, 1} with prior P(Y=1) = prior. A pool of K evidence
    pieces, conditionally independent given Y: x_k ~ N(+mu_k, 1) if Y=1 and
    N(-mu_k, 1) if Y=0, so piece k carries log-likelihood ratio 2 mu_k x_k.
    Each agent sees a random subset of pieces and reports its posterior
    log-odds. The global posterior counts each piece in the union once.

E3  adversarial variants of E2: Sybil copies of one agent, biased
    (miscalibrated) agents, colluders who pool their evidence and all report
    the pooled posterior.

E4  rollouts as sources. The world is E2's (Y, K evidence pieces, all K in
    the context). The sources are N rollouts of one imperfect any-order
    predictor: rollout i conditions on L_i pieces in its own random order
    and adds a*gamma^(L_i - t) * (lam_k + e_k) + nu_it for the piece at step t,
    starting from logit0 + b. Shared model error (misreading e_k of piece k,
    episode bias b, calibration a, recency decay gamma) correlates the
    rollouts' errors; nu is private sampling noise. Everything is linear-
    Gaussian given Y, so both ceilings are exact (see RolloutEnv).

Both kinds return a `Batch`. Aggregators must only see `batch.view(...)`;
`batch.oracle` holds the true structure and is for ceilings and checks only.
"""
from dataclasses import dataclass, asdict
import numpy as np

from .splits import episode_seeds

VISIBILITY = ("none", "noisy", "full")
ROLES = {"honest": 0, "colluder": 1, "biased": 2, "sybil": 3}


@dataclass
class View:
    """What an aggregator may use: reports, keys at one visibility, public constants."""
    kind: str
    reports: dict
    keys: np.ndarray  # (n, n_agents, key_dim); key_dim = 0 at visibility "none"
    public: dict


@dataclass
class Batch:
    env: str
    config: dict
    kind: str            # "gaussian" (E1) or "bernoulli" (E2, E3)
    seeds: np.ndarray    # one seed per episode
    truth: np.ndarray    # theta (n, d) or Y (n,)
    obs: np.ndarray      # per-agent raw observations; NaN = piece not seen (E2)
    local: dict          # honest exact local posterior of each agent
    reports: dict        # what each agent reports (= local unless adversarial)
    keys: dict           # visibility -> (n, n_agents, key_dim)
    bayes: dict          # exact posterior given ALL agents' observations
    oracle: dict         # true structure; never give to a learned aggregator
    public: dict

    def view(self, visibility="full"):
        if visibility not in VISIBILITY:
            raise ValueError(f"visibility must be one of {VISIBILITY}")
        return View(self.kind, {k: v.copy() for k, v in self.reports.items()},
                    self.keys[visibility].copy(), dict(self.public))

    def agent_inputs(self, visibility="full", alignment="aligned", mixing=False, own_model=False):
        """Per-agent local inputs for learned message encoders (high-dimensional values, 2026-10-20).

        Returns (obs, keys, public): obs[e, i] is agent i's own raw observation in episode e, keys as in
        view(). Rule: an encoder may map obs[e, i] (and keys[e, i]) to agent i's message only; nothing may
        mix agents before messages are sent. Check with aggbench.messages.check_locality. E3-biased: the
        biased agent's raw obs are honest; apply its distortion to its message (see README).

        alignment="aligned" (default) returns obs and keys indexed by shared slots (E2/E3: evidence piece k is
        coordinate k for every agent). alignment="unaligned" passes each agent's slot-indexed coordinates
        (observation and the slot-indexed key blocks alike) through an agent-specific, episode-fixed random
        permutation drawn from the episode seed and not revealed (see alignment_transforms); mixing=True
        (E1 only) uses a random orthogonal matrix instead. Same information, no shared slot positions.
        Headline results use "unaligned" with keys at "none" and "noisy" (README).

        own_model=True also returns, last, each agent's description of its OWN sensor / likelihood model
        (n, N, k), under the same hidden per-agent transform as obs (see own_model_from_full). It describes how
        the agent's own observation depends on the latent and nothing about which other agents share its
        sources: E1 (A_i, R_i = B_i B_i^T + sigma_i^2 I), not the factor loadings B_i; Z1/Z4 without the parent
        group / model identity. A real forecaster knows its own sources, so this is always available to the
        agent's own encoder, whatever the visibility; `visibility` only sets the provenance key (what the
        aggregator is told). Returns (obs, keys, public, own)."""
        if visibility not in VISIBILITY:
            raise ValueError(f"visibility must be one of {VISIBILITY}")
        if alignment not in ("aligned", "unaligned"):
            raise ValueError("alignment must be 'aligned' or 'unaligned'")
        obs, keys, full = self.obs.copy(), self.keys[visibility].copy(), self.keys["full"].copy()
        if alignment == "unaligned":
            T = alignment_transforms(self, mixing)
            obs, keys = apply_alignment(self, obs, keys, T)
            full = apply_alignment(self, self.obs.copy(), full, T)[1]
        elif mixing:
            raise ValueError("mixing=True needs alignment='unaligned'")
        if own_model:
            return obs, keys, dict(self.public), own_model_from_full(self, full)
        return obs, keys, dict(self.public)

    def __len__(self):
        return len(self.seeds)


# ----------------------------------------------------------------------------- alignment (2026-10-21)

ALIGN_SALT = 0x5107


def slot_layout(batch):
    """How a batch's agent inputs are indexed by shared slots: (n_slots, obs_slotted, key_blocks) with key_blocks
    ((start, n_slots, width), ...) = blocks of each agent's key that are an (n_slots, width) matrix whose rows are
    slots. Returns None when nothing is slot-indexed. E1: the m observed scalars (rows of A_i and B_i in the key);
    E2/E3: the K evidence pieces (coverage and coverage x mu); E4: the K pieces of the keys (the trace is in the
    rollout's own reading order already); zoo: identities of shared sources (Z1 parent group, Z4 model)."""
    c = batch.config
    if batch.kind == "gaussian":
        m, d, r = batch.obs.shape[-1], c["d"], c["n_factors"]
        return m, True, ((0, m, d), (m * d, m, r))
    if "W" in batch.oracle:
        K = c["n_pieces"]
        return K, False, ((0, K, 1), (K, K, 1), (2 * K, K, 1))
    if "cov" in batch.oracle:
        K = c["n_pieces"]
        return K, True, tuple((b * K, K, 1) for b in range(batch.keys["full"].shape[-1] // K))
    blocks = tuple(_env_of(batch).key_slot_blocks)
    return (blocks[0][1], False, blocks) if blocks else None


def own_model_from_full(batch, full):
    """Each agent's own likelihood description from its (possibly transformed) full key. E1: [A_i, R_i] with
    R_i = B_i B_i^T + sigma_i^2 I (the loadings B_i name the shared noise factors: provenance). E2-E4: the full
    key (which own coordinates hold evidence, their strengths; E4 also the reading order). Zoo: the full key
    minus the source-identity blocks (Z1 parent group, Z4 model)."""
    if batch.kind == "gaussian":
        c = batch.config
        n, N = full.shape[:2]
        m, d, r = batch.obs.shape[-1], c["d"], c["n_factors"]
        A = full[..., :m * d]
        B = full[..., m * d:m * d + m * r].reshape(n, N, m, r)
        sig2 = np.exp(2 * full[..., -1])
        R = np.einsum("enar,enbr->enab", B, B) + sig2[..., None, None] * np.eye(m)
        return np.concatenate([A, R.reshape(n, N, m * m)], -1)
    from .zoo import ZOO
    if batch.env not in ZOO:
        return full
    env = _env_of(batch)
    drop = np.zeros(full.shape[-1], dtype=bool)
    for start, n_slots, width in getattr(env, "provenance_blocks", env.key_slot_blocks):
        drop[start:start + n_slots * width] = True
    return full[..., ~drop]


def _env_of(batch):
    from .zoo import ZOO
    return ZOO[batch.env](**batch.config)


def _haar_orthogonal(rng, n):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def alignment_transforms(batch, mixing=False):
    """The hidden per-agent transforms of agent_inputs(alignment="unaligned"): (n, N, S, S) matrices T with
    unaligned slot vector = T @ aligned slot vector. Permutation matrices, or (mixing=True, E1 only) Haar-random
    orthogonal matrices. Drawn from each episode's seed; for ceilings and tests, never for an aggregator."""
    layout = slot_layout(batch)
    n, N = batch.obs.shape[:2]
    if layout is None:
        return np.zeros((n, N, 0, 0))
    if mixing and batch.kind != "gaussian":
        raise ValueError("mixing=True needs dense real-valued slots (E1); use permutations elsewhere")
    S = layout[0]
    T = np.zeros((n, N, S, S))
    for e, seed in enumerate(batch.seeds):
        rng = np.random.default_rng([int(seed), ALIGN_SALT])
        for i in range(N):
            T[e, i] = _haar_orthogonal(rng, S) if mixing else np.eye(S)[rng.permutation(S)]
    return T


def apply_alignment(batch, obs, keys, T):
    """Apply per-agent slot transforms T (from alignment_transforms) to obs and keys. Permutations move NaNs
    (unseen pieces) along with their slot, so an unseen piece is not identifiable by position."""
    layout = slot_layout(batch)
    if layout is None:
        return obs, keys
    S, obs_slotted, blocks = layout
    perm = np.isin(T, (0.0, 1.0)).all((-1, -2))           # (n, N): pure permutation matrices

    def move(x):  # x (n, N, S, w) -> T @ x, NaN-safe for permutations
        idx = np.argmax(T, -1)
        permuted = np.take_along_axis(x, idx[..., None], axis=2)
        rotated = np.einsum("enab,enbw->enaw", T, np.nan_to_num(x))
        return np.where(perm[..., None, None], permuted, rotated)

    if obs_slotted:
        obs = move(obs[..., None])[..., 0]
    if keys.shape[-1]:
        keys = keys.copy()
        for start, s, w in blocks:
            blk = keys[..., start:start + s * w].reshape(*keys.shape[:2], s, w)
            keys[..., start:start + s * w] = move(blk).reshape(*keys.shape[:2], s * w)
    return obs, keys


def clearing(K, B, Lam, dl):
    """Solve the clearing equation (K^T B K + Lam) w = K^T B dl. Returns (w, K^T B K + Lam)."""
    P = K.T @ B @ K + Lam
    return np.linalg.solve(P, K.T @ B @ dl), P


class _Env:
    name = "?"

    def sample(self, split, n, start=0, allow_test=False):
        """Generate n episodes of `split`. The test split needs allow_test=True."""
        if split == "test" and not allow_test:
            raise PermissionError("test split is reserved for confirmatory rounds; pass allow_test=True")
        seeds = episode_seeds(split, n, start)
        eps = [self.episode(int(s)) for s in seeds]
        grouped = {}
        for key in eps[0]:
            arr = np.stack([e[key] for e in eps])
            group, _, field = key.partition(".")
            if field:
                grouped.setdefault(group, {})[field] = arr
            else:
                grouped[group] = arr
        keys = {v: grouped["key"][v] for v in VISIBILITY}
        return Batch(self.name, asdict(self.config), self.kind, seeds, grouped["truth"], grouped["obs"],
                     grouped["local"], grouped["report"], keys, grouped["bayes"], grouped["oracle"],
                     self.public())


# ----------------------------------------------------------------------------- E1

@dataclass
class GaussianConfig:
    n_agents: int = 5
    d: int = 4                  # dimension of theta
    obs_dim: int = 2            # scalars observed per agent (must be <= coverage)
    coverage: int = 3           # coordinates of theta each agent's A_i touches
    n_factors: int = 2          # shared noise factors eta
    factor_scale: float = 1.0   # strength of shared (correlated) noise
    noise_range: tuple = (0.3, 2.0)  # private noise sd, log-uniform (reliability spread)
    key_noise: float = 0.3      # sd of noise added to keys at visibility "noisy"


def gaussian_local(A, Bf, sig, s):
    """Each agent's exact posterior of theta given only its own s_i."""
    N, m, d = A.shape
    means, precs = np.zeros((N, d)), np.zeros((N, d, d))
    for i in range(N):
        R = Bf[i] @ Bf[i].T + sig[i] ** 2 * np.eye(m)
        means[i], precs[i] = clearing(A[i], np.linalg.inv(R), np.eye(d), s[i])
    return means, precs


def gaussian_global(A, Bf, sig, s):
    """Exact posterior of theta given all agents' s, with the full cross-agent noise covariance."""
    N, m, d = A.shape
    K = A.reshape(N * m, d)
    Bk = Bf.reshape(N * m, -1)
    R = Bk @ Bk.T + np.diag(np.repeat(sig ** 2, m))
    return clearing(K, np.linalg.inv(R), np.eye(d), s.ravel())


class GaussianEnv(_Env):
    kind = "gaussian"

    def __init__(self, name="E1", **overrides):
        self.name = name
        self.config = c = GaussianConfig(**overrides)
        if not 1 <= c.obs_dim <= c.coverage <= c.d:
            raise ValueError("need 1 <= obs_dim <= coverage <= d (so reports are lossless per agent)")

    def public(self):
        return {"d": self.config.d, "n_agents": self.config.n_agents, "prior_mean": 0.0, "prior_prec": 1.0}

    def episode(self, seed):
        c = self.config
        rng = np.random.default_rng(seed)
        N, d, m, r = c.n_agents, c.d, c.obs_dim, c.n_factors
        A = np.zeros((N, m, d))
        for i in range(N):
            cols = rng.choice(d, c.coverage, replace=False)
            A[i][:, cols] = rng.standard_normal((m, c.coverage))
        Bf = c.factor_scale * rng.standard_normal((N, m, r)) / np.sqrt(max(r, 1))
        lo, hi = c.noise_range
        sig = np.exp(rng.uniform(np.log(lo), np.log(hi), N))
        theta = rng.standard_normal(d)
        eta = rng.standard_normal(r)
        s = A @ theta + Bf @ eta + sig[:, None] * rng.standard_normal((N, m))
        loc_mean, loc_prec = gaussian_local(A, Bf, sig, s)
        g_mean, g_prec = gaussian_global(A, Bf, sig, s)
        key_full = np.concatenate([A.reshape(N, -1), Bf.reshape(N, -1), np.log(sig)[:, None]], axis=1)
        key_noisy = key_full + c.key_noise * rng.standard_normal(key_full.shape)
        return {
            "truth": theta, "obs": s,
            "local.mean": loc_mean, "local.prec": loc_prec,
            "report.mean": loc_mean, "report.prec": loc_prec,
            "bayes.mean": g_mean, "bayes.prec": g_prec,
            "key.none": np.zeros((N, 0)), "key.noisy": key_noisy, "key.full": key_full,
            "oracle.A": A, "oracle.Bf": Bf, "oracle.sig": sig, "oracle.eta": eta,
        }


# ----------------------------------------------------------------------------- E2 / E3

@dataclass
class DiscreteConfig:
    n_agents: int = 5               # agents before Sybil copies are added
    n_pieces: int = 12              # K, size of the evidence pool
    pieces_per_agent: tuple = (2, 6)  # inclusive range; overlap ~ pieces_per_agent / n_pieces
    strength_range: tuple = (0.1, 0.6)  # mu_k ~ U(range): reliability spread of pieces
    prior: float = 0.5
    key_noise: float = 0.15         # bit-flip prob. of coverage keys at visibility "noisy"
    n_sybil: int = 0                # extra exact copies of one honest agent
    n_biased: int = 0               # agents reporting logit0 + temp*(local - logit0) + shift
    bias_temp: float = 3.0
    bias_shift: float = 1.5
    n_colluders: int = 0            # agents that pool their pieces and all report the pooled posterior


def report_posterior_logit(report, cov, mu, a, b, logit0):
    """Exact log-odds of Y given all agents' reports and the true structure (oracle clearing).

    Undo each agent's known distortion to get r = C lam (C = coverage, lam = piece LLRs).
    lam_k | Y ~ N(+-m_k, 2 m_k) with m_k = 2 mu_k^2, so r | Y ~ N(+-C m, 2 C M C^T) and
    log-odds = logit0 + m^T C^T (C M C^T)^+ r = logit0 + sum_k w_k, where
    w = M C^T (C M C^T)^+ r is the Lam -> 0 limit of the clearing solve
    (C^T C + Lam) w = C^T r with Lam proportional to M^{-1}.
    """
    r = (report - logit0 - b) / a
    C = cov.astype(float)
    m = 2 * mu ** 2
    S = (C * m) @ C.T
    w = (C * m).T @ np.linalg.pinv(S, rcond=1e-10, hermitian=True) @ r
    return logit0 + w.sum()


def union_identified(cov):
    """True iff the union's total LLR is a linear function of the agents' reports (1_U in rowspace C)."""
    C = cov.astype(float)
    u = cov.any(0).astype(float)
    coef = np.linalg.lstsq(C.T, u, rcond=None)[0]
    return bool(np.abs(C.T @ coef - u).max() < 1e-8)


class DiscreteEnv(_Env):
    kind = "bernoulli"

    def __init__(self, name="E2", **overrides):
        self.name = name
        self.config = c = DiscreteConfig(**overrides)
        lo, hi = c.pieces_per_agent
        if not 1 <= lo <= hi <= c.n_pieces:
            raise ValueError("need 1 <= pieces_per_agent[0] <= pieces_per_agent[1] <= n_pieces")
        if c.n_colluders == 1:
            raise ValueError("collusion needs at least 2 colluders")
        honest_needed = 1 if c.n_sybil else 0
        if c.n_colluders + c.n_biased + honest_needed > c.n_agents:
            raise ValueError("not enough agents for the requested adversaries")
        if not 0 < c.prior < 1:
            raise ValueError("prior must be in (0, 1)")
        self.logit0 = float(np.log(c.prior / (1 - c.prior)))

    def public(self):
        c = self.config
        return {"prior_logit": self.logit0, "n_agents": c.n_agents + c.n_sybil, "n_pieces": c.n_pieces}

    def episode(self, seed):
        c = self.config
        rng = np.random.default_rng(seed)
        K, N0 = c.n_pieces, c.n_agents
        N = N0 + c.n_sybil
        mu = rng.uniform(*c.strength_range, K)
        y = int(rng.random() < c.prior)
        x = (2 * y - 1) * mu + rng.standard_normal(K)
        lam = 2 * mu * x

        cov = np.zeros((N0, K), dtype=bool)
        for i in range(N0):
            size = rng.integers(c.pieces_per_agent[0], c.pieces_per_agent[1] + 1)
            cov[i, rng.choice(K, size, replace=False)] = True
        role = np.zeros(N0, dtype=int)
        nc, nb = c.n_colluders, c.n_biased
        if nc:
            cov[:nc] = cov[:nc].any(0)
            role[:nc] = ROLES["colluder"]
        role[nc:nc + nb] = ROLES["biased"]
        if c.n_sybil:  # copies of the last original agent, which is honest by validation
            cov = np.vstack([cov, np.repeat(cov[N0 - 1:N0], c.n_sybil, axis=0)])
            role = np.concatenate([role, np.full(c.n_sybil, ROLES["sybil"])])
        a = np.where(role == ROLES["biased"], c.bias_temp, 1.0)
        b = np.where(role == ROLES["biased"], c.bias_shift, 0.0)
        perm = rng.permutation(N)  # agent positions carry no information
        cov, role, a, b = cov[perm], role[perm], a[perm], b[perm]

        local = self.logit0 + cov @ lam
        report = self.logit0 + a * (local - self.logit0) + b
        bayes = self.logit0 + lam[cov.any(0)].sum()

        key_full = np.concatenate([cov.astype(float), cov * mu], axis=1)
        ncov = cov ^ (rng.random(cov.shape) < c.key_noise)
        nmu = mu * np.exp(c.key_noise * rng.standard_normal((N, K)))
        key_noisy = np.concatenate([ncov.astype(float), ncov * nmu], axis=1)
        return {
            "truth": np.int64(y), "obs": np.where(cov, x, np.nan),
            "local.logit": local, "report.logit": report, "bayes.logit": np.float64(bayes),
            "key.none": np.zeros((N, 0)), "key.noisy": key_noisy, "key.full": key_full,
            "oracle.cov": cov, "oracle.mu": mu, "oracle.x": x, "oracle.role": role,
            "oracle.a": a, "oracle.b": b,
        }


# ----------------------------------------------------------------------------- E4

@dataclass
class RolloutConfig:
    n_rollouts: int = 6              # sources = rollouts of one model on one context
    n_pieces: int = 8                # K evidence pieces in the context (all of them are "the evidence")
    rollout_len: tuple = (2, 6)      # inclusive range of pieces one rollout conditions on (its budget)
    strength_range: tuple = (0.1, 0.6)
    prior: float = 0.5
    calibration: float = 1.5         # a: the model scales every log-likelihood ratio by a (> 1: over-confident)
    recency: float = 0.7             # gamma: a piece read t steps before the end gets weight gamma^t (forgetting)
    read_noise: float = 0.3          # sd of e_k: the model's misreading of piece k, SHARED by all rollouts reading k
    shared_bias: float = 0.3         # sd of b: an episode-level bias SHARED by all rollouts
    step_noise: float = 0.3          # sd of nu: private per-step sampling noise of one rollout
    key_noise: float = 0.15          # bit-flip prob. / jitter of keys at visibility "noisy"


def gaussian_linear_logit(obs, offset, H, m, cov_noise, logit0):
    """Exact log-odds of Y given obs = offset + H lam + noise, noise ~ N(0, cov_noise) independent of Y,
    lam_k | Y ~ N(+-m_k, 2 m_k) independent (the E2 piece LLRs). obs | Y ~ N(offset +- H m, Sigma), with
    Sigma = H diag(2m) H^T + cov_noise, so the log-likelihood ratio is 2 (H m)^T Sigma^+ (obs - offset)."""
    S = (H * (2 * m)) @ H.T + cov_noise
    return logit0 + 2 * (H @ m) @ np.linalg.pinv(S, rcond=1e-10, hermitian=True) @ (obs - offset)


def rollout_weights(order, n_pieces, a, gamma):
    """W[i, k] = a gamma^(L_i - t) if rollout i reads piece k at step t (1-based) of L_i steps, else 0."""
    N, Lmax = order.shape
    W = np.zeros((N, n_pieces))
    for i in range(N):
        steps = order[i][order[i] >= 0]
        L = len(steps)
        W[i, steps] = a * gamma ** (L - 1 - np.arange(L))
    return W


def rollout_report_posterior_logit(report, W, length, mu, c, logit0):
    """Reports ceiling: exact log-odds of Y given the N rollout reports and the true error structure.
    report_i = logit0 + b + sum_k W_ik (lam_k + e_k) + sum_t nu_it."""
    N = len(report)
    noise = c.read_noise ** 2 * W @ W.T + c.shared_bias ** 2 * np.ones((N, N)) + np.diag(c.step_noise ** 2 * length)
    return gaussian_linear_logit(report, np.full(N, logit0), W, 2 * mu ** 2, noise, logit0)


def rollout_trace_posterior_logit(trace, order, mu, c, logit0):
    """Trace ceiling: exact log-odds of Y given every rollout's whole trace (start value and per-step
    increments, the richest message a rollout can send) and the true error structure."""
    N, T = trace.shape
    K = len(mu)
    rows, offs = [], []          # one row per observed trace entry: coefficients on (lam, e, b, nu)
    n_nu = N * (T - 1)
    for i in range(N):
        steps = order[i][order[i] >= 0]
        L = len(steps)
        r = np.zeros(2 * K + 1 + n_nu)
        r[2 * K] = 1.0           # start value logit0 + b
        rows.append(r)
        offs.append(logit0)
        for t, k in enumerate(steps):
            r = np.zeros(2 * K + 1 + n_nu)
            w = c.calibration * c.recency ** (L - 1 - t)
            r[k], r[K + k], r[2 * K + 1 + i * (T - 1) + t] = w, w, 1.0
            rows.append(r)
            offs.append(0.0)
    G = np.array(rows)
    obs = np.concatenate([trace[i][:1 + int((order[i] >= 0).sum())] for i in range(N)])
    sd = np.concatenate([np.full(K, c.read_noise), [c.shared_bias], np.full(n_nu, c.step_noise)])
    F = G[:, K:] * sd
    return gaussian_linear_logit(obs, np.array(offs), G[:, :K], 2 * mu ** 2, F @ F.T, logit0)


class RolloutEnv(_Env):
    """E4: sources are rollouts of an imperfect any-order predictor (see module docstring).

    Batch fields: truth Y; obs = each rollout's TRACE (n, N, Lmax + 1): entry 0 is its start value
    logit0 + b, entries 1..L_i its per-step increments in conditioning order, NaN after; local = the exact
    posterior given the rollout's conditioning set (what an exact model would report); report = the
    imperfect model's final log-odds (= sum of the trace); bayes = exact posterior given ALL K pieces of
    the context. oracle: x, mu, cov, order (-1 padded), length, W, e, b. The reports ceiling is
    baselines.oracle_clearing; the trace ceiling is rollout_trace_posterior_logit.
    """
    kind = "bernoulli"

    def __init__(self, name="E4", **overrides):
        self.name = name
        self.config = c = RolloutConfig(**overrides)
        lo, hi = c.rollout_len
        if not 1 <= lo <= hi <= c.n_pieces:
            raise ValueError("need 1 <= rollout_len[0] <= rollout_len[1] <= n_pieces")
        if not 0 < c.prior < 1:
            raise ValueError("prior must be in (0, 1)")
        if min(c.read_noise, c.shared_bias, c.step_noise) < 0 or c.calibration <= 0 or c.recency <= 0:
            raise ValueError("noise sds must be >= 0, calibration and recency > 0")
        self.logit0 = float(np.log(c.prior / (1 - c.prior)))

    def public(self):
        c = self.config
        return {"prior_logit": self.logit0, "n_agents": c.n_rollouts, "n_pieces": c.n_pieces}

    def structure(self, rng):
        """The episode's structure: piece strengths and each rollout's conditioning order."""
        c = self.config
        K, N, hi = c.n_pieces, c.n_rollouts, c.rollout_len[1]
        mu = rng.uniform(*c.strength_range, K)
        order = np.full((N, hi), -1, dtype=np.int64)
        for i in range(N):
            L = rng.integers(c.rollout_len[0], hi + 1)
            order[i, :L] = rng.choice(K, L, replace=False)
        return mu, order

    def simulate(self, mu, order, rng, n):
        """n independent draws of (Y, x, e, b, nu) for a fixed structure. Returns y, x, e, b, trace."""
        c = self.config
        K, (N, Lmax) = len(mu), order.shape
        y = (rng.random(n) < c.prior).astype(np.int64)
        x = (2 * y[:, None] - 1) * mu + rng.standard_normal((n, K))
        e = c.read_noise * rng.standard_normal((n, K))
        b = c.shared_bias * rng.standard_normal(n)
        nu = c.step_noise * rng.standard_normal((n, N, Lmax))
        lam = 2 * mu * x
        trace = np.full((n, N, Lmax + 1), np.nan)
        trace[:, :, 0] = self.logit0 + b[:, None]
        for i in range(N):
            steps = order[i][order[i] >= 0]
            L = len(steps)
            w = c.calibration * c.recency ** (L - 1 - np.arange(L))
            trace[:, i, 1:L + 1] = w * (lam[:, steps] + e[:, steps]) + nu[:, i, :L]
        return y, x, e, b, trace

    def episode(self, seed):
        c = self.config
        rng = np.random.default_rng(seed)
        K, N = c.n_pieces, c.n_rollouts
        mu, order = self.structure(rng)
        y, x, e, b, trace = (v[0] for v in self.simulate(mu, order, rng, 1))
        lam = 2 * mu * x
        cov = np.zeros((N, K), dtype=bool)
        recent = np.zeros((N, K))         # steps before the end at which the piece was read, / K
        for i in range(N):
            steps = order[i][order[i] >= 0]
            cov[i, steps] = True
            recent[i, steps] = (len(steps) - 1 - np.arange(len(steps))) / K
        length = cov.sum(1)
        W = rollout_weights(order, K, c.calibration, c.recency)
        report = np.nansum(trace, axis=1)
        key_full = np.concatenate([cov.astype(float), recent, cov * mu], axis=1)
        ncov = cov ^ (rng.random(cov.shape) < c.key_noise)
        nrec = ncov * np.abs(recent + c.key_noise * rng.standard_normal((N, K)))
        nmu = mu * np.exp(c.key_noise * rng.standard_normal((N, K)))
        key_noisy = np.concatenate([ncov.astype(float), nrec, ncov * nmu], axis=1)
        return {
            "truth": np.int64(y), "obs": trace,
            "local.logit": self.logit0 + cov @ lam, "report.logit": report,
            "bayes.logit": np.float64(self.logit0 + lam.sum()),
            "key.none": np.zeros((N, 0)), "key.noisy": key_noisy, "key.full": key_full,
            "oracle.cov": cov, "oracle.mu": mu, "oracle.x": x, "oracle.order": order, "oracle.length": length,
            "oracle.W": W, "oracle.e": e, "oracle.b": np.float64(b),
        }


# ----------------------------------------------------------------------------- presets

PRESETS = {
    "E1": lambda **kw: GaussianEnv("E1", **kw),
    "E2": lambda **kw: DiscreteEnv("E2", **kw),
    "E3-sybil": lambda **kw: DiscreteEnv("E3-sybil", **{"n_sybil": 3, **kw}),
    "E3-biased": lambda **kw: DiscreteEnv("E3-biased", **{"n_biased": 1, **kw}),
    "E3-collude": lambda **kw: DiscreteEnv("E3-collude", **{"n_colluders": 3, **kw}),
    "E4": lambda **kw: RolloutEnv("E4", **kw),
}


def make_env(name, allow_heldout=False, **overrides):
    """make_env("E2", n_agents=8) -> environment with the default config plus overrides.

    Also builds the structure zoo (aggbench.zoo, "Z1".."Z7"); held-out zoo structures (zoo.HELDOUT) need
    allow_heldout=True (confirmatory generalization-across-structure rounds only). zoo.list_envs() lists all."""
    if name in PRESETS:
        return PRESETS[name](**overrides)
    from .zoo import HELDOUT, ZOO
    if name not in ZOO:
        raise ValueError(f"unknown env {name!r}; choose from {list(PRESETS) + list(ZOO)}")
    if name in HELDOUT and not allow_heldout:
        raise PermissionError(f"{name} is a held-out structure (confirmatory rounds only); pass allow_heldout=True")
    return ZOO[name](**overrides)
