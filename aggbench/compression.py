"""Compression benchmark: aggregation when every agent's message is smaller than its evidence (2026-10-21).

Kelvin: "The important part, indeed, is a message smaller than the evidence read by each trader. We need to know
and measure how effective this 'compression' is, specifically for this purpose of the total aggregation." And: "We
need to specifically use the available evidence *to improve* aggregated estimates, and this needs to differ from
how an agent would summarize information for itself to predict the thing based only on its own evidence." And: "it
is essential for the consensus to query the agent or vice-versa before the compression happens."

Budget. b = message BITS per agent. A message is an integer symbol in [0, 2^b) produced by an agent-local encoder
from that agent's own `agent_inputs` (plus, in chain / queried protocols, what it received), and it passes an
explicit `Channel` that rejects anything else. A real-valued width without quantization is NOT a budget (one float
can hold ~1500 tokens, Kuratov et al. 2502.13063). b = None means "unbounded" (the uncompressed numbers), b = 0
means no message (the prior). Two axes are reported: b per agent and the total consensus size (`consensus_bits`:
N b for the append family, S for the fixed-size state). Queries (consensus -> agent) are unbounded and read-only in
the headline queried arm; q = b is a minor arm.

Metrics (`curve`): regret(b) = mean KL(Bayes || forecast) in nats; regret_to_floor(b) = mean log-loss minus the
log loss of `floor` (report.achievable_floor: Bayes, E4 the trace floor); rho(b) = (R(0) - R(b)) / (R(0) - R_full)
with R_full the floor (0 in E2/E3) and, second, the full-context transformer; b90 = the smallest swept b with
rho >= 0.9. `gain` = regret(self-summary, b) - regret(aggregation-aware, b), paired over the same dev episodes.

Baselines at b bits (literature rank in brackets, outbox/compressed-aggregation-literature-2026-10-21.md):
  llr_quantizer  [1]  SELF-SUMMARY (closed form; = queried with q = 0, one-shot). Each agent quantizes its own
                      local log-odds (its own-evidence sufficient statistic for Y) into 2^b cells; thresholds are a
                      companded family s * Phi^-1(j / 2^b) on the agent's standardized log-odds ("own" scale, from
                      its own model) or on the pooled scale ("shared"), family and s tuned on train against the
                      exact decoder. Decoder: EXACT Bayes given the cells and the TRUE structure (provenance full,
                      like oracle_clearing): the agents' log-odds are jointly Gaussian given Y in E2/E3/E4/Z8, so
                      P(cells | Y) is a Gaussian box probability, computed by Genz's separation-of-variables
                      integral with scrambled Sobol points shared by both classes (numerical, checked against Monte
                      Carlo in the tests). Tabular zoo: exact enumeration. b = None gives oracle_clearing.
  top_w          [3]  each agent sends its w pieces of largest |LLR|: which ones (ceil(log2 C(n_i, w)) bits, n_i =
                      its piece count) and each reading in q bits; w, q, s tuned on train. Decoder: exact per-piece
                      cell likelihoods over the union of distinct pieces sent (structure-aware: it knows coverage);
                      it ignores the small information in the selection event. E2/E3 only (Z8, E4: hooks).
  k_of_n         [6]  a scheduler that sees every agent's own model (not observations) lets the most informative
                      agents send ALL their readings at q bits each until the TOTAL N b bits are spent. Decoder:
                      exact over the distinct pieces sent. E2/E3 only.
  extremized_q / trimmed_q [7]  practitioner floor: extremized mean / 20%-trimmed mean of b-bit quantized local
                      log-odds (cell centroids from train), alpha fit on train. b = None equals
                      baselines.extremized.
  queried_llr_qinf    QUERIED, append family (headline queried arm): the consensus state is only the b-bit messages
  queried_llr_qb      so far; before agent t compresses, the consensus answers it a read-only query (a pure function
                      of the state; unbounded, or q = b bits); reference rule: the running log-odds estimate, and the
                      agent centres its quantizer at -kappa * query. Exact decoder unchanged (cells are still
                      boxes). q = 0 is one-shot exactly (tested). Parallel mode (first round, then queries): hook.
  coa_chain_scalar [4] Chain-of-Agents reference: one b-bit state rewritten by every agent (running log-odds sum).
  fixed_state         fixed-size family at the append family's total: one S = N b bit state rewritten by every
                      agent. `run_n_sweep`: regret vs N at fixed total S for append (b = S / N) and fixed-size.
Learned arms (torch, lazy import; `train_learned_pair`, `run_learned`, LEARNED_ARMS): dvib [2] (aggregation-aware:
  per-agent encoder -> b-bit straight-through Gumbel code -> shared set decoder, end to end) and self_summary_vib
  (the same encoder trained to predict Y from its own message only, frozen, then a set decoder) are implemented;
  coa_chain (learned) [4], tree [5] and learned queries are hooks.

Environments: E2, E3-sybil, E3-collude, E4, Z8 (multi-piece agents; closed forms for llr_quantizer, extremized_q,
trimmed_q, queried, chains; top_w and k_of_n on E2/E3), and the zoo with several readings per agent through
`PiecesPerAgent` ("Z9-p2", "Z1-p2", "Z5-p2", "Z6-p2": base readers grouped p at a time), with exact tabular
decoders. E3-biased is excluded (its distortion would have to be applied to the message).

    OMP_NUM_THREADS=2 python3 -m aggbench.compression --envs E2 E3-sybil --bits 1 2 4 8 --out results.json
    python3 -m aggbench.compression --learned --envs E2 E3-sybil Z9-p2 --out-dir runs/   # torch, GPU
"""
from dataclasses import asdict, dataclass
import itertools
import json
import math
import sys

import numpy as np

from . import make_env
from .envs import _Env, Batch
from .metrics import bernoulli_logloss, logsig, per_episode, summarize, summarize_values

UNBOUNDED = None
CLIP = 60.0


# ============================================================================= channels

class BudgetError(ValueError):
    pass


@dataclass
class Channel:
    """An explicit b-bit channel: symbols must be integers in [0, 2^b). b = None: unbounded (passes anything,
    counts as infinite bits); b = 0: carries nothing (every symbol must be 0)."""
    bits: object

    @property
    def levels(self):
        return None if self.bits is None else 2 ** int(self.bits)

    def send(self, symbols):
        if self.bits is None:
            return symbols
        s = np.asarray(symbols)
        if not np.issubdtype(s.dtype, np.integer):
            raise BudgetError(f"a {self.bits}-bit channel carries integer symbols, got {s.dtype}")
        if s.size and (s.min() < 0 or s.max() >= self.levels):
            raise BudgetError(f"symbol outside [0, 2^{self.bits})")
        return s

    def total_bits(self, n_agents):
        return math.inf if self.bits is None else int(self.bits) * n_agents


def companded_thresholds(b, s=1.0):
    """2^b - 1 increasing thresholds s * Phi^-1(j / 2^b): equal-mass cells for a N(0, s^2) input."""
    from scipy.special import ndtri
    if b == 0:
        return np.zeros(0)
    return s * ndtri(np.arange(1, 2 ** b) / 2 ** b)


def uniform_thresholds(b, lo, hi):
    """2^b - 1 thresholds of a uniform quantizer over [lo, hi] (the outer cells extend to +-inf)."""
    return np.linspace(lo, hi, 2 ** b + 1)[1:-1]


def quantize(x, thresholds):
    """Cell index of x for increasing thresholds (..., L); thresholds may carry leading axes matching x."""
    t = np.asarray(thresholds)
    if t.ndim == 1:
        return np.searchsorted(t, x, side="right").astype(np.int64)
    return (np.asarray(x)[..., None] >= t).sum(-1).astype(np.int64)


def cell_bounds(sym, thresholds):
    """(lower, upper) of cell `sym`; thresholds (..., L) broadcast against sym."""
    t = np.asarray(thresholds)
    pad = np.concatenate([np.full(t.shape[:-1] + (1,), -np.inf), t, np.full(t.shape[:-1] + (1,), np.inf)], -1)
    if t.ndim == 1:
        return pad[sym], pad[sym + 1]
    sym = np.asarray(sym)[..., None]
    return np.take_along_axis(pad, sym, -1)[..., 0], np.take_along_axis(pad, sym + 1, -1)[..., 0]


def ste_quantize(x, thresholds, values):
    """Torch: hard scalar quantization forward (x -> values[cell]), identity gradient backward (straight-through).
    thresholds (L,), values (L + 1,) tensors. The forward output takes at most 2^b distinct values."""
    import torch
    idx = torch.bucketize(x, thresholds, right=True)
    hard = values[idx]
    return x + (hard - x).detach()


def gumbel_code(logits, tau=1.0, hard=True):
    """Torch: a discrete code of 2^b symbols (logits (..., 2^b)) with the straight-through Gumbel-softmax
    estimator; forward is one-hot, so the message is exactly one of 2^b symbols."""
    import torch.nn.functional as F
    return F.gumbel_softmax(logits, tau=tau, hard=hard)


# ============================================================================= env adapters

def env_family(batch):
    c = batch.config
    if "n_rollouts" in c:
        return "rollout"
    if "read_noise" in c and "strength_levels" in c:
        return "privateread"
    if "n_sybil" in c:
        return "pieces"
    if "base" in c and "per_agent" in c:
        return "grouped"
    if "report_logit" in batch.oracle and "sym" in batch.oracle:
        return "tabular"
    if batch.kind == "gaussian":
        return "gaussian"
    raise ValueError(f"no compression adapter for {batch.env}")


def local_statistic(obs, own, family, config, public):
    """Agent-local log-odds r (n, N) and its own-model scale (n, N), from each agent's OWN inputs only
    (agent_inputs(..., own_model=True)): obs and own of agent i give r[:, i]. The scale is the agent's marginal
    sd of r - l0 under its own model (E2/E3/Z8: sqrt(h_i^2 + v_i) with h_i = sum 2 mu^2 w.r.t. its readings)."""
    l0 = float(public["prior_logit"])
    if family == "pieces":
        K = obs.shape[-1]
        mu = own[..., K:2 * K]
        r = l0 + np.nansum(2 * mu * obs, -1)
        h = (2 * mu ** 2).sum(-1)
        return r, np.sqrt(h ** 2 + 2 * h)
    if family == "privateread":
        K = obs.shape[-1]
        mu, tau2 = own[..., K:2 * K], config["read_noise"] ** 2
        w = 2 * mu / (1 + tau2)
        r = l0 + np.nansum(w * obs, -1)
        h = (w * mu).sum(-1)
        v = (w ** 2).sum(-1) * (1 + tau2)
        return r, np.sqrt(h ** 2 + v)
    if family == "rollout":
        r = np.nansum(obs, -1)              # the rollout's own output: start value + its increments
        K = config["n_pieces"]
        cov, rec, cmu = own[..., :K], own[..., K:2 * K], own[..., 2 * K:3 * K]
        W = cov * config["calibration"] * config["recency"] ** (rec * K)
        m = 2 * cmu ** 2
        h = (W * m).sum(-1)
        v = (W ** 2 * (2 * m + config["read_noise"] ** 2)).sum(-1) + config["shared_bias"] ** 2 \
            + config["step_noise"] ** 2 * cov.sum(-1)
        return r, np.sqrt(h ** 2 + v)
    raise ValueError(family)


def agent_statistic(batch):
    """(r, scale) for a batch, through the agent-local path (unaligned inputs, own model, provenance none)."""
    fam = env_family(batch)
    if fam in ("tabular", "grouped"):
        r = np.round(batch.local["logit"], 9)       # the agent's log-odds to 1e-9: float noise carries no message
        return r, np.ones_like(r)
    obs, _, public, own = batch.agent_inputs("none", alignment="unaligned", own_model=True)
    return local_statistic(obs, own, fam, batch.config, public)


def gaussian_report_model(batch):
    """Per episode (c, h, S): given Y, r ~ N(c + (2Y - 1) h, S) for the agents' local log-odds r. Uses the true
    structure (decoder side only). E2/E3: h = C m, S = C diag(2m) C^T, m = 2 mu^2; Z8, E4 as in envs/zoo."""
    fam, o, c = env_family(batch), batch.oracle, batch.config
    l0 = float(batch.public["prior_logit"])
    n, N = batch.local["logit"].shape
    if fam == "pieces":
        C, m = o["cov"].astype(float), 2 * o["mu"] ** 2
        h = np.einsum("nik,nk->ni", C, m)
        S = np.einsum("nik,nk,njk->nij", C, 2 * m, C)
    elif fam == "privateread":
        tau2 = c["read_noise"] ** 2
        Wt = o["cov"] * (2 * o["mu"] / (1 + tau2))[:, None, :]
        h = np.einsum("nik,nk->ni", Wt, o["mu"])
        S = np.einsum("nik,njk->nij", Wt, Wt) + tau2 * np.einsum("nij,ni->nij", np.broadcast_to(np.eye(N), (n, N, N)),
                                                                  (Wt ** 2).sum(-1))
    elif fam == "rollout":
        W, m = o["W"], 2 * o["mu"] ** 2
        h = np.einsum("nik,nk->ni", W, m)
        S = (np.einsum("nik,nk,njk->nij", W, 2 * m, W) + c["read_noise"] ** 2 * np.einsum("nik,njk->nij", W, W)
             + c["shared_bias"] ** 2 + np.einsum("ij,ni->nij", np.eye(N), c["step_noise"] ** 2 * o["length"]))
    else:
        raise ValueError(f"no Gaussian report model for {batch.env}")
    return np.full((n, N), l0), h, S


# ============================================================================= exact decoder: Gaussian boxes

def _chol(S, tol=1e-10):
    """Batched lower Cholesky that tolerates zero pivots (duplicates, exact linear dependence): a zero pivot leaves
    the column zero, so that variable is a deterministic function of the earlier ones."""
    n, N, _ = S.shape
    L = np.zeros_like(S)
    scale = np.maximum(np.einsum("nii->n", S) / N, 1e-300)
    for j in range(N):
        d = S[:, j, j] - (L[:, j, :j] ** 2).sum(-1)
        ok = d > tol * scale
        L[:, j, j] = np.where(ok, np.sqrt(np.maximum(d, 0)), 0.0)
        piv = np.where(ok, L[:, j, j], 1.0)
        for i in range(j + 1, N):
            v = S[:, i, j] - (L[:, i, :j] * L[:, j, :j]).sum(-1)
            L[:, i, j] = np.where(ok, v / piv, 0.0)
    return L


def _qmc(M, d, seed=0):
    from scipy.stats import qmc
    u = qmc.Sobol(d=max(d, 1), scramble=True, seed=seed).random(M)
    return np.clip(u, 1e-12, 1 - 1e-12)


def box_logprob(mean, L, lo, hi, u):
    """log P(lo <= X <= hi) for X ~ N(mean, L L^T), batched over episodes: mean, lo, hi (n, N), L (n, N, N),
    u (M, N) QMC points. Genz's separation of variables, all in log space (stable for narrow tail cells)."""
    from scipy.special import log_ndtr, ndtri_exp
    n, N = mean.shape
    M = len(u)
    z = np.zeros((n, M, N))
    logp = np.zeros((n, M))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i in range(N):
            shift = mean[:, i, None] + np.einsum("nk,nmk->nm", L[:, i, :i], z[:, :, :i])
            d = L[:, i, i][:, None]
            det = d <= 0
            dd = np.where(det, 1.0, d)
            a = (lo[:, i, None] - shift) / dd
            b = (hi[:, i, None] - shift) / dd
            flip = (np.nan_to_num(a, nan=0.0, posinf=1e300, neginf=-1e300)
                    + np.nan_to_num(b, nan=0.0, posinf=1e300, neginf=-1e300)) > 0
            a2, b2 = np.where(flip, -b, a), np.where(flip, -a, b)
            la, lb = log_ndtr(a2), log_ndtr(b2)
            r = np.where(np.isfinite(la), np.exp(la - lb), 0.0)
            lm = lb + np.log1p(-np.minimum(r, 1.0))
            t = lb + np.log(r + u[None, :, i] * (1 - r))
            zz = ndtri_exp(np.minimum(t, 0.0))
            zz = np.clip(np.where(flip, -zz, zz), np.minimum(a, b), np.maximum(a, b))
            inside = (lo[:, i, None] <= shift) & (shift <= hi[:, i, None])
            logp = logp + np.where(det, np.where(inside, 0.0, -np.inf), lm)
            z[:, :, i] = np.where(det, 0.0, np.nan_to_num(zz))
        mx = logp.max(1, keepdims=True)
        out = np.log(np.exp(logp - np.where(np.isfinite(mx), mx, 0.0)).mean(1)) + np.where(np.isfinite(mx), mx, -np.inf)[:, 0]
    return out


def gaussian_cell_decoder(model, lo, hi, l0, M=1024, seed=0, chunk=128):
    """Exact Bayes log-odds of Y given that every agent's log-odds lies in its cell [lo, hi] (n, N) under the
    Gaussian report model (c, h, S). Infinite cells on every agent give the prior."""
    c, h, S = model
    n, N = c.shape
    u = _qmc(M, N, seed)
    L = _chol(S)
    out = np.empty(n)
    for s in range(0, n, chunk):
        sl = slice(s, s + chunk)
        lp1 = box_logprob(c[sl] + h[sl], L[sl], lo[sl], hi[sl], u)
        lp0 = box_logprob(c[sl] - h[sl], L[sl], lo[sl], hi[sl], u)
        with np.errstate(invalid="ignore"):
            d = lp1 - lp0
        d = np.where(np.isfinite(lp1) | np.isfinite(lp0), d, 0.0)
        out[sl] = l0 + np.clip(np.nan_to_num(d, nan=0.0), -CLIP, CLIP)
    return out


def gaussian_exact_logit(model, r, l0):
    """b = None: exact log-odds given the unquantized reports (= oracle_clearing): l0 + 2 h^T S^+ (r - c)."""
    c, h, S = model
    Sp = np.linalg.pinv(S, rcond=1e-10, hermitian=True)
    return l0 + 2 * np.einsum("ni,nij,nj->n", h, Sp, r - c)


# ============================================================================= exact decoder: tabular

def tabular_message_decoder(prior_w, target, lik, msg_of_symbol, msgs):
    """Exact Bayes log-odds of Y given each agent's message, for deterministic per-agent encoders over finite
    alphabets: msg_of_symbol (N, A) maps agent i's symbol to its message; P(m_i | w) = sum_{o: enc_i(o) = m_i}
    lik[i, w, o]."""
    from .zoo import posterior_logit
    N = lik.shape[0]
    cols = np.stack([(lik[i] * (msg_of_symbol[i] == msgs[i])[None, :]).sum(-1) for i in range(N)])
    return posterior_logit(prior_w, target, cols)


# ============================================================================= pieces per agent (zoo)

class PiecesPerAgent(_Env):
    """A zoo structure where each agent reads p symbols: base readers are grouped p at a time (base agents are iid
    or randomly permuted per episode, so contiguous groups are random groups). Agents stay conditionally
    independent given the world state, so every ceiling is still a finite sum: agent j's table is the product of
    its readers' tables over symbol tuples (alphabet A^p). Bayes is the base env's (same total evidence).
    `make_grouped("Z9", 2)` -> "Z9-p2": 14 base readers -> 7 agents with 2 readings each. Z1: n_agents base
    readers = p * n_agents; Z9 (readers_per_bit scheme): the base reader count must be divisible by p."""
    kind = "bernoulli"
    family = "grouped"

    def __init__(self, base, p, n_agents=None, **overrides):
        from .zoo import ZOO, KParityEnv
        self.base_name, self.p = base, int(p)
        if base.startswith("Z9"):
            self.base = ZOO[base](**overrides)
        else:
            nb = (n_agents or 6) * self.p
            self.base = ZOO[base](**{"n_agents": nb, **overrides})
        nb = self.base.config.n_agents
        if nb % self.p:
            raise ValueError(f"{base}: {nb} base readers are not divisible by p = {p}")
        self.N = nb // self.p
        self.name = f"{base}-p{p}"
        self.config = _GroupedConfig(base=base, per_agent=self.p, base_config=asdict(self.base.config))
        self._kparity = isinstance(self.base, KParityEnv)

    def public(self):
        return {"prior_logit": self.base.prior_logit(), "n_agents": self.N}

    def grouped_tables(self, st):
        prior_w, target, lik, values = self.base.tables(st)
        nb, W, A = lik.shape
        p, N = self.p, self.N
        g = lik.reshape(N, p, W, A)
        out = g[:, 0]
        for j in range(1, p):
            out = (out[..., :, None] * g[:, j][..., None, :]).reshape(N, W, -1)
        return prior_w, target, out

    def episode(self, seed):
        from .zoo import tabular_ceilings
        rng = np.random.default_rng(seed)
        st = self.base.structure(rng)
        prior_w, target, lik = self.grouped_tables(st)
        _, _, base_lik, values = self.base.tables(st)
        w, sym = (v[0] for v in self.base.simulate(st, rng, 1))
        A = base_lik.shape[-1]
        N, p = self.N, self.p
        gsym = (sym.reshape(N, p) * (A ** np.arange(p - 1, -1, -1))).sum(1)
        report, bayes, ceiling = tabular_ceilings(prior_w, target, lik, gsym)
        kf, kn = self.base.make_keys(st, rng)
        obs = values[np.arange(len(sym)), sym].reshape(N, -1)
        out = {"truth": np.int64(target[w]), "obs": obs, "local.logit": report, "report.logit": report.copy(),
               "bayes.logit": np.float64(bayes), "key.none": np.zeros((N, 0)), "key.noisy": kn.reshape(N, -1),
               "key.full": kf.reshape(N, -1), "oracle.w": np.int64(w), "oracle.sym": gsym,
               "oracle.base_sym": sym, "oracle.report_logit": np.float64(ceiling)}
        out.update({f"oracle.{k}": v for k, v in st.items()})
        return out

    def tables_for(self, batch, e):
        st = {k: v[e] for k, v in batch.oracle.items() if k not in ("w", "sym", "base_sym", "report_logit")}
        return self.grouped_tables(st)


@dataclass
class _GroupedConfig:
    base: str
    per_agent: int
    base_config: dict

    @property
    def n_agents(self):
        return None


GROUPED = {"Z9-p2": ("Z9", 2), "Z1-p2": ("Z1", 2), "Z5-p2": ("Z5", 2), "Z6-p2": ("Z6", 2)}


def make_cenv(name, **overrides):
    """make_env plus the grouped zoo names in GROUPED (Z9-p2 etc.)."""
    if name in GROUPED:
        base, p = GROUPED[name]
        return PiecesPerAgent(base, p, **overrides)
    return make_env(name, **overrides)


def _tabular_tables(env, batch, e):
    if isinstance(env, PiecesPerAgent):
        return env.tables_for(batch, e)
    pw, t, lik, _ = env.tables_for(batch, e)
    return pw, t, lik


# ============================================================================= self-summary: LLR quantizer [1]

@dataclass
class LLRQuantizer:
    """Each agent quantizes its own local log-odds into 2^b cells: thresholds l0 + scale_i * s * Phi^-1(j/2^b),
    scale_i = the agent's own-model sd ("own") or a pooled train constant ("shared")."""
    bits: object
    s: float = 1.0
    family: str = "own"
    pooled: float = 1.0
    center: float = 0.0

    def thresholds(self, scale, l0, offset=0.0):
        t = companded_thresholds(self.bits, self.s)
        sc = scale if self.family == "own" else np.full_like(scale, self.pooled)
        return l0 + np.asarray(offset)[..., None] + self.center + sc[..., None] * t

    def encode(self, r, scale, l0, offset=0.0):
        if self.bits is None:
            return r
        if self.bits == 0:
            return np.zeros(r.shape, np.int64)
        return Channel(self.bits).send(quantize(r, self.thresholds(scale, l0, offset)))


def llr_quantizer_forecast(env, batch, quant, M=1024, seed=0, r_scale=None):
    """Exact-decoder forecast for the LLR quantizer on a batch. Returns ({"logit"}, symbols)."""
    l0 = float(batch.public["prior_logit"])
    r, scale = agent_statistic(batch) if r_scale is None else r_scale
    n = len(r)
    if quant.bits == 0:
        return {"logit": np.full(n, l0)}, np.zeros(r.shape, np.int64)
    fam = env_family(batch)
    if fam in ("tabular", "grouped"):
        return _tabular_llr_forecast(env, batch, quant, r, scale, l0)
    model = gaussian_report_model(batch)
    if quant.bits is None:
        return {"logit": gaussian_exact_logit(model, r, l0)}, r
    sym = quant.encode(r, scale, l0)
    lo, hi = cell_bounds(sym, quant.thresholds(scale, l0))
    return {"logit": gaussian_cell_decoder(model, lo, hi, l0, M=M, seed=seed)}, sym


def _tabular_llr_forecast(env, batch, quant, r, scale, l0):
    from .zoo import local_table
    out = np.empty(len(batch))
    syms = np.zeros(r.shape, np.int64)
    for e in range(len(batch)):
        pw, tg, lik = _tabular_tables(env, batch, e)
        g = np.round(np.nan_to_num(local_table(pw, tg, lik), nan=l0), 9)   # (N, A); rounding: no float-noise leak
        sym = batch.oracle["sym"][e]
        if quant.bits is None:                                     # message = the report itself
            from .zoo import report_classes
            m = np.stack([report_classes(g[i], sym[i]) for i in range(len(sym))]).astype(int)
            msgs = np.ones(len(sym), int)
        else:
            th = quant.thresholds(np.ones(g.shape[0]), l0)
            m = quantize(g, th[:, None, :])
            msgs = quantize(r[e], th[0] if th.ndim > 1 else th)
            syms[e] = msgs
        out[e] = tabular_message_decoder(pw, tg, lik, m, msgs)
    return {"logit": np.clip(out, -CLIP, CLIP)}, syms


def raw_code_forecast(env, batch, b):
    """Closed-form AGGREGATION-AWARE reference for the tabular zoo: each agent sends its first k raw readings, k =
    min(p, floor(b / log2 A)) (A = alphabet of one reading, p = readings per agent); exact decoder by enumeration.
    Lossless once A^p <= 2^b. It ignores what the agent itself would predict from them (in Z9-p2 nothing)."""
    l0 = float(batch.public["prior_logit"])
    p = env.p if isinstance(env, PiecesPerAgent) else 1
    out = np.empty(len(batch))
    for e in range(len(batch)):
        pw, tg, lik = _tabular_tables(env, batch, e)
        Ap = lik.shape[-1]
        A = int(round(Ap ** (1 / p)))
        k = 0 if b == 0 else (p if b is None else min(p, int(b // math.log2(A)) if A > 1 else p))
        div = A ** (p - k)
        enc = np.broadcast_to(np.arange(Ap) // div, (lik.shape[0], Ap))
        out[e] = tabular_message_decoder(pw, tg, lik, enc, batch.oracle["sym"][e] // div)
    return {"logit": np.clip(np.where(np.isfinite(out), out, l0), -CLIP, CLIP)}


def _kl(forecast, batch):
    return float(per_episode(forecast, batch)["kl"].mean())


def tune_llr_quantizer(env, train, b, M=256, n_fit=400, grid=(0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8)):
    """Family ("own" / "shared") and s chosen on train episodes by the exact decoder's KL to Bayes (grid search;
    the companded family has one free parameter, so this is optimal within the family, not over all
    quantizers). b = 1 with a symmetric prior: one threshold at the prior, s is irrelevant."""
    if b in (None, 0):
        return LLRQuantizer(b)
    sub = _subset(train, n_fit)
    r, scale = agent_statistic(sub)
    l0 = float(sub.public["prior_logit"])
    pooled = float(np.std(r - l0)) or 1.0
    best = None
    fams = ("own",) if env_family(sub) in ("tabular", "grouped") else ("own", "shared")
    for fam in fams:
        for s in (grid if b > 1 else (1.0,)):
            q = LLRQuantizer(b, s, fam, pooled)
            k = _kl(llr_quantizer_forecast(env, sub, q, M=M, r_scale=(r, scale))[0], sub)
            if best is None or k < best[0]:
                best = (k, q)
    return best[1]


def _subset(batch, n):
    if len(batch) <= n:
        return batch
    take = lambda d: {k: v[:n] for k, v in d.items()}
    return Batch(batch.env, batch.config, batch.kind, batch.seeds[:n], batch.truth[:n], batch.obs[:n],
                 take(batch.local), take(batch.reports), take(batch.keys), take(batch.bayes), take(batch.oracle),
                 dict(batch.public))


# ============================================================================= practitioner floor [7]

def _fit_scalar(loss, lo=0.05, hi=20.0):
    grid = np.linspace(np.log(lo), np.log(hi), 41)
    j = int(np.argmin([loss(np.exp(g)) for g in grid]))
    a, b = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
    g = (np.sqrt(5) - 1) / 2
    for _ in range(30):
        c, d = b - g * (b - a), a + g * (b - a)
        if loss(np.exp(c)) < loss(np.exp(d)):
            b = d
        else:
            a = c
    return float(np.exp((a + b) / 2))


def _pool(x, how):
    if how == "mean":
        return x.mean(1)
    from scipy.stats import trim_mean
    return trim_mean(x, 0.2, axis=1)


@dataclass
class PooledQuant:
    """extremized_q / trimmed_q: l0 + alpha * pool_i(centroid of agent i's cell), shared thresholds
    s * pooled_sd * Phi^-1(j / 2^b) on r - l0, cell centroids from train. bits None: the raw log-odds."""
    bits: object
    how: str = "mean"
    s: float = 1.0
    pooled: float = 1.0
    alpha: float = 1.0
    centroids: np.ndarray = None

    def thresholds(self):
        return self.s * self.pooled * companded_thresholds(self.bits)

    def values(self, r, l0):
        z = r - l0
        if self.bits is None:
            return z
        if self.bits == 0:
            return np.zeros_like(z)
        return self.centroids[quantize(z, self.thresholds())]

    def forecast(self, r, l0):
        return {"logit": l0 + self.alpha * _pool(self.values(r, l0), self.how)}


def fit_pooled_quant(train, b, how="mean", grid=(0.5, 0.75, 1.0, 1.5, 2.0)):
    r, _ = agent_statistic(train)
    l0 = float(train.public["prior_logit"])
    z = r - l0
    pooled = float(np.std(z)) or 1.0
    best = None
    for s in (grid if b not in (None, 0) else (1.0,)):
        pq = PooledQuant(b, how, s, pooled)
        if b not in (None, 0):
            cells = quantize(z, pq.thresholds())
            t = np.concatenate([[pq.thresholds()[0]], pq.thresholds(), [pq.thresholds()[-1]]])
            mids = (t[:-1] + t[1:]) / 2
            sums = np.bincount(cells.ravel(), z.ravel(), 2 ** b)
            cnt = np.bincount(cells.ravel(), minlength=2 ** b)
            pq.centroids = np.where(cnt > 0, sums / np.maximum(cnt, 1), mids)
        v = _pool(pq.values(r, l0), how)
        loss = lambda a: float(bernoulli_logloss(l0 + a * v, train.truth).mean())
        pq.alpha = _fit_scalar(loss)
        L = loss(pq.alpha)
        if best is None or L < best[0]:
            best = (L, pq)
    return best[1]


# ============================================================================= top-w pieces [3] and k-of-N [6]

def _piece_inputs(batch):
    """Aligned per-agent readings and own models in E2/E3: obs (n, N, K) with NaN, mu (n, N, K) (0 = unseen)."""
    if env_family(batch) != "pieces":
        raise ValueError("top_w / k_of_n are implemented for E2/E3 (pieces); Z8 and E4 are hooks")
    obs, _, public, own = batch.agent_inputs("none", alignment="aligned", own_model=True)
    K = obs.shape[-1]
    return obs, own[..., K:2 * K], float(public["prior_logit"])


def _piece_cell_llr(x, mu, q, s):
    """Exact log P(cell | Y=1) / P(cell | Y=0) of the q-bit cell of reading x ~ N(+-mu, 1)."""
    from scipy.special import log_ndtr
    if q is None:
        return 2 * mu * x
    sc = np.sqrt(1 + mu ** 2)
    base = companded_thresholds(q, s)
    cell = quantize(x / sc, base)
    lo, hi = cell_bounds(cell, base)
    lo, hi = lo * sc, hi * sc

    def logmass(m):
        a, b = lo - m, hi - m
        flip = (np.nan_to_num(a, posinf=1e300, neginf=-1e300) + np.nan_to_num(b, posinf=1e300, neginf=-1e300)) > 0
        a2, b2 = np.where(flip, -b, a), np.where(flip, -a, b)
        la, lb = log_ndtr(a2), log_ndtr(b2)
        with np.errstate(over="ignore", invalid="ignore"):
            return lb + np.log1p(-np.exp(la - lb))
    return logmass(mu) - logmass(-mu)


def _union_logit(sent, x, mu, q, s, l0):
    """sent (n, N, K) bool: which readings each agent sent; the decoder counts each distinct piece once (E2/E3
    readings of one piece are identical, and every agent uses the same per-piece quantizer)."""
    anyk = sent.any(1)                                            # (n, K)
    xk = np.nanmax(np.where(sent, x, -np.inf), 1)
    muk = np.max(np.where(sent, mu, 0.0), 1)
    with np.errstate(invalid="ignore"):
        llr = np.where(anyk, _piece_cell_llr(np.where(anyk, xk, 0.0), np.where(anyk, muk, 0.1), q, s), 0.0)
    return np.clip(l0 + llr.sum(-1), -CLIP, CLIP)


def top_w_messages(x, mu, b, q):
    """Which pieces each agent sends under b bits with q bits per reading: the w_i largest |2 mu x|, with
    w_i = max w such that w q + ceil(log2 C(n_i, w)) <= b. Returns (sent (n, N, K) bool, bits used (n, N))."""
    seen = np.isfinite(x)
    ni = seen.sum(-1)
    n, N, K = x.shape
    if b is None:
        return seen, np.full((n, N), np.inf)
    w = np.zeros((n, N), int)
    used = np.zeros((n, N), int)
    for cnt in np.unique(ni):
        best_w, best_c = 0, 0
        for ww in range(int(cnt) + 1):
            cost = ww * q + (math.ceil(math.log2(math.comb(int(cnt), ww))) if ww else 0)
            if cost <= b:
                best_w, best_c = ww, cost
        w[ni == cnt], used[ni == cnt] = best_w, best_c
    score = np.where(seen, np.abs(2 * mu * np.nan_to_num(x)), -1.0)
    rank = np.argsort(np.argsort(-score, -1), -1)
    return seen & (rank < w[..., None]), used


def top_w_forecast(batch, b, q, s):
    x, mu, l0 = _piece_inputs(batch)
    sent, used = top_w_messages(x, mu, b, q)
    return {"logit": _union_logit(sent, x, mu, None if b is None else q, s, l0)}, used


def fit_top_w(train, b, n_fit=1000):
    if b is None:
        return {"q": None, "s": 1.0}
    sub = _subset(train, n_fit)
    best = None
    for q in range(1, b + 1):
        for s in (0.75, 1.0, 1.5):
            k = _kl(top_w_forecast(sub, b, q, s)[0], sub)
            if best is None or k < best[0]:
                best = (k, {"q": q, "s": s})
    return best[1]


def k_of_n_messages(x, mu, b, q):
    """Scheduler: agents in decreasing own-model information sum 2 mu^2 send ALL their readings at q bits each,
    while the total stays within N b bits. Returns (sent (n, N, K), bits used (n, N))."""
    seen = np.isfinite(x)
    n, N, K = x.shape
    if b is None:
        return seen, np.full((n, N), np.inf)
    info = (2 * mu ** 2 * seen).sum(-1)
    cost = seen.sum(-1) * q
    order = np.argsort(-info, 1, kind="stable")
    chosen = np.zeros((n, N), bool)
    left = np.full(n, N * b)
    for j in range(N):
        a = order[:, j]
        c = cost[np.arange(n), a]
        ok = c <= left
        chosen[np.arange(n), a] = ok
        left = left - np.where(ok, c, 0)
    return seen & chosen[..., None], np.where(chosen, cost, 0)


def k_of_n_forecast(batch, b, q, s):
    x, mu, l0 = _piece_inputs(batch)
    sent, used = k_of_n_messages(x, mu, b, q)
    return {"logit": _union_logit(sent, x, mu, None if b is None else q, s, l0)}, used


def fit_k_of_n(train, b, n_fit=1000):
    if b is None:
        return {"q": None, "s": 1.0}
    sub = _subset(train, n_fit)
    best = None
    for q in range(1, max(b, 1) + 1):
        for s in (0.75, 1.0, 1.5):
            k = _kl(k_of_n_forecast(sub, b, q, s)[0], sub)
            if best is None or k < best[0]:
                best = (k, {"q": q, "s": s})
    return best[1]


# ============================================================================= queried protocol (append family)

@dataclass
class QueryProtocol:
    """APPEND family with queries (reference implementation; learned query / encoder modules are hooks).

    The consensus state is ONLY the b-bit messages received so far (n, t): S = N b bits at the end. Agents speak in
    index order (the environments shuffle agent positions per episode). Before agent t compresses, the consensus
    answers it a QUERY that is a pure function of the state (`consensus_estimate(state)`) and writes nothing to it.
    Headline (Kelvin): q = None, the query is unbounded (the agent may read the whole consensus state); q = 0 is
    one-shot (the key comparison); q = b is a minor arm (the query passes a q-bit downlink). Reference query: the
    running log-odds estimate sum_j (offset_j + centroid[m_j]) from the earlier symbols (pooled per-cell centroids
    in nats, from train; no structure, no agent scales). Agent t then quantizes its own log-odds with the LLR
    quantizer centred at -kappa * (query value): finer cells where its evidence would move the consensus across
    even odds. Its b-bit message depends only on its own inputs and the query. The exact decoder applies unchanged:
    given the realized messages each agent's cell is a known interval (its offset is a function of the earlier
    messages), so P(messages | Y) is the same Gaussian box probability. q = 0 or kappa = 0 is the one-shot LLR
    quantizer exactly (tested)."""
    quant: LLRQuantizer
    q: object = 0
    kappa: float = 0.5
    qs: float = 1.0
    centroids: np.ndarray = None      # (2^b,) mean of r - l0 - offset in each cell, pooled over agents (nats)

    def _qthresholds(self):
        return companded_thresholds(self.q, self.qs * self.quant.pooled)

    def query_symbol(self, est):
        """The downlink: unbounded (q None) passes the estimate, q = 0 nothing, else a q-bit symbol."""
        if self.q is None:
            return np.asarray(est, float)
        if self.q == 0:
            return np.zeros(np.shape(est), np.int64)
        return Channel(self.q).send(quantize(est, self._qthresholds()))

    def query_value(self, sym):
        if self.q is None:
            return np.asarray(sym, float)
        if self.q == 0:
            return np.zeros(np.shape(sym))
        t = self._qthresholds()
        step = t[1] - t[0] if len(t) > 1 else 1.0
        tt = np.concatenate([[t[0] - step], t, [t[-1] + step]])
        return ((tt[:-1] + tt[1:]) / 2)[sym]

    def offset_from_query(self, qsym):
        return np.zeros(np.shape(qsym)) if self.q == 0 else -self.kappa * self.query_value(qsym)

    def consensus_estimate(self, state):
        """Pure function of the consensus state (the b-bit symbols received so far, (n, t)): the running log-odds
        estimate minus l0 that the reference query reports. Reads the state, never writes it."""
        state = np.asarray(state)
        est = np.zeros(state.shape[0])
        if self.centroids is None:
            return est
        for j in range(state.shape[1]):
            off = self.offset_from_query(self.query_symbol(est))
            est = est + off + self.centroids[state[:, j]]
        return est

    def run(self, r, scale, l0):
        """Returns (uplink symbols = the final consensus state (n, N), offsets (n, N), queries sent (n, N))."""
        n, N = r.shape
        state = np.zeros((n, 0), np.int64)
        offs = np.zeros((n, N))
        qsym = np.zeros((n, N), float if self.q is None else np.int64)
        for t in range(N):
            view = state.view()
            view.setflags(write=False)                      # the query reads the state and cannot write it
            qs = self.query_symbol(self.consensus_estimate(view))
            qsym[:, t] = qs
            offs[:, t] = self.offset_from_query(qs)
            m = self.quant.encode(r[:, t], scale[:, t], l0, offs[:, t])
            state = np.concatenate([state, Channel(self.quant.bits).send(m)[:, None]], 1)   # the only write
        return state, offs, qsym


def queried_forecast(env, batch, proto, M=1024, seed=0, r_scale=None):
    l0 = float(batch.public["prior_logit"])
    r, scale = agent_statistic(batch) if r_scale is None else r_scale
    if proto.quant.bits in (None, 0):
        return llr_quantizer_forecast(env, batch, proto.quant, M=M, seed=seed, r_scale=(r, scale))[0]
    syms, offs, _ = proto.run(r, scale, l0)
    lo, hi = cell_bounds(syms, proto.quant.thresholds(scale, l0, offs))
    return {"logit": gaussian_cell_decoder(gaussian_report_model(batch), lo, hi, l0, M=M, seed=seed)}


def fit_query_protocol(env, train, quant, q, M=256, n_fit=400, kappas=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Pooled cell centroids from train (one-shot cells), then kappa (and the q-bit query scale) by the exact
    decoder's KL on train episodes."""
    sub = _subset(train, n_fit)
    r, scale = agent_statistic(sub)
    l0 = float(sub.public["prior_logit"])
    proto = QueryProtocol(quant, q)
    if quant.bits in (None, 0) or q == 0:
        proto.kappa = 0.0
        return proto
    sym = quant.encode(r, scale, l0)
    L = 2 ** quant.bits
    cnt = np.bincount(sym.ravel(), minlength=L)
    proto.centroids = np.bincount(sym.ravel(), (r - l0).ravel(), L) / np.maximum(cnt, 1)
    best = None
    for kappa in kappas:
        for qs in ((1.0,) if q is None else (0.5, 1.0, 2.0)):
            proto.kappa, proto.qs = kappa, qs
            k = _kl(queried_forecast(env, sub, proto, M=M, r_scale=(r, scale)), sub)
            if best is None or k < best[0]:
                best = (k, kappa, qs)
    proto.kappa, proto.qs = best[1], best[2]
    return proto


# ============================================================================= Chain-of-Agents reference [4]

@dataclass
class FixedStateChain:
    """Fixed-size consensus state: ONE S-bit state that every agent reads and rewrites (its update is its message and
    the only write), so the consensus does not grow with N. Two arms use it: `coa_chain_scalar` (Chain-of-Agents [4]
    reference, S = b, the per-agent message) and `fixed_state` (the fixed-size family at the append family's total,
    S = N b). Closed-form scalar reference: the state codes the
    running log-odds sum v (nats, minus l0) on a uniform S-bit grid over [-R, R]; agent t reads v, adds
    w (r_t - l0) from its own inputs and writes Q_S(v + w (r_t - l0)). Its update is its message and the only
    write. Readout l0 + v_N (w and R fit on train; w < 1 shrinks for overlap it cannot see). S = None: the
    unquantized running sum. A scalar state cannot dedupe overlap, so this reference stops improving after a few
    bits; the learned S-bit chain (LEARNED_ARMS["coa_chain"]) is where a richer state is tested."""
    S: object
    R: float = 8.0
    w: float = 1.0

    def run(self, r, l0):
        """Returns (final state value (n,), the writes (n, N) S-bit symbols or None if unbounded)."""
        n, N = r.shape
        v = np.zeros(n)
        writes = []
        for t in range(N):
            v = v + self.w * (r[:, t] - l0)
            if self.S is not None:
                if self.S == 0:
                    sym = np.zeros(n, np.int64)
                    v = np.zeros(n)
                else:
                    levels = 2 ** min(self.S, 52) - 1   # S > 52: a double already holds the exact value
                    sym = np.clip(np.rint((v + self.R) / (2 * self.R) * levels), 0, levels).astype(np.int64)
                    sym = Channel(min(self.S, 62)).send(sym)
                    v = -self.R + 2 * self.R * sym / levels
                writes.append(sym)
        return v, (np.stack(writes, 1) if writes else None)

    def forecast(self, r, l0):
        return {"logit": l0 + self.run(r, l0)[0]}


def fit_fixed_state(train, S, Rs=(2.0, 4.0, 6.0, 8.0, 12.0)):
    r, _ = agent_statistic(train)
    l0 = float(train.public["prior_logit"])
    best = None
    for R in (Rs if S not in (None, 0) else (8.0,)):
        for w in np.linspace(0.2, 1.2, 11):
            fs = FixedStateChain(S, R, float(w))
            L = float(bernoulli_logloss(fs.forecast(r, l0)["logit"], train.truth).mean())
            if best is None or L < best[0]:
                best = (L, fs)
    return best[1]


# ============================================================================= locality checks

def check_message_locality(encode, obs, own, received=None, trials=3, seed=0):
    """A message may depend only on the sender's own inputs and on what it RECEIVED (queries, chain state).
    encode(obs, own, received) -> (n, N) symbols. Perturbs the other agents' obs and own models (received held
    fixed) and requires agent i's message to stay the same. Then, if `received` is given, perturbs agent i's
    received input and requires only agent i's message to be free to change (the others must not)."""
    rng = np.random.default_rng(seed)
    base = np.asarray(encode(obs, own, received))
    N = obs.shape[1]
    for _ in range(trials):
        i = int(rng.integers(N))
        others = [j for j in range(N) if j != i]
        o2, w2 = obs.copy(), own.copy()
        fin = np.isfinite(o2[:, others])
        o2[:, others] = np.where(fin, o2[:, others] + rng.normal(size=fin.shape), o2[:, others])
        w2[:, others] = w2[:, others] * np.exp(0.3 * rng.normal(size=w2[:, others].shape))
        out = np.asarray(encode(o2, w2, received))
        if not np.array_equal(out[:, i], base[:, i]):
            raise AssertionError(f"agent {i}'s message changed when other agents' inputs changed")
        if received is not None:
            rc = np.array(received, copy=True)
            rc[:, i] = rc[:, i] + 1 if np.issubdtype(rc.dtype, np.integer) else rc[:, i] + rng.normal(size=rc[:, i].shape)
            out = np.asarray(encode(obs, own, rc))
            if not np.array_equal(out[:, others], base[:, others]):
                raise AssertionError(f"agent {i}'s received input changed other agents' messages")
    return True


def llr_encoder(batch, quant):
    """encode(obs, own, received) for the LLR quantizer (received = query offsets, or None): local by construction."""
    fam = env_family(batch)
    l0 = float(batch.public["prior_logit"])

    def encode(obs, own, received=None):
        r, scale = local_statistic(obs, own, fam, batch.config, batch.public)
        return quant.encode(r, scale, l0, 0.0 if received is None else received)
    return encode


# ============================================================================= learned arms (hooks)

LEARNED_ARMS = {
    "dvib": "IMPLEMENTED (train_learned_pair): aggregation-aware [2], per-agent encoder over its own token -> b-bit "
            "discrete code (straight-through Gumbel) -> shared set decoder, end to end on log loss (+ beta * rate)",
    "self_summary_vib": "IMPLEMENTED (train_learned_pair): self-summary, the same encoder trained to predict Y from "
                        "its OWN message only, frozen; then a set decoder trained on the frozen codes",
    "coa_chain": "HOOK [4]: fixed-width b-bit state passed agent to agent (state_t = enc(own_t, state_{t-1})), "
                 "manager readout of the last state; sequential.py broadcast mode with a b-bit channel.",
    "tree": "HOOK [5]: b-bit merge net over log2 N levels (merge(m_a, m_b) -> b-bit message).",
    "queried_learned": "HOOK (gemini): learned query / encoder modules for QueryProtocol (sequential and "
                       "parallel); the reference implementation is QueryProtocol.",
}


# ============================================================================= metrics

def curve(regrets, floor_regret=0.0, ref_regret=None, ref_name="full_context_transformer"):
    """regrets: {b: regret (float)} including b = 0. Adds rho(b) w.r.t. the floor and (if given) a reference
    (the full-context transformer), and b90 (smallest finite b with rho >= 0.9; None if never)."""
    R0 = regrets[0]
    out = {"rho": {}, "b90": None}
    for b, R in regrets.items():
        out["rho"][b] = (R0 - R) / (R0 - floor_regret) if R0 > floor_regret else float("nan")
    finite = sorted(b for b in regrets if b is not None and b > 0)
    out["b90"] = next((b for b in finite if out["rho"][b] >= 0.9), None)
    if ref_regret is not None:
        out[f"rho_{ref_name}"] = {b: (R0 - R) / (R0 - ref_regret) for b, R in regrets.items()}
        out[f"b90_{ref_name}"] = next((b for b in finite if out[f"rho_{ref_name}"][b] >= 0.9), None)
    return out


def gain(forecast_self, forecast_aware, batch, n_boot=2000):
    """Aggregation-aware gain(b) = regret(self-summary) - regret(aggregation-aware), paired, with a bootstrap CI."""
    a = per_episode(forecast_self, batch)["kl"]
    b = per_episode(forecast_aware, batch)["kl"]
    return summarize_values(a - b, n_boot)


def transformer_anchor(env_name):
    """Regret of the full-context transformer (fresh regime, unaligned, provenance none) from the cached table."""
    from pathlib import Path
    p = Path(__file__).resolve().parent / "baselines_transformer_fresh.json"
    try:
        for r in json.loads(p.read_text())["results"]:
            if r["env"]["name"] == env_name and r.get("visibility") in ("none", None) and \
                    r["run_id"].endswith("unaligned-none-fresh"):
                return float(r["methods"]["full_context_transformer"]["regret"]["mean"])
    except (OSError, KeyError, ValueError):
        return None
    return None


# ============================================================================= runner

APPEND_ARMS = ("llr_quantizer", "queried_llr_qinf", "queried_llr_qb", "extremized_q", "trimmed_q", "top_w", "k_of_n")


def _entry(fc, dv, ll_floor, n_boot, S, b, q, extra=None):
    m = summarize(fc, dv, n_boot=n_boot)
    rf = summarize_values(bernoulli_logloss(fc["logit"], dv.truth) - ll_floor, n_boot)
    ent = {"regret": m["kl"], "regret_to_floor": rf, "logloss": m["logloss"], "consensus_bits": S,
           "uplink_bits_per_agent": b, "query": "none (one-shot)" if q == 0 else ("unbounded" if q is None else f"{q} bits")}
    ent.update(extra or {})
    return ent


def run_closed_form(env_name, bits=(1, 2, 4, 8), n_train=2000, n_dev=2000, M=1024, n_boot=2000, verbose=True,
                    env_overrides=None, arms=None):
    """Every closed-form arm at every b on one env; returns a JSON-able dict. Primary axis: b bits per agent message
    written into the consensus. consensus_bits (N b; coa_chain: the b-bit state) is an extra column."""
    from .report import achievable_floor, prior_forecast
    env = make_cenv(env_name, **(env_overrides or {}))
    tr, dv = env.sample("train", n_train), env.sample("dev", n_dev)
    fam = env_family(dv)
    r_dv = agent_statistic(dv)
    N = r_dv[0].shape[1]
    l0 = float(dv.public["prior_logit"])
    floor = achievable_floor(dv) if fam not in ("tabular", "grouped") else dict(dv.bayes)
    ll_floor = bernoulli_logloss(floor["logit"], dv.truth)
    res = {"env": env_name, "overrides": env_overrides or {}, "family": fam, "n_agents": int(N), "n_train": n_train,
           "n_dev": n_dev, "qmc_points": M, "arms": {}, "anchors": {}}
    res["anchors"]["prior"] = summarize(prior_forecast(dv), dv, n_boot=n_boot)["kl"]
    res["anchors"]["floor"] = summarize(floor, dv, n_boot=n_boot)["kl"]
    res["anchors"]["full_context_transformer_regret"] = None if env_overrides else transformer_anchor(env_name)
    want = lambda a: arms is None or a in arms
    gaussian = fam not in ("tabular", "grouped")

    def record(arm, b, fc, q=0, extra=None, S=None):
        key = "inf" if b is None else str(b)
        S = (None if b is None else N * b) if S is None else S
        res["arms"].setdefault(arm, {})[key] = _entry(fc, dv, ll_floor, n_boot, S, b, q, extra)
        if verbose:
            m = res["arms"][arm][key]["regret"]
            print(f"[{env_name} N={N}] {arm:>16} b={key:>3} S={S}: regret {m['mean']:.4f} "
                  f"[{m['ci_low']:.4f}, {m['ci_high']:.4f}]", file=sys.stderr, flush=True)

    for b in [0] + list(bits) + [None]:
        quant = tune_llr_quantizer(env, tr, b)
        fc, sym = llr_quantizer_forecast(env, dv, quant, M=M, r_scale=r_dv)
        extra = {"family": quant.family, "s": quant.s}
        if b not in (None, 0):
            extra["distinct_symbols_max"] = int(max(len(np.unique(sym[:, i])) for i in range(N)))
        record("llr_quantizer", b, fc, 0, extra)
        if not gaussian:
            fr = raw_code_forecast(env, dv, b)
            record("raw_code", b, fr, 0, {"gain_vs_llr_quantizer": gain(fc, fr, dv, n_boot=n_boot)})
        if gaussian and b not in (0, None):
            for q, arm in ((None, "queried_llr_qinf"), (b, "queried_llr_qb")):
                if want(arm):
                    proto = fit_query_protocol(env, tr, quant, q, M=256)
                    record(arm, b, queried_forecast(env, dv, proto, M=M, r_scale=r_dv), q,
                           {"kappa": proto.kappa, "qs": proto.qs})
        if gaussian:
            for how, arm in (("mean", "extremized_q"), ("trim", "trimmed_q")):
                if want(arm):
                    pq = fit_pooled_quant(tr, b, how)
                    record(arm, b, pq.forecast(r_dv[0], l0), 0, {"alpha": pq.alpha, "s": pq.s})
            if want("coa_chain_scalar"):
                fs = fit_fixed_state(tr, b)
                record("coa_chain_scalar", b, fs.forecast(r_dv[0], l0), 0,
                       {"R": fs.R, "w": fs.w, "note": "Chain-of-Agents [4] reference: each agent rewrites one b-bit "
                                                      "state (its message)"}, S=b)
            if want("fixed_state"):
                S = None if b is None else N * b
                fs = fit_fixed_state(tr, S)
                record("fixed_state", b, fs.forecast(r_dv[0], l0), 0,
                       {"R": fs.R, "w": fs.w, "note": "fixed-size family: one S = N b bit state rewritten by every "
                                                      "agent (matched total consensus size); key = the matched b"}, S=S)
        if fam == "pieces":
            for arm, fit, fcast in (("top_w", fit_top_w, top_w_forecast), ("k_of_n", fit_k_of_n, k_of_n_forecast)):
                if not want(arm):
                    continue
                if b == 0:
                    record(arm, 0, prior_forecast(dv))
                    continue
                p = fit(tr, b)
                fc, used = fcast(dv, b, p["q"], p["s"])
                record(arm, b, fc, 0, {**p, "max_bits_used_per_agent": None if b is None else int(used.max()),
                                       "max_total_bits_used": None if b is None else int(used.sum(1).max())})
    fl = res["anchors"]["floor"]["mean"]
    ref = res["anchors"]["full_context_transformer_regret"]
    res["curves"] = {}
    for arm, by_b in res["arms"].items():
        regs = {(None if k == "inf" else int(k)): v["regret"]["mean"] for k, v in by_b.items()}
        regs.setdefault(0, res["anchors"]["prior"]["mean"])
        c = curve(regs, fl, ref)
        c["consensus_bits_at_b90"] = None if c["b90"] is None else (c["b90"] if arm == "coa_chain_scalar"
                                                                    else N * c["b90"])
        res["curves"][arm] = {k: ({str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v)
                              for k, v in c.items()}
    return res


def run_n_sweep(env_name="E2", Ns=(2, 4, 8, 16), S=16, n_train=2000, n_dev=2000, M=1024, n_boot=2000):
    """Second axis: regret vs N at a FIXED total consensus size S bits. Append family with b = S / N per agent
    (one-shot and queried), fixed-size family with one S-bit state."""
    out = []
    for N in Ns:
        if S % N:
            raise ValueError(f"S = {S} is not divisible by N = {N}")
        b = S // N
        r = run_closed_form(env_name, bits=(b,), n_train=n_train, n_dev=n_dev, M=M, n_boot=n_boot,
                            env_overrides={"n_agents": N},
                            arms=("llr_quantizer", "queried_llr_qinf", "extremized_q", "fixed_state"))
        out.append({"N": N, "S": S, "b": b, "n_agents_total": r["n_agents"],
                    "regret": {a: v[str(b)]["regret"] for a, v in r["arms"].items() if str(b) in v},
                    "regret_unbounded": {a: v["inf"]["regret"] for a, v in r["arms"].items() if "inf" in v},
                    "prior": r["anchors"]["prior"]})
    return out


# ============================================================================= learned arms (torch, lazy)

def agent_tokens(batch):
    """One token per agent from its OWN inputs only (unaligned, own model, provenance none): E2-E4 / Z8 the
    full-context transformer's token; grouped zoo [readings, own model per reading, local log-odds / 4]."""
    if env_family(batch) == "grouped":
        n, N = batch.local["logit"].shape
        return np.concatenate([batch.obs.reshape(n, N, -1), batch.keys["full"].reshape(n, N, -1),
                               batch.local["logit"][..., None] / 4], -1).astype(np.float32)
    from .transformer import full_context
    return full_context(batch, "none", "unaligned").tokens


def _learned_net(F, b, width=64, enc_hidden=128, layers=2, heads=4, cont_dim=16):
    import torch
    nn = torch.nn
    V = cont_dim if b is None else 2 ** b

    class Net(nn.Module):
        """encoder (per agent, own token only) -> b-bit channel -> symbol embedding -> set decoder (no positions)."""

        def __init__(self):
            super().__init__()
            self.register_buffer("x_mean", torch.zeros(F))
            self.register_buffer("x_sd", torch.ones(F))
            self.enc = nn.Sequential(nn.Linear(F, enc_hidden), nn.GELU(), nn.Linear(enc_hidden, enc_hidden), nn.GELU(),
                                     nn.Linear(enc_hidden, V))
            self.own_head = nn.Sequential(nn.Linear(V, 64), nn.GELU(), nn.Linear(64, 1))   # self-summary readout
            self.emb = nn.Linear(V, width)
            self.cls = nn.Parameter(0.02 * torch.randn(1, 1, width))
            layer = nn.TransformerEncoderLayer(width, heads, 4 * width, 0.0, batch_first=True, norm_first=True)
            self.dec = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

        def logits(self, x):
            return self.enc((x - self.x_mean) / self.x_sd)

        def message(self, x, tau=1.0, hard_eval=False):
            """The channel: training = straight-through Gumbel one-hot (exactly one of 2^b symbols forward);
            eval = argmax one-hot. b = None: the unquantized cont_dim vector (diagnostic, not a budget)."""
            z = self.logits(x)
            if b is None:
                return z
            if hard_eval:
                return torch.nn.functional.one_hot(z.argmax(-1), V).float()
            return gumbel_code(z, tau=tau)

        def decode(self, m):
            h = self.emb(m)
            h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1)
            return self.out(self.dec(h)[:, 0])[:, 0]

        def own_logit(self, m):
            return self.own_head(m)[..., 0]

    return Net()


def _train_loop(net, params, loss_fn, X, Y, steps, batch, lr, device, log, tag, tau0=1.0):
    import torch
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    n = len(X)
    g = torch.Generator(device="cpu").manual_seed(0)
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        x, y = X[idx].to(device), Y[idx].to(device)
        tau = max(0.5, tau0 - step / max(steps, 1))
        loss = loss_fn(x, y, tau)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps - 1:
            log.append({"arm": tag, "step": step, "loss": loss.item()})


def train_learned_pair(env_name, b, n_train=100_000, n_dev=2000, steps=4000, batch=512, lr=1e-3, beta=0.0,
                       device=None, seed=0, n_boot=2000, train_batch=None, dev_batch=None):
    """Aggregation-aware (D-VIB-style [2]: encoder + b-bit code + set decoder, end to end; beta * rate where rate =
    log 2^b - H(code distribution) per message) versus SELF-SUMMARY (the same encoder trained to predict Y from its
    own message alone through a per-agent head, then frozen; a set decoder trained on its frozen codes). Both
    decoders see only the b-bit messages. Returns forecasts on dev, regrets, gain with a paired CI, controls."""
    import torch
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = make_cenv(env_name)
    tr = train_batch if train_batch is not None else env.sample("train", n_train)
    dv = dev_batch if dev_batch is not None else env.sample("dev", n_dev)
    l0 = float(dv.public["prior_logit"])
    Xtr, Xdv = torch.tensor(agent_tokens(tr)), torch.tensor(agent_tokens(dv))
    Ytr = torch.tensor(np.asarray(tr.truth, np.float32))
    F = Xtr.shape[-1]
    mean, sd = Xtr.reshape(-1, F).mean(0), Xtr.reshape(-1, F).std(0).clamp_min(1e-3)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    out = {"env": env_name, "b": b, "n_train": len(tr), "steps": steps, "batch": batch, "beta": beta, "log": []}

    def make():
        net = _learned_net(F, b).to(device)
        net.x_mean.copy_(mean.to(device))
        net.x_sd.copy_(sd.to(device))
        return net

    def rate(net, x):
        if b is None:
            return torch.zeros((), device=x.device)
        p = torch.softmax(net.logits(x), -1)
        return (math.log(2 ** b) + (p * torch.log(p.clamp_min(1e-12))).sum(-1)).mean()

    def predict(net):
        net.eval()
        res = []
        with torch.no_grad():
            for s in range(0, len(Xdv), 1024):
                x = Xdv[s:s + 1024].to(device)
                res.append(net.decode(net.message(x, hard_eval=True)).cpu().numpy())
        net.train()
        return {"logit": l0 + np.concatenate(res).astype(float)}

    # aggregation-aware: end to end through the decoder
    aware = make()
    out["control_untrained"] = summarize(predict(aware), dv, n_boot=n_boot)["kl"]
    _train_loop(aware, list(aware.parameters()),
                lambda x, y, tau: bce(aware.decode(aware.message(x, tau)) + l0, y) + beta * rate(aware, x),
                Xtr, Ytr, steps, batch, lr, device, out["log"], "aware")
    fa = predict(aware)

    # self-summary: encoder + per-agent own head on own-evidence Y prediction, then freeze; decoder on frozen codes
    selfn = make()
    enc_params = list(selfn.enc.parameters()) + list(selfn.own_head.parameters())
    _train_loop(selfn, enc_params,
                lambda x, y, tau: bce(selfn.own_logit(selfn.message(x, tau)) + l0, y[:, None].expand(-1, x.shape[1])),
                Xtr, Ytr, steps, batch, lr, device, out["log"], "self_stage1")
    for p in enc_params:
        p.requires_grad_(False)
    dec_params = [p for n_, p in selfn.named_parameters() if not (n_.startswith("enc") or n_.startswith("own_head"))]
    _train_loop(selfn, dec_params,
                lambda x, y, tau: bce(selfn.decode(selfn.message(x, hard_eval=True)) + l0, y),
                Xtr, Ytr, steps, batch, lr, device, out["log"], "self_stage2")
    fs = predict(selfn)

    # aggregation-aware, warm start: the trained self-summary system, then encoder + decoder fine-tuned end to end
    # (separates "end-to-end training is hard at small b" from "aggregation-aware messages carry more")
    import copy
    ws = copy.deepcopy(selfn)
    for p in ws.parameters():
        p.requires_grad_(True)
    ws_params = [p for n_, p in ws.named_parameters() if not n_.startswith("own_head")]
    _train_loop(ws, ws_params, lambda x, y, tau: bce(ws.decode(ws.message(x, tau)) + l0, y) + beta * rate(ws, x),
                Xtr, Ytr, steps, batch, lr / 3, device, out["log"], "aware_ws", tau0=0.5)
    fw = predict(ws)

    if b is not None:   # the eval channel: count distinct codes per agent position, must be <= 2^b
        with torch.no_grad():
            codes = aware.logits(Xdv[:2000].to(device)).argmax(-1).cpu().numpy()
        Channel(b).send(codes.astype(np.int64))
        out["distinct_codes_aware"] = int(len(np.unique(codes)))
    out["regret_aware"] = summarize(fa, dv, n_boot=n_boot)["kl"]
    out["regret_self_summary"] = summarize(fs, dv, n_boot=n_boot)["kl"]
    out["gain"] = gain(fs, fa, dv, n_boot=n_boot)
    out["regret_aware_ws"] = summarize(fw, dv, n_boot=n_boot)["kl"]
    out["gain_ws"] = gain(fs, fw, dv, n_boot=n_boot)
    best = fa if out["regret_aware"]["mean"] <= out["regret_aware_ws"]["mean"] else fw   # chosen on dev: diagnostic
    out["gain_best_aware"] = gain(fs, best, dv, n_boot=n_boot)
    out["forecasts"] = {"aware": fa["logit"].tolist(), "aware_ws": fw["logit"].tolist(),
                        "self_summary": fs["logit"].tolist()}
    return out


def run_learned(envs=("E2", "E3-sybil", "Z9-p2"), bits=(1, 2, 4, 8, None), n_train=100_000, steps=4000, out_dir=None,
                **kw):
    results = []
    for e in envs:
        env = make_cenv(e)
        tr, dv = env.sample("train", n_train), env.sample("dev", 2000)
        for b in bits:
            r = train_learned_pair(e, b, steps=steps, train_batch=tr, dev_batch=dv, **kw)
            print(f"[learned {e}] b={b}: aware {r['regret_aware']['mean']:.4f} self {r['regret_self_summary']['mean']:.4f}"
                  f" gain {r['gain']['mean']:.4f} [{r['gain']['ci_low']:.4f}, {r['gain']['ci_high']:.4f}]"
                  f" aware_ws {r['regret_aware_ws']['mean']:.4f} gain_ws {r['gain_ws']['mean']:.4f}"
                  f" [{r['gain_ws']['ci_low']:.4f}, {r['gain_ws']['ci_high']:.4f}]"
                  f" untrained {r['control_untrained']['mean']:.4f}", file=sys.stderr, flush=True)
            results.append(r)
            if out_dir:
                import os
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "learned.json"), "w") as f:
                    json.dump({"schema": "aggbench.compression-learned/v1", "results": results}, f,
                              default=lambda x: None if x is None else float(x))
    return results


# ============================================================================= learned arms v2: trainable channels

GAUSS_SIGMA2 = 1.0 / 3.0     # peak power 1 (tanh), noise 1/3: capacity 1/2 log2(1 + 3) = 1 bit per dimension
ESCAPE_FRAC = 0.9           # steps to escape: first eval with validation KL < ESCAPE_FRAC x the prior KL

Z9_CURRICULUM = (          # k = 2 -> 3 -> 4 parity bits; n_vars = 6 throughout so the token width is fixed
    {"n_bits": 2, "n_distractor_bits": 4, "n_distractor_agents": 2},     # 8 readers -> 4 agents (p = 2)
    {"n_bits": 3, "n_distractor_bits": 3, "n_distractor_agents": 1},     # 10 readers -> 5 agents
    {},                                                                  # Z9 default, k = 4: 14 readers -> 7 agents
)


def gauss_bits(sigma2=GAUSS_SIGMA2, power=1.0):
    """Capacity of the additive Gaussian channel per dimension: 1/2 log2(1 + P / sigma^2)."""
    return 0.5 * math.log2(1 + power / sigma2)


def _net_v2(F, b, channel, n_aux=0, width=64, enc_hidden=128, layers=2, heads=4):
    """Encoder (own token only) -> channel -> set decoder (no positions), plus an own-message head (self-summary)
    and optional auxiliary heads (intermediate parities). channel "gauss": b dimensions, tanh (power <= 1) plus
    N(0, 1/3) noise in training = 1 bit per dimension by capacity; eval "quant" sends sign(.) = exactly b bits.
    channel "gumbel": a code of 2^b symbols, straight-through Gumbel in training, argmax at eval."""
    import torch
    nn = torch.nn
    V = F if channel == "none" else (b if channel == "gauss" else 2 ** b)
    sigma = math.sqrt(GAUSS_SIGMA2)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("x_mean", torch.zeros(F))
            self.register_buffer("x_sd", torch.ones(F))
            self.enc = nn.Sequential(nn.Linear(F, enc_hidden), nn.GELU(), nn.Linear(enc_hidden, enc_hidden), nn.GELU(),
                                     nn.Linear(enc_hidden, V))
            self.own_head = nn.Sequential(nn.Linear(V, 64), nn.GELU(), nn.Linear(64, 1))
            self.emb = nn.Linear(V, width)
            self.cls = nn.Parameter(0.02 * torch.randn(1, 1, width))
            layer = nn.TransformerEncoderLayer(width, heads, 4 * width, 0.0, batch_first=True, norm_first=True)
            self.dec = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(width)
            self.out = nn.Linear(width, 1)
            self.aux = nn.Linear(width, n_aux) if n_aux else None

        def message(self, x, mode="train", tau=1.0):
            if channel == "none":                                 # uncompressed anchor: the raw token itself
                return (x - self.x_mean) / self.x_sd
            u = self.enc((x - self.x_mean) / self.x_sd)
            if channel == "gauss":
                s = torch.tanh(u)
                if mode == "quant":
                    return torch.where(s >= 0, 1.0, -1.0)
                return s + sigma * torch.randn_like(s)            # "train" and "noisy" eval
            if mode == "train":
                return gumbel_code(u, tau=tau)
            return torch.nn.functional.one_hot(u.argmax(-1), V).float()

        def symbols(self, x):
            """The eval-channel symbols, (..., ) integers in [0, 2^b)."""
            u = self.enc((x - self.x_mean) / self.x_sd)
            if channel == "gauss":
                bitsv = (u >= 0).long()
                return (bitsv * (2 ** torch.arange(V, device=u.device))).sum(-1)
            return u.argmax(-1)

        def decode(self, m):
            h = self.emb(m)
            h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1)
            z = self.norm(self.dec(h)[:, 0])
            return self.out(z)[:, 0], (self.aux(z) if self.aux is not None else None)

        def own_logit(self, m):
            return self.own_head(m)[..., 0]

    return Net()


def _token_pool(env, split, n, start=0, k_max=None):
    """(tokens, truth, aux targets (n, k_max - 1) prefix parities of the parity bits, aux mask) for one env."""
    import torch
    batch = env.sample(split, n, start=start)
    X = torch.tensor(agent_tokens(batch))
    Y = torch.tensor(np.asarray(batch.truth, np.float32))
    if not k_max:
        return X, Y, None, None, batch
    k = env.base.config.n_bits
    w = np.asarray(batch.oracle["w"])
    bits = (w[:, None] >> np.arange(k)) & 1
    pref = np.cumsum(bits, 1) % 2                          # parity of the first j + 1 bits
    A = np.zeros((n, k_max - 1), np.float32)
    A[:, :k - 1] = pref[:, :k - 1]                         # the last prefix parity is Y itself: not auxiliary
    M = np.zeros(k_max - 1, np.float32)
    M[:k - 1] = 1
    return X, Y, torch.tensor(A), torch.tensor(M), batch


def _kl_logits(q, b):
    pb = 1 / (1 + np.exp(-b))
    return float(np.mean(pb * (logsig(b) - logsig(q)) + (1 - pb) * (logsig(-b) - logsig(-q))))


def train_learned_v2(env_name, b, arm, channel, seed, pools, val, dev, steps=4000, batch=512, lr=1e-3,
                     aux_weight=0.0, aux_warmup=0, curriculum=None, eval_every=200, device=None, n_boot=2000):
    """One learned arm. arm "self": encoder + own head on own-evidence Y prediction (stage 1), frozen, then a set
    decoder on its messages (stage 2), `steps` each. arm "aware": end to end through the decoder (D-VIB-style; a
    learned decoder that does not know the structure). curriculum: a list of (pool key, steps) stages for the
    aware arm (Z9: k = 2 -> 3 -> 4), else `steps` on the final pool. aux_weight: intermediate-parity loss weight,
    linearly annealed to 0 at step aux_warmup (Z9; labels from the latent bits, training only). Train noisy, test
    quantised: regret_quant uses the hard b-bit channel, regret_noisy (gauss) the noisy channel. steps_to_escape =
    first eval step with validation KL below ESCAPE_FRAC (0.9) x the prior's (leaving the plateau)."""
    import torch
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xf = pools["final"][0]
    F = Xf.shape[-1]
    import torch as _t                    # B6 (review 2026-10-22): statistics over every stage's pool; a feature that is
    Xs = _t.cat([v[0].reshape(-1, F) for v in pools.values()])       # constant in all of them is zeroed, not blown up
    mean, sd = Xs.mean(0), Xs.std(0)
    sd = _t.where(sd < 1e-6, _t.full_like(sd, 1e12), sd.clamp_min(1e-2))
    n_aux = 0 if pools["final"][2] is None else pools["final"][2].shape[1]
    net = _net_v2(F, b, channel, n_aux).to(device)
    net.x_mean.copy_(mean.to(device))
    net.x_sd.copy_(sd.to(device))
    l0 = float(dev[4].public["prior_logit"])
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    vX, vb = val[0], np.asarray(val[4].bayes["logit"], float)
    prior_kl = _kl_logits(np.full(len(vb), l0), vb)
    curve, rec = [], {"env": env_name, "b": b, "arm": arm, "channel": channel, "seed": seed, "steps": steps,
                      "aux_weight": aux_weight, "aux_warmup": aux_warmup, "curriculum": curriculum,
                      "channel_bits_per_agent": b}

    def predict(X, mode):
        net.eval()
        out = []
        g = torch.random.get_rng_state()     # B10 (review 2026-10-22): save and restore the CUDA states too
        gc = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(12345)
        with torch.no_grad():
            for s in range(0, len(X), 1024):
                out.append(net.decode(net.message(X[s:s + 1024].to(device), mode))[0].cpu().numpy())
        torch.random.set_rng_state(g)
        if gc is not None:
            torch.cuda.set_rng_state_all(gc)
        net.train()
        return l0 + np.concatenate(out).astype(float)

    gstep = [0]

    def loop(params, stages, loss_fn, tag):
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        total = sum(s for _, s in stages)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        t = 0
        for key, n_steps in stages:
            X, Y, A, M, _ = pools[key]
            for _ in range(n_steps):
                idx = torch.randint(0, len(X), (batch,), generator=gen)
                x, y = X[idx].to(device), Y[idx].to(device)
                a = None if A is None else (A[idx].to(device), M.to(device))
                tau = max(0.5, 1.0 - t / max(total, 1))
                loss = loss_fn(x, y, a, tau, t)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                if t % eval_every == 0 or t == total - 1:
                    kl = _kl_logits(predict(vX, "quant"), vb)
                    curve.append({"phase": tag, "step": t, "stage": key, "val_kl": kl, "loss": loss.item()})
                t += 1
                gstep[0] += 1

    rec["control_untrained"] = summarize({"logit": predict(dev[0], "quant")}, dev[4], n_boot=n_boot)["kl"]

    def aux_term(z_aux, a, t):
        if z_aux is None or a is None or not aux_weight or t >= aux_warmup:
            return 0.0
        A_, M_ = a
        w = aux_weight * (1 - t / aux_warmup)
        return w * (bce(z_aux, A_, reduction="none") * M_).sum(-1).mean() / M_.sum().clamp_min(1)

    if arm == "aware":
        stages = curriculum or [("final", steps)]

        def loss_aware(x, y, a, tau, t):
            logit, z_aux = net.decode(net.message(x, "train", tau))
            return bce(logit + l0, y) + aux_term(z_aux, a, t)
        loop(list(net.parameters()), stages, loss_aware, "aware")
    else:
        # "self": stage 1 trains the encoder on own-evidence Y prediction. "random_enc" (control): the encoder stays
        # at its random initialization; only the decoder is trained (stage 2), on Y labels.
        enc_params = list(net.enc.parameters()) + list(net.own_head.parameters())
        if arm == "self":
            loop(enc_params, [("final", steps)],
                 lambda x, y, a, tau, t: bce(net.own_logit(net.message(x, "train", tau)) + l0,
                                             y[:, None].expand(-1, x.shape[1])), "self_stage1")
        for p in enc_params:
            p.requires_grad_(False)
        dec_params = [p for n_, p in net.named_parameters() if not (n_.startswith("enc") or n_.startswith("own_head"))]

        def loss_dec(x, y, a, tau, t):
            with torch.no_grad():
                m = net.message(x, "train" if channel == "gauss" else "quant")
            logit, z_aux = net.decode(m)
            return bce(logit + l0, y)
        loop(dec_params, [("final", steps)], loss_dec, "self_stage2")

    fq = predict(dev[0], "quant")
    rec["regret_quant"] = summarize({"logit": fq}, dev[4], n_boot=n_boot)["kl"]
    if channel == "gauss":
        rec["regret_noisy"] = summarize({"logit": predict(dev[0], "noisy")}, dev[4], n_boot=n_boot)["kl"]
    if channel != "none":
        with torch.no_grad():
            sym = net.symbols(dev[0][:2000].to(device)).cpu().numpy().astype(np.int64)
        Channel(b).send(sym)                                # the eval channel carries b bits, exactly
        rec["distinct_symbols"] = int(len(np.unique(sym)))
    final_phase = "aware" if arm == "aware" else "self_stage2"
    esc = [c["step"] for c in curve if c["phase"] == final_phase and c["val_kl"] < ESCAPE_FRAC * prior_kl]
    rec["steps_to_escape"] = esc[0] if esc else None
    rec["escape_frac"] = ESCAPE_FRAC
    rec["val_prior_kl"] = prior_kl
    rec["curve"] = curve
    rec["forecast_quant"] = fq.tolist()
    return rec


def run_learned_v2(envs=("E2", "E3-sybil", "Z9-p2"), bits=(1, 2, 4), seeds=(0, 1, 2), channels=("gauss", "gumbel"),
                   steps=4000, z9_steps=8000, n_train=100_000, out_dir=None):
    """Self-summary vs end-to-end (aggregation-aware) at b bits, learned decoders, 3 seeds; Z9-p2 also with the
    intermediate-parity auxiliary loss and the k curriculum. Gains are paired over the same dev episodes."""
    import os
    results, gains = [], []

    def dump():
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "learned_v2.json"), "w") as f:
                json.dump({"schema": "aggbench.compression-learned/v2", "results": results, "gains": gains}, f,
                          default=lambda x: None if x is None else float(x))

    for e in envs:
        z9 = e == "Z9-p2"
        k_max = 4 if z9 else None
        env = make_cenv(e)
        pools = {"final": _token_pool(env, "train", n_train, k_max=k_max)}
        if z9:
            for j, ov in enumerate(Z9_CURRICULUM[:-1]):
                pools[f"k{j + 2}"] = _token_pool(PiecesPerAgent("Z9", 2, **ov), "train", 30_000, k_max=k_max)
        val = _token_pool(env, "dev", 1000, start=50_000, k_max=k_max)
        dev = _token_pool(env, "dev", 2000, k_max=k_max)
        S = z9_steps if z9 else steps
        variants = [("aware", {})]
        if z9:
            third = S // 4
            variants += [("aware_aux", {"aux_weight": 1.0, "aux_warmup": S // 2}),
                         ("aware_curr_aux", {"aux_weight": 1.0, "aux_warmup": S // 2,
                                             "curriculum": [("k2", third), ("k3", third), ("final", S - 2 * third)]})]
        for b in bits:
            for ch in channels:
                for seed in seeds:
                    base = train_learned_v2(e, b, "self", ch, seed, pools, val, dev, steps=S)
                    base["variant"] = "self"
                    results.append(base)
                    for name, kw in variants:
                        r = train_learned_v2(e, b, "aware", ch, seed, pools, val, dev, steps=S, **kw)
                        r["variant"] = name
                        results.append(r)
                        g = gain({"logit": np.array(base["forecast_quant"])}, {"logit": np.array(r["forecast_quant"])},
                                 dev[4])
                        gains.append({"env": e, "b": b, "channel": ch, "seed": seed, "variant": name, "gain": g,
                                      "self": base["regret_quant"]["mean"], "aware": r["regret_quant"]["mean"],
                                      "escape_self": base["steps_to_escape"], "escape_aware": r["steps_to_escape"]})
                        print(f"[v2 {e}] b={b} {ch} seed={seed} {name}: self {base['regret_quant']['mean']:.4f} "
                              f"aware {r['regret_quant']['mean']:.4f} gain {g['mean']:+.4f} [{g['ci_low']:+.4f}, "
                              f"{g['ci_high']:+.4f}] escape self/aware {base['steps_to_escape']}/{r['steps_to_escape']}"
                              + (f" noisy aware {r['regret_noisy']['mean']:.4f}" if ch == "gauss" else ""),
                              file=sys.stderr, flush=True)
                    dump()
    for r in results:
        r.pop("forecast_quant", None)
    dump()
    return results, gains


def run_z9_controls(bits=(1, 2, 4), seeds=(0, 1, 2), channels=("gauss", "gumbel"), steps=8000, n_train=100_000,
                    out_dir=None, env_name="Z9-p2"):
    """Controls for the Z9-p2 gains (gate 2026-10-21): (a) random frozen encoder + trained decoder at b bits (a
    non-aware code that is not collapsed: does the aware gain beat any fixed local code?), (b) the uncompressed
    full-context anchor: the same set decoder reading every agent's raw token (channel "none"), Y labels only.
    Both label-only, same data, steps and seeds as the v2 arms; the aware arm is rerun alongside for paired gains."""
    import os
    env = make_cenv(env_name)
    pools = {"final": _token_pool(env, "train", n_train, k_max=4)}
    val = _token_pool(env, "dev", 1000, start=50_000, k_max=4)
    dev = _token_pool(env, "dev", 2000, k_max=4)
    results, gains = [], []

    def dump():
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "z9_controls.json"), "w") as f:
                json.dump({"schema": "aggbench.compression-z9-controls/v1", "results": results, "gains": gains}, f,
                          default=lambda x: None if x is None else float(x))

    for seed in seeds:
        r = train_learned_v2(env_name, 0, "aware", "none", seed, pools, val, dev, steps=steps)
        r["variant"] = "full_context_anchor"
        results.append(r)
        print(f"[z9c] full-context anchor seed={seed}: {r['regret_quant']['mean']:.4f}", file=sys.stderr, flush=True)
        dump()
    for b in bits:
        for ch in channels:
            for seed in seeds:
                rnd = train_learned_v2(env_name, b, "random_enc", ch, seed, pools, val, dev, steps=steps)
                rnd["variant"] = "random_enc"
                aw = train_learned_v2(env_name, b, "aware", ch, seed, pools, val, dev, steps=steps)
                aw["variant"] = "aware"
                results += [rnd, aw]
                g = gain({"logit": np.array(rnd["forecast_quant"])}, {"logit": np.array(aw["forecast_quant"])}, dev[4])
                gains.append({"env": env_name, "b": b, "channel": ch, "seed": seed, "variant": "aware_vs_random_enc",
                              "gain": g, "random_enc": rnd["regret_quant"]["mean"], "aware": aw["regret_quant"]["mean"]})
                print(f"[z9c] b={b} {ch} seed={seed}: random_enc {rnd['regret_quant']['mean']:.4f} aware "
                      f"{aw['regret_quant']['mean']:.4f} gain {g['mean']:+.4f} [{g['ci_low']:+.4f}, {g['ci_high']:+.4f}]",
                      file=sys.stderr, flush=True)
                dump()
    for r in results:
        r.pop("forecast_quant", None)
    dump()
    return results, gains


# ============================================================================= FAIR closed forms (no episode structure)
#
# The closed-form decoders above (llr_quantizer, queried, top_w, k_of_n, raw_code) are ORACLE-STRUCTURE decoders:
# they read each episode's structure (which pieces / bits each agent reads, strengths, overlap) out of band from
# batch.oracle. A learned decoder sees only the messages. The fair closed forms below decode from the messages, the
# public constants and the environment's generative PRIOR over structures (sampled from fresh simulator seeds, never
# from the scored episodes); an agent that wants the decoder to know its structure must spend bits on it.

class NoOracle(dict):
    """Stands in for batch.oracle in fair paths: any access raises (tests use it to prove fair decoders are blind)."""

    def _deny(self, *a, **k):
        raise AssertionError("fair decoder touched batch.oracle")

    __getitem__ = __contains__ = get = keys = items = values = __iter__ = _deny


def blind(batch):
    """A copy of `batch` whose oracle is NoOracle (truth and bayes stay: they are only used for scoring)."""
    return Batch(batch.env, batch.config, batch.kind, batch.seeds, batch.truth, batch.obs, batch.local, batch.reports,
                 batch.keys, batch.bayes, NoOracle(), batch.public)


# ----------------------------------------------------------------------------- Z9-p2: send which bit + the reading

FAIR_Z9_SALT = 0x7A9


class FairZ9Code:
    """Fair code for grouped k-parity (Z9-p2). Each reading becomes a token: 0 = "distractor bit" (useless), else
    1 + ((parity-bit index j) * 2 + reading o) * 2^q + accuracy cell (q bits, uniform cells over the accuracy prior).
    kind "pair": the unordered pair of the agent's two tokens (lossless for the task up to accuracy quantization);
    kind "one": one token (the first parity reading). The symbol is the pair / token index mod 2^b, so below the
    lossless size symbols merge and the decoder accounts for it exactly. Bits to be lossless: ceil(log2 #symbols)."""

    def __init__(self, env, kind="pair", q=0, n_prior=50_000):
        base = env.base
        c = base.config
        self.k, self.nv, self.p = c.n_bits, base.n_vars(), env.p
        self.kind, self.q = kind, q
        self.acc_lo, self.acc_hi = c.accuracy_range
        self.T = 1 + 2 * self.k * 2 ** q
        self.n_sym = self.T * (self.T + 1) // 2 if kind == "pair" else self.T
        # the per-agent prior over (var_1, var_2): agents are exchangeable; tallied from fresh structures
        rng = np.random.default_rng([FAIR_Z9_SALT, 1])
        cnt = np.zeros((self.nv, self.nv))
        for _ in range(n_prior // base.config.n_agents * self.p + 1):
            v = base.structure(rng)["var"].reshape(-1, self.p)
            np.add.at(cnt, (v[:, 0], v[:, 1]), 1)
        self.var_prior = cnt / cnt.sum()

    def lossless_bits(self):
        return int(math.ceil(math.log2(self.n_sym)))

    def _acc_cell(self, acc):
        if self.q == 0:
            return np.zeros(np.shape(acc), np.int64)
        u = (np.asarray(acc) - self.acc_lo) / max(self.acc_hi - self.acc_lo, 1e-12)
        return np.clip((u * 2 ** self.q).astype(np.int64), 0, 2 ** self.q - 1)

    def _cell_acc(self, cell):
        """Mean accuracy in a cell (the decoder's likelihood for a reading with that cell)."""
        w = (self.acc_hi - self.acc_lo) / 2 ** self.q
        return self.acc_lo + (np.asarray(cell) + 0.5) * w

    def token(self, var, o, cell):
        par = var < self.k
        return np.where(par, 1 + ((np.minimum(var, self.k - 1) * 2 + o) * 2 ** self.q + cell), 0)

    def symbol(self, t1, t2):
        if self.kind == "one":
            t = np.where(t1 > 0, t1, t2)
            return t
        a, b = np.minimum(t1, t2), np.maximum(t1, t2)
        return a * self.T - a * (a - 1) // 2 + (b - a)

    def encode(self, obs, own, b):
        """Agent-local: obs (n, N, 2) readings, own (n, N, 2 * (nv + 1)) one-hot var + accuracy per reading."""
        own = own.reshape(*own.shape[:2], self.p, self.nv + 1)
        var = own[..., :self.nv].argmax(-1)
        cell = self._acc_cell(own[..., self.nv])
        o = np.rint(obs).astype(np.int64)
        t = self.token(var, o, cell)
        sym = self.symbol(t[..., 0], t[..., 1])
        return Channel(b).send((sym % 2 ** b).astype(np.int64)) if b is not None else sym

    def table(self, b):
        """log P(message | w) for every message and world state w, summing over the agent's local configurations
        (vars from the per-agent prior, readings, accuracy cells with their mean accuracy)."""
        nv, k = self.nv, self.k
        W = 2 ** nv
        bits = (np.arange(W)[:, None] >> np.arange(nv)) & 1           # (W, nv)
        L = self.n_sym if b is None else 2 ** b
        T = np.zeros((L, W))
        cells = np.arange(2 ** self.q)
        pc = 1.0 / len(cells)
        for v1 in range(nv):
            for v2 in range(nv):
                pv = self.var_prior[v1, v2]
                if pv == 0:
                    continue
                for c1 in cells:
                    for c2 in cells:
                        a1, a2 = self._cell_acc(c1), self._cell_acc(c2)
                        for o1 in (0, 1):
                            for o2 in (0, 1):
                                lik = (np.where(bits[:, v1] == o1, a1, 1 - a1) * np.where(bits[:, v2] == o2, a2, 1 - a2))
                                t1 = self.token(np.array(v1), np.array(o1), np.array(c1))
                                t2 = self.token(np.array(v2), np.array(o2), np.array(c2))
                                m = int(self.symbol(t1, t2))
                                m = m % L
                                T[m] += pv * pc * pc * lik
        with np.errstate(divide="ignore"):
            return np.log(T), bits

    def forecast(self, batch, b, table=None):
        """Fair forecast: exact posterior of Y = parity of the first k bits given the messages, with agents treated
        as independent draws from the per-agent structure prior (exact when the message reveals the vars, since the
        prior factor then cancels). Reads obs, keys['full'] (each agent's OWN model) and public only."""
        logT, bits = table if table is not None else self.table(b)
        n, N = batch.obs.shape[:2]
        sym = self.encode(batch.obs, batch.keys["full"], b)
        if b is not None:
            sym = sym % 2 ** b
        y = bits[:, :self.k].sum(1) % 2
        lp = logT[sym].sum(1)                                          # (n, W), uniform prior over w
        logit = np.logaddexp.reduce(np.where(y == 1, lp, -np.inf), -1) - np.logaddexp.reduce(
            np.where(y == 0, lp, -np.inf), -1)
        return {"logit": np.clip(np.nan_to_num(logit), -CLIP, CLIP)}, sym


def fair_z9_curve(env, train, dev, bits=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), qs=(0, 1, 2), n_boot=2000):
    """For each b: every fair code (pair / one, q accuracy bits) at b, the best chosen on train; returns rows."""
    codes = [FairZ9Code(env, kind, q) for kind in ("pair", "one") for q in qs]
    rows = []
    for b in bits:
        best = None
        for code in codes:
            tab = code.table(b)
            k_tr = _kl(code.forecast(blind(train), b, tab)[0], train)
            if best is None or k_tr < best[0]:
                best = (k_tr, code, tab)
        fc, sym = best[1].forecast(blind(dev), b, best[2])
        rows.append({"b": b, "code": best[1].kind, "acc_bits": best[1].q,
                     "lossless_bits_of_code": best[1].lossless_bits(),
                     "regret": summarize(fc, dev, n_boot=n_boot)["kl"],
                     "distinct_symbols": int(len(np.unique(sym)))})
    return rows


# ----------------------------------------------------------------------------- E2/E3: structure-prior decoder

def sample_structure_models(env, n, salt=0x5A3):
    """(c, h, S) Gaussian report models of n structures drawn from the environment's prior (fresh seeds from a
    reserved range, never scored)."""
    seeds = 80_000_000 + np.arange(n) + salt
    eps = [env.episode(int(s)) for s in seeds]
    o = {k[7:]: np.stack([e[k] for e in eps]) for k in eps[0] if k.startswith("oracle.")}
    local = {"logit": np.stack([e["local.logit"] for e in eps])}
    stub = Batch(env.name, asdict(env.config), env.kind, seeds, None, None, local, local, {}, {}, o, env.public())
    return gaussian_report_model(stub)


@dataclass
class FairLLR:
    """Fair self-summary for E2/E3/Z8/E4: shared thresholds l0 + s * pooled * Phi^-1(j / 2^b) (public constants fit
    on train), and a decoder that does not know the episode's structure: P(cells | Y) = E_structure[P(cells | Y,
    structure)] over n_struct structures from the prior (Genz box probabilities, M QMC points)."""
    bits: int
    s: float = 1.0
    pooled: float = 1.0

    def thresholds(self, l0):
        return l0 + self.s * self.pooled * companded_thresholds(self.bits)

    def forecast(self, batch, models, M=128, seed=0, chunk=4096):
        l0 = float(batch.public["prior_logit"])
        obs, _, public, own = batch.agent_inputs("none", alignment="aligned", own_model=True)
        r, _ = local_statistic(obs, own, env_family_from_config(batch.config), batch.config, public)
        th = self.thresholds(l0)
        sym = Channel(self.bits).send(quantize(r, th))
        lo, hi = cell_bounds(sym, th)
        c, h, S = models
        Ms, N = c.shape
        L = _chol(S)
        u = _qmc(M, N, seed)
        n = len(r)
        lp = np.empty((2, n, Ms))
        per = max(1, chunk // Ms)
        for s0 in range(0, n, per):
            e = np.arange(s0, min(n, s0 + per))
            E = len(e)
            LO, HI = np.repeat(lo[e], Ms, 0), np.repeat(hi[e], Ms, 0)
            LL = np.tile(L, (E, 1, 1))
            for yi, sign in ((0, -1.0), (1, 1.0)):
                mean = np.tile(c + sign * h, (E, 1))
                lp[yi, e] = box_logprob(mean, LL, LO, HI, u).reshape(E, Ms)
        with np.errstate(invalid="ignore"):
            l1 = np.logaddexp.reduce(lp[1], -1)
            l0_ = np.logaddexp.reduce(lp[0], -1)
            d = np.where(np.isfinite(l1) | np.isfinite(l0_), l1 - l0_, 0.0)
        return {"logit": l0 + np.clip(np.nan_to_num(d), -CLIP, CLIP)}, sym


def env_family_from_config(c):
    if "n_rollouts" in c:
        return "rollout"
    if "read_noise" in c and "strength_levels" in c:
        return "privateread"
    if "n_sybil" in c:
        return "pieces"
    raise ValueError("no fair LLR adapter")


def fair_llr_curve(env, train, dev, bits=(1, 2, 4, 8), n_struct=256, M=128, n_fit=300, n_boot=2000,
                   grid=(0.5, 0.75, 1.0, 1.5)):
    models = sample_structure_models(env, n_struct)
    r_tr = agent_statistic(train)[0]
    l0 = float(train.public["prior_logit"])
    pooled = float(np.std(r_tr - l0)) or 1.0
    sub = _subset(train, n_fit)
    rows = []
    for b in bits:
        best = None
        for s in (grid if b > 1 else (1.0,)):
            q = FairLLR(b, s, pooled)
            k = _kl(q.forecast(blind(sub), models, M=M)[0], sub)
            if best is None or k < best[0]:
                best = (k, q)
        fc, sym = best[1].forecast(blind(dev), models, M=M)
        rows.append({"b": b, "s": best[1].s, "n_struct": n_struct, "qmc_points": M,
                     "regret": summarize(fc, dev, n_boot=n_boot)["kl"],
                     "distinct_symbols_max": int(max(len(np.unique(sym[:, i])) for i in range(sym.shape[1])))})
        print(f"[fair {env.name}] b={b}: regret {rows[-1]['regret']['mean']:.4f}", file=sys.stderr, flush=True)
    return rows


def run_fair(envs=("E2", "E3-sybil", "Z9-p2"), n_train=2000, n_dev=2000, out=None):
    res = {"schema": "aggbench.compression-fair/v1", "decoder_access": "fair: messages + public + structure prior",
           "results": []}
    try:
        from .results import harness_info
        res["harness"] = harness_info()
    except Exception:   # noqa: BLE001
        pass
    for e in envs:
        env = make_cenv(e)
        tr, dv = env.sample("train", n_train), env.sample("dev", n_dev)
        rows = fair_z9_curve(env, tr, dv) if e.startswith("Z9") else fair_llr_curve(env, tr, dv)
        res["results"].append({"env": e, "rows": rows})
        if e.startswith("Z9"):
            for r in rows:
                print(f"[fair {e}] b={r['b']}: {r['code']} q={r['acc_bits']} regret {r['regret']['mean']:.4f}",
                      file=sys.stderr, flush=True)
        if out:
            with open(out, "w") as f:
                json.dump(res, f, indent=0, default=lambda x: None if x is None else float(x))
    return res


def run_anchor(envs=("E2", "E3-sybil"), seeds=(0, 1, 2), steps=4000, n_train=100_000, out_dir=None):
    """The uncompressed full-context anchor matched to the v2 learned arms: the same set decoder reading every
    agent's raw token (channel "none"), the same 100k pool, Y labels and `steps` (E2/E3: 4000, Z9-p2: 8000 via
    run_z9_controls). gap(b) = regret(learned arm, b) - regret(anchor) is the primary compression column."""
    import os
    results = []
    for e in envs:
        env = make_cenv(e)
        k_max = 4 if e.startswith("Z9") else None
        pools = {"final": _token_pool(env, "train", n_train, k_max=k_max)}
        val = _token_pool(env, "dev", 1000, start=50_000, k_max=k_max)
        dev = _token_pool(env, "dev", 2000, k_max=k_max)
        for seed in seeds:
            r = train_learned_v2(e, 0, "aware", "none", seed, pools, val, dev, steps=steps)
            r["variant"] = "full_context_anchor"
            r.pop("forecast_quant", None)
            results.append(r)
            print(f"[anchor {e}] seed={seed}: {r['regret_quant']['mean']:.4f}", file=sys.stderr, flush=True)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "anchor.json"), "w") as f:
                    json.dump({"schema": "aggbench.compression-anchor/v1", "results": results}, f,
                              default=lambda x: None if x is None else float(x))
    return results


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--envs", nargs="+", default=["E2", "E3-sybil"])
    ap.add_argument("--bits", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-dev", type=int, default=2000)
    ap.add_argument("--qmc", type=int, default=1024)
    ap.add_argument("--n-sweep", nargs="*", type=int, default=None, help="also regret vs N at fixed total S (E2)")
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--learned", action="store_true", help="train the learned arms (torch; GPU) instead")
    ap.add_argument("--learned-v2", action="store_true", help="v2 learned arms: Gaussian/Gumbel channels, seeds, "
                    "Z9 auxiliary loss and curriculum (torch; GPU)")
    ap.add_argument("--anchor", action="store_true", help="uncompressed full-context anchor matched to the v2 arms")
    ap.add_argument("--fair", action="store_true", help="fair closed forms (no episode structure) on --envs")
    ap.add_argument("--z9-controls", action="store_true", help="Z9-p2 random-encoder control and full-context anchor")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--channels", nargs="+", default=["gauss", "gumbel"])
    ap.add_argument("--z9-steps", type=int, default=8000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-train-learned", type=int, default=100_000)
    ap.add_argument("--learned-bits", nargs="+", default=["1", "2", "4", "8", "inf"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.anchor:
        run_anchor(tuple(a.envs), tuple(a.seeds), a.steps, a.n_train_learned, out_dir=a.out_dir)
        return
    if a.fair:
        run_fair(tuple(a.envs), a.n_train, a.n_dev, out=a.out)
        return
    if a.z9_controls:
        run_z9_controls(tuple(int(x) for x in a.learned_bits), tuple(a.seeds), tuple(a.channels), a.z9_steps,
                        a.n_train_learned, out_dir=a.out_dir)
        return
    if a.learned_v2:
        run_learned_v2(tuple(a.envs), tuple(int(x) for x in a.learned_bits), tuple(a.seeds), tuple(a.channels),
                       a.steps, a.z9_steps, a.n_train_learned, out_dir=a.out_dir)
        return
    if a.learned:
        bits = tuple(None if x == "inf" else int(x) for x in a.learned_bits)
        run_learned(tuple(a.envs), bits, a.n_train_learned, a.steps, out_dir=a.out_dir)
        return
    out = {"schema": "aggbench.compression/v1", "results": []}
    try:
        from .results import harness_info
        out["harness"] = harness_info()
    except Exception:   # noqa: BLE001 - provenance is best effort here
        pass
    def dump():
        txt = json.dumps(out, indent=1, default=lambda x: None if x is None else float(x))
        if a.out:
            with open(a.out, "w") as f:
                f.write(txt)
        return txt

    for e in a.envs:
        out["results"].append(run_closed_form(e, tuple(a.bits), a.n_train, a.n_dev, a.qmc))
        dump()                                   # after every env, so a later failure keeps the finished ones
    if a.n_sweep:
        out["n_sweep"] = run_n_sweep("E2", tuple(a.n_sweep), a.S, a.n_train, a.n_dev, a.qmc)
    txt = dump()
    if not a.out:
        print(txt)


if __name__ == "__main__":
    main()
