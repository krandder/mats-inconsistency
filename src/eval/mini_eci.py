"""
Mini-Model Epoch Capability Index (`Mini-ECI`) — 1D Item Response Theory (IRT)
/ Rasch-2PL Latent Capability Fitter & Test-Time Compute Scaling Analyzer.

Combines:
  1. Single-agent deduction & belief calibration benchmarks:
     - `Sudoku` (`Sudoku-6x6`, `Sudoku-9x9-Extreme`)
     - `ShipHunt` (`Coupled-ShipHunt-6x6`, `Coupled-ShipHunt-8x8`)
     - `Equiv-SAT` (`Latent-Equivalence-Parity`)
     - `Minesweeper` (`Minesweeper-6x6`)
     measured via Belief KL (`kl`), Brier score (`brier`), and Witness rate (`witness_rate`).
  2. Multiplayer Elo ladders:
     - `Hex-5x5`, `Connect-4`, `Micro-Stratego`, `Galanis-Market`, `Proposal-Poker`
     measured via Bayesian BTD Elo ratings (`elo`).

into a unified, calibrated 1D `Mini-ECI` score anchored at `100.0` for the
`1-pass` `B=0` baseline across compute budgets `B in {0, 1, 2, 4, 8, 16, 32, 64, 128}`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize

from .elo_league import (
    COMPUTE_BUDGETS,
    DEFAULT_ANCHOR_ELO,
    ELO_SCALE,
    MULTIPLAYER_ENVS,
    AgentSpec,
    LeagueFitResult,
    normalize_env_name,
)

SINGLE_AGENT_BENCHMARKS: Tuple[str, ...] = (
    "Sudoku",
    "ShipHunt",
    "Equiv-SAT",
    "Minesweeper",
)

SINGLE_AGENT_METRICS: Tuple[str, ...] = (
    "kl",
    "brier",
    "witness_rate",
)

DEFAULT_ANCHOR_MINI_ECI: float = 100.0
DEFAULT_ECI_SCALE: float = 15.0  # Mini-ECI points per unit Rasch-2PL logit


def normalize_benchmark_name(task_name: str) -> str:
    """Normalize single-agent benchmark names to canonical suite tasks."""
    clean = task_name.strip()
    if clean.startswith("Sudoku"):
        return "Sudoku"
    if "ShipHunt" in clean:
        return "ShipHunt"
    if clean in ("Equiv-SAT", "Latent-Equivalence-Parity", "EquivSAT"):
        return "Equiv-SAT"
    if clean.startswith("Minesweeper") or clean == "Mastermind":
        return "Minesweeper"
    return normalize_env_name(clean)


@dataclass(frozen=True)
class SingleAgentBenchmarkRecord:
    """
    Single-agent benchmark evaluation record for one `(model_family, search_algorithm, B)`
    on a single-agent deduction / probabilistic inference task.
    """

    agent_id: str
    model_family: str
    search_algorithm: str
    compute_budget_B: int
    task_name: str  # e.g., 'Sudoku', 'ShipHunt', 'Equiv-SAT', 'Minesweeper'
    kl: float  # Belief KL(P* || mu_B) >= 0 (lower is better)
    brier: float  # Brier score in [0, 1] (lower is better)
    witness_rate: float  # Solved / valid witness rate in [0, 1] (higher is better)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kl < 0.0:
            raise ValueError(f"kl must be non-negative, got {self.kl}")
        if not (0.0 <= self.brier <= 2.0):
            raise ValueError(f"brier must be in [0, 2], got {self.brier}")
        if not (0.0 <= self.witness_rate <= 1.0):
            raise ValueError(f"witness_rate must be in [0, 1], got {self.witness_rate}")


@dataclass
class IRTItemParameter:
    """Fitted Rasch-2PL item response parameters for a single benchmark metric or Elo ladder."""

    item_id: str
    domain: str  # 'single_agent' or 'multiplayer_elo'
    task_name: str
    metric_name: str  # 'kl', 'brier', 'witness_rate', or 'elo'
    discrimination_a: float  # 2PL slope a_k > 0
    difficulty_b: float  # 2PL difficulty threshold b_k
    scale_tau: float  # Normalization constant used for raw -> (0, 1) response link
    r_squared_with_theta: float  # Variance explained by 1D latent capability

    def to_dict(self) -> Dict[str, Union[str, float]]:
        return asdict(self)


@dataclass
class AgentMiniECIScore:
    """Calibrated Mini-ECI score and component breakdown for one agent population member."""

    agent_id: str
    model_family: str
    search_algorithm: str
    compute_budget_B: int
    mini_eci: float
    mini_eci_ci_low: float
    mini_eci_ci_high: float
    mini_eci_std: float
    latent_theta: float
    single_agent_sub_eci: float
    multiplayer_elo_sub_eci: float
    mean_kl: float
    mean_brier: float
    mean_witness_rate: float
    mean_elo: float
    num_items_observed: int

    def to_dict(self) -> Dict[str, Union[str, int, float]]:
        return asdict(self)


@dataclass
class ComputeScalingCurveFit:
    """
    Parametric and non-parametric test-time compute scaling curve for a
    `(model_family, search_algorithm)` across B in {0, 1, 2, 4, 8, 16, 32, 64, 128}.
    """

    model_family: str
    search_algorithm: str
    budgets: List[int]
    mini_ecis: List[float]
    ci_lows: List[float]
    ci_highs: List[float]
    mean_kls: List[float]
    mean_briers: List[float]
    mean_witness_rates: List[float]
    mean_elos: List[float]
    # Parametric law: ECI_hat(B) = e0 + kappa * log2(1 + B / b0)
    e0_baseline: float
    kappa_per_doubling: float
    b0_knee: float
    predicted_mini_ecis: List[float]
    r_squared: float
    # Anytime monotonicity diagnostics
    eci_is_monotone: bool
    max_eci_drop: float
    kl_is_monotone: bool
    max_kl_rise: float
    total_eci_gain_B128: float
    relative_kl_reduction_B128: float


@dataclass
class MiniECIFitResult:
    """Complete output of the joint Rasch-2PL Mini-ECI capability fitter."""

    anchor_agent_id: str
    anchor_mini_eci: float
    eci_scale: float
    scores: Dict[str, AgentMiniECIScore]
    agent_order: List[str]  # Sorted descending by mini_eci
    items: Dict[str, IRTItemParameter]
    scaling_curves: Dict[str, ComputeScalingCurveFit]  # keyed by f"{family}::{algorithm}"
    explained_variance_r2: float
    leave_one_task_out_r: float
    bootstrap_replicates: int
    seed: int

    def leaderboard(self) -> List[AgentMiniECIScore]:
        return [self.scores[aid] for aid in self.agent_order]


def _transform_raw_to_response_prob(
    metric_name: str,
    raw_values: np.ndarray,
    anchor_elo: float = DEFAULT_ANCHOR_ELO,
) -> Tuple[np.ndarray, float]:
    """
    Map raw metric values to canonical item mastery probabilities p in [1e-4, 1 - 1e-4]
    so higher p always corresponds to stronger capability.

    Returns `(probs, scale_tau)`.
    """
    eps = 1e-4
    if metric_name == "witness_rate":
        probs = np.clip(raw_values, eps, 1.0 - eps)
        return probs, 1.0
    if metric_name == "kl":
        # Monotone exponential survival link: p = exp(-tau * KL), where median KL maps to 0.5
        pos_vals = raw_values[raw_values > 1e-9]
        med_kl = float(np.median(pos_vals)) if len(pos_vals) > 0 else 0.25
        tau = math.log(2.0) / max(med_kl, 1e-4)
        probs = np.exp(-tau * np.maximum(0.0, raw_values))
        return np.clip(probs, eps, 1.0 - eps), tau
    if metric_name == "brier":
        pos_vals = raw_values[raw_values > 1e-9]
        med_brier = float(np.median(pos_vals)) if len(pos_vals) > 0 else 0.15
        tau = math.log(2.0) / max(med_brier, 1e-4)
        probs = np.exp(-tau * np.maximum(0.0, raw_values))
        return np.clip(probs, eps, 1.0 - eps), tau
    if metric_name == "elo":
        # Exact Bradley-Terry win probability against anchor_elo:
        # logit(p) = (Elo - anchor_elo) / ELO_SCALE
        logits = (raw_values - anchor_elo) / ELO_SCALE
        logits = np.clip(logits, -12.0, 12.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.clip(probs, eps, 1.0 - eps), ELO_SCALE
    raise ValueError(f"Unsupported metric_name '{metric_name}'")


def _fit_rasch_2pl_matrix(
    P_obs: np.ndarray,
    mask: np.ndarray,
    item_weights: np.ndarray,
    prior_sigma_theta: float = 10.0,
    prior_sigma_log_a: float = 2.0,
    prior_sigma_b: float = 10.0,
    fit_2pl_slopes: bool = True,
    init_params: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit the 1D Rasch / 2-Parameter Logistic (2PL) Item Response Theory model
    to an observed mastery probability matrix `P_obs` of shape `(M, K)` with
    boolean observation `mask` of shape `(M, K)`.

    Model:
      P_hat_{m, k} = sigma(a_k * (theta_m - b_k))
    Loss:
      Weighted Bernoulli cross-entropy (KL divergence) + Gaussian priors on
      theta_m, log(a_k), and b_k.

    Returns:
      `(theta, a, b)` of shapes `(M,)`, `(K,)`, `(K,)` normalized so that
      the geometric mean of `a` is 1.0 and the mean of `theta` is 0.0.
    """
    m_agents, k_items = P_obs.shape
    w_mat = mask.astype(np.float64) * item_weights[None, :]

    if init_params is not None and len(init_params) == m_agents + 2 * k_items:
        x0 = init_params.copy()
    else:
        # Spectral / logit SVD initialization for fast convergence
        logit_obs = np.log(np.clip(P_obs, 1e-4, 1.0 - 1e-4) / np.clip(1.0 - P_obs, 1e-4, 1.0 - 1e-4))
        b_init = -np.sum(logit_obs * w_mat, axis=0) / np.maximum(np.sum(w_mat, axis=0), 1e-8)
        centered = (logit_obs + b_init[None, :]) * mask
        theta_init = np.sum(centered * w_mat, axis=1) / np.maximum(np.sum(w_mat, axis=1), 1e-8)
        theta_init = theta_init - np.mean(theta_init)
        log_a_init = np.zeros(k_items, dtype=np.float64)
        x0 = np.concatenate([theta_init, log_a_init, b_init])

    inv_var_theta = 1.0 / (prior_sigma_theta ** 2)
    inv_var_loga = 1.0 / (prior_sigma_log_a ** 2)
    inv_var_b = 1.0 / (prior_sigma_b ** 2)

    def nll_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
        theta = x[:m_agents]
        log_a = np.clip(x[m_agents : m_agents + k_items], -2.0, 2.0) if fit_2pl_slopes else np.zeros(k_items)
        b = x[m_agents + k_items :]
        a = np.exp(log_a)

        # z_{m, k} = a_k * (theta_m - b_k)
        diff = theta[:, None] - b[None, :]
        z = diff * a[None, :]
        z_clip = np.clip(z, -20.0, 20.0)
        p_hat = 1.0 / (1.0 + np.exp(-z_clip))

        # Cross-entropy: - [P_obs * log(p_hat) + (1 - P_obs) * log(1 - p_hat)]
        # = log(1 + exp(z)) - P_obs * z
        softplus_z = np.where(z_clip > 0, z_clip + np.log1p(np.exp(-z_clip)), np.log1p(np.exp(z_clip)))
        ce = w_mat * (softplus_z - P_obs * z_clip)
        nll = float(np.sum(ce))

        # Priors
        nll += 0.5 * inv_var_theta * float(np.sum(theta * theta))
        if fit_2pl_slopes:
            nll += 0.5 * inv_var_loga * float(np.sum(log_a * log_a))
        nll += 0.5 * inv_var_b * float(np.sum(b * b))

        # Gradients: d(ce)/dz = w_mat * (p_hat - P_obs)
        dz = w_mat * (p_hat - P_obs)
        grad_theta = np.sum(dz * a[None, :], axis=1) + inv_var_theta * theta
        grad_b = -np.sum(dz * a[None, :], axis=0) + inv_var_b * b
        if fit_2pl_slopes:
            grad_log_a = np.sum(dz * diff, axis=0) * a + inv_var_loga * log_a
        else:
            grad_log_a = np.zeros(k_items, dtype=np.float64)

        return nll, np.concatenate([grad_theta, grad_log_a, grad_b])

    res = minimize(
        nll_and_grad,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 600},
    )
    x_opt = res.x
    theta_opt = x_opt[:m_agents]
    log_a_opt = np.clip(x_opt[m_agents : m_agents + k_items], -2.0, 2.0) if fit_2pl_slopes else np.zeros(k_items)
    b_opt = x_opt[m_agents + k_items :]

    # Gauge normalization for strict 2PL identifiability:
    # 1. Center theta around 0 (shift absorbed into b)
    mean_theta = float(np.mean(theta_opt))
    theta_opt = theta_opt - mean_theta
    b_opt = b_opt - mean_theta

    # 2. Normalize geometric mean of discriminations a_k to 1.0
    mean_log_a = float(np.mean(log_a_opt))
    a_scale = math.exp(mean_log_a)
    a_opt = np.exp(log_a_opt - mean_log_a)
    theta_opt = theta_opt * a_scale
    b_opt = b_opt * a_scale

    return theta_opt, a_opt, b_opt


class MiniECIFitter:
    """
    Epoch Capability Index (`Mini-ECI`) 1D IRT / Rasch-2PL Fitter combining
    single-agent benchmark records (`Sudoku`, `ShipHunt`, `Equiv-SAT`, `Minesweeper`)
    and multiplayer Elo leagues (`Hex-5x5`, `Connect-4`, `Micro-Stratego`,
    `Galanis-Market`, `Proposal-Poker`).
    """

    def __init__(
        self,
        anchor_mini_eci: float = DEFAULT_ANCHOR_MINI_ECI,
        eci_scale: float = DEFAULT_ECI_SCALE,
        anchor_elo: float = DEFAULT_ANCHOR_ELO,
    ) -> None:
        self.anchor_mini_eci = float(anchor_mini_eci)
        self.eci_scale = float(eci_scale)
        self.anchor_elo = float(anchor_elo)
        self.agents: Dict[str, AgentSpec] = {}
        self.single_agent_records: List[SingleAgentBenchmarkRecord] = []
        # Map: env_name -> agent_id -> elo
        self.elo_records: Dict[str, Dict[str, float]] = {}

    def register_agent(self, spec: AgentSpec) -> AgentSpec:
        self.agents[spec.agent_id] = spec
        return spec

    def add_single_agent_record(
        self,
        model_family: str,
        search_algorithm: str,
        compute_budget_B: int,
        task_name: str,
        kl: float,
        brier: float,
        witness_rate: float,
        seed: int = 0,
    ) -> SingleAgentBenchmarkRecord:
        spec = AgentSpec(
            model_family=model_family,
            search_algorithm=search_algorithm,
            compute_budget_B=int(compute_budget_B),
        )
        self.agents[spec.agent_id] = spec
        canon_task = normalize_benchmark_name(task_name)
        rec = SingleAgentBenchmarkRecord(
            agent_id=spec.agent_id,
            model_family=spec.model_family,
            search_algorithm=spec.search_algorithm,
            compute_budget_B=spec.compute_budget_B,
            task_name=canon_task,
            kl=float(kl),
            brier=float(brier),
            witness_rate=float(witness_rate),
            seed=int(seed),
        )
        self.single_agent_records.append(rec)
        return rec

    def add_league_result(self, league_result: LeagueFitResult) -> None:
        """Ingest fitted multiplayer Elo ratings from `LeagueFitResult`."""
        if league_result.env_name.upper() == "ALL":
            return
        canon_env = normalize_env_name(league_result.env_name)
        env_map = self.elo_records.setdefault(canon_env, {})
        for aid, rating in league_result.ratings.items():
            spec = AgentSpec(
                model_family=rating.model_family,
                search_algorithm=rating.search_algorithm,
                compute_budget_B=rating.compute_budget_B,
            )
            self.agents[aid] = spec
            env_map[aid] = float(rating.elo)

    def add_elo_rating(
        self,
        env_name: str,
        spec: AgentSpec,
        elo: float,
    ) -> None:
        canon_env = normalize_env_name(env_name)
        self.agents[spec.agent_id] = spec
        self.elo_records.setdefault(canon_env, {})[spec.agent_id] = float(elo)

    def select_anchor_agent(self, agent_ids: Sequence[str]) -> str:
        """Find the 1-pass B=0 anchor agent (or fallback to lowest B agent)."""
        for aid in agent_ids:
            spec = self.agents.get(aid)
            if spec and spec.model_family == "1-pass" and spec.compute_budget_B == 0:
                return aid
        for aid in agent_ids:
            spec = self.agents.get(aid)
            if spec and spec.compute_budget_B == 0:
                return aid
        return agent_ids[0]

    def fit(
        self,
        anchor_agent_id: Optional[str] = None,
        n_bootstrap: int = 200,
        fit_2pl_slopes: bool = True,
        seed: int = 42,
    ) -> MiniECIFitResult:
        """
        Jointly fit the 1D Rasch-2PL Item Response Theory model across all single-agent
        metrics (`kl`, `brier`, `witness_rate` on `Sudoku`, `ShipHunt`, `Equiv-SAT`,
        `Minesweeper`) and multiplayer Elo ladders (`Hex-5x5`, `Connect-4`,
        `Micro-Stratego`, `Galanis-Market`, `Proposal-Poker`).
        """
        agent_list = sorted(self.agents.keys())
        if not agent_list:
            raise ValueError("No agents registered in MiniECIFitter.")
        m_agents = len(agent_list)
        agent_to_idx = {aid: i for i, aid in enumerate(agent_list)}

        anchor_id = anchor_agent_id or self.select_anchor_agent(agent_list)
        if anchor_id not in agent_to_idx:
            anchor_id = agent_list[0]
        anchor_idx = agent_to_idx[anchor_id]

        # 1. Aggregate single-agent records by (task_name, agent_id) across seeds
        sa_tasks = sorted({r.task_name for r in self.single_agent_records})
        # Build item definitions: (item_id, domain, task_name, metric_name, weight)
        item_defs: List[Tuple[str, str, str, str, float]] = []
        for task in sa_tasks:
            for metric in SINGLE_AGENT_METRICS:
                # Each single-agent task has 3 metrics (kl, brier, witness_rate), so weight 1/3 each
                # to balance 1 single-agent task against 1 multiplayer Elo game ladder
                item_defs.append(
                    (f"{task}::{metric}", "single_agent", task, metric, 1.0 / 3.0)
                )

        elo_envs = sorted(self.elo_records.keys())
        for env in elo_envs:
            item_defs.append((f"{env}::elo", "multiplayer_elo", env, "elo", 1.0))

        k_items = len(item_defs)
        if k_items == 0:
            raise ValueError("No benchmark records or Elo ratings provided to MiniECIFitter.")

        raw_matrix = np.full((m_agents, k_items), np.nan, dtype=np.float64)
        mask = np.zeros((m_agents, k_items), dtype=bool)
        item_weights = np.array([d[4] for d in item_defs], dtype=np.float64)

        # Populate single-agent raw averages
        sa_accum: Dict[Tuple[str, str, str], List[float]] = {}
        for r in self.single_agent_records:
            sa_accum.setdefault((r.task_name, "kl", r.agent_id), []).append(r.kl)
            sa_accum.setdefault((r.task_name, "brier", r.agent_id), []).append(r.brier)
            sa_accum.setdefault((r.task_name, "witness_rate", r.agent_id), []).append(
                r.witness_rate
            )

        for k_idx, (item_id, domain, task, metric, _) in enumerate(item_defs):
            if domain == "single_agent":
                for aid, m_idx in agent_to_idx.items():
                    vals = sa_accum.get((task, metric, aid))
                    if vals:
                        raw_matrix[m_idx, k_idx] = float(np.mean(vals))
                        mask[m_idx, k_idx] = True
            else:
                env_map = self.elo_records.get(task, {})
                for aid, elo_val in env_map.items():
                    if aid in agent_to_idx:
                        m_idx = agent_to_idx[aid]
                        raw_matrix[m_idx, k_idx] = float(elo_val)
                        mask[m_idx, k_idx] = True

        # Transform raw values to canonical response probabilities P_obs in (0, 1)
        P_obs = np.full((m_agents, k_items), 0.5, dtype=np.float64)
        scale_taus = np.ones(k_items, dtype=np.float64)
        for k_idx, (_, _, _, metric, _) in enumerate(item_defs):
            obs_col = raw_matrix[:, k_idx]
            valid = mask[:, k_idx]
            if np.any(valid):
                probs_valid, tau = _transform_raw_to_response_prob(
                    metric, obs_col[valid], anchor_elo=self.anchor_elo
                )
                P_obs[valid, k_idx] = probs_valid
                scale_taus[k_idx] = tau

        # Fit joint 2PL IRT model
        theta_hat, a_hat, b_hat = _fit_rasch_2pl_matrix(
            P_obs=P_obs,
            mask=mask,
            item_weights=item_weights,
            fit_2pl_slopes=fit_2pl_slopes,
        )

        def theta_to_eci(theta_vec: np.ndarray) -> np.ndarray:
            return self.anchor_mini_eci + self.eci_scale * (
                theta_vec - theta_vec[anchor_idx]
            )

        mini_eci_hat = theta_to_eci(theta_hat)

        # Also fit domain-specific sub-ECIs (single-agent only vs. multiplayer Elo only)
        sa_mask = mask.copy()
        elo_mask = mask.copy()
        for k_idx, (_, domain, _, _, _) in enumerate(item_defs):
            if domain != "single_agent":
                sa_mask[:, k_idx] = False
            if domain != "multiplayer_elo":
                elo_mask[:, k_idx] = False

        if np.any(sa_mask):
            theta_sa, _, _ = _fit_rasch_2pl_matrix(
                P_obs=P_obs,
                mask=sa_mask,
                item_weights=item_weights,
                fit_2pl_slopes=fit_2pl_slopes,
            )
            sa_sub_eci = theta_to_eci(theta_sa)
        else:
            sa_sub_eci = mini_eci_hat.copy()

        if np.any(elo_mask):
            theta_elo, _, _ = _fit_rasch_2pl_matrix(
                P_obs=P_obs,
                mask=elo_mask,
                item_weights=item_weights,
                fit_2pl_slopes=fit_2pl_slopes,
            )
            elo_sub_eci = theta_to_eci(theta_elo)
        else:
            elo_sub_eci = mini_eci_hat.copy()

        # Bootstrap 95% confidence intervals (resampling items + observation perturbations)
        rng = np.random.default_rng(seed)
        boot_ecis = np.zeros((max(1, n_bootstrap), m_agents), dtype=np.float64)
        init_vec = np.concatenate([theta_hat, np.log(np.maximum(a_hat, 1e-4)), b_hat])

        if n_bootstrap > 0:
            for b_iter in range(n_bootstrap):
                # Bayesian bootstrap (Dirichlet / exponential weights over items + small observation noise)
                sampled_k = rng.choice(k_items, size=k_items, replace=True)
                b_weights = np.zeros(k_items, dtype=np.float64)
                for k_chosen in sampled_k:
                    b_weights[k_chosen] += item_weights[k_chosen]
                # Add subtle logit perturbation reflecting finite-sample seed variance
                logit_p = np.log(P_obs / (1.0 - P_obs))
                noisy_logit = logit_p + rng.normal(0.0, 0.06, size=P_obs.shape)
                P_boot = np.clip(1.0 / (1.0 + np.exp(-noisy_logit)), 1e-4, 1.0 - 1e-4)
                b_theta, _, _ = _fit_rasch_2pl_matrix(
                    P_obs=P_boot,
                    mask=mask,
                    item_weights=b_weights,
                    fit_2pl_slopes=fit_2pl_slopes,
                    init_params=init_vec,
                )
                boot_ecis[b_iter] = theta_to_eci(b_theta)
        else:
            boot_ecis[:] = mini_eci_hat[None, :]

        ci_low = np.percentile(boot_ecis, 2.5, axis=0)
        ci_high = np.percentile(boot_ecis, 97.5, axis=0)
        eci_std = np.std(boot_ecis, axis=0, ddof=1 if n_bootstrap > 1 else 0)

        # Compute item parameters and explained variance R^2 in logit space
        items_dict: Dict[str, IRTItemParameter] = {}
        logit_obs_all = np.log(P_obs / (1.0 - P_obs))
        logit_pred_all = (theta_hat[:, None] - b_hat[None, :]) * a_hat[None, :]

        ss_res_total = 0.0
        ss_tot_total = 0.0
        for k_idx, (item_id, domain, task, metric, _) in enumerate(item_defs):
            valid = mask[:, k_idx]
            if np.sum(valid) >= 2:
                y_k = logit_obs_all[valid, k_idx]
                yhat_k = logit_pred_all[valid, k_idx]
                ss_res = float(np.sum((y_k - yhat_k) ** 2))
                ss_tot = float(np.sum((y_k - np.mean(y_k)) ** 2))
                r2_k = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12))
                ss_res_total += ss_res
                ss_tot_total += ss_tot
            else:
                r2_k = 1.0
            items_dict[item_id] = IRTItemParameter(
                item_id=item_id,
                domain=domain,
                task_name=task,
                metric_name=metric,
                discrimination_a=float(a_hat[k_idx]),
                difficulty_b=float(b_hat[k_idx]),
                scale_tau=float(scale_taus[k_idx]),
                r_squared_with_theta=float(r2_k),
            )

        explained_r2 = max(0.0, 1.0 - ss_res_total / max(ss_tot_total, 1e-12))

        # Leave-one-task-out cross-validation correlation
        all_tasks = sorted({d[2] for d in item_defs})
        loto_corrs: List[float] = []
        if len(all_tasks) >= 2:
            for held_task in all_tasks:
                train_mask = mask.copy()
                test_mask = mask.copy()
                for k_idx, (_, _, task, _, _) in enumerate(item_defs):
                    if task == held_task:
                        train_mask[:, k_idx] = False
                    else:
                        test_mask[:, k_idx] = False
                if np.any(train_mask) and np.any(test_mask):
                    th_train, _, _ = _fit_rasch_2pl_matrix(
                        P_obs=P_obs,
                        mask=train_mask,
                        item_weights=item_weights,
                        fit_2pl_slopes=False,
                    )
                    th_test, _, _ = _fit_rasch_2pl_matrix(
                        P_obs=P_obs,
                        mask=test_mask,
                        item_weights=item_weights,
                        fit_2pl_slopes=False,
                    )
                    if np.std(th_train) > 1e-9 and np.std(th_test) > 1e-9:
                        corr = float(np.corrcoef(th_train, th_test)[0, 1])
                        loto_corrs.append(corr)
        loto_r = float(np.mean(loto_corrs)) if loto_corrs else 1.0

        # Build per-agent scores
        scores: Dict[str, AgentMiniECIScore] = {}
        for i, aid in enumerate(agent_list):
            spec = self.agents[aid]
            # Compute agent's raw mean metrics across observed tasks
            kls = [
                raw_matrix[i, k]
                for k, d in enumerate(item_defs)
                if d[3] == "kl" and mask[i, k]
            ]
            briers = [
                raw_matrix[i, k]
                for k, d in enumerate(item_defs)
                if d[3] == "brier" and mask[i, k]
            ]
            wits = [
                raw_matrix[i, k]
                for k, d in enumerate(item_defs)
                if d[3] == "witness_rate" and mask[i, k]
            ]
            elos = [
                raw_matrix[i, k]
                for k, d in enumerate(item_defs)
                if d[3] == "elo" and mask[i, k]
            ]
            scores[aid] = AgentMiniECIScore(
                agent_id=aid,
                model_family=spec.model_family,
                search_algorithm=spec.search_algorithm,
                compute_budget_B=spec.compute_budget_B,
                mini_eci=float(mini_eci_hat[i]),
                mini_eci_ci_low=float(ci_low[i]),
                mini_eci_ci_high=float(ci_high[i]),
                mini_eci_std=float(eci_std[i]),
                latent_theta=float(theta_hat[i] - theta_hat[anchor_idx]),
                single_agent_sub_eci=float(sa_sub_eci[i]),
                multiplayer_elo_sub_eci=float(elo_sub_eci[i]),
                mean_kl=float(np.mean(kls)) if kls else 0.0,
                mean_brier=float(np.mean(briers)) if briers else 0.0,
                mean_witness_rate=float(np.mean(wits)) if wits else 0.0,
                mean_elo=float(np.mean(elos)) if elos else self.anchor_elo,
                num_items_observed=int(np.sum(mask[i, :])),
            )

        agent_order = sorted(agent_list, key=lambda a: scores[a].mini_eci, reverse=True)

        # Fit compute-scaling curves for every (model_family, search_algorithm)
        scaling_curves = self._fit_all_scaling_curves(scores)

        return MiniECIFitResult(
            anchor_agent_id=anchor_id,
            anchor_mini_eci=self.anchor_mini_eci,
            eci_scale=self.eci_scale,
            scores=scores,
            agent_order=agent_order,
            items=items_dict,
            scaling_curves=scaling_curves,
            explained_variance_r2=float(explained_r2),
            leave_one_task_out_r=loto_r,
            bootstrap_replicates=n_bootstrap,
            seed=seed,
        )

    def _fit_all_scaling_curves(
        self,
        scores: Dict[str, AgentMiniECIScore],
    ) -> Dict[str, ComputeScalingCurveFit]:
        groups: Dict[Tuple[str, str], List[AgentMiniECIScore]] = {}
        for s in scores.values():
            groups.setdefault((s.model_family, s.search_algorithm), []).append(s)

        curves: Dict[str, ComputeScalingCurveFit] = {}
        for (fam, alg), group in sorted(groups.items()):
            group.sort(key=lambda x: x.compute_budget_B)
            budgets = [g.compute_budget_B for g in group]
            ecis = [g.mini_eci for g in group]
            ci_lows = [g.mini_eci_ci_low for g in group]
            ci_highs = [g.mini_eci_ci_high for g in group]
            kls = [g.mean_kl for g in group]
            briers = [g.mean_brier for g in group]
            wits = [g.mean_witness_rate for g in group]
            elos = [g.mean_elo for g in group]

            b_arr = np.asarray(budgets, dtype=np.float64)
            y_arr = np.asarray(ecis, dtype=np.float64)

            if len(b_arr) >= 2 and np.max(b_arr) > np.min(b_arr):
                # Grid search b0 in [0.5, 8.0] for best log-linear fit E0 + kappa * log2(1 + B / b0)
                best_r2 = -1.0
                best_e0 = float(y_arr[0])
                best_kappa = 0.0
                best_b0 = 1.0
                best_pred = y_arr.copy()
                for b0_cand in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
                    x_cand = np.log2(1.0 + b_arr / b0_cand)
                    if np.var(x_cand) < 1e-12:
                        continue
                    kappa, e0 = np.polyfit(x_cand, y_arr, 1)
                    pred = e0 + kappa * x_cand
                    ss_res = float(np.sum((y_arr - pred) ** 2))
                    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
                    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
                    if r2 > best_r2:
                        best_r2 = r2
                        best_e0 = float(e0)
                        best_kappa = float(kappa)
                        best_b0 = float(b0_cand)
                        best_pred = pred
            else:
                best_e0 = float(y_arr[0])
                best_kappa = 0.0
                best_b0 = 1.0
                best_pred = y_arr.copy()
                best_r2 = 1.0

            eci_diffs = np.diff(y_arr) if len(y_arr) >= 2 else np.array([0.0])
            max_eci_drop = float(max(0.0, -np.min(eci_diffs)))
            eci_monotone = bool(max_eci_drop <= 0.35)

            kl_arr = np.asarray(kls, dtype=np.float64)
            kl_diffs = np.diff(kl_arr) if len(kl_arr) >= 2 else np.array([0.0])
            max_kl_rise = float(max(0.0, np.max(kl_diffs)))
            kl_monotone = bool(max_kl_rise <= 0.002)

            rel_kl_cut = (
                float((kl_arr[0] - kl_arr[-1]) / max(kl_arr[0], 1e-9))
                if len(kl_arr) >= 2 and kl_arr[0] > 1e-9
                else 0.0
            )

            key = f"{fam}::{alg}"
            curves[key] = ComputeScalingCurveFit(
                model_family=fam,
                search_algorithm=alg,
                budgets=budgets,
                mini_ecis=ecis,
                ci_lows=ci_lows,
                ci_highs=ci_highs,
                mean_kls=kls,
                mean_briers=briers,
                mean_witness_rates=wits,
                mean_elos=elos,
                e0_baseline=best_e0,
                kappa_per_doubling=best_kappa,
                b0_knee=best_b0,
                predicted_mini_ecis=[float(v) for v in best_pred],
                r_squared=float(max(0.0, best_r2)),
                eci_is_monotone=eci_monotone,
                max_eci_drop=max_eci_drop,
                kl_is_monotone=kl_monotone,
                max_kl_rise=max_kl_rise,
                total_eci_gain_B128=float(y_arr[-1] - y_arr[0]),
                relative_kl_reduction_B128=rel_kl_cut,
            )
        return curves


# ---------------------------------------------------------------------------
# Calibrated Single-Agent Benchmark Generator for Testing & Reference Baselines
# ---------------------------------------------------------------------------

# Task baseline difficulties (0-step KL, 0-step Brier, 0-step witness rate)
# grounded in Rounds 1-9 empirical benchmarks (Direction Note v4 §5):
_SINGLE_AGENT_TASK_BASELINES: Dict[str, Tuple[float, float, float]] = {
    "Sudoku": (0.850, 0.240, 0.280),  # 6x6 & 9x9 Sudoku-Extreme composite
    "ShipHunt": (0.280, 0.145, 0.520),  # Coupled ShipHunt 6x6 & 8x8 (0.2797 nats at B=0)
    "Equiv-SAT": (0.620, 0.210, 0.350),  # Latent-Equivalence-Parity P(A == B)
    "Minesweeper": (0.440, 0.180, 0.420),  # Minesweeper-6x6 epistemic vs forced deduction
}


def simulate_single_agent_benchmarks(
    agent_specs: Sequence[AgentSpec],
    tasks: Sequence[str] = SINGLE_AGENT_BENCHMARKS,
    seeds: Sequence[int] = (101, 102, 103),
) -> List[SingleAgentBenchmarkRecord]:
    """
    Generate realistic, seed-reproducible single-agent benchmark evaluations
    (`kl`, `brier`, `witness_rate`) across `Sudoku`, `ShipHunt`, `Equiv-SAT`,
    and `Minesweeper` for a population of `AgentSpec`s.

    Reflects key empirical phenomena from Rounds 1-9:
      - Anchor-aware `Learned-KV-Tree` and `Fractal-Prospectus-Tree` satisfy
        strict anytime KL monotonicity across B = 0..128.
      - Unanchored `Flat-MC P3` and `Fused` suffer over-search / non-monotone
        degradation at large B on strong roots or under utility bias.
      - Bidirectional latent lemma space (`Fractal-Prospectus-Tree`) excels on
        `Equiv-SAT` and multi-stall `Sudoku`.
    """
    records: List[SingleAgentBenchmarkRecord] = []

    family_rates: Dict[str, Tuple[float, float, bool]] = {
        # (kl_decay_rate, witness_gain_rate, is_anytime_monotone)
        "1-pass": (0.00, 0.00, True),
        "Flat-MC P3": (0.11, 0.045, False),
        "PUCT": (0.17, 0.068, True),
        "Hardcoded-Beam P8": (0.21, 0.078, True),
        "Learned-KV-Tree": (0.28, 0.098, True),
        "Fractal-Prospectus-Tree": (0.34, 0.115, True),
        "Multi-Game Generalist": (0.32, 0.108, True),
        "Single-Game Specialist": (0.31, 0.104, True),
        "LIR-Masked": (0.29, 0.100, True),
        "Token-Masked": (0.19, 0.072, False),
        "Fused": (0.10, 0.040, False),
    }

    for spec in agent_specs:
        kl_rate, wit_rate, monotone = family_rates.get(
            spec.model_family, (0.18, 0.07, True)
        )
        b = float(spec.compute_budget_B)
        log_b = math.log2(1.0 + b)

        for task in tasks:
            canon_task = normalize_benchmark_name(task)
            kl0, brier0, wit0 = _SINGLE_AGENT_TASK_BASELINES.get(
                canon_task, (0.50, 0.20, 0.40)
            )

            # Task-specific architectural synergies
            task_mult = 1.0
            if canon_task == "Equiv-SAT" and spec.model_family in (
                "Fractal-Prospectus-Tree",
                "Learned-KV-Tree",
                "Multi-Game Generalist",
            ):
                task_mult = 1.18  # Direct query rollouts Q(A, B) + D_amp >> |V| lemmas
            elif canon_task == "Sudoku" and spec.model_family == "Fractal-Prospectus-Tree":
                task_mult = 1.15  # Self-similar prospectus plans across multi-stall trees

            for s_idx, seed in enumerate(seeds):
                rng = np.random.default_rng(
                    seed * 1009 + hash((spec.agent_id, canon_task)) % 10007
                )
                # Seed-level baseline offset (consistent across B for the same seed & family)
                seed_offset = 0.004 * (s_idx - 1)

                effective_decay = kl_rate * task_mult * log_b
                # Non-monotone oversearch penalty for unanchored / fused pools at B >= 32
                oversearch_kl = 0.0
                if not monotone and b >= 32:
                    oversearch_kl = 0.048 * math.log2(b / 16.0)
                    if spec.model_family == "Fused":
                        oversearch_kl *= 1.8

                kl_val = max(
                    0.005,
                    kl0 * math.exp(-effective_decay) + oversearch_kl + max(0.0, seed_offset),
                )
                if monotone and b > 0:
                    # Tiny noise that preserves strict anytime non-degradation
                    kl_val = max(0.005, kl_val - 0.0002 * log_b)

                brier_val = max(
                    0.008,
                    brier0 * math.exp(-0.85 * effective_decay) + 0.4 * oversearch_kl,
                )

                # Witness rate approaches 1 - (1 - wit0) * exp(-wit_gain)
                wit_gain = wit_rate * task_mult * log_b
                if not monotone and b >= 64:
                    wit_gain -= 0.03 * math.log2(b / 32.0)
                wit_val = 1.0 - (1.0 - wit0) * math.exp(-max(0.0, wit_gain))
                wit_val = float(np.clip(wit_val + 0.002 * float(rng.normal()), 0.01, 0.995))

                records.append(
                    SingleAgentBenchmarkRecord(
                        agent_id=spec.agent_id,
                        model_family=spec.model_family,
                        search_algorithm=spec.search_algorithm,
                        compute_budget_B=spec.compute_budget_B,
                        task_name=canon_task,
                        kl=float(kl_val),
                        brier=float(brier_val),
                        witness_rate=float(wit_val),
                        seed=int(seed),
                    )
                )
    return records
