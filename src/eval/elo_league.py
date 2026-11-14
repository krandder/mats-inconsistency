"""
Multiplayer Bayesian Elo / Bradley-Terry-Davidson (BTD) Rating Engine
and Tournament Framework for `futarchy-fi/inconsistency-search`.

Evaluates head-to-head and population tournaments across:
  - `Hex-5x5`
  - `Connect-4`
  - `Micro-Stratego` (`Micro-Stratego-4x4`, `Micro-Stratego-6x6`)
  - `Galanis-Market` (4 information-structure prediction-market equilibria)
  - `Proposal-Poker` (futarchy governance staking with private signal s_i and bias y_i)

Supports agent populations indexed by:
  `(model_family, search_algorithm, compute_budget_B)`
for `B in {0, 1, 2, 4, 8, 16, 32, 64, 128}`.

Mathematical Formulation (Bradley-Terry-Davidson with Bayesian Prior):
  For agents i, j with latent log-strengths theta_i, theta_j, first-mover advantage
  gamma (with position sign s in {-1, 0, +1}), and Davidson draw parameter nu >= 0:
    delta_{ij}(s) = (theta_i - theta_j) + s * gamma
    Z_{ij}(s) = exp(delta_{ij}(s) / 2) + exp(-delta_{ij}(s) / 2) + nu
    P(i beats j | s) = exp(delta_{ij}(s) / 2) / Z_{ij}(s)
    P(j beats i | s) = exp(-delta_{ij}(s) / 2) / Z_{ij}(s)
    P(draw | s)      = nu / Z_{ij}(s)

  Elo conversion (anchored at R_anchor = 1000.0 for the 1-pass B=0 baseline):
    Elo_i = R_anchor + (400.0 / ln(10)) * (theta_i - theta_anchor)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize

ELO_SCALE: float = 400.0 / math.log(10.0)  # ~173.7177927613007
DEFAULT_ANCHOR_ELO: float = 1000.0

COMPUTE_BUDGETS: Tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64, 128)

MULTIPLAYER_ENVS: Tuple[str, ...] = (
    "Hex-5x5",
    "Connect-4",
    "Micro-Stratego",
    "Galanis-Market",
    "Proposal-Poker",
)

MODEL_FAMILIES: Tuple[str, ...] = (
    "1-pass",
    "Flat-MC P3",
    "PUCT",
    "Hardcoded-Beam P8",
    "Learned-KV-Tree",
    "Fractal-Prospectus-Tree",
    "Multi-Game Generalist",
    "Single-Game Specialist",
    "LIR-Masked",
    "Token-Masked",
    "Fused",
)


@dataclass(frozen=True, order=True)
class AgentSpec:
    """Canonical identifier for an agent in the population tournament."""

    model_family: str
    search_algorithm: str
    compute_budget_B: int

    def __post_init__(self) -> None:
        if int(self.compute_budget_B) < 0:
            raise ValueError(
                f"compute_budget_B must be non-negative, got {self.compute_budget_B}"
            )

    @property
    def is_standard_budget(self) -> bool:
        return self.compute_budget_B in COMPUTE_BUDGETS

    @property
    def agent_id(self) -> str:
        return f"{self.model_family}::{self.search_algorithm}@B={self.compute_budget_B}"

    @classmethod
    def from_id(cls, agent_id: str) -> "AgentSpec":
        """Parse canonical string `'model_family::search_algorithm@B=16'`."""
        if "::" in agent_id and "@B=" in agent_id:
            fam, rest = agent_id.split("::", 1)
            alg, b_str = rest.rsplit("@B=", 1)
            return cls(
                model_family=fam.strip(),
                search_algorithm=alg.strip(),
                compute_budget_B=int(b_str.strip()),
            )
        raise ValueError(
            f"Invalid canonical agent_id '{agent_id}'. Expected 'family::algorithm@B=<int>'."
        )


@dataclass(frozen=True)
class MatchRecord:
    """Single pairwise match or pairwise projection of a multiplayer game."""

    env_name: str
    agent_a: str
    agent_b: str
    outcome_a: float  # 1.0 = A win, 0.5 = draw, 0.0 = B win (or continuous [0, 1] share)
    is_draw: bool = False
    first_player_sign: int = 0  # +1 if A moves first, -1 if B moves first, 0 if symmetric
    weight: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.outcome_a <= 1.0):
            raise ValueError(f"outcome_a must be in [0, 1], got {self.outcome_a}")
        if self.first_player_sign not in (-1, 0, 1):
            raise ValueError(
                f"first_player_sign must be -1, 0, or 1, got {self.first_player_sign}"
            )
        if self.weight <= 0.0:
            raise ValueError(f"weight must be positive, got {self.weight}")


@dataclass(frozen=True)
class MultiplayerGameRecord:
    """N-player game outcome (e.g., Galanis-Market or Proposal-Poker table)."""

    env_name: str
    agent_ids: Tuple[str, ...]
    payoffs: Tuple[float, ...]
    belief_kls: Optional[Tuple[float, ...]] = None
    honesty_scores: Optional[Tuple[float, ...]] = None
    seed: int = 0
    draw_atol: float = 1e-6

    def to_pairwise_matches(self) -> List[MatchRecord]:
        """Decompose N-player payoff vector into all (N choose 2) pairwise BTD comparisons."""
        n = len(self.agent_ids)
        if len(self.payoffs) != n or n < 2:
            raise ValueError("agent_ids and payoffs must have matching length >= 2")
        matches: List[MatchRecord] = []
        pair_weight = 2.0 / float(n)  # normalize so each player has total weight ~1 per game
        for i in range(n):
            for j in range(i + 1, n):
                diff = self.payoffs[i] - self.payoffs[j]
                if abs(diff) <= self.draw_atol:
                    outcome_a = 0.5
                    is_draw = True
                elif diff > 0.0:
                    outcome_a = 1.0
                    is_draw = False
                else:
                    outcome_a = 0.0
                    is_draw = False
                matches.append(
                    MatchRecord(
                        env_name=self.env_name,
                        agent_a=self.agent_ids[i],
                        agent_b=self.agent_ids[j],
                        outcome_a=outcome_a,
                        is_draw=is_draw,
                        first_player_sign=0,
                        weight=pair_weight,
                        seed=self.seed,
                    )
                )
        return matches


@dataclass
class AgentEloRating:
    """Calibrated Bayesian Elo / BTD rating with bootstrap 95% confidence interval."""

    agent_id: str
    model_family: str
    search_algorithm: str
    compute_budget_B: int
    env_name: str
    elo: float
    elo_ci_low: float
    elo_ci_high: float
    elo_std: float
    theta: float
    matches_played: float
    win_rate: float
    draw_rate: float

    def to_dict(self) -> Dict[str, Union[str, int, float]]:
        return asdict(self)


@dataclass
class LeagueFitResult:
    """Complete result of fitting a Bayesian BTD Elo league on an environment or suite."""

    env_name: str
    anchor_agent_id: str
    anchor_elo: float
    draw_nu: float
    draw_nu_ci: Tuple[float, float]
    first_mover_elo: float
    first_mover_elo_ci: Tuple[float, float]
    ratings: Dict[str, AgentEloRating]
    win_prob_matrix: np.ndarray  # shape (M, M), entry (i, j) = P(i beats j) + 0.5 * P(draw)
    dominance_matrix: np.ndarray  # shape (M, M), bootstrap P(Elo_i > Elo_j)
    agent_order: List[str]  # sorted descending by Elo
    num_matches: int
    bootstrap_replicates: int
    seed: int

    def get_rating(self, agent: Union[str, AgentSpec]) -> AgentEloRating:
        aid = agent.agent_id if isinstance(agent, AgentSpec) else agent
        return self.ratings[aid]

    def leaderboard(self) -> List[AgentEloRating]:
        return [self.ratings[aid] for aid in self.agent_order]


@dataclass
class ComputeScalingSummary:
    """Compute-scaling diagnostics across B in {0, 1, 2, 4, 8, 16, 32, 64, 128}."""

    model_family: str
    search_algorithm: str
    env_name: str
    budgets: List[int]
    elos: List[float]
    ci_lows: List[float]
    ci_highs: List[float]
    elo_gain_at_max_B: float
    elo_slope_per_doubling: float
    r_squared_log_linear: float
    is_monotone: bool
    max_monotonicity_drop_elo: float


def normalize_env_name(env_name: str) -> str:
    """Map environment aliases to canonical multiplayer benchmark names."""
    clean = env_name.strip()
    if clean.startswith("Micro-Stratego"):
        return "Micro-Stratego"
    if clean in ("Hex", "Hex-5x5"):
        return "Hex-5x5"
    if clean in ("Connect4", "Connect-4"):
        return "Connect-4"
    if clean in ("Galanis", "Galanis-Market", "galanis_market"):
        return "Galanis-Market"
    if clean in ("ProposalPoker", "Proposal-Poker", "proposal_poker"):
        return "Proposal-Poker"
    return clean


def btd_probabilities(
    theta_i: Union[float, np.ndarray],
    theta_j: Union[float, np.ndarray],
    nu: float = 0.25,
    first_mover_gamma: float = 0.0,
    first_player_sign: Union[int, np.ndarray] = 0,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Compute exact Bradley-Terry-Davidson (win_i, win_j, draw) probabilities.

    Numerically stable formulation using half-logit differences:
      u = 0.5 * ((theta_i - theta_j) + s * gamma)
      m = max(|u|, 0)
      exp(u - m) + exp(-u - m) + nu * exp(-m)
    """
    delta = (theta_i - theta_j) + first_player_sign * first_mover_gamma
    u = 0.5 * delta
    m = np.abs(u)
    eu = np.exp(u - m)
    emu = np.exp(-u - m)
    enu = nu * np.exp(-m)
    z = eu + emu + enu
    p_win_i = eu / z
    p_win_j = emu / z
    p_draw = enu / z
    return p_win_i, p_win_j, p_draw


def _aggregate_sufficient_stats(
    matches: Sequence[MatchRecord],
    agent_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate match records into unique (idx_a, idx_b, s) cells with
    weighted counts (w_win_a, w_win_b, w_draw).
    """
    cell_map: Dict[Tuple[int, int, int], List[float]] = {}
    for m in matches:
        ia = agent_to_idx[m.agent_a]
        ib = agent_to_idx[m.agent_b]
        if ia == ib:
            continue
        s = int(m.first_player_sign)
        key = (ia, ib, s)
        if key not in cell_map:
            cell_map[key] = [0.0, 0.0, 0.0]
        counts = cell_map[key]
        w = float(m.weight)
        if m.is_draw:
            counts[2] += w
        elif m.outcome_a == 1.0:
            counts[0] += w
        elif m.outcome_a == 0.0:
            counts[1] += w
        else:
            # Continuous outcome in (0, 1): split fractional wins
            counts[0] += w * m.outcome_a
            counts[1] += w * (1.0 - m.outcome_a)

    if not cell_map:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return empty_i, empty_i, empty_i, empty_f, empty_f, empty_f

    keys = list(cell_map.keys())
    vals = np.array([cell_map[k] for k in keys], dtype=np.float64)
    idx_a = np.array([k[0] for k in keys], dtype=np.int64)
    idx_b = np.array([k[1] for k in keys], dtype=np.int64)
    signs = np.array([k[2] for k in keys], dtype=np.float64)
    w_a = vals[:, 0]
    w_b = vals[:, 1]
    w_d = vals[:, 2]
    return idx_a, idx_b, signs, w_a, w_b, w_d


def _fit_btd_map_from_stats(
    num_agents: int,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    signs: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
    w_d: np.ndarray,
    prior_sigma: float = 2.0,
    fit_first_mover: bool = True,
    fit_draws: bool = True,
    init_params: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Solve the Maximum A Posteriori (MAP) Bradley-Terry-Davidson optimization
    with analytic gradients via L-BFGS-B.

    Returns:
      (theta, nu, gamma)
    """
    total_draws = float(np.sum(w_d))
    has_draws = fit_draws and (total_draws > 1e-9)
    has_signs = fit_first_mover and bool(np.any(signs != 0.0))

    if len(idx_a) == 0:
        return np.zeros(num_agents, dtype=np.float64), 0.0, 0.0

    inv_var = 1.0 / (prior_sigma * prior_sigma)

    # Parameter vector layout:
    # x[:num_agents] = theta
    # x[num_agents] = gamma (first-mover logit advantage)
    # x[num_agents + 1] = eta = log(nu) (log Davidson draw parameter)
    if init_params is not None and len(init_params) == num_agents + 2:
        x0 = init_params.copy()
    else:
        x0 = np.zeros(num_agents + 2, dtype=np.float64)
        total_games = float(np.sum(w_a + w_b + w_d))
        if has_draws and total_games > 0:
            draw_frac = np.clip(total_draws / total_games, 1e-3, 0.90)
            x0[num_agents + 1] = math.log(2.0 * draw_frac / (1.0 - draw_frac))
        else:
            x0[num_agents + 1] = -8.0

    def nll_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
        theta = x[:num_agents]
        gamma = x[num_agents] if has_signs else 0.0
        eta = np.clip(x[num_agents + 1], -12.0, 4.0) if has_draws else -20.0
        nu = math.exp(eta) if has_draws else 0.0

        delta = (theta[idx_a] - theta[idx_b]) + signs * gamma
        u = 0.5 * delta
        m = np.abs(u)
        eu = np.exp(u - m)
        emu = np.exp(-u - m)
        enu = nu * np.exp(-m)
        z = eu + emu + enu

        log_z = m + np.log(z)
        # Log probabilities:
        # log P(a) = u - log_z
        # log P(b) = -u - log_z
        # log P(d) = eta - log_z
        nll = float(
            np.sum(
                (w_a + w_b + w_d) * log_z
                - (w_a - w_b) * u
                - (w_d * eta if has_draws else 0.0)
            )
        )

        # Gaussian prior on theta, gamma, and weak prior on eta
        nll += 0.5 * inv_var * float(np.sum(theta * theta))
        if has_signs:
            nll += 0.5 * inv_var * (gamma * gamma)
        if has_draws:
            nll += 0.05 * (eta * eta)

        # Analytic gradient
        p_a = eu / z
        p_b = emu / z
        p_d = enu / z
        w_tot = w_a + w_b + w_d

        # d(nll)/du = w_tot * (p_a - p_b) - (w_a - w_b)
        du = w_tot * (p_a - p_b) - (w_a - w_b)
        d_delta = 0.5 * du

        grad = np.zeros_like(x)
        np.add.at(grad[:num_agents], idx_a, d_delta)
        np.add.at(grad[:num_agents], idx_b, -d_delta)
        grad[:num_agents] += inv_var * theta

        if has_signs:
            grad[num_agents] = float(np.sum(d_delta * signs)) + inv_var * gamma
        if has_draws:
            grad[num_agents + 1] = float(np.sum(w_tot * p_d - w_d)) + 0.10 * eta

        return nll, grad

    res = minimize(
        nll_and_grad,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
    )
    x_opt = res.x
    theta_opt = x_opt[:num_agents] - np.mean(x_opt[:num_agents])
    gamma_opt = float(x_opt[num_agents]) if has_signs else 0.0
    nu_opt = float(math.exp(np.clip(x_opt[num_agents + 1], -12.0, 4.0))) if has_draws else 0.0
    return theta_opt, nu_opt, gamma_opt


class BayesianEloLeague:
    """
    Multiplayer Bayesian Elo / Bradley-Terry-Davidson (BTD) tournament manager
    and rating engine with non-parametric bootstrap 95% confidence intervals.
    """

    def __init__(
        self,
        anchor_elo: float = DEFAULT_ANCHOR_ELO,
        prior_sigma: float = 2.0,
    ) -> None:
        self.anchor_elo = float(anchor_elo)
        self.prior_sigma = float(prior_sigma)
        self.agents: Dict[str, AgentSpec] = {}
        self.matches_by_env: Dict[str, List[MatchRecord]] = {
            env: [] for env in MULTIPLAYER_ENVS
        }

    def register_agent(
        self,
        model_family: str,
        search_algorithm: str,
        compute_budget_B: int,
    ) -> AgentSpec:
        spec = AgentSpec(
            model_family=model_family,
            search_algorithm=search_algorithm,
            compute_budget_B=int(compute_budget_B),
        )
        self.agents[spec.agent_id] = spec
        return spec

    def register_population(
        self,
        family_algorithms: Sequence[Tuple[str, str]],
        budgets: Sequence[int] = COMPUTE_BUDGETS,
    ) -> List[AgentSpec]:
        """Register a Cartesian grid of (model_family, search_algorithm) x compute_budget_B."""
        specs: List[AgentSpec] = []
        for fam, alg in family_algorithms:
            for b in budgets:
                # 1-pass baseline only needs B=0 unless explicitly requested for ablation
                specs.append(self.register_agent(fam, alg, b))
        return specs

    def _ensure_agent_registered(self, agent: Union[str, AgentSpec]) -> str:
        if isinstance(agent, AgentSpec):
            self.agents[agent.agent_id] = agent
            return agent.agent_id
        if agent not in self.agents:
            try:
                spec = AgentSpec.from_id(agent)
            except ValueError:
                spec = AgentSpec(
                    model_family=agent, search_algorithm="default", compute_budget_B=0
                )
            self.agents[agent] = spec
        return agent

    def add_match(
        self,
        env_name: str,
        agent_a: Union[str, AgentSpec],
        agent_b: Union[str, AgentSpec],
        outcome_a: float,
        is_draw: Optional[bool] = None,
        first_player_sign: int = 0,
        weight: float = 1.0,
        seed: int = 0,
    ) -> MatchRecord:
        canon_env = normalize_env_name(env_name)
        aid_a = self._ensure_agent_registered(agent_a)
        aid_b = self._ensure_agent_registered(agent_b)
        draw_flag = (abs(outcome_a - 0.5) < 1e-6) if is_draw is None else bool(is_draw)
        rec = MatchRecord(
            env_name=canon_env,
            agent_a=aid_a,
            agent_b=aid_b,
            outcome_a=float(outcome_a),
            is_draw=draw_flag,
            first_player_sign=int(first_player_sign),
            weight=float(weight),
            seed=int(seed),
        )
        self.matches_by_env.setdefault(canon_env, []).append(rec)
        return rec

    def add_multiplayer_game(self, record: MultiplayerGameRecord) -> List[MatchRecord]:
        canon_env = normalize_env_name(record.env_name)
        for aid in record.agent_ids:
            self._ensure_agent_registered(aid)
        pairwise = record.to_pairwise_matches()
        normalized_pairwise = [
            MatchRecord(
                env_name=canon_env,
                agent_a=m.agent_a,
                agent_b=m.agent_b,
                outcome_a=m.outcome_a,
                is_draw=m.is_draw,
                first_player_sign=m.first_player_sign,
                weight=m.weight,
                seed=m.seed,
            )
            for m in pairwise
        ]
        self.matches_by_env.setdefault(canon_env, []).extend(normalized_pairwise)
        return normalized_pairwise

    def select_anchor_agent(self, agent_ids: Sequence[str]) -> Optional[str]:
        """Select the 1-pass B=0 anchor agent if present, else the lowest-budget agent."""
        for aid in agent_ids:
            spec = self.agents.get(aid)
            if spec and spec.model_family == "1-pass" and spec.compute_budget_B == 0:
                return aid
        for aid in agent_ids:
            spec = self.agents.get(aid)
            if spec and spec.compute_budget_B == 0:
                return aid
        return agent_ids[0] if agent_ids else None

    def fit_environment(
        self,
        env_name: str,
        anchor_agent_id: Optional[str] = None,
        n_bootstrap: int = 200,
        seed: int = 42,
    ) -> LeagueFitResult:
        """
        Fit Bayesian BTD Elo ratings + bootstrap 95% confidence intervals for one environment,
        or across `"ALL"` environments pooled.
        """
        if env_name.upper() == "ALL":
            matches: List[MatchRecord] = []
            for env_matches in self.matches_by_env.values():
                matches.extend(env_matches)
            canon_env = "ALL"
        else:
            canon_env = normalize_env_name(env_name)
            matches = list(self.matches_by_env.get(canon_env, []))

        # Collect participating agents (or all registered agents if none)
        active_set = set()
        for m in matches:
            active_set.add(m.agent_a)
            active_set.add(m.agent_b)
        if not active_set:
            active_set = set(self.agents.keys())
        agent_list = sorted(active_set)
        num_agents = len(agent_list)
        agent_to_idx = {aid: i for i, aid in enumerate(agent_list)}

        anchor_id = anchor_agent_id or self.select_anchor_agent(agent_list) or ""
        anchor_idx = agent_to_idx.get(anchor_id, -1)

        idx_a, idx_b, signs, w_a, w_b, w_d = _aggregate_sufficient_stats(
            matches, agent_to_idx
        )
        theta_hat, nu_hat, gamma_hat = _fit_btd_map_from_stats(
            num_agents=num_agents,
            idx_a=idx_a,
            idx_b=idx_b,
            signs=signs,
            w_a=w_a,
            w_b=w_b,
            w_d=w_d,
            prior_sigma=self.prior_sigma,
        )

        def to_elo(theta_vec: np.ndarray) -> np.ndarray:
            ref = theta_vec[anchor_idx] if anchor_idx >= 0 else float(np.mean(theta_vec))
            return self.anchor_elo + ELO_SCALE * (theta_vec - ref)

        elo_hat = to_elo(theta_hat)
        first_mover_elo_hat = ELO_SCALE * gamma_hat

        # Non-parametric bootstrap over match outcomes using multinomial cell resampling
        rng = np.random.default_rng(seed)
        boot_elos = np.zeros((max(1, n_bootstrap), num_agents), dtype=np.float64)
        boot_nus = np.zeros(max(1, n_bootstrap), dtype=np.float64)
        boot_fm_elos = np.zeros(max(1, n_bootstrap), dtype=np.float64)

        init_vec = np.zeros(num_agents + 2, dtype=np.float64)
        init_vec[:num_agents] = theta_hat
        init_vec[num_agents] = gamma_hat
        init_vec[num_agents + 1] = math.log(max(nu_hat, 1e-6))

        num_cells = len(idx_a)
        if num_cells > 0 and n_bootstrap > 0:
            # Build flat vector of (win_a, win_b, draw) cell weights for multinomial resampling
            flat_weights = np.concatenate([w_a, w_b, w_d])
            total_weight = float(np.sum(flat_weights))
            probs = flat_weights / max(total_weight, 1e-12)
            n_draws = max(1, int(round(len(matches))))
            scale_factor = total_weight / float(n_draws)

            for b_idx in range(n_bootstrap):
                counts = rng.multinomial(n_draws, probs).astype(np.float64) * scale_factor
                b_wa = counts[:num_cells]
                b_wb = counts[num_cells : 2 * num_cells]
                b_wd = counts[2 * num_cells :]
                b_theta, b_nu, b_gamma = _fit_btd_map_from_stats(
                    num_agents=num_agents,
                    idx_a=idx_a,
                    idx_b=idx_b,
                    signs=signs,
                    w_a=b_wa,
                    w_b=b_wb,
                    w_d=b_wd,
                    prior_sigma=self.prior_sigma,
                    init_params=init_vec,
                )
                boot_elos[b_idx] = to_elo(b_theta)
                boot_nus[b_idx] = b_nu
                boot_fm_elos[b_idx] = ELO_SCALE * b_gamma
        else:
            boot_elos[:] = elo_hat[None, :]
            boot_nus[:] = nu_hat
            boot_fm_elos[:] = first_mover_elo_hat

        ci_low = np.percentile(boot_elos, 2.5, axis=0)
        ci_high = np.percentile(boot_elos, 97.5, axis=0)
        elo_std = np.std(boot_elos, axis=0, ddof=1 if n_bootstrap > 1 else 0)

        # Compute empirical match counts, win rates, draw rates per agent
        matches_played = np.zeros(num_agents, dtype=np.float64)
        wins_accum = np.zeros(num_agents, dtype=np.float64)
        draws_accum = np.zeros(num_agents, dtype=np.float64)
        if num_cells > 0:
            cell_tot = w_a + w_b + w_d
            np.add.at(matches_played, idx_a, cell_tot)
            np.add.at(matches_played, idx_b, cell_tot)
            np.add.at(wins_accum, idx_a, w_a + 0.5 * w_d)
            np.add.at(wins_accum, idx_b, w_b + 0.5 * w_d)
            np.add.at(draws_accum, idx_a, w_d)
            np.add.at(draws_accum, idx_b, w_d)

        # Pairwise expected score matrix (win_i + 0.5 * draw) under fitted BTD
        ti = theta_hat[:, None]
        tj = theta_hat[None, :]
        p_win_i, _, p_draw = btd_probabilities(ti, tj, nu=nu_hat, first_mover_gamma=0.0)
        win_prob_matrix = np.asarray(p_win_i + 0.5 * p_draw, dtype=np.float64)
        np.fill_diagonal(win_prob_matrix, 0.5)

        # Pairwise bootstrap dominance matrix P(Elo_i > Elo_j)
        dominance_matrix = np.mean(
            boot_elos[:, :, None] > boot_elos[:, None, :], axis=0
        ).astype(np.float64)
        np.fill_diagonal(dominance_matrix, 0.5)

        ratings: Dict[str, AgentEloRating] = {}
        for i, aid in enumerate(agent_list):
            spec = self.agents[aid]
            mp = float(matches_played[i])
            wr = float(wins_accum[i] / mp) if mp > 0 else 0.5
            dr = float(draws_accum[i] / mp) if mp > 0 else 0.0
            ratings[aid] = AgentEloRating(
                agent_id=aid,
                model_family=spec.model_family,
                search_algorithm=spec.search_algorithm,
                compute_budget_B=spec.compute_budget_B,
                env_name=canon_env,
                elo=float(elo_hat[i]),
                elo_ci_low=float(ci_low[i]),
                elo_ci_high=float(ci_high[i]),
                elo_std=float(elo_std[i]),
                theta=float(theta_hat[i]),
                matches_played=mp,
                win_rate=wr,
                draw_rate=dr,
            )

        agent_order = sorted(agent_list, key=lambda a: ratings[a].elo, reverse=True)
        return LeagueFitResult(
            env_name=canon_env,
            anchor_agent_id=anchor_id,
            anchor_elo=self.anchor_elo,
            draw_nu=float(nu_hat),
            draw_nu_ci=(
                float(np.percentile(boot_nus, 2.5)),
                float(np.percentile(boot_nus, 97.5)),
            ),
            first_mover_elo=float(first_mover_elo_hat),
            first_mover_elo_ci=(
                float(np.percentile(boot_fm_elos, 2.5)),
                float(np.percentile(boot_fm_elos, 97.5)),
            ),
            ratings=ratings,
            win_prob_matrix=win_prob_matrix,
            dominance_matrix=dominance_matrix,
            agent_order=agent_order,
            num_matches=len(matches),
            bootstrap_replicates=n_bootstrap,
            seed=seed,
        )

    def fit_all_environments(
        self,
        n_bootstrap: int = 200,
        seed: int = 42,
    ) -> Dict[str, LeagueFitResult]:
        """Fit per-environment Elo leagues plus a pooled `'ALL'` composite league."""
        results: Dict[str, LeagueFitResult] = {}
        for i, env in enumerate(MULTIPLAYER_ENVS):
            if self.matches_by_env.get(env):
                results[env] = self.fit_environment(
                    env_name=env,
                    n_bootstrap=n_bootstrap,
                    seed=seed + i * 17,
                )
        results["ALL"] = self.fit_environment(
            env_name="ALL",
            n_bootstrap=n_bootstrap,
            seed=seed + 999,
        )
        return results

    def compute_scaling_curve(
        self,
        fit_result: LeagueFitResult,
        model_family: str,
        search_algorithm: Optional[str] = None,
    ) -> ComputeScalingSummary:
        """
        Extract the test-time compute Elo scaling trajectory across B in {0, 1, 2, 4, 8, 16, 32, 64, 128}
        for a given `(model_family, search_algorithm)` in `fit_result`.
        """
        matched = [
            r
            for r in fit_result.ratings.values()
            if r.model_family == model_family
            and (search_algorithm is None or r.search_algorithm == search_algorithm)
        ]
        matched.sort(key=lambda r: r.compute_budget_B)
        if not matched:
            raise ValueError(
                f"No ratings found for model_family='{model_family}', search_algorithm='{search_algorithm}'"
            )

        alg_name = matched[0].search_algorithm
        budgets = [r.compute_budget_B for r in matched]
        elos = [r.elo for r in matched]
        ci_lows = [r.elo_ci_low for r in matched]
        ci_highs = [r.elo_ci_high for r in matched]

        # Fit log-linear scaling Elo(B) = a + slope * log2(1 + B)
        x = np.log2(1.0 + np.asarray(budgets, dtype=np.float64))
        y = np.asarray(elos, dtype=np.float64)
        if len(x) >= 2 and np.var(x) > 1e-12:
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = intercept + slope * x
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
        else:
            slope = 0.0
            r2 = 1.0

        diffs = np.diff(y) if len(y) >= 2 else np.array([0.0])
        max_drop = float(max(0.0, -np.min(diffs)))
        is_monotone = bool(max_drop <= 5.0)  # allow 5 Elo bootstrap noise tolerance

        return ComputeScalingSummary(
            model_family=model_family,
            search_algorithm=alg_name,
            env_name=fit_result.env_name,
            budgets=budgets,
            elos=elos,
            ci_lows=ci_lows,
            ci_highs=ci_highs,
            elo_gain_at_max_B=float(y[-1] - y[0]),
            elo_slope_per_doubling=float(slope),
            r_squared_log_linear=float(r2),
            is_monotone=is_monotone,
            max_monotonicity_drop_elo=max_drop,
        )

    def run_round_robin_tournament(
        self,
        env_name: str,
        agent_specs: Sequence[AgentSpec],
        match_fn: Callable[[str, AgentSpec, AgentSpec, int, int], Tuple[float, bool]],
        games_per_pair: int = 4,
        seed: int = 101,
    ) -> List[MatchRecord]:
        """
        Execute a balanced home-and-away round-robin tournament across `agent_specs`.
        `match_fn(env_name, agent_a, agent_b, first_player_sign, match_seed)` returns
        `(outcome_a, is_draw)`.
        """
        canon_env = normalize_env_name(env_name)
        for s in agent_specs:
            self.agents[s.agent_id] = s

        records: List[MatchRecord] = []
        n = len(agent_specs)
        has_first_mover = canon_env in ("Hex-5x5", "Connect-4", "Micro-Stratego")
        match_counter = 0

        for i in range(n):
            for j in range(i + 1, n):
                spec_a = agent_specs[i]
                spec_b = agent_specs[j]
                for g in range(games_per_pair):
                    match_counter += 1
                    m_seed = seed + match_counter * 31
                    sign = (1 if (g % 2 == 0) else -1) if has_first_mover else 0
                    outcome_a, is_draw = match_fn(canon_env, spec_a, spec_b, sign, m_seed)
                    records.append(
                        self.add_match(
                            env_name=canon_env,
                            agent_a=spec_a,
                            agent_b=spec_b,
                            outcome_a=outcome_a,
                            is_draw=is_draw,
                            first_player_sign=sign,
                            seed=m_seed,
                        )
                    )
        return records

    def run_swiss_tournament(
        self,
        env_name: str,
        agent_specs: Sequence[AgentSpec],
        match_fn: Callable[[str, AgentSpec, AgentSpec, int, int], Tuple[float, bool]],
        num_rounds: int = 7,
        games_per_pairing: int = 2,
        seed: int = 202,
    ) -> List[MatchRecord]:
        """
        Execute a Swiss-system tournament pairing agents with similar cumulative scores
        without repeating exact pairings when possible.
        """
        canon_env = normalize_env_name(env_name)
        for s in agent_specs:
            self.agents[s.agent_id] = s

        n = len(agent_specs)
        scores: Dict[str, float] = {s.agent_id: 0.0 for s in agent_specs}
        spec_map: Dict[str, AgentSpec] = {s.agent_id: s for s in agent_specs}
        played_pairs: set[Tuple[str, str]] = set()
        records: List[MatchRecord] = []
        has_first_mover = canon_env in ("Hex-5x5", "Connect-4", "Micro-Stratego")
        match_counter = 0

        for r in range(num_rounds):
            # Sort by score descending, then by agent_id for determinism
            ordered = sorted(
                spec_map.keys(),
                key=lambda aid: (scores[aid], aid),
                reverse=True,
            )
            unpaired = list(ordered)
            while len(unpaired) >= 2:
                a_id = unpaired.pop(0)
                # Find highest-ranked opponent not yet played
                chosen_idx = 0
                for idx, cand_id in enumerate(unpaired):
                    pair_key = (min(a_id, cand_id), max(a_id, cand_id))
                    if pair_key not in played_pairs:
                        chosen_idx = idx
                        break
                b_id = unpaired.pop(chosen_idx)
                played_pairs.add((min(a_id, b_id), max(a_id, b_id)))

                spec_a = spec_map[a_id]
                spec_b = spec_map[b_id]
                for g in range(games_per_pairing):
                    match_counter += 1
                    m_seed = seed + r * 1000 + match_counter * 19
                    sign = (1 if (g % 2 == 0) else -1) if has_first_mover else 0
                    outcome_a, is_draw = match_fn(canon_env, spec_a, spec_b, sign, m_seed)
                    scores[a_id] += outcome_a
                    scores[b_id] += 1.0 - outcome_a
                    records.append(
                        self.add_match(
                            env_name=canon_env,
                            agent_a=spec_a,
                            agent_b=spec_b,
                            outcome_a=outcome_a,
                            is_draw=is_draw,
                            first_player_sign=sign,
                            seed=m_seed,
                        )
                    )
        return records


# ---------------------------------------------------------------------------
# Calibrated Latent-Skill Reference Match Simulator across the 5 Multiplayer Envs
# ---------------------------------------------------------------------------

# Ground-truth architectural scaling profiles reflecting Society Charter & Rounds 1-9:
# (base_theta_at_B0, scaling_rate_alpha, saturation_B0, non_monotone_oversearch_drag)
_FAMILY_SCALING_PROFILES: Dict[str, Tuple[float, float, float, float]] = {
    "1-pass": (0.00, 0.00, 1.0, 0.00),
    "Flat-MC P3": (0.00, 0.24, 2.0, 0.025),
    "PUCT": (0.02, 0.36, 2.0, 0.008),
    "Hardcoded-Beam P8": (0.04, 0.42, 2.5, 0.004),
    "Learned-KV-Tree": (0.06, 0.56, 1.8, 0.000),
    "Fractal-Prospectus-Tree": (0.08, 0.68, 1.5, 0.000),
    "Multi-Game Generalist": (0.07, 0.64, 1.6, 0.000),
    "Single-Game Specialist": (0.10, 0.61, 1.7, 0.000),
    "LIR-Masked": (0.06, 0.58, 1.6, 0.000),
    "Token-Masked": (0.03, 0.41, 2.0, 0.015),
    "Fused": (-0.05, 0.28, 2.5, 0.045),
}

# Environment-specific draw propensities, first-mover advantages, and skill multipliers
_ENV_PARAMS: Dict[str, Tuple[float, float, float]] = {
    # (skill_discrimination, draw_nu, first_mover_gamma)
    "Hex-5x5": (1.20, 0.01, 0.22),  # Hex has no draws in exact rules, strong first-mover
    "Connect-4": (1.15, 0.18, 0.16),
    "Micro-Stratego": (1.30, 0.24, 0.06),
    "Galanis-Market": (1.25, 0.10, 0.00),
    "Proposal-Poker": (1.35, 0.08, 0.00),
}


def latent_skill_for_agent(env_name: str, spec: AgentSpec) -> float:
    """
    Compute the reference latent log-skill theta(env, family, alg, B) used by
    the calibrated benchmark tournament simulator.
    """
    canon_env = normalize_env_name(env_name)
    base, alpha, b0, drag = _FAMILY_SCALING_PROFILES.get(
        spec.model_family, (0.0, 0.35, 2.0, 0.0)
    )
    disc, _, _ = _ENV_PARAMS.get(canon_env, (1.0, 0.15, 0.0))

    b = float(spec.compute_budget_B)
    log_b = math.log2(1.0 + b / b0) if b > 0 else 0.0

    # Environment-specific bonuses matching theoretical desiderata:
    # - Micro-Stratego rewards episodic memory tokens M_t (Learned-KV-Tree / Fractal-Prospectus-Tree)
    # - Proposal-Poker & Galanis-Market penalize belief drift in Fused / Token-Masked RL
    env_bonus = 0.0
    if canon_env == "Micro-Stratego":
        if spec.model_family in ("Learned-KV-Tree", "Fractal-Prospectus-Tree", "Multi-Game Generalist"):
            env_bonus += 0.12 * math.log2(1.0 + b)
    elif canon_env in ("Proposal-Poker", "Galanis-Market"):
        if spec.model_family in ("LIR-Masked", "Fractal-Prospectus-Tree", "Learned-KV-Tree"):
            env_bonus += 0.15 * (1.0 + 0.25 * math.log2(1.0 + b))
        elif spec.model_family == "Fused":
            env_bonus -= 0.45  # Fused RL sycophancy & belief distortion under private bias y_i
        elif spec.model_family == "Token-Masked":
            env_bonus -= 0.18  # Shared-trunk representation spillover

    theta = disc * (base + alpha * log_b - drag * max(0.0, b - 16.0) / 32.0) + env_bonus
    return float(theta)


def simulate_calibrated_match(
    env_name: str,
    spec_a: AgentSpec,
    spec_b: AgentSpec,
    first_player_sign: int = 0,
    seed: int = 0,
) -> Tuple[float, bool]:
    """
    Simulate a calibrated head-to-head match between `spec_a` and `spec_b` in `env_name`
    using the Bradley-Terry-Davidson generative model.
    """
    canon_env = normalize_env_name(env_name)
    _, draw_nu, fm_gamma = _ENV_PARAMS.get(canon_env, (1.0, 0.15, 0.0))
    theta_a = latent_skill_for_agent(canon_env, spec_a)
    theta_b = latent_skill_for_agent(canon_env, spec_b)
    p_a, p_b, p_d = btd_probabilities(
        theta_a,
        theta_b,
        nu=draw_nu,
        first_mover_gamma=fm_gamma,
        first_player_sign=first_player_sign,
    )
    rng = np.random.default_rng(seed)
    u = float(rng.random())
    if u < float(p_a):
        return 1.0, False
    if u < float(p_a + p_b):
        return 0.0, False
    return 0.5, True
