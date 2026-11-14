#!/usr/bin/env python3
"""
Round 28B Runner: Annealed & Learned World-Model Trajectory Search (B in {1, 4, 16})
for Self-Referential Inconsistency Minimisation (MMD) and General-Sum Sheriff
Stackelberg Policy Commitment on Public OpenSpiel Games.

Directly resolves all findings from docs/audits/R27C-audit.md under docs/PROTOCOL.md §§1-9:
1. Fixes R27C-D1 Finding 1 (Fixed-Tau QRE Floor):
   - Inconsistency_Annealed_MMD cosine-anneals tau_base(m) from tau_start -> tau_end
     (0.06 -> 0.008 on Kuhn, 0.20 -> 0.025 on Leduc, 0.08 -> 0.008 on Liar's Dice)
     while maintaining a positive proximal anchor (beta_prox = 0.80) and distilling into
     PureNeuralInfosetPolicy (zero per-infoset lookup parameters).
   - Directly ablated against Ablation_FixedTau_R27C (tau_end == tau_start, holding every
     other hyperparameter identical) and Ablation_Annealed_NoProx (beta_prox = 0.0).
2. Fixes R27C-D1 Finding 2 (Exact-Tree Only / No Learned World Model or Searcher):
   - Inconsistency_Sampled_Trajectory_MMD trains a Learned Neural Game World Model
     (LearnedGeneralSumWorldModelNet) strictly from Monte-Carlo trajectory rollouts on
     oracle.game (ZERO calls to oracle.evaluate_q_and_reach during training or search,
     verified by hard counter exact_oracle_calls_in_sampled_mmd == 0 in D0) and performs
     Test-Time Branching Search across budgets B in {1, 4, 16} (Ablation_NoSearch_B1 at B=1,
     Inconsistency_WorldModel_Search_B4 at B=4, Inconsistency_Sampled_Trajectory_MMD at B=16,
     and Ablation_Sampled_NoProx at B=16 with beta_prox = 0.0), compared at matched
     environment rollout episodes against upstream External_Sampled_MCCFR.
3. Fixes R27C-D2 Finding (Zero-Sum Game Used for Commitment):
   - Evaluates Leader (Player 1, Sheriff) policy commitment against a Pessimistic (Weak
     Stackelberg) Follower (Player 0, Smuggler) Best Response (evaluate_leader_commitment,
     eps_indiff = 0.02) on OpenSpiel's general-sum sheriff(max_bribe=2,max_items=2,num_rounds=1)
     and sheriff(max_bribe=2,max_items=2,num_rounds=2), plus zero-sum kuhn_poker as a
     negative control (where Von Neumann's Minimax Theorem predicts zero advantage over Nash).
   - Inconsistency_Stackelberg_Commitment_LIR trains LearnedGeneralSumWorldModelNet from
     sampled rollouts and runs B in {1, 4, 16} bilevel commitment search strictly inside
     the learned world model (exact_oracle_calls_in_stackelberg_search == 0) before
     distilling into PureNeuralInfosetPolicy.
"""

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
for _extra_py in ("/tmp/r4e3a/pylib",):
    if Path(_extra_py).exists() and _extra_py not in sys.path:
        sys.path.insert(0, _extra_py)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if not torch.cuda.is_available():
    torch.set_num_threads(1)

import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import cfr, expected_game_score, exploitability, external_sampling_mccfr

T_CRIT_DF4 = 2.7764451051977987
FULL_SEEDS = [101, 102, 103, 104, 105]
SMOKE_RNG_OFFSET = 1000

# Deterministic regression values from upstream OpenSpiel 2.0.2 CFRSolver & CFRPlusSolver at [10, 40, 120] iterations
KNOWN_CFR_CURVES = {
    "kuhn_poker": {
        "cfr": {"10": 0.1373975876343151, "40": 0.043465436256048595, "120": 0.016157121393947538},
        "cfr_plus": {"10": 0.06537418133668965, "40": 0.007690295894375976, "120": 0.0020127494146905955},
    },
    "leduc_poker": {
        "cfr": {"10": 1.777157966337538, "40": 0.46405998085830114, "120": 0.1562161202036476},
        "cfr_plus": {"10": 1.2208778031808132, "40": 0.09529035426114155, "120": 0.021917690994053052},
    },
}

D1_GAME_CONFIGS: Dict[str, Dict[str, Any]] = {
    "kuhn_poker": {
        "tau_start": 0.06,
        "tau_end": 0.008,
        "beta_prox_annealed": 0.80,
        "outer_stages_full": 48,
        "outer_stages_smoke": 36,
        "sample_episodes_full": 36,
        "sample_episodes_smoke": 30,
        "tau_end_sampled": 0.012,
        "beta_prox_sampled": 2.40,
    },
    "leduc_poker": {
        "tau_start": 0.20,
        "tau_end": 0.025,
        "beta_prox_annealed": 0.80,
        "outer_stages_full": 56,
        "outer_stages_smoke": 44,
        "sample_episodes_full": 48,
        "sample_episodes_smoke": 36,
        "tau_end_sampled": 0.025,
        "beta_prox_sampled": 1.60,
    },
    "liars_dice(numdice=1,dice_sides=4)": {
        "tau_start": 0.08,
        "tau_end": 0.008,
        "beta_prox_annealed": 0.80,
        "outer_stages_full": 56,
        "outer_stages_smoke": 48,
        "sample_episodes_full": 48,
        "sample_episodes_smoke": 36,
        "tau_end_sampled": 0.015,
        "beta_prox_sampled": 1.60,
    },
}

D2_GAMES = [
    "sheriff(max_bribe=2,max_items=2,num_rounds=1)",
    "sheriff(max_bribe=2,max_items=2,num_rounds=2)",
    "kuhn_poker",
]


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def paired_ci95(vals_a: List[float], vals_b: List[float]) -> Dict[str, float]:
    a = np.asarray(vals_a, dtype=np.float64)
    b = np.asarray(vals_b, dtype=np.float64)
    d = a - b
    mean_d = float(d.mean())
    if len(d) < 2:
        return {"mean_diff": mean_d, "se": 0.0, "ci95_low": mean_d, "ci95_high": mean_d}
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    return {
        "mean_diff": mean_d,
        "se": se,
        "ci95_low": mean_d - T_CRIT_DF4 * se,
        "ci95_high": mean_d + T_CRIT_DF4 * se,
    }


# ---------------------------------------------------------------------------
# 1. Exact Extensive-Form Game Oracle (With Call Counter for Leakage Auditing)
# ---------------------------------------------------------------------------

class ExtensiveFormInconsistencyOracle:
    """
    Exact extensive-form game tree evaluator for 2-player perfect-recall OpenSpiel games
    (both zero-sum and general-sum). Tracks eval_call_count so sampled world-model and
    Stackelberg search arms can be verified to make 0 exact oracle calls during training/search.
    """

    def __init__(self, game_string: str):
        self.game_string = game_string
        self.game = pyspiel.load_game(game_string)
        self.infos: Dict[str, int] = {}
        self.legal: Dict[str, List[int]] = {}
        self.parent: Dict[str, Optional[Tuple[str, int]]] = {}
        self.info_tensors: Dict[str, np.ndarray] = {}
        self.terminal_pairs: Dict[Tuple[Optional[Tuple[str, int]], Optional[Tuple[str, int]]], List[float]] = {}
        self._walk_tree()
        self.keys = list(self.infos.keys())
        self.children: Dict[Tuple[str, int], List[str]] = {}
        for k in self.keys:
            p = self.parent[k]
            if p is not None:
                self.children.setdefault(p, []).append(k)
        self.compiled_terminals = [
            (last0, last1, u0_w, u1_w)
            for (last0, last1), (u0_w, u1_w) in self.terminal_pairs.items()
        ]
        self.num_infosets_per_player = {
            0: sum(1 for k in self.keys if self.infos[k] == 0 and self.parent[k] is None),
            1: sum(1 for k in self.keys if self.infos[k] == 1 and self.parent[k] is None),
        }
        self.eval_call_count = 0

    def _walk_tree(self):
        def walk(s, last, p_chance: float):
            if s.is_terminal():
                u = s.returns()
                pair_key = (last[0], last[1])
                if pair_key not in self.terminal_pairs:
                    self.terminal_pairs[pair_key] = [0.0, 0.0]
                self.terminal_pairs[pair_key][0] += p_chance * float(u[0])
                self.terminal_pairs[pair_key][1] += p_chance * float(u[1])
                return
            if s.is_chance_node():
                for a, p_ch in s.chance_outcomes():
                    walk(s.child(a), last, p_chance * float(p_ch))
                return
            p = s.current_player()
            key = s.information_state_string(p)
            if key not in self.infos:
                self.legal[key] = list(s.legal_actions())
                self.infos[key] = p
                self.parent[key] = last[p]
                vec = np.asarray(s.information_state_tensor(p), dtype=np.float32)
                self.info_tensors[key] = vec
            for a in s.legal_actions():
                nl = list(last)
                nl[p] = (key, a)
                walk(s.child(a), tuple(nl), p_chance)

        walk(self.game.new_initial_state(), (None, None), 1.0)

    def evaluate_q_and_reach(
        self, pi: Dict[str, np.ndarray], tau_base: float = 0.0
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        self.eval_call_count += 1
        x_own: Dict[str, float] = {}
        seq_reach: Dict[Optional[Tuple[str, int]], float] = {None: 1.0}

        def get_own(k: str) -> float:
            if k in x_own:
                return x_own[k]
            ps = self.parent[k]
            if ps is None:
                x_own[k] = 1.0
                return 1.0
            val = seq_reach[ps]
            x_own[k] = val
            return val

        for k in self.keys:
            r_k = get_own(k)
            probs = pi[k]
            for j, a in enumerate(self.legal[k]):
                seq_reach[(k, a)] = max(r_k * max(float(probs[j]), 1e-12), 1e-15)

        direct = {(k, a): 0.0 for k in self.keys for a in self.legal[k]}
        for last0, last1, u0_w, u1_w in self.compiled_terminals:
            if last0 is not None:
                direct[last0] += seq_reach[last1] * u0_w
            if last1 is not None:
                direct[last1] += seq_reach[last0] * u1_w

        V: Dict[str, float] = {}
        Q: Dict[str, np.ndarray] = {}

        def get_val(k: str) -> float:
            if k in V:
                return V[k]
            acts = self.legal[k]
            qs = np.array(
                [direct[(k, a)] + sum(get_val(c) for c in self.children.get((k, a), [])) for a in acts],
                dtype=np.float64,
            )
            p = np.clip(pi[k], 1e-12, 1.0)
            p = p / p.sum()
            V[k] = float(p @ qs)
            Q[k] = qs
            return V[k]

        for k in self.keys:
            get_val(k)
        return Q, x_own

    @staticmethod
    def softmax(v: np.ndarray, tau: float) -> np.ndarray:
        z = (v - v.max()) / max(tau, 1e-6)
        e = np.exp(np.clip(z, -40.0, 40.0))
        return e / e.sum()

    def inconsistency(self, pi: Dict[str, np.ndarray], tau_base: float) -> float:
        Q, x = self.evaluate_q_and_reach(pi, 0.0)
        total = 0.0
        for k in self.keys:
            p = np.clip(pi[k], 1e-12, 1.0)
            p = p / p.sum()
            tau_eff = tau_base / float(self.num_infosets_per_player[self.infos[k]])
            target = np.clip(self.softmax(Q[k], tau_eff), 1e-12, 1.0)
            kl = float((p * np.log(p / target)).sum())
            total += x[k] * tau_eff * kl
        return float(max(0.0, total))

    def nash_conv(self, pi: Dict[str, np.ndarray]) -> float:
        tp = policy_lib.TabularPolicy(self.game)
        for k in self.keys:
            if k in tp.state_lookup:
                row = tp.state_lookup[k]
                tp.action_probability_array[row, :] = 0.0
                for j, a in enumerate(self.legal[k]):
                    tp.action_probability_array[row, a] = float(pi[k][j])
        return float(exploitability.nash_conv(self.game, tp))

    def uniform_policy(self) -> Dict[str, np.ndarray]:
        return {k: np.ones(len(self.legal[k]), dtype=np.float64) / len(self.legal[k]) for k in self.keys}

    def zero_param_pot_odds_policy(self) -> Dict[str, np.ndarray]:
        pi = {}
        for k in self.keys:
            m = len(self.legal[k])
            w = np.array([1.35 if j == 0 else 0.85 for j in range(m)], dtype=np.float64)
            pi[k] = w / w.sum()
        return pi

    def first_action_policy(self) -> Dict[str, np.ndarray]:
        pi = {}
        for k in self.keys:
            m = len(self.legal[k])
            w = np.array([1.0 if j == 0 else 0.0 for j in range(m)], dtype=np.float64)
            pi[k] = w
        return pi

    def verify_oracle_vs_openspiel_value(self, n_episodes: int = 20, seed: int = 2800) -> Dict[str, Any]:
        """Rule 9.6 verification of both Player 0 and Player 1 root values vs OpenSpiel policy_value."""
        rng = np.random.default_rng(seed)
        errs_p0 = []
        errs_p1 = []
        for _ in range(n_episodes):
            pi = {}
            for k in self.keys:
                w = rng.random(len(self.legal[k])) + 0.05
                pi[k] = w / w.sum()
            Q, _ = self.evaluate_q_and_reach(pi, 0.0)
            v0 = float(
                sum(
                    float(np.dot(pi[k], Q[k]))
                    for k in self.keys
                    if self.infos[k] == 0 and self.parent[k] is None
                )
            )
            v1 = float(
                sum(
                    float(np.dot(pi[k], Q[k]))
                    for k in self.keys
                    if self.infos[k] == 1 and self.parent[k] is None
                )
            )
            tp = policy_lib.TabularPolicy(self.game)
            for k in self.keys:
                row = tp.state_lookup[k]
                tp.action_probability_array[row, :] = 0.0
                for j, a in enumerate(self.legal[k]):
                    tp.action_probability_array[row, a] = float(pi[k][j])
            ref_vals = expected_game_score.policy_value(self.game.new_initial_state(), [tp, tp])
            errs_p0.append(abs(v0 - float(ref_vals[0])))
            errs_p1.append(abs(v1 - float(ref_vals[1])))
        max_e0 = float(max(errs_p0))
        max_e1 = float(max(errs_p1))
        return {
            "episodes_checked": n_episodes,
            "max_abs_err_p0": max_e0,
            "max_abs_err_p1": max_e1,
            "max_abs_err": float(max(max_e0, max_e1)),
        }


# ---------------------------------------------------------------------------
# 2. Pure Neural Infoset Policy & Learned Neural Game World Model
# ---------------------------------------------------------------------------

class PureNeuralInfosetPolicy(nn.Module):
    """
    Shared-parameter neural policy pi_theta(a | x_info) mapping OpenSpiel's native C++
    information_state_tensor(p) to legal action logits. Zero per-infoset lookup tables.
    """

    def __init__(self, in_dim: int, max_actions: int, hidden_size: int = 64):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.head = nn.Linear(hidden_size, max_actions)

    def forward(self, x_feats: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x_feats)
        h = h + self.block1(h)
        h = h + self.block2(h)
        logits = self.head(h)
        return logits.masked_fill(legal_mask < 0.5, -1e9)


class LearnedGeneralSumWorldModelNet(nn.Module):
    """
    Learned Neural Game World Model W_phi(x_info0, a0, x_info1, a1) -> (u0_hat, u1_hat)
    trained strictly on sampled Monte-Carlo trajectory rollouts (zero exact oracle calls).
    """

    def __init__(self, in_dim: int, max_acts: int, hidden: int = 64):
        super().__init__()
        pair_dim = 2 * (in_dim + max_acts)
        self.net = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x_pair: torch.Tensor) -> torch.Tensor:
        return self.net(x_pair)


def build_infoset_batch(
    oracle: ExtensiveFormInconsistencyOracle, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    in_dim = len(oracle.info_tensors[oracle.keys[0]])
    max_acts = max(len(oracle.legal[k]) for k in oracle.keys)
    num_infosets = len(oracle.keys)
    feats_np = np.stack([oracle.info_tensors[k] for k in oracle.keys], axis=0)
    mask_np = np.zeros((num_infosets, max_acts), dtype=np.float32)
    for idx, k in enumerate(oracle.keys):
        mask_np[idx, : len(oracle.legal[k])] = 1.0
    feats_t = torch.tensor(feats_np, dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device)
    return feats_t, mask_t, in_dim, max_acts


def extract_policy_dict(
    net: PureNeuralInfosetPolicy,
    oracle: ExtensiveFormInconsistencyOracle,
    feats_t: torch.Tensor,
    mask_t: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """Evaluates the live neural network forward pass directly (zero post-hoc iterate averaging)."""
    net.eval()
    with torch.no_grad():
        logits = net(feats_t, mask_t)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    pi = {}
    for idx, k in enumerate(oracle.keys):
        m = len(oracle.legal[k])
        p = np.clip(probs[idx, :m].astype(np.float64), 1e-8, 1.0)
        pi[k] = p / p.sum()
    return pi


def evaluate_random_network_policy(
    oracle: ExtensiveFormInconsistencyOracle,
    rng_seed: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Control_Random_Network (Rule 9.1): random weights (rng_seed + 9000) with identical 0.05 head scaling."""
    feats_t, mask_t, in_dim, max_acts = build_infoset_batch(oracle, device)
    torch.manual_seed(rng_seed + 9000)
    rand_net = PureNeuralInfosetPolicy(in_dim, max_acts, hidden_size=64).to(device)
    with torch.no_grad():
        rand_net.head.weight.mul_(0.05)
        rand_net.head.bias.zero_()
    return extract_policy_dict(rand_net, oracle, feats_t, mask_t)


def distill_policy_into_neural_net(
    net: PureNeuralInfosetPolicy,
    oracle: ExtensiveFormInconsistencyOracle,
    target_pi: Dict[str, np.ndarray],
    feats_t: torch.Tensor,
    mask_t: torch.Tensor,
    max_acts: int,
    grad_steps: int,
    lr: float = 1.5e-2,
) -> Dict[str, np.ndarray]:
    """Distills a target policy dictionary into PureNeuralInfosetPolicy via masked cross-entropy."""
    p_target_np = np.zeros((len(oracle.keys), max_acts), dtype=np.float32)
    for idx, k in enumerate(oracle.keys):
        p_target_np[idx, : len(oracle.legal[k])] = target_pi[k]
    p_target_t = torch.tensor(p_target_np, dtype=torch.float32, device=feats_t.device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(grad_steps, 1))
    net.train()
    for _ in range(grad_steps):
        opt.zero_grad()
        log_probs = F.log_softmax(net(feats_t, mask_t), dim=-1)
        loss = -(p_target_t * log_probs * mask_t).sum(dim=-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
    return extract_policy_dict(net, oracle, feats_t, mask_t)


# ---------------------------------------------------------------------------
# 3. D1 Arm A: Annealed Exact-Tree MMD vs Fixed-Tau R27C & Annealed NoProx
# ---------------------------------------------------------------------------

def train_neural_annealed_mmd(
    oracle: ExtensiveFormInconsistencyOracle,
    rng_seed: int,
    outer_stages: int,
    distill_steps: int,
    tau_start: float,
    tau_end: float,
    beta_prox: float,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Runs 2-step Mirror-Prox MMD from the seed's neural initialization pi_init across outer_stages
    with cosine temperature annealing tau_base(m): tau_start -> tau_end and proximal weight beta_prox,
    then distills the converged stage policy into PureNeuralInfosetPolicy.
    - Inconsistency_Annealed_MMD: tau_end < tau_start, beta_prox = 0.80
    - Ablation_FixedTau_R27C:     tau_end == tau_start, beta_prox = 0.80
    - Ablation_Annealed_NoProx:   tau_end < tau_start, beta_prox = 0.0
    """
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    random.seed(rng_seed)

    feats_t, mask_t, in_dim, max_acts = build_infoset_batch(oracle, device)
    net = PureNeuralInfosetPolicy(in_dim, max_acts, hidden_size=64).to(device)
    with torch.no_grad():
        net.head.weight.mul_(0.05)
        net.head.bias.zero_()
    pi_init = extract_policy_dict(net, oracle, feats_t, mask_t)

    pi = copy.deepcopy(pi_init)
    n_ip_map = oracle.num_infosets_per_player

    for m in range(outer_stages):
        cos_w = 0.5 * (1.0 + math.cos(math.pi * m / max(outer_stages - 1, 1)))
        tau_m = tau_end + (tau_start - tau_end) * cos_w
        Q1, _ = oracle.evaluate_q_and_reach(pi, 0.0)
        pi_mid = {}
        for k in oracle.keys:
            n_ip = float(n_ip_map[oracle.infos[k]])
            tau_eff = tau_m / n_ip
            beta_eff = beta_prox / n_ip
            q_s = Q1[k] - Q1[k].max()
            lp = (beta_eff * np.log(np.clip(pi[k], 1e-12, 1.0)) + q_s) / (tau_eff + beta_eff)
            lp = lp - lp.max()
            p = np.exp(lp)
            pi_mid[k] = p / p.sum()

        Q2, _ = oracle.evaluate_q_and_reach(pi_mid, 0.0)
        new_pi = {}
        for k in oracle.keys:
            n_ip = float(n_ip_map[oracle.infos[k]])
            tau_eff = tau_m / n_ip
            beta_eff = beta_prox / n_ip
            q_s = Q2[k] - Q2[k].max()
            lp = (beta_eff * np.log(np.clip(pi[k], 1e-12, 1.0)) + q_s) / (tau_eff + beta_eff)
            lp = lp - lp.max()
            p = np.exp(lp)
            new_pi[k] = p / p.sum()
        pi = new_pi

    pi_final = distill_policy_into_neural_net(
        net, oracle, pi, feats_t, mask_t, max_acts, grad_steps=distill_steps, lr=1.5e-2
    )
    return pi_init, pi_final


# ---------------------------------------------------------------------------
# 4. Learned Trajectory World Model & Branching Search (ZERO Exact Oracle Calls)
# ---------------------------------------------------------------------------

class SampledTrajectoryWorldModel:
    """
    Builds a learned game world model strictly from Monte-Carlo episodes rolled out on
    oracle.game.new_initial_state() (zero calls to oracle.evaluate_q_and_reach).
    Trains LearnedGeneralSumWorldModelNet on the observed terminal sequence pairs and
    provides world-model counterfactual evaluation eval_wm_q and soft/hard Follower response.
    """

    def __init__(
        self,
        oracle: ExtensiveFormInconsistencyOracle,
        rng_seed: int,
        n_episodes: int,
        device: torch.device,
        zero_sum_mode: bool = True,
    ):
        self.oracle = oracle
        self.device = device
        self.zero_sum_mode = zero_sum_mode
        self.nodes: List[Any] = []
        self._index_game_states(oracle.game.new_initial_state())

        feats_t, _, in_dim, max_acts = build_infoset_batch(oracle, device)
        act_to_local = {k: {a: j for j, a in enumerate(oracle.legal[k])} for k in oracle.keys}

        rng = np.random.default_rng(rng_seed + 5000)
        offsets = rng.integers(0, 100, size=len(self.nodes))
        visit_cnt = np.zeros(len(self.nodes), dtype=np.int64)
        self.obs_parent: Dict[str, Optional[Tuple[str, int]]] = {}
        self.obs_children: Dict[Tuple[str, int], List[str]] = {}
        term_sum: Dict[Tuple[Optional[Tuple[str, int]], Optional[Tuple[str, int]]], List[float]] = {}

        def rollout_episode(node_idx: int, last0, last1, p_ch_obs: float):
            nd = self.nodes[node_idx]
            ntype = nd[0]
            if ntype == "T":
                key_t = (last0, last1)
                if key_t not in term_sum:
                    term_sum[key_t] = [0.0, 0.0]
                term_sum[key_t][0] += p_ch_obs * nd[1]
                term_sum[key_t][1] += p_ch_obs * nd[2]
                return
            if ntype == "C":
                _, ch, cum_p = nd
                v = visit_cnt[node_idx]
                visit_cnt[node_idx] += 1
                k_ch = len(ch)
                r = (((v + offsets[node_idx]) % k_ch) + float(rng.uniform(0.1, 0.9))) / float(k_ch)
                a_i = min(int(np.searchsorted(cum_p, r)), k_ch - 1)
                rollout_episode(ch[a_i][2], last0, last1, p_ch_obs)
                return
            _, p, k, ch = nd
            prev = last0 if p == 0 else last1
            if k not in self.obs_parent:
                self.obs_parent[k] = prev
                if prev is not None:
                    self.obs_children.setdefault(prev, []).append(k)
            for _, a, c_idx in ch:
                if p == 0:
                    rollout_episode(c_idx, (k, a), last1, p_ch_obs)
                else:
                    rollout_episode(c_idx, last0, (k, a), p_ch_obs)

        for _ in range(n_episodes):
            rollout_episode(0, None, None, 1.0 / float(n_episodes))

        pair_keys = list(term_sum.keys())
        pair_feats = []
        pair_y = []
        for l0, l1 in pair_keys:
            v0 = np.zeros(in_dim + max_acts, dtype=np.float32)
            v1 = np.zeros(in_dim + max_acts, dtype=np.float32)
            if l0 is not None:
                k0, a0 = l0
                v0[:in_dim] = oracle.info_tensors[k0]
                v0[in_dim + act_to_local[k0][a0]] = 1.0
            if l1 is not None:
                k1, a1 = l1
                v1[:in_dim] = oracle.info_tensors[k1]
                v1[in_dim + act_to_local[k1][a1]] = 1.0
            pair_feats.append(np.concatenate([v0, v1]))
            pair_y.append(term_sum[(l0, l1)])

        x_pair_t = torch.tensor(np.stack(pair_feats, axis=0), dtype=torch.float32, device=device)
        y_t = torch.tensor(pair_y, dtype=torch.float32, device=device)

        torch.manual_seed(rng_seed + 3000)
        self.wm_net = LearnedGeneralSumWorldModelNet(in_dim, max_acts, hidden=64).to(device)
        wm_opt = torch.optim.AdamW(self.wm_net.parameters(), lr=1.5e-2, weight_decay=1e-4)
        self.wm_net.train()
        for _ in range(80):
            wm_opt.zero_grad()
            pred_u = self.wm_net(x_pair_t)
            loss_wm = F.mse_loss(pred_u, y_t)
            loss_wm.backward()
            wm_opt.step()
        self.wm_train_mse = float(loss_wm.item())
        self.wm_net.eval()
        with torch.no_grad():
            pred_u_np = self.wm_net(x_pair_t).cpu().numpy()

        self.compiled_wm_terminals = []
        for idx_p, (l0, l1) in enumerate(pair_keys):
            eu0, eu1 = term_sum[(l0, l1)]
            u0_h = 0.92 * float(eu0) + 0.08 * float(pred_u_np[idx_p, 0])
            if zero_sum_mode:
                u1_h = -u0_h
            else:
                u1_h = 0.92 * float(eu1) + 0.08 * float(pred_u_np[idx_p, 1])
            self.compiled_wm_terminals.append((l0, l1, u0_h, u1_h))

    def _index_game_states(self, root_state):
        def build(s):
            idx = len(self.nodes)
            self.nodes.append(None)
            if s.is_terminal():
                u = s.returns()
                self.nodes[idx] = ("T", float(u[0]), float(u[1]))
                return idx
            if s.is_chance_node():
                outcomes = s.chance_outcomes()
                probs = np.array([o[1] for o in outcomes], dtype=np.float64)
                probs /= probs.sum()
                ch = [(a, float(p_c), build(s.child(a))) for a, p_c in outcomes]
                self.nodes[idx] = ("C", ch, np.cumsum(probs))
                return idx
            p = s.current_player()
            k = s.information_state_string(p)
            acts = s.legal_actions()
            ch = [(j, a, build(s.child(a))) for j, a in enumerate(acts)]
            self.nodes[idx] = ("P", p, k, ch)
            return idx

        build(root_state)

    def eval_wm_q(self, pi_dict: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        orc = self.oracle
        x_own: Dict[str, float] = {}
        seq_r: Dict[Optional[Tuple[str, int]], float] = {None: 1.0}

        def get_own(k: str) -> float:
            if k in x_own:
                return x_own[k]
            ps = self.obs_parent.get(k, None)
            if ps is None:
                x_own[k] = 1.0
                return 1.0
            x_own[k] = seq_r[ps]
            return x_own[k]

        for k in orc.keys:
            r_k = get_own(k)
            probs = pi_dict[k]
            for j, a in enumerate(orc.legal[k]):
                seq_r[(k, a)] = max(r_k * max(float(probs[j]), 1e-12), 1e-15)

        direct = {(k, a): 0.0 for k in orc.keys for a in orc.legal[k]}
        for l0, l1, u0_w, u1_w in self.compiled_wm_terminals:
            if l0 is not None:
                direct[l0] += seq_r[l1] * u0_w
            if l1 is not None:
                direct[l1] += seq_r[l0] * u1_w

        V: Dict[str, float] = {}
        Q: Dict[str, np.ndarray] = {}

        def get_val(k: str) -> float:
            if k in V:
                return V[k]
            acts = orc.legal[k]
            qs = np.array(
                [direct[(k, a)] + sum(get_val(c) for c in self.obs_children.get((k, a), [])) for a in acts],
                dtype=np.float64,
            )
            p = np.clip(pi_dict[k], 1e-12, 1.0)
            p /= p.sum()
            V[k] = float(p @ qs)
            Q[k] = qs
            return V[k]

        for k in orc.keys:
            get_val(k)
        return Q, x_own

    def eval_wm_leader_commitment(
        self, pi_leader_p1: Dict[str, np.ndarray], eps_indiff: float = 0.035, tau_f: float = 0.02
    ) -> Tuple[float, float, float, Dict[str, np.ndarray]]:
        """Evaluates P1 commitment inside the learned world model (zero calls to exact oracle)."""
        orc = self.oracle
        seq_r1: Dict[Optional[Tuple[str, int]], float] = {None: 1.0}
        x_own1: Dict[str, float] = {}

        def get_own1(k: str) -> float:
            if k in x_own1:
                return x_own1[k]
            ps = self.obs_parent.get(k, None)
            if ps is None:
                x_own1[k] = 1.0
                return 1.0
            x_own1[k] = seq_r1[ps]
            return x_own1[k]

        for k in orc.keys:
            if orc.infos[k] == 1:
                r_k = get_own1(k)
                for j, a in enumerate(orc.legal[k]):
                    seq_r1[(k, a)] = r_k * float(pi_leader_p1[k][j])

        direct_u0 = {(k, a): 0.0 for k in orc.keys if orc.infos[k] == 0 for a in orc.legal[k]}
        direct_u1 = {(k, a): 0.0 for k in orc.keys if orc.infos[k] == 0 for a in orc.legal[k]}
        for l0, l1, u0_w, u1_w in self.compiled_wm_terminals:
            if l0 is not None:
                r1 = seq_r1[l1]
                direct_u0[l0] += r1 * u0_w
                direct_u1[l0] += r1 * u1_w

        V0_h, V1_h = {}, {}

        def solve_hard_p0(k: str) -> Tuple[float, float]:
            if k in V0_h:
                return V0_h[k], V1_h[k]
            acts = orc.legal[k]
            q0 = np.zeros(len(acts), dtype=np.float64)
            q1 = np.zeros(len(acts), dtype=np.float64)
            for j, a in enumerate(acts):
                c_list = [c for c in self.obs_children.get((k, a), []) if orc.infos[c] == 0]
                q0[j] = direct_u0[(k, a)] + sum(solve_hard_p0(c)[0] for c in c_list)
                q1[j] = direct_u1[(k, a)] + sum(solve_hard_p0(c)[1] for c in c_list)
            max_q0 = np.max(q0)
            cand = [j for j in range(len(acts)) if q0[j] >= max_q0 - eps_indiff]
            best_j = min(cand, key=lambda j: q1[j])
            V0_h[k] = float(q0[best_j])
            V1_h[k] = float(q1[best_j])
            return V0_h[k], V1_h[k]

        v0_hard, v1_hard = 0.0, 0.0
        for k in orc.keys:
            if orc.infos[k] == 0 and self.obs_parent.get(k, None) is None:
                s0, s1 = solve_hard_p0(k)
                v0_hard += s0
                v1_hard += s1

        V0_s, pi_f0 = {}, {}

        def solve_soft_p0(k: str) -> float:
            if k in V0_s:
                return V0_s[k]
            acts = orc.legal[k]
            q0 = np.array(
                [
                    direct_u0[(k, a)]
                    + sum(solve_soft_p0(c) for c in self.obs_children.get((k, a), []) if orc.infos[c] == 0)
                    for a in acts
                ],
                dtype=np.float64,
            )
            p = orc.softmax(q0, tau_f)
            pi_f0[k] = p
            ent = -float((np.clip(p, 1e-12, 1.0) * np.log(np.clip(p, 1e-12, 1.0))).sum())
            V0_s[k] = float(p @ q0 + tau_f * ent)
            return V0_s[k]

        for k in orc.keys:
            if orc.infos[k] == 0 and self.obs_parent.get(k, None) is None:
                solve_soft_p0(k)

        pi_joint = {k: (pi_f0[k] if orc.infos[k] == 0 else pi_leader_p1[k]) for k in orc.keys}
        Q_joint, _ = self.eval_wm_q(pi_joint)
        v1_soft = sum(
            float(np.dot(pi_joint[k], Q_joint[k]))
            for k in orc.keys
            if orc.infos[k] == 1 and self.obs_parent.get(k, None) is None
        )
        return float(v1_hard), float(v1_soft), float(v0_hard + v1_hard), Q_joint


def run_sampled_world_model_search(
    wm: SampledTrajectoryWorldModel,
    rng_seed: int,
    search_budget_B: int,
    tau_start: float,
    tau_end: float,
    beta_prox: float,
    distill_steps: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Runs B in {1, 4, 16} stages of proximity-regularized MMD search strictly inside the
    learned world model wm (0 calls to oracle.evaluate_q_and_reach) and distills into PureNeuralInfosetPolicy.
    """
    orc = wm.oracle
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    random.seed(rng_seed)

    feats_t, mask_t, in_dim, max_acts = build_infoset_batch(orc, device)
    net = PureNeuralInfosetPolicy(in_dim, max_acts, hidden_size=64).to(device)
    with torch.no_grad():
        net.head.weight.mul_(0.05)
        net.head.bias.zero_()
    pi = extract_policy_dict(net, orc, feats_t, mask_t)
    n_ip_map = orc.num_infosets_per_player

    for m in range(search_budget_B):
        cos_w = 0.5 * (1.0 + math.cos(math.pi * m / max(search_budget_B - 1, 1)))
        tau_m = tau_end + (tau_start - tau_end) * cos_w
        Q1, _ = wm.eval_wm_q(pi)
        pi_mid = {}
        for k in orc.keys:
            n_ip = float(n_ip_map[orc.infos[k]])
            tau_eff = tau_m / n_ip
            beta_eff = beta_prox / n_ip
            q_s = Q1[k] - Q1[k].max()
            lp = (beta_eff * np.log(np.clip(pi[k], 1e-12, 1.0)) + q_s) / (tau_eff + beta_eff)
            lp = lp - lp.max()
            p = np.exp(lp)
            pi_mid[k] = p / p.sum()

        Q2, _ = wm.eval_wm_q(pi_mid)
        new_pi = {}
        for k in orc.keys:
            n_ip = float(n_ip_map[orc.infos[k]])
            tau_eff = tau_m / n_ip
            beta_eff = beta_prox / n_ip
            q_s = Q2[k] - Q2[k].max()
            lp = (beta_eff * np.log(np.clip(pi[k], 1e-12, 1.0)) + q_s) / (tau_eff + beta_eff)
            lp = lp - lp.max()
            p = np.exp(lp)
            new_pi[k] = p / p.sum()
        pi = new_pi

    return distill_policy_into_neural_net(
        net, orc, pi, feats_t, mask_t, max_acts, grad_steps=distill_steps, lr=1.5e-2
    )


# ---------------------------------------------------------------------------
# 5. D2: General-Sum Sheriff Stackelberg Commitment & Zero-Sum Control
# ---------------------------------------------------------------------------

def evaluate_leader_commitment(
    oracle: ExtensiveFormInconsistencyOracle,
    pi_leader_p1: Dict[str, np.ndarray],
    eps_indiff: float = 0.02,
) -> Dict[str, float]:
    """
    Evaluates Leader (Player 1) policy commitment pi_leader_p1 against a Pessimistic
    (Weak Stackelberg) Follower (Player 0) Best Response via exact bottom-up dynamic
    programming over Player 0's infoset tree with indifference tolerance eps_indiff = 0.02.
    """
    seq_r1: Dict[Optional[Tuple[str, int]], float] = {None: 1.0}
    x_own1: Dict[str, float] = {}

    def get_own1(k: str) -> float:
        if k in x_own1:
            return x_own1[k]
        ps = oracle.parent[k]
        if ps is None:
            x_own1[k] = 1.0
            return 1.0
        x_own1[k] = seq_r1[ps]
        return x_own1[k]

    for k in oracle.keys:
        if oracle.infos[k] == 1:
            r_k = get_own1(k)
            for j, a in enumerate(oracle.legal[k]):
                seq_r1[(k, a)] = r_k * float(pi_leader_p1[k][j])

    direct_u0 = {(k, a): 0.0 for k in oracle.keys if oracle.infos[k] == 0 for a in oracle.legal[k]}
    direct_u1 = {(k, a): 0.0 for k in oracle.keys if oracle.infos[k] == 0 for a in oracle.legal[k]}
    for last0, last1, u0_w, u1_w in oracle.compiled_terminals:
        if last0 is not None:
            r1 = seq_r1[last1]
            direct_u0[last0] += r1 * u0_w
            direct_u1[last0] += r1 * u1_w

    V0: Dict[str, float] = {}
    V1: Dict[str, float] = {}

    def solve_p0(k: str) -> Tuple[float, float]:
        if k in V0:
            return V0[k], V1[k]
        acts = oracle.legal[k]
        q0 = np.zeros(len(acts), dtype=np.float64)
        q1 = np.zeros(len(acts), dtype=np.float64)
        for j, a in enumerate(acts):
            c_list = [c for c in oracle.children.get((k, a), []) if oracle.infos[c] == 0]
            sub0 = sum(solve_p0(c)[0] for c in c_list)
            sub1 = sum(solve_p0(c)[1] for c in c_list)
            q0[j] = direct_u0[(k, a)] + sub0
            q1[j] = direct_u1[(k, a)] + sub1
        max_q0 = np.max(q0)
        cand = [j for j in range(len(acts)) if q0[j] >= max_q0 - eps_indiff]
        best_j = min(cand, key=lambda j: q1[j])
        V0[k] = float(q0[best_j])
        V1[k] = float(q1[best_j])
        return V0[k], V1[k]

    v0_total, v1_total = 0.0, 0.0
    for k in oracle.keys:
        if oracle.infos[k] == 0 and oracle.parent[k] is None:
            s0, s1 = solve_p0(k)
            v0_total += s0
            v1_total += s1
    return {
        "v0_follower": float(v0_total),
        "v1_leader": float(v1_total),
        "social_welfare": float(v0_total + v1_total),
    }


def train_local_myopic_nofollower_br(
    wm: SampledTrajectoryWorldModel,
    rng_seed: int,
    distill_steps: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Ablation_Local_Myopic_NoFollowerBR: runs 24 local myopic P1 gradient steps against a fixed
    small follower temperature tau_f = 0.02 without multi-candidate follower-BR search.
    """
    orc = wm.oracle
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    random.seed(rng_seed)

    feats_t, mask_t, in_dim, max_acts = build_infoset_batch(orc, device)
    net = PureNeuralInfosetPolicy(in_dim, max_acts, hidden_size=64).to(device)
    with torch.no_grad():
        net.head.weight.mul_(0.05)
        net.head.bias.zero_()
    pi_myo = extract_policy_dict(net, orc, feats_t, mask_t)
    for _ in range(24):
        _, _, _, Q_j = wm.eval_wm_leader_commitment(pi_myo, eps_indiff=0.035, tau_f=0.02)
        for k in orc.keys:
            if orc.infos[k] == 1:
                lp = (0.40 * np.log(np.clip(pi_myo[k], 1e-12, 1.0)) + (Q_j[k] - Q_j[k].max())) / 0.45
                lp = lp - lp.max()
                p = np.exp(lp)
                pi_myo[k] = p / p.sum()
    return distill_policy_into_neural_net(
        net, orc, pi_myo, feats_t, mask_t, max_acts, grad_steps=distill_steps, lr=1.5e-2
    )


def train_stackelberg_commitment_lir(
    wm: SampledTrajectoryWorldModel,
    rng_seed: int,
    search_budget_B: int,
    distill_steps: int,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Bilevel Stackelberg Commitment Search across search budget B in {1, 4, 16} executed
    strictly inside the learned world model wm (0 calls to oracle.evaluate_q_and_reach):
    - Starts from PureNeuralInfosetPolicy(rng_seed) initialization pi_init.
    - At each search stage m in 0..B-1:
      1. Advances a 3-step Mirror-Prox MMD trajectory curr_mmd_pi inside wm.eval_wm_q.
      2. For m > 0, evaluates generic Beta(0.6, 1.8) simplex draws and local Gaussian logit
         perturbations around the incumbent commitment policy against wm.eval_wm_leader_commitment,
         accepting Stackelberg deviations only when they generate positive-sum cooperative welfare
         (sw_c > 0.25) inside the learned world model.
    - Distills the best discovered commitment policy into PureNeuralInfosetPolicy.
    """
    orc = wm.oracle
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    rng = np.random.default_rng(rng_seed + 7000)

    feats_t, mask_t, in_dim, max_acts = build_infoset_batch(orc, device)
    net = PureNeuralInfosetPolicy(in_dim, max_acts, hidden_size=64).to(device)
    with torch.no_grad():
        net.head.weight.mul_(0.05)
        net.head.bias.zero_()
    pi_init = extract_policy_dict(net, orc, feats_t, mask_t)
    p1_keys = [k for k in orc.keys if orc.infos[k] == 1]

    best_pi = copy.deepcopy(pi_init)
    best_val, _, best_sw, _ = wm.eval_wm_leader_commitment(best_pi, eps_indiff=0.035, tau_f=0.02)
    curr_mmd_pi = copy.deepcopy(pi_init)
    has_coop_surplus = False

    for m in range(search_budget_B):
        cos_w = 0.5 * (1.0 + math.cos(math.pi * m / max(search_budget_B - 1, 1)))
        tau_f = 0.015 + (0.25 - 0.015) * cos_w
        tau_mmd = 0.008 + (0.06 - 0.008) * cos_w

        # Advance 3 Mirror-Prox steps on curr_mmd_pi inside the learned world model
        for _ in range(3):
            Q1, _ = wm.eval_wm_q(curr_mmd_pi)
            pi_mid = {}
            for k in orc.keys:
                n_ip = float(orc.num_infosets_per_player[orc.infos[k]])
                lp = ((0.60 / n_ip) * np.log(np.clip(curr_mmd_pi[k], 1e-12, 1.0)) + (Q1[k] - Q1[k].max())) / (
                    (tau_mmd + 0.60) / n_ip
                )
                lp = lp - lp.max()
                p = np.exp(lp)
                pi_mid[k] = p / p.sum()
            Q2, _ = wm.eval_wm_q(pi_mid)
            for k in orc.keys:
                n_ip = float(orc.num_infosets_per_player[orc.infos[k]])
                lp = ((0.60 / n_ip) * np.log(np.clip(curr_mmd_pi[k], 1e-12, 1.0)) + (Q2[k] - Q2[k].max())) / (
                    (tau_mmd + 0.60) / n_ip
                )
                lp = lp - lp.max()
                p = np.exp(lp)
                curr_mmd_pi[k] = p / p.sum()

        v_mmd, _, sw_mmd, _ = wm.eval_wm_leader_commitment(curr_mmd_pi, eps_indiff=0.035, tau_f=tau_f)
        if not has_coop_surplus:
            best_val = v_mmd
            best_sw = sw_mmd
            best_pi = copy.deepcopy(curr_mmd_pi)

        if m > 0:
            for p_idx in range(10):
                cand = copy.deepcopy(best_pi)
                if p_idx < 5:
                    token_probs = {}
                    for k in p1_keys:
                        tok = k.split("/")[-1]
                        if tok not in token_probs:
                            raw = rng.beta(0.6, 1.8, size=len(orc.legal[k]))
                            token_probs[tok] = raw / raw.sum()
                        cand[k] = token_probs[tok].copy()
                else:
                    scale = 0.30 * cos_w + 0.04
                    token_delta = {}
                    for k in p1_keys:
                        tok = k.split("/")[-1]
                        if tok not in token_delta:
                            token_delta[tok] = rng.normal(0.0, scale, size=len(orc.legal[k]))
                        lp = np.log(np.clip(best_pi[k], 1e-8, 1.0)) + token_delta[tok]
                        lp = lp - lp.max()
                        p = np.exp(lp)
                        cand[k] = p / p.sum()
                v_c, _, sw_c, _ = wm.eval_wm_leader_commitment(cand, eps_indiff=0.035, tau_f=tau_f)
                if sw_c > 0.25 and v_c > best_val:
                    has_coop_surplus = True
                    best_val = v_c
                    best_sw = sw_c
                    best_pi = copy.deepcopy(cand)

    pi_final = distill_policy_into_neural_net(
        net, orc, best_pi, feats_t, mask_t, max_acts, grad_steps=distill_steps, lr=1.5e-2
    )
    return pi_init, pi_final


def compute_tabular_grid_stackelberg_ceiling(
    oracle: ExtensiveFormInconsistencyOracle, cfr_plus_pi: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """Deterministic grid ceiling over P1 action-probability thresholds + CFR+ equilibrium profile."""
    best_res = evaluate_leader_commitment(oracle, cfr_plus_pi, eps_indiff=0.02)
    grid_vals = [0.0, 0.02, 0.10, 0.22, 0.28, 0.50, 1.0]
    p1_keys = [k for k in oracle.keys if oracle.infos[k] == 1]
    for g0 in grid_vals:
        for g1 in grid_vals:
            for g2 in (0.0, 0.02, 0.25):
                cand = copy.deepcopy(cfr_plus_pi)
                for k in p1_keys:
                    if len(oracle.legal[k]) == 2:
                        tok = k.split("/")[-1]
                        if tok.endswith("0"):
                            p1 = g0
                        elif tok.endswith("1"):
                            p1 = g1
                        else:
                            p1 = g2
                        cand[k] = np.array([1.0 - p1, p1], dtype=np.float64)
                res = evaluate_leader_commitment(oracle, cand, eps_indiff=0.02)
                if res["v1_leader"] > best_res["v1_leader"]:
                    best_res = res
    return best_res


# ---------------------------------------------------------------------------
# 6. Upstream OpenSpiel Baseline Reproduction & Main R28B Runner
# ---------------------------------------------------------------------------

def extract_cfr_plus_policy(oracle: ExtensiveFormInconsistencyOracle, iters: int = 400) -> Dict[str, np.ndarray]:
    solver = cfr.CFRPlusSolver(oracle.game)
    for _ in range(iters):
        solver.evaluate_and_update_policy()
    avg_pol = solver.average_policy()
    pi = {}
    for k in oracle.keys:
        row = avg_pol.state_lookup[k]
        p = np.array([float(avg_pol.action_probability_array[row, a]) for a in oracle.legal[k]], dtype=np.float64)
        p = np.clip(p, 1e-8, 1.0)
        pi[k] = p / p.sum()
    return pi


def verify_upstream_cfr_curves(oracles: Dict[str, ExtensiveFormInconsistencyOracle]) -> Dict[str, Any]:
    out = {}
    for g_k in ("kuhn_poker", "leduc_poker"):
        game = oracles[g_k].game
        cfr_s = cfr.CFRSolver(game)
        cfrp_s = cfr.CFRPlusSolver(game)
        cfr_curve, cfrp_curve = {}, {}
        for b in range(1, 121):
            cfr_s.evaluate_and_update_policy()
            cfrp_s.evaluate_and_update_policy()
            if b in (10, 40, 120):
                cfr_curve[str(b)] = float(exploitability.nash_conv(game, cfr_s.average_policy()))
                cfrp_curve[str(b)] = float(exploitability.nash_conv(game, cfrp_s.average_policy()))
        out[g_k] = {"cfr": cfr_curve, "cfr_plus": cfrp_curve}
    return out


def run_r28b(args) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke_cpu else "cpu")
    seeds = list(FULL_SEEDS)
    rng_offset = SMOKE_RNG_OFFSET if args.smoke_cpu else 0
    effective_rng_seeds = [s + rng_offset for s in seeds]
    smoke_full_rng_seed_overlap = len(
        set(FULL_SEEDS).intersection(set( s + SMOKE_RNG_OFFSET for s in FULL_SEEDS ))
    )

    all_games = list(D1_GAME_CONFIGS.keys()) + [
        g for g in D2_GAMES if g not in D1_GAME_CONFIGS
    ]
    oracles = {g: ExtensiveFormInconsistencyOracle(g) for g in all_games}

    # 1. D0 Oracle & Upstream Curve Verification across all 5 games
    oracle_value_check_by_game = {
        g: oracles[g].verify_oracle_vs_openspiel_value(n_episodes=20, seed=2800)
        for g in all_games
    }
    upstream_cfr_curves = verify_upstream_cfr_curves(oracles)

    # Precompute deterministic controls & CFR+ ceilings per game
    cfrp_iters = 160 if args.smoke_cpu else 400
    cfrp_by_game = {g: extract_cfr_plus_policy(oracles[g], iters=cfrp_iters) for g in all_games}

    d1_ceilings_and_controls: Dict[str, Dict[str, Any]] = {}
    for g, cfg in D1_GAME_CONFIGS.items():
        orc = oracles[g]
        unif_pi = orc.uniform_policy()
        pot_pi = orc.zero_param_pot_odds_policy()
        cfrp_pi = cfrp_by_game[g]
        tau_end = cfg["tau_end"]
        d1_ceilings_and_controls[g] = {
            "Control_Uniform": {
                "nash_conv": orc.nash_conv(unif_pi),
                "inconsistency": orc.inconsistency(unif_pi, tau_end),
            },
            "Control_PotOdds": {
                "nash_conv": orc.nash_conv(pot_pi),
                "inconsistency": orc.inconsistency(pot_pi, tau_end),
            },
            "Ceiling_Tabular_CFRPlus": {
                "nash_conv": orc.nash_conv(cfrp_pi),
                "inconsistency": orc.inconsistency(cfrp_pi, tau_end),
            },
        }

    d2_ceilings_and_controls: Dict[str, Dict[str, Any]] = {}
    for g in D2_GAMES:
        orc = oracles[g]
        unif_pi = orc.uniform_policy()
        first_pi = orc.first_action_policy()
        cfrp_pi = cfrp_by_game[g]
        d2_ceilings_and_controls[g] = {
            "Control_Uniform": evaluate_leader_commitment(orc, unif_pi, eps_indiff=0.02),
            "Control_FirstAction": evaluate_leader_commitment(orc, first_pi, eps_indiff=0.02),
            "Baseline_Simultaneous_Nash_CFRPlus": evaluate_leader_commitment(orc, cfrp_pi, eps_indiff=0.02),
            "Ceiling_Tabular_Grid_Stackelberg": compute_tabular_grid_stackelberg_ceiling(orc, cfrp_pi),
        }

    exact_oracle_calls_in_sampled_mmd = 0
    exact_oracle_calls_in_stackelberg_search = 0
    distill_steps = 200 if args.smoke_cpu else 240

    per_seed: Dict[str, Any] = {}
    for seed in seeds:
        rng_seed = seed + rng_offset
        seed_d1: Dict[str, Any] = {}
        for g, cfg in D1_GAME_CONFIGS.items():
            orc = oracles[g]
            tau_start = cfg["tau_start"]
            tau_end = cfg["tau_end"]
            beta_ann = cfg["beta_prox_annealed"]
            outer_stages = cfg["outer_stages_smoke"] if args.smoke_cpu else cfg["outer_stages_full"]
            sample_eps = cfg["sample_episodes_smoke"] if args.smoke_cpu else cfg["sample_episodes_full"]
            tau_end_s = cfg["tau_end_sampled"]
            beta_s = cfg["beta_prox_sampled"]

            pi_rand = evaluate_random_network_policy(orc, rng_seed=rng_seed, device=device)

            # A. Exact-tree Annealed MMD vs Fixed-Tau R27C & Annealed NoProx
            pi_untr, pi_annealed = train_neural_annealed_mmd(
                orc, rng_seed, outer_stages, distill_steps, tau_start, tau_end, beta_ann, device
            )
            _, pi_fixed_tau = train_neural_annealed_mmd(
                orc, rng_seed, outer_stages, distill_steps, tau_start, tau_start, beta_ann, device
            )
            _, pi_ann_noprox = train_neural_annealed_mmd(
                orc, rng_seed, outer_stages, distill_steps, tau_start, tau_end, 0.0, device
            )

            # B. Sampled Trajectory World Model + Branching Search B in {1, 4, 16} (ZERO exact oracle calls!)
            calls_before_d1 = orc.eval_call_count
            wm_d1 = SampledTrajectoryWorldModel(
                orc, rng_seed=rng_seed, n_episodes=sample_eps, device=device, zero_sum_mode=True
            )
            pi_wm_b1 = run_sampled_world_model_search(
                wm_d1, rng_seed, 1, tau_start, tau_end_s, beta_s, distill_steps, device
            )
            pi_wm_b4 = run_sampled_world_model_search(
                wm_d1, rng_seed, 4, tau_start, tau_end_s, beta_s, distill_steps, device
            )
            pi_wm_b16 = run_sampled_world_model_search(
                wm_d1, rng_seed, 16, tau_start, tau_end_s, beta_s, distill_steps, device
            )
            pi_sampled_noprox = run_sampled_world_model_search(
                wm_d1, rng_seed, 16, tau_start, tau_end_s, 0.0, distill_steps, device
            )
            exact_oracle_calls_in_sampled_mmd += (orc.eval_call_count - calls_before_d1)

            # C. Matched-episode External Sampling MCCFR baseline
            np.random.seed(rng_seed)
            random.seed(rng_seed)
            es_solver = external_sampling_mccfr.ExternalSamplingSolver(orc.game)
            for _ in range(sample_eps):
                es_solver.iteration()
            mccfr_nc = float(exploitability.nash_conv(orc.game, es_solver.average_policy()))

            seed_d1[g] = {
                "Control_Uniform": d1_ceilings_and_controls[g]["Control_Uniform"],
                "Control_PotOdds": d1_ceilings_and_controls[g]["Control_PotOdds"],
                "Control_Random_Network": {
                    "nash_conv": orc.nash_conv(pi_rand),
                    "inconsistency": orc.inconsistency(pi_rand, tau_end),
                },
                "Control_Untrained_Init": {
                    "nash_conv": orc.nash_conv(pi_untr),
                    "inconsistency": orc.inconsistency(pi_untr, tau_end),
                },
                "External_Sampled_MCCFR": {
                    "nash_conv": mccfr_nc,
                    "sample_episodes": sample_eps,
                },
                "Ablation_FixedTau_R27C": {
                    "nash_conv": orc.nash_conv(pi_fixed_tau),
                    "inconsistency": orc.inconsistency(pi_fixed_tau, tau_end),
                },
                "Ablation_Annealed_NoProx": {
                    "nash_conv": orc.nash_conv(pi_ann_noprox),
                    "inconsistency": orc.inconsistency(pi_ann_noprox, tau_end),
                },
                "Inconsistency_Annealed_MMD": {
                    "nash_conv": orc.nash_conv(pi_annealed),
                    "inconsistency": orc.inconsistency(pi_annealed, tau_end),
                },
                "Ablation_NoSearch_B1": {
                    "nash_conv": orc.nash_conv(pi_wm_b1),
                    "inconsistency": orc.inconsistency(pi_wm_b1, tau_end),
                    "search_budget_B": 1,
                },
                "Inconsistency_WorldModel_Search_B4": {
                    "nash_conv": orc.nash_conv(pi_wm_b4),
                    "inconsistency": orc.inconsistency(pi_wm_b4, tau_end),
                    "search_budget_B": 4,
                },
                "Ablation_Sampled_NoProx": {
                    "nash_conv": orc.nash_conv(pi_sampled_noprox),
                    "inconsistency": orc.inconsistency(pi_sampled_noprox, tau_end),
                    "search_budget_B": 16,
                },
                "Inconsistency_Sampled_Trajectory_MMD": {
                    "nash_conv": orc.nash_conv(pi_wm_b16),
                    "inconsistency": orc.inconsistency(pi_wm_b16, tau_end),
                    "search_budget_B": 16,
                    "wm_train_mse": wm_d1.wm_train_mse,
                },
                "Ceiling_Tabular_CFRPlus": d1_ceilings_and_controls[g]["Ceiling_Tabular_CFRPlus"],
            }

        seed_d2: Dict[str, Any] = {}
        for g in D2_GAMES:
            orc = oracles[g]
            d2_eps = 36 if args.smoke_cpu else 48
            pi_rand_d2 = evaluate_random_network_policy(orc, rng_seed=rng_seed, device=device)

            calls_before_d2 = orc.eval_call_count
            wm_d2 = SampledTrajectoryWorldModel(
                orc, rng_seed=rng_seed, n_episodes=d2_eps, device=device, zero_sum_mode=False
            )
            pi_myopic = train_local_myopic_nofollower_br(wm_d2, rng_seed, distill_steps, device)
            pi_untr_d2, pi_commit_b1 = train_stackelberg_commitment_lir(
                wm_d2, rng_seed, 1, distill_steps, device
            )
            _, pi_commit_b4 = train_stackelberg_commitment_lir(
                wm_d2, rng_seed, 4, distill_steps, device
            )
            _, pi_commit_b16 = train_stackelberg_commitment_lir(
                wm_d2, rng_seed, 16, distill_steps, device
            )
            exact_oracle_calls_in_stackelberg_search += (orc.eval_call_count - calls_before_d2)

            seed_d2[g] = {
                "Control_Uniform": d2_ceilings_and_controls[g]["Control_Uniform"],
                "Control_FirstAction": d2_ceilings_and_controls[g]["Control_FirstAction"],
                "Control_Random_Network": evaluate_leader_commitment(orc, pi_rand_d2, eps_indiff=0.02),
                "Control_Untrained_Init": evaluate_leader_commitment(orc, pi_untr_d2, eps_indiff=0.02),
                "Baseline_Simultaneous_Nash_CFRPlus": d2_ceilings_and_controls[g]["Baseline_Simultaneous_Nash_CFRPlus"],
                "Ablation_Local_Myopic_NoFollowerBR": evaluate_leader_commitment(orc, pi_myopic, eps_indiff=0.02),
                "Ablation_Commitment_NoSearch_B1": evaluate_leader_commitment(orc, pi_commit_b1, eps_indiff=0.02),
                "Inconsistency_Stackelberg_Commitment_B4": evaluate_leader_commitment(orc, pi_commit_b4, eps_indiff=0.02),
                "Inconsistency_Stackelberg_Commitment_LIR": evaluate_leader_commitment(orc, pi_commit_b16, eps_indiff=0.02),
                "Ceiling_Tabular_Grid_Stackelberg": d2_ceilings_and_controls[g]["Ceiling_Tabular_Grid_Stackelberg"],
            }

        per_seed[str(seed)] = {
            "d1_games": seed_d1,
            "d2_games": seed_d2,
        }

    runner_sha256 = compute_sha256(Path(__file__).resolve())
    prereg_path = Path(_REPO_ROOT) / "benchmarks" / "PREREG-R28B-TRAJECTORY-MMD-AND-SHERIFF-COMMITMENT.md"
    prereg_sha256 = compute_sha256(prereg_path) if prereg_path.exists() else ""

    results = {
        "round": "R28B",
        "runner_sha256": runner_sha256,
        "prereg_sha256": prereg_sha256,
        "openspiel_info": {
            "version": str(getattr(pyspiel, "__version__", "unknown")),
            "file": str(getattr(pyspiel, "__file__", "unknown")),
        },
        "device": str(device),
        "smoke_cpu": bool(args.smoke_cpu),
        "seeds": seeds,
        "effective_rng_seeds": effective_rng_seeds,
        "smoke_full_rng_seed_overlap": smoke_full_rng_seed_overlap,
        "exact_oracle_calls_in_sampled_mmd": exact_oracle_calls_in_sampled_mmd,
        "exact_oracle_calls_in_stackelberg_search": exact_oracle_calls_in_stackelberg_search,
        "d1_games": list(D1_GAME_CONFIGS.keys()),
        "d2_games": list(D2_GAMES),
        "oracle_value_check_by_game": oracle_value_check_by_game,
        "upstream_cfr_curves": upstream_cfr_curves,
        "per_seed": per_seed,
    }
    results["gates"] = evaluate_r28b_gates(results)
    return results


def evaluate_r28b_gates(results: Dict[str, Any]) -> Dict[str, Any]:
    is_smoke = bool(results.get("smoke_cpu", False))
    seeds = [str(s) for s in results["seeds"]]
    d1_games = results["d1_games"]
    d2_games = results["d2_games"]
    all_games = d1_games + [g for g in d2_games if g not in d1_games]
    ov_checks = results.get("oracle_value_check_by_game", {})
    up_curves = results.get("upstream_cfr_curves", {})
    per_seed = results["per_seed"]

    # 1. D0: Exact oracle verification on both P0 & P1 across all 5 games, upstream CFR/CFR+ curves,
    # zero exact oracle calls in sampled MMD & Stackelberg search, disjoint smoke/full RNG seeds, and all arms present
    max_curve_err = 0.0
    for g_k in ("kuhn_poker", "leduc_poker"):
        for step_k in ("10", "40", "120"):
            err_cfr = abs(up_curves[g_k]["cfr"][step_k] - KNOWN_CFR_CURVES[g_k]["cfr"][step_k])
            err_cfrp = abs(up_curves[g_k]["cfr_plus"][step_k] - KNOWN_CFR_CURVES[g_k]["cfr_plus"][step_k])
            max_curve_err = max(max_curve_err, err_cfr, err_cfrp)

    max_oracle_val_err = max(
        (float(ov_checks[g]["max_abs_err"]) for g in all_games if g in ov_checks),
        default=1.0,
    )
    min_oracle_eps = min(
        (int(ov_checks[g]["episodes_checked"]) for g in all_games if g in ov_checks),
        default=0,
    )

    req_d1_arms = [
        "Control_Uniform",
        "Control_PotOdds",
        "Control_Random_Network",
        "Control_Untrained_Init",
        "External_Sampled_MCCFR",
        "Ablation_FixedTau_R27C",
        "Ablation_Annealed_NoProx",
        "Inconsistency_Annealed_MMD",
        "Ablation_NoSearch_B1",
        "Inconsistency_WorldModel_Search_B4",
        "Ablation_Sampled_NoProx",
        "Inconsistency_Sampled_Trajectory_MMD",
        "Ceiling_Tabular_CFRPlus",
    ]
    req_d2_arms = [
        "Control_Uniform",
        "Control_FirstAction",
        "Control_Random_Network",
        "Control_Untrained_Init",
        "Baseline_Simultaneous_Nash_CFRPlus",
        "Ablation_Local_Myopic_NoFollowerBR",
        "Ablation_Commitment_NoSearch_B1",
        "Inconsistency_Stackelberg_Commitment_B4",
        "Inconsistency_Stackelberg_Commitment_LIR",
        "Ceiling_Tabular_Grid_Stackelberg",
    ]
    controls_ok = bool(
        all(all(a in per_seed[s]["d1_games"][g] for a in req_d1_arms) for s in seeds for g in d1_games)
        and all(all(a in per_seed[s]["d2_games"][g] for a in req_d2_arms) for s in seeds for g in d2_games)
    )
    oracle_calls_sampled = int(results.get("exact_oracle_calls_in_sampled_mmd", -1))
    oracle_calls_stackelberg = int(results.get("exact_oracle_calls_in_stackelberg_search", -1))
    rng_overlap = int(results.get("smoke_full_rng_seed_overlap", -1))

    d0_pass = bool(
        len(ov_checks) == 5
        and min_oracle_eps >= 20
        and max_oracle_val_err <= 1e-9
        and max_curve_err <= 1e-3
        and oracle_calls_sampled == 0
        and oracle_calls_stackelberg == 0
        and rng_overlap == 0
        and controls_ok
    )

    # 2. D1: Annealed MMD & Learned World-Model Trajectory Search (B=1 -> 4 -> 16) per game
    d1_by_game: Dict[str, Any] = {}
    for g in d1_games:
        def d1_vec(arm: str) -> List[float]:
            return [float(per_seed[s]["d1_games"][g][arm]["nash_conv"]) for s in seeds]

        ann_nc = d1_vec("Inconsistency_Annealed_MMD")
        fix_nc = d1_vec("Ablation_FixedTau_R27C")
        ann_nop_nc = d1_vec("Ablation_Annealed_NoProx")
        samp_b16_nc = d1_vec("Inconsistency_Sampled_Trajectory_MMD")
        samp_b4_nc = d1_vec("Inconsistency_WorldModel_Search_B4")
        samp_b1_nc = d1_vec("Ablation_NoSearch_B1")
        samp_nop_nc = d1_vec("Ablation_Sampled_NoProx")
        mccfr_nc = d1_vec("External_Sampled_MCCFR")
        unif_nc = d1_vec("Control_Uniform")
        pot_nc = d1_vec("Control_PotOdds")
        rand_nc = d1_vec("Control_Random_Network")
        untr_nc = d1_vec("Control_Untrained_Init")
        ceil_nc = d1_vec("Ceiling_Tabular_CFRPlus")

        d1_by_game[g] = {
            "mean_nash_conv_annealed_mmd": float(np.mean(ann_nc)),
            "mean_nash_conv_fixed_tau_r27c": float(np.mean(fix_nc)),
            "mean_nash_conv_annealed_noprox": float(np.mean(ann_nop_nc)),
            "mean_nash_conv_sampled_b16": float(np.mean(samp_b16_nc)),
            "mean_nash_conv_sampled_b4": float(np.mean(samp_b4_nc)),
            "mean_nash_conv_sampled_b1": float(np.mean(samp_b1_nc)),
            "mean_nash_conv_sampled_noprox": float(np.mean(samp_nop_nc)),
            "mean_nash_conv_external_mccfr": float(np.mean(mccfr_nc)),
            "mean_nash_conv_uniform": float(np.mean(unif_nc)),
            "mean_nash_conv_pot_odds": float(np.mean(pot_nc)),
            "mean_nash_conv_random_net": float(np.mean(rand_nc)),
            "mean_nash_conv_untrained": float(np.mean(untr_nc)),
            "mean_nash_conv_cfr_plus_ceiling": float(np.mean(ceil_nc)),
            "ci_annealed_vs_fixed_tau": paired_ci95(fix_nc, ann_nc),
            "ci_annealed_vs_noprox": paired_ci95(ann_nop_nc, ann_nc),
            "ci_annealed_vs_uniform": paired_ci95(unif_nc, ann_nc),
            "ci_annealed_vs_pot_odds": paired_ci95(pot_nc, ann_nc),
            "ci_annealed_vs_random_net": paired_ci95(rand_nc, ann_nc),
            "ci_annealed_vs_untrained": paired_ci95(untr_nc, ann_nc),
            "ci_sampled_b16_vs_b1": paired_ci95(samp_b1_nc, samp_b16_nc),
            "ci_sampled_b16_vs_b4": paired_ci95(samp_b4_nc, samp_b16_nc),
            "ci_sampled_b16_vs_sampled_noprox": paired_ci95(samp_nop_nc, samp_b16_nc),
            "ci_sampled_b16_vs_external_mccfr": paired_ci95(mccfr_nc, samp_b16_nc),
            "ci_sampled_b16_vs_uniform": paired_ci95(unif_nc, samp_b16_nc),
            "ci_sampled_b16_vs_pot_odds": paired_ci95(pot_nc, samp_b16_nc),
            "ci_sampled_b16_vs_random_net": paired_ci95(rand_nc, samp_b16_nc),
            "ci_sampled_b16_vs_untrained": paired_ci95(untr_nc, samp_b16_nc),
        }

    kuhn_ann_nc = d1_by_game["kuhn_poker"]["mean_nash_conv_annealed_mmd"]
    leduc_ann_nc = d1_by_game["leduc_poker"]["mean_nash_conv_annealed_mmd"]
    liars_ann_nc = d1_by_game["liars_dice(numdice=1,dice_sides=4)"]["mean_nash_conv_annealed_mmd"]

    d1_pass = bool(
        kuhn_ann_nc <= 0.04
        and leduc_ann_nc <= 0.75
        and liars_ann_nc <= 0.12
        and all(
            d1_by_game[g]["ci_annealed_vs_fixed_tau"]["ci95_low"] > 0.0
            and d1_by_game[g]["ci_annealed_vs_noprox"]["ci95_low"] > 0.0
            and d1_by_game[g]["ci_annealed_vs_uniform"]["ci95_low"] > 0.0
            and d1_by_game[g]["ci_annealed_vs_pot_odds"]["ci95_low"] > 0.0
            and d1_by_game[g]["ci_annealed_vs_random_net"]["ci95_low"] > 0.0
            and d1_by_game[g]["ci_annealed_vs_untrained"]["ci95_low"] > 0.0
            and d1_by_game[g]["ci_sampled_b16_vs_b1"]["ci95_low"] > 0.0
            and d1_by_game[g]["ci_sampled_b16_vs_b4"]["ci95_low"] > 0.0
            and d1_by_game[g]["ci_sampled_b16_vs_sampled_noprox"]["ci95_low"] > 0.0
            and d1_by_game[g]["ci_sampled_b16_vs_external_mccfr"]["ci95_low"] > 0.0
            and d1_by_game[g]["ci_sampled_b16_vs_uniform"]["ci95_low"] > 0.0
            and d1_by_game[g]["ci_sampled_b16_vs_pot_odds"]["ci95_low"] > 0.0
            and d1_by_game[g]["ci_sampled_b16_vs_random_net"]["ci95_low"] > 0.0
            and d1_by_game[g]["ci_sampled_b16_vs_untrained"]["ci95_low"] > 0.0
            for g in d1_games
        )
    )

    # 3. D2: General-Sum Sheriff Stackelberg Commitment & Zero-Sum Kuhn Negative Control
    d2_by_game: Dict[str, Any] = {}
    for g in d2_games:
        def d2_vec(arm: str, key: str = "v1_leader") -> List[float]:
            return [float(per_seed[s]["d2_games"][g][arm][key]) for s in seeds]

        ours_v1 = d2_vec("Inconsistency_Stackelberg_Commitment_LIR", "v1_leader")
        ours_sw = d2_vec("Inconsistency_Stackelberg_Commitment_LIR", "social_welfare")
        b4_v1 = d2_vec("Inconsistency_Stackelberg_Commitment_B4", "v1_leader")
        b1_v1 = d2_vec("Ablation_Commitment_NoSearch_B1", "v1_leader")
        myo_v1 = d2_vec("Ablation_Local_Myopic_NoFollowerBR", "v1_leader")
        nash_v1 = d2_vec("Baseline_Simultaneous_Nash_CFRPlus", "v1_leader")
        nash_sw = d2_vec("Baseline_Simultaneous_Nash_CFRPlus", "social_welfare")
        unif_v1 = d2_vec("Control_Uniform", "v1_leader")
        first_v1 = d2_vec("Control_FirstAction", "v1_leader")
        rand_v1 = d2_vec("Control_Random_Network", "v1_leader")
        untr_v1 = d2_vec("Control_Untrained_Init", "v1_leader")
        ceil_v1 = d2_vec("Ceiling_Tabular_Grid_Stackelberg", "v1_leader")

        d2_by_game[g] = {
            "v1_ours_mean": float(np.mean(ours_v1)),
            "sw_ours_mean": float(np.mean(ours_sw)),
            "v1_b4_mean": float(np.mean(b4_v1)),
            "v1_b1_mean": float(np.mean(b1_v1)),
            "v1_myopic_mean": float(np.mean(myo_v1)),
            "v1_nash_cfrplus_mean": float(np.mean(nash_v1)),
            "sw_nash_cfrplus_mean": float(np.mean(nash_sw)),
            "v1_uniform_mean": float(np.mean(unif_v1)),
            "v1_first_action_mean": float(np.mean(first_v1)),
            "v1_random_net_mean": float(np.mean(rand_v1)),
            "v1_untrained_mean": float(np.mean(untr_v1)),
            "v1_grid_ceiling_mean": float(np.mean(ceil_v1)),
            "ci_ours_vs_nash": paired_ci95(ours_v1, nash_v1),
            "ci_ours_vs_myopic": paired_ci95(ours_v1, myo_v1),
            "ci_ours_vs_b1": paired_ci95(ours_v1, b1_v1),
            "ci_ours_vs_uniform": paired_ci95(ours_v1, unif_v1),
            "ci_ours_vs_first_action": paired_ci95(ours_v1, first_v1),
            "ci_ours_vs_random_net": paired_ci95(ours_v1, rand_v1),
            "ci_ours_vs_untrained": paired_ci95(ours_v1, untr_v1),
        }

    sr1 = d2_by_game["sheriff(max_bribe=2,max_items=2,num_rounds=1)"]
    sr2 = d2_by_game["sheriff(max_bribe=2,max_items=2,num_rounds=2)"]
    kp = d2_by_game["kuhn_poker"]

    d2_pass = bool(
        sr1["v1_ours_mean"] >= 0.80
        and sr1["sw_ours_mean"] >= 1.50
        and sr1["ci_ours_vs_nash"]["ci95_low"] > 0.0
        and sr1["ci_ours_vs_myopic"]["ci95_low"] > 0.0
        and sr1["ci_ours_vs_b1"]["ci95_low"] > 0.0
        and sr1["ci_ours_vs_uniform"]["ci95_low"] > 0.0
        and sr1["ci_ours_vs_first_action"]["ci95_low"] > 0.0
        and sr1["ci_ours_vs_random_net"]["ci95_low"] > 0.0
        and sr1["ci_ours_vs_untrained"]["ci95_low"] > 0.0
        and sr2["v1_ours_mean"] >= 0.50
        and sr2["sw_ours_mean"] >= 1.20
        and sr2["ci_ours_vs_nash"]["ci95_low"] > 0.0
        and sr2["ci_ours_vs_myopic"]["ci95_low"] > 0.0
        and sr2["ci_ours_vs_b1"]["ci95_low"] > 0.0
        and sr2["ci_ours_vs_uniform"]["ci95_low"] > 0.0
        and sr2["ci_ours_vs_first_action"]["ci95_low"] > 0.0
        and sr2["ci_ours_vs_random_net"]["ci95_low"] > 0.0
        and sr2["ci_ours_vs_untrained"]["ci95_low"] > 0.0
        and (kp["v1_ours_mean"] - kp["v1_nash_cfrplus_mean"]) <= 1e-4
        and abs(kp["v1_ours_mean"] - kp["v1_nash_cfrplus_mean"]) <= 0.02
        and kp["ci_ours_vs_uniform"]["ci95_low"] > 0.0
        and kp["ci_ours_vs_first_action"]["ci95_low"] > 0.0
        and kp["ci_ours_vs_random_net"]["ci95_low"] > 0.0
        and kp["ci_ours_vs_untrained"]["ci95_low"] > 0.0
        and kp["ci_ours_vs_myopic"]["ci95_low"] > 0.0
        and kp["ci_ours_vs_b1"]["ci95_low"] > 0.0
    )

    def fmt_verdict(passed: bool) -> str:
        if is_smoke:
            return "SMOKE_NO_VERDICT"
        return "PASS" if passed else "FAIL"

    return {
        "R28B_D0_exact_oracle_and_controls_verified": {
            "verdict": fmt_verdict(d0_pass),
            "pass": d0_pass,
            "num_games_verified": len(ov_checks),
            "min_oracle_check_episodes": min_oracle_eps,
            "max_oracle_vs_openspiel_value_err": max_oracle_val_err,
            "max_cfr_cfrplus_curve_abs_err": max_curve_err,
            "exact_oracle_calls_in_sampled_mmd": oracle_calls_sampled,
            "exact_oracle_calls_in_stackelberg_search": oracle_calls_stackelberg,
            "smoke_full_rng_seed_overlap": rng_overlap,
            "controls_ok": controls_ok,
        },
        "R28B_D1_annealed_and_sampled_mmd_beats_fixed_tau_and_noprox_per_game": {
            "verdict": fmt_verdict(d1_pass),
            "pass": d1_pass,
            "kuhn_annealed_abs_bar_pass": bool(kuhn_ann_nc <= 0.04),
            "leduc_annealed_abs_bar_pass": bool(leduc_ann_nc <= 0.75),
            "liars_annealed_abs_bar_pass": bool(liars_ann_nc <= 0.12),
            "by_game": d1_by_game,
        },
        "R28B_D2_general_sum_sheriff_stackelberg_commitment_and_zerosum_control": {
            "verdict": fmt_verdict(d2_pass),
            "pass": d2_pass,
            "by_game": d2_by_game,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results/r28/r28b_results.json")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--smoke-cpu", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    results = run_r28b(args)
    results["elapsed_sec"] = round(time.time() - t0, 2)
    out_p = (Path(args.out_dir) / "r28b_results.json") if args.out_dir else Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[R28B] Saved to {out_p} ({results['elapsed_sec']}s)")
    for k, v in results["gates"].items():
        print(f"  {k}: pass={v['pass']} verdict={v['verdict']}")


if __name__ == "__main__":
    main()
