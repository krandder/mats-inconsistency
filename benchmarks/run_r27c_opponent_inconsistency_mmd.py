#!/usr/bin/env python3
"""
Round 27C Runner: Neural Infoset-Policy Distillation of Self-Referential Opponent
Rationality + Proximity + Prior Inconsistency Minimisation (MMD) & Policy Commitments
on Public OpenSpiel Games.

Strict compliance with docs/PROTOCOL.md (including Section 9, Rules 9.1-9.9) and
outbox/r27-freeze-review-2026-10-20.md:
1. Explicit Mechanism Statement (Blocking Item 1):
   Inconsistency_ProxPrior_Neural distills exact-tree Magnetic Mirror Descent (MMD)
   counterfactual action values Q^{pi_theta}(I, a) (computed by ExtensiveFormInconsistencyOracle
   over all infosets of the game tree) into a shared neural network PureNeuralInfosetPolicy
   taking OpenSpiel's native C++ information_state_tensor(p) with zero per-infoset lookup
   parameters (zero info_logit, zero info_emb). Ceiling_Tabular_MMD_SameStages is reported
   alongside Inconsistency_ProxPrior_Neural at the identical 2-step Mirror-Prox stage count
   and beta_prox = 2.4.
2. Clean Comparators & Rule 9.1 / 9.5 Compliance (Blocking Item 2):
   - Removed the broken 24-iteration DeepCFRSolver comparator (which called set_seed(42)
     internally and used uncalibrated sample traversals).
   - OpenSpiel_Upstream_MCCFR_OurRun is reported with its training_regime as an ungated
     diagnostic per Rule 9.5.
   - Control_Random_Network instantiates the exact PureNeuralInfosetPolicy class with
     random weights (seed + 9000) and the identical head scaling (mul_(0.05)) and runs
     through the identical extract_policy_dict pipeline (Rule 9.1).
   - Control_Untrained_Init extracts pi_init after head weight scaling (mul_(0.05)),
     recording the exact initial policy from which gradient descent starts.
3. Restored Original 928ddec Bars & Objective-Aligned D2 Readability Evaluation (Rule 9.9):
   - D0 verifies deterministic OpenSpiel 2.0.2 CFR & CFR+ curves at [10, 40, 120] to <= 1e-3,
     Rule 9.6 exact oracle root value vs OpenSpiel expected_game_score.policy_value across
     20 random profiles per game to <= 1e-9, and tabular MMD QRE inconsistency <= 1e-8 across
     all 3 games (kuhn_poker, leduc_poker, liars_dice(numdice=1,dice_sides=4)).
   - D1 gates that Inconsistency_ProxPrior_Neural (beta_prox = 2.4) achieves lower NashConv
     (ci95_low > 0) than Control_Random_Network, Control_Untrained_Init, Control_Uniform,
     Control_ZeroParam_PotOdds, and Ablation_Neural_NoProx (beta_prox = 0.0) across all 3
     games, and restores the original 928ddec absolute NashConv bars (kuhn_poker <= 0.05,
     leduc_poker <= 0.95, liars_dice <= 0.12; recording honest FAILs where Leduc/Liar's Dice
     exceed those bars).
   - D2 is an objective-alignment check: training against the reading opponent (alpha_train = 1.0)
     vs myopic exploitation (alpha_train = 0.0) and evaluating both payoff_vs_biased and
     payoff_vs_reader, with the original 928ddec bars restored (nc_a1 <= 0.05, max_drift_a1 <= 0.01).
4. Disjoint Smoke vs. Full Evaluation Seeds & SMOKE_NO_VERDICT (Rule 9.8, 9.9):
   - Self-contained sys.path insertion.
   - Full evaluation uses seeds [101, 102, 103, 104, 105]; --smoke-cpu uses disjoint seeds
     [201, 202, 203] and emits SMOKE_NO_VERDICT on all gates.
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
from open_spiel.python.algorithms import cfr, expected_game_score, exploitability, outcome_sampling_mccfr

T_CRIT_DF4 = 2.7764451051977987
T_CRIT_DF2 = 4.302652729911275
FULL_SEEDS = [101, 102, 103, 104, 105]
SMOKE_SEEDS = [201, 202, 203]

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
    t_val = T_CRIT_DF4 if len(d) >= 5 else T_CRIT_DF2
    return {
        "mean_diff": mean_d,
        "se": se,
        "ci95_low": mean_d - t_val * se,
        "ci95_high": mean_d + t_val * se,
    }


# ---------------------------------------------------------------------------
# 1. Exact Extensive-Form Self-Referential Inconsistency Oracle
# ---------------------------------------------------------------------------

class ExtensiveFormInconsistencyOracle:
    """
    Exact extensive-form game tree evaluator for 2-player zero-sum perfect-recall
    OpenSpiel games (kuhn_poker, leduc_poker, liars_dice(numdice=1,dice_sides=4)).
    Computes exact counterfactual action values Q^pi(I, a) and player reach probabilities
    x_own(I) with per-player root-infoset normalization:
      tau_eff(p)  = tau_base  / |I_p^root|,
      beta_eff(p) = beta_prox / |I_p^root|,
    where |I_p^root| is the number of infosets of player p with no prior action by player p
    (3 for P0 and 6 for P1 on Kuhn; 6 for P0 and 12 for P1 on Leduc; 4 for P0 and 32 for P1 on Liar's Dice).
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

    def evaluate_q_and_reach(self, pi: Dict[str, np.ndarray], tau_base: float) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
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
            ent = -float((p * np.log(p)).sum())
            tau_eff = tau_base / float(self.num_infosets_per_player[self.infos[k]])
            V[k] = float(p @ qs + tau_eff * ent)
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
        Q, x = self.evaluate_q_and_reach(pi, tau_base)
        total = 0.0
        for k in self.keys:
            p = np.clip(pi[k], 1e-12, 1.0)
            p = p / p.sum()
            tau_eff = tau_base / float(self.num_infosets_per_player[self.infos[k]])
            target = np.clip(self.softmax(Q[k], tau_eff), 1e-12, 1.0)
            kl = float((p * np.log(p / target)).sum())
            total += x[k] * tau_eff * kl
        return float(total)

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

    def run_tabular_mmd(self, tau_base: float, beta_prox: float, iters: int) -> Dict[str, np.ndarray]:
        """2-step Mirror-Prox midpoint tabular Magnetic Mirror Descent."""
        pi = self.uniform_policy()
        for _ in range(iters):
            Q1, _ = self.evaluate_q_and_reach(pi, tau_base)
            pi_mid = {}
            for k in self.keys:
                n_ip = float(self.num_infosets_per_player[self.infos[k]])
                tau_eff = tau_base / n_ip
                beta_eff = beta_prox / n_ip
                q_shift = Q1[k] - Q1[k].max()
                log_p = (beta_eff * np.log(np.clip(pi[k], 1e-12, 1.0)) + q_shift) / (tau_eff + beta_eff)
                log_p = log_p - log_p.max()
                p = np.exp(log_p)
                pi_mid[k] = p / p.sum()
            Q2, _ = self.evaluate_q_and_reach(pi_mid, tau_base)
            new_pi = {}
            for k in self.keys:
                n_ip = float(self.num_infosets_per_player[self.infos[k]])
                tau_eff = tau_base / n_ip
                beta_eff = beta_prox / n_ip
                q_shift = Q2[k] - Q2[k].max()
                log_p = (beta_eff * np.log(np.clip(pi[k], 1e-12, 1.0)) + q_shift) / (tau_eff + beta_eff)
                log_p = log_p - log_p.max()
                p = np.exp(log_p)
                new_pi[k] = p / p.sum()
            pi = new_pi
        return pi

    def verify_oracle_vs_openspiel_value(self, n_episodes: int = 20, seed: int = 2700) -> Dict[str, Any]:
        """Rule 9.6 verification of unregularized root value against OpenSpiel expected_game_score.policy_value."""
        rng = np.random.default_rng(seed)
        errs = []
        for _ in range(n_episodes):
            pi = {}
            for k in self.keys:
                w = rng.random(len(self.legal[k])) + 0.05
                pi[k] = w / w.sum()
            Q, _ = self.evaluate_q_and_reach(pi, 1e-12)
            v0 = float(
                sum(
                    float(np.dot(pi[k], Q[k]))
                    for k in self.keys
                    if self.infos[k] == 0 and self.parent[k] is None
                )
            )
            tp = policy_lib.TabularPolicy(self.game)
            for k in self.keys:
                row = tp.state_lookup[k]
                tp.action_probability_array[row, :] = 0.0
                for j, a in enumerate(self.legal[k]):
                    tp.action_probability_array[row, a] = float(pi[k][j])
            ref = float(expected_game_score.policy_value(self.game.new_initial_state(), [tp, tp])[0])
            errs.append(abs(v0 - ref))
        return {"episodes_checked": n_episodes, "max_abs_err": float(max(errs))}


# ---------------------------------------------------------------------------
# 2. Neural Infoset Policy Architecture (ZERO tabular info_logit / info_emb)
# ---------------------------------------------------------------------------

class PureNeuralInfosetPolicy(nn.Module):
    """
    Shared-parameter neural policy pi_theta(a | x_info) mapping OpenSpiel's native C++
    information_state_tensor(p) (in_dim floats: 11 on Kuhn, 30 on Leduc, 15 on Liar's Dice)
    to legal action logits. Zero per-infoset lookup tables (zero info_logit, zero info_emb).
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
    seed: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Control_Random_Network (Rule 9.1): instantiates the exact PureNeuralInfosetPolicy
    architecture with random weights (seed + 9000) and identical head scaling (0.05),
    and runs through the exact extract_policy_dict pipeline.
    """
    feats_t, mask_t, in_dim, max_acts = build_infoset_batch(oracle, device)
    torch.manual_seed(seed + 9000)
    rand_net = PureNeuralInfosetPolicy(in_dim, max_acts, hidden_size=64).to(device)
    with torch.no_grad():
        rand_net.head.weight.mul_(0.05)
        rand_net.head.bias.zero_()
    return extract_policy_dict(rand_net, oracle, feats_t, mask_t)


def train_pure_neural_inconsistency(
    oracle: ExtensiveFormInconsistencyOracle,
    seed: int,
    outer_stages: int,
    inner_grad_steps: int,
    tau_base: float,
    beta_prox: float,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Distills exact-tree MMD stage targets into PureNeuralInfosetPolicy across outer_stages.
    - At each outer stage, evaluates exact counterfactual values Q^{pi_curr} via oracle,
      takes a midpoint proximal mirror step p_mid with neural anchor anc_net, evaluates Q^{pi_mid},
      and forms the stage target p_stage = softmax((beta_prox * log pi_anc + n_ip * Q^{pi_mid}) / (tau_base + beta_prox)).
    - Trains net for inner_grad_steps via cross-entropy against p_stage over all infoset tensors.
    - When beta_prox == 0.0 (Ablation_Neural_NoProx), runs the identical pipeline with beta_prox = 0.0.
    - Returns (pi_init, pi_final), where pi_init is extracted after head scaling (0.05) before stage 0.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    feats_t, mask_t, in_dim, max_acts = build_infoset_batch(oracle, device)
    num_infosets = len(oracle.keys)

    net = PureNeuralInfosetPolicy(in_dim, max_acts, hidden_size=64).to(device)
    with torch.no_grad():
        net.head.weight.mul_(0.05)
        net.head.bias.zero_()
    pi_init = extract_policy_dict(net, oracle, feats_t, mask_t)
    anc_net = copy.deepcopy(net)

    opt = torch.optim.AdamW(net.parameters(), lr=8e-3, weight_decay=1e-4)
    n_ip_map = oracle.num_infosets_per_player

    for _ in range(outer_stages):
        pi_curr = extract_policy_dict(net, oracle, feats_t, mask_t)
        Q1_dict, _ = oracle.evaluate_q_and_reach(pi_curr, tau_base)

        q1_np = np.zeros((num_infosets, max_acts), dtype=np.float32)
        for idx, k in enumerate(oracle.keys):
            n_ip = float(n_ip_map[oracle.infos[k]])
            qv = (Q1_dict[k] - Q1_dict[k].max()) * n_ip
            q1_np[idx, : len(qv)] = qv
        q1_t = torch.tensor(q1_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            log_p_anc = F.log_softmax(anc_net(feats_t, mask_t), dim=-1)
            mid_logits = ((beta_prox * log_p_anc + q1_t) / (tau_base + beta_prox)).masked_fill(mask_t < 0.5, -1e9)
            p_mid_np = F.softmax(mid_logits, dim=-1).cpu().numpy()

        pi_mid = {}
        for idx, k in enumerate(oracle.keys):
            m = len(oracle.legal[k])
            p_m = np.clip(p_mid_np[idx, :m].astype(np.float64), 1e-8, 1.0)
            pi_mid[k] = p_m / p_m.sum()

        Q2_dict, _ = oracle.evaluate_q_and_reach(pi_mid, tau_base)
        q2_np = np.zeros((num_infosets, max_acts), dtype=np.float32)
        for idx, k in enumerate(oracle.keys):
            n_ip = float(n_ip_map[oracle.infos[k]])
            qv = (Q2_dict[k] - Q2_dict[k].max()) * n_ip
            q2_np[idx, : len(qv)] = qv
        q2_t = torch.tensor(q2_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            stage_logits = ((beta_prox * log_p_anc + q2_t) / (tau_base + beta_prox)).masked_fill(mask_t < 0.5, -1e9)
            p_stage = F.softmax(stage_logits, dim=-1)

        net.train()
        for _ in range(inner_grad_steps):
            opt.zero_grad()
            log_probs = F.log_softmax(net(feats_t, mask_t), dim=-1)
            loss = -(p_stage * log_probs * mask_t).sum(dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        anc_net.load_state_dict(net.state_dict())

    pi_final = extract_policy_dict(net, oracle, feats_t, mask_t)
    return pi_init, pi_final


# ---------------------------------------------------------------------------
# 3. Neural Policy-as-Commitment (D2) Under Opponent Readability
# ---------------------------------------------------------------------------

def evaluate_p0_payoff_against(
    kuhn_oracle: ExtensiveFormInconsistencyOracle,
    pi_p0: Dict[str, np.ndarray],
    pi_p1: Dict[str, np.ndarray],
) -> float:
    """Computes Player 0's exact expected game return when P0 plays pi_p0 and P1 plays pi_p1."""
    pi_joint = {k: (pi_p0[k] if kuhn_oracle.infos[k] == 0 else pi_p1[k]) for k in kuhn_oracle.keys}
    Q_eval, _ = kuhn_oracle.evaluate_q_and_reach(pi_joint, tau_base=1e-9)
    # Each root deal ("0", "1", "2") has probability 1/3; Q_eval already includes the 1/6 chance weights
    return float(np.sum([pi_p0[r_k] @ Q_eval[r_k] for r_k in ["0", "1", "2"]]))


def run_neural_policy_commitment_readability(
    kuhn_oracle: ExtensiveFormInconsistencyOracle,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Tests Policy-as-Commitment (Pi_theta) under opponent readability alpha_read in {0.0, 0.5, 1.0}:
      - For each alpha_read, trains a fresh PureNeuralInfosetPolicy(seed) against an opponent
        pi_opp(alpha_read; pi_theta) = (1 - alpha_read) * pi_biased + alpha_read * SoftBR(pi_theta).
      - Evaluates every trained policy against BOTH:
        1. pi_biased (payoff_vs_biased, testing exploitation of the non-reading opponent), and
        2. SoftBR(pi_theta) (payoff_vs_reader, testing commitment robustness when the opponent
           reads pi_theta and plays its soft best response!).
    """
    tau_base = 0.06
    beta_prox = 2.4
    n_ip_map = kuhn_oracle.num_infosets_per_player

    pi_biased = kuhn_oracle.uniform_policy()
    for k in kuhn_oracle.keys:
        if kuhn_oracle.infos[k] == 1:
            if k.startswith("0"):
                pi_biased[k] = np.array([0.25, 0.75])
            elif k.startswith("1"):
                pi_biased[k] = np.array([0.15, 0.85])
            elif k.startswith("2"):
                pi_biased[k] = np.array([0.30, 0.70])

    feats_t, mask_t, in_dim, max_acts = build_infoset_batch(kuhn_oracle, device)
    num_infosets = len(kuhn_oracle.keys)

    readability_results: Dict[str, Any] = {}
    for alpha_read in [0.0, 0.5, 1.0]:
        torch.manual_seed(seed + int(alpha_read * 100))
        net = PureNeuralInfosetPolicy(in_dim, max_acts, hidden_size=64).to(device)
        with torch.no_grad():
            net.head.weight.mul_(0.05)
            net.head.bias.zero_()
        anc_net = copy.deepcopy(net)
        opt = torch.optim.AdamW(net.parameters(), lr=8e-3, weight_decay=1e-4)

        pi_prev_stage = extract_policy_dict(net, kuhn_oracle, feats_t, mask_t)
        for _ in range(32):
            pi_curr = extract_policy_dict(net, kuhn_oracle, feats_t, mask_t)
            Q_self, _ = kuhn_oracle.evaluate_q_and_reach(pi_curr, tau_base)
            pi_joint = {}
            for k in kuhn_oracle.keys:
                if kuhn_oracle.infos[k] == 1:
                    tau_eff_p1 = tau_base / float(n_ip_map[1])
                    p_br = kuhn_oracle.softmax(Q_self[k], tau_eff_p1)
                    p_opp = (1.0 - alpha_read) * pi_biased[k] + alpha_read * p_br
                    pi_joint[k] = p_opp / p_opp.sum()
                else:
                    pi_joint[k] = pi_curr[k]

            Q_against_opp, _ = kuhn_oracle.evaluate_q_and_reach(pi_joint, tau_base)
            q_np = np.zeros((num_infosets, max_acts), dtype=np.float32)
            for idx, k in enumerate(kuhn_oracle.keys):
                p_id = kuhn_oracle.infos[k]
                q_src = Q_against_opp[k] if p_id == 0 else Q_self[k]
                qv = (q_src - q_src.max()) * float(n_ip_map[p_id])
                q_np[idx, : len(qv)] = qv
            q_t = torch.tensor(q_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                log_p_anc = F.log_softmax(anc_net(feats_t, mask_t), dim=-1)
                stage_logits = ((beta_prox * log_p_anc + q_t) / (tau_base + beta_prox)).masked_fill(mask_t < 0.5, -1e9)
                p_stage = F.softmax(stage_logits, dim=-1)

            pi_prev_stage = pi_curr
            net.train()
            for _ in range(18):
                opt.zero_grad()
                log_probs = F.log_softmax(net(feats_t, mask_t), dim=-1)
                loss = -(p_stage * log_probs * mask_t).sum(dim=-1).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            anc_net.load_state_dict(net.state_dict())

        pi_commit = extract_policy_dict(net, kuhn_oracle, feats_t, mask_t)
        max_l1_drift = max(
            float(np.abs(pi_commit[k] - pi_prev_stage[k]).sum()) for k in kuhn_oracle.keys
        )
        nc = kuhn_oracle.nash_conv(pi_commit)

        # Evaluate P0 payoff against (a) fixed pi_biased and (b) reading opponent SoftBR(pi_commit)
        Q_commit, _ = kuhn_oracle.evaluate_q_and_reach(pi_commit, tau_base)
        tau_eff_p1 = tau_base / float(n_ip_map[1])
        pi_reader = {
            k: (kuhn_oracle.softmax(Q_commit[k], tau_eff_p1) if kuhn_oracle.infos[k] == 1 else pi_commit[k])
            for k in kuhn_oracle.keys
        }
        payoff_vs_biased = evaluate_p0_payoff_against(kuhn_oracle, pi_commit, pi_biased)
        payoff_vs_reader = evaluate_p0_payoff_against(kuhn_oracle, pi_commit, pi_reader)

        readability_results[str(alpha_read)] = {
            "nash_conv": nc,
            "payoff_vs_biased": payoff_vs_biased,
            "payoff_vs_reader": payoff_vs_reader,
            "fixed_point_l1_drift": max_l1_drift,
        }

    return {"kuhn_by_readability": readability_results}


# ---------------------------------------------------------------------------
# 4. Upstream OpenSpiel Baseline Curve Verification & Main R27C Loop
# ---------------------------------------------------------------------------

def run_upstream_openspiel_baselines(
    oracle: ExtensiveFormInconsistencyOracle, smoke_cpu: bool, seed: int = 101
) -> Dict[str, Any]:
    game = oracle.game
    results: Dict[str, Any] = {}
    budgets = [10, 40, 120] if smoke_cpu else [10, 40, 120, 400]

    # 1. Upstream deterministic CFRSolver
    cfr_solver = cfr.CFRSolver(game)
    cfr_curve = {}
    for b in range(1, max(budgets) + 1):
        cfr_solver.evaluate_and_update_policy()
        if b in budgets:
            cfr_curve[str(b)] = float(exploitability.nash_conv(game, cfr_solver.average_policy()))
    results["OpenSpiel_Upstream_CFR"] = {
        "curve": cfr_curve,
        "final_nash_conv": cfr_curve[str(budgets[-1])],
    }

    # 2. Upstream deterministic CFRPlusSolver
    cfrp_solver = cfr.CFRPlusSolver(game)
    cfrp_curve = {}
    for b in range(1, max(budgets) + 1):
        cfrp_solver.evaluate_and_update_policy()
        if b in budgets:
            cfrp_curve[str(b)] = float(exploitability.nash_conv(game, cfrp_solver.average_policy()))
    results["OpenSpiel_Upstream_CFRPlus"] = {
        "curve": cfrp_curve,
        "final_nash_conv": cfrp_curve[str(budgets[-1])],
    }

    # 3. Upstream OutcomeSamplingMCCFR (diagnostic per Rule 9.5)
    np.random.seed(seed)
    random.seed(seed)
    mccfr_solver = outcome_sampling_mccfr.OutcomeSamplingSolver(game)
    mccfr_eps = 150 if smoke_cpu else 800
    for _ in range(mccfr_eps):
        mccfr_solver.iteration()
    results["OpenSpiel_Upstream_MCCFR_OurRun"] = {
        "episodes": mccfr_eps,
        "training_regime": f"upstream OutcomeSamplingSolver, {mccfr_eps} episodes (ungated diagnostic per Rule 9.5)",
        "final_nash_conv": float(exploitability.nash_conv(game, mccfr_solver.average_policy())),
    }

    return results


def run_r27c(args) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke_cpu else "cpu")
    seeds = SMOKE_SEEDS if args.smoke_cpu else FULL_SEEDS
    seed_overlap_count = len(set(SMOKE_SEEDS).intersection(set(FULL_SEEDS)))
    games = ["kuhn_poker", "leduc_poker", "liars_dice(numdice=1,dice_sides=4)"]
    tau_base = 0.06
    beta_prox = 2.4
    beta_prox_tabular = 1.2

    oracles = {g: ExtensiveFormInconsistencyOracle(g) for g in games}

    upstream_by_game: Dict[str, Any] = {}
    tabular_mmd_by_game: Dict[str, Any] = {}
    controls_by_game: Dict[str, Any] = {}
    game_structural_stats: Dict[str, Any] = {}
    oracle_value_check_by_game: Dict[str, Any] = {}

    for g in games:
        orc = oracles[g]
        in_dim = len(orc.info_tensors[orc.keys[0]])
        max_acts = max(len(orc.legal[k]) for k in orc.keys)
        game_structural_stats[g] = {
            "num_infosets_total": len(orc.keys),
            "num_root_infosets_per_player": orc.num_infosets_per_player,
            "info_tensor_dim": in_dim,
            "max_legal_actions": max_acts,
            "tau_eff_per_player": {
                str(p): tau_base / float(orc.num_infosets_per_player[p]) for p in (0, 1)
            },
        }
        oracle_value_check_by_game[g] = orc.verify_oracle_vs_openspiel_value(n_episodes=20, seed=2700)
        unif_pi = orc.uniform_policy()
        pot_pi = orc.zero_param_pot_odds_policy()
        controls_by_game[g] = {
            "Control_Uniform": {
                "nash_conv": orc.nash_conv(unif_pi),
                "inconsistency": orc.inconsistency(unif_pi, tau_base),
            },
            "Control_ZeroParam_PotOdds": {
                "nash_conv": orc.nash_conv(pot_pi),
                "inconsistency": orc.inconsistency(pot_pi, tau_base),
            },
        }
        upstream_by_game[g] = run_upstream_openspiel_baselines(orc, args.smoke_cpu, seed=seeds[0])

        b_tab = 0.6 if g == "leduc_poker" else beta_prox_tabular
        mmd_iters = (80 if g == "leduc_poker" else 400) if args.smoke_cpu else (600 if g == "leduc_poker" else 800)
        pi_mmd = orc.run_tabular_mmd(tau_base=tau_base, beta_prox=b_tab, iters=mmd_iters)
        tabular_mmd_by_game[g] = {
            "tau_base": tau_base,
            "beta_prox_tabular": b_tab,
            "iters": mmd_iters,
            "nash_conv": orc.nash_conv(pi_mmd),
            "inconsistency": max(0.0, orc.inconsistency(pi_mmd, tau_base)),
        }

    # Precompute deterministic same-stage 2-step Mirror-Prox tabular MMD ceiling (at beta_prox = 2.4) per game
    same_stage_tabular_by_game: Dict[str, Dict[str, float]] = {}
    for g in games:
        orc = oracles[g]
        outer_stages = (18 if g == "leduc_poker" else 36) if args.smoke_cpu else (32 if g == "leduc_poker" else 48)
        pi_tab_same = orc.run_tabular_mmd(tau_base=tau_base, beta_prox=beta_prox, iters=outer_stages)
        same_stage_tabular_by_game[g] = {
            "mirror_prox_stages": outer_stages,
            "q_eval_steps": outer_stages * 2,
            "beta_prox": beta_prox,
            "nash_conv": orc.nash_conv(pi_tab_same),
            "inconsistency": max(0.0, orc.inconsistency(pi_tab_same, tau_base)),
        }

    per_seed: Dict[str, Any] = {}
    for seed in seeds:
        seed_games: Dict[str, Any] = {}
        for g in games:
            orc = oracles[g]
            outer_stages = (18 if g == "leduc_poker" else 36) if args.smoke_cpu else (32 if g == "leduc_poker" else 48)
            inner_steps = 20 if args.smoke_cpu else 28

            pi_rand = evaluate_random_network_policy(orc, seed=seed, device=device)
            _, pi_noprox = train_pure_neural_inconsistency(
                orc,
                seed=seed,
                outer_stages=outer_stages,
                inner_grad_steps=inner_steps,
                tau_base=tau_base,
                beta_prox=0.0,
                device=device,
            )
            pi_untr, pi_trained = train_pure_neural_inconsistency(
                orc,
                seed=seed,
                outer_stages=outer_stages,
                inner_grad_steps=inner_steps,
                tau_base=tau_base,
                beta_prox=beta_prox,
                device=device,
            )

            # Seed-specific upstream MCCFR diagnostic (ungated per Rule 9.5)
            np.random.seed(seed)
            random.seed(seed)
            mccfr_s = outcome_sampling_mccfr.OutcomeSamplingSolver(orc.game)
            mccfr_eps = 150 if args.smoke_cpu else 800
            for _ in range(mccfr_eps):
                mccfr_s.iteration()
            mccfr_nc = float(exploitability.nash_conv(orc.game, mccfr_s.average_policy()))

            seed_games[g] = {
                "Control_Random_Network": {
                    "nash_conv": orc.nash_conv(pi_rand),
                    "inconsistency": orc.inconsistency(pi_rand, tau_base),
                },
                "Control_Untrained_Init": {
                    "nash_conv": orc.nash_conv(pi_untr),
                    "inconsistency": orc.inconsistency(pi_untr, tau_base),
                },
                "Ablation_Neural_NoProx": {
                    "nash_conv": orc.nash_conv(pi_noprox),
                    "inconsistency": orc.inconsistency(pi_noprox, tau_base),
                },
                "OpenSpiel_Upstream_MCCFR_OurRun": {
                    "nash_conv": mccfr_nc,
                    "training_regime": f"upstream OutcomeSamplingSolver, {mccfr_eps} episodes (ungated diagnostic per Rule 9.5)",
                },
                "Ceiling_Tabular_MMD_SameStages": same_stage_tabular_by_game[g],
                "Inconsistency_ProxPrior_Neural": {
                    "nash_conv": orc.nash_conv(pi_trained),
                    "inconsistency": orc.inconsistency(pi_trained, tau_base),
                },
            }

        seed_games["policy_commitment_readability"] = run_neural_policy_commitment_readability(
            oracles["kuhn_poker"], seed=seed, device=device
        )
        per_seed[str(seed)] = seed_games

    runner_sha256 = compute_sha256(Path(__file__).resolve())
    prereg_path = Path(_REPO_ROOT) / "benchmarks" / "PREREG-R27C-OPPONENT-INCONSISTENCY-MMD.md"
    prereg_sha256 = compute_sha256(prereg_path) if prereg_path.exists() else ""

    results = {
        "round": "R27C",
        "runner_sha256": runner_sha256,
        "prereg_sha256": prereg_sha256,
        "openspiel_info": {
            "version": str(getattr(pyspiel, "__version__", "unknown")),
            "file": str(getattr(pyspiel, "__file__", "unknown")),
        },
        "device": str(device),
        "smoke_cpu": bool(args.smoke_cpu),
        "seeds": seeds,
        "seed_overlap_count": seed_overlap_count,
        "games": games,
        "game_structural_stats": game_structural_stats,
        "oracle_value_check_by_game": oracle_value_check_by_game,
        "tau_base": tau_base,
        "beta_prox": beta_prox,
        "beta_prox_tabular": beta_prox_tabular,
        "controls_by_game": controls_by_game,
        "upstream_openspiel_by_game": upstream_by_game,
        "tabular_mmd_ceiling_by_game": tabular_mmd_by_game,
        "per_seed": per_seed,
    }
    results["gates"] = evaluate_r27c_gates(results)
    return results


def evaluate_r27c_gates(results: Dict[str, Any]) -> Dict[str, Any]:
    is_smoke = bool(results.get("smoke_cpu", False))
    seeds = [str(s) for s in results["seeds"]]
    games = results["games"]
    up = results["upstream_openspiel_by_game"]
    mmd = results["tabular_mmd_ceiling_by_game"]
    ctrls = results["controls_by_game"]
    per_seed = results["per_seed"]
    ov_checks = results.get("oracle_value_check_by_game", {})

    # D0: Deterministic regression check against OpenSpiel 2.0.2 CFR & CFR+ curves at [10, 40, 120]
    # to tolerance 1e-3, Rule 9.6 oracle root value check <= 1e-9, tabular MMD QRE inconsistency <= 1e-8
    # across all 3 games (restored 928ddec bar), and disjoint smoke/full seeds
    max_curve_err = 0.0
    for g_k in ("kuhn_poker", "leduc_poker"):
        for step_k in ("10", "40", "120"):
            err_cfr = abs(
                up[g_k]["OpenSpiel_Upstream_CFR"]["curve"][step_k] - KNOWN_CFR_CURVES[g_k]["cfr"][step_k]
            )
            err_cfrp = abs(
                up[g_k]["OpenSpiel_Upstream_CFRPlus"]["curve"][step_k] - KNOWN_CFR_CURVES[g_k]["cfr_plus"][step_k]
            )
            max_curve_err = max(max_curve_err, err_cfr, err_cfrp)

    kuhn_mmd_inc = float(mmd["kuhn_poker"]["inconsistency"])
    leduc_mmd_inc = float(mmd["leduc_poker"]["inconsistency"])
    liars_mmd_inc = float(mmd["liars_dice(numdice=1,dice_sides=4)"]["inconsistency"])
    max_oracle_val_err = max(
        (float(ov_checks[g]["max_abs_err"]) for g in games if g in ov_checks),
        default=1.0,
    )
    min_oracle_eps = min(
        (int(ov_checks[g]["episodes_checked"]) for g in games if g in ov_checks),
        default=0,
    )
    seed_overlap = int(results.get("seed_overlap_count", 0))

    d0_pass = bool(
        max_curve_err <= 1e-3
        and kuhn_mmd_inc <= 1e-8
        and leduc_mmd_inc <= 1e-8
        and liars_mmd_inc <= 1e-8
        and min_oracle_eps >= 20
        and max_oracle_val_err <= 1e-9
        and seed_overlap == 0
    )

    # D1: Proximity edge prevents cycling & Inconsistency_ProxPrior_Neural beats all 4 controls
    # (Control_Random_Network, Control_Untrained_Init, Control_Uniform, Control_ZeroParam_PotOdds)
    # and Ablation_Neural_NoProx (beta_prox = 0.0) across all 3 games, plus restored 928ddec
    # absolute NashConv bars (kuhn_poker <= 0.05, leduc_poker <= 0.95, liars_dice <= 0.12)
    game_metrics: Dict[str, Any] = {}
    all_games_beat = True
    for g in games:
        ours_nc = [per_seed[s][g]["Inconsistency_ProxPrior_Neural"]["nash_conv"] for s in seeds]
        rand_nc = [per_seed[s][g]["Control_Random_Network"]["nash_conv"] for s in seeds]
        untr_nc = [per_seed[s][g]["Control_Untrained_Init"]["nash_conv"] for s in seeds]
        noprox_nc = [per_seed[s][g]["Ablation_Neural_NoProx"]["nash_conv"] for s in seeds]
        mccfr_nc = [per_seed[s][g]["OpenSpiel_Upstream_MCCFR_OurRun"]["nash_conv"] for s in seeds]
        tab_same_nc = [per_seed[s][g]["Ceiling_Tabular_MMD_SameStages"]["nash_conv"] for s in seeds]
        unif_nc = [ctrls[g]["Control_Uniform"]["nash_conv"] for _ in seeds]
        pot_nc = [ctrls[g]["Control_ZeroParam_PotOdds"]["nash_conv"] for _ in seeds]

        ours_inc = [per_seed[s][g]["Inconsistency_ProxPrior_Neural"]["inconsistency"] for s in seeds]
        noprox_inc = [per_seed[s][g]["Ablation_Neural_NoProx"]["inconsistency"] for s in seeds]

        ci_vs_rand = paired_ci95(rand_nc, ours_nc)
        ci_vs_untr = paired_ci95(untr_nc, ours_nc)
        ci_vs_noprox = paired_ci95(noprox_nc, ours_nc)
        ci_vs_unif = paired_ci95(unif_nc, ours_nc)
        ci_vs_pot = paired_ci95(pot_nc, ours_nc)

        if not (
            ci_vs_rand["ci95_low"] > 0.0
            and ci_vs_untr["ci95_low"] > 0.0
            and ci_vs_noprox["ci95_low"] > 0.0
            and ci_vs_unif["ci95_low"] > 0.0
            and ci_vs_pot["ci95_low"] > 0.0
        ):
            all_games_beat = False

        game_metrics[g] = {
            "mean_nash_conv_ours": float(np.mean(ours_nc)),
            "mean_nash_conv_tabular_same_stages": float(np.mean(tab_same_nc)),
            "mean_nash_conv_random_net": float(np.mean(rand_nc)),
            "mean_nash_conv_untrained": float(np.mean(untr_nc)),
            "mean_nash_conv_noprox": float(np.mean(noprox_nc)),
            "mean_nash_conv_uniform": float(np.mean(unif_nc)),
            "mean_nash_conv_pot_odds": float(np.mean(pot_nc)),
            "mean_nash_conv_mccfr_ourrun": float(np.mean(mccfr_nc)),
            "mean_inconsistency_ours": float(np.mean(ours_inc)),
            "mean_inconsistency_noprox": float(np.mean(noprox_inc)),
            "ci_vs_random_net": ci_vs_rand,
            "ci_vs_untrained": ci_vs_untr,
            "ci_vs_noprox": ci_vs_noprox,
            "ci_vs_uniform": ci_vs_unif,
            "ci_vs_pot_odds": ci_vs_pot,
        }

    kuhn_ours_nc = game_metrics["kuhn_poker"]["mean_nash_conv_ours"]
    leduc_ours_nc = game_metrics["leduc_poker"]["mean_nash_conv_ours"]
    liars_ours_nc = game_metrics["liars_dice(numdice=1,dice_sides=4)"]["mean_nash_conv_ours"]
    d1_pass = bool(
        all_games_beat
        and kuhn_ours_nc <= 0.05
        and leduc_ours_nc <= 0.95
        and liars_ours_nc <= 0.12
    )

    # D2: Objective-aligned Neural Policy-as-Commitment under opponent readability alpha_read in {0.0, 0.5, 1.0}
    # with restored 928ddec bars (mean_nc_a1 <= 0.05 and max_drift_a1 <= 0.01)
    payoff_biased_a0 = [
        per_seed[s]["policy_commitment_readability"]["kuhn_by_readability"]["0.0"]["payoff_vs_biased"] for s in seeds
    ]
    payoff_biased_a1 = [
        per_seed[s]["policy_commitment_readability"]["kuhn_by_readability"]["1.0"]["payoff_vs_biased"] for s in seeds
    ]
    payoff_reader_a0 = [
        per_seed[s]["policy_commitment_readability"]["kuhn_by_readability"]["0.0"]["payoff_vs_reader"] for s in seeds
    ]
    payoff_reader_a1 = [
        per_seed[s]["policy_commitment_readability"]["kuhn_by_readability"]["1.0"]["payoff_vs_reader"] for s in seeds
    ]
    nc_a0 = [
        per_seed[s]["policy_commitment_readability"]["kuhn_by_readability"]["0.0"]["nash_conv"] for s in seeds
    ]
    nc_a1 = [
        per_seed[s]["policy_commitment_readability"]["kuhn_by_readability"]["1.0"]["nash_conv"] for s in seeds
    ]
    drift_a1 = [
        per_seed[s]["policy_commitment_readability"]["kuhn_by_readability"]["1.0"]["fixed_point_l1_drift"] for s in seeds
    ]

    ci_biased_a0_vs_a1 = paired_ci95(payoff_biased_a0, payoff_biased_a1)
    ci_reader_a1_vs_a0 = paired_ci95(payoff_reader_a1, payoff_reader_a0)
    ci_nc_a0_vs_a1 = paired_ci95(nc_a0, nc_a1)

    d2_pass = bool(
        ci_reader_a1_vs_a0["mean_diff"] >= 0.05
        and ci_reader_a1_vs_a0["ci95_low"] > 0.0
        and ci_biased_a0_vs_a1["mean_diff"] >= 0.03
        and ci_biased_a0_vs_a1["ci95_low"] > 0.0
        and ci_nc_a0_vs_a1["mean_diff"] >= 0.05
        and ci_nc_a0_vs_a1["ci95_low"] > 0.0
        and float(np.mean(nc_a1)) <= 0.05
        and float(np.max(drift_a1)) <= 0.01
    )

    def fmt_verdict(passed: bool) -> str:
        if is_smoke:
            return "SMOKE_NO_VERDICT"
        return "PASS" if passed else "FAIL"

    return {
        "R27C_D0_upstream_openspiel_exact_curve_reproduction": {
            "verdict": fmt_verdict(d0_pass),
            "max_cfr_cfrplus_curve_abs_err": max_curve_err,
            "kuhn_tabular_mmd_inconsistency": kuhn_mmd_inc,
            "leduc_tabular_mmd_inconsistency": leduc_mmd_inc,
            "liars_tabular_mmd_inconsistency": liars_mmd_inc,
            "max_oracle_vs_openspiel_value_err": max_oracle_val_err,
            "min_oracle_check_episodes": min_oracle_eps,
            "seed_overlap_count": seed_overlap,
        },
        "R27C_D1_proximity_edge_prevents_cycling_and_beats_baselines": {
            "verdict": fmt_verdict(d1_pass),
            "kuhn_abs_bar_pass": bool(kuhn_ours_nc <= 0.05),
            "leduc_abs_bar_pass": bool(leduc_ours_nc <= 0.95),
            "liars_abs_bar_pass": bool(liars_ours_nc <= 0.12),
            "all_games_beat_controls_and_noprox": bool(all_games_beat),
            "by_game": game_metrics,
        },
        "R27C_D2_neural_policy_commitment_readability_and_stability": {
            "verdict": fmt_verdict(d2_pass),
            "evaluation_type": "objective_alignment_verification",
            "mean_kuhn_payoff_vs_biased_unread_alpha0": float(np.mean(payoff_biased_a0)),
            "mean_kuhn_payoff_vs_biased_readable_alpha1": float(np.mean(payoff_biased_a1)),
            "mean_kuhn_payoff_vs_reader_unread_alpha0": float(np.mean(payoff_reader_a0)),
            "mean_kuhn_payoff_vs_reader_readable_alpha1": float(np.mean(payoff_reader_a1)),
            "mean_kuhn_nash_conv_unread_alpha0": float(np.mean(nc_a0)),
            "mean_kuhn_nash_conv_readable_alpha1": float(np.mean(nc_a1)),
            "max_fixed_point_l1_drift_alpha1": float(np.max(drift_a1)),
            "ci_payoff_vs_reader_alpha1_vs_alpha0": ci_reader_a1_vs_a0,
            "ci_payoff_vs_biased_alpha0_vs_alpha1": ci_biased_a0_vs_a1,
            "ci_nash_conv_alpha0_vs_alpha1": ci_nc_a0_vs_a1,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--smoke-cpu", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    results = run_r27c(args)
    results["elapsed_sec"] = round(time.time() - t0, 2)
    print(json.dumps(results["gates"], indent=2))
    if args.out_dir:
        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "r27c_results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
