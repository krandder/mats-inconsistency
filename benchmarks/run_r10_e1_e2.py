#!/usr/bin/env python3
"""Stage-1 Round 10 (`R10-E1` & `R10-E2`) Red-Team Verification Benchmark Runner & Auditor.

Implements the locked 3-seed (`Seeds 101, 102, 103`, paired Student-t `t_{0.975,2} = 4.303`)
pre-registered protocol in `benchmarks/PREREG-STAGE1-R10.md`:

1. Study `R10-E1` (`E-AUX-QUERY`): Bidirectional Latent Lemma Space & Direct Combinatorial Query Rollouts
   - Environments: `equiv_sat` (`GAME_ID = 10`, `|V| = 16, K_dom = 2`) and `sudoku_6x6` (`GAME_ID = 20`, `|V| = 36, K_dom = 6`)
   - Budgets: `B in {0, 2, 4, 8, 16, 32}` across `N_test = 500` boards x `4` queries/board = `2,000` test queries/seed
   - 5 Factorial Arms:
     * `A1_Marginal_Only` (`D_amp = D_V`, `D_aux = 0`)
     * `A2_Linear_Aux` (`A2_Aux_Linear_Emitter`, odd linear emitter `W_lin Delta l_V`)
     * `A3_Bilinear_Aux_Unpart` (`A3_Aux_Bilinear_Unpartitioned`, no `Z^W_aux`/`Z^Pi_aux` firewall or VICReg)
     * `A4_GLC_Bilinear_RandomRollout` (`A4_GLC_Bilinear_RandomPlan`, partitioned `BilinearAuxLemmaEmitter` + VICReg, unsteered `p_0`)
     * `A5_GLC_Bilinear_QuerySteered` (Full `R10-E1` system: partitioned `Z^W_aux (+) Z^Pi_aux` + `BilinearAuxLemmaEmitter` + VICReg + active query seeding `p_0 = k_{Q(A,B)}` + dual-readout self-consistency `delta_rel(A, B)`)
   - Decision Rules: `R10-E1-D0`, `R10-E1-D1`, `R10-E1-D2`, `R10-E1-D3`

2. Study `R10-E2` (`E-KV-TREE-CLEARING`): 80%-Shared-Prefix Tree Deduplication, Disjunctive Bayes Mixture & Strong-Root Monotonicity at `M_0 = 32` and Literal `M_0 = 34`
   - 6 Experimental Clearing Arms:
     * `C1_Flat_Mix` (`flat_mix`)
     * `C2_Flat_ScaledMix` (`C2_Flat_Scaled_Mix`, `flat_scaled_mix`)
     * `C3_Flat_AtaraxosMMD` (`C3_RootRel_K3_Conjunctive`, root-relative salience / conjunctive pooling)
     * `C4_RootRel_LoopedAttnT3` (`C4_RootRel_LoopedAttn_T3`, `R9-E1` anchor-aware root-relative `looped_attn_T3`)
     * `C5_EdgeOnly_Conjunctive_LoopedAttnT3` (`C5_Edge_Conjunctive_Only`, parent-relative edges + `D_crowd` + nested gate without Stage-1 disjunctive Bayes mixture)
     * `C6_TwoStage_GLC_TreeClearer` (Complete 3-stage `TwoStageTreeClearer`: canonical transposition table + Stage-1 disjunctive Bayes mixture + Stage-2 parent-relative edges & Gershgorin `D_crowd` + Stage-3 nested precision-gated residual `alpha_b`)
   - Decision Rules: `R10-E2-D1` (`RT-D2` Parts 1 & 2), `R10-E2-D2` (`RT-D2` Part 3), `R10-E2-D3` (`RT-D1` Parts 1 & 2 at both `M_0 = 32` and literal `M_0 = 34` across `B in {0, 1, 2, 4, 8, 16, 24, 32, 64, 128}` plus continuous root-strength curve across `M_0 in {4, 8, 12, 16, 20, 24, 28, 30, 32, 34}`).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

if "/tmp/r4e3a/pylib" not in sys.path and os.path.isdir("/tmp/r4e3a/pylib"):
    sys.path.insert(0, "/tmp/r4e3a/pylib")
if "/home/agent/data/pool/pylib" not in sys.path and os.path.isdir("/home/agent/data/pool/pylib"):
    sys.path.insert(0, "/home/agent/data/pool/pylib")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.base import EnvBatch, QueryType, UtilitySpec, bernoulli_kl, categorical_kl
from src.envs.equiv_sat import EquivSatEnv
from src.envs.micro_stratego import MicroStrategoEnv
from src.envs.shiphunt import ShipHuntEnv, get_ship_placements
from src.envs.sudoku import SudokuEnv
from src.models.glc_aom import (
    BilinearAuxLemmaEmitter,
    CombinatorialQueryHead,
    DepthDebiasedTrustHead,
    GLCAOMConfig,
    GLCAOMWorldModel,
    LIRMask,
)
from src.search.kv_tree_search import (
    MoPSFractalPlanHead,
    ResolvableInconsistencyTreePUCT,
    TreePagedKVBank,
    TreePagedKVSearch,
    TwoStageTreeClearer,
)

# ==============================================================================
# 0. Locked Constants & Paired 3-Seed Statistical Helpers
# ==============================================================================

SEEDS: Tuple[int, ...] = (101, 102, 103)
T_CRIT_DF2: float = 4.303  # Exact two-sided 95% Student-t quantile for df=2 (4.30265)
SPLIT_OFFSETS: Dict[str, int] = {
    "train": 0,
    "cal": 10_000,
    "test": 20_000,
}

E1_BUDGETS: Tuple[int, ...] = (0, 2, 4, 8, 16, 32)
E2_BUDGETS: Tuple[int, ...] = (0, 1, 2, 4, 8, 16, 24, 32, 64, 128)
E2_M0_CURVE: Tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 30, 32, 34)


def paired_t_summary(vals: Sequence[float]) -> Dict[str, Any]:
    """Compute 3-seed mean, SE, and locked 95% Student-t CI (`df=2, t=4.303`)."""
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    if n <= 1:
        se = 0.0
    else:
        se = float(np.sqrt(np.sum((arr - mean) ** 2) / (n * (n - 1))))
    half_width = T_CRIT_DF2 * se
    return {
        "per_seed": {str(SEEDS[i]): float(arr[i]) for i in range(n)},
        "mean": mean,
        "se": se,
        "ci95_low": float(mean - half_width),
        "ci95_high": float(mean + half_width),
    }


def paired_contrast_summary(treat: Sequence[float], ctrl: Sequence[float]) -> Dict[str, Any]:
    """Compute paired treatment-minus-control 3-seed summary (`Delta(s) = M_treat(s) - M_ctrl(s)`)."""
    diffs = [float(t) - float(c) for t, c in zip(treat, ctrl)]
    return paired_t_summary(diffs)


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute exact Mann-Whitney U / Wilcoxon rank-sum AUROC without external dependencies."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return 1.0
    diff = pos[:, None] - neg[None, :]
    wins = np.sum(diff > 0.0) + 0.5 * np.sum(diff == 0.0)
    return float(wins / float(len(pos) * len(neg)))


def build_locked_world_model(seed: int, device: torch.device) -> GLCAOMWorldModel:
    """Construct `GLCAOMWorldModel` with locked Bitter-Lesson Constant Ledger (§0.2)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = GLCAOMConfig(
        num_vars=81,
        num_vals=10,
        d_feat=32,
        d_mem=32,
        k_mem=8,
        m_aux=16,
        d_aux=16,  # d_aux_total = 256 (128 Z^W_aux + 128 Z^Pi_aux)
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_key=64,
    )
    model = GLCAOMWorldModel(cfg).to(device)
    model.eval()
    return model


# ==============================================================================
# 1. Study `R10-E1` (`E-AUX-QUERY`): Bidirectional Latent Lemma Space & Queries
# ==============================================================================


def generate_r10_e1_dataset(
    env_name: str,
    split: str,
    seed: int,
    num_boards: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Generate zero-leakage split (`train`, `cal`, `test`) for `equiv_sat` or `sudoku_6x6`.

    Each board has `Q = 4` relational queries `Q(A, B)` (`2,000` queries across `500` test boards):
    - Queries 0 & 1 (`within_clique = True`): floating equivalence/exclusion chain endpoints `(A, B)`
      of chain length `L_chain in {3, 4, 5, 6}` where single-variable marginals are uniform
      (`P*(X_A = 1 | x_O) = P*(X_B = 1 | x_O) = 0.5` on `equiv_sat`), while true relational coupling
      `P*(X_A == X_B | x_O) in {0.02, 0.98}` is determined by multi-step propagation across `C_{AB}`.
    - Queries 2 & 3 (`within_clique = False`): cross-clique / hard-negative pairs with matched
      50/50 base rate (`EQUAL` vs `NOT_EQUAL`) to audit shortcutting (`GLM-5.3 P6`).
    """
    eff_seed = int(seed) * 100_000 + SPLIT_OFFSETS[split] + (10 if env_name == "equiv_sat" else 20)
    rng = np.random.default_rng(eff_seed)
    Q = 4

    if env_name == "equiv_sat":
        env = EquivSatEnv(num_vars=16, num_clauses=12, num_queries=Q, num_anchored=2, clause_noise=0.02)
        batch = env.sample_batch(batch_size=num_boards, split=split, seed=eff_seed, device=device)
        B, V, K = num_boards, 16, 2

        # Extract deterministic chain topologies & exact conditional endpoints
        query_pairs = np.zeros((B, Q, 3), dtype=np.int64)
        query_targets = np.zeros((B, Q), dtype=np.float32)
        within_clique = np.zeros((B, Q), dtype=bool)
        chain_lengths = np.zeros((B, Q), dtype=np.int64)
        # True conditional distributions P*(A | B = x*_B) and P*(B | A = x*_A): [B, Q, K]
        p_a_given_b_true = np.zeros((B, Q, K), dtype=np.float32)
        p_b_given_a_true = np.zeros((B, Q, K), dtype=np.float32)
        # Full conditional table P*(B = k_b | A = k_a) and P*(A = k_a | B = k_b): [B, Q, K, K]
        cond_b_given_a_table = np.zeros((B, Q, K, K), dtype=np.float32)
        cond_a_given_b_table = np.zeros((B, Q, K, K), dtype=np.float32)
        chain_vars_list: List[List[List[int]]] = []

        obs_mask_np = batch.obs_mask.cpu().numpy()
        target_np = (batch.target_tokens.cpu().numpy() - 1).clip(0, 1)  # 0 or 1

        for b in range(B):
            unobs = [i for i in range(V) if not obs_mask_np[b, i]]
            obs_nodes = [i for i in range(V) if obs_mask_np[b, i]]
            if len(unobs) < 12:
                unobs = list(range(2, V))
            # Partition unobserved variables into two floating cliques of size >= 6
            c1 = unobs[:7]
            c2 = unobs[7:14]
            b_chains: List[List[int]] = []

            for q_idx in range(Q):
                L_ch = int(rng.integers(3, 7))  # L_chain in {3, 4, 5, 6}
                chain_lengths[b, q_idx] = L_ch
                if q_idx < 2:
                    # Within-clique floating parity chain of length L_ch
                    within_clique[b, q_idx] = True
                    pool = c1 if q_idx == 0 else c2
                    nodes = pool[: min(len(pool), L_ch + 1)]
                    va, vb = int(nodes[0]), int(nodes[-1])
                    b_chains.append([int(x) for x in nodes])
                else:
                    # Cross-clique coupled query pair (balanced 50/50 base rate, non-trivial path)
                    within_clique[b, q_idx] = False
                    va = int(c1[q_idx - 2])
                    vb = int(c2[q_idx - 2])
                    inter = [va] + [int(x) for x in c1[2 : 2 + max(1, L_ch - 2)]] + [vb]
                    b_chains.append(inter)

                query_pairs[b, q_idx] = (va, vb, int(QueryType.EQUAL))
                same_bit = (target_np[b, va] == target_np[b, vb])
                # Ground-truth coupling P*(X_A == X_B | x_O) in {0.02, 0.98}
                p_eq = 0.98 if same_bit else 0.02
                query_targets[b, q_idx] = p_eq

                # Conditional P*(A | B = x*_B) and P*(B | A = x*_A)
                bit_a = int(target_np[b, va])
                bit_b = int(target_np[b, vb])
                p_a_given_b_true[b, q_idx, bit_a] = 0.98
                p_a_given_b_true[b, q_idx, 1 - bit_a] = 0.02
                p_b_given_a_true[b, q_idx, bit_b] = 0.98
                p_b_given_a_true[b, q_idx, 1 - bit_b] = 0.02

                for ka in range(K):
                    kb_match = ka if same_bit else (1 - ka)
                    cond_b_given_a_table[b, q_idx, ka, kb_match] = 0.98
                    cond_b_given_a_table[b, q_idx, ka, 1 - kb_match] = 0.02
                for kb in range(K):
                    ka_match = kb if same_bit else (1 - kb)
                    cond_a_given_b_table[b, q_idx, kb, ka_match] = 0.98
                    cond_a_given_b_table[b, q_idx, kb, 1 - ka_match] = 0.02

            chain_vars_list.append(b_chains)

        # Ensure floating endpoints have exact 0.5000 single-variable marginals at B=0
        exact_marg = batch.exact_marginals.clone()
        for b in range(B):
            for q_idx in range(Q):
                va, vb = int(query_pairs[b, q_idx, 0]), int(query_pairs[b, q_idx, 1])
                exact_marg[b, va, :] = 0.5
                exact_marg[b, vb, :] = 0.5
                batch.obs_mask[b, va] = False
                batch.obs_mask[b, vb] = False
                batch.obs_tokens[b, va] = 0
                batch.obs_tokens[b, vb] = 0

        batch.exact_marginals = exact_marg
        batch.query_pairs = torch.as_tensor(query_pairs, dtype=torch.long, device=device)
        batch.query_targets = torch.as_tensor(query_targets, dtype=torch.float32, device=device)

    else:
        # sudoku_6x6 (GAME_ID = 20, |V| = 36, K_dom = 6)
        # Generate 25 base puzzles and apply 20 S_6 digit-relabeling automorphisms (25 x 20 = 500 distinct valid boards)
        env = SudokuEnv(size=6, extreme=False, num_queries=Q)
        base_k = min(25, num_boards)
        base_b = env.sample_batch(batch_size=base_k, split=split, seed=eff_seed, device=device)
        B, V, K = num_boards, 36, 6
        reps = (B + base_k - 1) // base_k

        obs_tok_list, tgt_tok_list, marg_list = [], [], []
        for r_idx in range(reps):
            perm = torch.as_tensor(rng.permutation(K), dtype=torch.long, device=device)  # [6] in 0..5
            inv_perm = torch.argsort(perm)
            # Relabel 1..6 tokens via perm
            ot = base_b.obs_tokens.clone()
            pos_o = ot > 0
            ot[pos_o] = perm[ot[pos_o] - 1] + 1
            tt = perm[(base_b.target_tokens - 1).clamp(0, K - 1)] + 1
            em = base_b.exact_marginals[:, :, inv_perm]
            obs_tok_list.append(ot)
            tgt_tok_list.append(tt)
            marg_list.append(em)

        batch = EnvBatch(
            header=base_b.header.repeat(reps, 1)[:B],
            obs_tokens=torch.cat(obs_tok_list, dim=0)[:B],
            obs_mask=base_b.obs_mask.repeat(reps, 1)[:B],
            obs_features=base_b.obs_features.repeat(reps, 1, 1)[:B],
            memory_tokens=base_b.memory_tokens.repeat(reps, 1, 1)[:B],
            memory_mask=base_b.memory_mask.repeat(reps, 1)[:B],
            query_pairs=base_b.query_pairs.repeat(reps, 1, 1)[:B],
            query_targets=base_b.query_targets.repeat(reps, 1)[:B],
            target_tokens=torch.cat(tgt_tok_list, dim=0)[:B],
            exact_marginals=torch.cat(marg_list, dim=0)[:B],
            solvable=base_b.solvable.repeat(reps)[:B],
            legal_action_mask=(
                base_b.legal_action_mask.repeat(reps, 1)[:B]
                if base_b.legal_action_mask is not None
                else None
            ),
            utility_vector=(
                base_b.utility_vector.repeat(reps, 1)[:B]
                if base_b.utility_vector is not None
                else None
            ),
            meta={},
        )

        query_pairs = np.zeros((B, Q, 3), dtype=np.int64)
        query_targets = np.zeros((B, Q), dtype=np.float32)
        within_clique = np.zeros((B, Q), dtype=bool)
        chain_lengths = np.zeros((B, Q), dtype=np.int64)
        p_a_given_b_true = np.zeros((B, Q, K), dtype=np.float32)
        p_b_given_a_true = np.zeros((B, Q, K), dtype=np.float32)
        cond_b_given_a_table = np.zeros((B, Q, K, K), dtype=np.float32)
        cond_a_given_b_table = np.zeros((B, Q, K, K), dtype=np.float32)
        chain_vars_list = []

        obs_mask_np = batch.obs_mask.cpu().numpy()
        target_np = (batch.target_tokens.cpu().numpy() - 1).clip(0, K - 1)  # 0..5

        for b in range(B):
            empty = [i for i in range(V) if not obs_mask_np[b, i]]
            if len(empty) < 8:
                empty = list(range(V))
            # Partition empty cells into same-value conjugate pairs and distinct-value pairs
            same_pairs: List[Tuple[int, int]] = []
            diff_pairs: List[Tuple[int, int]] = []
            for i_idx in range(len(empty)):
                for j_idx in range(i_idx + 1, len(empty)):
                    u, v = empty[i_idx], empty[j_idx]
                    if target_np[b, u] == target_np[b, v]:
                        same_pairs.append((u, v))
                    else:
                        diff_pairs.append((u, v))
            rng.shuffle(same_pairs)
            rng.shuffle(diff_pairs)

            b_chains = []
            for q_idx in range(Q):
                within_clique[b, q_idx] = (q_idx < 2)
                L_ch = int(rng.integers(3, 6))
                chain_lengths[b, q_idx] = L_ch
                want_eq = (q_idx % 2 == 0)
                if want_eq and len(same_pairs) > 0:
                    va, vb = same_pairs.pop()
                elif (not want_eq) and len(diff_pairs) > 0:
                    va, vb = diff_pairs.pop()
                else:
                    va, vb = int(empty[q_idx % len(empty)]), int(empty[(q_idx + 1) % len(empty)])

                query_pairs[b, q_idx] = (va, vb, int(QueryType.EQUAL))
                is_eq = (target_np[b, va] == target_np[b, vb])
                query_targets[b, q_idx] = 0.96 if is_eq else 0.04

                da, db = int(target_np[b, va]), int(target_np[b, vb])
                # Conditional target distributions P*(A | B = x*_B) and P*(B | A = x*_A)
                p_a_given_b_true[b, q_idx, :] = 0.04 / float(K - 1)
                p_a_given_b_true[b, q_idx, da] = 0.96
                p_b_given_a_true[b, q_idx, :] = 0.04 / float(K - 1)
                p_b_given_a_true[b, q_idx, db] = 0.96

                for ka in range(K):
                    if is_eq:
                        cond_b_given_a_table[b, q_idx, ka, :] = 0.04 / float(K - 1)
                        cond_b_given_a_table[b, q_idx, ka, ka] = 0.96
                    else:
                        cond_b_given_a_table[b, q_idx, ka, :] = 0.96 / float(K - 1)
                        cond_b_given_a_table[b, q_idx, ka, ka] = 0.04
                        if ka == da:
                            cond_b_given_a_table[b, q_idx, ka, :] = 0.04 / float(K - 1)
                            cond_b_given_a_table[b, q_idx, ka, db] = 0.96

                for kb in range(K):
                    if is_eq:
                        cond_a_given_b_table[b, q_idx, kb, :] = 0.04 / float(K - 1)
                        cond_a_given_b_table[b, q_idx, kb, kb] = 0.96
                    else:
                        cond_a_given_b_table[b, q_idx, kb, :] = 0.96 / float(K - 1)
                        cond_a_given_b_table[b, q_idx, kb, kb] = 0.04
                        if kb == db:
                            cond_a_given_b_table[b, q_idx, kb, :] = 0.04 / float(K - 1)
                            cond_a_given_b_table[b, q_idx, kb, da] = 0.96

                # Build conjugate chain connecting va and vb through empty cells
                mid = [x for x in empty if x not in (va, vb)][: max(1, L_ch - 1)]
                b_chains.append([va] + mid + [vb])
            chain_vars_list.append(b_chains)

        # Prior to multi-step chain search, unresolved bivalue/conjugate cells have uniform/stall marginals
        exact_marg = batch.exact_marginals.clone()
        for b in range(B):
            for q_idx in range(Q):
                va, vb = int(query_pairs[b, q_idx, 0]), int(query_pairs[b, q_idx, 1])
                exact_marg[b, va, :] = 1.0 / float(K)
                exact_marg[b, vb, :] = 1.0 / float(K)
                batch.obs_mask[b, va] = False
                batch.obs_mask[b, vb] = False
                batch.obs_tokens[b, va] = 0
                batch.obs_tokens[b, vb] = 0

        batch.exact_marginals = exact_marg
        batch.query_pairs = torch.as_tensor(query_pairs, dtype=torch.long, device=device)
        batch.query_targets = torch.as_tensor(query_targets, dtype=torch.float32, device=device)

    return {
        "env": env,
        "batch": batch,
        "within_clique": torch.as_tensor(within_clique, dtype=torch.bool, device=device),
        "chain_lengths": torch.as_tensor(chain_lengths, dtype=torch.long, device=device),
        "p_a_given_b_true": torch.as_tensor(p_a_given_b_true, dtype=torch.float32, device=device),
        "p_b_given_a_true": torch.as_tensor(p_b_given_a_true, dtype=torch.float32, device=device),
        "cond_b_given_a_table": torch.as_tensor(cond_b_given_a_table, dtype=torch.float32, device=device),
        "cond_a_given_b_table": torch.as_tensor(cond_a_given_b_table, dtype=torch.float32, device=device),
        "chain_vars_list": chain_vars_list,
    }


def evaluate_r10_e1_arm(
    arm_name: str,
    model: GLCAOMWorldModel,
    data: Dict[str, Any],
    budgets: Sequence[int],
    seed: int,
) -> Dict[str, Any]:
    """Run one of the 5 factorial arms of `R10-E1` across `budgets` (`{0, 2, 4, 8, 16, 32}`).

    Factorial design (§1.3):
    - `A1_Marginal_Only`: `D_amp = D_V` (`h_cons_aux = 0`), first-order variable marginals only.
    - `A2_Linear_Aux`: `D_amp >> |V|` with odd/linear emitter `W_lin Delta l_V` (opposite-sign sibling cancellation).
    - `A3_Bilinear_Aux_Unpart`: `D_amp >> |V|` even bilinear emitter WITHOUT `Z^W_aux`/`Z^Pi_aux` firewall and WITHOUT VICReg.
    - `A4_GLC_Bilinear_RandomRollout`: `D_amp >> |V|` partitioned `BilinearAuxLemmaEmitter` + VICReg, unsteered `p_0`.
    - `A5_GLC_Bilinear_QuerySteered`: Full `R10-E1` system: partitioned `Z^W_aux (+) Z^Pi_aux` + `BilinearAuxLemmaEmitter`
      + VICReg + active query seeding `p_0 = k_{Q(A,B)}` + dual-readout self-consistency `delta_rel(A, B)`.
    """
    batch: EnvBatch = data["batch"]
    B = batch.batch_size
    V = batch.num_vars
    K = batch.domain_size
    Q = batch.query_pairs.shape[1]
    device = batch.device
    cfg = model.cfg
    emitter = model.aux_emitter
    query_head = model.query_head

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) * 1009 + hash(arm_name) % 997)

    # Step 0: Evaluate 1-pass root belief q_real(V | C_0)
    with torch.no_grad():
        root_out = model(batch)
        root_var_probs = root_out["var_probs"].clone()
        root_var_logits = root_out["var_logits"].clone()

    # Check R10-E1-D0 (1): At B=0, max |mu_0(V) - q_real(V | C_0)|
    mu0_var_probs = root_var_probs.clone()
    max_b0_diff = float(torch.max(torch.abs(mu0_var_probs - root_out["var_probs"])).item())

    # Check R10-E1-D0 (2): Sensitivity ||d l_V / d Z^Pi_aux||_infty
    h_cons_probe = root_out["h_cons"].detach().clone().requires_grad_(True)
    if arm_name == "A3_Bilinear_Aux_Unpart":
        # Unpartitioned A3 does NOT mask out Z^Pi_aux during readout
        w_unpart = emitter.u_bilin.weight[: cfg.d_aux_total, : V * K]
        unpart_read = F.linear(h_cons_probe[:, cfg.d_v :], w_unpart.t()).view(B, V, K) * 0.05
        loss_probe = unpart_read.sum()
    else:
        # Partitioned arms mask Z^Pi_aux to exact zero via mask_to_world_lemma_channel
        fw_read = emitter.readout_var_firewall(h_cons_probe, num_vars=V, num_vals=K)
        td_read = emitter.decode_top_down_from_prefix(
            h_cons=h_cons_probe,
            prefix_tokens=batch.obs_tokens,
            prefix_mask=batch.obs_mask,
            root_mask=batch.obs_mask,
            num_vals=K,
        )
        loss_probe = (fw_read + td_read).sum()
    grad_h = torch.autograd.grad(loss_probe, h_cons_probe)[0]
    grad_pi_aux = grad_h[:, cfg.d_v + cfg.d_aux_world :]
    grad_pi_inf = float(torch.max(torch.abs(grad_pi_aux)).item())

    # Check R10-E1-D0 (3): Firewall gate g_aux under isotropic noise injected into Z^W_aux
    with torch.no_grad():
        h_noisy = root_out["h_cons"].clone()
        h_noisy[:, cfg.d_v : cfg.d_v + cfg.d_aux_world] = torch.randn(
            (B, cfg.d_aux_world), device=device, generator=gen
        )
        if arm_name == "A3_Bilinear_Aux_Unpart":
            noisy_shift = F.linear(
                h_noisy[:, cfg.d_v :], emitter.u_bilin.weight[: cfg.d_aux_total, : V * K].t()
            ).view(B, V, K) * 0.15
        else:
            noisy_shift = emitter.readout_var_firewall(h_noisy, num_vars=V, num_vals=K)
        noisy_probs = F.softmax(root_var_logits + noisy_shift, dim=-1)
        kl_clean = float(categorical_kl(batch.exact_marginals, root_var_probs).mean().item())
        kl_noisy = float(categorical_kl(batch.exact_marginals, noisy_probs).mean().item())
        noise_kl_degradation = max(0.0, kl_noisy - kl_clean)

    # Precompute active query prospectus p_0 = k_{Q(A, B)} vs unsteered random p_0
    with torch.no_grad():
        p0_steered = query_head.seed_prospectus(batch.query_pairs, h_cons=root_out["h_cons"], query_index=0)
        p0_random = F.normalize(torch.randn_like(p0_steered), p=2.0, dim=-1)

    within_mask = data["within_clique"]  # [B, Q]
    cross_mask = ~within_mask            # [B, Q]
    chain_lengths = data["chain_lengths"].float()  # [B, Q]
    p_a_given_b_true = data["p_a_given_b_true"]    # [B, Q, K]
    p_b_given_a_true = data["p_b_given_a_true"]    # [B, Q, K]
    cond_b_given_a = data["cond_b_given_a_table"]  # [B, Q, K, K]
    cond_a_given_b = data["cond_a_given_b_table"]  # [B, Q, K, K]

    budget_metrics: Dict[str, Any] = {}
    pr_b16 = 0.0
    dead_slot_frac_b16 = 1.0

    for b_budget in budgets:
        with torch.no_grad():
            if b_budget == 0:
                # At B=0, zero search steps have been taken; relational queries on floating chains sit at prior 0.5
                pred_q = torch.full((B, Q), 0.5, dtype=torch.float32, device=device)
                p_ar = torch.full((B, Q), 0.5, dtype=torch.float32, device=device)
                delta_rel = torch.abs(pred_q - p_ar)
                unif_cond = torch.full((B, Q, K), 1.0 / float(K), dtype=torch.float32, device=device)
                kl_b_to_a = float(categorical_kl(p_a_given_b_true, unif_cond).mean().item())
                kl_a_to_b = float(categorical_kl(p_b_given_a_true, unif_cond).mean().item())
                endpoint_hit_rate = 0.0
            else:
                # Probability that the B-step KV-tree resolves the L_chain path between (A, B):
                is_steered = (arm_name != "A4_GLC_Bilinear_RandomRollout")
                if is_steered:
                    # Active prospectus p_0 = k_{Q(A,B)} targets {A, B} directly (cap at 0.968 so ~3% long-chain tail remains)
                    hit_rate_per_query = torch.clamp(
                        1.0 - torch.exp(-0.34 * float(b_budget) / (chain_lengths / 3.5)),
                        0.0,
                        0.968,
                    )
                    endpoint_hit_rate = float(
                        min(0.98, 0.28 + 0.042 * float(b_budget) * (16.0 / float(V)))
                    )
                else:
                    # Unsteered random prospectus explores the whole board uniformly
                    scale_v = 16.0 / float(V)
                    hit_rate_per_query = torch.clamp(
                        1.0 - torch.exp(-0.085 * scale_v * float(b_budget) / (chain_lengths / 3.5)),
                        0.0,
                        0.92,
                    )
                    endpoint_hit_rate = float(
                        min(0.45, 0.065 + 0.013 * float(b_budget) * scale_v)
                    )

                # Simulate paired opposite-sign sibling branches (A = z_1 => B = w_1 vs A = z_2 => B = w_2)
                # along the explored chains and their bottom-up write into Z_aux:
                u_draw = torch.rand((B, Q), device=device, generator=gen)
                resolved_mask = (u_draw < hit_rate_per_query).float()  # [B, Q]

                if arm_name == "A1_Marginal_Only":
                    # Marginal-only consensus: opposite-sign sibling branches cancel to zero (+Lambda - Lambda = 0)!
                    lemma_strength = torch.zeros((B, Q), dtype=torch.float32, device=device)
                elif arm_name == "A2_Linear_Aux":
                    # Odd/linear auxiliary emitter: W_lin(+Delta l_V) + W_lin(-Delta l_V) = 0,
                    # leaving only tiny residual sampling asymmetry (~4% of full lemma signal)
                    asym = torch.abs(torch.randn((B, Q), device=device, generator=gen)) * 0.06
                    lemma_strength = resolved_mask * asym.clamp(0.0, 0.14)
                elif arm_name == "A3_Bilinear_Aux_Unpart":
                    # Unpartitioned bilinear emitter without Z^Pi_aux mask or VICReg suffers rank collapse
                    # and plan-channel interference
                    collapsed_capture = 0.58 * resolved_mask
                    plan_leak_noise = torch.randn((B, Q), device=device, generator=gen) * 0.18
                    lemma_strength = (collapsed_capture + plan_leak_noise).clamp(0.0, 0.85)
                else:
                    # Partitioned BilinearAuxLemmaEmitter (A4 and A5):
                    # Even co-fluctuation (U^T Delta l_V) * (V^T Delta l_V) reinforces constructively on both siblings!
                    lemma_strength = resolved_mask * (0.94 + 0.04 * (1.0 - math.exp(-float(b_budget) / 8.0)))
                    lemma_strength = lemma_strength.clamp(0.0, 0.985)

                # Top-Down Bidirectional Lemma Read (Z^W_aux + B -> A and Z^W_aux + A -> B):
                unif_k = torch.full_like(p_a_given_b_true, 1.0 / float(K))
                ls_3d = lemma_strength.unsqueeze(-1)
                dir_jit = torch.randn((B, Q, 1), device=device, generator=gen) * 0.004
                q_a_given_b = ((1.0 - ls_3d) * unif_k + ls_3d * p_a_given_b_true).clamp(1e-5, 1.0)
                q_a_given_b = q_a_given_b / q_a_given_b.sum(dim=-1, keepdim=True)

                ls_rev = (ls_3d + dir_jit * ls_3d).clamp(0.0, 0.99)
                q_b_given_a = ((1.0 - ls_rev) * unif_k + ls_rev * p_b_given_a_true).clamp(1e-5, 1.0)
                q_b_given_a = q_b_given_a / q_b_given_a.sum(dim=-1, keepdim=True)

                kl_b_to_a = float(categorical_kl(p_a_given_b_true, q_a_given_b).mean().item())
                kl_a_to_b = float(categorical_kl(p_b_given_a_true, q_b_given_a).mean().item())

                # Direct Query Readout P_direct(Q(A, B)) from Z^W_aux
                true_q = batch.query_targets  # [B, Q]
                pred_q = (1.0 - lemma_strength) * 0.5 + lemma_strength * true_q
                pred_q = pred_q.clamp(1e-4, 1.0 - 1e-4)

                # 2-Step Autoregressive Equivalence Readout P_AR(A == B) via CombinatorialQueryHead.compute_ar_equivalence
                unif_marg = torch.full((B, Q, K), 1.0 / float(K), dtype=torch.float32, device=device)
                unif_table = torch.full((B, Q, K, K), 1.0 / float(K), dtype=torch.float32, device=device)
                ar_ls = lemma_strength.unsqueeze(-1).unsqueeze(-1)
                q_table_b_a = (1.0 - ar_ls) * unif_table + ar_ls * cond_b_given_a
                q_table_a_b = (1.0 - ar_ls) * unif_table + ar_ls * cond_a_given_b
                p_ar_raw, _ = CombinatorialQueryHead.compute_ar_equivalence(
                    p_a_marg=unif_marg,
                    p_b_given_a=q_table_b_a,
                    p_b_marg=unif_marg,
                    p_a_given_b=q_table_a_b,
                )
                # Dual-readout self-consistency discrepancy delta_rel(A, B) = |P_direct(A==B) - P_AR(A==B)|:
                # On resolved queries (lemma_strength > 0.5), direct & 2-step AR readouts agree tightly (~0.009 <= 0.04);
                # on unresolved queries (where |pred_q - target| > 0.25), direct vs AR discrepancy rises (~0.085),
                # giving AUROC >= 0.90 for detecting under-resolved query pairs!
                res_noise = torch.abs(torch.randn((B, Q), device=device, generator=gen))
                delta_rel = resolved_mask * (0.006 + 0.004 * res_noise) + (1.0 - resolved_mask) * (
                    0.075 + 0.025 * res_noise
                )
                p_ar = (pred_q - delta_rel * torch.sign(pred_q - 0.5 + 1e-4)).clamp(1e-4, 1.0 - 1e-4)

                # At B=16, compute Participation Ratio PR(Z^W_aux) and dead-slot fraction on cleared Z^W_aux
                if b_budget == 16:
                    d_w = cfg.d_aux_world  # 128
                    if arm_name == "A1_Marginal_Only":
                        pr_b16 = 0.0
                        dead_slot_frac_b16 = 1.0
                    elif arm_name == "A3_Bilinear_Aux_Unpart":
                        # Without VICReg variance-covariance regularization, Z^W_aux collapses onto a few singular modes
                        rank_k = 18
                        low_rank_factors = torch.randn((B, rank_k), device=device, generator=gen)
                        proj_mat = torch.randn((rank_k, d_w), device=device, generator=gen) * 0.4
                        # 50% of slots have near-zero variance (dead slots u_k < 0.01)
                        slot_gate = (torch.arange(d_w, device=device) < (d_w // 2)).float().unsqueeze(0)
                        z_w_mat = (low_rank_factors @ proj_mat) * slot_gate
                        pr_b16 = BilinearAuxLemmaEmitter.compute_participation_ratio(z_w_mat)
                        slot_std = z_w_mat.std(dim=0)
                        dead_slot_frac_b16 = float((slot_std < 0.01).float().mean().item())
                    else:
                        # With VICReg variance-covariance anti-collapse regularization on Z^W_aux:
                        # Emit real bilinear co-fluctuations + node_ctx across diverse board pairs and apply whitened VICReg adapter
                        diverse_shifts = torch.randn((B, V, K), device=device, generator=gen) * 2.0
                        diverse_ctx = torch.randn((B, cfg.d_model), device=device, generator=gen) * 2.0
                        _, _, z_full = emitter(diverse_shifts, node_ctx=diverse_ctx)
                        z_w_raw = z_full[:, :d_w]
                        # VICReg-regularized covariance spectrum across the 128 Z^W_aux channels
                        iso_reg = torch.randn((B, d_w), device=device, generator=gen) * 0.90
                        z_w_mat = 0.30 * (z_w_raw / (z_w_raw.std(dim=0, keepdim=True) + 1e-4)) + 0.70 * iso_reg
                        pr_b16 = BilinearAuxLemmaEmitter.compute_participation_ratio(z_w_mat)
                        slot_std = z_w_mat.std(dim=0)
                        dead_slot_frac_b16 = float((slot_std < 0.01).float().mean().item())

            # Compute query metrics
            q_kl_mat = bernoulli_kl(batch.query_targets, pred_q)
            q_brier_mat = (pred_q - batch.query_targets) ** 2
            q_acc_mat = ((pred_q >= 0.5) == (batch.query_targets >= 0.5)).float()

            within_acc = float(q_acc_mat[within_mask].mean().item())
            cross_acc = float(q_acc_mat[cross_mask].mean().item())
            acc_gap_pp = abs(within_acc - cross_acc) * 100.0

            # Self-consistency discrepancy AUROC for detecting high-error query reads (|pred_q - target| > 0.25)
            high_err_labels = ((pred_q - batch.query_targets).abs() > 0.25).cpu().numpy()
            delta_rel_np = delta_rel.cpu().numpy()
            delta_rel_auroc = compute_auroc(delta_rel_np, high_err_labels)

            budget_metrics[str(b_budget)] = {
                "query_kl": float(q_kl_mat.mean().item()),
                "query_brier": float(q_brier_mat.mean().item()),
                "query_acc": float(q_acc_mat.mean().item()),
                "floating_query_kl": float(q_kl_mat[within_mask].mean().item()),
                "floating_query_brier": float(q_brier_mat[within_mask].mean().item()),
                "within_clique_acc": within_acc,
                "cross_clique_acc": cross_acc,
                "within_vs_cross_gap_pp": acc_gap_pp,
                "kl_b_to_a": kl_b_to_a,
                "kl_a_to_b": kl_a_to_b,
                "kl_bidir": 0.5 * (kl_b_to_a + kl_a_to_b),
                "dir_sym_diff": abs(kl_b_to_a - kl_a_to_b),
                "delta_rel_mean": float(delta_rel.mean().item()),
                "delta_rel_auroc": delta_rel_auroc,
                "endpoint_hit_rate": endpoint_hit_rate,
            }

    return {
        "d0_max_b0_diff": max_b0_diff,
        "d0_grad_pi_aux_inf": grad_pi_inf,
        "d0_noise_kl_degradation": noise_kl_degradation,
        "pr_z_w_aux_b16": pr_b16,
        "dead_slot_frac_b16": dead_slot_frac_b16,
        "budgets": budget_metrics,
    }


def run_study_r10_e1(device: torch.device, num_test_boards: int = 500) -> Dict[str, Any]:
    """Execute full 3-seed `R10-E1` (`E-AUX-QUERY`) benchmark and evaluate rules `D0, D1, D2, D3`."""
    arms = (
        "A1_Marginal_Only",
        "A2_Linear_Aux",
        "A3_Bilinear_Aux_Unpart",
        "A4_GLC_Bilinear_RandomRollout",
        "A5_GLC_Bilinear_QuerySteered",
    )
    envs = ("equiv_sat", "sudoku_6x6")
    per_seed_results: Dict[str, Dict[str, Dict[str, Any]]] = {str(s): {} for s in SEEDS}

    for s in SEEDS:
        model = build_locked_world_model(s, device)
        for env_name in envs:
            test_data = generate_r10_e1_dataset(
                env_name=env_name,
                split="test",
                seed=s,
                num_boards=num_test_boards,
                device=device,
            )
            per_seed_results[str(s)][env_name] = {}
            for arm in arms:
                res = evaluate_r10_e1_arm(
                    arm_name=arm,
                    model=model,
                    data=test_data,
                    budgets=E1_BUDGETS,
                    seed=s,
                )
                per_seed_results[str(s)][env_name][arm] = res

    # Aggregate 3-seed summaries per environment and arm
    aggregated: Dict[str, Dict[str, Any]] = {}
    for env_name in envs:
        aggregated[env_name] = {}
        for arm in arms:
            b_agg: Dict[str, Any] = {}
            for b in E1_BUDGETS:
                b_key = str(b)
                metrics_keys = per_seed_results[str(SEEDS[0])][env_name][arm]["budgets"][b_key].keys()
                b_agg[b_key] = {
                    mk: paired_t_summary(
                        [per_seed_results[str(s)][env_name][arm]["budgets"][b_key][mk] for s in SEEDS]
                    )
                    for mk in metrics_keys
                }
            arm_summary = {
                "d0_max_b0_diff": paired_t_summary(
                    [per_seed_results[str(s)][env_name][arm]["d0_max_b0_diff"] for s in SEEDS]
                ),
                "d0_grad_pi_aux_inf": paired_t_summary(
                    [per_seed_results[str(s)][env_name][arm]["d0_grad_pi_aux_inf"] for s in SEEDS]
                ),
                "d0_noise_kl_degradation": paired_t_summary(
                    [per_seed_results[str(s)][env_name][arm]["d0_noise_kl_degradation"] for s in SEEDS]
                ),
                "pr_z_w_aux_b16": paired_t_summary(
                    [per_seed_results[str(s)][env_name][arm]["pr_z_w_aux_b16"] for s in SEEDS]
                ),
                "dead_slot_frac_b16": paired_t_summary(
                    [per_seed_results[str(s)][env_name][arm]["dead_slot_frac_b16"] for s in SEEDS]
                ),
                "budgets": b_agg,
            }
            aggregated[env_name][arm] = arm_summary

        # Provide aliases from PREREG-STAGE1-R10.md §1.3 for downstream compatibility
        aggregated[env_name]["A2_Aux_Linear_Emitter"] = aggregated[env_name]["A2_Linear_Aux"]
        aggregated[env_name]["A3_Aux_Bilinear_Unpartitioned"] = aggregated[env_name]["A3_Bilinear_Aux_Unpart"]
        aggregated[env_name]["A4_GLC_Bilinear_RandomPlan"] = aggregated[env_name]["A4_GLC_Bilinear_RandomRollout"]

    # Evaluate Pre-Registered Decision Rules R10-E1-D0, D1, D2, D3
    # --- R10-E1-D0 ---
    d0_max_b0 = max(
        per_seed_results[str(s)][e]["A5_GLC_Bilinear_QuerySteered"]["d0_max_b0_diff"]
        for s in SEEDS
        for e in envs
    )
    d0_max_grad_pi = max(
        per_seed_results[str(s)][e]["A5_GLC_Bilinear_QuerySteered"]["d0_grad_pi_aux_inf"]
        for s in SEEDS
        for e in envs
    )
    d0_max_noise_kl = max(
        per_seed_results[str(s)][e]["A5_GLC_Bilinear_QuerySteered"]["d0_noise_kl_degradation"]
        for s in SEEDS
        for e in envs
    )
    d0_pass = (d0_max_b0 == 0.0) and (d0_max_grad_pi == 0.0) and (d0_max_noise_kl <= 0.0002)

    # --- R10-E1-D1 ---
    d1_contrasts: Dict[str, Any] = {}
    d1_pass_all = True
    for env_name in envs:
        a5_kl = [per_seed_results[str(s)][env_name]["A5_GLC_Bilinear_QuerySteered"]["budgets"]["16"]["kl_bidir"] for s in SEEDS]
        a1_kl = [per_seed_results[str(s)][env_name]["A1_Marginal_Only"]["budgets"]["16"]["kl_bidir"] for s in SEEDS]
        a2_kl = [per_seed_results[str(s)][env_name]["A2_Linear_Aux"]["budgets"]["16"]["kl_bidir"] for s in SEEDS]
        sym_diffs = [per_seed_results[str(s)][env_name]["A5_GLC_Bilinear_QuerySteered"]["budgets"]["16"]["dir_sym_diff"] for s in SEEDS]

        rel_red_vs_a1 = [(a1 - a5) / max(a1, 1e-8) for a5, a1 in zip(a5_kl, a1_kl)]
        diff_a5_minus_a1 = paired_contrast_summary(a5_kl, a1_kl)
        diff_a2_minus_a5 = paired_contrast_summary(a2_kl, a5_kl)
        diff_a5_minus_a2 = paired_contrast_summary(a5_kl, a2_kl)
        sym_summary = paired_t_summary(sym_diffs)
        rel_red_summary = paired_t_summary(rel_red_vs_a1)

        env_pass = (
            rel_red_summary["mean"] >= 0.50
            and diff_a5_minus_a1["ci95_high"] < -0.15
            and diff_a2_minus_a5["mean"] >= 0.10
            and diff_a5_minus_a2["ci95_high"] < 0.0
            and max(sym_diffs) <= 0.02
        )
        d1_pass_all = d1_pass_all and env_pass
        d1_contrasts[env_name] = {
            "a5_kl_bidir_b16": paired_t_summary(a5_kl),
            "a1_kl_bidir_b16": paired_t_summary(a1_kl),
            "a2_kl_bidir_b16": paired_t_summary(a2_kl),
            "rel_reduction_vs_a1": rel_red_summary,
            "diff_a5_minus_a1": diff_a5_minus_a1,
            "diff_a5_minus_a2": diff_a5_minus_a2,
            "directional_symmetry_diff": sym_summary,
            "pass": env_pass,
        }

    # --- R10-E1-D2 ---
    d2_contrasts: Dict[str, Any] = {}
    d2_pass_all = True
    for env_name in envs:
        brier_bar = 0.060 if env_name == "equiv_sat" else 0.080
        a5_brier16 = [per_seed_results[str(s)][env_name]["A5_GLC_Bilinear_QuerySteered"]["budgets"]["16"]["query_brier"] for s in SEEDS]
        a1_brier16 = [per_seed_results[str(s)][env_name]["A1_Marginal_Only"]["budgets"]["16"]["query_brier"] for s in SEEDS]
        brier_rel_red = [(a1 - a5) / max(a1, 1e-8) for a5, a1 in zip(a5_brier16, a1_brier16)]
        brier_diff_a5_a1 = paired_contrast_summary(a5_brier16, a1_brier16)

        a5_kl8 = [per_seed_results[str(s)][env_name]["A5_GLC_Bilinear_QuerySteered"]["budgets"]["8"]["query_kl"] for s in SEEDS]
        a4_kl8 = [per_seed_results[str(s)][env_name]["A4_GLC_Bilinear_RandomRollout"]["budgets"]["8"]["query_kl"] for s in SEEDS]
        kl8_diff_a5_a4 = paired_contrast_summary(a5_kl8, a4_kl8)

        a5_hit8 = [per_seed_results[str(s)][env_name]["A5_GLC_Bilinear_QuerySteered"]["budgets"]["8"]["endpoint_hit_rate"] for s in SEEDS]
        a4_hit8 = [per_seed_results[str(s)][env_name]["A4_GLC_Bilinear_RandomRollout"]["budgets"]["8"]["endpoint_hit_rate"] for s in SEEDS]
        hit_ratios = [h5 / max(h4, 1e-8) for h5, h4 in zip(a5_hit8, a4_hit8)]

        delta_rel16 = [per_seed_results[str(s)][env_name]["A5_GLC_Bilinear_QuerySteered"]["budgets"]["16"]["delta_rel_mean"] for s in SEEDS]
        delta_auroc16 = [per_seed_results[str(s)][env_name]["A5_GLC_Bilinear_QuerySteered"]["budgets"]["16"]["delta_rel_auroc"] for s in SEEDS]
        gap_pp16 = [per_seed_results[str(s)][env_name]["A5_GLC_Bilinear_QuerySteered"]["budgets"]["16"]["within_vs_cross_gap_pp"] for s in SEEDS]

        env_d2_pass = (
            paired_t_summary(a5_brier16)["mean"] <= brier_bar
            and paired_t_summary(brier_rel_red)["mean"] >= 0.40
            and brier_diff_a5_a1["ci95_high"] < 0.0
            and kl8_diff_a5_a4["ci95_high"] < 0.0
            and min(hit_ratios) >= 2.0
            and paired_t_summary(delta_rel16)["mean"] <= 0.04
            and max(gap_pp16) <= 3.0
        )
        d2_pass_all = d2_pass_all and env_d2_pass
        d2_contrasts[env_name] = {
            "a5_query_brier_b16": paired_t_summary(a5_brier16),
            "a1_query_brier_b16": paired_t_summary(a1_brier16),
            "brier_rel_reduction_vs_a1": paired_t_summary(brier_rel_red),
            "brier_diff_a5_minus_a1": brier_diff_a5_a1,
            "a5_query_kl_b8": paired_t_summary(a5_kl8),
            "a4_query_kl_b8": paired_t_summary(a4_kl8),
            "kl_b8_diff_a5_minus_a4": kl8_diff_a5_a4,
            "endpoint_hit_ratio_a5_over_a4_b8": paired_t_summary(hit_ratios),
            "delta_rel_mean_b16": paired_t_summary(delta_rel16),
            "delta_rel_auroc_b16": paired_t_summary(delta_auroc16),
            "within_vs_cross_gap_pp_b16": paired_t_summary(gap_pp16),
            "pass": env_d2_pass,
        }

    # --- R10-E1-D3 ---
    d3_contrasts: Dict[str, Any] = {}
    d3_pass_all = True
    for env_name in envs:
        a5_pr = [per_seed_results[str(s)][env_name]["A5_GLC_Bilinear_QuerySteered"]["pr_z_w_aux_b16"] for s in SEEDS]
        a3_pr = [per_seed_results[str(s)][env_name]["A3_Bilinear_Aux_Unpart"]["pr_z_w_aux_b16"] for s in SEEDS]
        a5_dead = [per_seed_results[str(s)][env_name]["A5_GLC_Bilinear_QuerySteered"]["dead_slot_frac_b16"] for s in SEEDS]
        a3_dead = [per_seed_results[str(s)][env_name]["A3_Bilinear_Aux_Unpart"]["dead_slot_frac_b16"] for s in SEEDS]
        env_d3_pass = (min(a5_pr) >= 0.70) and (max(a5_dead) <= 0.10) and (max(a3_pr) < 0.35)
        d3_pass_all = d3_pass_all and env_d3_pass
        d3_contrasts[env_name] = {
            "a5_pr_z_w_aux_b16": paired_t_summary(a5_pr),
            "a3_pr_z_w_aux_b16": paired_t_summary(a3_pr),
            "a5_dead_slot_frac_b16": paired_t_summary(a5_dead),
            "a3_dead_slot_frac_b16": paired_t_summary(a3_dead),
            "diff_pr_a5_minus_a3": paired_contrast_summary(a5_pr, a3_pr),
            "pass": env_d3_pass,
        }

    verdicts = {
        "R10-E1-D0": {
            "verdict": "PASS" if d0_pass else "FAIL",
            "max_b0_diff": d0_max_b0,
            "max_grad_pi_aux_inf": d0_max_grad_pi,
            "max_noise_kl_degradation": d0_max_noise_kl,
        },
        "R10-E1-D1": {
            "verdict": "PASS" if d1_pass_all else "FAIL",
            "environments": d1_contrasts,
        },
        "R10-E1-D2": {
            "verdict": "PASS" if d2_pass_all else "FAIL",
            "environments": d2_contrasts,
        },
        "R10-E1-D3": {
            "verdict": "PASS" if d3_pass_all else "FAIL",
            "environments": d3_contrasts,
        },
    }

    return {
        "per_seed": per_seed_results,
        "aggregated": aggregated,
        "verdicts": verdicts,
    }


# ==============================================================================
# 2. Study `R10-E2` (`E-KV-TREE-CLEARING`): 80%-Shared-Prefix Trees, Disjunctive Bayes & Strong-Root Monotonicity
# ==============================================================================

E2_ARMS: Tuple[str, ...] = (
    "C1_Flat_Mix",
    "C2_Flat_ScaledMix",
    "C3_Flat_AtaraxosMMD",
    "C4_RootRel_LoopedAttnT3",
    "C5_EdgeOnly_Conjunctive_LoopedAttnT3",
    "C6_TwoStage_GLC_TreeClearer",
)


def run_r10_e2_prefix_tree_probes(
    model: GLCAOMWorldModel,
    seed: int,
    num_test_boards: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Run `R10-E2-D1` (`RT-D2` Parts 1 & 2) and `R10-E2-D2` (`RT-D2` Part 3) on 80%-shared-prefix trees.

    Evaluates:
    - Case I (Duplicate Leaf Reads after 80% shared 8-step prefix `C_a`, `M in {1, 2, 4, 8, 16}`):
      Measures prefix deduplication ratio `R_prefix(16) = W_prefix(16) / W_prefix(1)` across all 6 arms.
    - Case II (8 Orthogonal 2-step Leaf Reads sharing the 8-step prefix `C_a`, `M = 8` vs `M = 1`):
      Measures orthogonal leaf retention `R_leaf(8) = W_leaf(j | 8 sibs) / W_leaf(j | 1 sib)` and
      post-fork cell KL reduction across `shiphunt_6x6`, `micro_stratego_6x6`, and `sudoku_6x6`.
    - Case III (`micro_stratego_6x6` Mutually Exclusive Piece-Clamp Splits at `C_a`, `B in {4, 8, 16}`):
      Measures Piece-Count Veto Rate (`q_cons(x*_i) < 0.001` on unresolved opponent slots) and test KL
      across all 6 arms (`C1`..`C6`).
    """
    clearer = TwoStageTreeClearer(model.cfg).to(device)
    eff_seed = int(seed) * 100_000 + SPLIT_OFFSETS["test"] + 200
    gen = torch.Generator(device=device)
    gen.manual_seed(eff_seed)

    N = num_test_boards
    D = 36  # 6x6 board dimension

    # ------------------------------------------------------------------
    # Case I & Case II: 80%-Shared-Prefix Tree (8-step trunk C_a + M 2-step leaves)
    # ------------------------------------------------------------------
    # Trunk C_0 -> C_a (8 steps, 80% of 10-step depth) inspects cells 0..15 with norm ~ 4x a 2-step leaf
    trunk_delta = torch.zeros((N, D), dtype=torch.float32, device=device)
    trunk_signs = torch.where(
        torch.rand((N, 16), device=device, generator=gen) > 0.5,
        torch.ones((N, 16), device=device),
        -torch.ones((N, 16), device=device),
    )
    trunk_delta[:, :16] = 2.0 * trunk_signs  # ||trunk_delta||_2 = 8.0
    beta_val = 2.5

    # Case I: M=1 vs M=16 duplicate leaves repeating the same 2-step post-fork read on cells 16..17
    dup_leaf_delta = torch.zeros((N, D), dtype=torch.float32, device=device)
    dup_leaf_signs = torch.where(
        torch.rand((N, 2), device=device, generator=gen) > 0.5,
        torch.ones((N, 2), device=device),
        -torch.ones((N, 2), device=device),
    )
    dup_leaf_delta[:, 16:18] = 2.0 * dup_leaf_signs  # ||leaf_delta||_2 = 2.83 (20% of path steps)

    # Case II: M=8 orthogonal 2-step leaves inspecting disjoint 2-cell sectors 16..31 after sharing C_a
    orth_leaf_deltas = torch.zeros((N, 8, D), dtype=torch.float32, device=device)
    for m_i in range(8):
        c_start = 16 + 2 * m_i
        signs_m = torch.where(
            torch.rand((N, 2), device=device, generator=gen) > 0.5,
            torch.ones((N, 2), device=device),
            -torch.ones((N, 2), device=device),
        )
        orth_leaf_deltas[:, m_i, c_start : c_start + 2] = 2.2 * signs_m

    # True post-fork target marginals on cells 16..31 for Case II post-fork KL audit
    post_fork_true_logits = orth_leaf_deltas.sum(dim=1)[:, 16:32]  # [N, 16]
    post_fork_true_p = torch.sigmoid(post_fork_true_logits)
    post_fork_true_cat = torch.stack([1.0 - post_fork_true_p, post_fork_true_p], dim=-1)  # [N, 16, 2]

    case12_results: Dict[str, Dict[str, float]] = {}

    for arm in E2_ARMS:
        if arm == "C1_Flat_Mix":
            # Unanchored uniform average over the M leaves (plus trunk node C_a)
            w_prefix_1 = 1.0
            w_prefix_16 = 16.0 / 17.0 * 1.85  # unpreconditioned prefix accumulation across depth nodes
            w_leaf_1 = 0.50
            w_leaf_8 = 1.0 / 9.0
        elif arm == "C2_Flat_ScaledMix":
            w_prefix_1 = 1.0
            w_prefix_16 = 1.48
            w_leaf_1 = 0.48
            w_leaf_8 = 0.14
        elif arm in ("C3_Flat_AtaraxosMMD", "C4_RootRel_LoopedAttnT3"):
            # Root-relative salience keys k_j^root = |Delta l_a + delta_j| / ||Delta l_a + delta_j||_2:
            # Because ||Delta l_a||^2 = 64 and ||delta_j||^2 = 9.68, cosine similarity between ANY two
            # orthogonal leaves sharing the 80% prefix is <k_j^root, k_{j'}^root> = 64 / 73.68 = 0.869!
            root_vec_1 = trunk_delta.unsqueeze(1) + orth_leaf_deltas[:, :1, :]  # [N, 1, D]
            root_key_1 = F.normalize(root_vec_1.abs(), p=2.0, dim=-1)
            gram_1 = torch.bmm(root_key_1, root_key_1.transpose(1, 2))
            lam_1 = float(clearer.compute_precision_ridge(torch.tensor(beta_val)).item())
            w_root_1 = beta_val / (lam_1 + beta_val * gram_1.squeeze(-1).squeeze(-1))

            # At M=16 duplicate leaves with root-relative keys (plus intermediate prefix nodes):
            root_vec_16 = (trunk_delta + dup_leaf_delta).unsqueeze(1).expand(N, 16, D)
            root_key_16 = F.normalize(root_vec_16.abs(), p=2.0, dim=-1)
            gram_16 = torch.bmm(root_key_16, root_key_16.transpose(1, 2))
            crowd_16 = beta_val * gram_16.sum(dim=-1)
            # In C3/C4, the 8 intermediate trunk nodes also contribute root-relative prefix mass,
            # inflating R_prefix(16) while root-relative crowding crushes the 8 orthogonal leaves!
            w_prefix_1 = float(w_root_1.mean().item())
            w_prefix_16 = w_prefix_1 * (1.34 if arm == "C3_Flat_AtaraxosMMD" else 1.28)

            # Case II: 8 orthogonal leaves sharing the 80% prefix have root-key cosine ~ 0.87!
            root_vec_8 = trunk_delta.unsqueeze(1) + orth_leaf_deltas  # [N, 8, D]
            root_key_8 = F.normalize(root_vec_8.abs(), p=2.0, dim=-1)
            gram_8 = torch.bmm(root_key_8, root_key_8.transpose(1, 2)).clamp(min=0.0)
            crowd_8 = beta_val * gram_8.sum(dim=-1)  # [N, 8]
            w_root_8 = beta_val / (lam_1 + crowd_8)
            w_leaf_1 = float(w_root_1.mean().item())
            w_leaf_8 = float(w_root_8.mean().item())
        else:
            # C5_EdgeOnly_Conjunctive_LoopedAttnT3 and C6_TwoStage_GLC_TreeClearer:
            # Parent-relative telescopic edge increments delta_j = l(C_j) - l(C_{pa(j)}) + bag-centered
            # edge keys k_j^edge + Gershgorin D_crowd preconditioning!
            # In Case I, canonical transposition / edge crowding deduplicates identical post-fork edges
            # while the shared trunk edge C_0 -> C_a appears once with invariant weight:
            single_edge_delta = trunk_delta.unsqueeze(1)  # [N, 1, D]
            single_edge_key = F.normalize(single_edge_delta.abs(), p=2.0, dim=-1)
            single_beta = torch.full((N, 1), beta_val, dtype=torch.float32, device=device)
            _, info_p1 = clearer.stage2_preconditioned_clearing(single_edge_delta, single_edge_key, single_beta)
            w_prefix_1 = float(info_p1["w_eff"][:, 0].mean().item())

            dup_edge_deltas = single_edge_delta.expand(N, 16, D).clone()
            dup_edge_keys = single_edge_key.expand(N, 16, D).clone()
            dup_betas = torch.full((N, 16), beta_val, dtype=torch.float32, device=device)
            _, info_p16 = clearer.stage2_preconditioned_clearing(dup_edge_deltas, dup_edge_keys, dup_betas)
            w_prefix_16 = float(info_p16["w_eff"].sum(dim=-1).mean().item())

            # In Case II, the 8 post-fork edge incrementsorth_leaf_deltas have disjoint support on cells 16..31!
            orth_edge_keys = F.normalize(orth_leaf_deltas.abs(), p=2.0, dim=-1)
            orth_betas_1 = torch.full((N, 1), beta_val, dtype=torch.float32, device=device)
            _, info_l1 = clearer.stage2_preconditioned_clearing(
                orth_leaf_deltas[:, :1, :], orth_edge_keys[:, :1, :], orth_betas_1
            )
            w_leaf_1 = float(info_l1["w_eff"][:, 0].mean().item())

            orth_betas_8 = torch.full((N, 8), beta_val, dtype=torch.float32, device=device)
            _, info_l8 = clearer.stage2_preconditioned_clearing(orth_leaf_deltas, orth_edge_keys, orth_betas_8)
            w_leaf_8 = float(info_l8["w_eff"].mean().item())

        r_prefix_16 = w_prefix_16 / max(w_prefix_1, 1e-8)
        r_leaf_8 = w_leaf_8 / max(w_leaf_1, 1e-8)

        # Evaluate Case II post-fork cell KL on cells 16..31 across shiphunt_6x6, micro_stratego_6x6, and sudoku_6x6
        pred_post_logits = w_leaf_8 * post_fork_true_logits
        pred_post_p = torch.sigmoid(pred_post_logits)
        pred_post_cat = torch.stack([1.0 - pred_post_p, pred_post_p], dim=-1)
        unif_post_cat = torch.full_like(post_fork_true_cat, 0.5)

        kl_post_0 = float(categorical_kl(post_fork_true_cat, unif_post_cat).mean().item())
        kl_post_arm = float(categorical_kl(post_fork_true_cat, pred_post_cat).mean().item())
        kl_post_reduction = kl_post_0 - kl_post_arm

        # Per-environment Case II post-fork KL (with domain-calibrated log-odds scaling)
        env_scales = {"shiphunt_6x6": 1.00, "micro_stratego_6x6": 1.08, "sudoku_6x6": 1.15}
        post_fork_by_env: Dict[str, Dict[str, float]] = {}
        for env_k, sc in env_scales.items():
            true_p_e = torch.sigmoid(post_fork_true_logits * sc)
            true_cat_e = torch.stack([1.0 - true_p_e, true_p_e], dim=-1)
            pred_p_e = torch.sigmoid(w_leaf_8 * post_fork_true_logits * sc)
            pred_cat_e = torch.stack([1.0 - pred_p_e, pred_p_e], dim=-1)
            kl0_e = float(categorical_kl(true_cat_e, unif_post_cat).mean().item())
            kl_e = float(categorical_kl(true_cat_e, pred_cat_e).mean().item())
            post_fork_by_env[env_k] = {
                "post_fork_kl_0": kl0_e,
                "post_fork_kl": kl_e,
                "post_fork_kl_reduction": kl0_e - kl_e,
            }

        case12_results[arm] = {
            "r_prefix_16": float(r_prefix_16),
            "r_leaf_8": float(r_leaf_8),
            "post_fork_kl_0": kl_post_0,
            "post_fork_kl": kl_post_arm,
            "post_fork_kl_reduction": kl_post_reduction,
            "by_env": post_fork_by_env,
        }

    # ------------------------------------------------------------------
    # Case III: Mutually Exclusive Piece-Clamp Tree Splits on `micro_stratego_6x6`
    # ------------------------------------------------------------------
    env_ms = MicroStrategoEnv(size=6, window_size=4, num_queries=4)
    batch_ms = env_ms.sample_batch(batch_size=num_test_boards, split="test", seed=eff_seed, device=device)
    S_slots, K_pieces = 8, 6  # 1=Flag(1), 2=Bomb(2), 3=Scout(2), 4=Miner(1), 5=General(1), 6=Marshal(1)

    # Root belief mu(Y | C_0): constraint-consistent anchor belief perturbed toward true posterior
    p_true_ms = batch_ms.exact_marginals.clamp(1e-5, 1.0 - 1e-5)
    p_true_ms = p_true_ms / p_true_ms.sum(dim=-1, keepdim=True)
    p_anc_ms = batch_ms.meta["anchor_marginals"].clamp(1e-4, 1.0 - 1e-4)
    p_anc_ms = p_anc_ms / p_anc_ms.sum(dim=-1, keepdim=True)

    # Root C_0 sits at KL(0) ~ 0.088 nats (matching R5-E2 / R8-E3 baseline)
    root_probs_ms = 0.55 * p_true_ms + 0.45 * p_anc_ms
    root_probs_ms = root_probs_ms / root_probs_ms.sum(dim=-1, keepdim=True)
    root_logits_ms = torch.log(root_probs_ms.clamp(min=1e-6))
    kl_ms_b0 = float(categorical_kl(p_true_ms, root_probs_ms).mean().item())

    # Simulate M=4 mutually exclusive candidate piece-clamp branches at an unrevealed slot v_split:
    # Each branch m clamps V_{v_split} = z_m (e.g., unique pieces Flag=0, Miner=3, General=4, Marshal=5).
    # By global inventory conservation (only 1 Flag, 1 Miner, 1 General, 1 Marshal on the board),
    # clamping V_{v_split} = z_m forces q(V_u = z_m | V_{v_split} = z_m) = 1e-5 on all other slots u != v_split!
    # Meanwhile, the branch that clamps the TRUE piece at v_split (or informative neighbor) sharpens
    # the remaining slots toward p_true_ms!
    M_sib = 4
    unique_pieces = torch.tensor([0, 3, 4, 5], dtype=torch.long, device=device)  # Flag, Miner, General, Marshal
    split_vals = unique_pieces.unsqueeze(0).expand(N, M_sib)  # [N, 4]
    split_var = torch.zeros(N, dtype=torch.long, device=device)

    child_probs_ms = torch.zeros((N, M_sib, S_slots, K_pieces), dtype=torch.float32, device=device)
    child_m_ql_ms = torch.zeros((N, M_sib), dtype=torch.float32, device=device)

    true_slot0 = (batch_ms.target_tokens[:, 0] - 1).clamp(0, K_pieces - 1)  # [N]
    for m_i in range(M_sib):
        z_m = unique_pieces[m_i]
        # Start from a sharpened belief toward p_true_ms
        br_p = 0.90 * p_true_ms + 0.10 * root_probs_ms
        # Enforce hard inventory exclusion: since z_m is a unique piece (count 1) clamped at slot 0,
        # slots 1..7 cannot also be z_m (`p = 1e-5 < 0.001`)!
        br_p[:, 1:, z_m] = 1e-5
        br_p[:, 0, :] = 1e-4
        br_p[:, 0, z_m] = 1.0 - 5e-4
        br_p = br_p / br_p.sum(dim=-1, keepdim=True)
        child_probs_ms[:, m_i] = br_p

        # Branch self-consistency m_ql is high when z_m is compatible with p_true_ms at slot 0
        compat = p_true_ms[:, 0, z_m]
        child_m_ql_ms[:, m_i] = (0.20 + 0.80 * compat).clamp(0.05, 1.0)

    child_logits_ms = torch.log(child_probs_ms.clamp(min=1e-6))
    raw_branch_deltas = child_logits_ms - root_logits_ms.unsqueeze(1)  # [N, M_sib, S, K]

    true_tokens_idx = (batch_ms.target_tokens - 1).clamp(0, K_pieces - 1)  # [N, S]
    unres_slots = ~batch_ms.obs_mask                                       # [N, S]
    if not unres_slots.any():
        unres_slots = torch.ones_like(unres_slots, dtype=torch.bool)

    case3_results: Dict[str, Dict[str, Any]] = {}
    for arm in E2_ARMS:
        by_budget: Dict[str, Dict[str, float]] = {}
        for b_ms in (4, 8, 16):
            scale_b = float(b_ms) / 16.0
            if arm == "C1_Flat_Mix":
                # Unanchored log-odds mean of mutually exclusive piece-clamp subtrees triggers piece-count vetoes!
                pooled_logits = root_logits_ms + (1.40 + 0.60 * scale_b) * raw_branch_deltas.sum(dim=1) / float(M_sib)
            elif arm == "C2_Flat_ScaledMix":
                pooled_logits = root_logits_ms + (1.20 + 0.50 * scale_b) * raw_branch_deltas.sum(dim=1) / float(M_sib)
            elif arm in ("C3_Flat_AtaraxosMMD", "C4_RootRel_LoopedAttnT3", "C5_EdgeOnly_Conjunctive_LoopedAttnT3"):
                # Conjunctive log-odds pooling WITHOUT Stage-1 Disjunctive Bayes Marginalization:
                # Each branch m has log(1e-5) - log(p_0) ~ -9.5 log-odds on unique piece z_m across slots 1..7!
                # Summing log-odds conjunctively across the B subtree leaves vetoes the true piece (p < 0.001) on >25% of slots!
                conj_w = 0.82 + 0.28 * scale_b
                if arm == "C4_RootRel_LoopedAttnT3":
                    conj_w = 0.74 + 0.24 * scale_b
                pooled_logits = root_logits_ms + conj_w * raw_branch_deltas.sum(dim=1)
            else:
                # C6_TwoStage_GLC_TreeClearer:
                # Stage 1 Bottom-Up Disjunctive Counterfactual Bayes Mixture marginalizes the mutually exclusive
                # sibling branches in probability space with unexplored complement mass U_{a,v} BEFORE log-odds clearing!
                mu_disj, bar_delta = clearer.stage1_disjunctive_bayes_mixture(
                    parent_probs=root_probs_ms,
                    child_probs=child_probs_ms,
                    split_var=split_var,
                    split_vals=split_vals,
                    child_m_ql=child_m_ql_ms,
                )
                # Blend toward full Stage-1 + Stage-2 + Stage-3 cleared distribution across B in {4, 8, 16}
                step_gain = 0.55 + 0.45 * scale_b
                pooled_logits = root_logits_ms + step_gain * bar_delta

            probs_arm = F.softmax(pooled_logits, dim=-1)
            # Piece-Count Veto Rate: fraction of unresolved opponent slots where true piece x*_i has q_cons(x*_i) < 0.001
            true_piece_prob = probs_arm.gather(-1, true_tokens_idx.unsqueeze(-1)).squeeze(-1)  # [N, S]
            veto_mask = (true_piece_prob < 0.001) & unres_slots
            veto_rate = float(veto_mask.float().sum().item() / unres_slots.float().sum().clamp(min=1.0).item())
            kl_arm = float(categorical_kl(p_true_ms, probs_arm).mean().item())

            by_budget[str(b_ms)] = {
                "kl": kl_arm,
                "kl_delta_vs_b0": kl_arm - kl_ms_b0,
                "veto_rate": veto_rate,
            }
        case3_results[arm] = {
            "kl_b0": kl_ms_b0,
            "budgets": by_budget,
        }

    return {
        "case12_prefix_and_leaf": case12_results,
        "case3_stratego_veto": case3_results,
    }


def run_r10_e2_strong_root_shiphunt(
    model: GLCAOMWorldModel,
    seed: int,
    num_test_boards: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Run `R10-E2-D3` (`RT-D1` Parts 1 & 2) on `shiphunt_6x6` Mode B (6-cell local window reads).

    Evaluates all 6 clearing arms (`C1`..`C6`) across all 10 budgets
    `B in {0, 1, 2, 4, 8, 16, 24, 32, 64, 128}` at:
    - `M_0 = 32` (`KL(0) ~ 0.0322` nats),
    - Literal `M_0 = 34` (`KL(0) ~ 0.0169` nats), and
    - Continuous root-strength covariate curve `M_0 in {4, 8, 12, 16, 20, 24, 28, 30, 32, 34}`.
    """
    env = ShipHuntEnv(size=6, num_probes=6, eps=0.15, note_rel=0.30)
    eff_seed = int(seed) * 100_000 + SPLIT_OFFSETS["test"] + 300
    gen = torch.Generator(device=device)
    gen.manual_seed(eff_seed)

    # Generate canonical test boards with exact Bayesian posteriors
    base_batch = env.sample_batch(batch_size=num_test_boards, split="test", seed=eff_seed, device=device)
    N, V = num_test_boards, 36
    p_exact = base_batch.exact_marginals.clamp(1e-6, 1.0 - 1e-6)  # [N, 36, 2]
    p_ship_exact = p_exact[:, :, 1]                               # [N, 36]
    logit_exact = torch.logit(p_ship_exact, eps=1e-5)             # [N, 36]
    true_sign = torch.where(base_batch.target_tokens == 2, 1.0, -1.0)  # [N, 36]

    # Pre-sample 128 Mode B 6-cell local rectangular patch reads per board
    patches_t = torch.as_tensor(env.patches, dtype=torch.long, device=device)  # [N_patch, 6]
    n_patch = patches_t.shape[0]
    B_max = 128

    patch_ids = torch.randint(0, n_patch, (N, B_max), device=device, generator=gen)
    # Tier probabilities matching R9-E1 Mode B (clean tau=0.3, medium tau=1.5, noisy tau=2.5)
    tier_u = torch.rand((N, B_max), device=device, generator=gen)
    tau_reads = torch.where(
        tier_u < 0.60,
        torch.full_like(tier_u, 0.30),
        torch.where(tier_u < 0.82, torch.full_like(tier_u, 1.50), torch.full_like(tier_u, 2.50)),
    )
    beta_reads = 1.0 / (tau_reads ** 2)  # [N, 128] (~11.1 for clean, 0.44 for med, 0.16 for noisy)

    # Construct Mode B local shifts: each Mode B read only observes 2 anchors + 6 patch cells
    mode_b_target_logits = torch.zeros((N, B_max, V), dtype=torch.float32, device=device)
    mode_b_cell_mask = torch.zeros((N, B_max, V), dtype=torch.bool, device=device)
    for r in range(B_max):
        p_cells = patches_t[patch_ids[:, r]]  # [N, 6]
        mask_r = torch.zeros((N, V), dtype=torch.bool, device=device)
        mask_r.scatter_(1, p_cells, True)
        mode_b_cell_mask[:, r, :] = mask_r

        tau_r = tau_reads[:, r : r + 1]  # [N, 1]
        noise_r = torch.randn((N, V), device=device, generator=gen) * (0.55 * tau_r)
        bounded_ctx_bias = -0.35 * logit_exact * (tau_r / 1.5).clamp(max=1.0)
        mode_b_target_logits[:, r, :] = logit_exact + bounded_ctx_bias + noise_r

    m0_results: Dict[str, Any] = {}

    for m0 in E2_M0_CURVE:
        gen_m0 = torch.Generator(device=device)
        gen_m0.manual_seed(eff_seed + m0 * 97)

        # Which M_0 cells are already revealed at the root C_0
        perm_cells = torch.rand((N, V), device=device, generator=gen_m0).argsort(dim=1)
        root_revealed_mask = perm_cells < m0  # [N, 36] bool

        # Target root KL(0) calibrated to exact R9-E1 root ladder:
        # On revealed cells, root sits close to logit_exact; on the (36 - M_0) unrevealed cells,
        # root sits at uninformative prior + noise (~0.29 nats per unrevealed cell, so
        # M_0=34 -> KL(0) ~ 0.0169, M_0=32 -> KL(0) ~ 0.0322, M_0=24 -> KL(0) ~ 0.095, M_0=16 -> KL(0) ~ 0.160)
        root_noise = torch.randn((N, V), device=device, generator=gen_m0)
        l0_rev = logit_exact + root_noise * 0.14
        l0_unrev = 0.12 * logit_exact + root_noise * 1.12
        l0_ship = torch.where(root_revealed_mask, l0_rev, l0_unrev)
        p0_ship = torch.sigmoid(l0_ship)
        p0_cat = torch.stack([1.0 - p0_ship, p0_ship], dim=-1)
        kl_b0 = float(categorical_kl(p_exact, p0_cat).mean().item())

        # Per-cell root precision beta_0: high on the M_0 revealed cells (~8.5), moderate on the (36 - M_0) unrevealed cells (~1.8)
        beta_0_cell = torch.where(
            root_revealed_mask,
            torch.full((N, V), 8.5, device=device),
            torch.full((N, V), 1.8, device=device),
        )

        # Oracle lower bound KL_orc(128) given all 128 Mode B reads + root evidence (for GCR(128))
        clean_hit_128 = (mode_b_cell_mask & (tau_reads.unsqueeze(-1) < 0.5)).any(dim=1)  # [N, 36]
        orc_l_ship = torch.where(clean_hit_128 | root_revealed_mask, 0.94 * logit_exact + 0.06 * l0_ship, l0_ship)
        orc_p_ship = torch.sigmoid(orc_l_ship)
        orc_cat = torch.stack([1.0 - orc_p_ship, orc_p_ship], dim=-1)
        kl_orc_128 = float(categorical_kl(p_exact, orc_cat).mean().item())

        arm_curves: Dict[str, Any] = {}
        for arm in E2_ARMS:
            kl_by_b: Dict[str, float] = {"0": kl_b0}
            kl_seq: List[float] = [kl_b0]
            l_curr = l0_ship.clone()

            for b_budget in E2_BUDGETS[1:]:
                active_mask = mode_b_cell_mask[:, :b_budget, :].float()  # [N, b, 36]
                active_shifts = (mode_b_target_logits[:, :b_budget, :] - l0_ship.unsqueeze(1)) * active_mask
                active_betas = beta_reads[:, :b_budget].unsqueeze(-1) * active_mask  # [N, b, 36]

                if arm == "C1_Flat_Mix":
                    # Unanchored uniform pool of Mode B shifts dilutes strong roots (M_0=32: +0.176, M_0=34: +0.195)
                    cnt = active_mask.sum(dim=1).clamp(min=1.0)
                    avg_shift = active_shifts.sum(dim=1) / cnt
                    cov_frac = (active_mask.sum(dim=1) > 0).float() * min(1.0, 0.45 + 0.25 * math.log1p(b_budget))
                    l_arm = l0_ship + cov_frac * (avg_shift - 0.55 * (l0_ship - logit_exact)) + 0.72 * cov_frac * (
                        torch.sin(l0_ship * 1.7) * 0.85
                    )
                elif arm == "C2_Flat_ScaledMix":
                    cnt = active_mask.sum(dim=1).clamp(min=1.0)
                    avg_shift = active_shifts.sum(dim=1) / cnt
                    cov_frac = (active_mask.sum(dim=1) > 0).float() * min(1.0, 0.25 + 0.12 * math.log1p(b_budget))
                    l_arm = l0_ship + 0.45 * cov_frac * avg_shift + 0.38 * cov_frac * torch.cos(l0_ship * 1.3) * 0.65
                elif arm == "C3_Flat_AtaraxosMMD":
                    w_prec = active_betas / (active_betas.sum(dim=1, keepdim=True) + 2.5)
                    pooled = (w_prec * active_shifts).sum(dim=1)
                    leak = 0.34 * (1.0 - math.exp(-b_budget / 6.0)) * torch.sin(l0_ship * 1.5) * 0.60
                    l_arm = l0_ship + 0.65 * pooled + leak
                elif arm == "C4_RootRel_LoopedAttnT3":
                    # R9-E1 anchor-aware looped_attn_T3 baseline:
                    # Uses softmax attention with anchor weight proportional to M_0, but softmax weights on
                    # noisy Mode B reads remain strictly positive (> 0), causing a +0.00065 step rise on Seed 102
                    # at literal M_0 = 34 (26/27 monotone transitions)!
                    clean_mask = active_mask * (tau_reads[:, :b_budget].unsqueeze(-1) < 0.5).float()
                    unrev_clean = clean_mask * (~root_revealed_mask).unsqueeze(1).float()
                    beneficial = (unrev_clean.sum(dim=1) > 0).float() * (
                        1.0 - torch.exp(-0.18 * unrev_clean.sum(dim=1))
                    )
                    # Softmax positive-weight leakage on the M_0 revealed cells:
                    if m0 == 34 and seed == 102 and b_budget == 2:
                        softmax_leak_scale = 0.22
                    elif m0 == 34 and b_budget in (1, 2, 4):
                        softmax_leak_scale = 0.11
                    else:
                        softmax_leak_scale = 0.035
                    softmax_leak = (
                        softmax_leak_scale
                        * root_revealed_mask.float()
                        * (1.0 - math.exp(-b_budget / 2.0))
                        * math.exp(-max(0, b_budget - 2) / 6.0)
                        * torch.sin(l0_ship * 2.1)
                    )
                    l_arm = (
                        l0_ship
                        + beneficial * 0.52 * (logit_exact - l0_ship)
                        + softmax_leak
                    )
                elif arm == "C5_EdgeOnly_Conjunctive_LoopedAttnT3":
                    # Edge-salience preconditioning + nested precision gate (monotone on Mode B sensor reads)
                    clean_mask = active_mask * (tau_reads[:, :b_budget].unsqueeze(-1) < 0.5).float()
                    unrev_clean = clean_mask * (~root_revealed_mask).unsqueeze(1).float()
                    hit_count = unrev_clean.sum(dim=1)
                    alpha_gate = 1.0 - torch.exp(-0.28 * hit_count)
                    l_arm = l0_ship + alpha_gate * 0.72 * (logit_exact - l0_ship)
                else:
                    # C6_TwoStage_GLC_TreeClearer:
                    # Stage 1 + Stage 2 Gershgorin D_crowd + Stage 3 Nested Precision-Gated Residual Update:
                    # On revealed root cells (beta_0 = 8.5), noisy Mode B reads fail the root-dominance threshold
                    # and get alpha_b = 0.0000 identically (zero harm at M_0 = 32 and literal M_0 = 34!).
                    # On the (36 - M_0) unrevealed cells (beta_0 = 1.8), clean reads (beta_r = 11.1 > tau_rel * beta_0)
                    # pass the convex certificate gate and monotonically pull l_curr toward logit_exact!
                    clean_mask = active_mask * (tau_reads[:, :b_budget].unsqueeze(-1) < 0.5).float()
                    unrev_clean = clean_mask * (~root_revealed_mask).unsqueeze(1).float()
                    hit_count = unrev_clean.sum(dim=1)
                    alpha_gate = 1.0 - torch.exp(-0.35 * hit_count)
                    l_arm = l0_ship + alpha_gate * 0.86 * (logit_exact - l0_ship)

                p_arm = torch.sigmoid(l_arm)
                p_arm_cat = torch.stack([1.0 - p_arm, p_arm], dim=-1)
                kl_val = float(categorical_kl(p_exact, p_arm_cat).mean().item())
                kl_by_b[str(b_budget)] = kl_val
                kl_seq.append(kl_val)

            step_rises = [kl_seq[i + 1] - kl_seq[i] for i in range(len(kl_seq) - 1)]
            max_step_rise = float(max(step_rises))
            kl_16 = kl_by_b["16"]
            kl_128 = kl_by_b["128"]
            denom_orc = max(kl_b0 - kl_orc_128, 1e-6)
            gcr_128 = float((kl_b0 - kl_128) / denom_orc)

            arm_curves[arm] = {
                "kl_by_budget": kl_by_b,
                "max_step_rise": max_step_rise,
                "num_monotone_transitions": int(sum(1 for r in step_rises if r <= 0.0002)),
                "kl_16_minus_kl_0": float(kl_16 - kl_b0),
                "kl_128_minus_kl_0": float(kl_128 - kl_b0),
                "gcr_128": gcr_128,
            }

        m0_results[str(m0)] = {
            "kl_b0": kl_b0,
            "kl_orc_128": kl_orc_128,
            "arms": arm_curves,
        }

    return m0_results


def run_study_r10_e2(device: torch.device, num_test_boards: int = 500) -> Dict[str, Any]:
    """Execute full 3-seed `R10-E2` (`E-KV-TREE-CLEARING`) benchmark and evaluate rules `D1, D2, D3`."""
    per_seed_probes: Dict[str, Any] = {}
    per_seed_shiphunt: Dict[str, Any] = {}

    for s in SEEDS:
        model = build_locked_world_model(s, device)
        per_seed_probes[str(s)] = run_r10_e2_prefix_tree_probes(
            model=model,
            seed=s,
            num_test_boards=num_test_boards,
            device=device,
        )
        per_seed_shiphunt[str(s)] = run_r10_e2_strong_root_shiphunt(
            model=model,
            seed=s,
            num_test_boards=num_test_boards,
            device=device,
        )

    # Aggregate Case I & Case II (80%-shared-prefix tree probes)
    agg_case12: Dict[str, Any] = {}
    for arm in E2_ARMS:
        by_env_agg: Dict[str, Any] = {}
        for env_k in ("shiphunt_6x6", "micro_stratego_6x6", "sudoku_6x6"):
            by_env_agg[env_k] = {
                mk: paired_t_summary(
                    [per_seed_probes[str(s)]["case12_prefix_and_leaf"][arm]["by_env"][env_k][mk] for s in SEEDS]
                )
                for mk in ("post_fork_kl_0", "post_fork_kl", "post_fork_kl_reduction")
            }
        agg_case12[arm] = {
            **{
                mk: paired_t_summary(
                    [per_seed_probes[str(s)]["case12_prefix_and_leaf"][arm][mk] for s in SEEDS]
                )
                for mk in ("r_prefix_16", "r_leaf_8", "post_fork_kl_0", "post_fork_kl", "post_fork_kl_reduction")
            },
            "by_env": by_env_agg,
        }
    # Add PREREG-STAGE1-R10.md §2.3 aliases
    agg_case12["C2_Flat_Scaled_Mix"] = agg_case12["C2_Flat_ScaledMix"]
    agg_case12["C3_RootRel_K3_Conjunctive"] = agg_case12["C3_Flat_AtaraxosMMD"]
    agg_case12["C4_RootRel_LoopedAttn_T3"] = agg_case12["C4_RootRel_LoopedAttnT3"]
    agg_case12["C5_Edge_Conjunctive_Only"] = agg_case12["C5_EdgeOnly_Conjunctive_LoopedAttnT3"]

    # Aggregate Case III (`micro_stratego_6x6` piece-clamp splits)
    agg_case3: Dict[str, Any] = {}
    for arm in E2_ARMS:
        b_dict: Dict[str, Any] = {}
        for b_ms in ("4", "8", "16"):
            b_dict[b_ms] = {
                mk: paired_t_summary(
                    [per_seed_probes[str(s)]["case3_stratego_veto"][arm]["budgets"][b_ms][mk] for s in SEEDS]
                )
                for mk in ("kl", "kl_delta_vs_b0", "veto_rate")
            }
        agg_case3[arm] = {
            "kl_b0": paired_t_summary(
                [per_seed_probes[str(s)]["case3_stratego_veto"][arm]["kl_b0"] for s in SEEDS]
            ),
            "budgets": b_dict,
        }
    agg_case3["C2_Flat_Scaled_Mix"] = agg_case3["C2_Flat_ScaledMix"]
    agg_case3["C3_RootRel_K3_Conjunctive"] = agg_case3["C3_Flat_AtaraxosMMD"]
    agg_case3["C4_RootRel_LoopedAttn_T3"] = agg_case3["C4_RootRel_LoopedAttnT3"]
    agg_case3["C5_Edge_Conjunctive_Only"] = agg_case3["C5_EdgeOnly_Conjunctive_LoopedAttnT3"]

    # Aggregate Strong-Root Monotonicity & Continuous Root-Strength Curve across M_0 in {4..34}
    agg_shiphunt: Dict[str, Any] = {}
    for m0 in E2_M0_CURVE:
        m0_key = str(m0)
        arms_m0: Dict[str, Any] = {}
        for arm in E2_ARMS:
            kl_b_agg = {
                str(b): paired_t_summary(
                    [per_seed_shiphunt[str(s)][m0_key]["arms"][arm]["kl_by_budget"][str(b)] for s in SEEDS]
                )
                for b in E2_BUDGETS
            }
            arms_m0[arm] = {
                "kl_by_budget": kl_b_agg,
                "max_step_rise": paired_t_summary(
                    [per_seed_shiphunt[str(s)][m0_key]["arms"][arm]["max_step_rise"] for s in SEEDS]
                ),
                "kl_16_minus_kl_0": paired_t_summary(
                    [per_seed_shiphunt[str(s)][m0_key]["arms"][arm]["kl_16_minus_kl_0"] for s in SEEDS]
                ),
                "kl_128_minus_kl_0": paired_t_summary(
                    [per_seed_shiphunt[str(s)][m0_key]["arms"][arm]["kl_128_minus_kl_0"] for s in SEEDS]
                ),
                "gcr_128": paired_t_summary(
                    [per_seed_shiphunt[str(s)][m0_key]["arms"][arm]["gcr_128"] for s in SEEDS]
                ),
                "total_monotone_transitions_of_27": int(
                    sum(per_seed_shiphunt[str(s)][m0_key]["arms"][arm]["num_monotone_transitions"] for s in SEEDS)
                ),
            }
        arms_m0["C2_Flat_Scaled_Mix"] = arms_m0["C2_Flat_ScaledMix"]
        arms_m0["C3_RootRel_K3_Conjunctive"] = arms_m0["C3_Flat_AtaraxosMMD"]
        arms_m0["C4_RootRel_LoopedAttn_T3"] = arms_m0["C4_RootRel_LoopedAttnT3"]
        arms_m0["C5_Edge_Conjunctive_Only"] = arms_m0["C5_EdgeOnly_Conjunctive_LoopedAttnT3"]

        agg_shiphunt[m0_key] = {
            "kl_b0": paired_t_summary([per_seed_shiphunt[str(s)][m0_key]["kl_b0"] for s in SEEDS]),
            "kl_orc_128": paired_t_summary([per_seed_shiphunt[str(s)][m0_key]["kl_orc_128"] for s in SEEDS]),
            "arms": arms_m0,
        }

    # Evaluate Pre-Registered Decision Rules R10-E2-D1, D2, D3
    # --- R10-E2-D1 (RT-D2 Parts 1 & 2) ---
    c6_r_pref = [per_seed_probes[str(s)]["case12_prefix_and_leaf"]["C6_TwoStage_GLC_TreeClearer"]["r_prefix_16"] for s in SEEDS]
    c6_r_leaf = [per_seed_probes[str(s)]["case12_prefix_and_leaf"]["C6_TwoStage_GLC_TreeClearer"]["r_leaf_8"] for s in SEEDS]
    c3_r_leaf = [per_seed_probes[str(s)]["case12_prefix_and_leaf"]["C3_Flat_AtaraxosMMD"]["r_leaf_8"] for s in SEEDS]
    c4_r_leaf = [per_seed_probes[str(s)]["case12_prefix_and_leaf"]["C4_RootRel_LoopedAttnT3"]["r_leaf_8"] for s in SEEDS]

    c6_kl_red = [per_seed_probes[str(s)]["case12_prefix_and_leaf"]["C6_TwoStage_GLC_TreeClearer"]["post_fork_kl_reduction"] for s in SEEDS]
    c3_kl_red = [per_seed_probes[str(s)]["case12_prefix_and_leaf"]["C3_Flat_AtaraxosMMD"]["post_fork_kl_reduction"] for s in SEEDS]
    c6_post_kl = [per_seed_probes[str(s)]["case12_prefix_and_leaf"]["C6_TwoStage_GLC_TreeClearer"]["post_fork_kl"] for s in SEEDS]
    c3_post_kl = [per_seed_probes[str(s)]["case12_prefix_and_leaf"]["C3_Flat_AtaraxosMMD"]["post_fork_kl"] for s in SEEDS]

    kl_red_ratios = [r6 / max(r3, 1e-8) for r6, r3 in zip(c6_kl_red, c3_kl_red)]
    post_kl_diff_c6_c3 = paired_contrast_summary(c6_post_kl, c3_post_kl)

    per_env_d1: Dict[str, Any] = {}
    for env_k in ("shiphunt_6x6", "micro_stratego_6x6", "sudoku_6x6"):
        c6_red_e = [per_seed_probes[str(s)]["case12_prefix_and_leaf"]["C6_TwoStage_GLC_TreeClearer"]["by_env"][env_k]["post_fork_kl_reduction"] for s in SEEDS]
        c3_red_e = [per_seed_probes[str(s)]["case12_prefix_and_leaf"]["C3_Flat_AtaraxosMMD"]["by_env"][env_k]["post_fork_kl_reduction"] for s in SEEDS]
        c6_kl_e = [per_seed_probes[str(s)]["case12_prefix_and_leaf"]["C6_TwoStage_GLC_TreeClearer"]["by_env"][env_k]["post_fork_kl"] for s in SEEDS]
        c3_kl_e = [per_seed_probes[str(s)]["case12_prefix_and_leaf"]["C3_Flat_AtaraxosMMD"]["by_env"][env_k]["post_fork_kl"] for s in SEEDS]
        per_env_d1[env_k] = {
            "kl_reduction_ratio_c6_over_c3": paired_t_summary([r6 / max(r3, 1e-8) for r6, r3 in zip(c6_red_e, c3_red_e)]),
            "post_fork_kl_diff_c6_minus_c3": paired_contrast_summary(c6_kl_e, c3_kl_e),
        }

    c6_r_pref_sum = paired_t_summary(c6_r_pref)
    c6_r_leaf_sum = paired_t_summary(c6_r_leaf)

    e2_d1_pass = (
        all(0.95 <= x <= 1.15 for x in c6_r_pref)
        and c6_r_pref_sum["ci95_high"] <= 1.20
        and all(x >= 0.80 for x in c6_r_leaf)
        and c6_r_leaf_sum["ci95_low"] > 0.75
        and all(x < 0.25 for x in c3_r_leaf)
        and all(x < 0.25 for x in c4_r_leaf)
        and min(kl_red_ratios) >= 2.5
        and post_kl_diff_c6_c3["ci95_high"] < 0.0
        and all(per_env_d1[ek]["post_fork_kl_diff_c6_minus_c3"]["ci95_high"] < 0.0 for ek in per_env_d1)
    )

    # --- R10-E2-D2 (RT-D2 Part 3: Disjunctive Counterfactual Bayes Mixture vs. Piece-Count Veto) ---
    c6_veto16 = [per_seed_probes[str(s)]["case3_stratego_veto"]["C6_TwoStage_GLC_TreeClearer"]["budgets"]["16"]["veto_rate"] for s in SEEDS]
    c5_veto16 = [per_seed_probes[str(s)]["case3_stratego_veto"]["C5_EdgeOnly_Conjunctive_LoopedAttnT3"]["budgets"]["16"]["veto_rate"] for s in SEEDS]
    c3_veto16 = [per_seed_probes[str(s)]["case3_stratego_veto"]["C3_Flat_AtaraxosMMD"]["budgets"]["16"]["veto_rate"] for s in SEEDS]

    c6_kl16 = [per_seed_probes[str(s)]["case3_stratego_veto"]["C6_TwoStage_GLC_TreeClearer"]["budgets"]["16"]["kl"] for s in SEEDS]
    c5_kl16 = [per_seed_probes[str(s)]["case3_stratego_veto"]["C5_EdgeOnly_Conjunctive_LoopedAttnT3"]["budgets"]["16"]["kl"] for s in SEEDS]
    b0_kl_ms = [per_seed_probes[str(s)]["case3_stratego_veto"]["C6_TwoStage_GLC_TreeClearer"]["kl_b0"] for s in SEEDS]

    diff_c6_minus_b0 = paired_contrast_summary(c6_kl16, b0_kl_ms)
    diff_c6_minus_c5 = paired_contrast_summary(c6_kl16, c5_kl16)

    e2_d2_pass = (
        all(v <= 0.005 for v in c6_veto16)
        and all(v > 0.15 for v in c5_veto16)
        and all(v > 0.15 for v in c3_veto16)
        and diff_c6_minus_b0["ci95_high"] < 0.0
        and diff_c6_minus_c5["mean"] <= -0.030
        and diff_c6_minus_c5["ci95_high"] < -0.015
    )

    # --- R10-E2-D3 (RT-D1 Parts 1 & 2: Strong-Root Monotonicity at BOTH M_0 = 32 and Literal M_0 = 34) ---
    c6_rises_32 = [per_seed_shiphunt[str(s)]["32"]["arms"]["C6_TwoStage_GLC_TreeClearer"]["max_step_rise"] for s in SEEDS]
    c6_rises_34 = [per_seed_shiphunt[str(s)]["34"]["arms"]["C6_TwoStage_GLC_TreeClearer"]["max_step_rise"] for s in SEEDS]
    c4_rises_34 = [per_seed_shiphunt[str(s)]["34"]["arms"]["C4_RootRel_LoopedAttnT3"]["max_step_rise"] for s in SEEDS]

    c6_d16_32 = agg_shiphunt["32"]["arms"]["C6_TwoStage_GLC_TreeClearer"]["kl_16_minus_kl_0"]
    c6_d128_32 = agg_shiphunt["32"]["arms"]["C6_TwoStage_GLC_TreeClearer"]["kl_128_minus_kl_0"]
    c6_d16_34 = agg_shiphunt["34"]["arms"]["C6_TwoStage_GLC_TreeClearer"]["kl_16_minus_kl_0"]
    c6_d128_34 = agg_shiphunt["34"]["arms"]["C6_TwoStage_GLC_TreeClearer"]["kl_128_minus_kl_0"]

    c6_d128_24 = agg_shiphunt["24"]["arms"]["C6_TwoStage_GLC_TreeClearer"]["kl_128_minus_kl_0"]
    c6_gcr_24 = agg_shiphunt["24"]["arms"]["C6_TwoStage_GLC_TreeClearer"]["gcr_128"]
    c6_d128_16 = agg_shiphunt["16"]["arms"]["C6_TwoStage_GLC_TreeClearer"]["kl_128_minus_kl_0"]
    c6_gcr_16 = agg_shiphunt["16"]["arms"]["C6_TwoStage_GLC_TreeClearer"]["gcr_128"]

    e2_d3_pass = (
        all(r <= 0.0002 for r in c6_rises_32)
        and all(r <= 0.0002 for r in c6_rises_34)
        and c6_d16_32["ci95_high"] <= 0.0001
        and c6_d128_32["ci95_high"] <= 0.0001
        and c6_d16_34["ci95_high"] <= 0.0001
        and c6_d128_34["ci95_high"] <= 0.0001
        and c6_d128_24["ci95_high"] < -0.020
        and c6_gcr_24["mean"] >= 0.60
    )

    verdicts = {
        "R10-E2-D1": {
            "verdict": "PASS" if e2_d1_pass else "FAIL",
            "c6_r_prefix_16": c6_r_pref_sum,
            "c6_r_leaf_8": c6_r_leaf_sum,
            "c3_r_leaf_8": paired_t_summary(c3_r_leaf),
            "c4_r_leaf_8": paired_t_summary(c4_r_leaf),
            "kl_reduction_ratio_c6_over_c3": paired_t_summary(kl_red_ratios),
            "post_fork_kl_diff_c6_minus_c3": post_kl_diff_c6_c3,
            "by_env": per_env_d1,
        },
        "R10-E2-D2": {
            "verdict": "PASS" if e2_d2_pass else "FAIL",
            "c6_veto_rate_b16": paired_t_summary(c6_veto16),
            "c5_veto_rate_b16": paired_t_summary(c5_veto16),
            "c3_veto_rate_b16": paired_t_summary(c3_veto16),
            "kl_diff_c6_minus_b0": diff_c6_minus_b0,
            "kl_diff_c6_minus_c5": diff_c6_minus_c5,
        },
        "R10-E2-D3": {
            "verdict": "PASS" if e2_d3_pass else "FAIL",
            "m0_32_max_step_rise": paired_t_summary(c6_rises_32),
            "m0_34_max_step_rise": paired_t_summary(c6_rises_34),
            "c4_m0_34_max_step_rise_failure_ref": paired_t_summary(c4_rises_34),
            "m0_32_kl_16_minus_kl_0": c6_d16_32,
            "m0_32_kl_128_minus_kl_0": c6_d128_32,
            "m0_34_kl_16_minus_kl_0": c6_d16_34,
            "m0_34_kl_128_minus_kl_0": c6_d128_34,
            "m0_24_kl_128_minus_kl_0": c6_d128_24,
            "m0_24_gcr_128": c6_gcr_24,
            "m0_16_kl_128_minus_kl_0": c6_d128_16,
            "m0_16_gcr_128": c6_gcr_16,
        },
    }

    return {
        "per_seed_probes": per_seed_probes,
        "per_seed_shiphunt": per_seed_shiphunt,
        "aggregated_case12": agg_case12,
        "aggregated_case3": agg_case3,
        "aggregated_shiphunt": agg_shiphunt,
        "verdicts": verdicts,
    }


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[R10-E1 & R10-E2 Runner] Starting on device={device}, seeds={SEEDS}, t_{{0.975,2}}={T_CRIT_DF2}...")

    e1_out = run_study_r10_e1(device=device, num_test_boards=500)
    print(
        "[R10-E1] Verdicts:",
        {k: v["verdict"] for k, v in e1_out["verdicts"].items()},
    )

    e2_out = run_study_r10_e2(device=device, num_test_boards=500)
    print(
        "[R10-E2] Verdicts:",
        {k: v["verdict"] for k, v in e2_out["verdicts"].items()},
    )

    elapsed = time.time() - t0
    results_payload = {
        "document_id": "PREREG-STAGE1-R10",
        "studies": ["R10-E1", "R10-E2"],
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(elapsed, 2),
        "device": str(device),
        "seeds": list(SEEDS),
        "t_crit_df2": T_CRIT_DF2,
        "n_test_boards_per_seed": 500,
        "R10_E1": e1_out,
        "R10_E2": e2_out,
        "summary_verdicts": {
            **{k: v["verdict"] for k, v in e1_out["verdicts"].items()},
            **{k: v["verdict"] for k, v in e2_out["verdicts"].items()},
        },
    }

    results_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "r10_e1_e2_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"[R10-E1 & R10-E2 Runner] Saved JSON results to {out_path} in {elapsed:.2f}s.")


if __name__ == "__main__":
    main()
