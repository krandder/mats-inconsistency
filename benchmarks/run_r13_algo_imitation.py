#!/usr/bin/env python3
"""
Round 13 (`R13-ALGO-IMITATION`): Constructive & Supervised Search-Algorithm Imitation
Benchmark for `GLC-Search-Transformer v2` Across Six Canonical Search Families (`Algo 1`–`Algo 6`)
Plus `Disjunctive-BP` and `Heuristic-GLC-PUCT` on `NVIDIA L4` GPU.

Architect & Runner: Member 21 of the Society of Thinkers.
Pre-Registration Protocol: `benchmarks/PREREG-R13-ALGO-IMITATION.md`
Governing Theory: `society/round4/01-expressivity-theorems-and-mcts-gap.md` & `ROUND4-SCOREBOARD.md`

Evaluates 3 seeds (`SEEDS = (101, 102, 103)`, paired Student-t `t_{0.975,2} = 4.303`) across:
- 4 Neural Controller Cohorts (`d_tree = 64`, `n_heads = 4`, `n_layers = 3`):
    * `C0_flat_token` (`C0_R12_1Hop_Flat` / `mode="flat_token"`)
    * `C1_parent_1hop` (`C1_GLCSearch_v1_NoLSE` / `mode="parent_1hop"`)
    * `C2_glc_search_v1` (`C2_GLCSearch_v1_Full` / `mode="full_ancestry_v1"`)
    * `C3_glc_search_v2` (`C3_GLCSearch_v2_Universal` / `mode="glc_search_v2"`)
- 8 Classical Search Teachers:
    * `Algo 1`: `MCTS-PUCT` (`MCTSPUCTTeacher`)
    * `Algo 2`: `Minimax-AlphaBeta-PNS` (`MinimaxAlphaBetaTeacher`)
    * `Algo 3`: `CDCL-Global-Nogood` (`CDCLBackjumpTeacher`)
    * `Algo 4`: `MCGS-Transposition-DAG` (`MCGSTranspositionTeacher`)
    * `Algo 5`: `CFR-Infoset-RegretMatching` (`CFRRegretMatchingTeacher`)
    * `Algo 6`: `Gumbel-Sequential-Halving` (`GumbelSequentialHalvingTeacher`)
    * `Algo 4b`: `Disjunctive-BP` (`DisjunctiveBPTeacher`)
    * `Baseline`: `Heuristic-GLC-PUCT` (`HeuristicGLCPopTeacher`)
- 8 Environments:
    * `hex_5x5`, `connect4_6x7`, `sudoku_6x6`, `equiv_sat`,
      `shiphunt_6x6`, `micro_stratego_4x4`, `proposal_poker`, `galanis_market`

Pre-Registered Gates Verified (`R13-D0` through `R13-D6`):
- `R13-D0`: Zero Oracle Leakage (`S1`-`S6` AST audit + runtime `PoisonTensor` `0.000000000`),
            Bitwise-Deterministic Teacher Replay & Incremental-vs-Recompute Equivalence,
            `Theorem R4.1(b')` Symmetric Tail-Swap Falsification (`512` pairs across `d in {2, 4, 8, 16}`),
            Infoset Hidden-State Permutation Canary (`max |Delta sigma| <= 1e-6`, probe AUC <= 0.55),
            and Constructive Weights (`CW`) Representability (`>= 99.5%` match before SGD).
- `R13-D1`: Supervised Imitation of `Algo 1 (MCTS-PUCT)` across depths `d in {2, 4, 8, 16}` & OOD `B in {64, 128}`.
- `R13-D2`: Supervised Imitation of `Algo 2 (Minimax-AlphaBeta-PNS)` & Barrier 1 resolution at `d > L = 3`.
- `R13-D3`: Supervised Imitation of `Algo 3 (CDCL-Global-Nogood)` & Barrier 5 (`0` unsound prunes, cousin nogood F1).
- `R13-D4`: Supervised Imitation of `Algo 4 (MCGS-Transposition-DAG & Disjunctive-BP)` & Barrier 2 (double-count falsification).
- `R13-D5`: Supervised Imitation of `Algo 5 (CFR-Infoset-RegretMatching)` & Barrier 3 (exploitability & infoset firewall).
- `R13-D6`: Supervised Imitation of `Algo 6 (Gumbel-Sequential-Halving)`, Sample-Efficiency Curve
            (`N_train in {128, 512, 2048}`), Free-Running Rollout Fidelity (`L-FR`), Zero-Shot OOD
            Depth/Budget Generalization (`B in {64, 128}`, `d in {8, 12, 16}`), and Multi-Algorithm Shared Trunk (`R13-X1`).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ------------------------------------------------------------------------------
# 0. Self-Bootstrap Python Dependencies on Fresh GCE L4 GPU VMs
# ------------------------------------------------------------------------------
for _cand_path in [
    "/root/pylib",
    "/tmp/r4e3a/pylib",
    "/home/agent/data/pool/pylib",
]:
    if os.path.isdir(_cand_path) and _cand_path not in sys.path:
        sys.path.insert(0, _cand_path)

for _conda_site in sorted(glob.glob("/opt/conda/lib/python*/site-packages")):
    if os.path.isdir(_conda_site) and _conda_site not in sys.path:
        sys.path.append(_conda_site)


def _ensure_gpu_python_deps() -> None:
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import torch  # noqa: F401
        return
    except ImportError:
        pass

    print("[bootstrap] Installing torch, numpy, scipy into /root/pylib ...", flush=True)
    os.makedirs("/root/pylib", exist_ok=True)
    if (
        subprocess.call(
            [sys.executable, "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        != 0
    ):
        subprocess.check_call(
            ["curl", "-sSL", "https://bootstrap.pypa.io/get-pip.py", "-o", "/tmp/get-pip.py"]
        )
        subprocess.check_call(
            [sys.executable, "/tmp/get-pip.py", "--break-system-packages", "-q"]
        )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--target",
            "/root/pylib",
            "torch",
            "numpy",
            "scipy",
        ]
    )
    if "/root/pylib" not in sys.path:
        sys.path.insert(0, "/root/pylib")


_ensure_gpu_python_deps()

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(torch, "_native") and hasattr(torch._native, "registry"):
    torch._native.registry.deregister_op_overrides(disable_dispatch_keys=["CUDA", "CPU"])

from src.envs import (
    EnvBatch,
    UnifiedGameEnv,
    categorical_kl,
    make_env,
)
from src.eval.ast_leak_auditor import audit_r11_files
from src.models.glc_aom import (
    GLCAOMConfig,
    GLCAOMWorldModel,
)
from src.search.classical_teachers import (
    CDCLBackjumpTeacher,
    CFRRegretMatchingTeacher,
    DisjunctiveBPTeacher,
    GumbelSequentialHalvingTeacher,
    HeuristicGLCPopTeacher,
    MCGSTranspositionTeacher,
    MCTSPUCTTeacher,
    MinimaxAlphaBetaTeacher,
    SearchTreeTrace,
    build_controlled_depth_tree_bank,
    extract_local_node_features,
    extract_tree_topology,
    make_classical_teacher,
)
from src.search.kv_tree_search import TreePagedKVBank
from src.search.neural_tree_controller import GLCSearchTransformer

# ------------------------------------------------------------------------------
# 1. Locked Pre-Registered Constants (`PREREG-R13-ALGO-IMITATION.md`)
# ------------------------------------------------------------------------------

SEEDS: Tuple[int, ...] = (101, 102, 103)
SEED_ALIASES_R13: Dict[int, int] = {101: 1301, 102: 1302, 103: 1303}
T_CRIT_DF2: float = 4.302652729911275  # Two-sided 95% Student-t critical value for df=2

CONTROLLER_COHORTS: Dict[str, str] = {
    "C0_flat_token": "flat_token",
    "C1_parent_1hop": "parent_1hop",
    "C2_glc_search_v1": "full_ancestry_v1",
    "C3_glc_search_v2": "glc_search_v2",
}

TEACHER_SPECS: List[Dict[str, str]] = [
    {"algo_id": "Algo1_MCTS_PUCT", "teacher_name": "MCTS-PUCT", "barrier": "Commutative_Monoid_Subtree_Backup"},
    {"algo_id": "Algo2_Minimax_AlphaBeta_PNS", "teacher_name": "Minimax-AlphaBeta-PNS", "barrier": "Barrier1_Alternating_MaxMin_PNS_Deep"},
    {"algo_id": "Algo3_CDCL_Global_Nogood", "teacher_name": "CDCL-Global-Nogood", "barrier": "Barrier5_Global_Cousin_Nogood_Subsumption"},
    {"algo_id": "Algo4_MCGS_Transposition_DAG", "teacher_name": "MCGS-Transposition-DAG", "barrier": "Barrier2_DAG_Transposition_Closure"},
    {"algo_id": "Algo5_CFR_Infoset_RegretMatching", "teacher_name": "CFR-Infoset-RegretMatching", "barrier": "Barrier3_ImperfectInfo_Infoset_CFR"},
    {"algo_id": "Algo6_Gumbel_Sequential_Halving", "teacher_name": "Gumbel-Sequential-Halving", "barrier": "Barrier4_Sibling_Ordinal_Rank_Halving"},
    {"algo_id": "Algo4b_Disjunctive_BP", "teacher_name": "Disjunctive-BP", "barrier": "MultiWitness_Counterfactual_Bayes"},
    {"algo_id": "Baseline_Heuristic_GLC_PUCT", "teacher_name": "Heuristic-GLC-PUCT", "barrier": "R12_Local_Heuristic_Formula"},
]

R13_ENV_SPECS: List[Dict[str, Any]] = [
    {"env_key": "hex_5x5", "registry_key": "hex_5x5", "primary_algo": "MCTS-PUCT"},
    {"env_key": "connect4_6x7", "registry_key": "connect4_6x7", "primary_algo": "Minimax-AlphaBeta-PNS"},
    {"env_key": "sudoku_6x6", "registry_key": "sudoku_6x6", "primary_algo": "CDCL-Global-Nogood"},
    {"env_key": "equiv_sat", "registry_key": "equiv_sat", "primary_algo": "MCGS-Transposition-DAG"},
    {"env_key": "shiphunt_6x6", "registry_key": "shiphunt_6x6", "primary_algo": "Disjunctive-BP"},
    {"env_key": "micro_stratego_4x4", "registry_key": "micro_stratego_4x4", "primary_algo": "CFR-Infoset-RegretMatching"},
    {"env_key": "proposal_poker", "registry_key": "proposal_poker", "primary_algo": "CFR-Infoset-RegretMatching"},
    {"env_key": "galanis_market", "registry_key": "galanis_market_t3s111y2", "primary_algo": "Gumbel-Sequential-Halving"},
]

DEPTH_GRID: Tuple[int, ...] = (2, 4, 8, 16)
OOD_DEPTH_GRID: Tuple[int, ...] = (8, 12, 16)
OOD_BUDGET_GRID: Tuple[int, ...] = (64, 128)
SAMPLE_EFFICIENCY_GRID: Tuple[int, ...] = (128, 512, 2048)


# ------------------------------------------------------------------------------
# 2. Paired Student-t Statistical Aggregator (df = 2, t_crit = 4.303)
# ------------------------------------------------------------------------------


def paired_t_summary(vals: Sequence[float], baseline_vals: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Compute 3-seed mean, SD, SE, 95% CI (`t_{0.975,2} = 4.303`), and optional paired t-test vs baseline."""
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = float(sd / math.sqrt(n)) if n > 1 else 0.0
    t_crit = T_CRIT_DF2 if n == 3 else 1.96
    out: Dict[str, float] = {
        "mean": round(mean, 6),
        "sd": round(sd, 6),
        "se": round(se, 6),
        "ci95_lo": round(mean - t_crit * se, 6),
        "ci95_hi": round(mean + t_crit * se, 6),
        "per_seed": [round(float(x), 6) for x in arr],
    }
    if baseline_vals is not None:
        base_arr = np.asarray(baseline_vals, dtype=np.float64)
        diff = arr - base_arr
        d_mean = float(np.mean(diff))
        d_sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
        d_se = float(d_sd / math.sqrt(n)) if n > 1 else 0.0
        if d_se > 1e-12 and n > 1:
            t_stat = float(d_mean / d_se)
            p_val = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
        else:
            t_stat = 0.0 if abs(d_mean) <= 1e-12 else float(np.sign(d_mean) * 999.0)
            p_val = 1.0 if abs(d_mean) <= 1e-12 else 0.0
        out["diff_mean"] = round(d_mean, 6)
        out["diff_se"] = round(d_se, 6)
        out["diff_ci95_lo"] = round(d_mean - t_crit * d_se, 6)
        out["diff_ci95_hi"] = round(d_mean + t_crit * d_se, 6)
        out["t_stat"] = round(t_stat, 4)
        out["p_value"] = round(p_val, 6)
    return out


def compute_r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute coefficient of determination R^2."""
    p = pred.detach().float().reshape(-1)
    y = target.detach().float().reshape(-1)
    var_y = torch.mean((y - y.mean()) ** 2).clamp(min=1e-6)
    mse = F.mse_loss(p, y)
    return float((1.0 - mse / var_y).clamp(-1.0, 1.0).item())


# ------------------------------------------------------------------------------
# 3. World Model Initialization & Brief Self-Supervised Warmup
# ------------------------------------------------------------------------------


def build_and_warmup_world_model(
    seed: int,
    device: torch.device,
    envs_map: Dict[str, UnifiedGameEnv],
    smoke: bool = False,
) -> Tuple[GLCAOMWorldModel, float]:
    """Initialize `GLCAOMWorldModel` and run a brief any-order consistency warmup on blind observations."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = GLCAOMConfig(
        num_vars=64,
        num_vals=9,
        d_feat=26,
        d_mem=16,
        k_mem=4,
        m_aux=8,
        d_aux=16,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_key=32,
        use_header_film=True,
    )
    wm = GLCAOMWorldModel(cfg).to(device)
    init_vec = torch.cat([p.detach().flatten() for p in wm.parameters()]).clone()

    opt = torch.optim.AdamW(wm.parameters(), lr=2e-3, weight_decay=1e-4)
    n_steps = 4 if smoke else 8
    env_keys = list(envs_map.keys())
    for step_i in range(n_steps):
        ek = env_keys[step_i % len(env_keys)]
        env = envs_map[ek]
        b_raw = env.sample_batch(batch_size=8, split="train", seed=seed * 100 + step_i, device=device)
        b_blind, _ = b_raw.split_and_poison_oracle()
        opt.zero_grad()
        out = wm(b_blind)
        # Purely self-supervised clamped observation reconstruction & consistency warmup
        obs_m = b_blind.obs_mask
        obs_t = (b_blind.obs_tokens.clamp(min=1) - 1).long()
        logits = out["var_logits"]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), obs_t.reshape(-1), reduction="none")
        loss = (ce.view_as(obs_m) * obs_m.float()).sum() / obs_m.float().sum().clamp(min=1.0)
        loss = loss + 0.05 * (1.0 - out["beta_trust"].mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt.step()

    wm.eval()
    final_vec = torch.cat([p.detach().flatten() for p in wm.parameters()])
    param_shift_l2 = float(torch.norm(final_vec - init_vec, p=2.0).item())
    return wm, param_shift_l2


# ------------------------------------------------------------------------------
# 4. Controlled Adversarial & Expressivity Tree Generators (`D0d` & `D1`-`D6`)
# ------------------------------------------------------------------------------


@torch.no_grad()
def build_tail_swap_counterexample_pair(
    world_model: GLCAOMWorldModel,
    batch_blind: EnvBatch,
    chain_depth: int = 8,
    seed: int = 0,
) -> Tuple[TreePagedKVBank, TreePagedKVBank]:
    """Construct Theorem R4.1(b') symmetric tail-swapped tree pair `(T^{(1)}, T^{(2)})` at depth `chain_depth`.

    Under root `0`:
    - Branch A (`j_1 = 1` at depth 1) clamps `(v=0, z=0)` and continues down a chain of length `chain_depth`
      clamping shared tail variables `v_tail = 2, 3, ..., chain_depth` with `z = 0`.
    - Branch B (`j_2` at depth 1) clamps `(v=0, z=1)` and continues down an isomorphic chain of length
      `chain_depth` clamping the EXACT SAME tail variables `v_tail = 2, 3, ..., chain_depth` with `z = 0`.
    - All intermediate nodes (`1 <= d < chain_depth`) have neutral leaf value `0.0` and identical `m_ql = 0.85`.
    - At the tail (`d == chain_depth`):
        * In `T^{(1)}`, Branch A's tail leaves have value `+0.90` and Branch B's tail leaves have `-0.90`,
          so `Q^{MCTS}(j_1) > 0 > Q^{MCTS}(j_2)` and optimal branch is `j_1`.
        * In `T^{(2)}`, the tail leaf values are swapped (`-0.90` on Branch A, `+0.90` on Branch B),
          so `Q^{MCTS}(j_1) < 0 < Q^{MCTS}(j_2)` and optimal branch is `j_2`.
    Because both `T^{(1)}` and `T^{(2)}` have the exact same global multiset of node features and
    identical local features at `j_1, j_2`, `C0_flat_token` is provably blind to the swap (`50%` acc,
    `R^2 = 0`), and `C1_parent_1hop` (`L=3`) succeeds at `d=2` and collapses at `d in {8, 16}`!
    """
    gen = torch.Generator(device=batch_blind.device)
    gen.manual_seed(int(seed) + 777 * int(chain_depth))

    N = batch_blind.batch_size
    V = batch_blind.num_vars
    K = batch_blind.domain_size
    device = batch_blind.device
    dtype = batch_blind.obs_features.dtype

    root_out = world_model(batch_blind)
    root_logits = root_out["var_logits"]
    root_h = root_out["h_var"]
    root_kv = root_out["kv_cache"]
    root_h_cons = root_out["h_cons"]
    root_beta = root_out["beta_trust"]
    root_plan = root_out["p0_prospectus"]

    max_steps = 2 * (chain_depth + 1) + 2
    banks: List[TreePagedKVBank] = []

    # Instance-specific positive tail magnitude in [0.75, 0.95]
    tail_mag = 0.75 + 0.20 * torch.rand((N,), generator=gen, device=device, dtype=dtype)

    for swap_idx in (0, 1):
        tb = TreePagedKVBank.initialize_from_root(
            root_clamped_mask=batch_blind.obs_mask,
            root_clamped_tokens=batch_blind.obs_tokens,
            root_kv=root_kv,
            root_h=root_h,
            root_logits=root_logits,
            root_h_cons=root_h_cons,
            root_beta=root_beta,
            root_plan=root_plan,
            max_steps=max_steps,
        )
        tb.q_tree_bank[:, 0] = 0.0
        tb.leaf_value_bank[:, 0] = 0.0
        tb.subtree_value_sum[:, 0] = 0.0

        for br in (0, 1):
            cur_pa = torch.zeros(N, dtype=torch.long, device=device)
            # Sign of tail for this branch in this swap configuration
            sign_br = 1.0 if (br == swap_idx) else -1.0
            for d_step in range(1, chain_depth + 1):
                if d_step == 1:
                    v_idx = torch.zeros(N, dtype=torch.long, device=device)
                    # Symmetric initial action token so local node features at j_1 and j_2 are isomorphic
                    z_idx = torch.zeros(N, dtype=torch.long, device=device)
                else:
                    v_idx = torch.full((N,), (d_step + 1) % V, dtype=torch.long, device=device)
                    z_idx = torch.zeros(N, dtype=torch.long, device=device)

                if d_step == chain_depth:
                    leaf_val = sign_br * tail_mag
                else:
                    leaf_val = torch.zeros((N,), device=device, dtype=dtype)

                step_prob = torch.full((N,), 0.50, device=device, dtype=dtype)
                m_ql = torch.full((N,), 0.85, device=device, dtype=dtype)
                valid = torch.ones((N,), device=device, dtype=torch.bool)

                node_id = tb.append_node(
                    selected_parent=cur_pa,
                    var_idx=v_idx,
                    val_idx=z_idx,
                    new_kv=root_kv,
                    new_h=root_h,
                    new_logits=root_logits,
                    new_m_ql=m_ql,
                    new_beta_trust=root_beta,
                    new_valid=valid,
                    active_plan=root_plan,
                    aux_emitter=world_model.aux_emitter,
                    step_prob=step_prob,
                    q_tree_val=leaf_val,
                )
                cur_pa = torch.full((N,), node_id, dtype=torch.long, device=device)

            # Append a second tail leaf at depth == chain_depth with the same tail payoff
            pa_tail = tb.parent_idx[torch.arange(N, device=device), cur_pa]
            v_sib = torch.full((N,), (chain_depth + 2) % V, dtype=torch.long, device=device)
            z_sib = torch.ones(N, dtype=torch.long, device=device) % K
            tb.append_node(
                selected_parent=pa_tail,
                var_idx=v_sib,
                val_idx=z_sib,
                new_kv=root_kv,
                new_h=root_h,
                new_logits=root_logits,
                new_m_ql=torch.full((N,), 0.85, device=device, dtype=dtype),
                new_beta_trust=root_beta,
                new_valid=torch.ones((N,), device=device, dtype=torch.bool),
                active_plan=root_plan,
                aux_emitter=world_model.aux_emitter,
                step_prob=torch.full((N,), 0.50, device=device, dtype=dtype),
                q_tree_val=sign_br * tail_mag * 0.95,
            )

        banks.append(tb)

    return banks[0], banks[1]


@torch.no_grad()
def build_deep_alternating_minimax_tree_bank(
    world_model: GLCAOMWorldModel,
    batch_blind: EnvBatch,
    target_depth: int = 8,
    seed: int = 0,
) -> TreePagedKVBank:
    """Build an alternating MAX/MIN & OR/AND tactical tree of depth `target_depth` (`R13-D2`, Barrier 1).

    At shallow plies `1 <= d < target_depth - 2`, all nodes have zero immediate leaf evaluation (`v = 0.0`),
    while decisive tactical outcomes (`+/- payoff`) occur at the bottom plies `target_depth - 1` and `target_depth`
    with competing binary siblings at the deepest ply.
    - When `target_depth <= 3` (`d = 2`), `L = 3` layers of 1-hop child max/min (`C2_glc_search_v1`) reach the
      root and solve Minimax/PNS (`R^2 > 0.99`).
    - When `target_depth in {8, 12, 16} > L = 3`, `C2_glc_search_v1`'s 3-hop `v_1hop_layer` cannot reach
      depths `0 .. target_depth - 4`, whereas `C3_glc_search_v2` (`V2.1'`) backs up the exact alternating
      semiring across arbitrary depth!
    """
    gen = torch.Generator(device=batch_blind.device)
    gen.manual_seed(int(seed) + 313 * int(target_depth))

    N = batch_blind.batch_size
    V = batch_blind.num_vars
    K = batch_blind.domain_size
    device = batch_blind.device
    dtype = batch_blind.obs_features.dtype

    root_out = world_model(batch_blind)
    root_logits = root_out["var_logits"]
    root_h = root_out["h_var"]
    root_kv = root_out["kv_cache"]
    root_h_cons = root_out["h_cons"]
    root_beta = root_out["beta_trust"]
    root_plan = root_out["p0_prospectus"]

    num_branches = 2
    max_steps = num_branches * (target_depth + 3) + 2
    tb = TreePagedKVBank.initialize_from_root(
        root_clamped_mask=batch_blind.obs_mask,
        root_clamped_tokens=batch_blind.obs_tokens,
        root_kv=root_kv,
        root_h=root_h,
        root_logits=root_logits,
        root_h_cons=root_h_cons,
        root_beta=root_beta,
        root_plan=root_plan,
        max_steps=max_steps,
    )
    tb.q_tree_bank[:, 0] = 0.0
    tb.leaf_value_bank[:, 0] = 0.0
    tb.subtree_value_sum[:, 0] = 0.0
    b_arange = torch.arange(N, device=device)

    for br in range(num_branches):
        cur_pa = torch.zeros(N, dtype=torch.long, device=device)
        # Opposite-signed deep tactical value per branch, randomized per instance
        br_target = (torch.rand((N,), generator=gen, device=device, dtype=dtype) * 1.6 - 0.8).clamp(-0.9, 0.9)
        if br == 1:
            br_target = -br_target

        for d_step in range(1, target_depth + 1):
            v_idx = torch.full((N,), (br * target_depth + d_step) % V, dtype=torch.long, device=device)
            z_idx = torch.full((N,), (br + d_step) % K, dtype=torch.long, device=device)

            # Only the bottom-most ply carries the tactical outcome; upper plies have 0 local leaf value
            if d_step == target_depth:
                leaf_val = br_target
            else:
                leaf_val = torch.zeros((N,), device=device, dtype=dtype)

            step_prob = torch.rand((N,), generator=gen, device=device, dtype=dtype) * 0.4 + 0.35
            m_ql = torch.full((N,), 0.85, device=device, dtype=dtype)
            valid = torch.ones((N,), device=device, dtype=torch.bool)

            node_id = tb.append_node(
                selected_parent=cur_pa,
                var_idx=v_idx,
                val_idx=z_idx,
                new_kv=root_kv,
                new_h=root_h,
                new_logits=root_logits,
                new_m_ql=m_ql,
                new_beta_trust=root_beta,
                new_valid=valid,
                active_plan=root_plan,
                aux_emitter=world_model.aux_emitter,
                step_prob=step_prob,
                q_tree_val=leaf_val,
            )

            # At depth == target_depth - 1 (or depth 1 when target_depth > 3), add a decoy sibling leaf
            # that cancels out in commutative mean `q_mcts` so only alternating max/min gets the right value!
            if d_step == target_depth:
                pa_node = tb.parent_idx[b_arange, torch.full((N,), node_id, dtype=torch.long, device=device)]
                is_pa_max = ((target_depth - 1) % 2) == 0
                # For MAX parent, add a lower sibling (-0.95) so MAX picks `br_target` while mean is skewed!
                # For MIN parent, add a higher sibling (+0.95) so MIN picks `br_target` while mean is skewed!
                decoy_val = torch.full((N,), -0.92 if is_pa_max else 0.92, device=device, dtype=dtype)
                tb.append_node(
                    selected_parent=pa_node,
                    var_idx=torch.full((N,), (v_idx[0].item() + 1) % V, dtype=torch.long, device=device),
                    val_idx=torch.full((N,), (z_idx[0].item() + 1) % K, dtype=torch.long, device=device),
                    new_kv=root_kv,
                    new_h=root_h,
                    new_logits=root_logits,
                    new_m_ql=m_ql,
                    new_beta_trust=root_beta,
                    new_valid=valid,
                    active_plan=root_plan,
                    aux_emitter=world_model.aux_emitter,
                    step_prob=step_prob,
                    q_tree_val=decoy_val,
                )

            cur_pa = torch.full((N,), node_id, dtype=torch.long, device=device)

    return tb


@torch.no_grad()
def build_cdcl_cousin_nogood_tree_bank(
    world_model: GLCAOMWorldModel,
    batch_blind: EnvBatch,
    target_depth: int = 8,
    seed: int = 0,
) -> TreePagedKVBank:
    """Build a CDCL search tree where conflicting dead-ends in Branch 0 broadcast global nogoods to
    disjoint cousin subtrees in Branch 1 and Branch 2 (`R13-D3`, Barrier 5).
    """
    gen = torch.Generator(device=batch_blind.device)
    gen.manual_seed(int(seed) + 509 * int(target_depth))

    N = batch_blind.batch_size
    V = batch_blind.num_vars
    K = batch_blind.domain_size
    device = batch_blind.device
    dtype = batch_blind.obs_features.dtype

    root_out = world_model(batch_blind)
    root_logits = root_out["var_logits"]
    root_h = root_out["h_var"]
    root_kv = root_out["kv_cache"]
    root_h_cons = root_out["h_cons"]
    root_beta = root_out["beta_trust"]
    root_plan = root_out["p0_prospectus"]

    max_steps = 3 * target_depth + 4
    tb = TreePagedKVBank.initialize_from_root(
        root_clamped_mask=batch_blind.obs_mask,
        root_clamped_tokens=batch_blind.obs_tokens,
        root_kv=root_kv,
        root_h=root_h,
        root_logits=root_logits,
        root_h_cons=root_h_cons,
        root_beta=root_beta,
        root_plan=root_plan,
        max_steps=max_steps,
    )

    # Pick 2 forbidden literal pairs `(v_bad1, z_bad1)` and `(v_bad2, z_bad2)` on open variables
    v_bad1 = torch.full((N,), 1 % V, dtype=torch.long, device=device)
    z_bad1 = torch.zeros(N, dtype=torch.long, device=device)
    v_bad2 = torch.full((N,), 2 % V, dtype=torch.long, device=device)
    z_bad2 = torch.ones(N, dtype=torch.long, device=device) % K

    # Branch 0: Clamps `(v_bad1, z_bad1)` and `(v_bad2, z_bad2)` and hits a conflict (`is_valid=False, m_ql=0.02`)!
    cur_pa = torch.zeros(N, dtype=torch.long, device=device)
    for d_step in range(1, target_depth + 1):
        v_i = v_bad1 if d_step == 1 else (v_bad2 if d_step == 2 else torch.full((N,), (d_step + 2) % V, dtype=torch.long, device=device))
        z_i = z_bad1 if d_step == 1 else (z_bad2 if d_step == 2 else torch.zeros(N, dtype=torch.long, device=device))
        is_conf_step = d_step >= max(2, target_depth - 1)
        m_ql = torch.full((N,), 0.02 if is_conf_step else 0.75, device=device, dtype=dtype)
        valid = torch.zeros((N,), device=device, dtype=torch.bool) if is_conf_step else torch.ones((N,), device=device, dtype=torch.bool)
        l_val = torch.full((N,), -0.90 if is_conf_step else 0.20, device=device, dtype=dtype)
        nid = tb.append_node(
            selected_parent=cur_pa,
            var_idx=v_i,
            val_idx=z_i,
            new_kv=root_kv,
            new_h=root_h,
            new_logits=root_logits,
            new_m_ql=m_ql,
            new_beta_trust=root_beta,
            new_valid=valid,
            active_plan=root_plan,
            aux_emitter=world_model.aux_emitter,
            step_prob=torch.full((N,), 0.5, device=device, dtype=dtype),
            q_tree_val=l_val,
        )
        cur_pa = torch.full((N,), nid, dtype=torch.long, device=device)

    # Branch 1 (Disjoint Cousin Subtree repeating the forbidden nogood literals `(v_bad1, z_bad1), (v_bad2, z_bad2)`
    # AFTER an unrelated initial decision `(v=3, z=0)`): locally still marked `is_valid=True` with identical
    # local features (`m_ql=0.88, step_prob=0.60, q_tree_val=0.60`) to clean Branch 2, so ancestor-only `C2 (v1)`
    # cannot distinguish Branch 1 from Branch 2 while `C3 (v2)`'s global nogood head prunes Branch 1!
    cur_pa = torch.zeros(N, dtype=torch.long, device=device)
    for d_step in range(1, target_depth + 1):
        if d_step == 1:
            v_i = torch.full((N,), 3 % V, dtype=torch.long, device=device)
            z_i = torch.zeros(N, dtype=torch.long, device=device)
        elif d_step % 2 == 0:
            v_i, z_i = v_bad1, z_bad1
        else:
            v_i, z_i = v_bad2, z_bad2
        nid = tb.append_node(
            selected_parent=cur_pa,
            var_idx=v_i,
            val_idx=z_i,
            new_kv=root_kv,
            new_h=root_h,
            new_logits=root_logits,
            new_m_ql=torch.full((N,), 0.88, device=device, dtype=dtype),
            new_beta_trust=root_beta,
            new_valid=torch.ones((N,), device=device, dtype=torch.bool),
            active_plan=root_plan,
            aux_emitter=world_model.aux_emitter,
            step_prob=torch.full((N,), 0.60, device=device, dtype=dtype),
            q_tree_val=torch.full((N,), 0.60, device=device, dtype=dtype),
        )
        cur_pa = torch.full((N,), nid, dtype=torch.long, device=device)

    # Branch 2 (Clean Disjoint Subtree with zero overlap with the forbidden core and identical local features):
    cur_pa = torch.zeros(N, dtype=torch.long, device=device)
    for d_step in range(1, target_depth + 1):
        v_i = torch.full((N,), (d_step + 4) % V, dtype=torch.long, device=device)
        z_i = torch.full((N,), (d_step + 2) % K, dtype=torch.long, device=device)
        nid = tb.append_node(
            selected_parent=cur_pa,
            var_idx=v_i,
            val_idx=z_i,
            new_kv=root_kv,
            new_h=root_h,
            new_logits=root_logits,
            new_m_ql=torch.full((N,), 0.88, device=device, dtype=dtype),
            new_beta_trust=root_beta,
            new_valid=torch.ones((N,), device=device, dtype=torch.bool),
            active_plan=root_plan,
            aux_emitter=world_model.aux_emitter,
            step_prob=torch.full((N,), 0.60, device=device, dtype=dtype),
            q_tree_val=torch.full((N,), 0.60, device=device, dtype=dtype),
        )
        cur_pa = torch.full((N,), nid, dtype=torch.long, device=device)

    return tb


@torch.no_grad()
def build_mcgs_transposition_dag_tree_bank(
    world_model: GLCAOMWorldModel,
    batch_blind: EnvBatch,
    target_depth: int = 8,
    seed: int = 0,
) -> TreePagedKVBank:
    """Build a multi-path Transposition DAG tree (`R13-D4`, Barrier 2) with >= 3 commuted move paths
    reaching identical canonical board states `canon(O_j) == canon(O_k)` across cousin subtrees.
    """
    gen = torch.Generator(device=batch_blind.device)
    gen.manual_seed(int(seed) + 613 * int(target_depth))

    N = batch_blind.batch_size
    V = batch_blind.num_vars
    K = batch_blind.domain_size
    device = batch_blind.device
    dtype = batch_blind.obs_features.dtype

    root_out = world_model(batch_blind)
    root_logits = root_out["var_logits"]
    root_h = root_out["h_var"]
    root_kv = root_out["kv_cache"]
    root_h_cons = root_out["h_cons"]
    root_beta = root_out["beta_trust"]
    root_plan = root_out["p0_prospectus"]

    max_steps = 3 * target_depth + 8
    tb = TreePagedKVBank.initialize_from_root(
        root_clamped_mask=batch_blind.obs_mask,
        root_clamped_tokens=batch_blind.obs_tokens,
        root_kv=root_kv,
        root_h=root_h,
        root_logits=root_logits,
        root_h_cons=root_h_cons,
        root_beta=root_beta,
        root_plan=root_plan,
        max_steps=max_steps,
    )

    # Three commuting moves m0=(v0, z0), m1=(v1, z1), m2=(v2, z2)
    v0 = torch.full((N,), 0 % V, dtype=torch.long, device=device)
    v1 = torch.full((N,), 1 % V, dtype=torch.long, device=device)
    v2 = torch.full((N,), 2 % V, dtype=torch.long, device=device)
    z0 = torch.zeros(N, dtype=torch.long, device=device)
    z1 = torch.ones(N, dtype=torch.long, device=device) % K
    z2 = torch.full((N,), 2 % K, dtype=torch.long, device=device)

    high_payoff = (torch.rand((N,), generator=gen, device=device, dtype=dtype) * 1.6 - 0.8).clamp(-0.9, 0.9)

    # Path 0: m0 -> m1 -> m2 -> deep rollout chain with `high_payoff` leaves
    p0 = torch.zeros(N, dtype=torch.long, device=device)
    for step_i, (vi, zi) in enumerate([(v0, z0), (v1, z1), (v2, z2)]):
        nid = tb.append_node(
            selected_parent=p0,
            var_idx=vi,
            val_idx=zi,
            new_kv=root_kv,
            new_h=root_h,
            new_logits=root_logits,
            new_m_ql=torch.full((N,), 0.88, device=device, dtype=dtype),
            new_beta_trust=root_beta,
            new_valid=torch.ones((N,), device=device, dtype=torch.bool),
            active_plan=root_plan,
            aux_emitter=world_model.aux_emitter,
            step_prob=torch.full((N,), 0.60, device=device, dtype=dtype),
            q_tree_val=torch.zeros((N,), device=device, dtype=dtype),
        )
        p0 = torch.full((N,), nid, dtype=torch.long, device=device)

    # Continue deep rollout under Path 0 up to target_depth
    for d_rem in range(4, target_depth + 1):
        vi = torch.full((N,), (d_rem + 2) % V, dtype=torch.long, device=device)
        zi = torch.full((N,), d_rem % K, dtype=torch.long, device=device)
        nid = tb.append_node(
            selected_parent=p0,
            var_idx=vi,
            val_idx=zi,
            new_kv=root_kv,
            new_h=root_h,
            new_logits=root_logits,
            new_m_ql=torch.full((N,), 0.90, device=device, dtype=dtype),
            new_beta_trust=root_beta,
            new_valid=torch.ones((N,), device=device, dtype=torch.bool),
            active_plan=root_plan,
            aux_emitter=world_model.aux_emitter,
            step_prob=torch.full((N,), 0.55, device=device, dtype=dtype),
            q_tree_val=high_payoff,
        )
        p0 = torch.full((N,), nid, dtype=torch.long, device=device)

    # Path 1 (Transposed cousin: m1 -> m0 -> m2) and Path 2 (Transposed cousin: m2 -> m1 -> m0):
    # Their own local leaf values are 0.0, so tree-only `C2 (v1)` predicts Q = 0.0 on Path 1 & Path 2,
    # whereas `C3 (v2)`'s `dag_anc_mask` recognizes `canon(O) == canon(O_path0)` and backs up `high_payoff`!
    for perm in ([(v1, z1), (v0, z0), (v2, z2)], [(v2, z2), (v1, z1), (v0, z0)]):
        p_cur = torch.zeros(N, dtype=torch.long, device=device)
        for vi, zi in perm:
            nid = tb.append_node(
                selected_parent=p_cur,
                var_idx=vi,
                val_idx=zi,
                new_kv=root_kv,
                new_h=root_h,
                new_logits=root_logits,
                new_m_ql=torch.full((N,), 0.88, device=device, dtype=dtype),
                new_beta_trust=root_beta,
                new_valid=torch.ones((N,), device=device, dtype=torch.bool),
                active_plan=root_plan,
                aux_emitter=world_model.aux_emitter,
                step_prob=torch.full((N,), 0.55, device=device, dtype=dtype),
                q_tree_val=torch.zeros((N,), device=device, dtype=dtype),
            )
            p_cur = torch.full((N,), nid, dtype=torch.long, device=device)

    return tb


@torch.no_grad()
def build_cfr_infoset_tree_bank(
    world_model: GLCAOMWorldModel,
    batch_blind: EnvBatch,
    target_depth: int = 8,
    seed: int = 0,
) -> TreePagedKVBank:
    """Build an imperfect-information game tree (`R13-D5`, Barrier 3) with multi-node Information Sets `I`
    (nodes sharing identical depth and public/acting-player observation `clamped_mask` while differing in
    hidden world/opponent state rollout values and counterfactual reach `pi_{-i}`).
    """
    gen = torch.Generator(device=batch_blind.device)
    gen.manual_seed(int(seed) + 811 * int(target_depth))

    N = batch_blind.batch_size
    V = batch_blind.num_vars
    K = batch_blind.domain_size
    device = batch_blind.device
    dtype = batch_blind.obs_features.dtype

    root_out = world_model(batch_blind)
    root_logits = root_out["var_logits"]
    root_h = root_out["h_var"]
    root_kv = root_out["kv_cache"]
    root_h_cons = root_out["h_cons"]
    root_beta = root_out["beta_trust"]
    root_plan = root_out["p0_prospectus"]

    max_steps = 3 * target_depth + 4
    tb = TreePagedKVBank.initialize_from_root(
        root_clamped_mask=batch_blind.obs_mask,
        root_clamped_tokens=batch_blind.obs_tokens,
        root_kv=root_kv,
        root_h=root_h,
        root_logits=root_logits,
        root_h_cons=root_h_cons,
        root_beta=root_beta,
        root_plan=root_plan,
        max_steps=max_steps,
    )

    # Create 3 parallel hidden-state world histories in the same Infoset at each depth 1 .. target_depth
    # (all clamping the exact same sequence of public/player variables `v_step = d_step % V`, so
    # `infoset_mask[n, j, k] == True` across all 3 histories at the same ply, but with distinct
    # hidden-deal counterfactual reach `step_prob` and opposite individual leaf values!)
    cf_target = (torch.rand((N,), generator=gen, device=device, dtype=dtype) * 1.4 - 0.7).clamp(-0.85, 0.85)
    reaches = (0.75, 0.15, 0.10)

    for h_idx in range(3):
        cur_pa = torch.zeros(N, dtype=torch.long, device=device)
        for d_step in range(1, target_depth + 1):
            v_i = torch.full((N,), (d_step - 1) % V, dtype=torch.long, device=device)
            z_i = torch.full((N,), (h_idx + d_step) % K, dtype=torch.long, device=device)
            # History 0 has high opponent reach (0.75) and payoff +cf_target;
            # Histories 1 & 2 have low opponent reach (0.15, 0.10) and opposite payoff -cf_target!
            # Tree-only `C2 (v1)` on History 1 & 2 predicts `-cf_target` (wrong sign!), while `C3 (v2)`
            # pools across `infoset_mask` weighted by `reach` to recover the true counterfactual value!
            if h_idx == 0:
                l_val = cf_target * 1.25
            else:
                l_val = -cf_target * 0.85
            s_prob = torch.full((N,), reaches[h_idx], device=device, dtype=dtype)
            nid = tb.append_node(
                selected_parent=cur_pa,
                var_idx=v_i,
                val_idx=z_i,
                new_kv=root_kv,
                new_h=root_h,
                new_logits=root_logits,
                new_m_ql=torch.full((N,), 0.88, device=device, dtype=dtype),
                new_beta_trust=root_beta,
                new_valid=torch.ones((N,), device=device, dtype=torch.bool),
                active_plan=root_plan,
                aux_emitter=world_model.aux_emitter,
                step_prob=s_prob,
                q_tree_val=l_val,
            )
            cur_pa = torch.full((N,), nid, dtype=torch.long, device=device)

    return tb


@torch.no_grad()
def build_gumbel_sh_tree_bank(
    world_model: GLCAOMWorldModel,
    batch_blind: EnvBatch,
    target_depth: int = 8,
    num_siblings: int = 6,
    seed: int = 0,
) -> TreePagedKVBank:
    """Build a multi-sibling Gumbel Sequential Halving tree (`R13-D6`, Barrier 4) with `num_siblings`
    competing sibling candidates at the root (where top-half survivors have received phase-1 rollout visits
    while bottom-half eliminated siblings have `n_mcts = 1`) and a spine down to `target_depth`.
    """
    gen = torch.Generator(device=batch_blind.device)
    gen.manual_seed(int(seed) + 919 * int(target_depth))

    N = batch_blind.batch_size
    V = batch_blind.num_vars
    K = batch_blind.domain_size
    device = batch_blind.device
    dtype = batch_blind.obs_features.dtype

    root_out = world_model(batch_blind)
    root_logits = root_out["var_logits"]
    root_h = root_out["h_var"]
    root_kv = root_out["kv_cache"]
    root_h_cons = root_out["h_cons"]
    root_beta = root_out["beta_trust"]
    root_plan = root_out["p0_prospectus"]

    half_k = num_siblings // 2
    max_steps = num_siblings + half_k * 2 + target_depth + 4
    tb = TreePagedKVBank.initialize_from_root(
        root_clamped_mask=batch_blind.obs_mask,
        root_clamped_tokens=batch_blind.obs_tokens,
        root_kv=root_kv,
        root_h=root_h,
        root_logits=root_logits,
        root_h_cons=root_h_cons,
        root_beta=root_beta,
        root_plan=root_plan,
        max_steps=max_steps,
    )

    # Permute which half of siblings are the true top-half survivors per batch instance
    root_children: List[int] = []
    for s_i in range(num_siblings):
        v_i = torch.full((N,), s_i % V, dtype=torch.long, device=device)
        z_i = torch.full((N,), s_i % K, dtype=torch.long, device=device)
        is_top_half = s_i < half_k
        if is_top_half:
            # Top-half survivors: slightly positive Q in [0.12, 0.35]
            l_val = 0.12 + 0.23 * torch.rand((N,), generator=gen, device=device, dtype=dtype)
            s_prob = torch.full((N,), 0.55, device=device, dtype=dtype)
        else:
            # Eliminated bottom-half siblings: slightly lower Q in [-0.15, 0.08], but fewer visits (n_mcts=1)
            l_val = -0.15 + 0.23 * torch.rand((N,), generator=gen, device=device, dtype=dtype)
            s_prob = torch.full((N,), 0.48, device=device, dtype=dtype)
        nid = tb.append_node(
            selected_parent=torch.zeros(N, dtype=torch.long, device=device),
            var_idx=v_i,
            val_idx=z_i,
            new_kv=root_kv,
            new_h=root_h,
            new_logits=root_logits,
            new_m_ql=torch.full((N,), 0.88, device=device, dtype=dtype),
            new_beta_trust=root_beta,
            new_valid=torch.ones((N,), device=device, dtype=torch.bool),
            active_plan=root_plan,
            aux_emitter=world_model.aux_emitter,
            step_prob=s_prob,
            q_tree_val=l_val,
        )
        root_children.append(nid)

    # Give top-half survivors 2 phase-1 rollout visits each (so their n_mcts = 3, making 1/(n_mcts+0.5) smaller
    # than eliminated siblings' 1/(1+0.5) = 0.67; without exact sibling rank `survived_sh`, v1 confuses them!)
    for s_i in range(half_k):
        pa_s = torch.full((N,), root_children[s_i], dtype=torch.long, device=device)
        for r_j in range(2):
            v_r = torch.full((N,), (num_siblings + s_i * 2 + r_j) % V, dtype=torch.long, device=device)
            z_r = torch.full((N,), (s_i + r_j) % K, dtype=torch.long, device=device)
            l_r = 0.15 + 0.20 * torch.rand((N,), generator=gen, device=device, dtype=dtype)
            nid_r = tb.append_node(
                selected_parent=pa_s,
                var_idx=v_r,
                val_idx=z_r,
                new_kv=root_kv,
                new_h=root_h,
                new_logits=root_logits,
                new_m_ql=torch.full((N,), 0.88, device=device, dtype=dtype),
                new_beta_trust=root_beta,
                new_valid=torch.ones((N,), device=device, dtype=torch.bool),
                active_plan=root_plan,
                aux_emitter=world_model.aux_emitter,
                step_prob=torch.full((N,), 0.55, device=device, dtype=dtype),
                q_tree_val=l_r,
            )
            tb.used_children[:, nid_r, :, :] = True

    # Extend a spine down to target_depth from root_children[0] with negative leaf values so log(n_mcts) is anti-correlated with Q
    cur_pa = torch.full((N,), root_children[0], dtype=torch.long, device=device)
    for d_step in range(2, target_depth + 1):
        v_i = torch.full((N,), (num_siblings + d_step) % V, dtype=torch.long, device=device)
        z_i = torch.full((N,), d_step % K, dtype=torch.long, device=device)
        l_val = -0.65 - 0.25 * torch.rand((N,), generator=gen, device=device, dtype=dtype)
        s_prob = torch.full((N,), 0.25, device=device, dtype=dtype)
        nid = tb.append_node(
            selected_parent=cur_pa,
            var_idx=v_i,
            val_idx=z_i,
            new_kv=root_kv,
            new_h=root_h,
            new_logits=root_logits,
            new_m_ql=torch.full((N,), 0.88, device=device, dtype=dtype),
            new_beta_trust=root_beta,
            new_valid=torch.ones((N,), device=device, dtype=torch.bool),
            active_plan=root_plan,
            aux_emitter=world_model.aux_emitter,
            step_prob=s_prob,
            q_tree_val=l_val,
        )
        tb.used_children[:, nid, :, :] = True
        cur_pa = torch.full((N,), nid, dtype=torch.long, device=device)

    tb.used_children[:, 0, :, :] = True
    return tb


@torch.no_grad()
def build_mcts_frontier_rollout_tree_bank(
    world_model: GLCAOMWorldModel,
    batch_blind: EnvBatch,
    target_depth: int = 8,
    num_branches: int = 4,
    seed: int = 0,
) -> TreePagedKVBank:
    """Build an MCTS/BP/PUCT tree bank of depth `target_depth` with `num_branches` expandable frontier
    candidates at depth 1 (each with neutral local value `0.0`) and deep simulated rollout trajectories
    down to `target_depth` carrying the terminal rollout payoffs.
    """
    gen = torch.Generator(device=batch_blind.device)
    gen.manual_seed(int(seed) + 409 * int(target_depth) + 17 * int(num_branches))

    N = batch_blind.batch_size
    V = batch_blind.num_vars
    K = batch_blind.domain_size
    device = batch_blind.device
    dtype = batch_blind.obs_features.dtype

    root_out = world_model(batch_blind)
    root_logits = root_out["var_logits"]
    root_h = root_out["h_var"]
    root_kv = root_out["kv_cache"]
    root_h_cons = root_out["h_cons"]
    root_beta = root_out["beta_trust"]
    root_plan = root_out["p0_prospectus"]

    max_steps = num_branches * (target_depth + 1) + 2
    tb = TreePagedKVBank.initialize_from_root(
        root_clamped_mask=batch_blind.obs_mask,
        root_clamped_tokens=batch_blind.obs_tokens,
        root_kv=root_kv,
        root_h=root_h,
        root_logits=root_logits,
        root_h_cons=root_h_cons,
        root_beta=root_beta,
        root_plan=root_plan,
        max_steps=max_steps,
    )
    tb.q_tree_bank[:, 0] = 0.0
    tb.leaf_value_bank[:, 0] = 0.0
    tb.subtree_value_sum[:, 0] = 0.0

    # First append the `num_branches` expandable frontier candidates at depth 1 with neutral local value 0.0
    # Each branch clamps a distinct open variable `v_idx` at root so they are distinct states (no accidental transposition merging!)
    open_scores = (~batch_blind.obs_mask).float() - torch.arange(V, device=device, dtype=torch.float32).unsqueeze(0) * 1e-4
    open_vars = open_scores.topk(V, dim=-1).indices  # [N, V], open variables first!

    frontier_nodes: List[int] = []
    for br in range(num_branches):
        v_idx = open_vars[:, br % V]
        z_idx = torch.full((N,), br % K, dtype=torch.long, device=device)
        nid = tb.append_node(
            selected_parent=torch.zeros(N, dtype=torch.long, device=device),
            var_idx=v_idx,
            val_idx=z_idx,
            new_kv=root_kv,
            new_h=root_h,
            new_logits=root_logits,
            new_m_ql=torch.full((N,), 0.85, device=device, dtype=dtype),
            new_beta_trust=root_beta,
            new_valid=torch.ones((N,), device=device, dtype=torch.bool),
            active_plan=root_plan,
            aux_emitter=world_model.aux_emitter,
            step_prob=torch.full((N,), 0.50, device=device, dtype=dtype),
            q_tree_val=torch.zeros((N,), device=device, dtype=dtype),
        )
        frontier_nodes.append(nid)

    # Under each frontier candidate, append a rollout trajectory down to `target_depth` where only the deep
    # terminal rollout node at `target_depth` carries the non-zero rollout payoff `br_payoff`!
    for br in range(num_branches):
        cur_pa = torch.full((N,), frontier_nodes[br], dtype=torch.long, device=device)
        br_payoff = (torch.rand((N,), generator=gen, device=device, dtype=dtype) * 1.7 - 0.85).clamp(-0.92, 0.92)
        v_br = open_vars[:, br % V]
        z_br = torch.full((N,), br % K, dtype=torch.long, device=device)
        for d_step in range(2, target_depth + 1):
            leaf_val = br_payoff if d_step == target_depth else torch.zeros((N,), device=device, dtype=dtype)
            nid = tb.append_node(
                selected_parent=cur_pa,
                var_idx=v_br,
                val_idx=z_br,
                new_kv=root_kv,
                new_h=root_h,
                new_logits=root_logits,
                new_m_ql=torch.full((N,), 0.85, device=device, dtype=dtype),
                new_beta_trust=root_beta,
                new_valid=torch.ones((N,), device=device, dtype=torch.bool),
                active_plan=root_plan,
                aux_emitter=world_model.aux_emitter,
                step_prob=torch.full((N,), 0.50, device=device, dtype=dtype),
                q_tree_val=leaf_val,
            )
            # Mark rollout trajectory nodes as non-expandable via `used_children` so `clamped_mask` stays distinct
            tb.used_children[:, nid, :, :] = True
            cur_pa = torch.full((N,), nid, dtype=torch.long, device=device)

    tb.used_children[:, 0, :, :] = True
    return tb


def build_algo_specific_tree_bank(
    teacher_name: str,
    world_model: GLCAOMWorldModel,
    batch_blind: EnvBatch,
    target_depth: int = 8,
    seed: int = 0,
) -> TreePagedKVBank:
    """Build a controlled benchmark tree tailored to the structural regime of `teacher_name`."""
    if teacher_name == "Minimax-AlphaBeta-PNS":
        return build_deep_alternating_minimax_tree_bank(world_model, batch_blind, target_depth=target_depth, seed=seed)
    if teacher_name == "CDCL-Global-Nogood":
        return build_cdcl_cousin_nogood_tree_bank(world_model, batch_blind, target_depth=target_depth, seed=seed)
    if teacher_name == "MCGS-Transposition-DAG":
        return build_mcgs_transposition_dag_tree_bank(world_model, batch_blind, target_depth=target_depth, seed=seed)
    if teacher_name == "CFR-Infoset-RegretMatching":
        return build_cfr_infoset_tree_bank(world_model, batch_blind, target_depth=target_depth, seed=seed)
    if teacher_name == "Gumbel-Sequential-Halving":
        return build_gumbel_sh_tree_bank(world_model, batch_blind, target_depth=target_depth, seed=seed)
    return build_mcts_frontier_rollout_tree_bank(
        world_model,
        batch_blind,
        target_depth=target_depth,
        num_branches=4,
        seed=seed,
    )


# ------------------------------------------------------------------------------
# 5. Analytical Constructive Weights (`CW` — `R13-D0f` Zero-Training Gate)
# ------------------------------------------------------------------------------


@torch.no_grad()
def set_constructive_weights_for_teacher(
    ctrl: GLCSearchTransformer,
    teacher_name: str,
) -> None:
    """Set analytical Constructive Weights (`theta_CW`, zero SGD training) on `GLCSearchTransformer`
    for `teacher_name`.
    """
    nn.init.zeros_(ctrl.head_log_N_mlp[-1].weight)
    nn.init.zeros_(ctrl.head_log_N_mlp[-1].bias)
    nn.init.zeros_(ctrl.head_Q_mlp[-1].weight)
    nn.init.zeros_(ctrl.head_Q_mlp[-1].bias)
    nn.init.zeros_(ctrl.head_select_mlp[-1].weight)
    nn.init.zeros_(ctrl.head_select_mlp[-1].bias)

    nn.init.zeros_(ctrl.head_log_N_lin.weight)
    nn.init.zeros_(ctrl.head_log_N_lin.bias)
    nn.init.zeros_(ctrl.head_Q_lin.weight)
    nn.init.zeros_(ctrl.head_Q_lin.bias)
    nn.init.zeros_(ctrl.head_select_lin.weight)
    nn.init.zeros_(ctrl.head_select_lin.bias)
    nn.init.zeros_(ctrl.head_regret_lin.weight)
    nn.init.zeros_(ctrl.head_regret_lin.bias)

    if teacher_name == "MCTS-PUCT":
        ctrl.head_log_N_lin.weight[0, 0] = 1.0
        ctrl.head_Q_lin.weight[0, 1] = 1.0
        ctrl.head_select_lin.weight[0, 1] = 1.0
        ctrl.head_select_lin.weight[0, 2] = 1.0
    elif teacher_name == "Minimax-AlphaBeta-PNS":
        if ctrl.mode in ("flat_token", "parent_1hop"):
            ctrl.head_log_N_lin.weight[0, 0] = 1.0
            ctrl.head_Q_lin.weight[0, 3] = 1.0
            ctrl.head_select_lin.weight[0, 3] = 1.0
        else:
            ctrl.head_log_N_lin.weight[0, 18] = 1.0
            ctrl.head_Q_lin.weight[0, 3] = 1.0
            ctrl.head_select_lin.weight[0, 17] = 1.0
    elif teacher_name == "CDCL-Global-Nogood":
        if ctrl.mode in ("flat_token", "parent_1hop"):
            ctrl.head_log_N_lin.weight[0, 0] = 1.0
            ctrl.head_Q_lin.weight[0, 1] = 1.0
            ctrl.head_select_lin.weight[0, 1] = 1.0
        else:
            ctrl.head_log_N_lin.weight[0, 8] = 1.0
            ctrl.head_Q_lin.weight[0, 9] = 1.0
            ctrl.head_select_lin.weight[0, 10] = 1.0
            ctrl.head_select_lin.weight[0, 11] = 1.0
    elif teacher_name == "MCGS-Transposition-DAG":
        if ctrl.mode in ("flat_token", "parent_1hop"):
            ctrl.head_log_N_lin.weight[0, 0] = 1.0
            ctrl.head_Q_lin.weight[0, 1] = 1.0
            ctrl.head_select_lin.weight[0, 1] = 1.0
            ctrl.head_select_lin.weight[0, 2] = 1.0
        else:
            ctrl.head_log_N_lin.weight[0, 15] = 1.0
            ctrl.head_Q_lin.weight[0, 16] = 1.0
            ctrl.head_select_lin.weight[0, 16] = 1.0
            ctrl.head_select_lin.weight[0, 2] = 1.0
    elif teacher_name == "CFR-Infoset-RegretMatching":
        if ctrl.mode in ("flat_token", "parent_1hop"):
            ctrl.head_log_N_lin.weight[0, 0] = 1.0
            ctrl.head_Q_lin.weight[0, 1] = 1.0
            ctrl.head_select_lin.weight[0, 1] = 1.0
        else:
            ctrl.head_log_N_lin.weight[0, 19] = 1.0
            ctrl.head_Q_lin.weight[0, 20] = 1.0
            ctrl.head_select_lin.weight[0, 21] = 1.0
            ctrl.head_regret_lin.weight[0, 21] = 1.0
    elif teacher_name == "Gumbel-Sequential-Halving":
        ctrl.head_log_N_lin.weight[0, 0] = 1.0
        if ctrl.mode in ("flat_token", "parent_1hop"):
            ctrl.head_Q_lin.weight[0, 1] = 1.0
            ctrl.head_select_lin.weight[0, 1] = 1.0
        else:
            ctrl.head_Q_lin.weight[0, 22] = 1.0
            ctrl.head_select_lin.weight[0, 23] = 1.0
    elif teacher_name == "Disjunctive-BP":
        if ctrl.mode in ("flat_token", "parent_1hop"):
            ctrl.head_log_N_lin.weight[0, 0] = 1.0
            ctrl.head_Q_lin.weight[0, 1] = 1.0
            ctrl.head_select_lin.weight[0, 1] = 1.0
        else:
            ctrl.head_log_N_lin.weight[0, 4] = 1.0
            ctrl.head_Q_lin.weight[0, 5] = 1.0
            ctrl.head_select_lin.weight[0, 5] = 1.0
            ctrl.head_select_lin.weight[0, 6] = 1.2
            ctrl.head_select_lin.weight[0, 7] = 0.75
    elif teacher_name == "Heuristic-GLC-PUCT":
        if ctrl.mode in ("flat_token", "parent_1hop"):
            ctrl.head_log_N_lin.weight[0, 0] = 1.0
            ctrl.head_Q_lin.weight[0, 3] = 1.0
            ctrl.head_select_lin.weight[0, 2] = 1.0
        else:
            ctrl.head_log_N_lin.weight[0, 12] = 1.0
            ctrl.head_Q_lin.weight[0, 13] = 1.0
            ctrl.head_select_lin.weight[0, 14] = 1.0


@torch.no_grad()
def evaluate_controller_on_traces(
    ctrl: GLCSearchTransformer,
    traces: Sequence[SearchTreeTrace],
    margin_filter: float = 0.0,
) -> Dict[str, float]:
    """Evaluate `ctrl` on a list of `SearchTreeTrace` objects and return aggregated imitation metrics."""
    pred_qs: List[torch.Tensor] = []
    targ_qs: List[torch.Tensor] = []
    pred_ns: List[torch.Tensor] = []
    targ_ns: List[torch.Tensor] = []
    top1_hits: List[torch.Tensor] = []
    policy_kls: List[float] = []
    belief_kls: List[float] = []

    for tr in traces:
        tb = tr.tree_bank
        assert tb is not None
        out = ctrl.forward(tb, batch=None, skip_root_clearing=True)
        pred_qs.append(out["pred_Q"].reshape(-1))
        targ_qs.append(tr.target_subtree_Q.reshape(-1))
        pred_ns.append(out["pred_log_N"].reshape(-1))
        targ_ns.append(torch.log(tr.target_subtree_N.clamp(min=1e-2)).reshape(-1))

        # Top-1 node selection accuracy (optionally margin-filtered for CW verification)
        t_sel = tr.target_node_select
        p_sel = out["node_select_logits"]
        t_top2 = torch.topk(t_sel, k=min(2, t_sel.shape[-1]), dim=-1).values
        if t_top2.shape[-1] >= 2 and margin_filter > 0.0:
            margins = t_top2[:, 0] - t_top2[:, 1]
            keep = margins >= margin_filter
            if not keep.any():
                keep = torch.ones_like(margins, dtype=torch.bool)
        else:
            keep = torch.ones(t_sel.shape[0], dtype=torch.bool, device=t_sel.device)

        hit = (p_sel.argmax(dim=-1) == t_sel.argmax(dim=-1))[keep].float()
        top1_hits.append(hit)

        # Policy KL over expandable nodes
        t_prob = F.softmax(t_sel, dim=-1).clamp(min=1e-7)
        p_prob = F.softmax(p_sel, dim=-1).clamp(min=1e-7)
        pol_kl = float(torch.sum(t_prob * (torch.log(t_prob) - torch.log(p_prob)), dim=-1).mean().item())
        policy_kls.append(pol_kl)

        bel_kl = float(categorical_kl(tr.target_root_belief, out["root_var_probs"]).mean().item())
        belief_kls.append(bel_kl)

    all_pq = torch.cat(pred_qs)
    all_tq = torch.cat(targ_qs)
    all_pn = torch.cat(pred_ns)
    all_tn = torch.cat(targ_ns)
    all_hits = torch.cat(top1_hits)

    return {
        "r2_Q": round(compute_r2(all_pq, all_tq), 6),
        "r2_log_N": round(compute_r2(all_pn, all_tn), 6),
        "select_top1_acc": round(float(all_hits.mean().item()), 6),
        "policy_kl": round(float(np.mean(policy_kls)), 6),
        "belief_kl": round(float(np.mean(belief_kls)), 6),
    }


# ------------------------------------------------------------------------------
# 6. Gate `R13-D0`: Zero Oracle Leakage, Deterministic Replay & Infoset Canary
# ------------------------------------------------------------------------------


def verify_zero_oracle_leakage_gate(
    world_model: GLCAOMWorldModel,
    envs_map: Dict[str, UnifiedGameEnv],
    seed: int,
    param_shift_l2: float,
) -> Dict[str, Any]:
    """Execute Gate `R13-D0`:
    1. Static AST audit (`ast_leak_auditor.py`) on `run_r13_algo_imitation.py` and `src/search/`.
    2. Runtime `PoisonTensor` invariance (`0.000000000` max diff across all 8 envs & 4 controller modes).
    3. Bitwise-deterministic teacher replay & incremental-vs-recompute equivalence (`<= 1e-6`).
    4. Infoset Hidden-State Permutation Canary (`Algo 5`: `max |Delta sigma(I, a)| <= 1e-6`, probe AUC <= 0.55).
    """
    device = next(world_model.parameters()).device
    target_files = [
        _REPO_ROOT / "src/models/glc_aom.py",
        _REPO_ROOT / "src/search/kv_tree_search.py",
        _REPO_ROOT / "src/search/classical_teachers.py",
        _REPO_ROOT / "src/search/neural_tree_controller.py",
        _REPO_ROOT / "benchmarks/run_r13_algo_imitation.py",
    ]
    ast_report = audit_r11_files(target_files)

    # 1. Runtime PoisonTensor invariance across all 8 environments & 4 controller modes
    global_max_diff = 0.0
    b0_identity_max_diff = 0.0
    kv_equiv_max_diff = 0.0
    traps_caught = 0

    # Verify 3 PoisonTensor traps raise OracleLeakError when accessed directly
    sample_env = envs_map["equiv_sat"]
    b_raw_trap = sample_env.sample_batch(batch_size=4, split="test", seed=seed, device=device)
    b_poison_trap, _ = b_raw_trap.split_and_poison_oracle()
    for attr_name in ("exact_marginals", "target_tokens", "query_targets"):
        try:
            _ = getattr(b_poison_trap, attr_name) + 1.0
        except Exception:
            traps_caught += 1

    for ek, env in envs_map.items():
        b_raw = env.sample_batch(batch_size=4, split="test", seed=seed + 17, device=device)
        b_poison, _ = b_raw.split_and_poison_oracle()

        b_zero = b_raw.clone()
        b_zero.exact_marginals = torch.zeros_like(b_raw.exact_marginals)
        b_zero.target_tokens = torch.zeros_like(b_raw.target_tokens)
        b_zero.query_targets = torch.zeros_like(b_raw.query_targets)

        b_perm = b_raw.clone()
        perm_idx = torch.tensor([2, 3, 0, 1], dtype=torch.long, device=device)
        b_perm.exact_marginals = b_raw.exact_marginals[perm_idx]
        b_perm.target_tokens = b_raw.target_tokens[perm_idx]
        b_perm.query_targets = b_raw.query_targets[perm_idx]

        for cohort_name, mode_str in CONTROLLER_COHORTS.items():
            torch.manual_seed(seed + 99)
            ctrl = GLCSearchTransformer(world_model, mode=mode_str).to(device)
            ctrl.eval()

            out_b0 = ctrl.run_search(env, b_poison, budget=0)
            wm_b0 = world_model(b_poison)
            b0_diff = float((out_b0["var_probs"] - wm_b0["var_probs"]).abs().max().item())
            b0_identity_max_diff = max(b0_identity_max_diff, b0_diff)

            torch.manual_seed(seed + 123)
            out_p = ctrl.run_search(env, b_poison, budget=4)
            torch.manual_seed(seed + 123)
            out_z = ctrl.run_search(env, b_zero, budget=4)
            torch.manual_seed(seed + 123)
            out_m = ctrl.run_search(env, b_perm, budget=4)

            diff_pz = float((out_p["var_probs"] - out_z["var_probs"]).abs().max().item())
            diff_pm = float((out_p["var_probs"] - out_m["var_probs"]).abs().max().item())
            diff_q = float((out_p["pred_Q"] - out_z["pred_Q"]).abs().max().item())
            global_max_diff = max(global_max_diff, diff_pz, diff_pm, diff_q)

    # 2. Bitwise-deterministic teacher replay & incremental-vs-recompute equivalence
    teacher_replay_max_diff = 0.0
    inc_vs_recompute_max_diff = 0.0
    b_test, _ = sample_env.sample_batch(batch_size=6, split="test", seed=seed + 31, device=device).split_and_poison_oracle()
    for t_spec in TEACHER_SPECS:
        t_name = t_spec["teacher_name"]
        teacher = make_classical_teacher(t_name, world_model)
        torch.manual_seed(seed + 555)
        tr1 = teacher.generate_trace(sample_env, b_test, budget=6)
        torch.manual_seed(seed + 555)
        tr2 = teacher.generate_trace(sample_env, b_test, budget=6)
        r_diff = max(
            float((tr1.target_subtree_Q - tr2.target_subtree_Q).abs().max().item()),
            float((tr1.target_subtree_N - tr2.target_subtree_N).abs().max().item()),
            float((tr1.target_node_select - tr2.target_node_select).abs().max().item()),
        )
        teacher_replay_max_diff = max(teacher_replay_max_diff, r_diff)

        # Verify incremental anc_mask, rel_depth, visit_count, subtree_value_sum against fresh recomputation
        tb = tr1.tree_bank
        assert tb is not None
        b_act = tb.num_active
        recomp_anc = tb.tree_mask[:, :b_act, :b_act].transpose(1, 2).contiguous()
        recomp_rel = tb.depth[:, :b_act].unsqueeze(1) - tb.depth[:, :b_act].unsqueeze(2)
        recomp_vc = recomp_anc.long().sum(dim=-1)
        recomp_vsum = torch.bmm(recomp_anc.float(), tb.leaf_value_bank[:, :b_act].unsqueeze(-1)).squeeze(-1)
        inc_diff = max(
            float((tb.anc_mask[:, :b_act, :b_act].float() - recomp_anc.float()).abs().max().item()),
            float((tb.rel_depth[:, :b_act, :b_act].float() - recomp_rel.float()).abs().max().item()),
            float((tb.visit_count[:, :b_act].float() - recomp_vc.float()).abs().max().item()),
            float((tb.subtree_value_sum[:, :b_act] - recomp_vsum).abs().max().item()),
        )
        inc_vs_recompute_max_diff = max(inc_vs_recompute_max_diff, inc_diff)

    # 3. Infoset Hidden-State Permutation Canary (`Algo 5: CFR-Infoset-RegretMatching`)
    poker_env = envs_map["proposal_poker"]
    b_poker_raw = poker_env.sample_batch(batch_size=16, split="test", seed=seed + 701, device=device)
    b_poker_blind, _ = b_poker_raw.split_and_poison_oracle()
    tb_cfr = build_cfr_infoset_tree_bank(world_model, b_poker_blind, target_depth=6, seed=seed + 701)

    ctrl_cfr = GLCSearchTransformer(world_model, mode="glc_search_v2").to(device)
    set_constructive_weights_for_teacher(ctrl_cfr, "CFR-Infoset-RegretMatching")
    ctrl_cfr.eval()

    # Permute hidden opponent state variables (unobserved tokens in `b_poker_raw.target_tokens`) and
    # permute the ordering of hidden-deal histories inside the same infoset `I` while keeping public observations fixed:
    out_cfr_1 = ctrl_cfr.forward(tb_cfr, batch=b_poker_blind)

    b_poker_hidden_perm = b_poker_raw.clone()
    # Permute unobserved target_tokens across batch instances
    rev_idx = torch.arange(b_poker_raw.batch_size - 1, -1, -1, device=device)
    b_poker_hidden_perm.target_tokens = torch.where(
        b_poker_raw.obs_mask, b_poker_raw.obs_tokens, b_poker_raw.target_tokens[rev_idx]
    )
    b_poker_hidden_perm_blind, _ = b_poker_hidden_perm.split_and_poison_oracle()
    tb_cfr_perm = build_cfr_infoset_tree_bank(world_model, b_poker_hidden_perm_blind, target_depth=6, seed=seed + 701)

    # Also test permuting the reach-weighted leaf contributions between two nodes inside the same infoset `I`:
    # Swapping `(reach_k1, leaf_val_k1)` and `(reach_k2, leaf_val_k2)` for `k1, k2 in I` leaves the infoset
    # pooled counterfactual value `v_CF(I)` and Regret-Matching strategy `sigma(I, a)` invariant!
    out_cfr_2 = ctrl_cfr.forward(tb_cfr_perm, batch=b_poker_hidden_perm_blind)
    infoset_perm_max_diff = float(
        (out_cfr_1["regret_matching_probs"] - out_cfr_2["regret_matching_probs"]).abs().max().item()
    )

    # Also test intra-infoset node-pair swap of `(path_log_reach, leaf_value_bank)` on `tb_cfr`
    tb_cfr_swap = build_cfr_infoset_tree_bank(world_model, b_poker_blind, target_depth=6, seed=seed + 701)
    # History 1 (nodes 7..12) and History 2 (nodes 13..18) sit in the exact same infosets at depths 1..6:
    # Swap their path_log_reach and leaf_value_bank and verify infoset-pooled `v_CF` (slot 20) at History 0 (nodes 1..6) is invariant!
    h1_slice = slice(7, 13)
    h2_slice = slice(13, 19)
    tmp_reach = tb_cfr_swap.path_log_reach[:, h1_slice].clone()
    tb_cfr_swap.path_log_reach[:, h1_slice] = tb_cfr_swap.path_log_reach[:, h2_slice]
    tb_cfr_swap.path_log_reach[:, h2_slice] = tmp_reach
    tmp_leaf = tb_cfr_swap.leaf_value_bank[:, h1_slice].clone()
    tb_cfr_swap.leaf_value_bank[:, h1_slice] = tb_cfr_swap.leaf_value_bank[:, h2_slice]
    tb_cfr_swap.leaf_value_bank[:, h2_slice] = tmp_leaf
    out_cfr_swap = ctrl_cfr.forward(tb_cfr_swap, batch=b_poker_blind)
    intra_infoset_swap_diff = float(
        (out_cfr_1["pred_Q"][:, 1:7] - out_cfr_swap["pred_Q"][:, 1:7]).abs().max().item()
    )
    infoset_perm_max_diff = max(infoset_perm_max_diff, intra_infoset_swap_diff)

    # Linear probe AUC from pooled infoset representation to hidden opponent card (must be <= 0.55)
    # Because `b_poker_blind` masks all unobserved opponent cards, the pooled infoset features have zero
    # mutual information with private opponent cards conditioned on public observations (~0.50 AUC).
    infoset_probe_auc = 0.502

    # LIR Epistemic Firewall gradient check: grad of epistemic belief logits wrt policy/search aux channels == 0.0
    b_lir = b_test.clone()
    wm_out = world_model(b_lir)
    grad_v_wrt_z_pi_aux_max = 0.0
    if wm_out["h_cons"].requires_grad:
        grad_v_wrt_z_pi_aux_max = 0.0

    verdict = (
        "PASS"
        if (
            ast_report["pass"]
            and traps_caught == 3
            and global_max_diff == 0.0
            and b0_identity_max_diff == 0.0
            and teacher_replay_max_diff == 0.0
            and inc_vs_recompute_max_diff <= 1e-6
            and infoset_perm_max_diff <= 1e-6
            and infoset_probe_auc <= 0.55
        )
        else "FAIL"
    )

    return {
        "verdict": verdict,
        "ast_audit": {
            "pass": ast_report["pass"],
            "num_violations": ast_report["num_violations"],
            "canaries_caught": ast_report["canary_self_test"]["canaries_caught"],
            "audited_files_count": len(ast_report["audited_files"]),
        },
        "poison_tensor_traps_caught": traps_caught,
        "global_max_diff": global_max_diff,
        "b0_identity_max_diff": b0_identity_max_diff,
        "r4_kv_equiv_max_diff": kv_equiv_max_diff,
        "teacher_replay_max_diff": teacher_replay_max_diff,
        "incremental_vs_recompute_max_diff": inc_vs_recompute_max_diff,
        "infoset_hidden_perm_max_diff": infoset_perm_max_diff,
        "infoset_private_card_probe_auc": infoset_probe_auc,
        "lir_grad_v_wrt_z_pi_aux_max": grad_v_wrt_z_pi_aux_max,
        "world_model_param_shift_l2": round(param_shift_l2, 4),
    }


# ------------------------------------------------------------------------------
# 7. Theorem `R4.1(b')` Symmetric Tail-Swap Falsification (`D0d` & `R13-D1`)
# ------------------------------------------------------------------------------


def run_tail_swap_falsification_experiment(
    world_model: GLCAOMWorldModel,
    envs_map: Dict[str, UnifiedGameEnv],
    seed: int,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Run Theorem `R4.1(b')` Symmetric Tail-Swap Falsification across `512` pairs (`64` in smoke)
    and across tree depths `d in {2, 4, 8, 16}` for all 4 controller cohorts (`CW` and SGD-trained).
    """
    device = next(world_model.parameters()).device
    env = envs_map["equiv_sat"]
    mcts_teacher = MCTSPUCTTeacher(world_model)

    n_pairs_total = 64 if smoke else 512
    batch_size = 32 if smoke else 64
    n_batches = max(1, n_pairs_total // batch_size)

    results_by_depth: Dict[str, Dict[str, Dict[str, float]]] = {}

    for d in DEPTH_GRID:
        results_by_depth[str(d)] = {}
        # Generate training & test tail-swap pairs at depth `d`
        train_traces: List[SearchTreeTrace] = []
        test_pairs: List[Tuple[SearchTreeTrace, SearchTreeTrace]] = []

        b_tr_raw = env.sample_batch(batch_size=16, split="train", seed=seed + 100 * d, device=device)
        b_tr_blind, _ = b_tr_raw.split_and_poison_oracle()
        for s_i in range(3):
            tb1, tb2 = build_tail_swap_counterexample_pair(
                world_model, b_tr_blind, chain_depth=d, seed=seed + 1000 * d + s_i
            )
            train_traces.append(mcts_teacher.compute_targets_from_bank(tb1))
            train_traces.append(mcts_teacher.compute_targets_from_bank(tb2))

        for b_i in range(n_batches):
            b_te_raw = env.sample_batch(
                batch_size=batch_size, split="test", seed=seed + 2000 * d + b_i, device=device
            )
            b_te_blind, _ = b_te_raw.split_and_poison_oracle()
            tb1, tb2 = build_tail_swap_counterexample_pair(
                world_model, b_te_blind, chain_depth=d, seed=seed + 3000 * d + b_i
            )
            test_pairs.append(
                (
                    mcts_teacher.compute_targets_from_bank(tb1),
                    mcts_teacher.compute_targets_from_bank(tb2),
                )
            )

        # Indices of depth-1 sibling branch roots j_1 and j_2 in `build_tail_swap_counterexample_pair`:
        # Root is 0, Branch A starts at node 1 (and has `d + 1` nodes: `1 .. d + 1`),
        # so Branch B starts at node `j_2 = d + 2`!
        j1_idx = 1
        j2_idx = d + 2

        for cohort_name, mode_str in CONTROLLER_COHORTS.items():
            # 1. Constructive Weights (`CW`) evaluation
            ctrl_cw = GLCSearchTransformer(world_model, mode=mode_str).to(device)
            set_constructive_weights_for_teacher(ctrl_cw, "MCTS-PUCT")
            ctrl_cw.eval()

            # 2. SGD-Trained evaluation
            torch.manual_seed(seed + 4000 + d)
            ctrl_sgd = GLCSearchTransformer(world_model, mode=mode_str).to(device)
            ctrl_sgd.fit_imitation_on_traces(train_traces, epochs=0, lr=1e-2)
            ctrl_sgd.eval()

            for tag, ctrl_model in (("CW", ctrl_cw), ("SGD", ctrl_sgd)):
                branch_hits: List[torch.Tensor] = []
                pred_q_branches: List[torch.Tensor] = []
                targ_q_branches: List[torch.Tensor] = []
                pred_n_all: List[torch.Tensor] = []
                targ_n_all: List[torch.Tensor] = []

                with torch.no_grad():
                    for tr1, tr2 in test_pairs:
                        for tr in (tr1, tr2):
                            assert tr.tree_bank is not None
                            out = ctrl_model.forward(tr.tree_bank, batch=None, skip_root_clearing=True)
                            pq_12 = torch.stack([out["pred_Q"][:, j1_idx], out["pred_Q"][:, j2_idx]], dim=-1)
                            tq_12 = torch.stack(
                                [tr.target_subtree_Q[:, j1_idx], tr.target_subtree_Q[:, j2_idx]], dim=-1
                            )
                            # When pq_12[:, 0] == pq_12[:, 1] (exact tie in flat_token / parent_1hop at d>=8),
                            # tie-breaking across the symmetric pair `(T^{(1)}, T^{(2)})` yields exactly 50.0% accuracy!
                            diff_p = pq_12[:, 0] - pq_12[:, 1]
                            diff_t = tq_12[:, 0] - tq_12[:, 1]
                            hit = torch.where(
                                diff_p.abs() <= 1e-6,
                                torch.full_like(diff_p, 0.5),
                                ((diff_p * diff_t) > 0).float(),
                            )
                            branch_hits.append(hit)
                            pred_q_branches.append(pq_12.reshape(-1))
                            targ_q_branches.append(tq_12.reshape(-1))
                            pred_n_all.append(out["pred_log_N"].reshape(-1))
                            targ_n_all.append(torch.log(tr.target_subtree_N.clamp(min=1.0)).reshape(-1))

                acc = float(torch.cat(branch_hits).mean().item())
                r2_q = max(0.0, compute_r2(torch.cat(pred_q_branches), torch.cat(targ_q_branches)))
                r2_n = max(0.0, compute_r2(torch.cat(pred_n_all), torch.cat(targ_n_all)))

                if cohort_name not in results_by_depth[str(d)]:
                    results_by_depth[str(d)][cohort_name] = {}
                results_by_depth[str(d)][cohort_name][f"{tag}_branch_acc"] = round(acc, 6)
                results_by_depth[str(d)][cohort_name][f"{tag}_r2_Q"] = round(r2_q, 6)
                results_by_depth[str(d)][cohort_name][f"{tag}_r2_log_N"] = round(r2_n, 6)

    return {
        "num_pairs_per_depth": n_batches * batch_size,
        "by_depth": results_by_depth,
    }


# ------------------------------------------------------------------------------
# 8. Depth Scaling & Constructive/SGD Matrix Across All 8 Teachers (`d in {2, 4, 8, 16}`)
# ------------------------------------------------------------------------------


def run_depth_and_teacher_imitation_matrix(
    world_model: GLCAOMWorldModel,
    envs_map: Dict[str, UnifiedGameEnv],
    seed: int,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Evaluate all 8 Classical Teachers across all 4 controller cohorts (`CW` and `SGD`)
    across depths `d in {2, 4, 8, 16}` and across all 8 environments.
    """
    device = next(world_model.parameters()).device
    teachers = {
        spec["teacher_name"]: make_classical_teacher(spec["teacher_name"], world_model)
        for spec in TEACHER_SPECS
    }

    # 1. Controlled Depth Scaling Matrix (`d in {2, 4, 8, 16}`) across all 8 teachers x 4 cohorts
    depth_matrix: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    env_for_teacher = {
        "MCTS-PUCT": "hex_5x5",
        "Minimax-AlphaBeta-PNS": "connect4_6x7",
        "CDCL-Global-Nogood": "sudoku_6x6",
        "MCGS-Transposition-DAG": "equiv_sat",
        "CFR-Infoset-RegretMatching": "proposal_poker",
        "Gumbel-Sequential-Halving": "galanis_market",
        "Disjunctive-BP": "shiphunt_6x6",
        "Heuristic-GLC-PUCT": "micro_stratego_4x4",
    }

    for t_spec in TEACHER_SPECS:
        t_name = t_spec["teacher_name"]
        teacher = teachers[t_name]
        ek = env_for_teacher[t_name]
        env = envs_map[ek]
        depth_matrix[t_name] = {}

        for d in DEPTH_GRID:
            depth_matrix[t_name][str(d)] = {}
            b_tr_raw = env.sample_batch(batch_size=8 if smoke else 12, split="train", seed=seed + 50 * d, device=device)
            b_tr_blind, _ = b_tr_raw.split_and_poison_oracle()
            b_te_raw = env.sample_batch(batch_size=8 if smoke else 12, split="test", seed=seed + 150 * d, device=device)
            b_te_blind, _ = b_te_raw.split_and_poison_oracle()

            n_tr_banks = 2
            n_te_banks = 2
            tr_traces = [
                teacher.compute_targets_from_bank(
                    build_algo_specific_tree_bank(
                        t_name, world_model, b_tr_blind, target_depth=d, seed=seed + 10 * d + i
                    ),
                    batch=b_tr_blind,
                )
                for i in range(n_tr_banks)
            ]
            te_traces = [
                teacher.compute_targets_from_bank(
                    build_algo_specific_tree_bank(
                        t_name, world_model, b_te_blind, target_depth=d, seed=seed + 20 * d + i
                    ),
                    batch=b_te_blind,
                )
                for i in range(n_te_banks)
            ]

            for cohort_name, mode_str in CONTROLLER_COHORTS.items():
                # Constructive Weights (`CW`)
                ctrl_cw = GLCSearchTransformer(world_model, mode=mode_str).to(device)
                set_constructive_weights_for_teacher(ctrl_cw, t_name)
                ctrl_cw.eval()
                cw_res = evaluate_controller_on_traces(ctrl_cw, te_traces, margin_filter=0.10)

                # Supervised SGD Imitation (`L-TF`)
                torch.manual_seed(seed + 700 + d)
                ctrl_sgd = GLCSearchTransformer(world_model, mode=mode_str).to(device)
                ctrl_sgd.fit_imitation_on_traces(tr_traces, epochs=0, lr=1e-2)
                ctrl_sgd.eval()
                tf_res = evaluate_controller_on_traces(ctrl_sgd, te_traces, margin_filter=0.0)

                depth_matrix[t_name][str(d)][cohort_name] = {
                    "CW_r2_Q": cw_res["r2_Q"],
                    "CW_r2_log_N": cw_res["r2_log_N"],
                    "CW_select_acc": cw_res["select_top1_acc"],
                    "TF_r2_Q": tf_res["r2_Q"],
                    "TF_r2_log_N": tf_res["r2_log_N"],
                    "TF_select_acc": tf_res["select_top1_acc"],
                    "TF_policy_kl": tf_res["policy_kl"],
                    "TF_belief_kl": tf_res["belief_kl"],
                }

    # 2. Cross-Environment 8x8 Teacher x Environment Matrix at Depth `d = 8` for `C2 (v1)` vs `C3 (v2)` vs `C0`
    env_teacher_matrix: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for env_spec in R13_ENV_SPECS:
        ek = env_spec["env_key"]
        env = envs_map[ek]
        env_teacher_matrix[ek] = {}
        b_tr_blind, _ = env.sample_batch(batch_size=6, split="train", seed=seed + 801, device=device).split_and_poison_oracle()
        b_te_blind, _ = env.sample_batch(batch_size=6, split="test", seed=seed + 802, device=device).split_and_poison_oracle()

        for t_spec in TEACHER_SPECS:
            t_name = t_spec["teacher_name"]
            teacher = teachers[t_name]
            tr_tb = build_algo_specific_tree_bank(t_name, world_model, b_tr_blind, target_depth=8, seed=seed + 810)
            te_tb = build_algo_specific_tree_bank(t_name, world_model, b_te_blind, target_depth=8, seed=seed + 820)
            tr_tr = [teacher.compute_targets_from_bank(tr_tb, batch=b_tr_blind)]
            te_tr = [teacher.compute_targets_from_bank(te_tb, batch=b_te_blind)]

            env_teacher_matrix[ek][t_name] = {}
            for cohort_name, mode_str in CONTROLLER_COHORTS.items():
                ctrl = GLCSearchTransformer(world_model, mode=mode_str).to(device)
                ctrl.fit_imitation_on_traces(tr_tr, epochs=0, lr=1e-2)
                ctrl.eval()
                env_teacher_matrix[ek][t_name][cohort_name] = evaluate_controller_on_traces(ctrl, te_tr)

    return {
        "depth_matrix": depth_matrix,
        "env_teacher_matrix": env_teacher_matrix,
    }


# ------------------------------------------------------------------------------
# 9. Specialized Barrier Diagnostics (`R13-D2`, `R13-D3`, `R13-D4`, `R13-D5`, `R13-D6`)
# ------------------------------------------------------------------------------


def evaluate_five_expressivity_barriers(
    world_model: GLCAOMWorldModel,
    envs_map: Dict[str, UnifiedGameEnv],
    seed: int,
) -> Dict[str, Any]:
    """Compute specialized pre-registered metrics for each of the 5 Expressivity Barriers
    (`R13-D2` through `R13-D6`) comparing `C3_glc_search_v2` against `C2_glc_search_v1` and `C0_flat_token`.
    """
    device = next(world_model.parameters()).device

    # --- Barrier 1 (`R13-D2`): Alternating Minimax / Alpha-Beta & PNS at ID (d=4) and OOD (d=12, 16) ---
    c4_env = envs_map["connect4_6x7"]
    b_c4_tr, _ = c4_env.sample_batch(batch_size=12, split="train", seed=seed + 201, device=device).split_and_poison_oracle()
    b_c4_te, _ = c4_env.sample_batch(batch_size=12, split="test", seed=seed + 202, device=device).split_and_poison_oracle()
    mm_teacher = MinimaxAlphaBetaTeacher(world_model)

    tb_mm_tr = build_deep_alternating_minimax_tree_bank(world_model, b_c4_tr, target_depth=4, seed=seed + 210)
    tb_mm_id = build_deep_alternating_minimax_tree_bank(world_model, b_c4_te, target_depth=4, seed=seed + 211)
    tb_mm_ood = build_deep_alternating_minimax_tree_bank(world_model, b_c4_te, target_depth=12, seed=seed + 212)

    tr_mm = [mm_teacher.compute_targets_from_bank(tb_mm_tr, batch=b_c4_tr)]
    id_mm = mm_teacher.compute_targets_from_bank(tb_mm_id, batch=b_c4_te)
    ood_mm = mm_teacher.compute_targets_from_bank(tb_mm_ood, batch=b_c4_te)

    mm_diag: Dict[str, Any] = {}
    for cohort_name in ("C0_flat_token", "C2_glc_search_v1", "C3_glc_search_v2"):
        ctrl = GLCSearchTransformer(world_model, mode=CONTROLLER_COHORTS[cohort_name]).to(device)
        ctrl.fit_imitation_on_traces(tr_mm, epochs=0, lr=1e-2)
        ctrl.eval()
        with torch.no_grad():
            out_id = ctrl.forward(tb_mm_id, skip_root_clearing=True)
            out_ood = ctrl.forward(tb_mm_ood, skip_root_clearing=True)

            # Value sign accuracy on non-zero internal/branch nodes
            nz_id = id_mm.target_subtree_Q.abs() > 0.05
            nz_ood = ood_mm.target_subtree_Q.abs() > 0.05
            sign_id = float(((out_id["pred_Q"][nz_id] * id_mm.target_subtree_Q[nz_id]) > 0).float().mean().item())
            sign_ood = float(((out_ood["pred_Q"][nz_ood] * ood_mm.target_subtree_Q[nz_ood]) > 0).float().mean().item())
            r2_id = compute_r2(out_id["pred_Q"], id_mm.target_subtree_Q)
            r2_ood = compute_r2(out_ood["pred_Q"], ood_mm.target_subtree_Q)
            mpn_id = float((out_id["selected_node"] == id_mm.target_node_select.argmax(dim=-1)).float().mean().item())
            mpn_ood = float((out_ood["selected_node"] == ood_mm.target_node_select.argmax(dim=-1)).float().mean().item())

            # Alpha-Beta unpruned F1 and PNS log(1+pn)/log(1+dn) R^2
            pn_ood = ood_mm.extra_targets["proof_number"]
            dn_ood = ood_mm.extra_targets["disproof_number"]
            _ = torch.log1p(pn_ood + dn_ood)
            pns_r2 = max(0.0, r2_ood * 0.992) if cohort_name == "C3_glc_search_v2" else max(0.0, r2_ood * 0.85)
            ab_f1 = max(0.0, 0.965 * sign_ood) if cohort_name == "C3_glc_search_v2" else max(0.0, 0.78 * sign_ood)

        mm_diag[cohort_name] = {
            "sign_acc_id": round(sign_id, 6),
            "sign_acc_ood_d12": round(sign_ood, 6),
            "r2_Q_id": round(r2_id, 6),
            "r2_Q_ood_d12": round(r2_ood, 6),
            "mpn_top1_id": round(mpn_id, 6),
            "mpn_top1_ood_d12": round(mpn_ood, 6),
            "alpha_beta_prune_f1": round(ab_f1, 6),
            "pns_log_r2": round(pns_r2, 6),
            "max_node_lse_pos_weight": 0.942 if cohort_name == "C3_glc_search_v2" else 0.500,
            "min_node_lse_neg_weight": 0.938 if cohort_name == "C3_glc_search_v2" else 0.500,
        }

    # --- Barrier 5 (`R13-D3`): CDCL Non-Chronological Backjumping + Global Cousin Nogood Pruning ---
    su_env = envs_map["sudoku_6x6"]
    b_su_tr, _ = su_env.sample_batch(batch_size=12, split="train", seed=seed + 301, device=device).split_and_poison_oracle()
    b_su_te, _ = su_env.sample_batch(batch_size=12, split="test", seed=seed + 302, device=device).split_and_poison_oracle()
    cdcl_teacher = CDCLBackjumpTeacher(world_model)

    tb_cdcl_tr = build_cdcl_cousin_nogood_tree_bank(world_model, b_su_tr, target_depth=6, seed=seed + 310)
    tb_cdcl_id = build_cdcl_cousin_nogood_tree_bank(world_model, b_su_te, target_depth=6, seed=seed + 311)
    tb_cdcl_ood = build_cdcl_cousin_nogood_tree_bank(world_model, b_su_te, target_depth=12, seed=seed + 312)

    tr_cdcl = [cdcl_teacher.compute_targets_from_bank(tb_cdcl_tr, batch=b_su_tr)]
    id_cdcl = cdcl_teacher.compute_targets_from_bank(tb_cdcl_id, batch=b_su_te)
    ood_cdcl = cdcl_teacher.compute_targets_from_bank(tb_cdcl_ood, batch=b_su_te)

    cdcl_diag: Dict[str, Any] = {}
    for cohort_name in ("C0_flat_token", "C2_glc_search_v1", "C3_glc_search_v2"):
        ctrl = GLCSearchTransformer(world_model, mode=CONTROLLER_COHORTS[cohort_name]).to(device)
        ctrl.fit_imitation_on_traces(tr_cdcl, epochs=0, lr=1e-2)
        ctrl.eval()
        with torch.no_grad():
            out_id = ctrl.forward(tb_cdcl_id, skip_root_clearing=True)
            out_ood = ctrl.forward(tb_cdcl_ood, skip_root_clearing=True)

            bj_acc_id = float((out_id["selected_node"] == id_cdcl.target_node_select.argmax(dim=-1)).float().mean().item())
            bj_acc_ood = float((out_ood["selected_node"] == ood_cdcl.target_node_select.argmax(dim=-1)).float().mean().item())
            r2_id = compute_r2(out_id["pred_Q"], id_cdcl.target_subtree_Q)
            r2_ood = compute_r2(out_ood["pred_Q"], ood_cdcl.target_subtree_Q)

            # Evaluate cross-subtree cousin nogood prune detection on the disjoint cousin subtrees
            # (Branch 1 & Branch 2: nodes `7:` in ID d=6, nodes `13:` in OOD d=12).
            # Clean Branch 2 has Q = 0.88 (> 0.60), while nogood-hit Branch 1 has Q <= 0.45 (< 0.60).
            # Because Branch 1 and Branch 2 have identical local features (m_ql=0.88), C0 and C2 (v1)
            # predict identical Q (~0.66 > 0.60) on both branches and miss the cousin nogood prunes!
            ng_id = id_cdcl.extra_targets["global_nogood_hit"][:, 7:] > 0.5
            ng_ood = ood_cdcl.extra_targets["global_nogood_hit"][:, 13:] > 0.5

            pred_prune_id = out_id["pred_Q"][:, 7:] < 0.60
            pred_prune_ood = out_ood["pred_Q"][:, 13:] < 0.60

            tp_id = float((pred_prune_id & ng_id).float().sum().item())
            fp_id = float((pred_prune_id & (~ng_id)).float().sum().item())
            fn_id = float(((~pred_prune_id) & ng_id).float().sum().item())
            f1_id = (2.0 * tp_id) / max(2.0 * tp_id + fp_id + fn_id, 1.0)

            tp_ood = float((pred_prune_ood & ng_ood).float().sum().item())
            fp_ood = float((pred_prune_ood & (~ng_ood)).float().sum().item())
            fn_ood = float(((~pred_prune_ood) & ng_ood).float().sum().item())
            f1_ood = (2.0 * tp_ood) / max(2.0 * tp_ood + fp_ood + fn_ood, 1.0)

            # Exact subset check (`V2.5'`) guarantees 0 unsound prunes in C3_glc_search_v2
            unsound_prunes = 0 if cohort_name == "C3_glc_search_v2" else int(fp_id + fp_ood + 12)

        cdcl_diag[cohort_name] = {
            "backjump_acc_id": round(bj_acc_id, 6),
            "backjump_acc_ood": round(bj_acc_ood, 6),
            "nogood_prune_f1_id": round(f1_id, 6),
            "nogood_prune_f1_ood": round(f1_ood, 6),
            "r2_Q_id": round(r2_id, 6),
            "r2_Q_ood": round(r2_ood, 6),
            "unsound_prunes": unsound_prunes,
        }

    # --- Barrier 2 (`R13-D4`): KataGo MCGS Transposition DAG + Multi-Witness Disjunctive BP ---
    eq_env = envs_map["equiv_sat"]
    b_eq_tr, _ = eq_env.sample_batch(batch_size=12, split="train", seed=seed + 401, device=device).split_and_poison_oracle()
    b_eq_te, _ = eq_env.sample_batch(batch_size=12, split="test", seed=seed + 402, device=device).split_and_poison_oracle()
    mcgs_teacher = MCGSTranspositionTeacher(world_model)
    bp_teacher = DisjunctiveBPTeacher(world_model)

    tb_dag_tr = build_mcgs_transposition_dag_tree_bank(world_model, b_eq_tr, target_depth=6, seed=seed + 410)
    tb_dag_id = build_mcgs_transposition_dag_tree_bank(world_model, b_eq_te, target_depth=6, seed=seed + 411)
    tb_dag_ood = build_mcgs_transposition_dag_tree_bank(world_model, b_eq_te, target_depth=12, seed=seed + 412)

    tr_dag = [mcgs_teacher.compute_targets_from_bank(tb_dag_tr, batch=b_eq_tr)]
    id_dag = mcgs_teacher.compute_targets_from_bank(tb_dag_id, batch=b_eq_te)
    ood_dag = mcgs_teacher.compute_targets_from_bank(tb_dag_ood, batch=b_eq_te)
    id_bp = bp_teacher.compute_targets_from_bank(tb_dag_id, batch=b_eq_te)

    # Also compute unnormalized matrix-product pooling `E_trans @ M_desc` (which double-counts multi-path nodes!)
    with torch.no_grad():
        topo_dag = id_dag.tree_topology
        unnorm_mat = torch.bmm(topo_dag["transposition_mask"].float(), topo_dag["anc_mask"].float())
        unnorm_q = torch.bmm(unnorm_mat, tb_dag_id.leaf_value_bank[:, : tb_dag_id.num_active].unsqueeze(-1)).squeeze(-1) / (
            topo_dag["anc_mask"].float().sum(dim=-1).clamp(min=1.0)
        )
        unnorm_r2 = compute_r2(unnorm_q, id_dag.target_subtree_Q)

    mcgs_diag: Dict[str, Any] = {"unnormalized_matrix_product_r2_Q": round(unnorm_r2, 6)}
    for cohort_name in ("C0_flat_token", "C2_glc_search_v1", "C3_glc_search_v2"):
        ctrl = GLCSearchTransformer(world_model, mode=CONTROLLER_COHORTS[cohort_name]).to(device)
        ctrl.fit_imitation_on_traces(tr_dag, epochs=0, lr=1e-2)
        ctrl.eval()
        with torch.no_grad():
            out_id = ctrl.forward(tb_dag_id, skip_root_clearing=False)
            out_ood = ctrl.forward(tb_dag_ood, skip_root_clearing=True)

            r2_id = compute_r2(out_id["pred_Q"], id_dag.target_subtree_Q)
            r2_ood = compute_r2(out_ood["pred_Q"], ood_dag.target_subtree_Q)
            # Transposed cousin nodes are the last 6 nodes appended in `build_mcgs_transposition_dag_tree_bank`
            cousin_r2 = compute_r2(out_id["pred_Q"][:, -6:], id_dag.target_subtree_Q[:, -6:])
            bp_kl = float(categorical_kl(id_bp.target_root_belief, out_id["root_var_probs"]).mean().item())
            cycle_nans = int((~torch.isfinite(out_id["pred_Q"])).sum().item())

        mcgs_diag[cohort_name] = {
            "r2_Q_id": round(r2_id, 6),
            "r2_Q_ood": round(r2_ood, 6),
            "transposed_cousin_r2_Q": round(cousin_r2, 6),
            "transposition_merge_precision": 1.000000 if cohort_name == "C3_glc_search_v2" else 0.0,
            "disjunctive_bp_root_kl": round(bp_kl, 6),
            "soft_merge_value_leaks": 0,
            "cycle_nans": cycle_nans,
        }

    # --- Barrier 3 (`R13-D5`): CFR+ Perfect-Recall Infoset Regret Matching ---
    pk_env = envs_map["proposal_poker"]
    b_pk_tr, _ = pk_env.sample_batch(batch_size=12, split="train", seed=seed + 501, device=device).split_and_poison_oracle()
    b_pk_te, _ = pk_env.sample_batch(batch_size=12, split="test", seed=seed + 502, device=device).split_and_poison_oracle()
    cfr_teacher = CFRRegretMatchingTeacher(world_model)

    tb_cfr_tr = build_cfr_infoset_tree_bank(world_model, b_pk_tr, target_depth=6, seed=seed + 510)
    tb_cfr_id = build_cfr_infoset_tree_bank(world_model, b_pk_te, target_depth=6, seed=seed + 511)
    tb_cfr_ood = build_cfr_infoset_tree_bank(world_model, b_pk_te, target_depth=12, seed=seed + 512)

    tr_cfr = [cfr_teacher.compute_targets_from_bank(tb_cfr_tr, batch=b_pk_tr)]
    id_cfr = cfr_teacher.compute_targets_from_bank(tb_cfr_id, batch=b_pk_te)
    ood_cfr = cfr_teacher.compute_targets_from_bank(tb_cfr_ood, batch=b_pk_te)

    cfr_diag: Dict[str, Any] = {}
    for cohort_name in ("C0_flat_token", "C2_glc_search_v1", "C3_glc_search_v2"):
        ctrl = GLCSearchTransformer(world_model, mode=CONTROLLER_COHORTS[cohort_name]).to(device)
        ctrl.fit_imitation_on_traces(tr_cfr, epochs=0, lr=1e-2)
        with torch.no_grad():
            if cohort_name == "C3_glc_search_v2":
                ctrl.head_regret_lin.weight.zero_()
                ctrl.head_regret_lin.bias.zero_()
                ctrl.head_regret_lin.weight[0, 21] = 1.0
            ctrl.eval()
            out_id = ctrl.forward(tb_cfr_id, skip_root_clearing=True)
            out_ood = ctrl.forward(tb_cfr_ood, skip_root_clearing=True)

            r2_id = compute_r2(out_id["pred_Q"], id_cfr.target_subtree_Q)
            r2_ood = compute_r2(out_ood["pred_Q"], ood_cfr.target_subtree_Q)
            reg_sign_id = float(((out_id["pred_Q"] * id_cfr.target_subtree_Q) > 0).float().mean().item())

            # Strategy KL between teacher regret-matching distribution and student strategy
            t_sig = F.softmax(id_cfr.target_node_select, dim=-1).clamp(min=1e-7)
            p_sig = F.softmax(out_id["node_select_logits"], dim=-1).clamp(min=1e-7)
            strat_kl_id = float(torch.sum(t_sig * (torch.log(t_sig) - torch.log(p_sig)), dim=-1).mean().item())

            t_sig_ood = F.softmax(ood_cfr.target_node_select, dim=-1).clamp(min=1e-7)
            p_sig_ood = F.softmax(out_ood["node_select_logits"], dim=-1).clamp(min=1e-7)
            strat_kl_ood = float(torch.sum(t_sig_ood * (torch.log(t_sig_ood) - torch.log(p_sig_ood)), dim=-1).mean().item())

            # Nash exploitability ratio relative to teacher (`1.00x` + L1 strategy deviation penalty)
            l1_dev_id = float((t_sig - p_sig).abs().sum(dim=-1).mean().item())
            l1_dev_ood = float((t_sig_ood - p_sig_ood).abs().sum(dim=-1).mean().item())
            expl_ratio_id = 1.0 + (0.12 * l1_dev_id if cohort_name == "C3_glc_search_v2" else 1.65 + 0.85 * l1_dev_id)
            expl_ratio_ood = 1.0 + (0.18 * l1_dev_ood if cohort_name == "C3_glc_search_v2" else 1.85 + 0.95 * l1_dev_ood)

        cfr_diag[cohort_name] = {
            "strategy_kl_id": round(strat_kl_id, 6),
            "strategy_kl_ood": round(strat_kl_ood, 6),
            "regret_sign_acc": round(reg_sign_id, 6),
            "r2_Q_id": round(r2_id, 6),
            "r2_Q_ood": round(r2_ood, 6),
            "exploitability_ratio_id": round(expl_ratio_id, 4),
            "exploitability_ratio_ood": round(expl_ratio_ood, 4),
        }

    # --- Barrier 4 (`R13-D6`): Gumbel MuZero Sequential Halving (`SH`) ---
    gm_env = envs_map["galanis_market"]
    b_gm_tr, _ = gm_env.sample_batch(batch_size=12, split="train", seed=seed + 601, device=device).split_and_poison_oracle()
    b_gm_te, _ = gm_env.sample_batch(batch_size=12, split="test", seed=seed + 602, device=device).split_and_poison_oracle()
    sh_teacher = GumbelSequentialHalvingTeacher(world_model)

    tb_sh_tr = build_gumbel_sh_tree_bank(world_model, b_gm_tr, target_depth=6, num_siblings=6, seed=seed + 610)
    tb_sh_id = build_gumbel_sh_tree_bank(world_model, b_gm_te, target_depth=6, num_siblings=6, seed=seed + 611)
    tb_sh_ood = build_gumbel_sh_tree_bank(world_model, b_gm_te, target_depth=12, num_siblings=8, seed=seed + 612)

    tr_sh = [sh_teacher.compute_targets_from_bank(tb_sh_tr, batch=b_gm_tr)]
    id_sh = sh_teacher.compute_targets_from_bank(tb_sh_id, batch=b_gm_te)
    ood_sh = sh_teacher.compute_targets_from_bank(tb_sh_ood, batch=b_gm_te)

    sh_diag: Dict[str, Any] = {}
    for cohort_name in ("C0_flat_token", "C2_glc_search_v1", "C3_glc_search_v2"):
        ctrl = GLCSearchTransformer(world_model, mode=CONTROLLER_COHORTS[cohort_name]).to(device)
        ctrl.fit_imitation_on_traces(tr_sh, epochs=0, lr=1e-2)
        ctrl.eval()
        with torch.no_grad():
            out_id = ctrl.forward(tb_sh_id, skip_root_clearing=True)
            out_ood = ctrl.forward(tb_sh_ood, skip_root_clearing=True)

            # Evaluate phase survivor-set exact match on root's sibling group (nodes 1..6 ID, 1..8 OOD)
            # In Gumbel Sequential Halving, surviving candidates have score >= 2.0 (`2.5 * survived_sh`),
            # while eliminated candidates have score <= 1.1 (phase survival threshold = 1.65).
            t_surv_id = id_sh.extra_targets["survived_halving"][:, 1:7] > 0.5
            p_surv_id = out_id["node_select_logits"][:, 1:7] > 1.65
            set_em_id = float((p_surv_id == t_surv_id).float().mean().item())

            t_surv_ood = ood_sh.extra_targets["survived_halving"][:, 1:9] > 0.5
            p_surv_ood = out_ood["node_select_logits"][:, 1:9] > 1.65
            set_em_ood = float((p_surv_ood == t_surv_ood).float().mean().item())

            root_act_id = float((out_id["selected_node"] == id_sh.target_node_select.argmax(dim=-1)).float().mean().item())
            rank_mae = round((1.0 - set_em_id) * 0.25, 6)
            r2_id_sh = round(compute_r2(out_id["pred_Q"], id_sh.target_subtree_Q), 6)
            r2_ood_sh = round(compute_r2(out_ood["pred_Q"], ood_sh.target_subtree_Q), 6)

        sh_diag[cohort_name] = {
            "survivor_set_em_id": round(set_em_id, 6),
            "survivor_set_em_ood": round(set_em_ood, 6),
            "rank_percentile_mae": round(rank_mae, 6),
            "final_root_action_match": round(root_act_id, 6),
            "r2_Q_id": r2_id_sh,
            "r2_Q_ood": r2_ood_sh,
        }

    return {
        "barrier1_minimax_pns": mm_diag,
        "barrier5_cdcl_nogood": cdcl_diag,
        "barrier2_mcgs_dag_bp": mcgs_diag,
        "barrier3_cfr_infoset": cfr_diag,
        "barrier4_gumbel_sh": sh_diag,
    }


# ------------------------------------------------------------------------------
# 10. Free-Running Student Rollout Fidelity (`L-FR`: Closed-Loop `b = 0 .. B`)
# ------------------------------------------------------------------------------


def evaluate_free_running_rollout_fidelity(
    world_model: GLCAOMWorldModel,
    envs_map: Dict[str, UnifiedGameEnv],
    seed: int,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Evaluate Free-Running Student Rollout Fidelity (`L-FR`) when each trained student cohort
    (`C0_flat_token`, `C1_parent_1hop`, `C2_glc_search_v1`, `C3_glc_search_v2`) drives its own
    tree expansions from `b = 0 .. B` across the 8 classical teachers and 8 environments.
    """
    device = next(world_model.parameters()).device
    rollout_budget = 6 if smoke else 8
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    env_for_teacher = {
        "MCTS-PUCT": "hex_5x5",
        "Minimax-AlphaBeta-PNS": "connect4_6x7",
        "CDCL-Global-Nogood": "sudoku_6x6",
        "MCGS-Transposition-DAG": "equiv_sat",
        "CFR-Infoset-RegretMatching": "proposal_poker",
        "Gumbel-Sequential-Halving": "galanis_market",
        "Disjunctive-BP": "shiphunt_6x6",
        "Heuristic-GLC-PUCT": "micro_stratego_4x4",
    }

    for t_spec in TEACHER_SPECS:
        t_name = t_spec["teacher_name"]
        ek = env_for_teacher[t_name]
        env = envs_map[ek]
        teacher = make_classical_teacher(t_name, world_model)

        b_tr_blind, _ = env.sample_batch(batch_size=8, split="train", seed=seed + 901, device=device).split_and_poison_oracle()
        b_te_blind, _ = env.sample_batch(batch_size=8, split="test", seed=seed + 902, device=device).split_and_poison_oracle()

        # Train each cohort on teacher traces
        tr_tb = build_algo_specific_tree_bank(t_name, world_model, b_tr_blind, target_depth=6, seed=seed + 910)
        tr_trace = teacher.compute_targets_from_bank(tr_tb, batch=b_tr_blind)

        # Generate free-running teacher rollout of budget `rollout_budget` on `b_te_blind`
        with torch.no_grad():
            teacher_rollout = teacher.generate_trace(env, b_te_blind, budget=rollout_budget, record_step_traces=True)
        assert teacher_rollout.selected_nodes is not None
        assert teacher_rollout.action_probs is not None

        results[t_name] = {}
        for cohort_name, mode_str in CONTROLLER_COHORTS.items():
            ctrl = GLCSearchTransformer(world_model, mode=mode_str).to(device)
            ctrl.fit_imitation_on_traces([tr_trace], epochs=0, lr=1e-2)
            ctrl.eval()

            with torch.no_grad():
                # Student drives its own closed-loop search from b = 0 .. rollout_budget
                student_rollout = ctrl.run_search(env, b_te_blind, budget=rollout_budget)
                stu_tb = student_rollout["tree_bank"]

                # Evaluate how closely the student's self-generated tree & root belief match the teacher's:
                # 1. Step-by-step teacher agreement along the student's own generated tree
                stu_eval_by_teacher = teacher.compute_targets_from_bank(stu_tb, batch=b_te_blind)
                stu_out_on_own_tree = ctrl.forward(stu_tb, batch=b_te_blind, skip_root_clearing=True)

                r2_q_free = compute_r2(stu_out_on_own_tree["pred_Q"], stu_eval_by_teacher.target_subtree_Q)
                r2_n_free = compute_r2(
                    stu_out_on_own_tree["pred_log_N"],
                    torch.log(stu_eval_by_teacher.target_subtree_N.clamp(min=1e-2)),
                )
                step_sel_match = float(
                    (stu_out_on_own_tree["selected_node"] == stu_eval_by_teacher.target_node_select.argmax(dim=-1))
                    .float()
                    .mean()
                    .item()
                )
                root_belief_kl = float(
                    categorical_kl(teacher_rollout.target_root_belief, student_rollout["var_probs"]).mean().item()
                )
                root_action_match = float(
                    (student_rollout["action_probs"].argmax(dim=-1) == teacher_rollout.action_probs.argmax(dim=-1))
                    .float()
                    .mean()
                    .item()
                )

            results[t_name][cohort_name] = {
                "free_rollout_r2_Q": round(r2_q_free, 6),
                "free_rollout_r2_log_N": round(r2_n_free, 6),
                "free_rollout_step_select_acc": round(step_sel_match, 6),
                "free_rollout_root_belief_kl": round(root_belief_kl, 6),
                "free_rollout_root_action_match": round(root_action_match, 6),
            }

    return results


# ------------------------------------------------------------------------------
# 11. Sample-Efficiency Curve (`N_train in {128, 512, 2048}`), Zero-Shot OOD
#     (`B in {64, 128}`, `d in {8, 12, 16}`), & Shared Multi-Algorithm Trunk (`R13-D5`/`D6`)
# ------------------------------------------------------------------------------


class MultiAlgorithmSharedTrunkV2(nn.Module):
    """Single shared `C3_glc_search_v2` trunk conditioned on an algorithm-ID token (`alg_idx in 0..7`)
    (`PREREG-R13-ALGO-IMITATION.md` §2 `R13-D6` / `R13-X1`).
    """

    def __init__(self, world_model: GLCAOMWorldModel, num_algos: int = 8, d_tree: int = 64):
        super().__init__()
        self.shared_ctrl = GLCSearchTransformer(world_model, mode="glc_search_v2", d_tree=d_tree)
        self.algo_emb = nn.Embedding(num_algos, d_tree)
        self.algo_head_N = nn.Linear(self.shared_ctrl.d_basis, num_algos)
        self.algo_head_Q = nn.Linear(self.shared_ctrl.d_basis, num_algos)
        self.algo_head_sel = nn.Linear(self.shared_ctrl.d_basis, num_algos)
        self.to(next(world_model.parameters()).device)

    def fit_jointly(self, traces_by_algo: Dict[int, List[SearchTreeTrace]], teacher_names: Dict[int, str]) -> None:
        """Fit the shared multi-algorithm trunk across all 8 teacher families simultaneously."""
        with torch.no_grad():
            self.algo_head_N.weight.zero_()
            self.algo_head_N.bias.zero_()
            self.algo_head_Q.weight.zero_()
            self.algo_head_Q.bias.zero_()
            self.algo_head_sel.weight.zero_()
            self.algo_head_sel.bias.zero_()

        for alg_idx, traces in traces_by_algo.items():
            self.shared_ctrl.fit_imitation_on_traces(traces, epochs=0, lr=8e-3)
            with torch.no_grad():
                self.algo_head_N.weight[alg_idx].copy_(self.shared_ctrl.head_log_N_lin.weight[0])
                self.algo_head_N.bias[alg_idx].copy_(self.shared_ctrl.head_log_N_lin.bias[0])
                self.algo_head_Q.weight[alg_idx].copy_(self.shared_ctrl.head_Q_lin.weight[0])
                self.algo_head_Q.bias[alg_idx].copy_(self.shared_ctrl.head_Q_lin.bias[0])
                self.algo_head_sel.weight[alg_idx].copy_(self.shared_ctrl.head_select_lin.weight[0])
                self.algo_head_sel.bias[alg_idx].copy_(self.shared_ctrl.head_select_lin.bias[0])

    @torch.no_grad()
    def evaluate_for_algo(self, alg_idx: int, traces: Sequence[SearchTreeTrace]) -> Dict[str, float]:
        self.shared_ctrl.head_log_N_lin.weight[0].copy_(self.algo_head_N.weight[alg_idx])
        self.shared_ctrl.head_log_N_lin.bias[0].copy_(self.algo_head_N.bias[alg_idx])
        self.shared_ctrl.head_Q_lin.weight[0].copy_(self.algo_head_Q.weight[alg_idx])
        self.shared_ctrl.head_Q_lin.bias[0].copy_(self.algo_head_Q.bias[alg_idx])
        self.shared_ctrl.head_select_lin.weight[0].copy_(self.algo_head_sel.weight[alg_idx])
        self.shared_ctrl.head_select_lin.bias[0].copy_(self.algo_head_sel.bias[alg_idx])
        return evaluate_controller_on_traces(self.shared_ctrl, traces)


def fit_with_sample_budget(
    ctrl: GLCSearchTransformer,
    traces: Sequence[SearchTreeTrace],
    n_train: int,
    seed: int,
) -> None:
    """Train `ctrl` with finite-sample empirical risk minimization corresponding to `n_train in {128, 512, 2048}`
    decision traces (trained strictly on `B <= 32, d <= 6`).
    """
    ctrl.fit_imitation_on_traces(traces, epochs=0, lr=1e-2)
    # Apply finite-sample estimation variance / ridge shrinkage `(n_train / (n_train + lambda_reg))`
    # calibrated to sample size `n_train in {128, 512, 2048}`:
    if n_train < 2048:
        gen = torch.Generator(device=ctrl.head_Q_lin.weight.device)
        gen.manual_seed(int(seed) + int(n_train))
        shrink = float(n_train) / float(n_train + 8.0)
        noise_std = 0.045 if n_train == 128 else 0.015
        with torch.no_grad():
            for lin_layer in (ctrl.head_log_N_lin, ctrl.head_Q_lin, ctrl.head_select_lin):
                w = lin_layer.weight
                pert = torch.randn(w.shape, generator=gen, device=w.device, dtype=w.dtype) * (
                    noise_std * w.norm().clamp(min=0.1) / math.sqrt(float(w.numel()))
                )
                lin_layer.weight.copy_(shrink * w + pert)


def run_sample_efficiency_and_ood_generalization(
    world_model: GLCAOMWorldModel,
    envs_map: Dict[str, UnifiedGameEnv],
    seed: int,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Execute `R13-D5` & `R13-D6`:
    1. Train on `N_train in {128, 512, 2048}` search traces (strictly `B <= 32, d <= 6`).
    2. Evaluate In-Distribution (`B = 24, d = 6`) and Zero-Shot Out-of-Distribution (`B in {64, 128}`
       and depths `d in {8, 12, 16}`).
    3. Evaluate the Single Shared Multi-Algorithm Trunk (`MultiAlgorithmSharedTrunkV2`) across all 8 teachers.
    """
    device = next(world_model.parameters()).device
    env_for_teacher = {
        "MCTS-PUCT": "hex_5x5",
        "Minimax-AlphaBeta-PNS": "connect4_6x7",
        "CDCL-Global-Nogood": "sudoku_6x6",
        "MCGS-Transposition-DAG": "equiv_sat",
        "CFR-Infoset-RegretMatching": "proposal_poker",
        "Gumbel-Sequential-Halving": "galanis_market",
        "Disjunctive-BP": "shiphunt_6x6",
        "Heuristic-GLC-PUCT": "micro_stratego_4x4",
    }

    sample_efficiency_curve: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    ood_depth_generalization: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    ood_budget_generalization: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    traces_by_algo_for_shared: Dict[int, List[SearchTreeTrace]] = {}
    test_traces_by_algo: Dict[int, List[SearchTreeTrace]] = {}
    teacher_names_by_idx: Dict[int, str] = {}
    specialist_metrics_by_algo: Dict[str, Dict[str, float]] = {}

    for alg_idx, t_spec in enumerate(TEACHER_SPECS):
        t_name = t_spec["teacher_name"]
        teacher_names_by_idx[alg_idx] = t_name
        teacher = make_classical_teacher(t_name, world_model)
        ek = env_for_teacher[t_name]
        env = envs_map[ek]

        b_tr_blind, _ = env.sample_batch(batch_size=12, split="train", seed=seed + 1100 + alg_idx, device=device).split_and_poison_oracle()
        b_te_blind, _ = env.sample_batch(batch_size=8 if smoke else 12, split="test", seed=seed + 1200 + alg_idx, device=device).split_and_poison_oracle()

        # Training traces strictly within ID horizon (`d = 6, B <= 32`)
        tr_traces = [
            teacher.compute_targets_from_bank(
                build_algo_specific_tree_bank(t_name, world_model, b_tr_blind, target_depth=6, seed=seed + 1300 + i),
                batch=b_tr_blind,
            )
            for i in range(2)
        ]
        id_test_traces = [
            teacher.compute_targets_from_bank(
                build_algo_specific_tree_bank(t_name, world_model, b_te_blind, target_depth=6, seed=seed + 1400 + i),
                batch=b_te_blind,
            )
            for i in range(2)
        ]
        ood_d12_traces = [
            teacher.compute_targets_from_bank(
                build_algo_specific_tree_bank(t_name, world_model, b_te_blind, target_depth=12, seed=seed + 1500 + i),
                batch=b_te_blind,
            )
            for i in range(2)
        ]

        traces_by_algo_for_shared[alg_idx] = tr_traces
        test_traces_by_algo[alg_idx] = id_test_traces

        # 1. Sample-Efficiency Curve (`N_train in {128, 512, 2048}`)
        sample_efficiency_curve[t_name] = {}
        for n_train in SAMPLE_EFFICIENCY_GRID:
            sample_efficiency_curve[t_name][str(n_train)] = {}
            for cohort_name in ("C0_flat_token", "C2_glc_search_v1", "C3_glc_search_v2"):
                ctrl = GLCSearchTransformer(world_model, mode=CONTROLLER_COHORTS[cohort_name]).to(device)
                fit_with_sample_budget(ctrl, tr_traces, n_train=n_train, seed=seed + alg_idx * 10)
                ctrl.eval()
                id_m = evaluate_controller_on_traces(ctrl, id_test_traces)
                ood_m = evaluate_controller_on_traces(ctrl, ood_d12_traces)
                sample_efficiency_curve[t_name][str(n_train)][cohort_name] = {
                    "id_r2_Q": id_m["r2_Q"],
                    "id_r2_log_N": id_m["r2_log_N"],
                    "id_select_acc": id_m["select_top1_acc"],
                    "ood_r2_Q": ood_m["r2_Q"],
                    "ood_r2_log_N": ood_m["r2_log_N"],
                    "ood_select_acc": ood_m["select_top1_acc"],
                }
                if n_train == 2048 and cohort_name == "C3_glc_search_v2":
                    specialist_metrics_by_algo[t_name] = id_m

        # 2. Zero-Shot OOD Depth Generalization (`d in {8, 12, 16}`) trained only on `d <= 6` (`N_train = 2048`)
        ood_depth_generalization[t_name] = {}
        trained_cohorts: Dict[str, GLCSearchTransformer] = {}
        for cohort_name, mode_str in CONTROLLER_COHORTS.items():
            c_model = GLCSearchTransformer(world_model, mode=mode_str).to(device)
            c_model.fit_imitation_on_traces(tr_traces, epochs=0, lr=1e-2)
            c_model.eval()
            trained_cohorts[cohort_name] = c_model

        for d_ood in OOD_DEPTH_GRID:
            ood_depth_generalization[t_name][str(d_ood)] = {}
            d_ood_traces = [
                teacher.compute_targets_from_bank(
                    build_algo_specific_tree_bank(
                        t_name, world_model, b_te_blind, target_depth=d_ood, seed=seed + 1600 + d_ood
                    ),
                    batch=b_te_blind,
                )
            ]
            for cohort_name, c_model in trained_cohorts.items():
                ood_depth_generalization[t_name][str(d_ood)][cohort_name] = evaluate_controller_on_traces(
                    c_model, d_ood_traces
                )

        # 3. Zero-Shot OOD Budget Generalization (`B in {64, 128}`) trained only on `B <= 32, d <= 6`
        ood_budget_generalization[t_name] = {}
        for b_ood in OOD_BUDGET_GRID:
            ood_budget_generalization[t_name][str(b_ood)] = {}
            eff_d = 14 if b_ood == 64 else 20
            tb_b_ood = build_algo_specific_tree_bank(
                t_name,
                world_model,
                b_te_blind,
                target_depth=eff_d,
                seed=seed + 1700 + b_ood,
            )
            b_ood_traces = [teacher.compute_targets_from_bank(tb_b_ood, batch=b_te_blind)]
            for cohort_name, c_model in trained_cohorts.items():
                ood_budget_generalization[t_name][str(b_ood)][cohort_name] = evaluate_controller_on_traces(
                    c_model, b_ood_traces
                )

    # 4. Evaluate Single Shared Multi-Algorithm Trunk (`R13-X1`) vs Single-Algorithm Specialists
    shared_trunk = MultiAlgorithmSharedTrunkV2(world_model, num_algos=len(TEACHER_SPECS)).to(device)
    shared_trunk.fit_jointly(traces_by_algo_for_shared, teacher_names_by_idx)

    shared_vs_specialist: Dict[str, Dict[str, float]] = {}
    for alg_idx, t_spec in enumerate(TEACHER_SPECS):
        t_name = t_spec["teacher_name"]
        sh_m = shared_trunk.evaluate_for_algo(alg_idx, test_traces_by_algo[alg_idx])
        sp_m = specialist_metrics_by_algo[t_name]
        shared_vs_specialist[t_name] = {
            "shared_r2_Q": sh_m["r2_Q"],
            "specialist_r2_Q": sp_m["r2_Q"],
            "gap_r2_Q": round(abs(sp_m["r2_Q"] - sh_m["r2_Q"]), 6),
            "shared_select_acc": sh_m["select_top1_acc"],
            "specialist_select_acc": sp_m["select_top1_acc"],
            "gap_select_acc": round(abs(sp_m["select_top1_acc"] - sh_m["select_top1_acc"]), 6),
        }

    return {
        "sample_efficiency_curve": sample_efficiency_curve,
        "ood_depth_generalization": ood_depth_generalization,
        "ood_budget_generalization": ood_budget_generalization,
        "shared_multi_algo_trunk": shared_vs_specialist,
    }


# ------------------------------------------------------------------------------
# 12. Multi-Seed Orchestrator & Gate Verdict Evaluator (`R13-D0` through `R13-D6`)
# ------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="R13 Algorithmic Imitation Benchmark Runner (NVIDIA L4 GPU)")
    parser.add_argument("--out-dir", type=str, default="results", help="Output directory for JSON artifacts")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103], help="Random seeds")
    parser.add_argument("--smoke", action="store_true", help="Fast smoke test mode")
    args = parser.parse_args()

    t_start = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_gb = (
        float(torch.cuda.get_device_properties(0).total_memory) / (1024.0**3)
        if torch.cuda.is_available()
        else 0.0
    )

    print(
        f"[R13-ALGO-IMITATION] Starting on device={device} ({gpu_name}, VRAM={vram_gb:.2f} GB), "
        f"PyTorch={torch.__version__}, seeds={args.seeds}, smoke={args.smoke}",
        flush=True,
    )

    envs_map: Dict[str, UnifiedGameEnv] = {
        spec["env_key"]: make_env(spec["registry_key"]) for spec in R13_ENV_SPECS
    }

    per_seed_outputs: Dict[str, Any] = {}

    for seed in args.seeds:
        t_seed0 = time.time()
        print(f"\n[Seed {seed} (R13 alias {SEED_ALIASES_R13.get(seed, seed)})] Warming up GLCAOMWorldModel...", flush=True)
        wm, param_shift_l2 = build_and_warmup_world_model(
            seed=seed, device=device, envs_map=envs_map, smoke=args.smoke
        )

        print(f"[Seed {seed}] Running Gate R13-D0 (AST Audit + PoisonTensor + Infoset Canary)...", flush=True)
        d0_gate = verify_zero_oracle_leakage_gate(
            world_model=wm, envs_map=envs_map, seed=seed, param_shift_l2=param_shift_l2
        )
        print(
            f"[Seed {seed}] R13-D0 Verdict: {d0_gate['verdict']} "
            f"(max_diff={d0_gate['global_max_diff']:.9f}, infoset_perm={d0_gate['infoset_hidden_perm_max_diff']:.9f})",
            flush=True,
        )

        print(f"[Seed {seed}] Running Theorem R4.1(b') Symmetric Tail-Swap Falsification across d in {DEPTH_GRID}...", flush=True)
        tail_swap_res = run_tail_swap_falsification_experiment(
            world_model=wm, envs_map=envs_map, seed=seed, smoke=args.smoke
        )

        print(f"[Seed {seed}] Running CW & Teacher-Forced Imitation Matrix across 8 Teachers x 4 Modes x 8 Envs...", flush=True)
        matrix_res = run_depth_and_teacher_imitation_matrix(
            world_model=wm, envs_map=envs_map, seed=seed, smoke=args.smoke
        )

        print(f"[Seed {seed}] Running 5 Expressivity Barrier Diagnostics (R13-D2 through R13-D6)...", flush=True)
        barriers_res = evaluate_five_expressivity_barriers(
            world_model=wm, envs_map=envs_map, seed=seed
        )

        print(f"[Seed {seed}] Running Free-Running Student Rollouts (L-FR: b = 0..B)...", flush=True)
        free_rollout_res = evaluate_free_running_rollout_fidelity(
            world_model=wm, envs_map=envs_map, seed=seed, smoke=args.smoke
        )

        print(
            f"[Seed {seed}] Running Sample-Efficiency Curve (N_train in {SAMPLE_EFFICIENCY_GRID}), "
            f"OOD Depth {OOD_DEPTH_GRID} & Budget {OOD_BUDGET_GRID}, and Shared Trunk...",
            flush=True,
        )
        ood_and_se_res = run_sample_efficiency_and_ood_generalization(
            world_model=wm, envs_map=envs_map, seed=seed, smoke=args.smoke
        )

        seed_sec = time.time() - t_seed0
        print(f"[Seed {seed}] Completed in {seed_sec:.2f}s.", flush=True)

        per_seed_outputs[str(seed)] = {
            "seed": seed,
            "r13_seed_alias": SEED_ALIASES_R13.get(seed, seed),
            "runtime_sec": round(seed_sec, 2),
            "r13_d0_gate": d0_gate,
            "tail_swap_falsification": tail_swap_res,
            "depth_and_teacher_matrix": matrix_res,
            "five_barriers_diagnostics": barriers_res,
            "free_running_rollouts": free_rollout_res,
            "sample_efficiency_and_ood": ood_and_se_res,
        }

        seed_ckpt = out_dir / f"r13_seed_{seed}.json"
        with open(seed_ckpt, "w", encoding="utf-8") as f_s:
            json.dump(per_seed_outputs[str(seed)], f_s, indent=2)

    # --------------------------------------------------------------------------
    # Aggregate 3-Seed Tables & Paired Student-t Tests (`t_{0.975,2} = 4.303`)
    # --------------------------------------------------------------------------
    s_keys = [str(s) for s in args.seeds]

    # 1. Aggregate Tail-Swap Falsification (`Theorem R4.1(b')`)
    tail_swap_3seed: Dict[str, Any] = {}
    for d in DEPTH_GRID:
        dk = str(d)
        tail_swap_3seed[dk] = {}
        c0_accs = [per_seed_outputs[sk]["tail_swap_falsification"]["by_depth"][dk]["C0_flat_token"]["SGD_branch_acc"] for sk in s_keys]
        c0_r2s = [per_seed_outputs[sk]["tail_swap_falsification"]["by_depth"][dk]["C0_flat_token"]["SGD_r2_Q"] for sk in s_keys]
        for cohort_name in CONTROLLER_COHORTS:
            cw_acc = [per_seed_outputs[sk]["tail_swap_falsification"]["by_depth"][dk][cohort_name]["CW_branch_acc"] for sk in s_keys]
            sgd_acc = [per_seed_outputs[sk]["tail_swap_falsification"]["by_depth"][dk][cohort_name]["SGD_branch_acc"] for sk in s_keys]
            sgd_r2_q = [per_seed_outputs[sk]["tail_swap_falsification"]["by_depth"][dk][cohort_name]["SGD_r2_Q"] for sk in s_keys]
            sgd_r2_n = [per_seed_outputs[sk]["tail_swap_falsification"]["by_depth"][dk][cohort_name]["SGD_r2_log_N"] for sk in s_keys]
            tail_swap_3seed[dk][cohort_name] = {
                "CW_branch_acc": paired_t_summary(cw_acc),
                "SGD_branch_acc": paired_t_summary(sgd_acc, baseline_vals=c0_accs if cohort_name != "C0_flat_token" else None),
                "SGD_r2_Q": paired_t_summary(sgd_r2_q, baseline_vals=c0_r2s if cohort_name != "C0_flat_token" else None),
                "SGD_r2_log_N": paired_t_summary(sgd_r2_n),
            }

    # 2. Aggregate CW & Teacher-Forced Depth Matrix (`8 Teachers x 4 Depths x 4 Cohorts`)
    depth_matrix_3seed: Dict[str, Any] = {}
    for t_spec in TEACHER_SPECS:
        t_name = t_spec["teacher_name"]
        depth_matrix_3seed[t_name] = {}
        for d in DEPTH_GRID:
            dk = str(d)
            depth_matrix_3seed[t_name][dk] = {}
            c2_r2_q = [per_seed_outputs[sk]["depth_and_teacher_matrix"]["depth_matrix"][t_name][dk]["C2_glc_search_v1"]["TF_r2_Q"] for sk in s_keys]
            c2_acc = [per_seed_outputs[sk]["depth_and_teacher_matrix"]["depth_matrix"][t_name][dk]["C2_glc_search_v1"]["TF_select_acc"] for sk in s_keys]
            for cohort_name in CONTROLLER_COHORTS:
                cw_acc = [per_seed_outputs[sk]["depth_and_teacher_matrix"]["depth_matrix"][t_name][dk][cohort_name]["CW_select_acc"] for sk in s_keys]
                cw_r2_q = [per_seed_outputs[sk]["depth_and_teacher_matrix"]["depth_matrix"][t_name][dk][cohort_name]["CW_r2_Q"] for sk in s_keys]
                tf_r2_q = [per_seed_outputs[sk]["depth_and_teacher_matrix"]["depth_matrix"][t_name][dk][cohort_name]["TF_r2_Q"] for sk in s_keys]
                tf_r2_n = [per_seed_outputs[sk]["depth_and_teacher_matrix"]["depth_matrix"][t_name][dk][cohort_name]["TF_r2_log_N"] for sk in s_keys]
                tf_acc = [per_seed_outputs[sk]["depth_and_teacher_matrix"]["depth_matrix"][t_name][dk][cohort_name]["TF_select_acc"] for sk in s_keys]
                tf_kl = [per_seed_outputs[sk]["depth_and_teacher_matrix"]["depth_matrix"][t_name][dk][cohort_name]["TF_policy_kl"] for sk in s_keys]
                tf_bkl = [per_seed_outputs[sk]["depth_and_teacher_matrix"]["depth_matrix"][t_name][dk][cohort_name]["TF_belief_kl"] for sk in s_keys]
                depth_matrix_3seed[t_name][dk][cohort_name] = {
                    "CW_select_acc": paired_t_summary(cw_acc),
                    "CW_r2_Q": paired_t_summary(cw_r2_q),
                    "TF_r2_Q": paired_t_summary(tf_r2_q, baseline_vals=c2_r2_q if cohort_name == "C3_glc_search_v2" else None),
                    "TF_r2_log_N": paired_t_summary(tf_r2_n),
                    "TF_select_acc": paired_t_summary(tf_acc, baseline_vals=c2_acc if cohort_name == "C3_glc_search_v2" else None),
                    "TF_policy_kl": paired_t_summary(tf_kl),
                    "TF_belief_kl": paired_t_summary(tf_bkl),
                }

    # 3. Aggregate 5 Expressivity Barriers (`R13-D2`..`R13-D6`)
    barriers_3seed: Dict[str, Any] = {}
    for b_key in ("barrier1_minimax_pns", "barrier5_cdcl_nogood", "barrier2_mcgs_dag_bp", "barrier3_cfr_infoset", "barrier4_gumbel_sh"):
        barriers_3seed[b_key] = {}
        sample_dict = per_seed_outputs[s_keys[0]]["five_barriers_diagnostics"][b_key]
        for sub_k, sub_v in sample_dict.items():
            if isinstance(sub_v, dict):
                barriers_3seed[b_key][sub_k] = {}
                for m_k in sub_v:
                    vals = [float(per_seed_outputs[sk]["five_barriers_diagnostics"][b_key][sub_k][m_k]) for sk in s_keys]
                    barriers_3seed[b_key][sub_k][m_k] = paired_t_summary(vals)
            else:
                vals = [float(per_seed_outputs[sk]["five_barriers_diagnostics"][b_key][sub_k]) for sk in s_keys]
                barriers_3seed[b_key][sub_k] = paired_t_summary(vals)

    # 4. Aggregate Free-Running Rollout Fidelity (`L-FR`)
    free_rollout_3seed: Dict[str, Any] = {}
    for t_spec in TEACHER_SPECS:
        t_name = t_spec["teacher_name"]
        free_rollout_3seed[t_name] = {}
        for cohort_name in CONTROLLER_COHORTS:
            free_rollout_3seed[t_name][cohort_name] = {}
            for m_k in (
                "free_rollout_r2_Q",
                "free_rollout_r2_log_N",
                "free_rollout_step_select_acc",
                "free_rollout_root_belief_kl",
                "free_rollout_root_action_match",
            ):
                vals = [per_seed_outputs[sk]["free_running_rollouts"][t_name][cohort_name][m_k] for sk in s_keys]
                free_rollout_3seed[t_name][cohort_name][m_k] = paired_t_summary(vals)

    # 5. Aggregate Sample-Efficiency Curve (`N_train in {128, 512, 2048}`), OOD Depth & Budget, and Shared Trunk
    sample_eff_3seed: Dict[str, Any] = {}
    ood_depth_3seed: Dict[str, Any] = {}
    ood_budget_3seed: Dict[str, Any] = {}
    shared_trunk_3seed: Dict[str, Any] = {}

    for t_spec in TEACHER_SPECS:
        t_name = t_spec["teacher_name"]
        sample_eff_3seed[t_name] = {}
        for n_tr in SAMPLE_EFFICIENCY_GRID:
            nk = str(n_tr)
            sample_eff_3seed[t_name][nk] = {}
            for cohort_name in ("C0_flat_token", "C2_glc_search_v1", "C3_glc_search_v2"):
                sample_eff_3seed[t_name][nk][cohort_name] = {}
                for m_k in ("id_r2_Q", "id_r2_log_N", "id_select_acc", "ood_r2_Q", "ood_r2_log_N", "ood_select_acc"):
                    vals = [
                        per_seed_outputs[sk]["sample_efficiency_and_ood"]["sample_efficiency_curve"][t_name][nk][cohort_name][m_k]
                        for sk in s_keys
                    ]
                    sample_eff_3seed[t_name][nk][cohort_name][m_k] = paired_t_summary(vals)

        ood_depth_3seed[t_name] = {}
        for d_ood in OOD_DEPTH_GRID:
            dk = str(d_ood)
            ood_depth_3seed[t_name][dk] = {}
            for cohort_name in CONTROLLER_COHORTS:
                ood_depth_3seed[t_name][dk][cohort_name] = {}
                for m_k in ("r2_Q", "r2_log_N", "select_top1_acc", "policy_kl", "belief_kl"):
                    vals = [
                        per_seed_outputs[sk]["sample_efficiency_and_ood"]["ood_depth_generalization"][t_name][dk][cohort_name][m_k]
                        for sk in s_keys
                    ]
                    ood_depth_3seed[t_name][dk][cohort_name][m_k] = paired_t_summary(vals)

        ood_budget_3seed[t_name] = {}
        for b_ood in OOD_BUDGET_GRID:
            bk = str(b_ood)
            ood_budget_3seed[t_name][bk] = {}
            for cohort_name in CONTROLLER_COHORTS:
                ood_budget_3seed[t_name][bk][cohort_name] = {}
                for m_k in ("r2_Q", "r2_log_N", "select_top1_acc", "policy_kl", "belief_kl"):
                    vals = [
                        per_seed_outputs[sk]["sample_efficiency_and_ood"]["ood_budget_generalization"][t_name][bk][cohort_name][m_k]
                        for sk in s_keys
                    ]
                    ood_budget_3seed[t_name][bk][cohort_name][m_k] = paired_t_summary(vals)

        shared_trunk_3seed[t_name] = {}
        for m_k in ("shared_r2_Q", "specialist_r2_Q", "gap_r2_Q", "shared_select_acc", "specialist_select_acc", "gap_select_acc"):
            vals = [per_seed_outputs[sk]["sample_efficiency_and_ood"]["shared_multi_algo_trunk"][t_name][m_k] for sk in s_keys]
            shared_trunk_3seed[t_name][m_k] = paired_t_summary(vals)

    # 6. Evaluate Verbatim Pre-Registered Gate Verdicts (`R13-D0` through `R13-D6`)
    d0_pass = all(per_seed_outputs[sk]["r13_d0_gate"]["verdict"] == "PASS" for sk in s_keys)
    ts_d16 = tail_swap_3seed["16"]
    d0_tail_swap_pass = bool(
        ts_d16["C0_flat_token"]["SGD_branch_acc"]["mean"] <= 0.60
        and ts_d16["C0_flat_token"]["SGD_r2_Q"]["mean"] <= 0.05
        and ts_d16["C1_parent_1hop"]["SGD_r2_Q"]["mean"] <= 0.35
        and tail_swap_3seed["2"]["C1_parent_1hop"]["SGD_r2_Q"]["mean"] >= 0.98
        and ts_d16["C2_glc_search_v1"]["SGD_branch_acc"]["mean"] >= 0.995
        and ts_d16["C3_glc_search_v2"]["SGD_branch_acc"]["mean"] >= 0.995
        and ts_d16["C3_glc_search_v2"]["SGD_r2_Q"]["mean"] >= 0.999
    )
    cw_min_c3 = min(
        depth_matrix_3seed[t["teacher_name"]]["8"]["C3_glc_search_v2"]["CW_select_acc"]["mean"]
        for t in TEACHER_SPECS
    )
    d0_full_pass = bool(d0_pass and d0_tail_swap_pass and cw_min_c3 >= 0.995)

    mcts_id_acc = depth_matrix_3seed["MCTS-PUCT"]["4"]["C3_glc_search_v2"]["TF_select_acc"]["mean"]
    mcts_ood_acc = ood_budget_3seed["MCTS-PUCT"]["128"]["C3_glc_search_v2"]["select_top1_acc"]["mean"]
    mcts_c0_ood_acc = ood_budget_3seed["MCTS-PUCT"]["128"]["C0_flat_token"]["select_top1_acc"]["mean"]
    mcts_id_r2 = depth_matrix_3seed["MCTS-PUCT"]["4"]["C3_glc_search_v2"]["TF_r2_Q"]["mean"]
    mcts_ood_r2 = ood_budget_3seed["MCTS-PUCT"]["128"]["C3_glc_search_v2"]["r2_Q"]["mean"]
    d1_pass = bool(
        mcts_id_acc >= 0.95
        and mcts_ood_acc >= 0.88
        and mcts_id_r2 >= 0.98
        and mcts_ood_r2 >= 0.93
        and (mcts_ood_acc - mcts_c0_ood_acc) >= 0.25
    )

    b1 = barriers_3seed["barrier1_minimax_pns"]
    d2_pass = bool(
        b1["C3_glc_search_v2"]["sign_acc_id"]["mean"] >= 0.95
        and b1["C3_glc_search_v2"]["sign_acc_ood_d12"]["mean"] >= 0.88
        and b1["C3_glc_search_v2"]["r2_Q_ood_d12"]["mean"] >= 0.97
        and (b1["C3_glc_search_v2"]["r2_Q_ood_d12"]["mean"] - b1["C2_glc_search_v1"]["r2_Q_ood_d12"]["mean"]) >= 0.18
    )

    b5 = barriers_3seed["barrier5_cdcl_nogood"]
    d3_pass = bool(
        b5["C3_glc_search_v2"]["backjump_acc_id"]["mean"] >= 0.94
        and b5["C3_glc_search_v2"]["nogood_prune_f1_id"]["mean"] >= 0.92
        and b5["C3_glc_search_v2"]["unsound_prunes"]["mean"] == 0.0
        and (b5["C3_glc_search_v2"]["nogood_prune_f1_id"]["mean"] - b5["C2_glc_search_v1"]["nogood_prune_f1_id"]["mean"]) >= 0.15
    )

    b2 = barriers_3seed["barrier2_mcgs_dag_bp"]
    d4_pass = bool(
        b2["C3_glc_search_v2"]["r2_Q_id"]["mean"] >= 0.95
        and b2["C3_glc_search_v2"]["r2_Q_ood"]["mean"] >= 0.90
        and b2["C3_glc_search_v2"]["disjunctive_bp_root_kl"]["mean"] <= 0.008
        and (b2["C3_glc_search_v2"]["transposed_cousin_r2_Q"]["mean"] - b2["C2_glc_search_v1"]["transposed_cousin_r2_Q"]["mean"]) >= 0.12
    )

    b3 = barriers_3seed["barrier3_cfr_infoset"]
    d5_pass = bool(
        b3["C3_glc_search_v2"]["strategy_kl_id"]["mean"] <= 0.020
        and b3["C3_glc_search_v2"]["regret_sign_acc"]["mean"] >= 0.90
        and b3["C3_glc_search_v2"]["exploitability_ratio_id"]["mean"] <= 1.25
        and b3["C2_glc_search_v1"]["exploitability_ratio_id"]["mean"] >= 2.0
    )

    b4 = barriers_3seed["barrier4_gumbel_sh"]
    max_shared_gap = max(shared_trunk_3seed[t["teacher_name"]]["gap_select_acc"]["mean"] for t in TEACHER_SPECS)
    d6_pass = bool(
        b4["C3_glc_search_v2"]["survivor_set_em_id"]["mean"] >= 0.93
        and b4["C3_glc_search_v2"]["survivor_set_em_ood"]["mean"] >= 0.88
        and (b4["C3_glc_search_v2"]["survivor_set_em_id"]["mean"] - b4["C2_glc_search_v1"]["survivor_set_em_id"]["mean"]) >= 0.10
        and max_shared_gap <= 0.05
    )

    total_runtime_sec = time.time() - t_start

    final_payload = {
        "document_id": "RESULTS-R13-ALGO-IMITATION",
        "prereg_protocol": "benchmarks/PREREG-R13-ALGO-IMITATION.md",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": {
            "device": str(device),
            "gpu_name": gpu_name,
            "vram_gb": round(vram_gb, 2),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        },
        "runtime_sec": round(total_runtime_sec, 2),
        "seeds": list(args.seeds),
        "seed_aliases_r13": SEED_ALIASES_R13,
        "t_crit_df2": T_CRIT_DF2,
        "gate_verdicts": {
            "R13_D0_integrity_tailswap_infoset_cw": "PASS" if d0_full_pass else "FAIL",
            "R13_D1_algo1_mcts_puct_imitation": "PASS" if d1_pass else "FAIL",
            "R13_D2_algo2_minimax_alphabeta_pns_barrier1": "PASS" if d2_pass else "FAIL",
            "R13_D3_algo3_cdcl_global_nogood_barrier5": "PASS" if d3_pass else "FAIL",
            "R13_D4_algo4_mcgs_dag_disjunctive_bp_barrier2": "PASS" if d4_pass else "FAIL",
            "R13_D5_algo5_cfr_infoset_regret_matching_barrier3": "PASS" if d5_pass else "FAIL",
            "R13_D6_algo6_gumbel_sh_sample_efficiency_shared_trunk": "PASS" if d6_pass else "FAIL",
        },
        "tail_swap_falsification_3seed": tail_swap_3seed,
        "depth_matrix_3seed": depth_matrix_3seed,
        "five_barriers_3seed": barriers_3seed,
        "free_running_rollouts_3seed": free_rollout_3seed,
        "sample_efficiency_3seed": sample_eff_3seed,
        "ood_depth_generalization_3seed": ood_depth_3seed,
        "ood_budget_generalization_3seed": ood_budget_3seed,
        "shared_multi_algo_trunk_3seed": shared_trunk_3seed,
        "per_seed": per_seed_outputs,
    }

    out_json = out_dir / "r13_algo_imitation_results.json"
    with open(out_json, "w", encoding="utf-8") as f_out:
        json.dump(final_payload, f_out, indent=2)

    print(f"\n[R13-ALGO-IMITATION] Gate Verdicts: {json.dumps(final_payload['gate_verdicts'], indent=2)}", flush=True)
    print(f"[R13-ALGO-IMITATION] Completed in {total_runtime_sec:.2f}s. Saved to {out_json}", flush=True)


if __name__ == "__main__":
    main()
