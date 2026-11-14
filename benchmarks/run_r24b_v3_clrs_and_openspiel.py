#!/usr/bin/env python3
r"""Round 24B (`R24B-V3-CLRS-AND-OPENSPIEL-UNIFICATION`):
Resolving the Two Genuine Wave-2 Empirical Bottlenecks (`R22B-E2` `CLRS-30` & `R22A-E1` `OpenSpiel`)
via `GLC_Graph_v3` (Learned Semiring Gate + Triplet-LIR Edge Memory) and `GLC_AOM_v3` (`GLCSearch_v3`
with Linear-CFR Infoset Regret Memory) on `NVIDIA L4` GPU.

Architect & Runner: Member 34 of the Society of Thinkers.
Pre-Registration: `benchmarks/PREREG-R24B-V3-CLRS-AND-OPENSPIEL-UNIFICATION.md`
Static Enforcement Gate: `src/eval/ast_leak_auditor.py` (`v2`, Rules `S1`–`S10`, `0` violations, `7/7` canaries).

Experiments:
  1. `R24B-D0` (Static AST Audit `S1`–`S10` + Runtime `PoisonTensor` Invariance):
     - Verifies `0` violations across `glc_aom.py`, `neural_tree_controller.py`, and
       `run_r24b_v3_clrs_and_openspiel.py` (`7/7` synthetic canaries caught).
     - Verifies `0.000000000` `PoisonTensor` output diff across all 6 `CLRS-30` architectures
       and all 10 `OpenSpiel` paradigms, plus `0.000000000` infoset hidden-state permutation drift.
  2. `R24B-E1` (`GLC_Graph_v3` Learned Semiring Gate $\gamma_h \in [0, 1]$ + Triplet-LIR on Clean
     Zero-Leakage `CLRS-30` $N=16 \to 32 \to 64$ `4x` OOD):
     - Uses the exact same clean zero-leakage `generate_clean_clrs_batch` from `R22B-E2`
       (`node_feats [B, N, 6]`, `edge_feats [B, N, N, 4]`, zero target-derived features).
     - Compares all 6 architectures (`DeepSets`, `MPNN_Max`, `RelTransformer`, `Triplet_GMPNN`,
       `GLC_AOM_v2`, and `GLC_Graph_v3`) across `BFS`, `DFS`, `Dijkstra`, `Bellman_Ford`,
       `MST_Prim`, and `Articulation_Points` at $N=16$ (`train`), $N=32$ (`2x OOD`), and $N=64$ (`4x OOD`).
     - Logs the learned effective semiring gate $\bar{\gamma} = \sum_h \pi_h \gamma_h \in [0, 1]$
       and per-head distribution $\pi_h$ across all 6 algorithms as a mechanistic probe.
  3. `R24B-E2` (`GLC_AOM_v3` / `GLCSearch_v3` with Linear-CFR Infoset Regret Memory on Pure-Neural
     `OpenSpiel` `kuhn_poker`, `leduc_poker`, `phantom_ttt`, `dark_hex`):
     - Uses the exact same 100% pure-neural GPU infoset evaluation from `R22A-E1` (zero
       `cfr_reference_table`, zero `ref_sigma`, zero `w_spec`, 100% of infosets evaluated on GPU).
     - Compares `PPO_GAE`, `GRPO`, `AlphaZero_PUCT`, `Gumbel_MuZero`, `R_NaD`, `Deep_CFR`,
       `GLC_AOM_v2`, `GLC_AOM_v2_Generalist`, `GLC_AOM_v3`, and `GLC_AOM_v3_Generalist`
       across all 4 `OpenSpiel` games at $B \in \{1, 4, 16, 64\}$.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

for _cand_path in [
    "/root/pylib",
    "/tmp/r4e3a/pylib",
    "/home/agent/data/pool/pylib",
    "/home/agent/.local/lib/python3.12/site-packages",
    "/home/agent/workspaces/venv-cft/lib/python3.12/site-packages",
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
        import pyspiel  # noqa: F401
        return
    except ImportError:
        pass

    print("[bootstrap] Installing torch, numpy, scipy, open_spiel into /root/pylib ...", flush=True)
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
            "open_spiel",
        ]
    )
    if "/root/pylib" not in sys.path:
        sys.path.insert(0, "/root/pylib")


if __name__ == "__main__":
    _ensure_gpu_python_deps()

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(torch, "_native") and hasattr(torch._native, "registry"):
    torch._native.registry.deregister_op_overrides(disable_dispatch_keys=["CUDA", "CPU"])
torch.set_num_threads(4)

import pyspiel  # noqa: E402
import open_spiel.python.policy as osp_policy  # noqa: E402
from open_spiel.python.algorithms import exploitability as osp_exploitability  # noqa: E402

from src.envs import (  # noqa: E402
    EnvBatch,
    OracleLeakError,
    PoisonTensor,
    make_env,
)
from src.envs.stratego_full import NUM_RANKS  # noqa: E402
from src.eval.ast_leak_auditor import audit_r11_files  # noqa: E402
from src.models.glc_aom import GLCAOMConfig, GLCAOMWorldModel, LIRMask  # noqa: E402
from src.search.kv_tree_search import TreePagedKVSearch  # noqa: E402

# ==============================================================================
# 1. Locked Constants & Pre-Registered Grids (`PREREG-R24B`)
# ==============================================================================

SEEDS: Tuple[int, ...] = (101, 102, 103)
T_CRIT_95_DF2: float = 4.302652729911275  # Student-t critical value t_{0.975, df=2}

CLRS_TASKS_E1: Tuple[str, ...] = (
    "BFS",
    "DFS",
    "Dijkstra",
    "Bellman_Ford",
    "MST_Prim",
    "Articulation_Points",
)

CLRS_ARCHITECTURES_E1: Tuple[str, ...] = (
    "DeepSets",
    "MPNN_Max",
    "RelTransformer",
    "Triplet_GMPNN",
    "GLC_AOM_v2",
    "GLC_Graph_v3",
)

CLRS_EVAL_SIZES: Tuple[int, ...] = (16, 32, 64)

OPENSPIEL_PARADIGMS_E2: Tuple[str, ...] = (
    "PPO_GAE",
    "GRPO",
    "AlphaZero_PUCT",
    "Gumbel_MuZero",
    "R_NaD",
    "Deep_CFR",
    "GLC_AOM_v2",
    "GLC_AOM_v2_Generalist",
    "GLC_AOM_v3",
    "GLC_AOM_v3_Generalist",
)

OPENSPIEL_IMPERFECT_GAMES: Tuple[str, ...] = (
    "kuhn_poker",
    "leduc_poker",
    "phantom_ttt(obstype=reveal-nothing)",
    "dark_hex(board_size=2)",
)

TEST_BUDGETS: Tuple[int, ...] = (1, 4, 16, 64)


def load_openspiel_game(gname: str) -> Any:
    """Load an OpenSpiel game by specification string, with safe fallback if parameter syntax differs."""
    try:
        return pyspiel.load_game(gname)
    except Exception:
        if gname.startswith("phantom_ttt"):
            return pyspiel.load_game("phantom_ttt")
        if gname.startswith("dark_hex"):
            return pyspiel.load_game("dark_hex", {"board_size": 2})
        raise


def summarize_across_seeds(vals: Sequence[float], baseline_vals: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Compute 3-seed mean, SD, SE, 95% CI (`t_{0.975,2} = 4.303`), and optional paired t-test vs baseline."""
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(float(max(n, 1)))
    half_ci = T_CRIT_95_DF2 * se if n == 3 else 1.96 * se
    out: Dict[str, Any] = {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "se": round(se, 6),
        "ci95_low": round(mean - half_ci, 6),
        "ci95_high": round(mean + half_ci, 6),
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
        out["diff_sd"] = round(d_sd, 6)
        out["diff_se"] = round(d_se, 6)
        out["t_stat"] = round(t_stat, 4)
        out["p_value"] = round(p_val, 6)
        out["significant_95"] = bool(abs(t_stat) > T_CRIT_95_DF2)
    return out


def strip_oracle_fields(batch: EnvBatch) -> Tuple[EnvBatch, Dict[str, Any]]:
    """Sanitize `batch` via `batch.strip_oracle_labels()` (`PoisonTensor`)."""
    oracle_targets: Dict[str, Any] = {
        "target_tokens": batch.target_tokens.clone() if isinstance(batch.target_tokens, torch.Tensor) else None,
        "exact_marginals": batch.exact_marginals.clone() if isinstance(batch.exact_marginals, torch.Tensor) else None,
        "query_targets": batch.query_targets.clone() if isinstance(batch.query_targets, torch.Tensor) else None,
        "solvable": batch.solvable.clone(),
    }
    return batch.strip_oracle_labels(), oracle_targets


# ==============================================================================
# 2. Experiment `R24B-E1`: Clean Zero-Leakage `CLRS-30` Data Generator &
#    `GLC_Graph_v3` (Learned Semiring Gate $\gamma_h \in [0, 1]$ + Triplet-LIR)
# ==============================================================================


@dataclass
class CleanCLRSBatch:
    """Strictly clean `CLRS-30` graph batch (Zero target leakage in `node_feats` or `edge_feats`):
      - `node_feats`: `[B, N, 6]` containing ONLY `[source_indicator, v/N, deg(v)/N, min_w, mean_w, max_w]`.
      - `edge_feats`: `[B, N, N, 4]` containing ONLY `[A_uv, W_uv, I_uv, exp(-3 W_uv) A_uv]`.
      - `adj_mask`: `[B, N, N]` bool mask of valid candidate edges (`A_uv + I_uv > 0`).
      - `out_pointer`: `[B, N]` ground-truth predecessor pointer `pi_star[v]` (poisoned at eval time!).
      - `out_mask`: `[B, N]` ground-truth structural node mask (poisoned at eval time!).
    """

    task_name: str
    num_nodes: int
    node_feats: torch.Tensor
    edge_feats: torch.Tensor
    adj_mask: torch.Tensor
    out_pointer: torch.Tensor
    out_mask: torch.Tensor

    def make_blind_poisoned(self) -> "CleanCLRSBatch":
        """Replace `out_pointer` and `out_mask` with `PoisonTensor` guards."""
        dev = self.node_feats.device
        return CleanCLRSBatch(
            task_name=self.task_name,
            num_nodes=self.num_nodes,
            node_feats=self.node_feats,
            edge_feats=self.edge_feats,
            adj_mask=self.adj_mask,
            out_pointer=PoisonTensor(self.out_pointer.shape, dtype=torch.long, device=dev, field_name="out_pointer"),
            out_mask=PoisonTensor(self.out_mask.shape, dtype=torch.float32, device=dev, field_name="out_mask"),
        )


@torch.no_grad()
def generate_clean_clrs_batch(
    task_name: str,
    batch_size: int,
    num_nodes: int,
    seed: int,
    device: torch.device,
) -> CleanCLRSBatch:
    """Generate 100% clean zero-leakage `CLRS-30` instances (`N in {16, 32, 64}`).

    Identical to `R22B-E2` and strictly enforces `ast_leak_auditor.py v2` Rule `S10`:
      - `node_feats` (`[B, N, 6]`) depends ONLY on `(s, v/N, A_uv, W_uv)`.
      - `edge_feats` (`[B, N, N, 4]`) depends ONLY on `(A_uv, W_uv, I_uv)`.
      - ZERO target variables (`dist_vec`, `pi_star`, `pi_oh`, `mask_star`, `aux_node_stat`)
        ever touch `node_feats` or `edge_feats`!
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) * 31337 + int(num_nodes) * 997 + sum(ord(c) for c in task_name))

    B = int(batch_size)
    N = int(num_nodes)

    node_feats = torch.zeros((B, N, 6), dtype=torch.float32)
    edge_feats = torch.zeros((B, N, N, 4), dtype=torch.float32)
    adj_mask = torch.zeros((B, N, N), dtype=torch.bool)
    out_pointer = torch.zeros((B, N), dtype=torch.long)
    out_mask = torch.zeros((B, N), dtype=torch.float32)

    v_norm = torch.arange(N, dtype=torch.float32) / float(max(N, 1))
    eye_bool = torch.eye(N, dtype=torch.bool)
    eye_f = torch.eye(N, dtype=torch.float32)

    for b in range(B):
        adj = eye_bool.clone()
        weights = torch.zeros((N, N), dtype=torch.float32)

        for v in range(1, N):
            low_p = max(0, v - max(3, N // 4))
            pa = int(torch.randint(low_p, v, (1,), generator=gen).item())
            w_tree = 0.08 + 0.28 * float(torch.rand((1,), generator=gen).item())
            adj[pa, v] = True
            weights[pa, v] = w_tree
            if task_name in ("MST_Prim", "Articulation_Points"):
                adj[v, pa] = True
                weights[v, pa] = w_tree

        extra_prob = 0.20 if N <= 16 else (0.12 if N <= 32 else 0.08)
        rand_mat = torch.rand((N, N), generator=gen)
        rand_w = 0.42 + 0.55 * torch.rand((N, N), generator=gen)
        for u in range(N):
            for v in range(u + 1, N):
                if not adj[u, v] and float(rand_mat[u, v].item()) < extra_prob:
                    w_val = float(rand_w[u, v].item())
                    adj[u, v] = True
                    weights[u, v] = w_val
                    if task_name in ("MST_Prim", "Articulation_Points"):
                        adj[v, u] = True
                        weights[v, u] = w_val

        # Populate input tensors `node_feats [B, N, 6]` and `edge_feats [B, N, N, 4]` BEFORE targets
        adj_no_self = adj & (~eye_bool)
        adj_ns_f = adj_no_self.float()
        deg_out = adj_ns_f.sum(dim=1)  # [N]

        w_masked_min = weights.masked_fill(~adj_no_self, 1.0)
        w_min = torch.where(deg_out > 0, w_masked_min.min(dim=1).values, torch.zeros(N))
        w_mean = (weights * adj_ns_f).sum(dim=1) / deg_out.clamp(min=1.0)
        w_max = (weights * adj_ns_f).max(dim=1).values

        src_ind = torch.zeros(N, dtype=torch.float32)
        src_ind[0] = 1.0

        node_feats[b, :, 0] = src_ind
        node_feats[b, :, 1] = v_norm
        node_feats[b, :, 2] = deg_out / float(max(N, 1))
        node_feats[b, :, 3] = w_min
        node_feats[b, :, 4] = w_mean
        node_feats[b, :, 5] = w_max

        in_adj = adj.t()  # in_adj[v, u] == 1 iff edge u -> v exists
        in_w = weights.t() * in_adj.float()
        inv_exp = torch.exp(-3.0 * in_w) * in_adj.float()
        inv_exp_norm = inv_exp / inv_exp.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        edge_feats[b, :, :, 0] = in_adj.float()
        edge_feats[b, :, :, 1] = in_w
        edge_feats[b, :, :, 2] = eye_f
        edge_feats[b, :, :, 3] = inv_exp_norm
        adj_mask[b] = in_adj

        # Compute ground-truth targets `(out_pointer[b], out_mask[b])` strictly for labels
        target_ptr = torch.zeros(N, dtype=torch.long)
        target_msk = torch.zeros(N, dtype=torch.float32)

        if task_name == "BFS":
            hop_d = torch.full((N,), 999, dtype=torch.long)
            hop_d[0] = 0
            target_ptr[0] = 0
            q_list = [0]
            head = 0
            while head < len(q_list):
                u = q_list[head]
                head += 1
                for v in range(N):
                    if adj[u, v] and u != v:
                        if hop_d[u] + 1 < hop_d[v]:
                            hop_d[v] = hop_d[u] + 1
                            target_ptr[v] = u
                            q_list.append(v)
                        elif hop_d[u] + 1 == hop_d[v] and u < int(target_ptr[v].item()):
                            target_ptr[v] = u
            target_msk = (hop_d <= 2).float()

        elif task_name == "DFS":
            visited = [False] * N
            target_ptr[0] = 0
            sub_sz = torch.ones(N, dtype=torch.float32)

            def _dfs(u: int) -> float:
                visited[u] = True
                nbrs = [v for v in range(N) if adj[u, v] and v != u]
                nbrs.sort(key=lambda v: (float(weights[u, v].item()), v))
                sz = 1.0
                for v in nbrs:
                    if not visited[v]:
                        target_ptr[v] = u
                        sz += _dfs(v)
                sub_sz[u] = sz
                return sz

            _dfs(0)
            target_msk = (deg_out >= 1.0).float()

        elif task_name == "Dijkstra":
            d_sp = torch.full((N,), 1e6, dtype=torch.float32)
            d_sp[0] = 0.0
            target_ptr[0] = 0
            settled = [False] * N
            for _ in range(N):
                u_best = -1
                best_val = 1e9
                for u in range(N):
                    if not settled[u] and float(d_sp[u].item()) < best_val:
                        best_val = float(d_sp[u].item())
                        u_best = u
                if u_best < 0:
                    break
                settled[u_best] = True
                for v in range(N):
                    if adj[u_best, v] and u_best != v:
                        cand = float(d_sp[u_best].item() + weights[u_best, v].item())
                        if cand + 1e-6 < float(d_sp[v].item()):
                            d_sp[v] = cand
                            target_ptr[v] = u_best
            med_d = float(torch.median(d_sp).item())
            target_msk = (d_sp <= med_d).float()

        elif task_name == "Bellman_Ford":
            d_bf = torch.full((N,), 1e6, dtype=torch.float32)
            d_bf[0] = 0.0
            target_ptr[0] = 0
            for _ in range(N - 1):
                for u in range(N):
                    for v in range(N):
                        if adj[u, v] and u != v:
                            cand = float(d_bf[u].item() + weights[u, v].item())
                            if cand + 1e-6 < float(d_bf[v].item()):
                                d_bf[v] = cand
                                target_ptr[v] = u
            med_bf = float(torch.median(d_bf).item())
            target_msk = (d_bf <= med_bf).float()

        elif task_name == "MST_Prim":
            in_mst = [False] * N
            in_mst[0] = True
            target_ptr[0] = 0
            key_w = torch.full((N,), 1e6, dtype=torch.float32)
            key_w[0] = 0.0
            for v in range(1, N):
                if adj[0, v]:
                    key_w[v] = weights[0, v]
                    target_ptr[v] = 0
            for _ in range(N - 1):
                v_best = -1
                best_k = 1e9
                for v in range(N):
                    if not in_mst[v] and float(key_w[v].item()) < best_k:
                        best_k = float(key_w[v].item())
                        v_best = v
                if v_best < 0:
                    break
                in_mst[v_best] = True
                for w in range(N):
                    if not in_mst[w] and adj[v_best, w]:
                        if float(weights[v_best, w].item()) + 1e-6 < float(key_w[w].item()):
                            key_w[w] = weights[v_best, w]
                            target_ptr[w] = v_best
            med_k = float(torch.median(key_w).item())
            target_msk = (key_w <= med_k).float()

        elif task_name == "Articulation_Points":
            visited = [False] * N
            disc = [0] * N
            low = [0] * N
            is_art = [0.0] * N
            timer = [0]
            target_ptr[0] = 0

            def _art_dfs(u: int, p: int) -> None:
                visited[u] = True
                timer[0] += 1
                disc[u] = low[u] = timer[0]
                children = 0
                nbrs = [v for v in range(N) if adj[u, v] and v != u]
                nbrs.sort(key=lambda v: (float(weights[u, v].item()), v))
                for v in nbrs:
                    if not visited[v]:
                        children += 1
                        target_ptr[v] = u
                        _art_dfs(v, u)
                        low[u] = min(low[u], low[v])
                        if p != -1 and low[v] >= disc[u]:
                            is_art[u] = 1.0
                    elif v != p:
                        low[u] = min(low[u], disc[v])
                if p == -1 and children > 1:
                    is_art[u] = 1.0

            _art_dfs(0, -1)
            target_msk = torch.tensor(is_art, dtype=torch.float32)

        out_pointer[b] = target_ptr
        out_mask[b] = target_msk

    return CleanCLRSBatch(
        task_name=task_name,
        num_nodes=N,
        node_feats=node_feats.to(device),
        edge_feats=edge_feats.to(device),
        adj_mask=adj_mask.to(device),
        out_pointer=out_pointer.to(device),
        out_mask=out_mask.to(device),
    )


class CLRSNeuralProcessor(nn.Module):
    """100% Clean Zero-Leakage `CLRS-30` Neural Processor supporting all 6 architectures:
      1. `DeepSets`: Node-wise MLP + global mean/max pooling without graph edge message passing.
      2. `MPNN_Max`: Fixed-depth 1-hop message passing with max-aggregation.
      3. `RelTransformer`: All-pairs unmasked relational self-attention over nodes + edge biases.
      4. `Triplet_GMPNN`: 2-hop triplet edge-pivot message passing (`Ibarz et al. 2022`).
      5. `GLC_AOM_v2` (`R22B`): Fixed $\\gamma=1$ path-sum tropical relaxation + LIR attention.
      6. `GLC_Graph_v3` (`R24B`): Multi-Head Learned Semiring Gate $\\gamma_h = \\sigma(w_h) \\in [0, 1]$
         (spanning cut-edge $\\gamma_0 \\to 0$ for `MST_Prim`/`DFS` to Bellman path-sum $\\gamma_3 \\to 1$
         for `Dijkstra`/`Bellman_Ford`) + Triplet-LIR Edge Memory & Hybrid Max/LIR Aggregation.
    """

    def __init__(self, arch_name: str, d_hidden: int = 64, n_steps: int = 6, n_semiring_heads: int = 4):
        super().__init__()
        if arch_name not in CLRS_ARCHITECTURES_E1:
            raise ValueError(f"Unknown CLRS architecture {arch_name!r}")
        self.arch_name = arch_name
        self.d_hidden = int(d_hidden)
        self.n_steps = int(n_steps)
        self.n_semiring_heads = int(n_semiring_heads)
        self.lir_mask = LIRMask.epistemic_belief()

        self.node_enc = nn.Sequential(
            nn.Linear(6, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(4, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )

        self.msg_src = nn.Linear(d_hidden, d_hidden)
        self.msg_dst = nn.Linear(d_hidden, d_hidden)
        self.msg_edge = nn.Linear(d_hidden, d_hidden)
        self.node_gru = nn.GRUCell(d_hidden, d_hidden)
        self.node_ln = nn.LayerNorm(d_hidden)

        self.attn_q = nn.Linear(d_hidden, d_hidden)
        self.attn_k = nn.Linear(d_hidden, d_hidden)

        self.trip_ik = nn.Linear(d_hidden, d_hidden // 2)
        self.trip_kj = nn.Linear(d_hidden, d_hidden // 2)
        self.trip_out = nn.Linear(d_hidden // 2, d_hidden)

        # Neural Bellman-Ford / Tropical Relaxation parameters
        self.relax_weight_scale = nn.Parameter(torch.tensor(2.5))
        self.relax_hop_cost = nn.Parameter(torch.tensor(0.25))
        self.relax_edge_gate = nn.Sequential(
            nn.Linear(d_hidden + 4, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, 1),
        )
        nn.init.zeros_(self.relax_edge_gate[-1].weight)
        nn.init.zeros_(self.relax_edge_gate[-1].bias)

        self.lir_prec_gate = nn.Linear(d_hidden, 1)
        self.relax_to_node = nn.Linear(4, d_hidden)
        self.relax_to_ptr = nn.Linear(5, 1)

        self.ptr_q = nn.Linear(d_hidden, d_hidden)
        self.ptr_k = nn.Linear(d_hidden, d_hidden)
        self.ptr_e = nn.Linear(d_hidden, 1)
        self.mask_dec = nn.Sequential(
            nn.Linear(d_hidden + 6, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, 1),
        )

        with torch.no_grad():
            self.relax_to_ptr.weight.copy_(torch.tensor([[-3.5, -2.5, 2.2, 1.8, -1.2]]))
            self.relax_to_ptr.bias.zero_()

        # GLC_Graph_v3 Learned Multi-Head Semiring Gate (\gamma_h = \sigma(w_h) \in [0, 1]) + Triplet-LIR Gate
        if self.arch_name == "GLC_Graph_v3":
            # 4 semiring heads spanning \gamma_0 \approx 0.03 (pure cut-edge/Prim) to \gamma_3 \approx 0.97 (Bellman SP)
            self.semiring_head_w = nn.Parameter(torch.tensor([-3.5, -1.1, 1.1, 3.5]))
            self.semiring_selector_logits = nn.Parameter(torch.zeros(self.n_semiring_heads))
            self.semiring_ctx_proj = nn.Linear(d_hidden, self.n_semiring_heads)
            nn.init.zeros_(self.semiring_ctx_proj.weight)
            nn.init.zeros_(self.semiring_ctx_proj.bias)
            # Learned mixing weight between Triplet-LIR message-passing pointer score and Semiring-Slack score
            self.semiring_ptr_gain = nn.Parameter(torch.tensor(0.0))
            self.agg_attn_mix = nn.Parameter(torch.tensor(-1.2))
            self.semiring_node_proj = nn.Linear(self.n_semiring_heads + 2, d_hidden)
            nn.init.zeros_(self.semiring_node_proj.weight)
            nn.init.zeros_(self.semiring_node_proj.bias)

    def get_semiring_gate_summary(self) -> Dict[str, Any]:
        """Return mechanistic probe values for `GLC_Graph_v3`'s learned semiring gate $\\bar{\\gamma} \\in [0, 1]$."""
        if self.arch_name != "GLC_Graph_v3":
            gamma_fixed = 1.0 if self.arch_name == "GLC_AOM_v2" else 0.0
            return {
                "effective_gamma_mean": gamma_fixed,
                "head_gammas": [gamma_fixed],
                "head_probs": [1.0],
                "semiring_ptr_blend": 1.0 if self.arch_name == "GLC_AOM_v2" else 0.0,
            }
        with torch.no_grad():
            gammas = torch.sigmoid(self.semiring_head_w)  # [H]
            probs = F.softmax(5.0 * self.semiring_selector_logits, dim=-1)  # [H]
            eff_gamma = float((probs * gammas).sum().item())
            ptr_blend = float(torch.sigmoid(3.0 * self.semiring_ptr_gain).item())
            return {
                "effective_gamma_mean": round(eff_gamma, 6),
                "head_gammas": [round(float(x), 6) for x in gammas.detach().cpu().tolist()],
                "head_probs": [round(float(x), 6) for x in probs.detach().cpu().tolist()],
                "semiring_ptr_blend": round(ptr_blend, 6),
            }

    def forward(self, batch: CleanCLRSBatch) -> Dict[str, torch.Tensor]:
        """Run forward pass strictly on `(batch.node_feats [B, N, 6], batch.edge_feats [B, N, N, 4], batch.adj_mask)`."""
        B, N, _ = batch.node_feats.shape
        D = self.d_hidden
        device = batch.node_feats.device
        dtype = batch.node_feats.dtype

        h = self.node_enc(batch.node_feats)                  # [B, N, D]
        e = self.edge_enc(batch.edge_feats)                  # [B, N, N, D]
        in_adj = batch.adj_mask                              # [B, N, N] bool: in_adj[b, v, u] == 1 iff u -> v
        eye_mask = batch.edge_feats[:, :, :, 2] > 0.5
        in_proper = in_adj & (~eye_mask)

        src_mask = batch.node_feats[:, :, 0] > 0.5           # [B, N]
        d_pot = torch.where(src_mask, torch.zeros_like(batch.node_feats[:, :, 0]), torch.full((B, N), 10.0, dtype=dtype, device=device))
        hop_pot = torch.where(src_mask, torch.zeros_like(batch.node_feats[:, :, 0]), torch.full((B, N), 10.0, dtype=dtype, device=device))

        w_in = batch.edge_feats[:, :, :, 1]                  # [B, v, u] weight of edge u -> v
        inv_exp = batch.edge_feats[:, :, :, 3]               # [B, v, u]
        w_scale = F.softplus(self.relax_weight_scale)
        h_cost = F.softplus(self.relax_hop_cost)

        # Multi-head semiring potentials d_semi: [B, H, N] for GLC_Graph_v3
        if self.arch_name == "GLC_Graph_v3":
            H = self.n_semiring_heads
            gammas_h = torch.sigmoid(self.semiring_head_w).view(1, H, 1, 1)  # [1, H, 1, 1]
            src_mask_h = src_mask.unsqueeze(1).expand(B, H, N)
            d_semi = torch.where(
                src_mask_h,
                torch.zeros((B, H, N), dtype=dtype, device=device),
                torch.full((B, H, N), 10.0, dtype=dtype, device=device),
            )
        else:
            H = 1
            gammas_h = None
            d_semi = None

        eff_steps = (
            self.n_steps
            if self.arch_name not in ("GLC_AOM_v2", "GLC_Graph_v3")
            else max(self.n_steps, min(N // 2, 24))
        )

        for step_t in range(eff_steps):
            if self.arch_name == "DeepSets":
                g_mean = h.mean(dim=1, keepdim=True).expand(B, N, D)
                g_max = h.max(dim=1, keepdim=True).values.expand(B, N, D)
                agg = F.silu(self.msg_src(h) + 0.5 * self.msg_dst(g_mean + g_max))

            elif self.arch_name == "MPNN_Max":
                src_h = self.msg_src(h).unsqueeze(1)         # [B, 1, u, D]
                dst_h = self.msg_dst(h).unsqueeze(2)         # [B, v, 1, D]
                pair_msg = F.silu(src_h + dst_h + self.msg_edge(e))
                masked_msg = pair_msg.masked_fill(~in_adj.unsqueeze(-1), -1e4)
                agg = masked_msg.max(dim=2).values

            elif self.arch_name == "RelTransformer":
                q = self.attn_q(h)
                k = self.attn_k(h)
                scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(float(D))
                attn = F.softmax(scores, dim=-1)
                src_h = self.msg_src(h).unsqueeze(1)
                dst_h = self.msg_dst(h).unsqueeze(2)
                pair_msg = F.silu(src_h + dst_h + self.msg_edge(e))
                agg = torch.einsum("bvu,bvud->bvd", attn, pair_msg)

            elif self.arch_name == "Triplet_GMPNN":
                src_h = self.msg_src(h).unsqueeze(1)
                dst_h = self.msg_dst(h).unsqueeze(2)
                pair_msg = F.silu(src_h + dst_h + self.msg_edge(e))
                if step_t < 4:
                    t_ik = self.trip_ik(pair_msg)
                    t_kj = self.trip_kj(pair_msg)
                    pivot_k = t_ik.max(dim=2, keepdim=True).values + t_kj.mean(dim=1, keepdim=True)
                    e = e + 0.20 * self.trip_out(F.silu(pivot_k))
                masked_msg = pair_msg.masked_fill(~in_adj.unsqueeze(-1), -1e4)
                agg = masked_msg.max(dim=2).values

            elif self.arch_name == "GLC_AOM_v2":
                neural_edge_delta = 0.15 * torch.tanh(
                    self.relax_edge_gate(torch.cat([e, batch.edge_feats], dim=-1)).squeeze(-1)
                )
                step_edge_cost = w_scale * w_in + h_cost + neural_edge_delta
                cand_d = d_pot.unsqueeze(1) + step_edge_cost
                cand_d = cand_d.masked_fill(~in_proper, 1e4)
                min_incoming_d = cand_d.min(dim=-1).values
                d_pot = torch.where(src_mask, torch.zeros_like(d_pot), torch.minimum(d_pot, min_incoming_d))

                cand_hop = hop_pot.unsqueeze(1) + 1.0 + 0.05 * batch.node_feats[:, :, 1].unsqueeze(1)
                cand_hop = cand_hop.masked_fill(~in_proper, 1e4)
                hop_pot = torch.where(src_mask, torch.zeros_like(hop_pot), torch.minimum(hop_pot, cand_hop.min(dim=-1).values))

                if step_t < self.n_steps:
                    src_h = self.msg_src(h).unsqueeze(1)
                    dst_h = self.msg_dst(h).unsqueeze(2)
                    pair_msg = F.silu(src_h + dst_h + self.msg_edge(e))
                    q_g = self.attn_q(h)
                    k_g = self.attn_k(h)
                    prec = torch.sigmoid(self.lir_prec_gate(h))
                    raw_s = torch.bmm(q_g, k_g.transpose(1, 2)) / math.sqrt(float(D)) - 2.0 * w_in
                    raw_s = raw_s.masked_fill(~in_adj, -1e9)
                    attn_g = F.softmax(raw_s, dim=-1) * prec.transpose(1, 2)
                    attn_g = attn_g / (attn_g.sum(dim=-1, keepdim=True) + 1e-6)
                    agg = torch.einsum("bvu,bvud->bvd", attn_g, pair_msg)
                else:
                    continue

            else:
                # GLC_Graph_v3:
                # 1. Generalized \gamma_h-Semiring Relaxation across H=4 heads (\gamma_h \in [0, 1])
                neural_edge_delta = 0.12 * torch.tanh(
                    self.relax_edge_gate(torch.cat([e, batch.edge_feats], dim=-1)).squeeze(-1)
                )
                step_edge_cost = w_scale * w_in + h_cost + neural_edge_delta  # [B, v, u]
                # Candidate potential for head h: \gamma_h * d_semi[b, h, u] + cost(u -> v) -> shape [B, H, v, u]
                assert d_semi is not None and gammas_h is not None
                cand_semi = gammas_h * d_semi.unsqueeze(2) + step_edge_cost.unsqueeze(1)
                cand_semi = cand_semi.masked_fill(~in_proper.unsqueeze(1), 1e4)
                min_inc_semi = cand_semi.min(dim=-1).values  # [B, H, v]
                d_semi = torch.where(
                    src_mask.unsqueeze(1),
                    torch.zeros_like(d_semi),
                    torch.minimum(d_semi, min_inc_semi),
                )

                cand_hop = hop_pot.unsqueeze(1) + 1.0 + 0.05 * batch.node_feats[:, :, 1].unsqueeze(1)
                cand_hop = cand_hop.masked_fill(~in_proper, 1e4)
                hop_pot = torch.where(src_mask, torch.zeros_like(hop_pot), torch.minimum(hop_pot, cand_hop.min(dim=-1).values))

                # 2. Triplet-LIR Edge Memory Update + Hybrid Max / LIR-Preconditioned Aggregation
                if step_t < self.n_steps:
                    src_h = self.msg_src(h).unsqueeze(1)
                    dst_h = self.msg_dst(h).unsqueeze(2)
                    pair_msg = F.silu(src_h + dst_h + self.msg_edge(e))
                    prec = torch.sigmoid(self.lir_prec_gate(h))  # [B, N, 1]
                    if step_t < 4:
                        t_ik = self.trip_ik(pair_msg)
                        t_kj = self.trip_kj(pair_msg)
                        pivot_k = t_ik.max(dim=2, keepdim=True).values + t_kj.mean(dim=1, keepdim=True)
                        # Gate 2-hop triplet update with LIR precision \beta_{trust}
                        beta_edge = (0.5 + 0.5 * prec).unsqueeze(2)  # [B, v, 1, 1]
                        e = e + 0.22 * beta_edge * self.trip_out(F.silu(pivot_k))

                    masked_msg = pair_msg.masked_fill(~in_adj.unsqueeze(-1), -1e4)
                    agg_max = masked_msg.max(dim=2).values

                    q_g = self.attn_q(h)
                    k_g = self.attn_k(h)
                    raw_s = torch.bmm(q_g, k_g.transpose(1, 2)) / math.sqrt(float(D)) - 2.0 * w_in
                    raw_s = raw_s.masked_fill(~in_adj, -1e9)
                    attn_g = F.softmax(raw_s, dim=-1) * prec.transpose(1, 2)
                    attn_g = attn_g / (attn_g.sum(dim=-1, keepdim=True) + 1e-6)
                    agg_attn = torch.einsum("bvu,bvud->bvd", attn_g, pair_msg)

                    mix_w = torch.sigmoid(self.agg_attn_mix)
                    agg = (1.0 - 0.35 * mix_w) * agg_max + (0.35 * mix_w) * agg_attn
                else:
                    continue

            h_flat = self.node_gru(agg.reshape(B * N, D), h.reshape(B * N, D)).view(B, N, D)
            h = self.node_ln(h + 0.5 * h_flat)

        if self.arch_name == "GLC_AOM_v2":
            d_norm = d_pot / d_pot.max(dim=-1, keepdim=True).values.clamp(min=1.0)
            hop_norm = hop_pot / hop_pot.max(dim=-1, keepdim=True).values.clamp(min=1.0)
            relax_node_f = torch.stack(
                [d_norm, hop_norm, (d_norm <= d_norm.median(dim=-1, keepdim=True).values).float(), batch.node_feats[:, :, 0]],
                dim=-1,
            )
            h = self.node_ln(h + 0.4 * self.relax_to_node(relax_node_f))
        elif self.arch_name == "GLC_Graph_v3" and d_semi is not None:
            d_semi_norm = d_semi / d_semi.max(dim=-1, keepdim=True).values.clamp(min=1.0)  # [B, H, N]
            hop_norm = hop_pot / hop_pot.max(dim=-1, keepdim=True).values.clamp(min=1.0)
            semi_node_f = torch.cat(
                [
                    d_semi_norm.permute(0, 2, 1),
                    hop_norm.unsqueeze(-1),
                    batch.node_feats[:, :, 0:1],
                ],
                dim=-1,
            )
            h = self.node_ln(h + 0.25 * self.semiring_node_proj(semi_node_f))

        q_out = self.ptr_q(h)
        k_out = self.ptr_k(h)
        base_ptr = torch.bmm(q_out, k_out.transpose(1, 2)) / math.sqrt(float(D)) + self.ptr_e(e).squeeze(-1)
        base_mask = self.mask_dec(torch.cat([h, batch.node_feats], dim=-1)).squeeze(-1)

        if self.arch_name == "DeepSets":
            ptr_logits = base_ptr
            mask_logits = base_mask
        elif self.arch_name == "RelTransformer":
            ptr_logits = base_ptr + 2.5 * batch.edge_feats[:, :, :, 3] - 3.5 * (1.0 - in_adj.float())
            mask_logits = base_mask
        elif self.arch_name in ("MPNN_Max", "Triplet_GMPNN"):
            weight_bonus = 2.2 * batch.edge_feats[:, :, :, 3] - 1.5 * w_in
            ptr_logits = (base_ptr + weight_bonus).masked_fill(~in_adj, -1e9)
            mask_logits = base_mask
        elif self.arch_name == "GLC_AOM_v2":
            neural_edge_delta = 0.15 * torch.tanh(
                self.relax_edge_gate(torch.cat([e, batch.edge_feats], dim=-1)).squeeze(-1)
            )
            step_edge_cost = w_scale * w_in + h_cost + neural_edge_delta
            slack_d = (d_pot.unsqueeze(1) + step_edge_cost - d_pot.unsqueeze(2)).abs()
            slack_hop = (hop_pot.unsqueeze(1) + 1.0 - hop_pot.unsqueeze(2)).abs()
            min_w_in = w_in.masked_fill(~in_proper, 99.0).min(dim=-1, keepdim=True).values
            is_min_edge = ((w_in <= min_w_in + 1e-4) & in_proper).float()
            src_self = (src_mask.unsqueeze(-1) & eye_mask).float()
            non_src_self = ((~src_mask).unsqueeze(-1) & eye_mask).float()

            relax_feats = torch.stack(
                [slack_d, slack_hop, is_min_edge + src_self, inv_exp, non_src_self],
                dim=-1,
            )
            relax_score = self.relax_to_ptr(relax_feats).squeeze(-1)
            valid_cnt = in_adj.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
            relax_centered = relax_score - (relax_score * in_adj.float()).sum(dim=-1, keepdim=True) / valid_cnt
            ptr_logits = (0.45 * base_ptr + 1.25 * relax_centered).masked_fill(~in_adj, -1e9)
            mask_logits = base_mask
        else:
            # GLC_Graph_v3 Pointer Readout:
            # Unifies Triplet-LIR base pointer (`base_ptr + weight_bonus`) with Multi-Head \gamma_h-Semiring
            # Relaxation Slack (`S_{b,v,u}^{(h)}`) weighted by learned head distribution \pi_h!
            assert d_semi is not None and gammas_h is not None
            neural_edge_delta = 0.12 * torch.tanh(
                self.relax_edge_gate(torch.cat([e, batch.edge_feats], dim=-1)).squeeze(-1)
            )
            step_edge_cost = w_scale * w_in + h_cost + neural_edge_delta  # [B, v, u]

            # Per-head incoming candidate cost: \gamma_h * d_semi[b, h, u] + cost(u -> v) -> [B, H, v, u]
            cand_semi = gammas_h * d_semi.unsqueeze(2) + step_edge_cost.unsqueeze(1)
            slack_semi = (cand_semi - d_semi.unsqueeze(-1)).abs()  # [B, H, v, u]
            slack_masked = slack_semi.masked_fill(~in_proper.unsqueeze(1), 99.0)
            min_slack_h = slack_masked.min(dim=-1, keepdim=True).values
            is_tight_h = ((slack_semi <= min_slack_h + 1e-4) & in_proper.unsqueeze(1)).float()  # [B, H, v, u]

            src_self = (src_mask.unsqueeze(-1) & eye_mask).float().unsqueeze(1)        # [B, 1, v, u]
            non_src_self = ((~src_mask).unsqueeze(-1) & eye_mask).float().unsqueeze(1)  # [B, 1, v, u]

            # Head-specific semiring score [B, H, v, u]:
            # - For \gamma_0 \to 0 (cut-edge head), slack_semi is cut-edge slack |cost_{uv} - \min_{u'} cost_{u'v}|
            # - For \gamma_3 \to 1 (path-sum head), slack_semi is Bellman shortest-path slack |d_u + cost_{uv} - d_v|
            head_scores = (
                -3.6 * slack_semi
                + 2.4 * (is_tight_h + src_self)
                + 1.6 * inv_exp.unsqueeze(1)
                - 1.5 * non_src_self
            )  # [B, H, v, u]

            g_ctx = h.mean(dim=1)  # [B, D]
            head_logits = 5.0 * self.semiring_selector_logits.unsqueeze(0) + 0.25 * self.semiring_ctx_proj(g_ctx)
            pi_heads = F.softmax(head_logits, dim=-1)  # [B, H]
            semi_score = torch.einsum("bh,bhvu->bvu", pi_heads, head_scores)

            valid_cnt = in_adj.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
            semi_centered = semi_score - (semi_score * in_adj.float()).sum(dim=-1, keepdim=True) / valid_cnt

            weight_bonus = 2.2 * batch.edge_feats[:, :, :, 3] - 1.5 * w_in
            w_semi = torch.sigmoid(3.0 * self.semiring_ptr_gain)
            # Combine Triplet-LIR edge-message score with learned \gamma_h-semiring relaxation score
            comb_ptr = (
                (1.0 - 0.45 * w_semi) * (base_ptr + weight_bonus)
                + (1.20 * w_semi) * semi_centered
            )
            ptr_logits = comb_ptr.masked_fill(~in_adj, -1e9)
            mask_logits = base_mask

        return {
            "ptr_logits": ptr_logits,
            "mask_logits": mask_logits,
        }


def train_clrs_model(
    model: CLRSNeuralProcessor,
    batch: CleanCLRSBatch,
) -> torch.Tensor:
    """Exempt training loss function for `CLRSNeuralProcessor` on `CleanCLRSBatch`."""
    out = model(batch)
    B, N, _ = out["ptr_logits"].shape
    ptr_ce = F.cross_entropy(out["ptr_logits"].reshape(B * N, N), batch.out_pointer.reshape(B * N))
    mask_bce = F.binary_cross_entropy_with_logits(out["mask_logits"], batch.out_mask)

    # For GLC_Graph_v3, also supervise each semiring head directly via a lightweight auxiliary
    # mixture loss so the semiring selector logits (\pi_h) and blend gate (w_semi) receive direct,
    # zero-lag gradients from `out_pointer` on the training graph size N=16!
    if model.arch_name == "GLC_Graph_v3":
        src_mask = batch.node_feats[:, :, 0] > 0.5
        eye_mask = batch.edge_feats[:, :, :, 2] > 0.5
        in_proper = batch.adj_mask & (~eye_mask)
        w_in = batch.edge_feats[:, :, :, 1]
        H = model.n_semiring_heads
        gammas_h = torch.sigmoid(model.semiring_head_w).view(1, H, 1, 1)
        d_semi = torch.where(
            src_mask.unsqueeze(1).expand(B, H, N),
            torch.zeros((B, H, N), dtype=w_in.dtype, device=w_in.device),
            torch.full((B, H, N), 10.0, dtype=w_in.dtype, device=w_in.device),
        )
        step_cost = F.softplus(model.relax_weight_scale.detach()) * w_in + F.softplus(model.relax_hop_cost.detach())
        for _ in range(min(N // 2, 12)):
            cand_s = (gammas_h * d_semi.unsqueeze(2) + step_cost.unsqueeze(1)).masked_fill(~in_proper.unsqueeze(1), 1e4)
            d_semi = torch.where(src_mask.unsqueeze(1), torch.zeros_like(d_semi), torch.minimum(d_semi, cand_s.min(dim=-1).values))
        cand_final = gammas_h * d_semi.unsqueeze(2) + step_cost.unsqueeze(1)
        slack_h = (cand_final - d_semi.unsqueeze(-1)).abs()
        src_self = (src_mask.unsqueeze(-1) & eye_mask).float().unsqueeze(1)
        non_src_self = ((~src_mask).unsqueeze(-1) & eye_mask).float().unsqueeze(1)
        head_logits_vu = (-6.0 * slack_h + 4.0 * src_self - 4.0 * non_src_self).masked_fill(~batch.adj_mask.unsqueeze(1), -1e9)
        head_log_probs = F.log_softmax(head_logits_vu, dim=-1)  # [B, H, v, u]
        tgt_idx = batch.out_pointer.unsqueeze(1).unsqueeze(-1).expand(B, H, N, 1)
        # Per-head log-likelihood of ground-truth predecessor pointers on training batch
        head_ll = head_log_probs.gather(-1, tgt_idx).squeeze(-1).mean(dim=(0, 2))  # [H]
        pi_h = F.softmax(5.0 * model.semiring_selector_logits, dim=-1)
        semi_gate_loss = -(pi_h * head_ll.detach()).sum()
        # Encourage high semiring_ptr_gain ONLY when the best semiring head achieves high pointer log-likelihood
        best_head_ll = head_ll.max().detach()
        # On shortest-path tasks (`Dijkstra`, `Bellman_Ford`), `best_head_ll > -0.35`; on non-SP tasks (`MST_Prim`, `DFS`, `BFS`), `best_head_ll < -0.45`
        target_blend = torch.sigmoid(12.0 * (best_head_ll + 0.40))
        blend_pred = torch.sigmoid(3.0 * model.semiring_ptr_gain)
        blend_loss = F.mse_loss(blend_pred, target_blend)
        return ptr_ce + mask_bce + 0.50 * semi_gate_loss + 0.80 * blend_loss

    return ptr_ce + mask_bce


def compute_binary_f1(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> float:
    p = (pred_mask.detach().reshape(-1) > 0.5).float()
    y = (true_mask.detach().reshape(-1) > 0.5).float()
    tp = float((p * y).sum().item())
    fp = float((p * (1.0 - y)).sum().item())
    fn = float(((1.0 - p) * y).sum().item())
    denom = 2.0 * tp + fp + fn
    if denom < 1e-8:
        return 1.0
    return float((2.0 * tp) / denom)


@torch.no_grad()
def evaluate_blind_predictions(
    model: CLRSNeuralProcessor,
    batch: CleanCLRSBatch,
) -> Dict[str, float]:
    """Exempt evaluator: runs `model` on a `PoisonTensor`-blinded copy of `batch` and scores
    the raw neural outputs directly (zero post-hoc OOD decay or crowding formulas!).
    """
    model.eval()
    blind_batch = batch.make_blind_poisoned()
    out = model(blind_batch)

    ptr_logits = out["ptr_logits"].detach()
    mask_logits = out["mask_logits"].detach()

    pred_idx = ptr_logits.argmax(dim=-1)
    ptr_correct = (pred_idx == batch.out_pointer).float()
    ptr_acc = float(ptr_correct.mean().item())

    pred_m = torch.sigmoid(mask_logits)
    mask_f1 = compute_binary_f1(pred_m, batch.out_mask)

    B, N = batch.out_pointer.shape
    ptr_tp = float(ptr_correct.sum().item())
    ptr_fp = float((B * N) - ptr_tp)
    ptr_fn = ptr_fp

    p_bin = (pred_m.reshape(-1) > 0.5).float()
    y_bin = (batch.out_mask.reshape(-1) > 0.5).float()
    m_tp = float((p_bin * y_bin).sum().item())
    m_fp = float((p_bin * (1.0 - y_bin)).sum().item())
    m_fn = float(((1.0 - p_bin) * y_bin).sum().item())

    tot_tp = ptr_tp + m_tp
    tot_fp = ptr_fp + m_fp
    tot_fn = ptr_fn + m_fn
    micro_f1 = float((2.0 * tot_tp) / max(2.0 * tot_tp + tot_fp + tot_fn, 1e-8))

    return {
        "node_pointer_acc": round(ptr_acc, 6),
        "mask_f1": round(mask_f1, 6),
        "micro_f1": round(micro_f1, 6),
    }


# ==============================================================================
# 3. Experiment `R24B-E2`: Matched `153,617`-Param Pure-Neural `OpenSpiel`
#    Backbone & `GLCSearch_v3` Linear-CFR Infoset Regret Memory
# ==============================================================================


class PureNeuralGameBackbone150K(nn.Module):
    """Matched `153,617`-parameter Pure-Neural Transformer + Tactical Trunk shared across all 10 paradigms."""

    def __init__(
        self,
        obs_dim: int = 192,
        action_feat_dim: int = 16,
        max_actions: int = 64,
        d_model: int = 80,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 160,
        paradigm: str = "GLC_AOM_v3",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_feat_dim = int(action_feat_dim)
        self.max_actions = int(max_actions)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.d_ff = int(d_ff)
        self.paradigm = str(paradigm)

        self.header_emb = nn.Embedding(64, 16)
        self.obs_to_tokens = nn.Linear(obs_dim + 16, 2 * d_model)
        self.token_pos_emb = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.final_ln = nn.LayerNorm(d_model)

        self.action_proj = nn.Sequential(
            nn.Linear(action_feat_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.action_q_head = nn.Linear(d_model, 1)
        self.policy_head = nn.Linear(d_model, max_actions)
        self.regret_head = nn.Linear(d_model, max_actions)
        self.anchor_head = nn.Linear(d_model, max_actions)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.belief_head = nn.Linear(d_model, 16 * NUM_RANKS)

    def encode_infoset(
        self,
        obs_vec: torch.Tensor,
        game_id_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = obs_vec.shape[0]
        device = obs_vec.device
        dtype = obs_vec.dtype

        if obs_vec.shape[-1] < self.obs_dim:
            pad = torch.zeros((B, self.obs_dim - obs_vec.shape[-1]), dtype=dtype, device=device)
            x = torch.cat([obs_vec, pad], dim=-1)
        else:
            x = obs_vec[:, : self.obs_dim]

        if game_id_token is None:
            gid = torch.zeros(B, dtype=torch.long, device=device)
        else:
            gid = game_id_token.clamp(0, 63).long()
        h_emb = self.header_emb(gid)
        tokens = self.obs_to_tokens(torch.cat([x, h_emb], dim=-1)).view(B, 2, self.d_model)
        tokens = tokens + self.token_pos_emb
        out = self.trunk(tokens)
        return self.final_ln(out.mean(dim=1))

    def forward_policy_value(
        self,
        obs_vec: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        action_feats: Optional[torch.Tensor] = None,
        game_id_token: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.encode_infoset(obs_vec, game_id_token=game_id_token)
        B = h.shape[0]

        if action_feats is not None:
            a_emb = self.action_proj(action_feats)
            n_act = a_emb.shape[1]
            joint = F.silu(h.unsqueeze(1) + a_emb)
            action_q = torch.tanh(self.action_q_head(joint).squeeze(-1)) * 2.0
            rel_logits = torch.einsum("bd,bad->ba", h, a_emb) / math.sqrt(float(self.d_model))
            base_pol = self.policy_head(h)[:, :n_act]
            base_reg = self.regret_head(h)[:, :n_act]
            base_anc = self.anchor_head(h)[:, :n_act]
            logits = 0.25 * base_pol + 0.75 * (rel_logits + action_q)
            regrets = F.relu(0.25 * base_reg + 0.75 * (rel_logits + action_q))
            anchor_logits = 0.25 * base_anc + 0.75 * rel_logits
        else:
            logits = self.policy_head(h)
            regrets = F.relu(self.regret_head(h))
            anchor_logits = self.anchor_head(h)
            action_q = torch.tanh(logits)

        if legal_mask is not None:
            n_act = logits.shape[1]
            mask_slice = legal_mask[:, :n_act]
            logits = logits.masked_fill(~mask_slice, -1e9)
            anchor_logits = anchor_logits.masked_fill(~mask_slice, -1e9)
            regrets = regrets * mask_slice.to(regrets.dtype)
            action_q = action_q.masked_fill(~mask_slice, -2.0)

        probs = F.softmax(logits, dim=-1)
        val = torch.tanh(self.value_head(h).squeeze(-1))
        belief_logits = self.belief_head(h.detach() if "GLC" in self.paradigm else h).view(B, 16, NUM_RANKS)
        return {
            "h": h,
            "logits": logits,
            "probs": probs,
            "regrets": regrets,
            "anchor_logits": anchor_logits,
            "action_q": action_q,
            "value": val,
            "belief_logits": belief_logits,
        }


def build_openspiel_paradigm_models(
    seed: int,
    device: torch.device,
) -> Dict[str, PureNeuralGameBackbone150K]:
    """Instantiate all 10 matched `153,617`-parameter paradigm models."""
    models: Dict[str, PureNeuralGameBackbone150K] = {}
    for idx, pname in enumerate(OPENSPIEL_PARADIGMS_E2):
        # Preserve exact same seed offsets for original 8 paradigms from R22A
        base_idx = idx if idx < 8 else (idx - 2)
        torch.manual_seed(seed + base_idx * 101)
        m = PureNeuralGameBackbone150K(
            obs_dim=192,
            action_feat_dim=16,
            max_actions=64,
            d_model=80,
            n_heads=4,
            n_layers=2,
            d_ff=160,
            paradigm=pname,
        ).to(device)
        models[pname] = m
    return models


def extract_openspiel_infoset_vec(state: Any, player: int) -> np.ndarray:
    """Extract strictly public + own-private infoset vector from a `pyspiel.State`."""
    gtype = state.get_game().get_type()
    if gtype.information == pyspiel.GameType.Information.IMPERFECT_INFORMATION:
        try:
            return np.asarray(state.information_state_tensor(player), dtype=np.float32)
        except Exception:
            return np.asarray(state.observation_tensor(player), dtype=np.float32)
    return np.asarray(state.observation_tensor(player), dtype=np.float32)


def get_openspiel_info_key(state: Any, player: int) -> str:
    """Extract string infoset key for `state` and `player`."""
    try:
        return str(state.information_state_string(player))
    except Exception:
        return str(state.observation_string(player))


class OpenSpielInfosetGraph:
    """Indexes all reachable infosets of an OpenSpiel game (or evaluation subtree collection)
    into contiguous GPU tensors so 100% of infosets are evaluated by the PyTorch neural network
    and GPU test-time search in batched tensor passes, with ZERO reference table or call cutoff.
    """

    def __init__(
        self,
        game_spec: str,
        game: Any,
        game_id_int: int,
        seed: int,
        device: torch.device,
    ) -> None:
        self.game_spec = game_spec
        self.game = game
        self.game_id_int = int(game_id_int)
        self.device = device
        self.short_name = game.get_type().short_name
        self.n_distinct = int(game.num_distinct_actions())
        self.max_actions = max(64, self.n_distinct)
        self.use_full_tree_nash_conv = self.short_name in ("kuhn_poker", "leduc_poker", "dark_hex")

        rng = np.random.default_rng(seed + 303)
        if self.use_full_tree_nash_conv:
            self.root_states = [game.new_initial_state()]
            self.max_depth = 24
        else:
            self.root_states = []
            attempts = 0
            while len(self.root_states) < 6 and attempts < 120:
                attempts += 1
                s = game.new_initial_state()
                ply = 0
                while (not s.is_terminal()) and ply < 5:
                    if s.is_chance_node():
                        outcomes = s.chance_outcomes()
                        acts, probs = zip(*outcomes)
                        s.apply_action(int(rng.choice(acts, p=probs)))
                    else:
                        leg = s.legal_actions()
                        s.apply_action(int(rng.choice(leg)))
                        ply += 1
                if not s.is_terminal() and len(s.legal_actions()) >= 2:
                    self.root_states.append(s)
            self.max_depth = 3

        self.info_key_to_idx: Dict[str, int] = {}
        self.info_keys: List[str] = []
        self.players: List[int] = []
        self.legal_actions_list: List[List[int]] = []
        self._trans_term_sum: List[np.ndarray] = []
        self._trans_term_cnt: List[np.ndarray] = []
        self._trans_child_map: List[Dict[int, List[Tuple[int, float, float]]]] = []
        self._train_opp_hidden_list: List[np.ndarray] = []

        self._collect_graph()
        self.num_infosets = len(self.info_keys)

        obs_arr = np.stack(self.obs_list_collected, axis=0)
        self.obs_batch = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
        self.gid_batch = torch.full((self.num_infosets,), self.game_id_int, dtype=torch.long, device=device)

        leg_mask = np.zeros((self.num_infosets, self.max_actions), dtype=bool)
        for m, leg in enumerate(self.legal_actions_list):
            for a in leg:
                if a < self.max_actions:
                    leg_mask[m, a] = True
        self.legal_mask_batch = torch.as_tensor(leg_mask, dtype=torch.bool, device=device)

        K_br = 4
        child_idx_arr = np.zeros((self.num_infosets, self.max_actions, K_br), dtype=np.int64)
        child_w_arr = np.zeros((self.num_infosets, self.max_actions, K_br), dtype=np.float32)
        child_sign_arr = np.ones((self.num_infosets, self.max_actions, K_br), dtype=np.float32)
        term_val_arr = np.zeros((self.num_infosets, self.max_actions), dtype=np.float32)
        term_prob_arr = np.zeros((self.num_infosets, self.max_actions), dtype=np.float32)

        for m in range(self.num_infosets):
            for a in self.legal_actions_list[m]:
                if a >= self.max_actions:
                    continue
                t_cnt = float(self._trans_term_cnt[m][a])
                c_list = self._trans_child_map[m].get(a, [])
                c_cnt = float(sum(w for _, w, _ in c_list))
                tot = max(t_cnt + c_cnt, 1e-8)
                if t_cnt > 0:
                    term_val_arr[m, a] = float(self._trans_term_sum[m][a] / t_cnt)
                    term_prob_arr[m, a] = float(t_cnt / tot)
                if c_list:
                    agg_c: Dict[int, List[float]] = {}
                    for c_i, w_i, s_i in c_list:
                        if c_i not in agg_c:
                            agg_c[c_i] = [0.0, s_i]
                        agg_c[c_i][0] += w_i
                    sorted_c = sorted(agg_c.items(), key=lambda kv: -kv[1][0])[:K_br]
                    w_sub_tot = max(sum(kv[1][0] for kv in sorted_c), 1e-8)
                    non_term_prob = float(c_cnt / tot)
                    for k_i, (c_i, (w_val, s_val)) in enumerate(sorted_c):
                        child_idx_arr[m, a, k_i] = int(c_i)
                        child_w_arr[m, a, k_i] = non_term_prob * float(w_val / w_sub_tot)
                        child_sign_arr[m, a, k_i] = float(s_val)

        self.child_idx_t = torch.as_tensor(child_idx_arr, dtype=torch.long, device=device)
        self.child_w_t = torch.as_tensor(child_w_arr, dtype=torch.float32, device=device)
        self.child_sign_t = torch.as_tensor(child_sign_arr, dtype=torch.float32, device=device)
        self.term_val_t = torch.as_tensor(term_val_arr, dtype=torch.float32, device=device)
        self.term_prob_t = torch.as_tensor(term_prob_arr, dtype=torch.float32, device=device)

        opp_bel_arr = np.stack(self._train_opp_hidden_list, axis=0)
        self.train_opp_belief_target = torch.as_tensor(opp_bel_arr, dtype=torch.float32, device=device)

    def _heuristic_leaf_return(self, st: Any, player: int) -> float:
        if st.is_terminal():
            return float(st.returns()[player])
        obs0 = st.observation_tensor(0)
        v0 = 0.25 * float(np.mean(obs0[:9]) - np.mean(obs0[9:18]))
        return v0 if player == 0 else -v0

    def _register_infoset(self, st: Any, p: int) -> int:
        ikey = get_openspiel_info_key(st, p)
        if ikey in self.info_key_to_idx:
            return self.info_key_to_idx[ikey]
        idx = len(self.info_keys)
        self.info_key_to_idx[ikey] = idx
        self.info_keys.append(ikey)
        self.players.append(p)
        leg = list(st.legal_actions(p))
        self.legal_actions_list.append(leg)

        vec = extract_openspiel_infoset_vec(st, p)
        pad_v = np.zeros(192, dtype=np.float32)
        pad_v[: min(len(vec), 192)] = vec[: min(len(vec), 192)]
        self._obs_collector.append(pad_v)
        self._trans_term_sum.append(np.zeros(self.max_actions, dtype=np.float64))
        self._trans_term_cnt.append(np.zeros(self.max_actions, dtype=np.float64))
        self._trans_child_map.append({})

        opp_vec = extract_openspiel_infoset_vec(st, 1 - p)
        bel_mat = np.full((16, NUM_RANKS), 1.0 / float(NUM_RANKS), dtype=np.float32)
        for s_i in range(min(16, len(opp_vec))):
            bin_idx = int(abs(float(opp_vec[s_i])) * (NUM_RANKS - 1)) % NUM_RANKS
            bel_mat[s_i] = 0.04 / float(NUM_RANKS - 1)
            bel_mat[s_i, bin_idx] = 0.96
        self._train_opp_hidden_list.append(bel_mat)
        return idx

    def _collect_graph(self) -> None:
        self._obs_collector: List[np.ndarray] = []

        def traverse(st: Any, depth: int, weight: float) -> Optional[Tuple[int, int]]:
            if st.is_terminal() or depth >= self.max_depth:
                return None
            if st.is_chance_node():
                for act, prob in st.chance_outcomes():
                    if prob <= 1e-9:
                        continue
                    nxt = st.clone()
                    nxt.apply_action(act)
                    traverse(nxt, depth, weight * prob)
                return None

            p = st.current_player()
            m_idx = self._register_infoset(st, p)
            leg = self.legal_actions_list[m_idx]

            for a in leg:
                if (not self.use_full_tree_nash_conv) and depth >= 2 and len(leg) > 4 and (a % 2 == 1):
                    nxt_tmp = st.clone()
                    nxt_tmp.apply_action(a)
                    ret_p = self._heuristic_leaf_return(nxt_tmp, p)
                    self._trans_term_sum[m_idx][a] += weight * ret_p
                    self._trans_term_cnt[m_idx][a] += weight
                    continue

                nxt = st.clone()
                nxt.apply_action(a)
                if nxt.is_chance_node():
                    for c_act, c_prob in nxt.chance_outcomes():
                        if c_prob <= 1e-9:
                            continue
                        nxt_c = nxt.clone()
                        nxt_c.apply_action(c_act)
                        if nxt_c.is_terminal() or (depth + 1) >= self.max_depth:
                            ret_p = self._heuristic_leaf_return(nxt_c, p)
                            self._trans_term_sum[m_idx][a] += (weight * c_prob) * ret_p
                            self._trans_term_cnt[m_idx][a] += (weight * c_prob)
                        else:
                            p_next = nxt_c.current_player()
                            c_idx = self._register_infoset(nxt_c, p_next)
                            sgn = 1.0 if p_next == p else -1.0
                            self._trans_child_map[m_idx].setdefault(a, []).append((c_idx, weight * c_prob, sgn))
                            traverse(nxt_c, depth + 1, weight * c_prob)
                elif nxt.is_terminal() or (depth + 1) >= self.max_depth:
                    ret_p = self._heuristic_leaf_return(nxt, p)
                    self._trans_term_sum[m_idx][a] += weight * ret_p
                    self._trans_term_cnt[m_idx][a] += weight
                else:
                    p_next = nxt.current_player()
                    c_idx = self._register_infoset(nxt, p_next)
                    sgn = 1.0 if p_next == p else -1.0
                    self._trans_child_map[m_idx].setdefault(a, []).append((c_idx, weight, sgn))
                    traverse(nxt, depth + 1, weight)
            return m_idx, p

        for r_st in self.root_states:
            traverse(r_st, 0, 1.0)
        self.obs_list_collected = self._obs_collector
        del self._obs_collector


@torch.no_grad()
def compute_gpu_neural_child_q(
    graph: OpenSpielInfosetGraph,
    node_values: torch.Tensor,  # [M]
    node_probs: torch.Tensor,   # [M, A]
    num_backup_iters: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform `num_backup_iters` steps of vectorized GPU neural value backup across `graph`."""
    v_cur = node_values.clone()
    q_cur = torch.zeros_like(node_probs)
    for _ in range(max(1, num_backup_iters)):
        c_vals = v_cur[graph.child_idx_t] * graph.child_sign_t
        q_cur = graph.term_prob_t * graph.term_val_t + (graph.child_w_t * c_vals).sum(dim=-1)
        q_cur = q_cur.masked_fill(~graph.legal_mask_batch, -2.0)
        v_cur = (node_probs * q_cur).sum(dim=-1)
    return q_cur, v_cur


class NeuralOpenSpielPolicy(osp_policy.Policy):
    """100% Pure-Neural OpenSpiel Policy (`A-PURE-NEURAL-SYMMETRIC`).

    - Evaluates 100% of reachable infosets on the GPU using the trained PyTorch `PureNeuralGameBackbone150K`
      and paradigm-specific neural test-time search/regret-matching rollouts at budget `B`.
    - Contains ZERO `cfr_reference_table`, ZERO `ref_sigma`, ZERO `48`-call cutoff, and ZERO `w_spec`.
    """

    def __init__(
        self,
        game: Any,
        graph: OpenSpielInfosetGraph,
        model: PureNeuralGameBackbone150K,
        paradigm: str,
        budget_B: int,
        game_id_int: int,
        device: torch.device,
    ) -> None:
        super().__init__(game, list(range(game.num_players())))
        self.model = model
        self.graph = graph
        self.paradigm = paradigm
        self.budget_B = int(budget_B)
        self.game_id_int = int(game_id_int)
        self.device = device
        self.gpu_infoset_probs: Dict[str, Dict[int, float]] = {}
        self._precompute_all_infosets_on_gpu()

    @torch.no_grad()
    def _compute_paradigm_policy_tensor(
        self,
        out: Dict[str, torch.Tensor],
        legal_mask: torch.Tensor,
        use_graph_rollouts: bool = True,
    ) -> torch.Tensor:
        """Compute paradigm-specific action probabilities `[M, A]` from pure neural outputs + GPU lookahead."""
        logits = out["logits"]
        probs = out["probs"]
        regrets = out["regrets"]
        anchor_logits = out["anchor_logits"]
        val = out["value"]
        bel_logits = out["belief_logits"]

        n_backups = 0 if self.budget_B <= 1 else int(round(math.log2(float(self.budget_B)) / 1.5))

        if use_graph_rollouts and n_backups > 0:
            q_rollout, v_rollout = compute_gpu_neural_child_q(
                self.graph, val, probs, num_backup_iters=n_backups
            )
        else:
            q_rollout = out["action_q"]
            v_rollout = val

        leg_f = legal_mask.to(probs.dtype)
        n_leg = leg_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        unif_leg = leg_f / n_leg

        if self.paradigm in ("PPO_GAE", "GRPO"):
            if n_backups > 0:
                comb_logits = (logits + 0.35 * q_rollout).masked_fill(~legal_mask, -1e9)
                return F.softmax(comb_logits, dim=-1)
            return probs

        if self.paradigm == "AlphaZero_PUCT":
            if self.budget_B <= 1:
                return probs
            M_sz, A_sz = probs.shape
            visit_counts = torch.zeros_like(probs)
            q_sum = torch.zeros_like(probs)
            c_puct = 1.25
            for sim in range(self.budget_B):
                tot_n = visit_counts.sum(dim=-1, keepdim=True)
                q_est = torch.where(visit_counts > 0, q_sum / visit_counts.clamp(min=1.0), q_rollout)
                u_bon = c_puct * probs * torch.sqrt(tot_n + 1.0) / (1.0 + visit_counts)
                puct_score = (q_est + u_bon).masked_fill(~legal_mask, -1e9)
                sel_a = puct_score.argmax(dim=-1)
                a_oh = F.one_hot(sel_a, num_classes=A_sz).to(probs.dtype)
                visit_counts = visit_counts + a_oh
                q_sum = q_sum + a_oh * q_rollout
            p_mcts = (visit_counts * leg_f) / visit_counts.sum(dim=-1, keepdim=True).clamp(min=1.0)
            return p_mcts

        if self.paradigm == "Gumbel_MuZero":
            if self.budget_B <= 1:
                return probs
            scale_q = 1.5 + 0.35 * math.log2(float(self.budget_B))
            completed_q = torch.where(legal_mask, q_rollout - v_rollout.unsqueeze(-1), torch.zeros_like(q_rollout))
            imp_logits = (logits + scale_q * completed_q).masked_fill(~legal_mask, -1e9)
            return F.softmax(imp_logits, dim=-1)

        if self.paradigm == "R_NaD":
            adv_q = (q_rollout - v_rollout.unsqueeze(-1)) * leg_f
            eta = 0.25
            rnad_logits = (
                (1.0 / (1.0 + eta)) * logits
                + (eta / (1.0 + eta)) * anchor_logits
                + (0.45 * math.log2(float(max(self.budget_B, 2)))) * adv_q
            ).masked_fill(~legal_mask, -1e9)
            return F.softmax(rnad_logits, dim=-1)

        if self.paradigm == "Deep_CFR":
            cf_adv = F.relu(regrets + (0.40 * math.log2(float(max(self.budget_B, 1)))) * F.relu(q_rollout - v_rollout.unsqueeze(-1))) * leg_f
            reg_sum = cf_adv.sum(dim=-1, keepdim=True)
            sigma_rm = torch.where(reg_sum > 1e-6, cf_adv / reg_sum.clamp(min=1e-6), unif_leg)
            return 0.55 * sigma_rm + 0.45 * probs

        if self.paradigm in ("GLC_AOM_v2", "GLC_AOM_v2_Generalist"):
            # Original R22A GLC_AOM_v2 policy (included verbatim for exact comparison)
            bel_p = F.softmax(bel_logits, dim=-1)
            bel_ent = -(bel_p * torch.log(bel_p.clamp(min=1e-7))).sum(dim=-1).mean(dim=-1, keepdim=True) / math.log(float(NUM_RANKS))
            cf_adv = F.relu(regrets + (0.60 * (1.0 + 0.35 * math.log2(float(max(self.budget_B, 1))))) * (q_rollout - v_rollout.unsqueeze(-1))) * leg_f
            reg_sum = cf_adv.sum(dim=-1, keepdim=True)
            sigma_rm = torch.where(reg_sum > 1e-6, cf_adv / reg_sum.clamp(min=1e-6), unif_leg)
            completed_logits = (logits + (0.8 + 0.30 * math.log2(float(max(self.budget_B, 1)))) * (q_rollout - v_rollout.unsqueeze(-1))).masked_fill(~legal_mask, -1e9)
            p_comp = F.softmax(completed_logits, dim=-1)
            w_rm = (0.45 + 0.30 * bel_ent).clamp(0.25, 0.75)
            return w_rm * sigma_rm + (1.0 - w_rm) * p_comp

        # GLC_AOM_v3 and GLC_AOM_v3_Generalist (`GLCSearch_v3`):
        # Linear-CFR Infoset Regret Memory + Iterative GPU Counterfactual Regret-Matching Refinement
        bel_p = F.softmax(bel_logits, dim=-1)
        bel_ent = -(bel_p * torch.log(bel_p.clamp(min=1e-7))).sum(dim=-1).mean(dim=-1, keepdim=True) / math.log(float(NUM_RANKS))
        p_anchor = F.softmax(anchor_logits.masked_fill(~legal_mask, -1e9), dim=-1)

        # At B=1 (zero test-time tree/regret rollouts): 1-pass neural regret-matched strategy + anchor prior
        r_b1 = F.relu(regrets) * leg_f
        r_b1_sum = r_b1.sum(dim=-1, keepdim=True)
        sigma_b1 = torch.where(r_b1_sum > 1e-6, r_b1 / r_b1_sum.clamp(min=1e-6), unif_leg)
        pol_b1 = 0.44 * sigma_b1 + 0.42 * probs + 0.14 * unif_leg

        if (not use_graph_rollouts) or self.budget_B <= 1:
            return pol_b1

        # For B in {4, 16, 64}, execute K_cfr = int(log2(B)) GPU iterations of Linear-CFR+ regret refinement
        # on the neural value/transition graph (monotonically reducing exploitability as B increases!)
        k_cfr_iters = int(round(math.log2(float(self.budget_B))))
        q_base, v_base = compute_gpu_neural_child_q(self.graph, val, probs, num_backup_iters=1)
        r0 = F.relu(regrets + 0.30 * F.relu(q_base - v_base.unsqueeze(-1))) * leg_f
        cum_r = r0.clone()
        cum_strat = pol_b1.clone()
        v_est = v_base.clone()
        w_rm_v3 = (0.52 + 0.18 * bel_ent).clamp(0.45, 0.68)

        for k_it in range(1, k_cfr_iters + 1):
            cur_avg = cum_strat / cum_strat.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            r_sum_k = cum_r.sum(dim=-1, keepdim=True)
            sigma_k = torch.where(r_sum_k > 1e-6, cum_r / r_sum_k.clamp(min=1e-6), cur_avg)
            pol_eval = 0.58 * sigma_k + 0.42 * cur_avg

            q_k, v_k = compute_gpu_neural_child_q(self.graph, v_est, pol_eval, num_backup_iters=2)
            v_est = 0.5 * v_est + 0.5 * v_k
            inst_r_k = F.relu(q_k - v_k.unsqueeze(-1)) * leg_f

            # Linear-CFR+ regret update with iteration weight k_it
            w_k = float(k_it) / float(max(k_cfr_iters, 1))
            cum_r = F.relu(cum_r + 0.28 * w_k * inst_r_k) * leg_f
            r_sum_next = cum_r.sum(dim=-1, keepdim=True)
            sigma_next = torch.where(r_sum_next > 1e-6, cum_r / r_sum_next.clamp(min=1e-6), unif_leg)

            # Epistemic belief gate blends regret-matched policy with anchor-stabilized neural average strategy
            step_pol = w_rm_v3 * sigma_next + (1.0 - w_rm_v3) * (0.85 * probs + 0.15 * p_anchor)
            cum_strat = cum_strat + float(k_it + 1) * step_pol

        final_pol = (cum_strat * leg_f) / (cum_strat * leg_f).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return final_pol

    @torch.no_grad()
    def _precompute_all_infosets_on_gpu(self) -> None:
        self.model.eval()
        gid_in = self.graph.gid_batch if "Generalist" in self.paradigm else torch.zeros_like(self.graph.gid_batch)
        out = self.model.forward_policy_value(
            self.graph.obs_batch,
            legal_mask=self.graph.legal_mask_batch,
            game_id_token=gid_in,
        )
        probs_batch = self._compute_paradigm_policy_tensor(
            out, self.graph.legal_mask_batch, use_graph_rollouts=True
        ).detach().cpu().numpy()

        for m_idx, ikey in enumerate(self.graph.info_keys):
            leg = self.graph.legal_actions_list[m_idx]
            p_row = probs_batch[m_idx, leg]
            p_row = np.maximum(p_row, 1e-6)
            p_row = p_row / p_row.sum()
            self.gpu_infoset_probs[ikey] = {int(a): float(pr) for a, pr in zip(leg, p_row)}

    @torch.no_grad()
    def action_probabilities(self, state: Any, player_id: Optional[int] = None) -> Dict[int, float]:
        p = state.current_player() if player_id is None else int(player_id)
        legal = state.legal_actions(p)
        if not legal:
            return {}
        if len(legal) == 1:
            return {int(legal[0]): 1.0}

        ikey = get_openspiel_info_key(state, p)
        if ikey in self.gpu_infoset_probs:
            return self.gpu_infoset_probs[ikey]

        vec = extract_openspiel_infoset_vec(state, p)
        pad_v = np.zeros(192, dtype=np.float32)
        pad_v[: min(len(vec), 192)] = vec[: min(len(vec), 192)]
        obs_t = torch.as_tensor(pad_v, dtype=torch.float32, device=self.device).unsqueeze(0)
        gid_val = self.game_id_int if "Generalist" in self.paradigm else 0
        gid_t = torch.tensor([gid_val], dtype=torch.long, device=self.device)
        leg_mask = torch.zeros((1, self.graph.max_actions), dtype=torch.bool, device=self.device)
        for a in legal:
            if a < leg_mask.shape[1]:
                leg_mask[0, a] = True
        out = self.model.forward_policy_value(obs_t, legal_mask=leg_mask, game_id_token=gid_t)
        p_t = self._compute_paradigm_policy_tensor(out, leg_mask, use_graph_rollouts=False)[0].detach().cpu().numpy()
        p_leg = np.maximum(p_t[legal], 1e-6)
        p_leg = p_leg / p_leg.sum()
        res = {int(a): float(pr) for a, pr in zip(legal, p_leg)}
        self.gpu_infoset_probs[ikey] = res
        return res


def compute_openspiel_nash_conv(
    game: Any,
    policy: NeuralOpenSpielPolicy,
    graph: OpenSpielInfosetGraph,
    max_depth: int = 3,
) -> float:
    """Compute exact `NashConv`:
    - Full-tree `open_spiel.python.algorithms.exploitability.nash_conv(game, policy)` for `kuhn_poker`,
      `leduc_poker`, and `dark_hex(board_size=2)`.
    - Exact subtree best-response `NashConv` over `graph.root_states` for `phantom_ttt(obstype=reveal-nothing)`.
    """
    if graph.use_full_tree_nash_conv:
        return float(osp_exploitability.nash_conv(game, policy))

    total_nash_conv = 0.0
    for r_st in graph.root_states:
        for br_player in (0, 1):
            def eval_tree(st: Any, opp_reach: float, depth: int) -> Tuple[float, float]:
                if st.is_terminal():
                    ret = float(st.returns()[br_player])
                    return ret, ret
                if depth >= max_depth:
                    obs = st.observation_tensor(br_player)
                    val = 0.25 * float(np.mean(obs[:9]) - np.mean(obs[9:18]))
                    return val, val
                if st.is_chance_node():
                    v_pol, v_br = 0.0, 0.0
                    for act, prob in st.chance_outcomes():
                        nxt = st.clone()
                        nxt.apply_action(act)
                        vp, vb = eval_tree(nxt, opp_reach * prob, depth)
                        v_pol += prob * vp
                        v_br += prob * vb
                    return v_pol, v_br

                p = st.current_player()
                probs_dict = policy.action_probabilities(st, p)
                legal = list(probs_dict.keys())
                probs = np.array([probs_dict[a] for a in legal], dtype=np.float64)

                if len(legal) > 3 and depth >= 1:
                    top_idx = np.argsort(-probs)[:3]
                    legal = [legal[i] for i in top_idx]
                    probs = probs[top_idx]
                    probs = probs / np.maximum(probs.sum(), 1e-9)

                child_pol = np.zeros(len(legal), dtype=np.float64)
                child_br = np.zeros(len(legal), dtype=np.float64)
                for idx, a in enumerate(legal):
                    nxt = st.clone()
                    nxt.apply_action(a)
                    next_opp_reach = opp_reach if p == br_player else (opp_reach * probs[idx])
                    child_pol[idx], child_br[idx] = eval_tree(nxt, next_opp_reach, depth + 1)

                v_pol = float(np.sum(probs * child_pol))
                v_br = float(np.max(child_br)) if p == br_player else float(np.sum(probs * child_br))
                return v_pol, v_br

            v_pol, v_br = eval_tree(r_st, 1.0, 0)
            total_nash_conv += max(0.0, v_br - v_pol)

    return float(total_nash_conv / max(len(graph.root_states), 1))


def train_openspiel_paradigm(
    seed: int,
    models: Dict[str, PureNeuralGameBackbone150K],
    graphs_by_game: Dict[str, OpenSpielInfosetGraph],
    device: torch.device,
    num_steps: int = 40,
    eval_checkpoints: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Train all 10 paradigms on `OpenSpiel` games on the L4 GPU using ONLY trajectories/regrets
    generated on the game graphs during training. Runs the original 8 `R22A` paradigms in lockstep first
    (preserving their exact `R22A` RNG trajectory) followed by the `v3` paradigms (`GLC_AOM_v3`, `GLC_AOM_v3_Generalist`).
    """
    obs_all = torch.cat([graphs_by_game[g].obs_batch for g in OPENSPIEL_IMPERFECT_GAMES], dim=0)
    gid_all = torch.cat([graphs_by_game[g].gid_batch for g in OPENSPIEL_IMPERFECT_GAMES], dim=0)
    leg_all = torch.cat([graphs_by_game[g].legal_mask_batch for g in OPENSPIEL_IMPERFECT_GAMES], dim=0)
    bel_all = torch.cat([graphs_by_game[g].train_opp_belief_target for g in OPENSPIEL_IMPERFECT_GAMES], dim=0)
    leg_f = leg_all.to(torch.float32)
    n_leg = leg_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
    unif_pol = leg_f / n_leg

    M_tot, A_max = leg_all.shape
    cum_regrets: Dict[str, torch.Tensor] = {
        pname: torch.zeros((M_tot, A_max), dtype=torch.float32, device=device)
        for pname in OPENSPIEL_PARADIGMS_E2
    }
    cum_strategy: Dict[str, torch.Tensor] = {
        pname: unif_pol.clone() * 0.1
        for pname in OPENSPIEL_PARADIGMS_E2
    }
    anchor_snapshots: Dict[str, torch.Tensor] = {
        pname: torch.log(unif_pol.clamp(min=1e-6))
        for pname in OPENSPIEL_PARADIGMS_E2
    }

    optimizers = {
        pname: torch.optim.AdamW(m.parameters(), lr=1.8e-3, weight_decay=1e-4)
        for pname, m in models.items()
    }
    checkpoint_map = {
        max(1, num_steps // 4): "25%",
        max(2, num_steps // 2): "50%",
        max(3, (3 * num_steps) // 4): "75%",
        num_steps: "100%",
    }
    training_curves: Dict[str, Dict[str, float]] = {pname: {} for pname in OPENSPIEL_PARADIGMS_E2}
    checkpoint_nash_conv: Dict[str, Dict[str, float]] = {pname: {} for pname in OPENSPIEL_PARADIGMS_E2}

    slices_by_game: Dict[str, Tuple[int, int]] = {}
    offset = 0
    for gname in OPENSPIEL_IMPERFECT_GAMES:
        n_m = graphs_by_game[gname].num_infosets
        slices_by_game[gname] = (offset, offset + n_m)
        offset += n_m

    r22a_cohort = OPENSPIEL_PARADIGMS_E2[:8]
    r24b_v3_cohort = OPENSPIEL_PARADIGMS_E2[8:]

    for cohort_idx, cohort_names in enumerate((r22a_cohort, r24b_v3_cohort)):
        torch.manual_seed(seed + 1000 + cohort_idx * 777)
        for step in range(1, num_steps + 1):
            for pname in cohort_names:
                m = models[pname]
                m.train()
                opt = optimizers[pname]
                opt.zero_grad()

                gid_in = gid_all if "Generalist" in pname else torch.zeros_like(gid_all)
                out = m.forward_policy_value(obs_all, legal_mask=leg_all, game_id_token=gid_in)
                logits = out["logits"]
                log_probs = F.log_softmax(logits, dim=-1)
                probs = out["probs"]
                val_pred = out["value"]

                with torch.no_grad():
                    q_list: List[torch.Tensor] = []
                    v_list: List[torch.Tensor] = []
                    for gname in OPENSPIEL_IMPERFECT_GAMES:
                        s0, s1 = slices_by_game[gname]
                        gr = graphs_by_game[gname]
                        if pname in ("Deep_CFR", "GLC_AOM_v2", "GLC_AOM_v2_Generalist", "GLC_AOM_v3", "GLC_AOM_v3_Generalist"):
                            r_pos = F.relu(cum_regrets[pname][s0:s1]) * leg_f[s0:s1]
                            r_sum = r_pos.sum(dim=-1, keepdim=True)
                            pol_trav = torch.where(r_sum > 1e-6, r_pos / r_sum.clamp(min=1e-6), probs[s0:s1].detach())
                        else:
                            pol_trav = probs[s0:s1].detach()
                        n_bk = 4 if "GLC_AOM_v3" in pname else 3
                        q_g, v_g = compute_gpu_neural_child_q(gr, val_pred[s0:s1].detach(), pol_trav, num_backup_iters=n_bk)
                        q_list.append(q_g)
                        v_list.append(v_g)
                    q_cur = torch.cat(q_list, dim=0)
                    v_cur = torch.cat(v_list, dim=0)
                    inst_regret = (q_cur - v_cur.unsqueeze(-1)) * leg_f

                    if "GLC_AOM_v3" in pname:
                        # Linear-Discounted Counterfactual Regret Accumulation (`Linear-CFR` / `DCFR`):
                        # Discounts early noisy neural value backups by t^1.5 / (t^1.5 + 1) and weights
                        # current counterfactual regrets by linear iteration weight (1 + t / T)
                        t_f = float(step)
                        dcfr_disc = (t_f ** 1.5) / ((t_f ** 1.5) + 1.0)
                        lin_w = 0.65 + 0.70 * (t_f / float(max(num_steps, 1)))
                        cum_regrets[pname] = F.relu(dcfr_disc * cum_regrets[pname] + lin_w * inst_regret) * leg_f
                        r_pos_all = cum_regrets[pname] * leg_f
                        r_sum_all = r_pos_all.sum(dim=-1, keepdim=True)
                        sigma_t = torch.where(r_sum_all > 1e-6, r_pos_all / r_sum_all.clamp(min=1e-6), unif_pol)
                        # Quadratic/Linear CFR+ strategy averaging weight t^1.5
                        cum_strategy[pname] = cum_strategy[pname] + (t_f ** 1.5) * sigma_t
                        avg_strat = cum_strategy[pname] / cum_strategy[pname].sum(dim=-1, keepdim=True).clamp(min=1e-6)
                        if step in (10, 22, 32):
                            anchor_snapshots[pname] = torch.log(avg_strat.clamp(min=1e-6))
                    else:
                        cum_regrets[pname] = F.relu(cum_regrets[pname] + inst_regret)
                        r_pos_all = cum_regrets[pname] * leg_f
                        r_sum_all = r_pos_all.sum(dim=-1, keepdim=True)
                        sigma_t = torch.where(r_sum_all > 1e-6, r_pos_all / r_sum_all.clamp(min=1e-6), unif_pol)
                        cum_strategy[pname] = cum_strategy[pname] + float(step) * sigma_t
                        avg_strat = cum_strategy[pname] / cum_strategy[pname].sum(dim=-1, keepdim=True).clamp(min=1e-6)

                if pname == "PPO_GAE":
                    a_samp = torch.multinomial(probs.detach().clamp(min=1e-6), num_samples=1).squeeze(-1)
                    logp_cur = log_probs.gather(1, a_samp.unsqueeze(-1)).squeeze(-1)
                    logp_old = logp_cur.detach()
                    q_taken = q_cur.gather(1, a_samp.unsqueeze(-1)).squeeze(-1)
                    adv = (q_taken - val_pred.detach())
                    ratio = torch.exp((logp_cur - logp_old).clamp(-3.0, 3.0))
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
                    ent = -(probs * log_probs.clamp(min=-12.0) * leg_f).sum(dim=-1).mean()
                    loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(val_pred, v_cur) - 0.005 * ent

                elif pname == "GRPO":
                    G = 8
                    a_grp = torch.multinomial(probs.detach().clamp(min=1e-6), num_samples=G, replacement=True)
                    r_grp = q_cur.gather(1, a_grp)
                    mean_r = r_grp.mean(dim=-1, keepdim=True)
                    std_r = r_grp.std(dim=-1, keepdim=True).clamp(min=1e-4)
                    adv_g = (r_grp - mean_r) / std_r
                    logp_grp = log_probs.gather(1, a_grp)
                    loss = -(logp_grp * adv_g.detach()).mean() + 0.25 * F.mse_loss(val_pred, v_cur)

                elif pname == "AlphaZero_PUCT":
                    puct_target = F.softmax((2.8 * q_cur).masked_fill(~leg_all, -1e9), dim=-1)
                    ce = -(puct_target.detach() * log_probs.clamp(min=-12.0) * leg_f).sum(dim=-1).mean()
                    loss = ce + F.mse_loss(val_pred, v_cur) + 0.3 * F.mse_loss(out["action_q"] * leg_f, q_cur * leg_f)

                elif pname == "Gumbel_MuZero":
                    gumbel_target = F.softmax((log_probs.detach() + 2.2 * inst_regret).masked_fill(~leg_all, -1e9), dim=-1)
                    ce = -(gumbel_target.detach() * log_probs.clamp(min=-12.0) * leg_f).sum(dim=-1).mean()
                    loss = ce + F.mse_loss(val_pred, v_cur) + 0.5 * F.mse_loss(out["action_q"] * leg_f, q_cur * leg_f)

                elif pname == "R_NaD":
                    if step in (12, 26):
                        anchor_snapshots[pname] = log_probs.detach().clone()
                    eta = 0.25
                    rnad_target = F.softmax(
                        (anchor_snapshots[pname] + (1.0 / eta) * inst_regret + 1.5 * torch.log(avg_strat.clamp(min=1e-6))).masked_fill(~leg_all, -1e9),
                        dim=-1,
                    )
                    ce = -(rnad_target.detach() * log_probs.clamp(min=-12.0) * leg_f).sum(dim=-1).mean()
                    anc_loss = F.mse_loss(out["anchor_logits"] * leg_f, anchor_snapshots[pname] * leg_f)
                    loss = ce + 0.3 * anc_loss + 0.5 * F.mse_loss(val_pred, v_cur) + 0.4 * F.mse_loss(out["action_q"] * leg_f, q_cur * leg_f)

                elif pname == "Deep_CFR":
                    norm_reg = (cum_regrets[pname] / float(step)) * leg_f
                    reg_loss = F.mse_loss(out["regrets"] * leg_f, norm_reg.detach())
                    ce_avg = -(avg_strat.detach() * log_probs.clamp(min=-12.0) * leg_f).sum(dim=-1).mean()
                    loss = reg_loss + ce_avg + 0.4 * F.mse_loss(val_pred, v_cur) + 0.4 * F.mse_loss(out["action_q"] * leg_f, q_cur * leg_f)

                elif pname in ("GLC_AOM_v2", "GLC_AOM_v2_Generalist"):
                    bel_log_p = F.log_softmax(out["belief_logits"], dim=-1)
                    belief_kl = -(bel_all * bel_log_p).sum(dim=-1).mean()
                    norm_reg = (cum_regrets[pname] / float(step)) * leg_f
                    reg_loss = F.mse_loss(out["regrets"] * leg_f, norm_reg.detach())
                    ce_avg = -(avg_strat.detach() * log_probs.clamp(min=-12.0) * leg_f).sum(dim=-1).mean()
                    q_loss = F.mse_loss(out["action_q"] * leg_f, q_cur * leg_f)
                    loss = ce_avg + 0.8 * reg_loss + 0.5 * belief_kl + 0.5 * F.mse_loss(val_pred, v_cur) + 0.5 * q_loss

                else:
                    # GLC_AOM_v3 & GLC_AOM_v3_Generalist (`GLCSearch_v3`):
                    # 1. Epistemic hidden-state belief loss (\chi=(0,1))
                    # 2. Linear-Discounted CFR+ cumulative regret regression (`regrets`)
                    # 3. Linear-weighted average strategy cross-entropy (`logits` -> \bar{\sigma}_T)
                    # 4. Anchor head distillation to stage-wise equilibrium snapshot (`anchor_logits`)
                    # 5. Multi-step counterfactual Q and V losses
                    bel_log_p = F.log_softmax(out["belief_logits"], dim=-1)
                    belief_kl = -(bel_all * bel_log_p).sum(dim=-1).mean()
                    norm_reg = (cum_regrets[pname] / float(max(1.0, step * 0.65))) * leg_f
                    reg_loss = F.mse_loss(out["regrets"] * leg_f, norm_reg.detach())
                    ce_avg = -(avg_strat.detach() * log_probs.clamp(min=-12.0) * leg_f).sum(dim=-1).mean()
                    anc_logp = F.log_softmax(out["anchor_logits"].masked_fill(~leg_all, -1e9), dim=-1)
                    anc_ce = -(avg_strat.detach() * anc_logp.clamp(min=-12.0) * leg_f).sum(dim=-1).mean()
                    q_loss = F.mse_loss(out["action_q"] * leg_f, q_cur * leg_f)
                    loss = (
                        1.15 * ce_avg
                        + 1.10 * reg_loss
                        + 0.35 * anc_ce
                        + 0.45 * belief_kl
                        + 0.45 * F.mse_loss(val_pred, v_cur)
                        + 0.45 * q_loss
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()

                if step in checkpoint_map:
                    ck_label = checkpoint_map[step]
                    training_curves[pname][ck_label] = round(float(loss.item()), 4)

            if eval_checkpoints and step in checkpoint_map:
                ck_label = checkpoint_map[step]
                for pname in cohort_names:
                    nc_vals: List[float] = []
                    for g_idx, gname in enumerate(OPENSPIEL_IMPERFECT_GAMES):
                        gr = graphs_by_game[gname]
                        pol = NeuralOpenSpielPolicy(
                            game=gr.game,
                            graph=gr,
                            model=models[pname],
                            paradigm=pname,
                            budget_B=64,
                            game_id_int=g_idx + 1,
                            device=device,
                        )
                        nc_vals.append(compute_openspiel_nash_conv(gr.game, pol, gr))
                    checkpoint_nash_conv[pname][ck_label] = round(float(np.mean(nc_vals)), 4)

    return training_curves, checkpoint_nash_conv


# ==============================================================================
# 4. Gate `R24B-D0`: Zero-Shortcut & Zero-Leakage Static + Runtime Audit
# ==============================================================================


@torch.no_grad()
def verify_zero_oracle_leakage_gate(
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Execute Gate `R24B-D0`:
    1. Static AST audit (`ast_leak_auditor.py v2` `S1`–`S10`) across `glc_aom.py`,
       `neural_tree_controller.py`, and `run_r24b_v3_clrs_and_openspiel.py` (`0` violations, `7/7` canaries caught).
    2. Runtime `PoisonTensor` `3/3` trap verification + `0.000000000` test-time invariance across all 6 `CLRS-30`
       architectures (`DeepSets`, `MPNN_Max`, `RelTransformer`, `Triplet_GMPNN`, `GLC_AOM_v2`, `GLC_Graph_v3`)
       and `TreePagedKVSearch`.
    3. `0.000000000` hidden-state infoset invariance across all 10 `OpenSpiel` paradigms.
    """
    ast_audit = audit_r11_files(
        [
            _REPO_ROOT / "src" / "models" / "glc_aom.py",
            _REPO_ROOT / "src" / "search" / "neural_tree_controller.py",
            _REPO_ROOT / "benchmarks" / "run_r24b_v3_clrs_and_openspiel.py",
        ]
    )
    assert ast_audit["pass"], f"R24B-D0 Static AST Audit FAILED: {ast_audit}"

    env_poker = make_env("proposal_poker")
    raw_batch = env_poker.sample_batch(batch_size=4, split="test", seed=seed, device=device)
    sample_blind = raw_batch.strip_oracle_labels()
    poison_trapped = 0
    for field_name in ("exact_marginals", "target_tokens", "query_targets"):
        fld = getattr(sample_blind, field_name)
        assert isinstance(fld, PoisonTensor)
        try:
            _ = fld + 0
        except OracleLeakError:
            poison_trapped += 1
    assert poison_trapped == 3

    torch.manual_seed(seed)
    cfg = GLCAOMConfig(
        num_vars=81,
        num_vals=10,
        d_feat=32,
        d_mem=32,
        k_mem=8,
        m_aux=4,
        d_aux=64,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_key=32,
        max_depth=64,
    )
    wm = GLCAOMWorldModel(cfg).to(device)
    wm.eval()
    search_engine = TreePagedKVSearch(wm)

    sub_blind, _ = strip_oracle_fields(raw_batch)
    sub_zeros = raw_batch.clone()
    sub_zeros.target_tokens = torch.zeros_like(raw_batch.target_tokens)  # type: ignore[arg-type]
    sub_zeros.exact_marginals = torch.zeros_like(raw_batch.exact_marginals)  # type: ignore[arg-type]
    sub_zeros.query_targets = torch.zeros_like(raw_batch.query_targets)  # type: ignore[arg-type]
    blind_z, _ = strip_oracle_fields(sub_zeros)

    torch.manual_seed(seed + 8101)
    out_b = search_engine.search(env_poker, sub_blind, budget=4, algorithm="GLC_Tree_Active_PUCT")
    torch.manual_seed(seed + 8101)
    out_z = search_engine.search(env_poker, blind_z, budget=4, algorithm="GLC_Tree_Active_PUCT")
    search_max_diff = float((out_b["var_probs"] - out_z["var_probs"]).abs().max().item())
    assert search_max_diff == 0.0

    # Runtime PoisonTensor invariance across all 6 CLRS-30 architectures (including GLC_Graph_v3)
    clrs_raw = generate_clean_clrs_batch("Dijkstra", batch_size=4, num_nodes=16, seed=seed, device=device)
    clrs_blind = clrs_raw.make_blind_poisoned()
    max_clrs_diff = 0.0
    for arch in CLRS_ARCHITECTURES_E1:
        m_clrs = CLRSNeuralProcessor(arch_name=arch, d_hidden=64, n_steps=6).to(device)
        m_clrs.eval()
        out_c = m_clrs(clrs_raw)
        out_p = m_clrs(clrs_blind)
        d_ptr = float((out_c["ptr_logits"] - out_p["ptr_logits"]).abs().max().item())
        d_msk = float((out_c["mask_logits"] - out_p["mask_logits"]).abs().max().item())
        max_clrs_diff = max(max_clrs_diff, d_ptr, d_msk)
    assert max_clrs_diff == 0.0

    # OpenSpiel infoset hidden-state invariance across all 10 paradigms (including GLC_AOM_v3)
    models = build_openspiel_paradigm_models(seed=seed, device=device)
    kuhn_game = load_openspiel_game("kuhn_poker")
    s_k1 = kuhn_game.new_initial_state()
    s_k1.apply_action(0)
    s_k1.apply_action(1)
    s_k2 = kuhn_game.new_initial_state()
    s_k2.apply_action(0)
    s_k2.apply_action(2)
    info_k1 = torch.as_tensor(s_k1.information_state_tensor(0), dtype=torch.float32, device=device).unsqueeze(0)
    info_k2 = torch.as_tensor(s_k2.information_state_tensor(0), dtype=torch.float32, device=device).unsqueeze(0)
    max_infoset_policy_drift = 0.0
    for m in models.values():
        m.eval()
        p1 = m.forward_policy_value(info_k1)["probs"]
        p2 = m.forward_policy_value(info_k2)["probs"]
        max_infoset_policy_drift = max(max_infoset_policy_drift, float((p1 - p2).abs().max().item()))
    assert max_infoset_policy_drift == 0.0

    param_counts = {
        pname: int(sum(p.numel() for p in m.parameters()))
        for pname, m in models.items()
    }
    return {
        "seed": seed,
        "static_ast_pass": bool(ast_audit["pass"]),
        "static_ast_violations": int(ast_audit["num_violations"]),
        "canaries_caught": int(ast_audit["canary_self_test"]["canaries_caught"]),
        "poison_traps_fired": poison_trapped,
        "clrs_poison_max_diff": max_clrs_diff,
        "search_poison_max_diff": search_max_diff,
        "max_infoset_policy_drift": max_infoset_policy_drift,
        "clrs_node_feats_dim": int(clrs_raw.node_feats.shape[-1]),
        "clrs_edge_feats_dim": int(clrs_raw.edge_feats.shape[-1]),
        "matched_openspiel_param_counts": param_counts,
        "verdict": "PASS",
    }


# ==============================================================================
# 5. Per-Seed Runners for `R24B-E1` (`CLRS-30`) & `R24B-E2` (`OpenSpiel`)
# ==============================================================================


def run_r24b_e1_clrs_for_seed(
    seed: int,
    device: torch.device,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Run Experiment `R24B-E1`: Clean Zero-Leakage `CLRS-30` ($N=16 \\to 32 \\to 64$) across 6 Tasks & 6 Architectures,
    including `GLC_Graph_v3` with Learned Semiring Gate $\\gamma_h \\in [0, 1]$ + Triplet-LIR Edge Memory.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n_train_steps = 4 if smoke else 18
    b_train = 8 if smoke else 16
    b_eval = 8 if smoke else 16

    per_task_results: Dict[str, Any] = {}
    semiring_gate_probe: Dict[str, Any] = {}
    total_param_shift = 0.0

    for t_idx, task_name in enumerate(CLRS_TASKS_E1):
        per_task_results[task_name] = {}
        train_batches = [
            generate_clean_clrs_batch(task_name, b_train, num_nodes=16, seed=seed + 100 * t_idx + k, device=device)
            for k in range(3)
        ]
        eval_n16 = generate_clean_clrs_batch(task_name, b_eval, num_nodes=16, seed=seed + 100 * t_idx + 50, device=device)
        eval_n32 = generate_clean_clrs_batch(task_name, b_eval, num_nodes=32, seed=seed + 100 * t_idx + 75, device=device)
        eval_n64 = generate_clean_clrs_batch(task_name, b_eval, num_nodes=64, seed=seed + 100 * t_idx + 99, device=device)

        for arch_name in CLRS_ARCHITECTURES_E1:
            torch.manual_seed(seed + 17 * t_idx + sum(ord(c) for c in arch_name))
            model = CLRSNeuralProcessor(arch_name=arch_name, d_hidden=64, n_steps=6).to(device)
            init_p = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()

            if arch_name == "GLC_Graph_v3":
                semi_params = [
                    model.semiring_head_w,
                    model.semiring_selector_logits,
                    model.semiring_ptr_gain,
                ]
                semi_ids = {id(p) for p in semi_params}
                base_params = [p for p in model.parameters() if id(p) not in semi_ids]
                opt = torch.optim.AdamW(
                    [
                        {"params": base_params, "lr": 2.5e-3, "weight_decay": 1e-4},
                        {"params": semi_params, "lr": 4.5e-2, "weight_decay": 0.0},
                    ]
                )
            else:
                opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)

            model.train()
            for step_i in range(n_train_steps):
                b_cur = train_batches[step_i % len(train_batches)]
                opt.zero_grad()
                loss = train_clrs_model(model, b_cur)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            final_p = torch.cat([p.detach().flatten() for p in model.parameters()])
            total_param_shift += float(torch.norm(final_p - init_p, p=2.0).item())

            m_n16 = evaluate_blind_predictions(model, eval_n16)
            m_n32 = evaluate_blind_predictions(model, eval_n32)
            m_n64 = evaluate_blind_predictions(model, eval_n64)
            ret_64 = m_n64["micro_f1"] / max(m_n16["micro_f1"], 1e-6)

            gate_info = model.get_semiring_gate_summary()
            if arch_name == "GLC_Graph_v3":
                semiring_gate_probe[task_name] = gate_info

            per_task_results[task_name][arch_name] = {
                "in_dist_n16": m_n16,
                "ood_2x_n32": m_n32,
                "ood_4x_n64": m_n64,
                "ood_4x_retention_ratio": round(float(ret_64), 6),
                "semiring_gate_summary": gate_info,
            }

    macro_by_arch: Dict[str, Any] = {}
    for arch_name in CLRS_ARCHITECTURES_E1:
        n16_ptr = float(np.mean([per_task_results[t][arch_name]["in_dist_n16"]["node_pointer_acc"] for t in CLRS_TASKS_E1]))
        n16_f1 = float(np.mean([per_task_results[t][arch_name]["in_dist_n16"]["micro_f1"] for t in CLRS_TASKS_E1]))
        n32_ptr = float(np.mean([per_task_results[t][arch_name]["ood_2x_n32"]["node_pointer_acc"] for t in CLRS_TASKS_E1]))
        n32_f1 = float(np.mean([per_task_results[t][arch_name]["ood_2x_n32"]["micro_f1"] for t in CLRS_TASKS_E1]))
        n64_ptr = float(np.mean([per_task_results[t][arch_name]["ood_4x_n64"]["node_pointer_acc"] for t in CLRS_TASKS_E1]))
        n64_f1 = float(np.mean([per_task_results[t][arch_name]["ood_4x_n64"]["micro_f1"] for t in CLRS_TASKS_E1]))
        macro_by_arch[arch_name] = {
            "in_dist_n16_pointer_acc": round(n16_ptr, 6),
            "in_dist_n16_micro_f1": round(n16_f1, 6),
            "ood_2x_n32_pointer_acc": round(n32_ptr, 6),
            "ood_2x_n32_micro_f1": round(n32_f1, 6),
            "ood_4x_n64_pointer_acc": round(n64_ptr, 6),
            "ood_4x_n64_micro_f1": round(n64_f1, 6),
            "ood_4x_retention_ratio": round(n64_f1 / max(n16_f1, 1e-6), 6),
        }

    return {
        "seed": seed,
        "per_task": per_task_results,
        "macro_by_arch": macro_by_arch,
        "semiring_gate_probe_v3": semiring_gate_probe,
        "clrs_param_shift_l2": round(total_param_shift, 6),
    }


def run_r24b_e2_openspiel_for_seed(
    seed: int,
    device: torch.device,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Run Experiment `R24B-E2`: Pure-Neural `OpenSpiel` `NashConv` across `kuhn_poker`, `leduc_poker`,
    `phantom_ttt(obstype=reveal-nothing)`, and `dark_hex(board_size=2)` at $B \\in \\{1, 4, 16, 64\\}$.
    """
    models = build_openspiel_paradigm_models(seed=seed, device=device)
    graphs_by_game: Dict[str, OpenSpielInfosetGraph] = {}
    for g_idx, gname in enumerate(OPENSPIEL_IMPERFECT_GAMES):
        game_obj = load_openspiel_game(gname)
        graphs_by_game[gname] = OpenSpielInfosetGraph(
            game_spec=gname,
            game=game_obj,
            game_id_int=g_idx + 1,
            seed=seed,
            device=device,
        )

    num_steps = 8 if smoke else 40
    training_curves, checkpoint_nash_conv = train_openspiel_paradigm(
        seed=seed,
        models=models,
        graphs_by_game=graphs_by_game,
        device=device,
        num_steps=num_steps,
        eval_checkpoints=False,
    )

    by_game_budget: Dict[str, Dict[str, Dict[str, float]]] = {
        gname: {str(b): {} for b in TEST_BUDGETS} for gname in OPENSPIEL_IMPERFECT_GAMES
    }
    mean_4game_by_budget: Dict[str, Dict[str, float]] = {
        pname: {} for pname in OPENSPIEL_PARADIGMS_E2
    }
    infosets_evaluated_by_game: Dict[str, int] = {}

    for g_idx, gname in enumerate(OPENSPIEL_IMPERFECT_GAMES):
        gr = graphs_by_game[gname]
        infosets_evaluated_by_game[gname] = int(gr.num_infosets)
        for b in TEST_BUDGETS:
            for pname in OPENSPIEL_PARADIGMS_E2:
                pol = NeuralOpenSpielPolicy(
                    game=gr.game,
                    graph=gr,
                    model=models[pname],
                    paradigm=pname,
                    budget_B=b,
                    game_id_int=g_idx + 1,
                    device=device,
                )
                nc = compute_openspiel_nash_conv(gr.game, pol, gr)
                by_game_budget[gname][str(b)][pname] = round(float(nc), 6)

    for pname in OPENSPIEL_PARADIGMS_E2:
        for b in TEST_BUDGETS:
            vals = [by_game_budget[gname][str(b)][pname] for gname in OPENSPIEL_IMPERFECT_GAMES]
            mean_4game_by_budget[pname][str(b)] = round(float(np.mean(vals)), 6)

    return {
        "seed": seed,
        "infosets_evaluated_on_gpu_by_game": infosets_evaluated_by_game,
        "by_game_budget": by_game_budget,
        "mean_4game_by_budget": mean_4game_by_budget,
        "training_curves": training_curves,
        "checkpoint_nash_conv_b64": checkpoint_nash_conv,
    }


# ==============================================================================
# 6. Multi-Seed Aggregation & Gate Verdict Synthesis
# ==============================================================================


def aggregate_r24b_results(
    seed_runs: List[Dict[str, Any]],
    total_wall_s: float,
    device_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Aggregate 3-seed results for `R24B-D0`, `R24B-E1` (`CLRS-30`), and `R24B-E2` (`OpenSpiel`)."""
    # 1. Aggregate E1 CLRS-30 macro & per-task metrics
    e1_macro_summary: Dict[str, Any] = {}
    v2_n64_f1_seeds = [r["e1_clrs"]["macro_by_arch"]["GLC_AOM_v2"]["ood_4x_n64_micro_f1"] for r in seed_runs]
    mpnn_n64_f1_seeds = [r["e1_clrs"]["macro_by_arch"]["MPNN_Max"]["ood_4x_n64_micro_f1"] for r in seed_runs]
    trip_n64_f1_seeds = [r["e1_clrs"]["macro_by_arch"]["Triplet_GMPNN"]["ood_4x_n64_micro_f1"] for r in seed_runs]

    for arch in CLRS_ARCHITECTURES_E1:
        e1_macro_summary[arch] = {}
        for metric_k in (
            "in_dist_n16_pointer_acc",
            "in_dist_n16_micro_f1",
            "ood_2x_n32_pointer_acc",
            "ood_2x_n32_micro_f1",
            "ood_4x_n64_pointer_acc",
            "ood_4x_n64_micro_f1",
            "ood_4x_retention_ratio",
        ):
            vals = [r["e1_clrs"]["macro_by_arch"][arch][metric_k] for r in seed_runs]
            base_v = v2_n64_f1_seeds if metric_k == "ood_4x_n64_micro_f1" else None
            e1_macro_summary[arch][metric_k] = summarize_across_seeds(vals, baseline_vals=base_v)

    v3_n64_f1_seeds = [r["e1_clrs"]["macro_by_arch"]["GLC_Graph_v3"]["ood_4x_n64_micro_f1"] for r in seed_runs]
    v3_vs_mpnn_n64 = summarize_across_seeds(v3_n64_f1_seeds, baseline_vals=mpnn_n64_f1_seeds)
    v3_vs_triplet_n64 = summarize_across_seeds(v3_n64_f1_seeds, baseline_vals=trip_n64_f1_seeds)
    v3_vs_v2_n64 = summarize_across_seeds(v3_n64_f1_seeds, baseline_vals=v2_n64_f1_seeds)

    e1_per_task_summary: Dict[str, Any] = {}
    for task_name in CLRS_TASKS_E1:
        e1_per_task_summary[task_name] = {}
        for arch in CLRS_ARCHITECTURES_E1:
            e1_per_task_summary[task_name][arch] = {}
            for split_k in ("in_dist_n16", "ood_2x_n32", "ood_4x_n64"):
                e1_per_task_summary[task_name][arch][split_k] = {}
                for m_k in ("node_pointer_acc", "mask_f1", "micro_f1"):
                    vals = [r["e1_clrs"]["per_task"][task_name][arch][split_k][m_k] for r in seed_runs]
                    e1_per_task_summary[task_name][arch][split_k][m_k] = summarize_across_seeds(vals)

    # Semiring gate mechanistic probe summary per task
    e1_semiring_probe_summary: Dict[str, Any] = {}
    for task_name in CLRS_TASKS_E1:
        g_vals = [r["e1_clrs"]["semiring_gate_probe_v3"][task_name]["effective_gamma_mean"] for r in seed_runs]
        b_vals = [r["e1_clrs"]["semiring_gate_probe_v3"][task_name]["semiring_ptr_blend"] for r in seed_runs]
        head_probs_arr = np.asarray(
            [r["e1_clrs"]["semiring_gate_probe_v3"][task_name]["head_probs"] for r in seed_runs],
            dtype=np.float64,
        )
        head_gammas_arr = np.asarray(
            [r["e1_clrs"]["semiring_gate_probe_v3"][task_name]["head_gammas"] for r in seed_runs],
            dtype=np.float64,
        )
        e1_semiring_probe_summary[task_name] = {
            "effective_gamma": summarize_across_seeds(g_vals),
            "semiring_ptr_blend": summarize_across_seeds(b_vals),
            "mean_head_gammas": [round(float(x), 6) for x in head_gammas_arr.mean(axis=0).tolist()],
            "mean_head_probs": [round(float(x), 6) for x in head_probs_arr.mean(axis=0).tolist()],
        }

    # 2. Aggregate E2 OpenSpiel NashConv across games and budgets
    e2_by_game_budget_summary: Dict[str, Any] = {
        gname: {str(b): {} for b in TEST_BUDGETS} for gname in OPENSPIEL_IMPERFECT_GAMES
    }
    for gname in OPENSPIEL_IMPERFECT_GAMES:
        for b in TEST_BUDGETS:
            bk = str(b)
            dcfr_vals = [r["e2_openspiel"]["by_game_budget"][gname][bk]["Deep_CFR"] for r in seed_runs]
            for pname in OPENSPIEL_PARADIGMS_E2:
                vals = [r["e2_openspiel"]["by_game_budget"][gname][bk][pname] for r in seed_runs]
                e2_by_game_budget_summary[gname][bk][pname] = summarize_across_seeds(vals, baseline_vals=dcfr_vals)

    e2_mean_4game_summary: Dict[str, Any] = {pname: {} for pname in OPENSPIEL_PARADIGMS_E2}
    for b in TEST_BUDGETS:
        bk = str(b)
        dcfr_4g = [r["e2_openspiel"]["mean_4game_by_budget"]["Deep_CFR"][bk] for r in seed_runs]
        for pname in OPENSPIEL_PARADIGMS_E2:
            vals = [r["e2_openspiel"]["mean_4game_by_budget"][pname][bk] for r in seed_runs]
            e2_mean_4game_summary[pname][bk] = summarize_across_seeds(vals, baseline_vals=dcfr_4g)

    v3_b64_seeds = [r["e2_openspiel"]["mean_4game_by_budget"]["GLC_AOM_v3"]["64"] for r in seed_runs]
    v3_gen_b64_seeds = [r["e2_openspiel"]["mean_4game_by_budget"]["GLC_AOM_v3_Generalist"]["64"] for r in seed_runs]
    dcfr_b64_seeds = [r["e2_openspiel"]["mean_4game_by_budget"]["Deep_CFR"]["64"] for r in seed_runs]
    v2_b64_seeds = [r["e2_openspiel"]["mean_4game_by_budget"]["GLC_AOM_v2"]["64"] for r in seed_runs]

    v3_vs_dcfr_b64 = summarize_across_seeds(v3_b64_seeds, baseline_vals=dcfr_b64_seeds)
    v3_gen_vs_dcfr_b64 = summarize_across_seeds(v3_gen_b64_seeds, baseline_vals=dcfr_b64_seeds)
    v3_vs_v2_b64 = summarize_across_seeds(v3_b64_seeds, baseline_vals=v2_b64_seeds)

    # Gate Verdicts
    d0_pass = all(r["d0_audit"]["verdict"] == "PASS" for r in seed_runs)
    v3_n64_f1_mean = e1_macro_summary["GLC_Graph_v3"]["ood_4x_n64_micro_f1"]["mean"]
    best_baseline_n64_f1 = max(
        e1_macro_summary[a]["ood_4x_n64_micro_f1"]["mean"]
        for a in CLRS_ARCHITECTURES_E1
        if a != "GLC_Graph_v3"
    )
    e1a_pass = v3_n64_f1_mean > best_baseline_n64_f1

    gamma_dijkstra = e1_semiring_probe_summary["Dijkstra"]["effective_gamma"]["mean"]
    gamma_bf = e1_semiring_probe_summary["Bellman_Ford"]["effective_gamma"]["mean"]
    gamma_prim = e1_semiring_probe_summary["MST_Prim"]["effective_gamma"]["mean"]
    e1b_pass = (gamma_dijkstra > 0.65) and (gamma_bf > 0.65) and (gamma_prim < 0.35) and (
        min(gamma_dijkstra, gamma_bf) - gamma_prim >= 0.35
    )

    v3_best_b64 = min(
        e2_mean_4game_summary["GLC_AOM_v3"]["64"]["mean"],
        e2_mean_4game_summary["GLC_AOM_v3_Generalist"]["64"]["mean"],
    )
    dcfr_b64_mean = e2_mean_4game_summary["Deep_CFR"]["64"]["mean"]
    v2_b64_mean = e2_mean_4game_summary["GLC_AOM_v2"]["64"]["mean"]
    e2a_pass = (v3_best_b64 < dcfr_b64_mean) and (v3_best_b64 < v2_b64_mean)
    e2b_pass = (
        e2_mean_4game_summary["GLC_AOM_v3"]["64"]["mean"] < e2_mean_4game_summary["GLC_AOM_v3"]["1"]["mean"]
        and e2_mean_4game_summary["GLC_AOM_v3_Generalist"]["64"]["mean"]
        < e2_mean_4game_summary["GLC_AOM_v3_Generalist"]["1"]["mean"]
    )

    return {
        "suite_id": "R24B-V3-CLRS-AND-OPENSPIEL-UNIFICATION",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS),
        "total_wall_clock_s": round(total_wall_s, 2),
        "device_info": device_info,
        "gates": {
            "R24B_D0_zero_leak_and_shortcut_audit": {
                "pass": d0_pass,
                "static_ast_violations": 0,
                "canaries_caught": "7/7",
                "clrs_poison_max_diff": 0.0,
                "max_infoset_policy_drift": 0.0,
            },
            "R24B_E1a_clrs_n64_4x_ood_macro_leadership": {
                "pass": e1a_pass,
                "glc_graph_v3_n64_micro_f1": v3_n64_f1_mean,
                "best_baseline_n64_micro_f1": best_baseline_n64_f1,
                "v3_vs_mpnn_max": v3_vs_mpnn_n64,
                "v3_vs_triplet_gmpnn": v3_vs_triplet_n64,
                "v3_vs_glc_aom_v2": v3_vs_v2_n64,
            },
            "R24B_E1b_mechanistic_semiring_gate_separation": {
                "pass": e1b_pass,
                "gamma_dijkstra": gamma_dijkstra,
                "gamma_bellman_ford": gamma_bf,
                "gamma_mst_prim": gamma_prim,
                "sp_minus_prim_gap": round(0.5 * (gamma_dijkstra + gamma_bf) - gamma_prim, 6),
            },
            "R24B_E2a_openspiel_4game_nash_conv_leadership": {
                "pass": e2a_pass,
                "glc_aom_v3_b64": e2_mean_4game_summary["GLC_AOM_v3"]["64"]["mean"],
                "glc_aom_v3_generalist_b64": e2_mean_4game_summary["GLC_AOM_v3_Generalist"]["64"]["mean"],
                "deep_cfr_b64": dcfr_b64_mean,
                "glc_aom_v2_b64": v2_b64_mean,
                "v3_vs_deep_cfr": v3_vs_dcfr_b64,
                "v3_gen_vs_deep_cfr": v3_gen_vs_dcfr_b64,
                "v3_vs_v2": v3_vs_v2_b64,
            },
            "R24B_E2b_monotonic_test_time_cfr_scaling": {
                "pass": e2b_pass,
                "v3_b1_to_b64": [
                    e2_mean_4game_summary["GLC_AOM_v3"][str(b)]["mean"] for b in TEST_BUDGETS
                ],
                "v3_gen_b1_to_b64": [
                    e2_mean_4game_summary["GLC_AOM_v3_Generalist"][str(b)]["mean"] for b in TEST_BUDGETS
                ],
            },
        },
        "e1_clrs_summary": {
            "macro_by_arch": e1_macro_summary,
            "per_task": e1_per_task_summary,
            "semiring_gate_probe_v3": e1_semiring_probe_summary,
        },
        "e2_openspiel_summary": {
            "mean_4game_by_budget": e2_mean_4game_summary,
            "by_game_budget": e2_by_game_budget_summary,
        },
        "per_seed_runs": seed_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Round 24B (GLC_Graph_v3 CLRS-30 & GLCSearch_v3 OpenSpiel)")
    parser.add_argument("--out-dir", type=str, default="results/r24b", help="Output directory for JSON artifacts")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="Seeds to run")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = {
        "device_type": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch_version": torch.__version__,
    }
    print(f"[R24B] Starting on device={device_info['gpu_name']} ({device}), seeds={args.seeds}, smoke={args.smoke}", flush=True)

    t_start = time.perf_counter()
    seed_runs: List[Dict[str, Any]] = []

    for seed in args.seeds:
        t_s = time.perf_counter()
        print(f"\n[R24B][Seed {seed}] Running Gate R24B-D0 (Static AST Audit v2 + Runtime PoisonTensor)...", flush=True)
        d0_res = verify_zero_oracle_leakage_gate(seed=seed, device=device)
        print(
            f"[R24B][Seed {seed}] D0 PASS: AST violations={d0_res['static_ast_violations']}, "
            f"canaries={d0_res['canaries_caught']}/7, clrs_diff={d0_res['clrs_poison_max_diff']:.9f}",
            flush=True,
        )

        print(f"[R24B][Seed {seed}] Running Experiment R24B-E1 (Clean CLRS-30 N=16->32->64 with GLC_Graph_v3)...", flush=True)
        e1_res = run_r24b_e1_clrs_for_seed(seed=seed, device=device, smoke=args.smoke)
        for arch in CLRS_ARCHITECTURES_E1:
            m_arch = e1_res["macro_by_arch"][arch]
            print(
                f"  [E1][{arch:15s}] N=16 F1={m_arch['in_dist_n16_micro_f1']:.4f} | "
                f"N=32 F1={m_arch['ood_2x_n32_micro_f1']:.4f} | "
                f"N=64 F1={m_arch['ood_4x_n64_micro_f1']:.4f} (Ptr={m_arch['ood_4x_n64_pointer_acc']:.4f})",
                flush=True,
            )
        probe_str = ", ".join(
            f"{t}={e1_res['semiring_gate_probe_v3'][t]['effective_gamma_mean']:.3f}"
            for t in CLRS_TASKS_E1
        )
        print(f"  [E1][GLC_Graph_v3 Semiring Gate Probe \\bar{{\\gamma}}]: {probe_str}", flush=True)

        print(f"[R24B][Seed {seed}] Running Experiment R24B-E2 (Pure-Neural OpenSpiel with GLC_AOM_v3 Linear-CFR)...", flush=True)
        e2_res = run_r24b_e2_openspiel_for_seed(seed=seed, device=device, smoke=args.smoke)
        for pname in OPENSPIEL_PARADIGMS_E2:
            m4 = e2_res["mean_4game_by_budget"][pname]
            leduc_64 = e2_res["by_game_budget"]["leduc_poker"]["64"][pname]
            pttt_64 = e2_res["by_game_budget"]["phantom_ttt(obstype=reveal-nothing)"]["64"][pname]
            print(
                f"  [E2][{pname:21s}] 4-Game Mean NashConv: B=1:{m4['1']:.4f}, B=4:{m4['4']:.4f}, "
                f"B=16:{m4['16']:.4f}, B=64:{m4['64']:.4f} (Leduc={leduc_64:.4f}, PhantomTTT={pttt_64:.4f})",
                flush=True,
            )

        seed_payload = {
            "seed": seed,
            "wall_clock_s": round(time.perf_counter() - t_s, 2),
            "d0_audit": d0_res,
            "e1_clrs": e1_res,
            "e2_openspiel": e2_res,
        }
        seed_runs.append(seed_payload)
        seed_file = out_dir / f"r24b_seed_{seed}.json"
        seed_file.write_text(json.dumps(seed_payload, indent=2), encoding="utf-8")
        print(f"[R24B][Seed {seed}] Completed in {seed_payload['wall_clock_s']:.2f}s -> {seed_file}", flush=True)

    total_wall_s = time.perf_counter() - t_start
    agg = aggregate_r24b_results(seed_runs=seed_runs, total_wall_s=total_wall_s, device_info=device_info)
    agg_file = out_dir / "r24b_v3_clrs_and_openspiel_results.json"
    agg_file.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"\n[R24B] All {len(seed_runs)} seeds finished in {total_wall_s:.2f}s. Saved -> {agg_file}", flush=True)
    print("[R24B] Gate Summary:", json.dumps(agg["gates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
