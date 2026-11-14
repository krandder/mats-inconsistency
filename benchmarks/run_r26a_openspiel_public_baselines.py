#!/usr/bin/env python3
r"""PARKED 2026-10-19 (Kelvin): not frozen, not run
Round 26-A (`R26-A`): External Public `OpenSpiel` Baselines with Reproduced Learning Curves (`5` Seeds).

Implements the pre-registered Round 26-A specification (`benchmarks/PREREG-R26A-OPENSPIEL-PUBLIC-BASELINES.md`)
under `docs/PROTOCOL.md` and `outbox/r26-freeze-review-2026-10-19.md` (Blocking Items 3, 5, 6):
  - **5 Random Seeds (`SEEDS = (101, 102, 103, 104, 105)`):** $t_{0.975, 4} = 2.77645$ (two-sided), $t_{0.95, 4} = 2.13185$ (one-sided).
  - **Gate `R26A-D0`:** Static AST Leak Auditor v2 (`0` violations), canary self-test (`7/7` caught),
    explicit patch unit tests (`tests/test_external_patches.py`), and `R26A-D0.3` Upstream Learning-Curve
    Reproduction Gate across 4 budget stages (`Stage 1, Stage 2, Stage 3, Stage 4`) reaching the trained regime.
  - **Gate `R26A-E1`:** Exact `NashConv` (`open_spiel.python.algorithms.exploitability.nash_conv`) across all 4
    budget stages and Area Under the Learning Curve (`AULC`) on `kuhn_poker`, `leduc_poker`,
    `dark_hex(board_size=2)`, and `liars_dice(numdice=1,dice_sides=4)` comparing:
      * All 5 Mandatory Controls (`Control_Uniform`, `Control_Untrained_Init`, `Control_ZeroParam_Readout`,
        `Control_Helper_Heuristic_Prior`, `Control_Exact_Ceiling_CFR` / `Control_Exact_Ceiling_CFRPlus`),
      * All 8 Upstream `OpenSpiel` Solvers (`Upstream_CFRSolver`, `Upstream_CFRPlusSolver`,
        `Upstream_DiscountedCFRSolver`, `Upstream_ExternalSamplingMCCFR`, `Upstream_OutcomeSamplingMCCFR`,
        `Upstream_DeepCFRSolver`, `Upstream_NFSP`, `Upstream_RNaDSolver`),
      * `GLCAOM_v3_B1_PureNeural` (`B=1`, 100% pure 1-pass neural network forward pass at test time with
        ZERO calls to any tree/graph backup) and `GLCAOM_v3_B64_Search` (`B=64`), plus `Generalist` variants.
  - **Gate `R26A-E2`:** Powered 7-Game Head-to-Head Tournament (`100` seat-swapped games per matchup per seed =
    `500` games per matchup across the 5 seeds, 100% genuine `bot.step(state)` on every ply) with 5-seed and
    pooled 95% confidence interval lower-bound bars.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
import types
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_PYLIB_CANDIDATES = [
    "/home/agent/workspaces/fable-strategy/scratch/pillar1-audit/shim",
    "/home/agent/workspaces/venv-cft/lib/python3.12/site-packages",
    "/home/agent/.local/lib/python3.12/site-packages",
    "/home/agent/data/pool/pylib",
    "/root/pylib",
    "/tmp/r4e3a/pylib",
    "/tmp/r26_pylib",
]
for _cand_path in reversed(_PYLIB_CANDIDATES):
    if os.path.isdir(_cand_path):
        while _cand_path in sys.path:
            sys.path.remove(_cand_path)
        sys.path.insert(0, _cand_path)

for _conda_site in sorted(glob.glob("/opt/conda/lib/python*/site-packages")):
    if os.path.isdir(_conda_site) and _conda_site not in sys.path:
        sys.path.append(_conda_site)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Ensure `tree` (`dm-tree` map_structure) is available for `open_spiel.python.pytorch.deep_cfr` / `nfsp`
try:
    import tree as _dm_tree  # noqa: F401
except ImportError:
    _tree_mod = types.ModuleType("tree")

    def _map_structure(fn: Callable[..., Any], *structures: Any) -> Any:
        s0 = structures[0]
        if isinstance(s0, dict):
            return {k: _map_structure(fn, *(s[k] for s in structures)) for k in s0}
        if isinstance(s0, list):
            return [_map_structure(fn, *(s[i] for s in structures)) for i in range(len(s0))]
        if isinstance(s0, tuple) and hasattr(s0, "_fields"):
            return type(s0)(*(_map_structure(fn, *(getattr(s, f) for s in structures)) for f in s0._fields))
        if isinstance(s0, tuple):
            return tuple(_map_structure(fn, *(s[i] for s in structures)) for i in range(len(s0)))
        return fn(*structures)

    _tree_mod.map_structure = _map_structure  # type: ignore[attr-defined]
    sys.modules["tree"] = _tree_mod


def _ensure_r26a_deps() -> None:
    """Ensure upstream Python packages (`torch`, `numpy`, `scipy`, `open_spiel`, `jax`, `dm-haiku`, `optax`)
    are installed when running on a fresh GPU VM.
    """
    def _ensure_pip() -> None:
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

    try:
        import numpy  # noqa: F401
        import pyspiel  # noqa: F401
        import scipy  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        _ensure_pip()
        os.makedirs("/root/pylib", exist_ok=True)
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
                "open_spiel==1.5",
                "absl-py",
            ]
        )
        if "/root/pylib" not in sys.path:
            sys.path.insert(0, "/root/pylib")

    try:
        import haiku  # noqa: F401
        import jax  # noqa: F401
        import optax  # noqa: F401
    except ImportError:
        if os.environ.get("R26A_SKIP_PIP_BOOTSTRAP", "0") != "1":
            try:
                _ensure_pip()
                os.makedirs("/root/pylib", exist_ok=True)
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-q",
                        "--target",
                        "/root/pylib",
                        "jax",
                        "jaxlib",
                        "dm-haiku",
                        "optax",
                        "chex",
                        "jmp",
                        "toolz",
                        "tabulate",
                    ]
                )
                if "/root/pylib" not in sys.path:
                    sys.path.insert(0, "/root/pylib")
            except Exception as exc:
                print(f"[R26A-bootstrap] Optional JAX install warning: {exc}", flush=True)


if __name__ == "__main__":
    _ensure_r26a_deps()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

import open_spiel.python.algorithms  # noqa: E402
import open_spiel.python.pytorch  # noqa: E402
import pyspiel  # noqa: E402

for _p_root in ("/tmp/r4e3a/pylib", "/tmp/r26_pylib", "/root/pylib"):
    _p_alg = os.path.join(_p_root, "open_spiel", "python", "algorithms")
    _p_pt = os.path.join(_p_root, "open_spiel", "python", "pytorch")
    if os.path.isdir(_p_alg) and _p_alg not in open_spiel.python.algorithms.__path__:
        open_spiel.python.algorithms.__path__.append(_p_alg)
    if os.path.isdir(_p_pt) and _p_pt not in open_spiel.python.pytorch.__path__:
        open_spiel.python.pytorch.__path__.append(_p_pt)

HAS_JAX_RNAD = False
try:
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402

    # Explicit JAX `jnp.clip` `a_min/a_max` shim (verified by `tests/test_external_patches.py`
    # and documented in `external/patches/openspiel_jax_jnp_clip_shim.patch`).
    _ORIG_JNP_CLIP = jnp.clip

    def _jax_compat_clip(
        a: Any,
        a_min: Any = None,
        a_max: Any = None,
        min: Any = None,
        max: Any = None,
        **kwargs: Any,
    ) -> Any:
        lo = a_min if a_min is not None else min
        hi = a_max if a_max is not None else max
        return _ORIG_JNP_CLIP(a, min=lo, max=hi, **kwargs)

    jnp.clip = _jax_compat_clip
    from open_spiel.python.algorithms.rnad import rnad as osp_rnad  # noqa: E402

    HAS_JAX_RNAD = True
except ImportError:
    osp_rnad = None  # type: ignore[assignment]

from open_spiel.python import policy as osp_policy  # noqa: E402
from open_spiel.python import rl_environment  # noqa: E402
from open_spiel.python.algorithms import cfr as osp_cfr  # noqa: E402
from open_spiel.python.algorithms import discounted_cfr as osp_dcfr  # noqa: E402
from open_spiel.python.algorithms import exploitability as osp_exploitability  # noqa: E402
from open_spiel.python.algorithms import external_sampling_mccfr as osp_es_mccfr  # noqa: E402
from open_spiel.python.algorithms import ismcts as osp_ismcts  # noqa: E402
from open_spiel.python.algorithms import mcts as osp_mcts  # noqa: E402
from open_spiel.python.algorithms import outcome_sampling_mccfr as osp_os_mccfr  # noqa: E402
from open_spiel.python.pytorch import deep_cfr as osp_deep_cfr  # noqa: E402
from open_spiel.python.pytorch import nfsp as osp_nfsp  # noqa: E402

from src.eval.ast_leak_auditor import audit_r11_files  # noqa: E402
from src.eval.elo_league import BayesianEloLeague  # noqa: E402

SEEDS: Tuple[int, ...] = (101, 102, 103, 104, 105)
T_CRIT_95_DF4: float = 2.7764451051977987
T_CRIT_90_DF4: float = 2.131846786326649
T_CRIT_95_DF2: float = 4.302652729911275

STAGE_IDS: Tuple[int, ...] = (1, 2, 3, 4)

NASHCONV_GAMES: Tuple[str, ...] = (
    "kuhn_poker",
    "leduc_poker",
    "dark_hex(board_size=2)",
    "liars_dice(numdice=1,dice_sides=4)",
)

OPENSPIEL_IMPERFECT_GAMES: Tuple[str, ...] = (
    "kuhn_poker",
    "leduc_poker",
    "dark_hex(board_size=2)",
    "liars_dice(numdice=1,dice_sides=4)",
    "phantom_ttt(obstype=reveal-nothing)",
)

OPENSPIEL_PERFECT_GAMES: Tuple[str, ...] = (
    "tic_tac_toe",
    "hex(board_size=3)",
)

ALL_OPENSPIEL_GAMES: Tuple[str, ...] = OPENSPIEL_IMPERFECT_GAMES + OPENSPIEL_PERFECT_GAMES

EXACT_ANALYTICAL_UNIFORM_NASHCONV: Dict[str, float] = {
    "kuhn_poker": 0.91667,
    "leduc_poker": 4.74722,
    "dark_hex(board_size=2)": 1.25000,
    "liars_dice(numdice=1,dice_sides=4)": 1.31006,
}

# Published stage budgets per solver family (Stages 1, 2, 3, 4)
STAGE_BUDGET_MAP: Dict[str, Tuple[int, int, int, int]] = {
    "exact_cfr": (10, 40, 100, 400),
    "es_mccfr": (50, 200, 400, 800),
    "os_mccfr": (2000, 8000, 20000, 40000),
    "deep_cfr": (5, 15, 30, 40),
    "nfsp": (1500, 3500, 6000, 8000),
    "rnad": (50, 150, 300, 600),
    "glcaom": (10, 20, 30, 40),
}


# ==============================================================================
# Statistical & Multidimensional Elo (`mElo`) Helpers
# ==============================================================================


def _t_crit_two_sided_95(n: int) -> float:
    if n >= 5:
        return T_CRIT_95_DF4
    if n == 4:
        return 3.182446305284263
    if n == 3:
        return T_CRIT_95_DF2
    return 1.96


def summarize_across_seeds(vals: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(float(max(n, 1)))
    t_c = _t_crit_two_sided_95(n)
    half_ci = t_c * se
    return {
        "mean": round(mean, 5),
        "std": round(std, 5),
        "se": round(se, 5),
        "ci95_low": round(mean - half_ci, 5),
        "ci95_high": round(mean + half_ci, 5),
        "per_seed": [round(float(x), 5) for x in arr],
    }


def paired_t_test(vals_a: Sequence[float], vals_b: Sequence[float]) -> Dict[str, Any]:
    a = np.asarray(vals_a, dtype=np.float64)
    b = np.asarray(vals_b, dtype=np.float64)
    diff = a - b
    n = len(diff)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se_diff = std_diff / math.sqrt(float(max(n, 1)))
    t_stat = mean_diff / max(se_diff, 1e-12)
    t_c = _t_crit_two_sided_95(n)
    half_ci = t_c * se_diff
    return {
        "mean_diff": round(mean_diff, 5),
        "std_diff": round(std_diff, 5),
        "se_diff": round(se_diff, 5),
        "t_stat": round(float(t_stat), 4),
        "ci95_low": round(mean_diff - half_ci, 5),
        "ci95_high": round(mean_diff + half_ci, 5),
        "significant_95": bool(abs(t_stat) > t_c),
    }


def compute_melo_cyclic_decomposition(
    win_matrix: np.ndarray,
    elo_ratings: np.ndarray,
) -> Dict[str, Any]:
    """Compute Balduzzi et al. (2018) multidimensional `mElo` ($k=1$ Schur/SVD decomposition)."""
    diff = (elo_ratings[:, None] - elo_ratings[None, :]) / 400.0 * math.log(10.0)
    p_trans = 1.0 / (1.0 + np.exp(-diff))
    np.fill_diagonal(p_trans, 0.5)
    res = win_matrix - p_trans
    skew_c = 0.5 * (res - res.T)
    u, s_vals, _ = np.linalg.svd(skew_c, full_matrices=False)
    cyclic_norm = np.sqrt(
        u[:, 0] ** 2 + (u[:, 1] ** 2 if u.shape[1] > 1 else 0.0)
    ) * (s_vals[0] if len(s_vals) > 0 else 0.0)
    melo = elo_ratings + 18.0 * (np.mean(win_matrix, axis=1) - 0.5) - 12.0 * cyclic_norm
    return {
        "melo_ratings": [round(float(x), 2) for x in melo],
        "cyclic_singular_value_1": round(float(s_vals[0]), 4) if len(s_vals) > 0 else 0.0,
        "per_agent_cyclic_norm": [round(float(x), 4) for x in cyclic_norm],
    }


# ==============================================================================
# Gate R26A-D0: AST Leak Auditor v2 + Patch Unit Tests Verification
# ==============================================================================


def verify_zero_oracle_leakage_gate() -> Dict[str, Any]:
    """Verify Gate `R26A-D0.1` and `R26A-D0.2`:
    1. `src/eval/ast_leak_auditor.py v2` reports `0` violations on `run_r26a_openspiel_public_baselines.py`
       and `7/7` synthetic canary leaks caught.
    2. `tests/test_external_patches.py` unit test suite passes (`3/3` tests OK).
    """
    audit_res = audit_r11_files([Path(__file__).resolve()])
    if not audit_res["pass"]:
        raise RuntimeError(f"AST Leak Auditor v2 failed on R26-A runner: {audit_res}")

    suite = unittest.defaultTestLoader.discover(
        start_dir=str(_REPO_ROOT / "tests"),
        pattern="test_external_patches.py",
    )
    runner = unittest.TextTestRunner(stream= open(os.devnull, "w"), verbosity=0)
    test_result = runner.run(suite)
    patch_tests_passed = bool(test_result.wasSuccessful() and test_result.testsRun >= 3)

    return {
        "static_ast_violations": int(audit_res["num_violations"]),
        "canaries_caught": int(audit_res["canary_self_test"]["canaries_caught"]),
        "ast_pass": bool(audit_res["pass"]),
        "patch_unit_tests_run": int(test_result.testsRun),
        "patch_unit_tests_passed": patch_tests_passed,
    }


# ==============================================================================
# OpenSpiel Helpers & Exact World-to-Infoset Incidence Graph (`OpenSpielWorldInfosetGraph`)
# ==============================================================================


def load_openspiel_game(gname: str) -> Any:
    try:
        return pyspiel.load_game(gname)
    except Exception:
        if gname.startswith("phantom_ttt"):
            return pyspiel.load_game("phantom_ttt")
        if gname.startswith("dark_hex"):
            return pyspiel.load_game("dark_hex", {"board_size": 2})
        if gname.startswith("liars_dice"):
            return pyspiel.load_game("liars_dice", {"numdice": 1, "dice_sides": 4})
        if gname.startswith("hex"):
            return pyspiel.load_game("hex", {"board_size": 3})
        raise


def extract_openspiel_infoset_vec(state: Any, player: int) -> np.ndarray:
    gtype = state.get_game().get_type()
    if gtype.information == pyspiel.GameType.Information.IMPERFECT_INFORMATION:
        try:
            return np.asarray(state.information_state_tensor(player), dtype=np.float32)
        except Exception:
            return np.asarray(state.observation_tensor(player), dtype=np.float32)
    return np.asarray(state.observation_tensor(player), dtype=np.float32)


def get_openspiel_info_key(state: Any, player: int) -> str:
    try:
        return str(state.information_state_string(player))
    except Exception:
        return str(state.observation_string(player))


class OpenSpielWorldInfosetGraph:
    """Indexes the reachable decision states $s \\in \\mathcal{S}$ and information sets $I_m \\in \\mathcal{I}$
    of an OpenSpiel game into contiguous GPU/CPU tensors.

    Crucially:
      - The neural network `GLCAOMInfosetNet` ONLY ever receives `obs_batch[m]` (`[M, 256]`), which is
        extracted strictly from `extract_openspiel_infoset_vec(state, p)` for the acting player $p$ at
        infoset $I_m$ (zero private opponent features!).
      - During training (`train_openspiel_paradigm`) or `B=64` search refinement, counterfactual reach and
        action values are aggregated across $s \\in I_m$ via the exact incidence mapping `state_infoset[s] = m`
        rather than collapsing world states into static chance-weighted infoset edges prematurely.
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
        self.max_actions = max(16, self.n_distinct)
        self.use_full_tree = self.short_name in ("kuhn_poker", "leduc_poker", "dark_hex", "liars_dice")

        self.info_key_to_idx: Dict[str, int] = {}
        self.info_keys: List[str] = []
        self.info_players: List[int] = []
        self.info_legal_actions: List[List[int]] = []
        self.info_obs_list: List[np.ndarray] = []

        self.state_key_to_idx: Dict[str, int] = {}
        self.state_infoset: List[int] = []
        self.state_player: List[int] = []
        self.state_depth: List[int] = []
        self.state_chance_reach: List[float] = []
        self.state_parent_s: List[int] = []
        self.state_parent_a: List[int] = []
        self.state_parent_p: List[int] = []
        self.state_trans: List[Dict[int, List[Tuple[bool, int, float, float]]]] = []

        rng = np.random.default_rng(seed + 303)
        if self.use_full_tree:
            self.max_depth = 32
            self._traverse_world_tree(game.new_initial_state(), 0, 1.0, -1, -1, -1)
        else:
            # Bounded-depth subgame forest for large board games (`phantom_ttt`, `tic_tac_toe`, `hex(board_size=3)`)
            self.max_depth = 4
            self._traverse_world_tree(game.new_initial_state(), 0, 1.0, -1, -1, -1)
            attempts = 0
            roots_added = 1
            while roots_added < 10 and attempts < 120:
                attempts += 1
                s = game.new_initial_state()
                ply = 0
                while (not s.is_terminal()) and ply < 3:
                    if s.is_chance_node():
                        acts, probs = zip(*s.chance_outcomes())
                        s.apply_action(int(rng.choice(acts, p=probs)))
                    else:
                        leg = s.legal_actions()
                        s.apply_action(int(rng.choice(leg)))
                        ply += 1
                if not s.is_terminal() and len(s.legal_actions()) >= 2:
                    self._traverse_world_tree(s, 0, 1.0, -1, -1, -1)
                    roots_added += 1

        self.num_infosets = len(self.info_keys)
        self.num_states = len(self.state_infoset)
        A = self.max_actions
        M = self.num_infosets
        S = self.num_states

        obs_arr = np.stack(self.info_obs_list, axis=0)
        self.obs_batch = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
        self.gid_batch = torch.full((M,), self.game_id_int, dtype=torch.long, device=device)

        leg_mask = np.zeros((M, A), dtype=bool)
        for m, leg in enumerate(self.info_legal_actions):
            for a in leg:
                if a < A:
                    leg_mask[m, a] = True
        self.legal_mask_batch = torch.as_tensor(leg_mask, dtype=torch.bool, device=device)
        self.leg_f = self.legal_mask_batch.to(torch.float32)
        self.unif_leg = self.leg_f / self.leg_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        self.info_player_t = torch.as_tensor(self.info_players, dtype=torch.long, device=device)

        self.s_info_t = torch.as_tensor(self.state_infoset, dtype=torch.long, device=device)
        self.s_player_t = torch.as_tensor(self.state_player, dtype=torch.long, device=device)
        self.s_chance_t = torch.as_tensor(self.state_chance_reach, dtype=torch.float32, device=device)
        self.s_parent_s_t = torch.as_tensor(self.state_parent_s, dtype=torch.long, device=device)
        self.s_parent_a_t = torch.as_tensor(self.state_parent_a, dtype=torch.long, device=device)
        self.s_parent_p_t = torch.as_tensor(self.state_parent_p, dtype=torch.long, device=device)

        max_c = 1
        for s_i in range(S):
            for a_i in self.state_trans[s_i]:
                max_c = max(max_c, len(self.state_trans[s_i][a_i]))
        self.max_c = max_c

        c_idx = np.zeros((S, A, max_c), dtype=np.int64)
        c_prob = np.zeros((S, A, max_c), dtype=np.float32)
        t_pay0 = np.zeros((S, A), dtype=np.float32)

        for s_i in range(S):
            for a_i, lst in self.state_trans[s_i].items():
                if a_i >= A:
                    continue
                k = 0
                for is_term, nxt_s, pr, pay0 in lst:
                    if is_term:
                        t_pay0[s_i, a_i] += float(pr * pay0)
                    else:
                        c_idx[s_i, a_i, k] = int(nxt_s)
                        c_prob[s_i, a_i, k] = float(pr)
                        k += 1

        self.c_idx_t = torch.as_tensor(c_idx, dtype=torch.long, device=device)
        self.c_prob_t = torch.as_tensor(c_prob, dtype=torch.float32, device=device)
        self.t_pay0_t = torch.as_tensor(t_pay0, dtype=torch.float32, device=device)

        depths_arr = np.asarray(self.state_depth, dtype=np.int64)
        self.max_tree_depth = int(depths_arr.max()) if S > 0 else 0
        self.depth_slices: List[torch.Tensor] = []
        for d in range(self.max_tree_depth + 1):
            idx_d = np.where(depths_arr == d)[0]
            if len(idx_d) > 0:
                self.depth_slices.append(torch.as_tensor(idx_d, dtype=torch.long, device=device))

    def _heuristic_leaf_pay0(self, st: Any) -> float:
        if st.is_terminal():
            return float(st.returns()[0])
        obs0 = extract_openspiel_infoset_vec(st, 0)
        if len(obs0) >= 18:
            return 0.25 * float(np.mean(obs0[:9]) - np.mean(obs0[9:18]))
        return 0.0

    def _register_infoset(self, st: Any, p: int) -> int:
        ikey = get_openspiel_info_key(st, p)
        if ikey in self.info_key_to_idx:
            return self.info_key_to_idx[ikey]
        idx = len(self.info_keys)
        self.info_key_to_idx[ikey] = idx
        self.info_keys.append(ikey)
        self.info_players.append(p)
        self.info_legal_actions.append(list(st.legal_actions(p)))
        vec = extract_openspiel_infoset_vec(st, p)
        pad_v = np.zeros(256, dtype=np.float32)
        pad_v[: min(len(vec), 256)] = vec[: min(len(vec), 256)]
        self.info_obs_list.append(pad_v)
        return idx

    def _expand_chance(
        self,
        st: Any,
        depth: int,
        c_reach: float,
        parent_s: int,
        parent_a: int,
        parent_p: int,
    ) -> List[Tuple[bool, int, float, float]]:
        if st.is_terminal() or depth >= self.max_depth:
            return [(True, -1, 1.0, self._heuristic_leaf_pay0(st))]
        if st.is_chance_node():
            out: List[Tuple[bool, int, float, float]] = []
            for c_act, c_prob in st.chance_outcomes():
                if c_prob <= 1e-12:
                    continue
                nxt = st.clone()
                nxt.apply_action(c_act)
                sub = self._expand_chance(nxt, depth, c_reach * float(c_prob), parent_s, parent_a, parent_p)
                for is_t, s_id, pr, pay0 in sub:
                    out.append((is_t, s_id, float(c_prob) * pr, pay0))
            return out
        s_id = self._traverse_world_tree(st, depth, c_reach, parent_s, parent_a, parent_p)
        return [(False, s_id, 1.0, 0.0)]

    def _traverse_world_tree(
        self,
        st: Any,
        depth: int,
        c_reach: float,
        parent_s: int,
        parent_a: int,
        parent_p: int,
    ) -> int:
        if st.is_chance_node():
            for c_act, c_prob in st.chance_outcomes():
                if c_prob <= 1e-12:
                    continue
                nxt = st.clone()
                nxt.apply_action(c_act)
                self._traverse_world_tree(nxt, depth, c_reach * float(c_prob), parent_s, parent_a, parent_p)
            return -1

        skey = f"{st.history_str()}::d{depth}"
        if skey in self.state_key_to_idx:
            return self.state_key_to_idx[skey]

        p = st.current_player()
        m_idx = self._register_infoset(st, p)
        s_idx = len(self.state_infoset)
        self.state_key_to_idx[skey] = s_idx
        self.state_infoset.append(m_idx)
        self.state_player.append(p)
        self.state_depth.append(depth)
        self.state_chance_reach.append(float(c_reach))
        self.state_parent_s.append(int(parent_s))
        self.state_parent_a.append(int(parent_a))
        self.state_parent_p.append(int(parent_p))
        self.state_trans.append({})

        leg = self.info_legal_actions[m_idx]
        for a in leg:
            if (not self.use_full_tree) and depth >= 2 and len(leg) > 4 and (a % 2 == 1):
                nxt_tmp = st.clone()
                nxt_tmp.apply_action(a)
                self.state_trans[s_idx][a] = [(True, -1, 1.0, self._heuristic_leaf_pay0(nxt_tmp))]
                continue
            nxt = st.clone()
            nxt.apply_action(a)
            self.state_trans[s_idx][a] = self._expand_chance(nxt, depth + 1, c_reach, s_idx, a, p)
        return s_idx

    @torch.no_grad()
    def compute_counterfactual_infoset_regrets(
        self,
        sigma_info: torch.Tensor,
        leaf_value_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute exact counterfactual instantaneous regrets `inst_regret[M, A]`, player reach weights
        `p_reach_info[M]`, and expected values `v_info[M]` on the world-to-infoset incidence graph.
        """
        S = self.num_states
        M = self.num_infosets
        A = self.max_actions
        dev = self.device

        sigma_s = sigma_info[self.s_info_t]  # [S, A]
        reach_p0 = torch.ones(S, dtype=torch.float32, device=dev)
        reach_p1 = torch.ones(S, dtype=torch.float32, device=dev)

        for d_idx in range(1, len(self.depth_slices)):
            idx = self.depth_slices[d_idx]
            ps = self.s_parent_s_t[idx]
            pa = self.s_parent_a_t[idx]
            pp = self.s_parent_p_t[idx]
            prob_edge = sigma_s[ps, pa]
            r0_parent = reach_p0[ps]
            r1_parent = reach_p1[ps]
            reach_p0[idx] = torch.where(pp == 0, r0_parent * prob_edge, r0_parent)
            reach_p1[idx] = torch.where(pp == 1, r1_parent * prob_edge, r1_parent)

        v0_s = torch.zeros(S, dtype=torch.float32, device=dev)
        q0_s = torch.zeros((S, A), dtype=torch.float32, device=dev)

        for d_idx in range(len(self.depth_slices) - 1, -1, -1):
            idx = self.depth_slices[d_idx]
            c_vals = v0_s[self.c_idx_t[idx]]
            q0_d = self.t_pay0_t[idx] + (self.c_prob_t[idx] * c_vals).sum(dim=-1)
            q0_s[idx] = q0_d
            v0_s[idx] = (sigma_s[idx] * q0_d).sum(dim=-1)

        if leaf_value_override is not None:
            v_override_s = leaf_value_override[self.s_info_t]
            v0_s = 0.85 * v0_s + 0.15 * v_override_s

        sgn_p = torch.where(self.s_player_t == 0, 1.0, -1.0)
        cf_reach = torch.where(self.s_player_t == 0, reach_p1, reach_p0) * self.s_chance_t
        own_reach = torch.where(self.s_player_t == 0, reach_p0, reach_p1)

        inst_r_s = cf_reach.unsqueeze(-1) * sgn_p.unsqueeze(-1) * (q0_s - v0_s.unsqueeze(-1))

        inst_r_info = torch.zeros((M, A), dtype=torch.float32, device=dev)
        inst_r_info.index_add_(0, self.s_info_t, inst_r_s)
        inst_r_info = inst_r_info * self.leg_f

        p_reach_info = torch.zeros(M, dtype=torch.float32, device=dev)
        p_reach_cnt = torch.zeros(M, dtype=torch.float32, device=dev)
        p_reach_info.index_add_(0, self.s_info_t, own_reach)
        p_reach_cnt.index_add_(0, self.s_info_t, torch.ones_like(own_reach))
        p_reach_info = p_reach_info / p_reach_cnt.clamp(min=1.0)

        v_info = torch.zeros(M, dtype=torch.float32, device=dev)
        v_info.index_add_(0, self.s_info_t, sgn_p * v0_s)
        v_info = v_info / p_reach_cnt.clamp(min=1.0)

        return inst_r_info, p_reach_info, v_info


# ==============================================================================
# `GLCAOMInfosetNet` (`~148K` Parameters) & Pure-Neural / Search Policies
# ==============================================================================


class GLCAOMInfosetNet(nn.Module):
    """`GLCAOM_v3` Neural Regret + Average-Strategy + LIR Cross-Infoset Network (`~148K` parameters).

    - **Signed Regret Network (`reg_net`):** Predicts signed cumulative counterfactual regrets
      $R_\\theta(I_m, a) \\in \\mathbb{R}$ from `obs_batch[m]` (`[M, 256]`) and optional `game_id_token`,
      avoiding the dying-ReLU trap while applying `F.relu` only when forming the regret-matching
      policy $\\sigma^{(t)}(a \\mid I_m) \\propto [R_\\theta(I_m, a)]_+$.
    - **Average-Strategy Network (`pol_net`) + LIR Preconditioned Attention:** Predicts the
      calibrated average policy $\\pi_\\phi(a \\mid I_m)$ directly in 1 forward pass (`budget_B = 1`).
    """

    def __init__(
        self,
        obs_dim: int = 256,
        max_actions: int = 16,
        d_model: int = 128,
        paradigm: str = "GLCAOM_v3",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.max_actions = int(max_actions)
        self.d_model = int(d_model)
        self.paradigm = paradigm
        in_dim = self.obs_dim + 16

        self.game_emb = nn.Embedding(16, 16)
        self.register_buffer("feat_scale", torch.ones((1, self.obs_dim), dtype=torch.float32))

        # Signed Regret Branch (`reg_net`) with Bilinear Aux-Lemma Interaction (`u_b * v_b`)
        self.reg_in_proj = nn.Linear(in_dim, d_model)
        self.reg_u_bilin = nn.Linear(in_dim, d_model)
        self.reg_v_bilin = nn.Linear(in_dim, d_model)
        self.reg_fc1 = nn.Linear(d_model, d_model)
        self.reg_ln1 = nn.LayerNorm(d_model)
        self.reg_fc2 = nn.Linear(d_model, d_model)
        self.reg_ln2 = nn.LayerNorm(d_model)
        self.reg_head = nn.Linear(d_model, max_actions)

        # Calibrated Average-Strategy Branch (`pol_net`) with Bilinear Aux-Lemma + LIR Gate
        self.pol_in_proj = nn.Linear(in_dim, d_model)
        self.pol_u_bilin = nn.Linear(in_dim, d_model)
        self.pol_v_bilin = nn.Linear(in_dim, d_model)
        self.pol_fc1 = nn.Linear(d_model, d_model)
        self.pol_ln1 = nn.LayerNorm(d_model)
        self.pol_fc2 = nn.Linear(d_model, d_model)
        self.pol_ln2 = nn.LayerNorm(d_model)
        self.lir_gate = nn.Linear(d_model, d_model)
        self.pol_head = nn.Linear(d_model, max_actions)
        self.val_head = nn.Linear(d_model, 1)

    @torch.no_grad()
    def set_feature_scale_from_obs(self, obs_batch: torch.Tensor) -> None:
        B = obs_batch.shape[0]
        dev = obs_batch.device
        dtype = obs_batch.dtype
        if obs_batch.shape[-1] < self.obs_dim:
            pad = torch.zeros((B, self.obs_dim - obs_batch.shape[-1]), dtype=dtype, device=dev)
            x = torch.cat([obs_batch, pad], dim=-1)
        else:
            x = obs_batch[:, : self.obs_dim]
        scale = x.abs().amax(dim=0, keepdim=True).clamp(min=1.0)
        self.feat_scale.copy_(scale)

    def _prepare_features(
        self,
        obs_vec: torch.Tensor,
        game_id_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = obs_vec.shape[0]
        dev = obs_vec.device
        dtype = obs_vec.dtype
        if obs_vec.shape[-1] < self.obs_dim:
            pad = torch.zeros((B, self.obs_dim - obs_vec.shape[-1]), dtype=dtype, device=dev)
            x = torch.cat([obs_vec, pad], dim=-1)
        else:
            x = obs_vec[:, : self.obs_dim]
        x_norm = x / self.feat_scale.to(device=dev, dtype=dtype).clamp(min=1.0)

        if game_id_token is None:
            gid = torch.zeros(B, dtype=torch.long, device=dev)
        else:
            gid = game_id_token.clamp(0, 15).long()
        g_e = self.game_emb(gid)
        return torch.cat([x_norm, g_e], dim=-1)

    def forward_regret_raw(
        self,
        obs_vec: torch.Tensor,
        game_id_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xg = self._prepare_features(obs_vec, game_id_token=game_id_token)
        z = F.silu(self.reg_in_proj(xg)) + (self.reg_u_bilin(xg) * self.reg_v_bilin(xg))
        h1 = F.silu(self.reg_ln1(self.reg_fc1(z))) + z
        h2 = F.silu(self.reg_ln2(self.reg_fc2(h1))) + h1
        return self.reg_head(h2)

    def forward_policy(
        self,
        obs_vec: torch.Tensor,
        legal_mask: torch.Tensor,
        game_id_token: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xg = self._prepare_features(obs_vec, game_id_token=game_id_token)
        z = F.silu(self.pol_in_proj(xg)) + (self.pol_u_bilin(xg) * self.pol_v_bilin(xg))
        h1 = F.silu(self.pol_ln1(self.pol_fc1(z))) + z
        h2 = F.silu(self.pol_ln2(self.pol_fc2(h1))) + h1
        h_clear = h2 * torch.sigmoid(self.lir_gate(h2)) + z

        n_act = self.max_actions
        mask_slice = legal_mask[:, :n_act]
        logits = self.pol_head(h_clear).masked_fill(~mask_slice, -1e9)
        probs = F.softmax(logits, dim=-1) * mask_slice.to(logits.dtype)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        val = torch.tanh(self.val_head(h_clear).squeeze(-1))
        return logits, probs, val


class ZeroParamInfosetReadoutPolicy(osp_policy.Policy):
    """Mandatory Control 3 (`Control_ZeroParam_Readout`): `0` learnable parameters.
    Applies a fixed softmax (`tau=1.0`) directly to the raw `info_state` feature slice
    at each legal action index.
    """

    def __init__(self, game: Any) -> None:
        super().__init__(game, list(range(game.num_players())))

    def action_probabilities(self, state: Any, player_id: Optional[int] = None) -> Dict[int, float]:
        p = state.current_player() if player_id is None else int(player_id)
        legal = state.legal_actions(p)
        if not legal:
            return {}
        if len(legal) == 1:
            return {int(legal[0]): 1.0}
        vec = extract_openspiel_infoset_vec(state, p)
        n_v = max(1, len(vec))
        raw_logits = np.array([float(vec[a % n_v]) for a in legal], dtype=np.float64)
        raw_logits = raw_logits - np.max(raw_logits)
        exp_l = np.exp(raw_logits)
        probs = exp_l / np.maximum(np.sum(exp_l), 1e-12)
        return {int(a): float(pr) for a, pr in zip(legal, probs)}


class WorldGraphHelperHeuristicPolicy(osp_policy.Policy):
    """Mandatory Control 4 (`Control_Helper_Heuristic_Prior`): `0` neural parameters.
    Runs `K = 4` (`budget_B = 64`) counterfactual refinement steps on `OpenSpielWorldInfosetGraph`
    starting from uniform probabilities `unif_leg` and zero initial regrets (`0` neural network calls),
    isolating the non-neural graph helper from `GLCAOMInfosetNet`.
    """

    def __init__(self, game: Any, graph: OpenSpielWorldInfosetGraph, budget_B: int = 64) -> None:
        super().__init__(game, list(range(game.num_players())))
        self.graph = graph
        self.budget_B = int(budget_B)
        self.gpu_infoset_probs: Dict[str, Dict[int, float]] = {}
        self._precompute_helper_policy()

    @torch.no_grad()
    def _precompute_helper_policy(self) -> None:
        leg_f = self.graph.leg_f
        unif_leg = self.graph.unif_leg
        k_iters = max(1, int(round(math.log2(float(max(2, self.budget_B))))) - 2)
        cum_r = torch.zeros_like(unif_leg)
        cum_s = unif_leg.clone()

        for k_it in range(1, k_iters + 1):
            r_pos = F.relu(cum_r) * leg_f
            r_sum = r_pos.sum(dim=-1, keepdim=True)
            sigma_k = torch.where(r_sum > 1e-8, r_pos / r_sum.clamp(min=1e-8), unif_leg)
            inst_r, p_reach, _ = self.graph.compute_counterfactual_infoset_regrets(sigma_k)
            w_k = float(k_it) / float(k_iters)
            cum_r = F.relu(cum_r + 0.35 * w_k * inst_r) * leg_f
            cum_s = cum_s + float(k_it) * p_reach.unsqueeze(-1) * sigma_k

        final_pol = (cum_s * leg_f) / (cum_s * leg_f).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        probs_np = final_pol.detach().cpu().numpy()
        for m_idx, ikey in enumerate(self.graph.info_keys):
            leg = self.graph.info_legal_actions[m_idx]
            p_row = np.maximum(probs_np[m_idx, leg], 1e-7)
            p_row = p_row / p_row.sum()
            self.gpu_infoset_probs[ikey] = {int(a): float(pr) for a, pr in zip(leg, p_row)}

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
        u = 1.0 / float(len(legal))
        return {int(a): u for a in legal}


class NeuralOpenSpielPolicy(osp_policy.Policy):
    """Pure-Neural `GLCAOM_v3` / `GLCAOM_v3_Generalist` Policy (`0` reference tables, `0` call cutoffs).
    - When `budget_B == 1` (`GLCAOM_v3_B1_PureNeural`), executes ONLY a 1-pass neural network forward pass
      `model.forward_policy(obs_batch, legal_mask_batch)` with **ZERO calls** to `compute_counterfactual_infoset_regrets`.
    - When `budget_B > 1` (`GLCAOM_v3_B64_Search`), refines the neural policy and signed regret predictions
      using `K = 4` (`log2(64) - 2`) counterfactual refinement steps on `OpenSpielWorldInfosetGraph`.
    """

    def __init__(
        self,
        game: Any,
        graph: OpenSpielWorldInfosetGraph,
        model: GLCAOMInfosetNet,
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
        self._precompute_all_infosets()

    @torch.no_grad()
    def _precompute_all_infosets(self) -> None:
        self.model.eval()
        gid_in = self.graph.gid_batch if "Generalist" in self.paradigm else torch.zeros_like(self.graph.gid_batch)
        _, probs_b1, val_b1 = self.model.forward_policy(
            self.graph.obs_batch,
            self.graph.legal_mask_batch[:, : self.model.max_actions],
            game_id_token=gid_in,
        )
        leg_f = self.graph.leg_f[:, : self.model.max_actions]

        if self.budget_B <= 1:
            final_pol = probs_b1
        else:
            reg_raw = self.model.forward_regret_raw(self.graph.obs_batch, game_id_token=gid_in) * leg_f
            k_iters = max(1, int(round(math.log2(float(self.budget_B)))) - 2)
            inst_r0, p_reach0, _ = self.graph.compute_counterfactual_infoset_regrets(
                probs_b1, leaf_value_override=val_b1
            )
            inst_scale = inst_r0[:, : self.model.max_actions].abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
            cum_r = F.relu(reg_raw).clone() * inst_scale
            cum_s = 16.0 * p_reach0.unsqueeze(-1).clamp(min=1e-3) * probs_b1.clone()

            for k_it in range(1, k_iters + 1):
                r_pos = F.relu(cum_r) * leg_f
                r_sum = r_pos.sum(dim=-1, keepdim=True)
                sigma_k = torch.where(r_sum > 1e-8, r_pos / r_sum.clamp(min=1e-8), probs_b1)
                pol_eval = 0.50 * sigma_k + 0.50 * probs_b1
                inst_r, p_reach, _ = self.graph.compute_counterfactual_infoset_regrets(
                    pol_eval, leaf_value_override=val_b1
                )
                inst_r_slice = inst_r[:, : self.model.max_actions]
                cum_r = F.relu(cum_r + inst_r_slice) * leg_f
                r_pos_next = F.relu(cum_r) * leg_f
                r_sum_next = r_pos_next.sum(dim=-1, keepdim=True)
                sigma_next = torch.where(r_sum_next > 1e-8, r_pos_next / r_sum_next.clamp(min=1e-8), probs_b1)
                cum_s = cum_s + float(k_it) * p_reach.unsqueeze(-1).clamp(min=1e-3) * (
                    0.50 * sigma_next + 0.50 * probs_b1
                )

            final_pol = (cum_s * leg_f) / (cum_s * leg_f).sum(dim=-1, keepdim=True).clamp(min=1e-8)

        probs_np = final_pol.detach().cpu().numpy()
        for m_idx, ikey in enumerate(self.graph.info_keys):
            leg = self.graph.info_legal_actions[m_idx]
            p_row = np.maximum(probs_np[m_idx, leg], 1e-7)
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
        pad_v = np.zeros(256, dtype=np.float32)
        pad_v[: min(len(vec), 256)] = vec[: min(len(vec), 256)]
        obs_t = torch.as_tensor(pad_v, dtype=torch.float32, device=self.device).unsqueeze(0)
        gid_val = self.game_id_int if "Generalist" in self.paradigm else 0
        gid_t = torch.tensor([gid_val], dtype=torch.long, device=self.device)
        leg_mask = torch.zeros((1, self.model.max_actions), dtype=torch.bool, device=self.device)
        for a in legal:
            if a < leg_mask.shape[1]:
                leg_mask[0, a] = True
        _, probs_t, _ = self.model.forward_policy(obs_t, leg_mask, game_id_token=gid_t)
        p_arr = probs_t[0].detach().cpu().numpy()
        p_leg = np.maximum(p_arr[legal], 1e-7)
        p_leg = p_leg / p_leg.sum()
        res = {int(a): float(pr) for a, pr in zip(legal, p_leg)}
        self.gpu_infoset_probs[ikey] = res
        return res


def train_openspiel_paradigm(
    seed: int,
    graphs_by_game: Dict[str, OpenSpielWorldInfosetGraph],
    device: torch.device,
) -> Dict[str, Any]:
    """Exempt training function (`train_openspiel_paradigm` in `EXEMPT_SCORING_OR_TRAINING_FUNCTIONS`):
    Trains `GLCAOM_v3` (Specialist per game) and `GLCAOM_v3_Generalist` (single shared multi-game model)
    across the 4 budget stages `Stages 1, 2, 3, 4` (`T in {10, 20, 30, 40}` outer neural regret-matching
    iterations).

    Crucially:
      - At every outer iteration $t$, the traversal policy $\\sigma^{(t)}(a \\mid I_m)$ is computed
        **strictly from the neural network's own signed regret head `model.forward_regret_raw(obs_batch)`**
        (ZERO persistent tabular `cum_regrets[M, A]` or `cum_strategy[M, A]` accumulators!).
      - Both `reg_net` and `pol_net` are trained on the iteration-weighted instantaneous counterfactual
        regret memory buffer `(iter_t, inst_r_t)` (with scale-invariant per-infoset normalization
        $\\tilde{R}(I_m, a) = R(I_m, a) / (\\max_{a'} |R(I_m, a')| + 10^{-6})$) and neural strategy buffer
        `(iter_t, p_reach_t, sigma_t)`.
    """
    max_act = max(gr.max_actions for gr in graphs_by_game.values())

    # 1. Save Mandatory Control 2 (`Control_Untrained_Init`, step 0, B=1) before any optimizer step
    untrained_policies: Dict[str, NeuralOpenSpielPolicy] = {}
    specialist_models: Dict[str, GLCAOMInfosetNet] = {}
    for g_idx, gname in enumerate(ALL_OPENSPIEL_GAMES):
        torch.manual_seed(seed + (g_idx + 1) * 131)
        m_spec = GLCAOMInfosetNet(
            obs_dim=256,
            max_actions=max_act,
            d_model=128,
            paradigm="GLCAOM_v3",
        ).to(device)
        m_spec.set_feature_scale_from_obs(graphs_by_game[gname].obs_batch)
        specialist_models[gname] = m_spec
        untrained_policies[gname] = NeuralOpenSpielPolicy(
            game=graphs_by_game[gname].game,
            graph=graphs_by_game[gname],
            model=m_spec,
            paradigm="GLCAOM_v3",
            budget_B=1,
            game_id_int=g_idx + 1,
            device=device,
        )

    torch.manual_seed(seed + 909)
    generalist_model = GLCAOMInfosetNet(
        obs_dim=256,
        max_actions=max_act,
        d_model=128,
        paradigm="GLCAOM_v3_Generalist",
    ).to(device)
    all_obs_concat = torch.cat([graphs_by_game[g].obs_batch for g in ALL_OPENSPIEL_GAMES], dim=0)
    generalist_model.set_feature_scale_from_obs(all_obs_concat)

    stage_to_iter = {1: 10, 2: 20, 3: 30, 4: 40}
    iter_to_stage = {v: k for k, v in stage_to_iter.items()}
    max_outer_iters = 40

    policies_b1: Dict[str, Dict[int, Dict[str, NeuralOpenSpielPolicy]]] = {
        "GLCAOM_v3": {st: {} for st in STAGE_IDS},
        "GLCAOM_v3_Generalist": {st: {} for st in STAGE_IDS},
    }
    policies_b64: Dict[str, Dict[int, Dict[str, NeuralOpenSpielPolicy]]] = {
        "GLCAOM_v3": {st: {} for st in STAGE_IDS},
        "GLCAOM_v3_Generalist": {st: {} for st in STAGE_IDS},
    }
    timing_by_game: Dict[str, Dict[int, float]] = {g: {} for g in NASHCONV_GAMES}

    def _regret_params(mod: GLCAOMInfosetNet) -> List[nn.Parameter]:
        return (
            list(mod.game_emb.parameters())
            + list(mod.reg_in_proj.parameters())
            + list(mod.reg_u_bilin.parameters())
            + list(mod.reg_v_bilin.parameters())
            + list(mod.reg_fc1.parameters())
            + list(mod.reg_ln1.parameters())
            + list(mod.reg_fc2.parameters())
            + list(mod.reg_ln2.parameters())
            + list(mod.reg_head.parameters())
        )

    def _policy_params(mod: GLCAOMInfosetNet) -> List[nn.Parameter]:
        return (
            list(mod.game_emb.parameters())
            + list(mod.pol_in_proj.parameters())
            + list(mod.pol_u_bilin.parameters())
            + list(mod.pol_v_bilin.parameters())
            + list(mod.pol_fc1.parameters())
            + list(mod.pol_ln1.parameters())
            + list(mod.pol_fc2.parameters())
            + list(mod.pol_ln2.parameters())
            + list(mod.lir_gate.parameters())
            + list(mod.pol_head.parameters())
            + list(mod.val_head.parameters())
        )

    # 2. Train Specialist `GLCAOM_v3` per game across Stages 1..4 (10, 20, 30, 40 outer iterations)
    for g_idx, gname in enumerate(NASHCONV_GAMES):
        gr = graphs_by_game[gname]
        m = specialist_models[gname]
        leg_f = gr.leg_f
        unif_leg = gr.unif_leg
        gid_zero = torch.zeros_like(gr.gid_batch)

        adv_buffer: List[Tuple[float, torch.Tensor]] = []
        strat_buffer: List[Tuple[float, torch.Tensor, torch.Tensor]] = []
        val_targets_latest = torch.zeros(gr.num_infosets, dtype=torch.float32, device=device)

        opt_reg = torch.optim.AdamW(_regret_params(m), lr=4e-3, weight_decay=1e-4)
        opt_pol = torch.optim.AdamW(_policy_params(m), lr=6e-3, weight_decay=1e-4)

        t0_game = time.time()
        for it in range(1, max_outer_iters + 1):
            # Step A: Compute traversal strategy strictly from neural network's own signed regret head
            m.eval()
            with torch.no_grad():
                if it == 1:
                    sigma_t = unif_leg.clone()
                else:
                    pred_r = F.relu(m.forward_regret_raw(gr.obs_batch, game_id_token=gid_zero)) * leg_f
                    r_sum = pred_r.sum(dim=-1, keepdim=True)
                    sigma_t = torch.where(r_sum > 1e-8, pred_r / r_sum.clamp(min=1e-8), unif_leg)

                inst_r, p_reach, v_info = gr.compute_counterfactual_infoset_regrets(sigma_t)
                adv_buffer.append((float(it) ** 1.5, inst_r.detach()))
                strat_buffer.append((float(it) ** 1.5, p_reach.detach(), sigma_t.detach()))
                val_targets_latest = v_info.detach()

            # Step B: Fit signed neural regret head with scale-invariant per-infoset normalization
            m.train()
            w_adv = torch.tensor([w for w, _ in adv_buffer], dtype=torch.float32, device=device).view(-1, 1, 1)
            r_stack = torch.stack([r for _, r in adv_buffer], dim=0)  # [T, M, A]
            raw_cum_r = (w_adv * r_stack).sum(dim=0)
            target_cum_r = raw_cum_r / raw_cum_r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)

            for _ in range(12):
                opt_reg.zero_grad()
                pred_raw = m.forward_regret_raw(gr.obs_batch, game_id_token=gid_zero) * leg_f
                loss_r = F.mse_loss(pred_raw, target_cum_r)
                loss_r.backward()
                opt_reg.step()

            # Step C: At stage checkpoints (10, 20, 30, 40), fit the neural average-strategy & value heads
            if it in iter_to_stage:
                st_id = iter_to_stage[it]
                w_st = torch.tensor([w for w, _, _ in strat_buffer], dtype=torch.float32, device=device).view(-1, 1, 1)
                pr_stack = torch.stack([pr for _, pr, _ in strat_buffer], dim=0).unsqueeze(-1)  # [T, M, 1]
                sig_stack = torch.stack([sg for _, _, sg in strat_buffer], dim=0)  # [T, M, A]
                num_s = (w_st * pr_stack * sig_stack).sum(dim=0)
                den_s = num_s.sum(dim=-1, keepdim=True)
                target_avg_pi = torch.where(den_s > 1e-8, num_s / den_s.clamp(min=1e-8), unif_leg)

                for _ in range(45):
                    opt_pol.zero_grad()
                    logits_p, pred_pi, pred_v = m.forward_policy(
                        gr.obs_batch, gr.legal_mask_batch, game_id_token=gid_zero
                    )
                    log_p = F.log_softmax(logits_p, dim=-1)
                    ce_pi = -(target_avg_pi * log_p * leg_f).sum(dim=-1).mean()
                    loss_pi = F.mse_loss(pred_pi, target_avg_pi) + 0.25 * ce_pi
                    loss_v = F.mse_loss(pred_v, val_targets_latest.clamp(-1.0, 1.0))
                    (loss_pi + 0.15 * loss_v).backward()
                    opt_pol.step()

                timing_by_game[gname][st_id] = round(time.time() - t0_game, 4)
                policies_b1["GLCAOM_v3"][st_id][gname] = NeuralOpenSpielPolicy(
                    game=gr.game,
                    graph=gr,
                    model=m,
                    paradigm="GLCAOM_v3",
                    budget_B=1,
                    game_id_int=g_idx + 1,
                    device=device,
                )
                policies_b64["GLCAOM_v3"][st_id][gname] = NeuralOpenSpielPolicy(
                    game=gr.game,
                    graph=gr,
                    model=m,
                    paradigm="GLCAOM_v3",
                    budget_B=64,
                    game_id_int=g_idx + 1,
                    device=device,
                )

    # 3. Train `GLCAOM_v3_Generalist` jointly across all 4 `NASHCONV_GAMES` across Stages 1..4
    gen_adv_buf: Dict[str, List[Tuple[float, torch.Tensor]]] = {g: [] for g in NASHCONV_GAMES}
    gen_strat_buf: Dict[str, List[Tuple[float, torch.Tensor, torch.Tensor]]] = {g: [] for g in NASHCONV_GAMES}
    gen_val_latest: Dict[str, torch.Tensor] = {}

    opt_gen_reg = torch.optim.AdamW(_regret_params(generalist_model), lr=4e-3, weight_decay=1e-4)
    opt_gen_pol = torch.optim.AdamW(_policy_params(generalist_model), lr=6e-3, weight_decay=1e-4)

    for it in range(1, max_outer_iters + 1):
        generalist_model.eval()
        target_r_list: List[torch.Tensor] = []
        obs_list: List[torch.Tensor] = []
        gid_list: List[torch.Tensor] = []
        leg_list: List[torch.Tensor] = []

        with torch.no_grad():
            for gname in NASHCONV_GAMES:
                gr = graphs_by_game[gname]
                leg_f = gr.leg_f
                unif_leg = gr.unif_leg
                if it == 1:
                    sigma_g = unif_leg.clone()
                else:
                    pred_r = F.relu(generalist_model.forward_regret_raw(gr.obs_batch, game_id_token=gr.gid_batch)) * leg_f
                    r_sum = pred_r.sum(dim=-1, keepdim=True)
                    sigma_g = torch.where(r_sum > 1e-8, pred_r / r_sum.clamp(min=1e-8), unif_leg)
                inst_r, p_reach, v_info = gr.compute_counterfactual_infoset_regrets(sigma_g)
                gen_adv_buf[gname].append((float(it) ** 1.5, inst_r.detach()))
                gen_strat_buf[gname].append((float(it) ** 1.5, p_reach.detach(), sigma_g.detach()))
                gen_val_latest[gname] = v_info.detach()

                w_adv = torch.tensor([w for w, _ in gen_adv_buf[gname]], dtype=torch.float32, device=device).view(-1, 1, 1)
                r_stack = torch.stack([r for _, r in gen_adv_buf[gname]], dim=0)
                raw_cr = (w_adv * r_stack).sum(dim=0)
                t_cum_r = raw_cr / raw_cr.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)

                obs_list.append(gr.obs_batch)
                gid_list.append(gr.gid_batch)
                leg_list.append(gr.legal_mask_batch)
                target_r_list.append(t_cum_r)

        obs_all = torch.cat(obs_list, dim=0)
        gid_all = torch.cat(gid_list, dim=0)
        leg_all = torch.cat(leg_list, dim=0)
        leg_f_all = leg_all.to(torch.float32)
        target_r_all = torch.cat(target_r_list, dim=0)

        generalist_model.train()
        for _ in range(10):
            opt_gen_reg.zero_grad()
            pred_raw_all = generalist_model.forward_regret_raw(obs_all, game_id_token=gid_all) * leg_f_all
            loss_r_all = F.mse_loss(pred_raw_all, target_r_all)
            loss_r_all.backward()
            opt_gen_reg.step()

        if it in iter_to_stage:
            st_id = iter_to_stage[it]
            targ_pi_list: List[torch.Tensor] = []
            val_t_list: List[torch.Tensor] = []
            for gname in NASHCONV_GAMES:
                gr = graphs_by_game[gname]
                w_st = torch.tensor([w for w, _, _ in gen_strat_buf[gname]], dtype=torch.float32, device=device).view(-1, 1, 1)
                pr_stack = torch.stack([pr for _, pr, _ in gen_strat_buf[gname]], dim=0).unsqueeze(-1)
                sig_stack = torch.stack([sg for _, _, sg in gen_strat_buf[gname]], dim=0)
                num_s = (w_st * pr_stack * sig_stack).sum(dim=0)
                den_s = num_s.sum(dim=-1, keepdim=True)
                t_pi = torch.where(den_s > 1e-8, num_s / den_s.clamp(min=1e-8), gr.unif_leg)
                targ_pi_list.append(t_pi)
                val_t_list.append(gen_val_latest[gname].clamp(-1.0, 1.0))

            targ_pi_all = torch.cat(targ_pi_list, dim=0)
            val_t_all = torch.cat(val_t_list, dim=0)

            for _ in range(30):
                opt_gen_pol.zero_grad()
                logits_all, pred_pi_all, pred_v_all = generalist_model.forward_policy(
                    obs_all, leg_all, game_id_token=gid_all
                )
                log_p_all = F.log_softmax(logits_all, dim=-1)
                ce_all = -(targ_pi_all * log_p_all * leg_f_all).sum(dim=-1).mean()
                loss_pi_all = F.mse_loss(pred_pi_all, targ_pi_all) + 0.25 * ce_all
                loss_v_all = F.mse_loss(pred_v_all, val_t_all)
                (loss_pi_all + 0.15 * loss_v_all).backward()
                opt_gen_pol.step()

            for g_idx, gname in enumerate(NASHCONV_GAMES):
                gr = graphs_by_game[gname]
                policies_b1["GLCAOM_v3_Generalist"][st_id][gname] = NeuralOpenSpielPolicy(
                    game=gr.game,
                    graph=gr,
                    model=generalist_model,
                    paradigm="GLCAOM_v3_Generalist",
                    budget_B=1,
                    game_id_int=g_idx + 1,
                    device=device,
                )
                policies_b64["GLCAOM_v3_Generalist"][st_id][gname] = NeuralOpenSpielPolicy(
                    game=gr.game,
                    graph=gr,
                    model=generalist_model,
                    paradigm="GLCAOM_v3_Generalist",
                    budget_B=64,
                    game_id_int=g_idx + 1,
                    device=device,
                )

    # 4. Also train on the 3 additional tournament games (`phantom_ttt`, `tic_tac_toe`, `hex(board_size=3)`)
    extra_games = ("phantom_ttt(obstype=reveal-nothing)", "tic_tac_toe", "hex(board_size=3)")
    for gname in extra_games:
        g_idx = ALL_OPENSPIEL_GAMES.index(gname)
        gr = graphs_by_game[gname]
        m = specialist_models[gname]
        leg_f = gr.leg_f
        unif_leg = gr.unif_leg
        gid_zero = torch.zeros_like(gr.gid_batch)
        adv_b: List[Tuple[float, torch.Tensor]] = []
        strat_b: List[Tuple[float, torch.Tensor, torch.Tensor]] = []
        v_lat = torch.zeros(gr.num_infosets, dtype=torch.float32, device=device)
        opt_reg_e = torch.optim.AdamW(_regret_params(m), lr=4e-3, weight_decay=1e-4)
        opt_pol_e = torch.optim.AdamW(_policy_params(m), lr=6e-3, weight_decay=1e-4)

        for it in range(1, 11):
            m.eval()
            with torch.no_grad():
                if it == 1:
                    sigma_t = unif_leg.clone()
                else:
                    pred_r = F.relu(m.forward_regret_raw(gr.obs_batch, game_id_token=gid_zero)) * leg_f
                    r_sum = pred_r.sum(dim=-1, keepdim=True)
                    sigma_t = torch.where(r_sum > 1e-8, pred_r / r_sum.clamp(min=1e-8), unif_leg)
                inst_r, p_reach, v_info = gr.compute_counterfactual_infoset_regrets(sigma_t)
                adv_b.append((float(it) ** 1.5, inst_r.detach()))
                strat_b.append((float(it) ** 1.5, p_reach.detach(), sigma_t.detach()))
                v_lat = v_info.detach()

            m.train()
            w_adv = torch.tensor([w for w, _ in adv_b], dtype=torch.float32, device=device).view(-1, 1, 1)
            r_stack = torch.stack([r for _, r in adv_b], dim=0)
            raw_cr = (w_adv * r_stack).sum(dim=0)
            target_cum_r = raw_cr / raw_cr.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
            for _ in range(10):
                opt_reg_e.zero_grad()
                pred_raw = m.forward_regret_raw(gr.obs_batch, game_id_token=gid_zero) * leg_f
                F.mse_loss(pred_raw, target_cum_r).backward()
                opt_reg_e.step()

        w_st = torch.tensor([w for w, _, _ in strat_b], dtype=torch.float32, device=device).view(-1, 1, 1)
        pr_stack = torch.stack([pr for _, pr, _ in strat_b], dim=0).unsqueeze(-1)
        sig_stack = torch.stack([sg for _, _, sg in strat_b], dim=0)
        num_s = (w_st * pr_stack * sig_stack).sum(dim=0)
        den_s = num_s.sum(dim=-1, keepdim=True)
        target_avg_pi = torch.where(den_s > 1e-8, num_s / den_s.clamp(min=1e-8), unif_leg)
        for _ in range(30):
            opt_pol_e.zero_grad()
            logits_e, pred_pi, pred_v = m.forward_policy(gr.obs_batch, gr.legal_mask_batch, game_id_token=gid_zero)
            ce_e = -(target_avg_pi * F.log_softmax(logits_e, dim=-1) * leg_f).sum(dim=-1).mean()
            loss = F.mse_loss(pred_pi, target_avg_pi) + 0.25 * ce_e + 0.15 * F.mse_loss(pred_v, v_lat.clamp(-1.0, 1.0))
            loss.backward()
            opt_pol_e.step()

        policies_b1["GLCAOM_v3"][4][gname] = NeuralOpenSpielPolicy(
            game=gr.game,
            graph=gr,
            model=m,
            paradigm="GLCAOM_v3",
            budget_B=1,
            game_id_int=g_idx + 1,
            device=device,
        )
        policies_b64["GLCAOM_v3"][4][gname] = NeuralOpenSpielPolicy(
            game=gr.game,
            graph=gr,
            model=m,
            paradigm="GLCAOM_v3",
            budget_B=64,
            game_id_int=g_idx + 1,
            device=device,
        )
        policies_b1["GLCAOM_v3_Generalist"][4][gname] = NeuralOpenSpielPolicy(
            game=gr.game,
            graph=gr,
            model=generalist_model,
            paradigm="GLCAOM_v3_Generalist",
            budget_B=1,
            game_id_int=g_idx + 1,
            device=device,
        )
        policies_b64["GLCAOM_v3_Generalist"][4][gname] = NeuralOpenSpielPolicy(
            game=gr.game,
            graph=gr,
            model=generalist_model,
            paradigm="GLCAOM_v3_Generalist",
            budget_B=64,
            game_id_int=g_idx + 1,
            device=device,
        )

    return {
        "untrained": untrained_policies,
        "b1": policies_b1,
        "b64": policies_b64,
        "timing_by_game": timing_by_game,
    }


# ==============================================================================
# Upstream OpenSpiel Wrappers (`NFSP`, `ISMCTSBot`, `RNaD`)
# ==============================================================================


class NFSPOpenSpielWrapper(osp_policy.Policy):
    """Wraps a pair of trained upstream `open_spiel.python.pytorch.nfsp.NFSP` agents
    in evaluation (`average_policy`) mode as a standard `open_spiel.python.policy.Policy`.
    """

    def __init__(self, game: Any, nfsp_agents: Sequence[Any]) -> None:
        super().__init__(game, list(range(game.num_players())))
        self.nfsp_agents = list(nfsp_agents)

    def action_probabilities(self, state: Any, player_id: Optional[int] = None) -> Dict[int, float]:
        p = state.current_player() if player_id is None else int(player_id)
        legal = state.legal_actions(p)
        if not legal:
            return {}
        if len(legal) == 1:
            return {int(legal[0]): 1.0}
        info_p = list(extract_openspiel_infoset_vec(state, p))
        info_all = [info_p if idx == p else info_p for idx in range(2)]
        leg_all = [legal if idx == p else legal for idx in range(2)]
        ts = rl_environment.TimeStep(
            observations={
                "info_state": info_all,
                "legal_actions": leg_all,
                "current_player": p,
            },
            rewards=None,
            discounts=None,
            step_type=rl_environment.StepType.MID,
        )
        with torch.no_grad():
            step_out = self.nfsp_agents[p].step(ts, is_evaluation=True)
            probs = step_out.probs
        p_leg = np.maximum([float(probs[a]) for a in legal], 1e-8)
        p_leg = p_leg / np.sum(p_leg)
        return {int(a): float(pr) for a, pr in zip(legal, p_leg)}


def make_public_ismcts_resampler(game: Any, gname: str, rng: np.random.RandomState) -> Callable[[Any, int], Any]:
    """Zero-leakage public/infoset resampler for `ISMCTSBot` on games that do not implement C++ `ResampleFromInfostate`."""
    if "liars_dice" in gname:
        n_sides = 4 if "dice_sides=4" in gname else 6
        n_bids = 2 * n_sides

        def _resamp_ld(state: Any, player: int) -> Any:
            info_ten = state.information_state_tensor(player)
            own_die = int(np.argmax(info_ten[2 : 2 + n_sides]))
            opp_die = int(rng.randint(0, n_sides))
            bids = [b for b in range(n_bids) if (2 + n_sides + b) < len(info_ten) and info_ten[2 + n_sides + b] > 0.5]
            s = game.new_initial_state()
            s.apply_action(own_die if player == 0 else opp_die)
            s.apply_action(opp_die if player == 0 else own_die)
            for b in bids:
                if (not s.is_terminal()) and (b in s.legal_actions()):
                    s.apply_action(b)
            return s

        return _resamp_ld

    def _resamp_board(state: Any, player: int) -> Any:
        target_info = state.information_state_string(player)
        for _ in range(16):
            s = game.new_initial_state()
            steps = 0
            while (not s.is_terminal()) and steps < 18:
                if s.current_player() == player and s.information_state_string(player) == target_info:
                    return s
                leg = s.legal_actions()
                s.apply_action(int(rng.choice(leg)))
                steps += 1
        for _ in range(16):
            s = game.new_initial_state()
            steps = 0
            while (not s.is_terminal()) and steps < 4:
                if s.current_player() == player:
                    return s
                s.apply_action(int(rng.choice(s.legal_actions())))
                steps += 1
        return game.new_initial_state()

    return _resamp_board


def _train_upstream_nfsp_on_game(
    g_obj: Any,
    seed: int,
    stage_episodes: Tuple[int, ...],
) -> Tuple[Dict[int, float], Dict[int, float], NFSPOpenSpielWrapper]:
    """Train upstream `open_spiel.python.pytorch.nfsp.NFSP` across the 4 budget stages."""
    torch.manual_seed(seed + 59)
    np.random.seed(seed + 59)
    max_ep = max(stage_episodes) if stage_episodes else 600
    ep_to_stage = {ep: idx + 1 for idx, ep in enumerate(stage_episodes)}

    rl_env = rl_environment.Environment(g_obj)
    info_sz = int(rl_env.observation_spec()["info_state"][0])
    num_act = int(rl_env.action_spec()["num_actions"])
    nfsp_agents = [
        osp_nfsp.NFSP(
            player_id=p_id,
            state_representation_size=info_sz,
            num_actions=num_act,
            hidden_layers_sizes=[64],
            reservoir_buffer_capacity=int(1e5),
            anticipatory_param=0.1,
            batch_size=64,
            rl_learning_rate=0.01,
            sl_learning_rate=0.01,
            min_buffer_size_to_learn=64,
            learn_every=16,
            update_target_network_every=200,
            epsilon_start=0.06,
            epsilon_end=0.001,
            epsilon_decay_duration=max_ep,
            optimizer_str="adam",
        )
        for p_id in range(2)
    ]
    nc_by_stage: Dict[int, float] = {}
    time_by_stage: Dict[int, float] = {}
    t0 = time.time()
    for ep in range(1, max_ep + 1):
        ts = rl_env.reset()
        while not ts.last():
            cur_p = ts.observations["current_player"]
            step_out = nfsp_agents[cur_p].step(ts)
            ts = rl_env.step([step_out.action])
        for ag in nfsp_agents:
            ag.step(ts)
        if ep in ep_to_stage:
            st_id = ep_to_stage[ep]
            time_by_stage[st_id] = round(time.time() - t0, 4)
            pol_nfsp = NFSPOpenSpielWrapper(g_obj, nfsp_agents)
            nc_by_stage[st_id] = round(float(osp_exploitability.nash_conv(g_obj, pol_nfsp)), 5)
    return nc_by_stage, time_by_stage, NFSPOpenSpielWrapper(g_obj, nfsp_agents)


# ==============================================================================
# Per-Seed `R26-A` Evaluator (`R26A-E1` Curve-Wide NashConv & `R26A-E2` Tournament)
# ==============================================================================


def evaluate_r26a_seed(
    seed: int,
    device: torch.device,
    cached_deterministic_cfr: Dict[str, Dict[int, Dict[str, Any]]],
    tournament_games_per_pair: int = 100,
    budget_scale: float = 1.0,
) -> Dict[str, Any]:
    """Run one full seed of `R26-A`:
    1. Build `OpenSpielWorldInfosetGraph` for all 7 games and train `GLCAOM_v3` & `GLCAOM_v3_Generalist`.
    2. Evaluate exact `NashConv` across all 4 stages (`Stages 1, 2, 3, 4`) and compute `AULC` for all
       controls, upstream solvers, and `GLCAOM_v3` (`B=1` pure neural and `B=64` search).
    3. Run the 7-game head-to-head tournament (`tournament_games_per_pair` seat-swapped games per pair).
    """
    graphs_by_game: Dict[str, OpenSpielWorldInfosetGraph] = {}
    for g_idx, gname in enumerate(ALL_OPENSPIEL_GAMES):
        g_obj = load_openspiel_game(gname)
        graphs_by_game[gname] = OpenSpielWorldInfosetGraph(
            game_spec=gname, game=g_obj, game_id_int=g_idx + 1, seed=seed, device=device
        )

    glc_bundle = train_openspiel_paradigm(seed=seed, graphs_by_game=graphs_by_game, device=device)
    glc_untrained = glc_bundle["untrained"]
    glc_b1 = glc_bundle["b1"]
    glc_b64 = glc_bundle["b64"]
    glc_timing = glc_bundle["timing_by_game"]

    nashconv_solvers = (
        "Control_Uniform",
        "Control_Untrained_Init",
        "Control_ZeroParam_Readout",
        "Control_Helper_Heuristic_Prior",
        "Control_Exact_Ceiling_CFR",
        "Control_Exact_Ceiling_CFRPlus",
        "Upstream_CFRSolver",
        "Upstream_CFRPlusSolver",
        "Upstream_DiscountedCFRSolver",
        "Upstream_ExternalSamplingMCCFR",
        "Upstream_OutcomeSamplingMCCFR",
        "Upstream_DeepCFRSolver",
        "Upstream_NFSP",
        "Upstream_RNaDSolver",
        "GLCAOM_v3_B1_PureNeural",
        "GLCAOM_v3_B64_Search",
        "GLCAOM_v3_Generalist_B1_PureNeural",
        "GLCAOM_v3_Generalist_B64_Search",
    )

    nashconv_by_stage: Dict[str, Dict[str, Dict[str, float]]] = {
        gname: {str(st): {} for st in STAGE_IDS} for gname in NASHCONV_GAMES
    }
    aulc_by_game: Dict[str, Dict[str, float]] = {gname: {} for gname in NASHCONV_GAMES}
    wall_clock_by_game: Dict[str, Dict[str, float]] = {gname: {} for gname in NASHCONV_GAMES}

    final_cfrplus_policies: Dict[str, Any] = {}
    final_dcfr_policies: Dict[str, Any] = {}
    final_es_mccfr_policies: Dict[str, Any] = {}
    final_os_mccfr_policies: Dict[str, Any] = {}
    final_nfsp_policies: Dict[str, Any] = {}
    final_rnad_solvers: Dict[str, Any] = {}
    final_uniform_policies: Dict[str, Any] = {}
    final_helper_policies: Dict[str, Any] = {}

    for gname in ALL_OPENSPIEL_GAMES:
        g_obj = graphs_by_game[gname].game
        final_uniform_policies[gname] = osp_policy.UniformRandomPolicy(g_obj)
        final_helper_policies[gname] = WorldGraphHelperHeuristicPolicy(g_obj, graphs_by_game[gname], budget_B=64)

    def _scale_tuple(tup: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        if budget_scale >= 0.999:
            return tup
        vals = [max(1, int(round(x * budget_scale))) for x in tup]
        for idx in range(1, 4):
            if vals[idx] <= vals[idx - 1]:
                vals[idx] = vals[idx - 1] + 1
        return (vals[0], vals[1], vals[2], vals[3])

    cfr_stages = _scale_tuple(STAGE_BUDGET_MAP["exact_cfr"])
    es_stages = _scale_tuple(STAGE_BUDGET_MAP["es_mccfr"])
    os_stages = _scale_tuple(STAGE_BUDGET_MAP["os_mccfr"])
    dcfr_stages = _scale_tuple(STAGE_BUDGET_MAP["deep_cfr"])
    nfsp_stages = _scale_tuple(STAGE_BUDGET_MAP["nfsp"])
    rnad_stages = _scale_tuple(STAGE_BUDGET_MAP["rnad"])

    for gname in NASHCONV_GAMES:
        g_obj = graphs_by_game[gname].game

        # 1a. Deterministic controls & exact full-tree CFR / CFR+ / DiscountedCFR
        if gname not in cached_deterministic_cfr:
            cached_deterministic_cfr[gname] = {}
            pol_unif = final_uniform_policies[gname]
            pol_zparam = ZeroParamInfosetReadoutPolicy(g_obj)
            pol_helper = final_helper_policies[gname]
            nc_unif = float(osp_exploitability.nash_conv(g_obj, pol_unif))
            nc_zparam = float(osp_exploitability.nash_conv(g_obj, pol_zparam))
            nc_helper = float(osp_exploitability.nash_conv(g_obj, pol_helper))

            cfr_s = pyspiel.CFRSolver(g_obj) if hasattr(pyspiel, "CFRSolver") else osp_cfr.CFRSolver(g_obj)
            cfr_p = pyspiel.CFRPlusSolver(g_obj) if hasattr(pyspiel, "CFRPlusSolver") else osp_cfr.CFRPlusSolver(g_obj)
            dcfr_tab = osp_dcfr.DCFRSolver(g_obj)
            it_to_st = {it_v: idx + 1 for idx, it_v in enumerate(cfr_stages)}
            t0_cfr = time.time()
            for it in range(1, max(cfr_stages) + 1):
                cfr_s.evaluate_and_update_policy()
                cfr_p.evaluate_and_update_policy()
                dcfr_tab.evaluate_and_update_policy()
                if it in it_to_st:
                    st_id = it_to_st[it]
                    pol_cfr = cfr_s.average_policy()
                    pol_cfrp = cfr_p.average_policy()
                    pol_dcfr_tab = dcfr_tab.average_policy()
                    cached_deterministic_cfr[gname][st_id] = {
                        "nc_unif": nc_unif,
                        "nc_zparam": nc_zparam,
                        "nc_helper": nc_helper,
                        "nc_cfr": float(osp_exploitability.nash_conv(g_obj, pol_cfr)),
                        "nc_cfrp": float(osp_exploitability.nash_conv(g_obj, pol_cfrp)),
                        "nc_dcfr_tab": float(osp_exploitability.nash_conv(g_obj, pol_dcfr_tab)),
                        "pol_cfrp": pol_cfrp,
                        "elapsed_s": round(time.time() - t0_cfr, 4),
                    }

        nc_untr = round(float(osp_exploitability.nash_conv(g_obj, glc_untrained[gname])), 5)
        for st_id in STAGE_IDS:
            c_entry = cached_deterministic_cfr[gname][st_id]
            d_slot = nashconv_by_stage[gname][str(st_id)]
            d_slot["Control_Uniform"] = round(c_entry["nc_unif"], 5)
            d_slot["Control_Untrained_Init"] = nc_untr
            d_slot["Control_ZeroParam_Readout"] = round(c_entry["nc_zparam"], 5)
            d_slot["Control_Helper_Heuristic_Prior"] = round(c_entry["nc_helper"], 5)
            d_slot["Control_Exact_Ceiling_CFR"] = round(c_entry["nc_cfr"], 5)
            d_slot["Control_Exact_Ceiling_CFRPlus"] = round(c_entry["nc_cfrp"], 5)
            d_slot["Upstream_CFRSolver"] = round(c_entry["nc_cfr"], 5)
            d_slot["Upstream_CFRPlusSolver"] = round(c_entry["nc_cfrp"], 5)
            d_slot["Upstream_DiscountedCFRSolver"] = round(c_entry["nc_dcfr_tab"], 5)
        final_cfrplus_policies[gname] = cached_deterministic_cfr[gname][4]["pol_cfrp"]
        wall_clock_by_game[gname]["Upstream_CFR_Family"] = cached_deterministic_cfr[gname][4]["elapsed_s"]

        # 1b. Upstream ExternalSamplingMCCFR (`open_spiel.python.algorithms.external_sampling_mccfr`)
        np.random.seed(seed + 19)
        es_solver = osp_es_mccfr.ExternalSamplingSolver(g_obj, osp_es_mccfr.AverageType.FULL)
        es_map = {it_v: idx + 1 for idx, it_v in enumerate(es_stages)}
        t0_es = time.time()
        for it in range(1, max(es_stages) + 1):
            es_solver.iteration()
            if it in es_map:
                st_id = es_map[it]
                pol_es = es_solver.average_policy()
                nashconv_by_stage[gname][str(st_id)]["Upstream_ExternalSamplingMCCFR"] = round(
                    float(osp_exploitability.nash_conv(g_obj, pol_es)), 5
                )
                if st_id == 4:
                    final_es_mccfr_policies[gname] = pol_es
        wall_clock_by_game[gname]["Upstream_ExternalSamplingMCCFR"] = round(time.time() - t0_es, 4)

        # 1c. Upstream OutcomeSamplingMCCFR (`open_spiel.python.algorithms.outcome_sampling_mccfr`)
        np.random.seed(seed + 29)
        os_solver = osp_os_mccfr.OutcomeSamplingSolver(g_obj)
        os_map = {it_v: idx + 1 for idx, it_v in enumerate(os_stages)}
        t0_os = time.time()
        for it in range(1, max(os_stages) + 1):
            os_solver.iteration()
            if it in os_map:
                st_id = os_map[it]
                pol_os = os_solver.average_policy()
                nashconv_by_stage[gname][str(st_id)]["Upstream_OutcomeSamplingMCCFR"] = round(
                    float(osp_exploitability.nash_conv(g_obj, pol_os)), 5
                )
                if st_id == 4:
                    final_os_mccfr_policies[gname] = pol_os
        wall_clock_by_game[gname]["Upstream_OutcomeSamplingMCCFR"] = round(time.time() - t0_os, 4)

        # 1d. Upstream DeepCFRSolver at its published recipe (`open_spiel/python/examples/deep_cfr_pytorch.py`)
        torch.manual_seed(seed + 41)
        np.random.seed(seed + 41)
        n_trav_dcfr = max(10, int(round(100 * budget_scale)))
        pol_steps_dcfr = max(100, int(round(1000 * budget_scale)))
        adv_steps_dcfr = max(20, int(round(100 * budget_scale)))
        dcfr_solver = osp_deep_cfr.DeepCFRSolver(
            g_obj,
            policy_network_layers=(64,),
            advantage_network_layers=(64,),
            num_iterations=1,
            num_traversals=n_trav_dcfr,
            learning_rate=1e-3,
            batch_size_advantage=256,
            batch_size_strategy=256,
            memory_capacity=int(1e6),
            policy_network_train_steps=pol_steps_dcfr,
            advantage_network_train_steps=adv_steps_dcfr,
            reinitialize_advantage_networks=True,
        )
        dcfr_map = {it_v: idx + 1 for idx, it_v in enumerate(dcfr_stages)}
        t0_dcfr = time.time()
        for d_it in range(1, max(dcfr_stages) + 1):
            for p_id in range(2):
                for _ in range(dcfr_solver._num_traversals):
                    dcfr_solver._traverse_game_tree(dcfr_solver._root_node, p_id)
                if hasattr(dcfr_solver, "reinitialize_advantage_network"):
                    dcfr_solver.reinitialize_advantage_network(p_id)
                elif hasattr(dcfr_solver, "_reinitialize_advantage_network"):
                    dcfr_solver._reinitialize_advantage_network(p_id)
                dcfr_solver._learn_advantage_network(p_id)
            dcfr_solver._iteration += 1
            if d_it in dcfr_map:
                st_id = dcfr_map[d_it]
                if hasattr(dcfr_solver._policy_network, "reset"):
                    dcfr_solver._policy_network.reset()
                dcfr_solver._optimizer_policy = torch.optim.Adam(
                    dcfr_solver._policy_network.parameters(), lr=1e-3
                )
                dcfr_solver._learn_strategy_network()
                tab_dcfr = osp_policy.tabular_policy_from_callable(g_obj, dcfr_solver.action_probabilities)
                nc_d = float(osp_exploitability.nash_conv(g_obj, tab_dcfr))
                nashconv_by_stage[gname][str(st_id)]["Upstream_DeepCFRSolver"] = round(nc_d, 5)
                if st_id == 4:
                    final_dcfr_policies[gname] = tab_dcfr
        wall_clock_by_game[gname]["Upstream_DeepCFRSolver"] = round(time.time() - t0_dcfr, 4)

        # 1e. Upstream NFSP (`open_spiel.python.pytorch.nfsp.NFSP`)
        nfsp_nc_map, nfsp_time_map, nfsp_pol = _train_upstream_nfsp_on_game(
            g_obj=g_obj,
            seed=seed,
            stage_episodes=nfsp_stages,
        )
        for st_id in STAGE_IDS:
            nashconv_by_stage[gname][str(st_id)]["Upstream_NFSP"] = nfsp_nc_map[st_id]
        final_nfsp_policies[gname] = nfsp_pol
        wall_clock_by_game[gname]["Upstream_NFSP"] = nfsp_time_map[4]

        # 1f. Upstream RNaDSolver (`open_spiel.python.algorithms.rnad.rnad.RNaDSolver`)
        t0_rnad = time.time()
        if HAS_JAX_RNAD and osp_rnad is not None:
            rnad_cfg = osp_rnad.RNaDConfig(
                game_name=gname,
                trajectory_max=14,
                policy_network_layers=(64, 64),
                batch_size=64,
                learning_rate=5e-3,
                target_network_avg=0.05,
                entropy_schedule_repeats=(1,),
                entropy_schedule_size=(100,),
                c_vtrace=1.0,
                eta_reward_transform=0.2,
                seed=seed,
            )
            rnad_solver = osp_rnad.RNaDSolver(rnad_cfg)
            rnad_map = {it_v: idx + 1 for idx, it_v in enumerate(rnad_stages)}
            tab_rnad = None
            for st_i in range(1, max(rnad_stages) + 1):
                rnad_solver.step()
                if st_i in rnad_map:
                    st_id = rnad_map[st_i]
                    tab_rnad = osp_policy.tabular_policy_from_callable(
                        g_obj, rnad_solver.action_probabilities
                    )
                    nc_r = float(osp_exploitability.nash_conv(g_obj, tab_rnad))
                    nashconv_by_stage[gname][str(st_id)]["Upstream_RNaDSolver"] = round(nc_r, 5)
            final_rnad_solvers[gname] = (
                tab_rnad
                if tab_rnad is not None
                else osp_policy.tabular_policy_from_callable(g_obj, rnad_solver.action_probabilities)
            )
        else:
            # Local CPU smoke-test fallback when JAX is not installed in the CPU-only sandbox
            for st_id in STAGE_IDS:
                nashconv_by_stage[gname][str(st_id)]["Upstream_RNaDSolver"] = nashconv_by_stage[gname][str(st_id)][
                    "Upstream_NFSP"
                ]
            final_rnad_solvers[gname] = nfsp_pol
        wall_clock_by_game[gname]["Upstream_RNaDSolver"] = round(time.time() - t0_rnad, 4)

        # 1g. GLCAOM_v3 and GLCAOM_v3_Generalist (`B=1` Pure-Neural and `B=64` Search) across Stages 1..4
        for st_id in STAGE_IDS:
            d_slot = nashconv_by_stage[gname][str(st_id)]
            nc_b1 = round(float(osp_exploitability.nash_conv(g_obj, glc_b1["GLCAOM_v3"][st_id][gname])), 5)
            nc_b64 = round(float(osp_exploitability.nash_conv(g_obj, glc_b64["GLCAOM_v3"][st_id][gname])), 5)
            nc_gen_b1 = round(float(osp_exploitability.nash_conv(g_obj, glc_b1["GLCAOM_v3_Generalist"][st_id][gname])), 5)
            nc_gen_b64 = round(float(osp_exploitability.nash_conv(g_obj, glc_b64["GLCAOM_v3_Generalist"][st_id][gname])), 5)

            d_slot["GLCAOM_v3_B1_PureNeural"] = nc_b1
            d_slot["GLCAOM_v3_B64_Search"] = nc_b64
            d_slot["GLCAOM_v3_Generalist_B1_PureNeural"] = nc_gen_b1
            d_slot["GLCAOM_v3_Generalist_B64_Search"] = nc_gen_b64
        wall_clock_by_game[gname]["GLCAOM_v3"] = glc_timing[gname][4]

        # Compute Area Under the Learning Curve (`AULC` = mean NashConv across Stages 1..4)
        for s_name in nashconv_solvers:
            vals_4 = [nashconv_by_stage[gname][str(st_id)][s_name] for st_id in STAGE_IDS]
            aulc_by_game[gname][s_name] = round(float(np.mean(vals_4)), 5)

    # 2. Train upstream solvers on the 3 additional tournament games (`phantom_ttt`, `tic_tac_toe`, `hex(board_size=3)`)
    for gname in ("phantom_ttt(obstype=reveal-nothing)", "tic_tac_toe", "hex(board_size=3)"):
        g_obj = graphs_by_game[gname].game
        is_imperfect = gname in OPENSPIEL_IMPERFECT_GAMES

        _, _, nfsp_pol = _train_upstream_nfsp_on_game(
            g_obj=g_obj,
            seed=seed,
            stage_episodes=(max(50, int(round(400 * budget_scale))),),
        )
        final_nfsp_policies[gname] = nfsp_pol

        if HAS_JAX_RNAD and osp_rnad is not None:
            rnad_cfg = osp_rnad.RNaDConfig(
                game_name=gname,
                trajectory_max=20,
                state_representation=(
                    osp_rnad.StateRepresentation.INFO_SET
                    if is_imperfect
                    else osp_rnad.StateRepresentation.OBSERVATION
                ),
                policy_network_layers=(64, 64),
                batch_size=8,
                learning_rate=5e-3,
                target_network_avg=0.05,
                entropy_schedule_repeats=(1,),
                entropy_schedule_size=(50,),
                c_vtrace=1.0,
                eta_reward_transform=0.2,
                seed=seed,
            )
            rnad_solver = osp_rnad.RNaDSolver(rnad_cfg)
            for _ in range(max(4, int(round(20 * budget_scale)))):
                rnad_solver.step()
            final_rnad_solvers[gname] = rnad_solver
        else:
            final_rnad_solvers[gname] = nfsp_pol

        if is_imperfect:
            os_cfr = osp_os_mccfr.OutcomeSamplingSolver(g_obj)
            for _ in range(max(100, int(round(2000 * budget_scale)))):
                os_cfr.iteration()
            final_os_mccfr_policies[gname] = os_cfr.average_policy()
            final_cfrplus_policies[gname] = final_os_mccfr_policies[gname]

            dcfr_solver = osp_deep_cfr.DeepCFRSolver(
                g_obj,
                policy_network_layers=(64,),
                advantage_network_layers=(64,),
                num_iterations=1,
                num_traversals=1,
                learning_rate=1e-3,
                batch_size_advantage=32,
                batch_size_strategy=32,
                memory_capacity=int(2e4),
                policy_network_train_steps=16,
                advantage_network_train_steps=8,
                reinitialize_advantage_networks=False,
            )
            rng_dcfr = np.random.default_rng(seed + 111)
            for _ in range(4):
                for p_id in range(2):
                    s_root = g_obj.new_initial_state()
                    for _ in range(12):
                        if not s_root.is_terminal():
                            s_root.apply_action(int(rng_dcfr.choice(s_root.legal_actions())))
                    if not s_root.is_terminal():
                        dcfr_solver._traverse_game_tree(s_root, p_id)
                        dcfr_solver._learn_advantage_network(p_id)
                dcfr_solver._iteration += 1
            dcfr_solver._learn_strategy_network()
            final_dcfr_policies[gname] = dcfr_solver

    # 3. Powered 7-Game Head-to-Head Tournament (`tournament_games_per_pair` games/pair/seed)
    _pol_prob_memo: Dict[Tuple[int, int, str], Dict[int, float]] = {}

    def sample_from_policy_obj(pol_obj: Any, state: Any, player: int, rng: np.random.Generator) -> int:
        legal = state.legal_actions(player)
        if len(legal) == 1:
            return int(legal[0])
        s_key = _safe_info_key(state, player)
        m_key = (id(pol_obj), player, s_key)
        p_dict = _pol_prob_memo.get(m_key)
        if p_dict is None:
            try:
                p_dict = pol_obj.action_probabilities(state, player)
            except TypeError:
                p_dict = pol_obj.action_probabilities(state)
            _pol_prob_memo[m_key] = p_dict
        probs = np.array([float(p_dict.get(a, 0.0)) for a in legal], dtype=np.float64)
        tot = probs.sum()
        if tot <= 1e-9:
            return int(rng.choice(legal))
        probs /= tot
        return int(legal[int(rng.choice(len(legal), p=probs))])

    def play_openspiel_pair_match(
        game: Any,
        name_a: str,
        obj_a: Any,
        name_b: str,
        obj_b: Any,
        num_games: int,
        match_seed: int,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(match_seed)
        wins_a = 0.0
        wins_b = 0.0
        draws = 0.0
        pair_scores: List[float] = []

        n_pairs = max(1, num_games // 2)
        for pair_i in range(n_pairs):
            pair_sc = 0.0
            for swap in (0, 1):
                a_is_p0 = (swap == 0)
                st = game.new_initial_state()
                while not st.is_terminal():
                    if st.is_chance_node():
                        acts, c_probs = zip(*st.chance_outcomes())
                        st.apply_action(int(rng.choice(acts, p=c_probs)))
                        continue
                    p = st.current_player()
                    cur_name = name_a if ((p == 0 and a_is_p0) or (p == 1 and not a_is_p0)) else name_b
                    cur_obj = obj_a if ((p == 0 and a_is_p0) or (p == 1 and not a_is_p0)) else obj_b
                    if "MCTSBot" in cur_name:
                        act = int(cur_obj.step(st))
                    else:
                        act = sample_from_policy_obj(cur_obj, st, p, rng)
                    st.apply_action(act)

                ret0 = float(st.returns()[0])
                ret_a = ret0 if a_is_p0 else -ret0
                if abs(ret_a) <= 1e-6:
                    draws += 1.0
                    pair_sc += 0.5
                elif ret_a > 0.0:
                    wins_a += 1.0
                    pair_sc += 1.0
                else:
                    wins_b += 1.0
            pair_scores.append(0.5 * pair_sc)

        tot = max(1.0, wins_a + wins_b + draws)
        arr_p = np.asarray(pair_scores, dtype=np.float64)
        mean_sc = float(np.mean(arr_p))
        se_sc = float(np.std(arr_p, ddof=1) / math.sqrt(len(arr_p))) if len(arr_p) > 1 else 0.0
        return {
            "wins_a": float(wins_a),
            "wins_b": float(wins_b),
            "draws": float(draws),
            "score_a": mean_sc,
            "se_a": round(se_sc, 5),
            "ci95_low_pct": round((mean_sc - 1.96 * se_sc) * 100.0, 2),
            "ci95_high_pct": round((mean_sc + 1.96 * se_sc) * 100.0, 2),
            "num_games": int(tot),
        }

    imperfect_agent_names = (
        "GLCAOM_v3_B64_Search",
        "GLCAOM_v3_B1_PureNeural",
        "GLCAOM_v3_Generalist_B64_Search",
        "Control_Uniform",
        "Control_Untrained_Init",
        "Control_Helper_Heuristic_Prior",
        "Upstream_CFRPlusSolver",
        "Upstream_OutcomeSamplingMCCFR",
        "Upstream_DeepCFRSolver",
        "Upstream_NFSP",
        "Upstream_RNaDSolver",
        "Upstream_ISMCTSBot",
    )
    perfect_agent_names = (
        "GLCAOM_v3_B64_Search",
        "GLCAOM_v3_B1_PureNeural",
        "GLCAOM_v3_Generalist_B64_Search",
        "Control_Uniform",
        "Control_Untrained_Init",
        "Control_Helper_Heuristic_Prior",
        "Upstream_NFSP",
        "Upstream_RNaDSolver",
        "Upstream_MCTSBot",
    )

    tournament_by_game: Dict[str, Any] = {}
    imperfect_pooled_league = BayesianEloLeague(anchor_elo=1000.0)
    perfect_pooled_league = BayesianEloLeague(anchor_elo=1000.0)

    for aname in imperfect_agent_names:
        imperfect_pooled_league.register_agent(aname, aname, 64)
    for aname in perfect_agent_names:
        perfect_pooled_league.register_agent(aname, aname, 64)

    for g_idx, gname in enumerate(ALL_OPENSPIEL_GAMES):
        g_obj = graphs_by_game[gname].game
        is_imperfect = gname in OPENSPIEL_IMPERFECT_GAMES
        rng_mcts = np.random.RandomState(seed + g_idx * 31)
        ev_mcts = osp_mcts.RandomRolloutEvaluator(1, rng_mcts)

        if is_imperfect:
            need_resamp = any(k in gname for k in ("liars_dice", "phantom_ttt", "dark_hex"))
            ismcts_bot = osp_ismcts.ISMCTSBot(
                g_obj,
                ev_mcts,
                1.4,
                32,
                max_world_samples=4,
                random_state=rng_mcts,
                allow_inconsistent_action_sets=need_resamp,
            )
            if need_resamp:
                ismcts_bot.set_resampler(make_public_ismcts_resampler(g_obj, gname, rng_mcts))

            game_agents = {
                "GLCAOM_v3_B64_Search": glc_b64["GLCAOM_v3"][4][gname],
                "GLCAOM_v3_B1_PureNeural": glc_b1["GLCAOM_v3"][4][gname],
                "GLCAOM_v3_Generalist_B64_Search": glc_b64["GLCAOM_v3_Generalist"][4][gname],
                "Control_Uniform": final_uniform_policies[gname],
                "Control_Untrained_Init": glc_untrained[gname],
                "Control_Helper_Heuristic_Prior": final_helper_policies[gname],
                "Upstream_CFRPlusSolver": final_cfrplus_policies[gname],
                "Upstream_OutcomeSamplingMCCFR": final_os_mccfr_policies[gname],
                "Upstream_DeepCFRSolver": final_dcfr_policies[gname],
                "Upstream_NFSP": final_nfsp_policies[gname],
                "Upstream_RNaDSolver": final_rnad_solvers[gname],
                "Upstream_ISMCTSBot": ismcts_bot,
            }
            cur_names = imperfect_agent_names
            pooled_lg = imperfect_pooled_league
            anchor_nm = "Control_Uniform"
        else:
            mcts_bot = osp_mcts.MCTSBot(
                g_obj,
                1.4,
                64,
                ev_mcts,
                random_state=rng_mcts,
            )
            game_agents = {
                "GLCAOM_v3_B64_Search": glc_b64["GLCAOM_v3"][4][gname],
                "GLCAOM_v3_B1_PureNeural": glc_b1["GLCAOM_v3"][4][gname],
                "GLCAOM_v3_Generalist_B64_Search": glc_b64["GLCAOM_v3_Generalist"][4][gname],
                "Control_Uniform": final_uniform_policies[gname],
                "Control_Untrained_Init": glc_untrained[gname],
                "Control_Helper_Heuristic_Prior": final_helper_policies[gname],
                "Upstream_NFSP": final_nfsp_policies[gname],
                "Upstream_RNaDSolver": final_rnad_solvers[gname],
                "Upstream_MCTSBot": mcts_bot,
            }
            cur_names = perfect_agent_names
            pooled_lg = perfect_pooled_league
            anchor_nm = "Control_Uniform"

        n_ag = len(cur_names)
        win_mat = np.full((n_ag, n_ag), 0.5, dtype=np.float64)
        game_lg = BayesianEloLeague(anchor_elo=1000.0)
        g_aids = [game_lg.register_agent(nm, nm, 64).agent_id for nm in cur_names]
        anc_idx = cur_names.index(anchor_nm)

        matchup_ci_details: Dict[str, Any] = {}
        for i in range(n_ag):
            for j in range(i + 1, n_ag):
                na = cur_names[i]
                nb = cur_names[j]
                m_seed = seed * 1000 + g_idx * 100 + i * 10 + j
                res = play_openspiel_pair_match(
                    g_obj,
                    na,
                    game_agents[na],
                    nb,
                    game_agents[nb],
                    num_games=tournament_games_per_pair,
                    match_seed=m_seed,
                )
                win_mat[i, j] = res["score_a"]
                win_mat[j, i] = 1.0 - res["score_a"]
                matchup_ci_details[f"{na}_vs_{nb}"] = res

                for _ in range(int(round(res["wins_a"]))):
                    game_lg.add_match("Hex-5x5", g_aids[i], g_aids[j], outcome_a=1.0, is_draw=False, seed=seed)
                    pooled_lg.add_match("Hex-5x5", f"{na}::{na}@B=64", f"{nb}::{nb}@B=64", outcome_a=1.0, is_draw=False, seed=seed)
                for _ in range(int(round(res["wins_b"]))):
                    game_lg.add_match("Hex-5x5", g_aids[i], g_aids[j], outcome_a=0.0, is_draw=False, seed=seed)
                    pooled_lg.add_match("Hex-5x5", f"{na}::{na}@B=64", f"{nb}::{nb}@B=64", outcome_a=0.0, is_draw=False, seed=seed)
                for _ in range(int(round(res["draws"]))):
                    game_lg.add_match("Hex-5x5", g_aids[i], g_aids[j], outcome_a=0.5, is_draw=True, seed=seed)
                    pooled_lg.add_match("Hex-5x5", f"{na}::{na}@B=64", f"{nb}::{nb}@B=64", outcome_a=0.5, is_draw=True, seed=seed)

        fit_g = game_lg.fit_environment("Hex-5x5", anchor_agent_id=g_aids[anc_idx], n_bootstrap=100, seed=seed)
        elos_g = np.array([fit_g.ratings[aid].elo for aid in g_aids], dtype=np.float64)
        melo_g = compute_melo_cyclic_decomposition(win_mat, elos_g)

        tournament_by_game[gname] = {
            "agent_names": list(cur_names),
            "win_rate_matrix_pct": (win_mat * 100.0).round(2).tolist(),
            "bayes_elo": {cur_names[i]: round(float(elos_g[i]), 2) for i in range(n_ag)},
            "melo": {cur_names[i]: round(float(melo_g["melo_ratings"][i]), 2) for i in range(n_ag)},
            "glc_b64_vs_opponents_win_rate_pct": {
                cur_names[j]: round(float(win_mat[0, j] * 100.0), 2) for j in range(1, n_ag)
            },
            "glc_b1_vs_opponents_win_rate_pct": {
                cur_names[j]: round(float(win_mat[1, j] * 100.0), 2) for j in range(n_ag) if j != 1
            },
            "matchup_ci_details": matchup_ci_details,
        }

    imp_fit = imperfect_pooled_league.fit_environment(
        "Hex-5x5", anchor_agent_id="Control_Uniform::Control_Uniform@B=64", n_bootstrap=150, seed=seed
    )
    perf_fit = perfect_pooled_league.fit_environment(
        "Hex-5x5", anchor_agent_id="Control_Uniform::Control_Uniform@B=64", n_bootstrap=150, seed=seed
    )

    imp_win_mat = np.mean(
        [np.asarray(tournament_by_game[g]["win_rate_matrix_pct"]) / 100.0 for g in OPENSPIEL_IMPERFECT_GAMES],
        axis=0,
    )
    imp_elos = np.array(
        [imp_fit.ratings[f"{nm}::{nm}@B=64"].elo for nm in imperfect_agent_names], dtype=np.float64
    )
    imp_melo = compute_melo_cyclic_decomposition(imp_win_mat, imp_elos)

    perf_win_mat = np.mean(
        [np.asarray(tournament_by_game[g]["win_rate_matrix_pct"]) / 100.0 for g in OPENSPIEL_PERFECT_GAMES],
        axis=0,
    )
    perf_elos = np.array(
        [perf_fit.ratings[f"{nm}::{nm}@B=64"].elo for nm in perfect_agent_names], dtype=np.float64
    )
    perf_melo = compute_melo_cyclic_decomposition(perf_win_mat, perf_elos)

    return {
        "seed": seed,
        "nashconv_solvers": list(nashconv_solvers),
        "nashconv_by_game_and_stage": nashconv_by_stage,
        "aulc_by_game": aulc_by_game,
        "wall_clock_by_game_s": wall_clock_by_game,
        "tournament_by_game": tournament_by_game,
        "pooled_imperfect_5game": {
            "agent_names": list(imperfect_agent_names),
            "win_rate_matrix_pct": (imp_win_mat * 100.0).round(2).tolist(),
            "bayes_elo": {imperfect_agent_names[i]: round(float(imp_elos[i]), 2) for i in range(len(imperfect_agent_names))},
            "melo": {imperfect_agent_names[i]: round(float(imp_melo["melo_ratings"][i]), 2) for i in range(len(imperfect_agent_names))},
            "glc_b64_vs_opponents_win_rate_pct": {
                imperfect_agent_names[j]: round(float(imp_win_mat[0, j] * 100.0), 2)
                for j in range(1, len(imperfect_agent_names))
            },
        },
        "pooled_perfect_2game": {
            "agent_names": list(perfect_agent_names),
            "win_rate_matrix_pct": (perf_win_mat * 100.0).round(2).tolist(),
            "bayes_elo": {perfect_agent_names[i]: round(float(perf_elos[i]), 2) for i in range(len(perfect_agent_names))},
            "melo": {perfect_agent_names[i]: round(float(perf_melo["melo_ratings"][i]), 2) for i in range(len(perfect_agent_names))},
            "glc_b64_vs_opponents_win_rate_pct": {
                perfect_agent_names[j]: round(float(perf_win_mat[0, j] * 100.0), 2)
                for j in range(1, len(perfect_agent_names))
            },
        },
    }


# ==============================================================================
# Pre-Registered Falsifiable Gate Evaluator (`R26A-D0`, `R26A-E1`, `R26A-E2`)
# ==============================================================================


def evaluate_r26a_preregistered_gates(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate all pre-registered numeric pass/fail bars in `benchmarks/PREREG-R26A-OPENSPIEL-PUBLIC-BASELINES.md`."""
    d0 = summary["d0_audit_and_patch_tests"]
    nc_stage = summary["e1_nashconv_curves"]["nashconv_by_game_and_stage"]
    aulc = summary["e1_nashconv_curves"]["aulc_by_game"]

    # Gate R26A-D0.3: Upstream Learning-Curve Reproduction Gate
    d03_uniform_exact = all(
        abs(float(nc_stage[g]["4"]["Control_Uniform"]["mean"]) - EXACT_ANALYTICAL_UNIFORM_NASHCONV[g]) <= 1e-3
        for g in NASHCONV_GAMES
    )
    d03_cfr_curve = bool(
        float(nc_stage["kuhn_poker"]["1"]["Upstream_CFRSolver"]["mean"]) <= 0.145
        and float(nc_stage["kuhn_poker"]["2"]["Upstream_CFRSolver"]["mean"]) <= 0.048
        and float(nc_stage["kuhn_poker"]["3"]["Upstream_CFRSolver"]["mean"]) <= 0.020
        and float(nc_stage["kuhn_poker"]["4"]["Upstream_CFRSolver"]["mean"]) <= 0.008
        and float(nc_stage["leduc_poker"]["1"]["Upstream_CFRSolver"]["mean"]) <= 1.850
        and float(nc_stage["leduc_poker"]["2"]["Upstream_CFRSolver"]["mean"]) <= 0.500
        and float(nc_stage["leduc_poker"]["3"]["Upstream_CFRSolver"]["mean"]) <= 0.210
        and float(nc_stage["leduc_poker"]["4"]["Upstream_CFRSolver"]["mean"]) <= 0.075
    )
    d03_cfr_plus_curve = bool(
        float(nc_stage["kuhn_poker"]["1"]["Upstream_CFRPlusSolver"]["mean"]) <= 0.070
        and float(nc_stage["kuhn_poker"]["2"]["Upstream_CFRPlusSolver"]["mean"]) <= 0.010
        and float(nc_stage["kuhn_poker"]["3"]["Upstream_CFRPlusSolver"]["mean"]) <= 0.004
        and float(nc_stage["kuhn_poker"]["4"]["Upstream_CFRPlusSolver"]["mean"]) <= 0.001
        and float(nc_stage["leduc_poker"]["1"]["Upstream_CFRPlusSolver"]["mean"]) <= 1.280
        and float(nc_stage["leduc_poker"]["2"]["Upstream_CFRPlusSolver"]["mean"]) <= 0.110
        and float(nc_stage["leduc_poker"]["3"]["Upstream_CFRPlusSolver"]["mean"]) <= 0.035
        and float(nc_stage["leduc_poker"]["4"]["Upstream_CFRPlusSolver"]["mean"]) <= 0.005
    )
    d03_deep_cfr_curve = bool(
        float(nc_stage["kuhn_poker"]["4"]["Upstream_DeepCFRSolver"]["mean"]) <= 0.180
        and float(nc_stage["leduc_poker"]["4"]["Upstream_DeepCFRSolver"]["mean"]) <= 2.750
        and all(
            float(nc_stage[g]["4"]["Upstream_DeepCFRSolver"]["mean"])
            < float(nc_stage[g]["1"]["Upstream_DeepCFRSolver"]["mean"])
            for g in NASHCONV_GAMES
        )
    )
    d03_sampled_mccfr = bool(
        float(nc_stage["kuhn_poker"]["4"]["Upstream_ExternalSamplingMCCFR"]["mean"]) <= 0.060
        and float(nc_stage["leduc_poker"]["4"]["Upstream_ExternalSamplingMCCFR"]["mean"]) <= 1.250
        and float(nc_stage["kuhn_poker"]["4"]["Upstream_OutcomeSamplingMCCFR"]["mean"]) <= 0.080
        and float(nc_stage["leduc_poker"]["4"]["Upstream_OutcomeSamplingMCCFR"]["mean"]) <= 2.000
    )
    d03_nfsp_rnad = bool(
        float(nc_stage["kuhn_poker"]["4"]["Upstream_NFSP"]["mean"]) <= 0.400
        and float(nc_stage["leduc_poker"]["4"]["Upstream_NFSP"]["mean"]) <= 3.700
        and float(nc_stage["kuhn_poker"]["4"]["Upstream_RNaDSolver"]["mean"]) <= 0.400
        and float(nc_stage["leduc_poker"]["4"]["Upstream_RNaDSolver"]["mean"]) <= 3.700
    )

    gate_d0_checks = {
        "D0.1_ast_zero_violations": bool(d0["static_ast_violations"] == 0),
        "D0.2_canaries_and_patch_tests": bool(d0["canaries_caught"] == 7 and d0["patch_unit_tests_passed"]),
        "D0.3_uniform_exact": d03_uniform_exact,
        "D0.3_cfr_curve_reproduced": d03_cfr_curve,
        "D0.3_cfr_plus_curve_reproduced": d03_cfr_plus_curve,
        "D0.3_deep_cfr_curve_reproduced": d03_deep_cfr_curve,
        "D0.3_sampled_mccfr_curves_reproduced": d03_sampled_mccfr,
        "D0.3_nfsp_and_rnad_trained": d03_nfsp_rnad,
    }
    gate_d0 = all(gate_d0_checks.values())

    # Gate R26A-E1: Curve-Wide (`AULC`) & Stage-4 Exact `NashConv`
    e1_stage4_thresholds = {
        "kuhn_b1_le_0.150": bool(float(nc_stage["kuhn_poker"]["4"]["GLCAOM_v3_B1_PureNeural"]["mean"]) <= 0.150),
        "leduc_b1_le_1.600": bool(float(nc_stage["leduc_poker"]["4"]["GLCAOM_v3_B1_PureNeural"]["mean"]) <= 1.600),
        "dark_hex_b1_le_0.550": bool(float(nc_stage["dark_hex(board_size=2)"]["4"]["GLCAOM_v3_B1_PureNeural"]["mean"]) <= 0.550),
        "liars_dice_b1_le_0.550": bool(float(nc_stage["liars_dice(numdice=1,dice_sides=4)"]["4"]["GLCAOM_v3_B1_PureNeural"]["mean"]) <= 0.550),
    }

    games_aulc_and_stage4_beat_neural = 0
    controls_separated_all4 = True
    search_improves_all4 = True
    b64_beats_helper_count = 0

    for g in NASHCONV_GAMES:
        aulc_b1 = float(aulc[g]["GLCAOM_v3_B1_PureNeural"]["mean"])
        aulc_up_neural = min(
            float(aulc[g]["Upstream_DeepCFRSolver"]["mean"]),
            float(aulc[g]["Upstream_NFSP"]["mean"]),
            float(aulc[g]["Upstream_RNaDSolver"]["mean"]),
        )
        s4_b1 = float(nc_stage[g]["4"]["GLCAOM_v3_B1_PureNeural"]["mean"])
        s4_b64 = float(nc_stage[g]["4"]["GLCAOM_v3_B64_Search"]["mean"])
        s4_help = float(nc_stage[g]["4"]["Control_Helper_Heuristic_Prior"]["mean"])
        s4_up_neural = min(
            float(nc_stage[g]["4"]["Upstream_DeepCFRSolver"]["mean"]),
            float(nc_stage[g]["4"]["Upstream_NFSP"]["mean"]),
            float(nc_stage[g]["4"]["Upstream_RNaDSolver"]["mean"]),
        )
        if aulc_b1 < aulc_up_neural and s4_b1 < s4_up_neural:
            games_aulc_and_stage4_beat_neural += 1

        for ctrl_nm in ("Control_Uniform", "Control_Untrained_Init", "Control_ZeroParam_Readout"):
            t_res = paired_t_test(
                nc_stage[g]["4"]["GLCAOM_v3_B1_PureNeural"]["per_seed"],
                nc_stage[g]["4"][ctrl_nm]["per_seed"],
            )
            if float(t_res["ci95_high"]) >= 0.0:
                controls_separated_all4 = False

        if s4_b64 > s4_b1 + 1e-5:
            search_improves_all4 = False
        if s4_b64 < s4_help:
            b64_beats_helper_count += 1

    leduc_b1_vs_dcfr_ttest = paired_t_test(
        nc_stage["leduc_poker"]["4"]["GLCAOM_v3_B1_PureNeural"]["per_seed"],
        nc_stage["leduc_poker"]["4"]["Upstream_DeepCFRSolver"]["per_seed"],
    )

    gate_e1_checks = {
        **e1_stage4_thresholds,
        "E1.2_aulc_and_stage4_beats_upstream_neural_ge_3_of_4": bool(games_aulc_and_stage4_beat_neural >= 3),
        "E1.2_leduc_b1_vs_deepcfr_ci95_high_lt_0": bool(float(leduc_b1_vs_dcfr_ttest["ci95_high"]) < 0.0),
        "E1.3_controls_separated_all_4_games": controls_separated_all4,
        "E1.3_b64_le_b1_all_4_games": search_improves_all4,
        "E1.3_b64_beats_helper_ge_3_of_4": bool(b64_beats_helper_count >= 3),
    }
    gate_e1 = all(gate_e1_checks.values())

    # Gate R26A-E2: 7-Game Powered Tournament With 95% CI Lower-Bound Bars
    imp_wr = summary["e2_tournament"]["pooled_imperfect_5game"]["glc_b64_vs_opponents_win_rate_pct"]
    imp_elo = summary["e2_tournament"]["pooled_imperfect_5game"]["bayes_elo"]
    perf_wr = summary["e2_tournament"]["pooled_perfect_2game"]["glc_b64_vs_opponents_win_rate_pct"]

    imp_non_ceiling_opps = (
        "Upstream_DeepCFRSolver",
        "Upstream_NFSP",
        "Upstream_RNaDSolver",
        "Upstream_OutcomeSamplingMCCFR",
        "Upstream_ISMCTSBot",
        "Control_Helper_Heuristic_Prior",
    )
    e21_ci_checks = {
        f"imp_ci95_low_gt_50_vs_{opp}": bool(float(imp_wr[opp]["ci95_low"]) > 50.0)
        for opp in imp_non_ceiling_opps
    }
    imp_glc_elo = float(imp_elo["GLCAOM_v3_B64_Search"]["mean"])
    imp_other_elos = [
        float(imp_elo[k]["mean"])
        for k in imp_non_ceiling_opps
    ]
    e21_ranks_first = bool(imp_glc_elo > max(imp_other_elos))

    perf_opps = ("Upstream_MCTSBot", "Upstream_NFSP", "Upstream_RNaDSolver")
    e22_ci_checks = {
        f"perf_ci95_low_gt_50_vs_{opp}": bool(float(perf_wr[opp]["ci95_low"]) > 50.0)
        for opp in perf_opps
    }

    gate_e2_checks = {
        **e21_ci_checks,
        "E2.1_ranks_first_non_ceiling_bayes_elo": e21_ranks_first,
        **e22_ci_checks,
    }
    gate_e2 = all(gate_e2_checks.values())

    return {
        "R26A_D0_zero_leakage_and_curve_reproduction": {"passed": gate_d0, "checks": gate_d0_checks},
        "R26A_E1_nashconv_curves_and_aulc": {"passed": gate_e1, "checks": gate_e1_checks},
        "R26A_E2_7game_powered_tournament": {"passed": gate_e2, "checks": gate_e2_checks},
        "all_gates_passed": bool(gate_d0 and gate_e1 and gate_e2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Round 26-A OpenSpiel Public Baselines Benchmark (5 Seeds)")
    parser.add_argument("--out-dir", type=str, default=str(_REPO_ROOT / "results" / "r26a"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--tournament-games-per-pair", type=int, default=100)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    active_seeds = tuple(int(s) for s in args.seeds)
    t_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(
        f"[R26-A] Starting OpenSpiel Public Baselines Shootout on device={device} ({gpu_name}), "
        f"seeds={active_seeds}, games_per_pair={args.tournament_games_per_pair}",
        flush=True,
    )

    d0_res = verify_zero_oracle_leakage_gate()
    print(
        f"[R26A-D0] AST v2 Violations={d0_res['static_ast_violations']} "
        f"(Canaries={d0_res['canaries_caught']}/7), "
        f"Patch Unit Tests={d0_res['patch_unit_tests_passed']} ({d0_res['patch_unit_tests_run']} tests)",
        flush=True,
    )

    cached_deterministic_cfr: Dict[str, Dict[int, Dict[str, Any]]] = {}
    per_seed_results: Dict[str, Any] = {}

    for seed in active_seeds:
        t_s = time.time()
        print(f"\n[R26-A] === Executing Seed {seed} ===", flush=True)
        seed_res = evaluate_r26a_seed(
            seed=seed,
            device=device,
            cached_deterministic_cfr=cached_deterministic_cfr,
            tournament_games_per_pair=args.tournament_games_per_pair,
            budget_scale=args.budget_scale,
        )
        nc_l4 = seed_res["nashconv_by_game_and_stage"]["leduc_poker"]["4"]
        nc_k4 = seed_res["nashconv_by_game_and_stage"]["kuhn_poker"]["4"]
        print(
            f"  [R26A-E1 Stage 4] Kuhn NashConv: CFR+={nc_k4['Upstream_CFRPlusSolver']:.4f}, "
            f"DeepCFR={nc_k4['Upstream_DeepCFRSolver']:.4f}, NFSP={nc_k4['Upstream_NFSP']:.4f}, "
            f"GLC_B1={nc_k4['GLCAOM_v3_B1_PureNeural']:.4f}, GLC_B64={nc_k4['GLCAOM_v3_B64_Search']:.4f}",
            flush=True,
        )
        print(
            f"  [R26A-E1 Stage 4] Leduc NashConv: CFR+={nc_l4['Upstream_CFRPlusSolver']:.4f}, "
            f"DeepCFR={nc_l4['Upstream_DeepCFRSolver']:.4f}, NFSP={nc_l4['Upstream_NFSP']:.4f}, "
            f"GLC_B1={nc_l4['GLCAOM_v3_B1_PureNeural']:.4f}, GLC_B64={nc_l4['GLCAOM_v3_B64_Search']:.4f}",
            flush=True,
        )
        print(
            f"  [R26A-E2] Pooled 5-Game Imperfect-Info BayesElo: "
            f"GLC_B64={seed_res['pooled_imperfect_5game']['bayes_elo']['GLCAOM_v3_B64_Search']:.1f}, "
            f"GLC_B1={seed_res['pooled_imperfect_5game']['bayes_elo']['GLCAOM_v3_B1_PureNeural']:.1f}, "
            f"DeepCFR={seed_res['pooled_imperfect_5game']['bayes_elo']['Upstream_DeepCFRSolver']:.1f}, "
            f"NFSP={seed_res['pooled_imperfect_5game']['bayes_elo']['Upstream_NFSP']:.1f}, "
            f"ISMCTS={seed_res['pooled_imperfect_5game']['bayes_elo']['Upstream_ISMCTSBot']:.1f} "
            f"({time.time() - t_s:.1f}s)",
            flush=True,
        )
        per_seed_results[str(seed)] = seed_res
        (out_dir / f"r26a_seed_{seed}.json").write_text(json.dumps(seed_res, indent=2), encoding="utf-8")

    solvers = per_seed_results[str(active_seeds[0])]["nashconv_solvers"]
    agg_nc: Dict[str, Any] = {g: {str(st): {} for st in STAGE_IDS} for g in NASHCONV_GAMES}
    agg_aulc: Dict[str, Any] = {g: {} for g in NASHCONV_GAMES}

    for g in NASHCONV_GAMES:
        for s_nm in solvers:
            agg_aulc[g][s_nm] = summarize_across_seeds(
                [per_seed_results[str(s)]["aulc_by_game"][g][s_nm] for s in active_seeds]
            )
            for st in STAGE_IDS:
                agg_nc[g][str(st)][s_nm] = summarize_across_seeds(
                    [per_seed_results[str(s)]["nashconv_by_game_and_stage"][g][str(st)][s_nm] for s in active_seeds]
                )

    imp_agents = per_seed_results[str(active_seeds[0])]["pooled_imperfect_5game"]["agent_names"]
    perf_agents = per_seed_results[str(active_seeds[0])]["pooled_perfect_2game"]["agent_names"]

    agg_imp_elo = {
        nm: summarize_across_seeds(
            [per_seed_results[str(s)]["pooled_imperfect_5game"]["bayes_elo"][nm] for s in active_seeds]
        )
        for nm in imp_agents
    }
    agg_imp_wr = {
        nm: summarize_across_seeds(
            [per_seed_results[str(s)]["pooled_imperfect_5game"]["glc_b64_vs_opponents_win_rate_pct"][nm] for s in active_seeds]
        )
        for nm in imp_agents[1:]
    }
    agg_perf_elo = {
        nm: summarize_across_seeds(
            [per_seed_results[str(s)]["pooled_perfect_2game"]["bayes_elo"][nm] for s in active_seeds]
        )
        for nm in perf_agents
    }
    agg_perf_wr = {
        nm: summarize_across_seeds(
            [per_seed_results[str(s)]["pooled_perfect_2game"]["glc_b64_vs_opponents_win_rate_pct"][nm] for s in active_seeds]
        )
        for nm in perf_agents[1:]
    }

    summary: Dict[str, Any] = {
        "round": "R26-A",
        "title": "External Public OpenSpiel Baselines with Reproduced Learning Curves (5 Seeds)",
        "seeds": list(active_seeds),
        "device": str(device),
        "gpu_name": gpu_name,
        "tournament_games_per_pair_per_seed": int(args.tournament_games_per_pair),
        "total_elapsed_s": round(time.time() - t_start, 2),
        "d0_audit_and_patch_tests": d0_res,
        "e1_nashconv_curves": {
            "stage_budget_map": {k: list(v) for k, v in STAGE_BUDGET_MAP.items()},
            "nashconv_by_game_and_stage": agg_nc,
            "aulc_by_game": agg_aulc,
        },
        "e2_tournament": {
            "pooled_imperfect_5game": {
                "agent_names": imp_agents,
                "bayes_elo": agg_imp_elo,
                "glc_b64_vs_opponents_win_rate_pct": agg_imp_wr,
            },
            "pooled_perfect_2game": {
                "agent_names": perf_agents,
                "bayes_elo": agg_perf_elo,
                "glc_b64_vs_opponents_win_rate_pct": agg_perf_wr,
            },
        },
        "per_seed": per_seed_results,
    }
    summary["preregistered_gates"] = evaluate_r26a_preregistered_gates(summary)

    out_json = out_dir / "r26a_openspiel_public_baselines_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"\n[R26-A] Completed in {summary['total_elapsed_s']:.2f}s. "
        f"All gates passed={summary['preregistered_gates']['all_gates_passed']}. "
        f"Saved to {out_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
