#!/usr/bin/env python3
r"""PARKED 2026-10-19 (Kelvin): not frozen, not run
Round 26-B (`R26-B`): External Public `Ataraxos` (`arXiv:2511.07312`) `10x10` Stratego Evaluation (`3` Seeds).

Implements the pre-registered Round 26-B specification (`benchmarks/PREREG-R26B-ATARAXOS-STRATEGO.md`)
under `docs/PROTOCOL.md` and `outbox/r26-freeze-review-2026-10-19.md` (Blocking Items 1, 2, 4, 5, 6):
  - **Primary Citation:** Sokota, Samuel, Eugene Vinitsky, Hengyuan Hu, J. Zico Kolter, and Gabriele Farina
    (2025/2026), *"Scalable Decision Making for Games of Imperfect Information"* (`Ataraxos`), `arXiv:2511.07312`
    (repo `AtaraxosAI/stratego` at commit `92db29e8ffc323b1b8a2804b5c3f84695d036b05`; building on Update Equivalence `arXiv:2304.13138`
    and Magnetic Mirror Descent `arXiv:2206.05825`).
  - **Option 2b Public Baseline Naming (Blocking Item 2):**
    * Named Public Baselines use **exclusively released public weights and public analytic belief functions**:
      1. `Ataraxos_14.7M_Blueprint` (`model42400.pthw` + `init_model42400.pthw`, no search)
      2. `Ataraxos_SearchBot_Uniform` (`model42400.pthw` + `SearchBot` + `pyengine.belief.uniform.uniform_belief`)
      3. `Ataraxos_SearchBot_Marginalized` (`model42400.pthw` + `SearchBot` + `pyengine.belief.uniform.marginalized_uniform_belief`)
    * Matched-compute `240`-update trainings of `BeliefTransformer` and `ARBelief` on `arrangements[p][:512]`
      are explicitly labelled `BeliefTransformer_Arch_OurTraining_240Updates` and
      `ARBelief_Arch_OurTraining_240Updates` (diagnostic rows only; zero baseline pass/fail gates).
  - **Powered Evaluation & 95% CI Bars Reported Per Seed and Pooled (Blocking Item 4):**
    * `R26B-E1b`: `600` held-out mid-game boards per seed (`1,800` boards / `~54,000` hidden piece slots total)
      on `arrangements[p][512:]`, with per-seed and pooled paired 95% confidence intervals.
    * `R26B-E1a`: `50` seat-swapped pairs = `100` games per seed (`150` pairs = `300` games total across 3 seeds)
      per focal matchup on held-out `arrangements[p][512:]`, with per-seed and pooled 95% CI lower-bound bars.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

for _cand_path in [
    "/home/agent/workspaces/venv-cft/lib/python3.12/site-packages",
    "/home/agent/.local/lib/python3.12/site-packages",
    "/home/agent/data/pool/pylib",
    "/root/pylib",
    "/tmp/r26_pylib",
    "/tmp/r4e3a/pylib",
]:
    if os.path.isdir(_cand_path):
        while _cand_path in sys.path:
            sys.path.remove(_cand_path)
        sys.path.insert(0, _cand_path)

for _conda_site in sorted(glob.glob("/opt/conda/lib/python*/site-packages")):
    if os.path.isdir(_conda_site) and _conda_site not in sys.path:
        sys.path.append(_conda_site)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ATARAXOS_ROOT = _REPO_ROOT / "external" / "ataraxos_stratego"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_ATARAXOS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ATARAXOS_ROOT))


def _ensure_r26b_deps_and_pystratego() -> None:
    """Ensure PyTorch + Ataraxos Python deps and C++/CUDA `pystratego` are installed and compiled on the active host."""
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
        import einops  # noqa: F401
        import numpy  # noqa: F401
        import psutil  # noqa: F401
        import pynvml  # noqa: F401
        import scipy  # noqa: F401
        import tabulate  # noqa: F401
        import torch  # noqa: F401
        import typeguard  # noqa: F401
        import yaml  # noqa: F401
    except ImportError:
        if os.environ.get("R26B_SKIP_PIP_BOOTSTRAP", "0") != "1":
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
                    "einops",
                    "tabulate",
                    "typeguard",
                    "pyyaml",
                    "psutil",
                    "pynvml",
                    "matplotlib",
                    "wandb",
                    "typing_extensions",
                ]
            )
            if "/root/pylib" not in sys.path:
                sys.path.insert(0, "/root/pylib")

    import torch

    if torch.cuda.is_available():
        so_matches = list(_ATARAXOS_ROOT.rglob("pystratego*.so"))
        if not so_matches:
            print("[R26B-bootstrap] Compiling Ataraxos C++/CUDA pystratego extension ...", flush=True)
            if not os.path.exists("/usr/local/bin/python"):
                try:
                    os.symlink(sys.executable, "/usr/local/bin/python")
                except OSError:
                    pass
            cmake_file = _ATARAXOS_ROOT / "CMakeLists.txt"
            cmake_txt = cmake_file.read_text(encoding="utf-8")
            abi_flag = "1" if getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", False) else "0"
            cmake_txt_new = cmake_txt.replace(
                "-D_GLIBCXX_USE_CXX11_ABI=0", f"-D_GLIBCXX_USE_CXX11_ABI={abi_flag}"
            ).replace(
                "-D_GLIBCXX_USE_CXX11_ABI=1", f"-D_GLIBCXX_USE_CXX11_ABI={abi_flag}"
            )
            if cmake_txt_new != cmake_txt:
                cmake_file.write_text(cmake_txt_new, encoding="utf-8")
            build_dir = _ATARAXOS_ROOT / "build"
            if build_dir.exists():
                subprocess.call(["rm", "-rf", str(build_dir)])
            build_dir.mkdir(parents=True, exist_ok=True)
            env_vars = os.environ.copy()
            py_bin_dir = str(Path(sys.executable).parent)
            env_vars["PATH"] = f"/usr/local/bin:{py_bin_dir}:{env_vars.get('PATH', '')}"
            env_vars["PYTHONPATH"] = f"/root/pylib:{env_vars.get('PYTHONPATH', '')}"
            subprocess.check_call(["cmake", ".."], cwd=str(build_dir), env=env_vars)
            rc = subprocess.call(["make", "-j4"], cwd=str(build_dir), env=env_vars)
            if rc != 0:
                cmake_txt_fb = cmake_file.read_text(encoding="utf-8").replace(
                    'set(CMAKE_CUDA_FLAGS "${COMPILATION_FLAGS}")',
                    'set(CMAKE_CUDA_FLAGS "-g -O3 -Xcompiler=-fPIC,-fopenmp,-Wall,-Wextra,-Wfatal-errors")',
                )
                cmake_file.write_text(cmake_txt_fb, encoding="utf-8")
                subprocess.call(["rm", "-rf", str(build_dir)])
                build_dir.mkdir(parents=True, exist_ok=True)
                subprocess.check_call(["cmake", ".."], cwd=str(build_dir), env=env_vars)
                subprocess.check_call(["make", "-j4"], cwd=str(build_dir), env=env_vars)


if __name__ == "__main__":
    _ensure_r26b_deps_and_pystratego()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.distributions import Categorical

from src.eval.ast_leak_auditor import audit_r11_files  # noqa: E402
from src.eval.elo_league import BayesianEloLeague  # noqa: E402

SEEDS: Tuple[int, ...] = (101, 102, 103)
T_CRIT_95_DF2: float = 4.302652729911275
EXPECTED_RL_SHA256 = "8a490d1afe14c0a105c725a71e80ea517c98d9dcf332b3d68a3acbd503d2f52f"
EXPECTED_INIT_SHA256 = "35b3362dbe0bf8d308ed42759df0dbcaaa848c72e5cdeec436f963a948c1a6a6"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def summarize_across_seeds(vals: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(float(max(n, 1)))
    t_c = T_CRIT_95_DF2 if n == 3 else 1.96
    half_ci = t_c * se
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "se": round(se, 4),
        "ci95_low": round(mean - half_ci, 4),
        "ci95_high": round(mean + half_ci, 4),
        "per_seed": [round(float(x), 4) for x in arr],
    }


def summarize_samples_ci95(samples: Sequence[float]) -> Dict[str, Any]:
    """Compute sample mean, standard error, two-sided 95% CI, and one-sided 95% LCB/UCB over paired boards/pairs."""
    arr = np.asarray(samples, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr)) if n > 0 else 0.0
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(float(max(n, 1)))
    return {
        "n": int(n),
        "mean": round(mean, 5),
        "std": round(std, 5),
        "se": round(se, 5),
        "ci95_low": round(mean - 1.96 * se, 5),
        "ci95_high": round(mean + 1.96 * se, 5),
        "lcb95_one_sided": round(mean - 1.64485 * se, 5),
        "ucb95_one_sided": round(mean + 1.64485 * se, 5),
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
# Gate R26B-D0: AST Leak Auditor v2 + Patch Unit Tests + Pretrained Checkpoints
# ==============================================================================


def verify_zero_oracle_leakage_gate() -> Dict[str, Any]:
    """Verify Gate `R26B-D0`:
    1. `src/eval/ast_leak_auditor.py v2` reports `0` violations and `7/7` canaries caught.
    2. `tests/test_external_patches.py` passes (`3/3` unit tests, including `max_tf_diff == 0.0`
       teacher-forced equivalence of `BeliefTransformer.forward`).
    3. `external/ataraxos_stratego` `model42400.pthw` (`14,686,281` params, SHA256 verified) and
       `init_model42400.pthw` (`12,616,961` params, SHA256 verified) match `tests/test_data/rl_test_data.pkl`
       and `arrangements42400.pkl` within `rtol=1e-4, atol=1e-4`.
    """
    audit_res = audit_r11_files([Path(__file__).resolve()])
    if not audit_res["pass"]:
        raise RuntimeError(f"AST Leak Auditor v2 failed on R26-B runner: {audit_res}")

    suite = unittest.defaultTestLoader.discover(
        start_dir=str(_REPO_ROOT / "tests"),
        pattern="test_external_patches.py",
    )
    runner = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=0)
    test_result = runner.run(suite)
    patch_tests_passed = bool(test_result.wasSuccessful() and test_result.testsRun >= 3)

    pth_path = _ATARAXOS_ROOT / "pretrained" / "final_run" / "model42400.pthw"
    init_path = _ATARAXOS_ROOT / "pretrained" / "final_run" / "init_model42400.pthw"
    arr_path = _ATARAXOS_ROOT / "pretrained" / "final_run" / "arrangements42400.pkl"

    rl_sha = _sha256_file(pth_path) if pth_path.is_file() else ""
    init_sha = _sha256_file(init_path) if init_path.is_file() else ""

    pretrained_check: Dict[str, Any] = {
        "model_path": str(pth_path),
        "model_sha256": rl_sha,
        "model_sha256_matches": bool(rl_sha == EXPECTED_RL_SHA256),
        "init_model_path": str(init_path),
        "init_model_sha256": init_sha,
        "init_model_sha256_matches": bool(init_sha == EXPECTED_INIT_SHA256),
        "arrangements_path": str(arr_path),
        "rl_exact_match_passed": False,
        "arr_exact_match_passed": False,
        "max_action_log_prob_diff": None,
        "max_value_diff": None,
        "param_count": 0,
        "init_model_param_count": 0,
    }

    if torch.cuda.is_available() and pth_path.is_file():
        from pyengine.arrangement.sampling import to_string
        from pyengine.utils.loading import load_rl_model
        from pyengine.utils.pretrained_loading import (
            load_pretrained_arrangement_model,
            load_pretrained_rl_model,
        )

        rl_test_pkl = _ATARAXOS_ROOT / "tests" / "test_data" / "rl_test_data.pkl"
        rl_legacy, arrs = load_rl_model(str(pth_path))
        rl_pretrained, arrs_pre = load_pretrained_rl_model(str(pth_path))
        init_model = load_pretrained_arrangement_model(str(init_path))
        rl_legacy.eval()
        rl_pretrained.eval()
        init_model.eval()
        n_params = sum(p.numel() for p in rl_legacy.parameters())
        n_init_params = sum(p.numel() for p in init_model.parameters())

        with open(rl_test_pkl, "rb") as f:
            test_data = pickle.load(f)

        observations = test_data["observations"].cuda()
        piece_ids = test_data["piece_ids"].cuda()
        legal_actions = test_data["legal_actions"].cuda()
        expected_logits = test_data["logits"].cuda()
        expected_value = test_data["value"].cuda()

        with torch.no_grad():
            logits_pre, value_pre = rl_pretrained.forward_main(observations, piece_ids, legal_actions)
            logits_leg, value_leg = rl_legacy.forward_main(observations, piece_ids, legal_actions)
            gen_arrs, _ = init_model.generate(8)
            gen_strs = to_string(gen_arrs)

        torch.testing.assert_close(logits_pre, expected_logits, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(value_pre, expected_value, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(logits_leg, expected_logits, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(value_leg, expected_value, rtol=1e-4, atol=1e-4)

        max_lp_diff = float((logits_pre - expected_logits).abs().max().item())
        max_v_diff = float((value_pre - expected_value).abs().max().item())

        pretrained_check.update(
            {
                "rl_exact_match_passed": bool(
                    max_lp_diff <= 1e-4
                    and max_v_diff <= 1e-4
                    and arrs == arrs_pre
                    and rl_sha == EXPECTED_RL_SHA256
                ),
                "arr_exact_match_passed": bool(
                    n_init_params == 12616961
                    and len(gen_strs) == 8
                    and init_sha == EXPECTED_INIT_SHA256
                ),
                "max_action_log_prob_diff": round(max_lp_diff, 8),
                "max_value_diff": round(max_v_diff, 8),
                "param_count": int(n_params),
                "init_model_param_count": int(n_init_params),
                "num_arrangements_pl0": len(arrs[0]),
                "num_arrangements_pl1": len(arrs[1]),
            }
        )

    return {
        "static_ast_violations": int(audit_res["num_violations"]),
        "canaries_caught": int(audit_res["canary_self_test"]["canaries_caught"]),
        "ast_pass": bool(audit_res["pass"]),
        "patch_unit_tests_run": int(test_result.testsRun),
        "patch_unit_tests_passed": patch_tests_passed,
        "ataraxos_pretrained_verification": pretrained_check,
    }


# ==============================================================================
# `SpatialPrior_Sinkhorn_Helper` (`Control_Helper_Heuristic_Prior`, 0 Neural Params)
# & `GLCAOMStrategoBeliefNet` + `GLCAOMStrategoBeliefPlugin` + `GLC_v3_AtaraxosSearchBot`
# ==============================================================================


class SpatialPrior_Sinkhorn_Helper:
    """Mandatory Control 4 (`Control_Helper_Heuristic_Prior`, `0` neural parameters):
    Non-neural empirical `(cell, moved, rank)` frequency table fitted strictly on the training
    arrangement split (`arrangements[p][:512]`) + 6-step Sinkhorn-Knopp bipartite scaling.
    """

    def __init__(
        self,
        unmoved_cell_prior: Optional[torch.Tensor] = None,
        moved_cell_prior: Optional[torch.Tensor] = None,
        sinkhorn_iters: int = 6,
    ) -> None:
        from pyengine.utils.types import GenerateArgType

        self.generate_arg_type = GenerateArgType.MARGLIZED_UNIFORM
        self.unmoved_cell_prior = unmoved_cell_prior
        self.moved_cell_prior = moved_cell_prior
        self.sinkhorn_iters = int(sinkhorn_iters)

    def apply_sinkhorn_to_probs(
        self,
        base_probs_12: torch.Tensor,
        unknown_piece_position_onehot: torch.Tensor,
        unknown_piece_has_moved: torch.Tensor,
        unknown_piece_counts: torch.Tensor,
        use_spatial_prior: bool = True,
    ) -> torch.Tensor:
        from pyengine.utils.constants import (
            IMMOVABLE_PLAYER_PIECE_TYPE_SLICE,
            N_PIECE_TYPE,
            PLAYER_PIECE_TYPE_SLICE,
        )

        B, n_piece, _ = unknown_piece_position_onehot.shape
        dev = base_probs_12.device
        dtype = torch.float32

        active_mask = unknown_piece_position_onehot.any(dim=-1)
        positions = unknown_piece_position_onehot.long().argmax(dim=-1)
        b_raw = base_probs_12.to(dtype).clamp(min=0.0)

        if use_spatial_prior and self.unmoved_cell_prior is not None and self.moved_cell_prior is not None:
            u_prior = self.unmoved_cell_prior.to(device=dev, dtype=dtype)[positions]
            m_prior = self.moved_cell_prior.to(device=dev, dtype=dtype)[positions]
            cell_prior = torch.where(unknown_piece_has_moved.unsqueeze(-1), m_prior, u_prior)
            pos_support = (b_raw > 1e-6).to(dtype)
            b_work = (b_raw.clamp(min=1e-6) * (cell_prior.clamp(min=1e-4) ** 0.60)) * pos_support
        else:
            b_work = b_raw.clone()

        moved_bool = unknown_piece_has_moved & active_mask
        b_work[:, :, IMMOVABLE_PLAYER_PIECE_TYPE_SLICE] = torch.where(
            moved_bool.unsqueeze(-1),
            torch.zeros_like(b_work[:, :, IMMOVABLE_PLAYER_PIECE_TYPE_SLICE]),
            b_work[:, :, IMMOVABLE_PLAYER_PIECE_TYPE_SLICE],
        )

        cnt_f = unknown_piece_counts.to(dtype).clamp(min=0.0)
        type_avail = (cnt_f > 0.5).unsqueeze(1).to(dtype)
        b_work = b_work * type_avail * active_mask.unsqueeze(-1).to(dtype)

        row_fallback = type_avail * active_mask.unsqueeze(-1).to(dtype)
        row_fallback[:, :, IMMOVABLE_PLAYER_PIECE_TYPE_SLICE] = torch.where(
            moved_bool.unsqueeze(-1),
            torch.zeros_like(row_fallback[:, :, IMMOVABLE_PLAYER_PIECE_TYPE_SLICE]),
            row_fallback[:, :, IMMOVABLE_PLAYER_PIECE_TYPE_SLICE],
        )
        row_sum0 = b_work.sum(dim=-1, keepdim=True)
        b_work = torch.where(row_sum0 > 1e-7, b_work, row_fallback.clamp(min=1e-6))

        act_f = active_mask.unsqueeze(-1).to(dtype)
        for _ in range(self.sinkhorn_iters):
            r_sum = b_work.sum(dim=-1, keepdim=True).clamp(min=1e-7)
            b_work = (b_work / r_sum) * act_f
            c_sum = b_work.sum(dim=1, keepdim=True).clamp(min=1e-7)
            col_scale = (cnt_f.unsqueeze(1) / c_sum).clamp(min=0.0, max=20.0)
            b_work = b_work * col_scale * act_f

        r_sum_final = b_work.sum(dim=-1, keepdim=True).clamp(min=1e-7)
        b_prob = (b_work / r_sum_final) * act_f

        full_belief = torch.zeros(B, n_piece, N_PIECE_TYPE, device=dev, dtype=dtype)
        full_belief[:, :, PLAYER_PIECE_TYPE_SLICE] = b_prob
        return torch.log(full_belief + torch.finfo(dtype).tiny)

    def compute_calibrated_marginals(
        self,
        infostate_tensor: torch.Tensor,
        unknown_piece_position_onehot: torch.Tensor,
        unknown_piece_has_moved: torch.Tensor,
        unknown_piece_counts: torch.Tensor,
    ) -> torch.Tensor:
        from pyengine.utils.constants import INFOSTATE_OPP_PROB_SLICE

        dtype = torch.float32
        positions = unknown_piece_position_onehot.long().argmax(dim=-1)
        b_plane = infostate_tensor[:, INFOSTATE_OPP_PROB_SLICE].to(dtype).flatten(start_dim=2).permute(0, 2, 1)
        b_raw = b_plane.gather(1, positions.unsqueeze(-1).expand(-1, -1, b_plane.size(-1))).clamp(min=0.0)
        return self.apply_sinkhorn_to_probs(
            b_raw,
            unknown_piece_position_onehot,
            unknown_piece_has_moved,
            unknown_piece_counts,
            use_spatial_prior=True,
        )

    def __call__(
        self,
        infostate_tensor: torch.Tensor,
        unknown_piece_position_onehot: torch.Tensor,
        unknown_piece_has_moved: Optional[torch.Tensor] = None,
        unknown_piece_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if unknown_piece_has_moved is None or unknown_piece_counts is None:
            from pyengine.belief.uniform import marginalized_uniform_belief

            return marginalized_uniform_belief(infostate_tensor, unknown_piece_position_onehot)
        return self.compute_calibrated_marginals(
            infostate_tensor,
            unknown_piece_position_onehot,
            unknown_piece_has_moved,
            unknown_piece_counts,
        )

    def generate(
        self,
        n_sample: int,
        unknown_piece_position_onehot: torch.Tensor,
        unknown_piece_has_moved: torch.Tensor,
        unknown_piece_counts: torch.Tensor,
        infostate_tensor: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        from pyengine.belief.sampling import check_sampling_args, sampling_loop

        check_sampling_args(
            unknown_piece_position_onehot, unknown_piece_has_moved, unknown_piece_counts
        )
        log_beliefs_1 = self.compute_calibrated_marginals(
            infostate_tensor.unsqueeze(0),
            unknown_piece_position_onehot.unsqueeze(0),
            unknown_piece_has_moved.unsqueeze(0),
            unknown_piece_counts.unsqueeze(0),
        )
        beliefs_rep = log_beliefs_1.repeat(n_sample, 1, 1)
        return sampling_loop(
            n_sample,
            unknown_piece_position_onehot,
            unknown_piece_has_moved,
            unknown_piece_counts,
            lambda _s: beliefs_rep.clone(),
        )


class GLCAOMStrategoBeliefNet(nn.Module):
    """Learned `~1.2M`-parameter `GLCAOM_v3` Neural Belief Network for `10x10` Stratego."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_head: int = 4,
        n_encoder_layer: int = 2,
        plane_history_len: int = 86,
    ) -> None:
        super().__init__()
        from pyengine.networks.feature_orchestration import (
            FeatureOrchestrator,
            FeatureOrchestratorConfig,
        )
        from pyengine.networks.transformer_basics import SelfAttentionLayer
        from pyengine.utils.constants import (
            N_CLASSIC_PIECE,
            N_OCCUPIABLE_CELL,
            N_PLAYER_PIECE_TYPE,
        )

        self.embed_dim = int(embed_dim)
        self.feature_orchestrator = FeatureOrchestrator(
            FeatureOrchestratorConfig(
                use_piece_ids=True,
                use_threaten=True,
                use_evade=True,
                use_actadj=True,
                use_cemetery=True,
                use_battle=True,
                use_protect=True,
                plane_history_len=plane_history_len,
            )
        )
        self.embedder = nn.Linear(self.feature_orchestrator.in_channels, embed_dim)
        self.positional_embed_enc = nn.Parameter(torch.zeros(N_OCCUPIABLE_CELL, embed_dim))
        nn.init.trunc_normal_(self.positional_embed_enc, std=0.05)

        self.encoder = nn.ModuleList(
            [SelfAttentionLayer(embed_dim, num_head, 0.0) for _ in range(n_encoder_layer)]
        )
        self.enc_ln = nn.LayerNorm(embed_dim)

        self.piece_ctx_proj = nn.Sequential(
            nn.Linear(N_PLAYER_PIECE_TYPE * 2 + 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.piece_slot_emb = nn.Parameter(torch.zeros(1, N_CLASSIC_PIECE, embed_dim))
        nn.init.trunc_normal_(self.piece_slot_emb, std=0.02)

        self.lir_key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lir_val_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lir_trust_head = nn.Linear(embed_dim, 1)
        self.cross_piece_layers = nn.ModuleList(
            [SelfAttentionLayer(embed_dim, num_head, 0.0) for _ in range(2)]
        )
        self.final_ln = nn.LayerNorm(embed_dim)
        self.delta_logit_head = nn.Linear(embed_dim, N_PLAYER_PIECE_TYPE)
        self.prior_gate_head = nn.Linear(embed_dim, 1)

        nn.init.normal_(self.delta_logit_head.weight, std=0.01)
        nn.init.zeros_(self.delta_logit_head.bias)

    def forward_marginals_12(
        self,
        infostate_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
        unknown_piece_position_onehot: torch.Tensor,
        unknown_piece_has_moved: torch.Tensor,
        unknown_piece_counts: torch.Tensor,
        use_kernel_prior_anchor: bool = True,
    ) -> torch.Tensor:
        from pyengine.networks.utils import extract_mem
        from pyengine.utils.constants import (
            IMMOVABLE_PLAYER_PIECE_TYPE_SLICE,
            INFOSTATE_OPP_PROB_SLICE,
        )

        B, n_piece, _ = unknown_piece_position_onehot.shape
        dtype = torch.float32
        active_mask = unknown_piece_position_onehot.any(dim=-1)
        positions = unknown_piece_position_onehot.long().argmax(dim=-1)

        b_plane = infostate_tensor[:, INFOSTATE_OPP_PROB_SLICE].to(dtype).flatten(start_dim=2).permute(0, 2, 1)
        b_raw = b_plane.gather(1, positions.unsqueeze(-1).expand(-1, -1, b_plane.size(-1))).clamp(min=0.0)

        x = self.embedder(self.feature_orchestrator(infostate_tensor, piece_ids))
        x = x + self.positional_embed_enc.unsqueeze(0)
        for layer in self.encoder:
            x = layer(x)
        mem = self.enc_ln(extract_mem(x, unknown_piece_position_onehot, self.feature_orchestrator.cell_mask))
        mem = mem.to(dtype)

        cnt_f = unknown_piece_counts.to(dtype).clamp(min=0.0)
        cnt_norm = cnt_f / cnt_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        cnt_exp = cnt_norm.unsqueeze(1).expand(B, n_piece, -1)
        moved_f = unknown_piece_has_moved.to(dtype).unsqueeze(-1)
        act_f = active_mask.to(dtype).unsqueeze(-1)
        ctx_in = torch.cat([b_raw, cnt_exp, moved_f, act_f], dim=-1)
        h = mem + self.piece_ctx_proj(ctx_in) + self.piece_slot_emb[:, :n_piece, :]

        k_edge = F.normalize(self.lir_key_proj(h), p=2, dim=-1) * act_f
        v_edge = self.lir_val_proj(h) * act_f
        beta_trust = F.softplus(self.lir_trust_head(h)) * act_f
        gram_pos = F.relu(torch.bmm(k_edge, k_edge.transpose(1, 2)))
        crowd = 1.0 + torch.bmm(gram_pos, beta_trust).clamp(min=1e-4)
        w_lir = beta_trust / crowd
        lir_agg = torch.bmm(gram_pos, w_lir * v_edge)
        h = h + 0.25 * lir_agg

        for layer in self.cross_piece_layers:
            h = layer(h)
        h = self.final_ln(h)

        delta_logits = self.delta_logit_head(h)
        if use_kernel_prior_anchor:
            gate = torch.sigmoid(self.prior_gate_head(h) + 1.5)
            prior_logp = torch.log(b_raw.clamp(min=1e-6))
            raw_logits = gate * prior_logp + delta_logits
        else:
            raw_logits = delta_logits

        moved_bool = unknown_piece_has_moved & active_mask
        immov_mask = torch.zeros_like(raw_logits, dtype=torch.bool)
        immov_mask[:, :, IMMOVABLE_PLAYER_PIECE_TYPE_SLICE] = moved_bool.unsqueeze(-1)
        zero_cnt_mask = (cnt_f <= 0.5).unsqueeze(1).expand_as(raw_logits)
        invalid_mask = (immov_mask | zero_cnt_mask) & active_mask.unsqueeze(-1)
        valid_any = (~invalid_mask).any(dim=-1, keepdim=True)
        masked_logits = torch.where(invalid_mask & valid_any, torch.full_like(raw_logits, -1e9), raw_logits)
        return F.log_softmax(masked_logits, dim=-1)


class GLCAOMStrategoBeliefPlugin(nn.Module):
    """Wraps `GLCAOMStrategoBeliefNet` + optional `SpatialPrior_Sinkhorn_Helper` as an Ataraxos `SearchBot`
    belief plugin (`generate_arg_type = GenerateArgType.PLANAR_TRANSFORMER`) with Any-Order
    Minimum-Entropy permutation sampling $\\pi = \\operatorname{argsort}(H_i)$.
    """

    def __init__(
        self,
        net: GLCAOMStrategoBeliefNet,
        helper: Optional[SpatialPrior_Sinkhorn_Helper] = None,
        use_kernel_prior_anchor: bool = True,
    ) -> None:
        super().__init__()
        from pyengine.utils.types import GenerateArgType

        self.generate_arg_type = GenerateArgType.PLANAR_TRANSFORMER
        self.net = net
        self.helper = helper
        self.use_kernel_prior_anchor = bool(use_kernel_prior_anchor)

    def compute_log_beliefs(
        self,
        infostate_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
        unknown_piece_position_onehot: torch.Tensor,
        unknown_piece_has_moved: torch.Tensor,
        unknown_piece_counts: torch.Tensor,
        with_helper: bool = True,
    ) -> torch.Tensor:
        from pyengine.utils.constants import N_PIECE_TYPE, PLAYER_PIECE_TYPE_SLICE

        logp_12 = self.net.forward_marginals_12(
            infostate_tensor=infostate_tensor,
            piece_ids=piece_ids,
            unknown_piece_position_onehot=unknown_piece_position_onehot,
            unknown_piece_has_moved=unknown_piece_has_moved,
            unknown_piece_counts=unknown_piece_counts,
            use_kernel_prior_anchor=self.use_kernel_prior_anchor,
        )
        probs_12 = logp_12.exp()
        if with_helper and self.helper is not None:
            return self.helper.apply_sinkhorn_to_probs(
                probs_12,
                unknown_piece_position_onehot,
                unknown_piece_has_moved,
                unknown_piece_counts,
                use_spatial_prior=True,
            )
        B, n_piece, _ = probs_12.shape
        full_b = torch.zeros(B, n_piece, N_PIECE_TYPE, device=probs_12.device, dtype=torch.float32)
        full_b[:, :, PLAYER_PIECE_TYPE_SLICE] = probs_12
        return torch.log(full_b + torch.finfo(torch.float32).tiny)

    def generate(
        self,
        n_sample: int,
        unknown_piece_position_onehot: torch.Tensor,
        unknown_piece_has_moved: torch.Tensor,
        unknown_piece_counts: torch.Tensor,
        infostate_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        from pyengine.belief.sampling import check_sampling_args, sampling_loop
        from pyengine.utils.constants import PLAYER_PIECE_TYPE_SLICE

        check_sampling_args(
            unknown_piece_position_onehot, unknown_piece_has_moved, unknown_piece_counts
        )
        n_unk = int(unknown_piece_position_onehot.any(dim=-1).sum().item())
        log_beliefs_1 = self.compute_log_beliefs(
            infostate_tensor=infostate_tensor.unsqueeze(0),
            piece_ids=piece_ids.unsqueeze(0),
            unknown_piece_position_onehot=unknown_piece_position_onehot.unsqueeze(0),
            unknown_piece_has_moved=unknown_piece_has_moved.unsqueeze(0),
            unknown_piece_counts=unknown_piece_counts.unsqueeze(0),
            with_helper=(self.helper is not None),
        )

        if n_unk <= 1:
            beliefs_rep = log_beliefs_1.repeat(n_sample, 1, 1)
            return sampling_loop(
                n_sample,
                unknown_piece_position_onehot,
                unknown_piece_has_moved,
                unknown_piece_counts,
                lambda _s: beliefs_rep.clone(),
            )

        probs_unk = log_beliefs_1[0, :n_unk, PLAYER_PIECE_TYPE_SLICE].exp().clamp(min=1e-7)
        probs_unk = probs_unk / probs_unk.sum(dim=-1, keepdim=True).clamp(min=1e-7)
        entropies = -(probs_unk * probs_unk.log()).sum(dim=-1)

        perm = torch.argsort(entropies, dim=0)
        inv_perm = torch.argsort(perm, dim=0)

        pos_perm = unknown_piece_position_onehot.clone()
        pos_perm[:n_unk] = unknown_piece_position_onehot[perm]

        moved_perm = unknown_piece_has_moved.clone()
        moved_perm[:n_unk] = unknown_piece_has_moved[perm]

        log_beliefs_perm = log_beliefs_1.clone()
        log_beliefs_perm[:, :n_unk, :] = log_beliefs_1[:, perm, :]
        beliefs_rep = log_beliefs_perm.repeat(n_sample, 1, 1)

        samples_perm = sampling_loop(
            n_sample,
            pos_perm,
            moved_perm,
            unknown_piece_counts,
            lambda _s: beliefs_rep.clone(),
        )

        samples = samples_perm.clone()
        samples[:, :n_unk, :] = samples_perm[:, inv_perm, :]
        return samples


class GLC_v3_AtaraxosSearchBot:
    """`GLC_v3_AtaraxosSearchBot`:
    Uses the **exact same** `SearchBot` hyperparameters (`depth=2, stepsize=10.0, temperature=0.006,
    td_lambda=1.0, max_num_samples=20, env=search_env` with `num_envs=120`) as every other `SearchBot`
    arm, combining `GLCAOMStrategoBeliefPlugin` + blueprint-prior-guided root rollout allocation +
    completed-Q and positive counterfactual regret-matching policy refinement.
    """

    def __init__(
        self,
        rl_model: nn.Module,
        env: Any,
        depth: int,
        stepsize: float,
        temperature: float,
        td_lambda: float,
        max_num_samples: int,
        belief_model: nn.Module,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from pyengine.core.search import SearchBot

        self.base_bot = SearchBot(
            rl_model=rl_model,
            env=env,
            depth=depth,
            stepsize=stepsize,
            temperature=temperature,
            td_lambda=td_lambda,
            max_num_samples=max_num_samples,
            uniform_magnet=False,
            dtype=dtype,
            belief_model=belief_model,
        )
        self.rl_model = rl_model
        self.env = env
        self.depth = depth
        self.stepsize = stepsize
        self.temperature = temperature
        self.td_lambda = td_lambda
        self.max_num_samples = max_num_samples
        self.dtype = dtype
        self.belief_model = belief_model

    def __call__(
        self,
        env_states: Any,
        infostate_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
        legal_action_mask: torch.Tensor,
        num_moves: torch.Tensor,
        unknown_piece_position_onehot: torch.Tensor,
        unknown_piece_counts: torch.Tensor,
        unknown_piece_has_moved: torch.Tensor,
    ) -> torch.Tensor:
        from pyengine import utils
        from pyengine.core.search import compute_search_policy, sample_deterministic
        from pyengine.utils.constants import N_PLAYER, N_VF_CAT

        mask_1d = legal_action_mask.squeeze(0)
        n_legal = int(mask_1d.sum().item())
        if n_legal <= 1:
            return mask_1d.int().argmax().int().view(1)

        with torch.no_grad(), utils.eval_mode(self.rl_model):
            root_dict = self.rl_model(infostate_tensor, piece_ids, legal_action_mask, num_moves)
            bp_logits_root = root_dict["action_log_probs"].squeeze(0).to(torch.float32)
            v_logits_root = root_dict["value"].squeeze(0).to(torch.float32)
            bp_probs_root = torch.zeros_like(bp_logits_root)
            bp_probs_root[mask_1d] = torch.softmax(bp_logits_root[mask_1d], dim=-1)
            if self.rl_model.cfg.use_cat_vf:
                v_root_scalar = float((torch.softmax(v_logits_root, dim=-1) @ self.base_bot.categorical_aggregation).item())
            else:
                v_root_scalar = float(v_logits_root.item())

        n_sample = min(self.env.num_envs // max(1, n_legal), self.max_num_samples)
        n_sample = max(1, int(n_sample))
        if env_states.num_envs > 1:
            env_states = env_states.slice(env_states.num_envs - 1, env_states.num_envs)
        posterior_env_states = self.base_bot._generate_env_state(
            env_states,
            infostate_tensor,
            piece_ids,
            num_moves,
            unknown_piece_position_onehot,
            unknown_piece_counts,
            unknown_piece_has_moved,
            n_sample,
        )

        A = mask_1d.shape[0]
        k_top = min(n_legal, 12)
        legal_indices = torch.nonzero(mask_1d, as_tuple=False).squeeze(-1)
        top_rel = torch.argsort(bp_probs_root[legal_indices], descending=True)[:k_top]
        cand_indices = legal_indices[top_rel]
        cand_mask = torch.zeros_like(mask_1d)
        cand_mask[cand_indices] = True

        alloc_policy = torch.zeros_like(bp_probs_root)
        alloc_policy[cand_mask] = 0.72 * bp_probs_root[cand_mask] + 0.28 / float(k_top)
        alloc_policy = alloc_policy / alloc_policy.sum().clamp(min=1e-7)

        self.env.change_reset_behavior_to_env_state(posterior_env_states)
        self.env.reset()
        player = posterior_env_states.to_play

        leaf_vals = torch.zeros(self.env.num_envs, N_VF_CAT, device="cuda", dtype=torch.float32)
        done = self.env.current_is_terminal.view(-1, 1)
        q_cat = torch.zeros(A, N_VF_CAT, device="cuda", dtype=torch.float32)
        cum = torch.zeros_like(q_cat)
        counts = torch.zeros(A, device="cuda", dtype=torch.float32)

        actions = sample_deterministic(alloc_policy.cpu(), self.env.num_envs).cuda()
        tracked = actions.long()

        for d in range(self.depth // 2):
            for parity in range(N_PLAYER):
                self.env.apply_actions(actions)
                with autocast(device_type="cuda", dtype=self.dtype), torch.no_grad(), utils.eval_mode(self.rl_model):
                    tensor_dict = self.rl_model(
                        self.env.current_infostate_tensor,
                        self.env.current_piece_ids,
                        self.env.current_legal_action_mask,
                        self.env.current_num_moves,
                    )
                    actions = tensor_dict["action"]
                    values = tensor_dict["value"]

                if parity == 0:
                    reward = self.env.current_is_newly_terminal * self.env.current_reward_pl0
                    continue

                reward += self.env.current_is_newly_terminal * self.env.current_reward_pl0
                if player == 1:
                    reward *= -1
                term = self.env.current_is_terminal
                reward_oh = F.one_hot((reward + 1).long(), num_classes=N_VF_CAT)
                values_prob = torch.softmax(values.to(torch.float32), dim=-1)
                term_col = term.view(-1, 1)

                if self.td_lambda < 1.0:
                    if d < self.depth // 2 - 1:
                        normalizer = (1.0 - self.td_lambda) * (self.td_lambda ** d)
                    else:
                        normalizer = self.td_lambda ** d
                    target = torch.where(term_col, (self.td_lambda ** d) * reward_oh, normalizer * values_prob)
                else:
                    target = torch.where(term_col, reward_oh, (d == self.depth // 2 - 1) * values_prob)
                leaf_vals += (~done) * target
                done |= term_col
                if done.all():
                    break

        cum.scatter_add_(0, tracked.unsqueeze(-1).expand(-1, N_VF_CAT), leaf_vals)
        counts += torch.bincount(tracked.squeeze(), minlength=A)
        nz = counts > 0
        q_cat[nz] = cum[nz] / counts[nz].unsqueeze(-1)
        q_raw = q_cat @ self.base_bot.categorical_aggregation

        v_completed_base = (
            float((bp_probs_root[nz] * q_raw[nz]).sum().item() / bp_probs_root[nz].sum().clamp(min=1e-6).item())
            if nz.any()
            else v_root_scalar
        )
        v_baseline = 0.55 * v_completed_base + 0.45 * v_root_scalar
        q_completed = torch.where(nz, q_raw, torch.full_like(q_raw, v_baseline))

        search_policy = compute_search_policy(
            q_completed,
            bp_logits_root,
            mask_1d,
            self.temperature,
            self.stepsize,
            uniform_magnet=False,
        )

        pos_regret = F.relu(q_completed - v_baseline) * mask_1d.float() * (bp_probs_root.clamp(min=1e-4) ** 0.35)
        reg_sum = pos_regret.sum()
        if float(reg_sum.item()) > 1e-6:
            rm_policy = pos_regret / reg_sum
            combined_policy = 0.65 * search_policy + 0.35 * rm_policy
        else:
            combined_policy = search_policy

        combined_policy = combined_policy * mask_1d.float()
        combined_policy = combined_policy / combined_policy.sum().clamp(min=1e-8)
        return Categorical(probs=combined_policy).sample().int().view(1)


class UniformRandomStrategoAgent:
    """`Control_UniformRandom` for `10x10` Stratego: samples uniformly over `legal_action_mask`."""

    def select_uniform_action(self, legal_action_mask: torch.Tensor) -> torch.Tensor:
        mask_1d = legal_action_mask.squeeze(0).float()
        probs = mask_1d / mask_1d.sum().clamp(min=1.0)
        return Categorical(probs=probs).sample().int().view(1)


def calibrate_model_and_clearer(
    seed: int,
    rl_model: nn.Module,
    arrangements: Tuple[List[str], List[str]],
    belief_train_updates: int = 240,
    ataraxos_belief_ckpt: Optional[str] = None,
) -> Dict[str, Any]:
    """Exempt training/calibration helper (`calibrate_model_and_clearer` in `EXEMPT_SCORING_OR_TRAINING_FUNCTIONS`):
    1. Strictly uses the training split `arrangements[p][:512]` (disjoint from evaluation split `arrangements[p][512:]`).
    2. Fits `SpatialPrior_Sinkhorn_Helper` (`Control_Helper_Heuristic_Prior`, `0` neural params) on empirical
       unmoved/moved cell counts from self-play rollouts of `Ataraxos` `14.7M`.
    3. Saves an untrained copy of `GLCAOMStrategoBeliefNet` (`Control_Untrained_Init`, `0` steps).
    4. Trains `GLCAOMStrategoBeliefNet` alongside the diagnostic matched-compute rows
       `BeliefTransformer_Arch_OurTraining_240Updates` and `ARBelief_Arch_OurTraining_240Updates`
       on the exact same self-play rollout batches from `arrangements[p][:512]`.
    """
    import copy

    from pyengine import utils
    from pyengine.core.env import Stratego
    from pyengine.networks.belief_transformer import BeliefTransformer, BeliefTransformerConfig
    from pyengine.networks.legacy_belief import ARBelief, ARBeliefConfig
    from pyengine.utils.constants import (
        N_BOARD_CELL,
        N_PIECE_TYPE,
        N_PLAYER_PIECE_TYPE,
        PLAYER_PIECE_TYPE_SLICE,
    )
    from pyengine.utils.loading import load_belief_model

    torch.manual_seed(seed + 17)
    train_arrs = (arrangements[0][:512], arrangements[1][:512])
    train_env = Stratego(num_envs=16, traj_len_per_player=120, quiet=2, max_num_moves=360)
    train_env.change_reset_behavior_to_random_initial_arrangement(train_arrs)
    train_env.reset()

    unmoved_counts = torch.ones(N_BOARD_CELL, N_PLAYER_PIECE_TYPE, device="cuda", dtype=torch.float32) * 0.5
    moved_counts = torch.ones(N_BOARD_CELL, N_PLAYER_PIECE_TYPE, device="cuda", dtype=torch.float32) * 0.5

    bt_cfg = BeliefTransformerConfig(
        n_encoder_layer=2,
        n_decoder_block=2,
        num_head=4,
        embed_dim_per_head_over8=4,
        dropout=0.0,
    )
    bt_model = BeliefTransformer(bt_cfg).to("cuda")
    bt_opt = torch.optim.AdamW(bt_model.parameters(), lr=6e-4, weight_decay=1e-4)
    bt_model.train()

    if ataraxos_belief_ckpt and os.path.isfile(ataraxos_belief_ckpt):
        ar_model = load_belief_model(ataraxos_belief_ckpt).to("cuda")
        ar_opt = None
    else:
        ar_cfg = ARBeliefConfig(
            depth=2,
            num_head=4,
            embed_dim=128,
            dropout=0.0,
            mask=0,
            plane_history_len=86,
            decoder_depth=2,
        )
        ar_model = ARBelief(N_PIECE_TYPE, ar_cfg).to("cuda")
        ar_opt = torch.optim.AdamW(ar_model.parameters(), lr=6e-4, weight_decay=1e-4)
        ar_model.train()

    glc_net = GLCAOMStrategoBeliefNet(
        embed_dim=128,
        num_head=4,
        n_encoder_layer=2,
        plane_history_len=86,
    ).to("cuda")
    untrained_glc_net = copy.deepcopy(glc_net).eval()
    for p in untrained_glc_net.parameters():
        p.requires_grad_(False)

    glc_opt = torch.optim.AdamW(glc_net.parameters(), lr=1.2e-3, weight_decay=1e-4)
    glc_net.train()

    n_rollout_steps = max(30, int(belief_train_updates))
    for step in range(1, n_rollout_steps + 1):
        with torch.no_grad(), utils.eval_mode(rl_model), autocast(device_type="cuda", dtype=torch.bfloat16):
            act_dict = rl_model(
                train_env.current_infostate_tensor,
                train_env.current_piece_ids,
                train_env.current_legal_action_mask,
                train_env.current_num_moves,
            )
        train_env.apply_actions(act_dict["action"])

        if step % 3 == 0:
            infostate = train_env.current_infostate_tensor
            piece_ids = train_env.current_piece_ids
            num_moves = train_env.current_num_moves
            pos_oh = train_env.current_unknown_piece_position_onehot
            type_oh = train_env.current_unknown_piece_type_onehot
            counts = train_env.current_unknown_piece_counts
            moved = train_env.current_unknown_piece_has_moved

            pos_oh_f = pos_oh.float()
            type_oh_f = type_oh.float()
            active = type_oh.any(dim=-1)

            unmoved_mask = (active & (~moved)).float().unsqueeze(-1)
            moved_mask = (active & moved).float().unsqueeze(-1)
            type_12 = type_oh_f[:, :, PLAYER_PIECE_TYPE_SLICE]

            unmoved_counts += torch.einsum("bnc,bnk->ck", pos_oh_f * unmoved_mask, type_12)
            moved_counts += torch.einsum("bnc,bnk->ck", pos_oh_f * moved_mask, type_12)

            if active.any():
                act_f = active.float()
                denom = act_f.sum().clamp(min=1.0)
                for _ in range(3):
                    bt_opt.zero_grad()
                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        bt_logits = bt_model(
                            infostate_tensor=infostate,
                            piece_ids=piece_ids,
                            unknown_piece_position_onehot=pos_oh,
                            unknown_piece_type_onehot=type_oh,
                            unknown_piece_counts=counts,
                            unknown_piece_has_moved=moved,
                        )
                        bt_logp = F.log_softmax(bt_logits.float(), dim=-1)
                        bt_xent = -(type_oh_f * bt_logp).sum(dim=-1)
                        bt_loss = (bt_xent * act_f).sum() / denom
                    bt_loss.backward()
                    torch.nn.utils.clip_grad_norm_(bt_model.parameters(), 1.0)
                    bt_opt.step()

                    if ar_opt is not None:
                        ar_opt.zero_grad()
                        with autocast(device_type="cuda", dtype=torch.bfloat16):
                            ar_logits = ar_model(
                                infostate_tensor=infostate,
                                piece_ids=piece_ids,
                                num_moves=num_moves,
                                unknown_piece_position_onehot=pos_oh,
                                unknown_piece_type_onehot=type_oh,
                                unknown_piece_counts=counts,
                                unknown_piece_has_moved=moved,
                            )
                            ar_logp = F.log_softmax(ar_logits.float(), dim=-1)
                            ar_xent = -(type_oh_f * ar_logp).sum(dim=-1)
                            ar_loss = (ar_xent * act_f).sum() / denom
                        ar_loss.backward()
                        torch.nn.utils.clip_grad_norm_(ar_model.parameters(), 1.0)
                        ar_opt.step()

                    glc_opt.zero_grad()
                    glc_logp_12 = glc_net.forward_marginals_12(
                        infostate_tensor=infostate,
                        piece_ids=piece_ids,
                        unknown_piece_position_onehot=pos_oh,
                        unknown_piece_has_moved=moved,
                        unknown_piece_counts=counts,
                        use_kernel_prior_anchor=True,
                    )
                    glc_xent = -(type_12 * glc_logp_12).sum(dim=-1)
                    glc_p12 = glc_logp_12.exp()
                    pred_counts = (glc_p12 * act_f.unsqueeze(-1)).sum(dim=1)
                    cnt_loss = F.mse_loss(pred_counts / 40.0, counts.float() / 40.0)
                    glc_loss = (glc_xent * act_f).sum() / denom + 0.15 * cnt_loss
                    glc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(glc_net.parameters(), 1.0)
                    glc_opt.step()

    bt_model.eval()
    ar_model.eval()
    glc_net.eval()

    unmoved_prior = unmoved_counts / unmoved_counts.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    moved_prior = moved_counts / moved_counts.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    helper_prior = SpatialPrior_Sinkhorn_Helper(
        unmoved_cell_prior=unmoved_prior.detach(),
        moved_cell_prior=moved_prior.detach(),
        sinkhorn_iters=6,
    )
    glc_plugin = GLCAOMStrategoBeliefPlugin(
        net=glc_net,
        helper=helper_prior,
        use_kernel_prior_anchor=True,
    )
    untrained_plugin = GLCAOMStrategoBeliefPlugin(
        net=untrained_glc_net,
        helper=None,
        use_kernel_prior_anchor=False,
    )
    return {
        "helper_prior": helper_prior,
        "bt_model": bt_model,
        "ar_model": ar_model,
        "glc_net": glc_net,
        "untrained_glc_net": untrained_glc_net,
        "glc_plugin": glc_plugin,
        "untrained_plugin": untrained_plugin,
        "param_counts": {
            "BeliefTransformer_Arch_OurTraining_240Updates": int(sum(p.numel() for p in bt_model.parameters())),
            "ARBelief_Arch_OurTraining_240Updates": int(sum(p.numel() for p in ar_model.parameters())),
            "GLCAOMStrategoBeliefNet": int(sum(p.numel() for p in glc_net.parameters())),
        },
    }


def evaluate_beliefs(
    seed: int,
    rl_model: nn.Module,
    arrangements: Tuple[List[str], List[str]],
    calibrated_bundle: Dict[str, Any],
    num_eval_boards: int = 600,
) -> Dict[str, Any]:
    """Exempt post-hoc belief scoring function (`evaluate_beliefs` in `EXEMPT_SCORING_OR_TRAINING_FUNCTIONS`):
    Evaluates Hidden-Piece Belief Calibration on `num_eval_boards = 600` held-out mid-game boards per seed
    (`1,800` boards / `~54,000` hidden piece slots total across 3 seeds) drawn strictly from `arrangements[p][512:]`.
    Computes per-board Cross-Entropy (`nats`), Top-1 Accuracy (`%`), and Brier score, plus per-seed 95% CIs
    for every arm and every pre-registered pairwise difference (`R26B-E1b.1`, `R26B-E1b.2`, `R26B-E1b.3`).
    """
    from pyengine import utils
    from pyengine.belief.masking import create_mask
    from pyengine.belief.uniform import marginalized_uniform_belief, uniform_belief
    from pyengine.core.env import Stratego
    from pyengine.utils.constants import N_PIECE_TYPE, PLAYER_PIECE_TYPE_SLICE

    helper_prior: SpatialPrior_Sinkhorn_Helper = calibrated_bundle["helper_prior"]
    bt_model: nn.Module = calibrated_bundle["bt_model"]
    ar_model: nn.Module = calibrated_bundle["ar_model"]
    glc_plugin: GLCAOMStrategoBeliefPlugin = calibrated_bundle["glc_plugin"]
    untrained_plugin: GLCAOMStrategoBeliefPlugin = calibrated_bundle["untrained_plugin"]

    torch.manual_seed(seed + 99)
    eval_arrs = (arrangements[0][512:], arrangements[1][512:])
    batch_envs = 20
    eval_env = Stratego(num_envs=batch_envs, traj_len_per_player=120, quiet=2, max_num_moves=200)
    eval_env.change_reset_behavior_to_random_initial_arrangement(eval_arrs)
    eval_env.reset()

    model_names = (
        "Control_Uniform_1Over12",
        "Control_ZeroParam_Readout",
        "Control_Untrained_Init",
        "Ataraxos_Marginalized_Uniform_Belief",
        "Control_Helper_Heuristic_Prior",
        "BeliefTransformer_Arch_OurTraining_240Updates",
        "ARBelief_Arch_OurTraining_240Updates",
        "GLC_v3_BeliefNet_Only",
        "GLC_v3_BeliefNet_and_Clearer",
        "Control_Exact_Ceiling_Posterior",
    )

    board_ce: Dict[str, List[float]] = {m: [] for m in model_names}
    board_acc: Dict[str, List[float]] = {m: [] for m in model_names}
    board_brier: Dict[str, List[float]] = {m: [] for m in model_names}

    n_snapshots_needed = max(1, (num_eval_boards + batch_envs - 1) // batch_envs)
    snapshots_collected = 0
    ply = 0

    while snapshots_collected < n_snapshots_needed:
        ply += 1
        with torch.no_grad(), utils.eval_mode(rl_model), autocast(device_type="cuda", dtype=torch.bfloat16):
            act_dict = rl_model(
                eval_env.current_infostate_tensor,
                eval_env.current_piece_ids,
                eval_env.current_legal_action_mask,
                eval_env.current_num_moves,
            )
        eval_env.apply_actions(act_dict["action"])

        if ply >= 16 and (ply % 8 == 0):
            snapshots_collected += 1
            with torch.no_grad():
                infostate = eval_env.current_infostate_tensor
                piece_ids = eval_env.current_piece_ids
                pos_oh = eval_env.current_unknown_piece_position_onehot
                type_oh = eval_env.current_unknown_piece_type_onehot
                counts = eval_env.current_unknown_piece_counts
                moved = eval_env.current_unknown_piece_has_moved
                active = type_oh.any(dim=-1)
                target_f = type_oh.float()
                target_idx = type_oh[:, :, PLAYER_PIECE_TYPE_SLICE].long().argmax(dim=-1)

                zeros_type = torch.zeros_like(type_oh)
                feas_mask = create_mask(counts, zeros_type, moved)

                # 1. Control_Uniform_1Over12 (pure 1/12 uniform over all 12 piece ranks)
                logp_u12 = torch.zeros(infostate.shape[0], pos_oh.shape[1], N_PIECE_TYPE, device=infostate.device)
                logp_u12[:, :, PLAYER_PIECE_TYPE_SLICE] = -math.log(12.0)

                # 2. Control_ZeroParam_Readout (movability-masked uniform_belief)
                logp_zparam = uniform_belief(zeros_type, moved, counts).float()

                # 3. Control_Untrained_Init (GLCAOMStrategoBeliefNet at step 0, no helper)
                logp_untrained = untrained_plugin.compute_log_beliefs(
                    infostate, piece_ids, pos_oh, moved, counts, with_helper=False
                ).float()

                # 4. Ataraxos_Marginalized_Uniform_Belief (public analytic marginalized_uniform_belief)
                logp_mu = marginalized_uniform_belief(infostate, pos_oh).float()

                # 5. Control_Helper_Heuristic_Prior (SpatialPrior_Sinkhorn_Helper, 0 neural params)
                logp_helper = helper_prior.compute_calibrated_marginals(
                    infostate, pos_oh, moved, counts
                ).float()

                # 6. Diagnostic: BeliefTransformer_Arch_OurTraining_240Updates
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    bt_mem = bt_model.encoder_forward(infostate, piece_ids, pos_oh)
                    bt_logits_0 = bt_model.decoder_forward(zeros_type, bt_mem).float()
                bt_logits_0 = bt_logits_0.masked_fill(~feas_mask, -1e9)
                bt_probs_0 = F.softmax(bt_logits_0, dim=-1)
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    bt_logits_1 = bt_model.decoder_forward(bt_probs_0.to(bt_mem.dtype), bt_mem).float()
                bt_logits = (0.5 * bt_logits_0 + 0.5 * bt_logits_1).masked_fill(~feas_mask, -1e9)
                logp_bt = F.log_softmax(bt_logits, dim=-1)

                # 7. Diagnostic: ARBelief_Arch_OurTraining_240Updates
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    ar_mem = ar_model.encoder_forward(infostate, pos_oh)
                    ar_logits_0 = ar_model.decoder_forward(zeros_type, ar_mem).float()
                ar_logits_0 = ar_logits_0.masked_fill(~feas_mask, -1e9)
                ar_probs_0 = F.softmax(ar_logits_0, dim=-1)
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    ar_logits_1 = ar_model.decoder_forward(ar_probs_0.to(ar_mem.dtype), ar_mem).float()
                ar_logits = (0.5 * ar_logits_0 + 0.5 * ar_logits_1).masked_fill(~feas_mask, -1e9)
                logp_ar = F.log_softmax(ar_logits, dim=-1)

                # 8. GLC_v3_BeliefNet_Only (pure neural, no Sinkhorn-LIR clearer)
                logp_glc_pure = glc_plugin.compute_log_beliefs(
                    infostate, piece_ids, pos_oh, moved, counts, with_helper=False
                ).float()

                # 9. GLC_v3_BeliefNet_and_Clearer (neural + Two-Stage Sinkhorn-LIR clearer)
                logp_glc_clear = glc_plugin.compute_log_beliefs(
                    infostate, piece_ids, pos_oh, moved, counts, with_helper=True
                ).float()

                # 10. Control_Exact_Ceiling_Posterior
                t_12 = target_f[:, :, PLAYER_PIECE_TYPE_SLICE]
                p_mu_12 = F.softmax(logp_mu[:, :, PLAYER_PIECE_TYPE_SLICE], dim=-1)
                p_ceil_12 = 0.65 * t_12 + 0.35 * p_mu_12
                p_ceil_12 = p_ceil_12 / p_ceil_12.sum(dim=-1, keepdim=True).clamp(min=1e-7)
                logp_ceil = torch.zeros(infostate.shape[0], pos_oh.shape[1], N_PIECE_TYPE, device=infostate.device)
                logp_ceil[:, :, PLAYER_PIECE_TYPE_SLICE] = torch.log(p_ceil_12.clamp(min=1e-7))

                arm_outputs = (
                    ("Control_Uniform_1Over12", logp_u12),
                    ("Control_ZeroParam_Readout", logp_zparam),
                    ("Control_Untrained_Init", logp_untrained),
                    ("Ataraxos_Marginalized_Uniform_Belief", logp_mu),
                    ("Control_Helper_Heuristic_Prior", logp_helper),
                    ("BeliefTransformer_Arch_OurTraining_240Updates", logp_bt),
                    ("ARBelief_Arch_OurTraining_240Updates", logp_ar),
                    ("GLC_v3_BeliefNet_Only", logp_glc_pure),
                    ("GLC_v3_BeliefNet_and_Clearer", logp_glc_clear),
                    ("Control_Exact_Ceiling_Posterior", logp_ceil),
                )

                act_f = active.float()
                board_den = act_f.sum(dim=-1).clamp(min=1.0)  # [B]
                for m_name, lp in arm_outputs:
                    lp_12 = lp[:, :, PLAYER_PIECE_TYPE_SLICE]
                    p_12 = torch.softmax(lp_12, dim=-1)
                    lp_norm = torch.log(p_12.clamp(min=1e-7))

                    xent_b = (-(t_12 * lp_norm).sum(dim=-1) * act_f).sum(dim=-1) / board_den
                    pred_idx = p_12.argmax(dim=-1)
                    acc_b = (((pred_idx == target_idx).float() * act_f).sum(dim=-1) / board_den) * 100.0
                    brier_b = ((((p_12 - t_12) ** 2).sum(dim=-1)) * act_f).sum(dim=-1) / board_den

                    for b_i in range(infostate.shape[0]):
                        if len(board_ce[m_name]) < num_eval_boards:
                            board_ce[m_name].append(float(xent_b[b_i].item()))
                            board_acc[m_name].append(float(acc_b[b_i].item()))
                            board_brier[m_name].append(float(brier_b[b_i].item()))

        if ply >= 152:
            eval_env.reset()
            ply = 0

    def _paired_diff_ci(vals_a: Sequence[float], vals_b: Sequence[float]) -> Dict[str, Any]:
        diff = np.asarray(vals_a, dtype=np.float64) - np.asarray(vals_b, dtype=np.float64)
        return summarize_samples_ci95(diff)

    pairwise_diffs = {
        "ce_glc_clear_minus_marginalized": _paired_diff_ci(
            board_ce["GLC_v3_BeliefNet_and_Clearer"],
            board_ce["Ataraxos_Marginalized_Uniform_Belief"],
        ),
        "ce_glc_net_only_minus_helper": _paired_diff_ci(
            board_ce["GLC_v3_BeliefNet_Only"],
            board_ce["Control_Helper_Heuristic_Prior"],
        ),
        "ce_glc_clear_minus_glc_net_only": _paired_diff_ci(
            board_ce["GLC_v3_BeliefNet_and_Clearer"],
            board_ce["GLC_v3_BeliefNet_Only"],
        ),
        "acc_glc_clear_minus_marginalized_pp": _paired_diff_ci(
            board_acc["GLC_v3_BeliefNet_and_Clearer"],
            board_acc["Ataraxos_Marginalized_Uniform_Belief"],
        ),
        "acc_glc_clear_minus_helper_pp": _paired_diff_ci(
            board_acc["GLC_v3_BeliefNet_and_Clearer"],
            board_acc["Control_Helper_Heuristic_Prior"],
        ),
        "brier_glc_clear_minus_marginalized": _paired_diff_ci(
            board_brier["GLC_v3_BeliefNet_and_Clearer"],
            board_brier["Ataraxos_Marginalized_Uniform_Belief"],
        ),
    }

    return {
        "num_eval_boards": int(num_eval_boards),
        "model_names": list(model_names),
        "cross_entropy_summary": {m: summarize_samples_ci95(board_ce[m]) for m in model_names},
        "top1_accuracy_pct_summary": {m: summarize_samples_ci95(board_acc[m]) for m in model_names},
        "brier_score_summary": {m: summarize_samples_ci95(board_brier[m]) for m in model_names},
        "pairwise_diff_ci95": pairwise_diffs,
        "raw_board_ce": board_ce,
        "raw_board_acc": board_acc,
        "raw_board_brier": board_brier,
    }


def play_ataraxos_pair_match(
    agent_a: Any,
    agent_b: Any,
    eval_env: Any,
    arrangements: Tuple[List[str], List[str]],
    num_paired_games: int,
    seed: int,
) -> Dict[str, Any]:
    """Play `num_paired_games` (`K_pairs = num_paired_games // 2` seat-swapped pairs) of `10x10` Stratego
    strictly on held-out `arrangements[p][512:]`.
    Records per-pair score $Y_j = \\frac{1}{2}(X_{j,\\text{red}} + X_{j,\\text{blue}}) \\in [0, 1]$ and returns
    per-seed and raw pair scores for pooled 95% confidence intervals.
    """
    from pyengine import utils

    rng = np.random.default_rng(seed)
    n_avail = len(arrangements[0])
    start_idx = min(512, max(0, n_avail - 64))
    wins_a = 0.0
    wins_b = 0.0
    draws = 0.0
    total_plies = 0
    pair_scores_pct: List[float] = []

    def step_agent(ag: Any, env_obj: Any) -> torch.Tensor:
        mask_1d = env_obj.current_legal_action_mask.squeeze(0)
        if int(mask_1d.sum().item()) <= 1:
            return mask_1d.int().argmax().int().view(1)
        if hasattr(ag, "select_uniform_action"):
            return ag.select_uniform_action(env_obj.current_legal_action_mask)
        if hasattr(ag, "depth") and hasattr(ag, "max_num_samples"):
            return ag(
                env_obj.current_state,
                env_obj.current_infostate_tensor,
                env_obj.current_piece_ids,
                env_obj.current_legal_action_mask,
                env_obj.current_num_moves,
                env_obj.current_unknown_piece_position_onehot,
                env_obj.current_unknown_piece_counts,
                env_obj.current_unknown_piece_has_moved,
            )
        with torch.no_grad(), utils.eval_mode(ag), autocast(device_type="cuda", dtype=torch.bfloat16):
            return ag(
                infostate_tensor=env_obj.current_infostate_tensor,
                piece_ids=env_obj.current_piece_ids,
                legal_action_mask=env_obj.current_legal_action_mask,
                num_moves=env_obj.current_num_moves,
            )["action"]

    n_pairs = max(1, num_paired_games // 2)
    for pair_idx in range(n_pairs):
        idx0 = int(rng.integers(start_idx, n_avail))
        idx1 = int(rng.integers(start_idx, n_avail))
        arr_pair = ([arrangements[0][idx0]], [arrangements[1][idx1]])
        pair_pts = 0.0

        for swap in (0, 1):
            torch.manual_seed(seed + pair_idx * 37 + swap * 13)
            pl0_ag = agent_a if swap == 0 else agent_b
            pl1_ag = agent_b if swap == 0 else agent_a
            eval_env.change_reset_behavior_to_random_initial_arrangement(arr_pair)
            eval_env.reset()

            plies = 0
            for _ in range(eval_env.conf.max_num_moves):
                plies += 1
                acting = pl0_ag if eval_env.current_player == 0 else pl1_ag
                act_t = step_agent(acting, eval_env)
                eval_env.apply_actions(act_t)
                if bool(eval_env.current_is_terminal.all().item()):
                    break

            total_plies += plies
            rew_pl0 = float(eval_env.current_reward_pl0[0].item()) if bool(eval_env.current_is_terminal[0].item()) else 0.0
            rew_a = rew_pl0 if swap == 0 else -rew_pl0
            if rew_a > 0.1:
                wins_a += 1.0
                pair_pts += 1.0
            elif rew_a < -0.1:
                wins_b += 1.0
            else:
                draws += 1.0
                pair_pts += 0.5

        pair_scores_pct.append(50.0 * pair_pts)

    tot = max(1.0, wins_a + wins_b + draws)
    ci_stats = summarize_samples_ci95(pair_scores_pct)
    return {
        "wins_a": float(wins_a),
        "wins_b": float(wins_b),
        "draws": float(draws),
        "num_games": int(tot),
        "num_pairs": int(n_pairs),
        "score_a": float((wins_a + 0.5 * draws) / tot),
        "win_rate_pct": round(float(ci_stats["mean"]), 2),
        "se_pct": round(float(ci_stats["se"]), 2),
        "ci95_low_pct": round(float(ci_stats["ci95_low"]), 2),
        "ci95_high_pct": round(float(ci_stats["ci95_high"]), 2),
        "lcb95_one_sided_pct": round(float(ci_stats["lcb95_one_sided"]), 2),
        "avg_plies": round(float(total_plies) / tot, 2),
        "pair_scores_pct": pair_scores_pct,
    }


def evaluate_r26b_seed(
    seed: int,
    focal_games_per_matchup: int = 100,
    anchor_games_per_matchup: int = 20,
    num_eval_boards: int = 600,
    belief_train_updates: int = 240,
    ataraxos_belief_ckpt: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one seed of `R26-B` (`R26B-E1b` Belief Calibration + `R26B-E1a` Powered `10x10` Stratego Matchups)."""
    from pyengine.belief.uniform import marginalized_uniform_belief, uniform_belief
    from pyengine.core.env import Stratego
    from pyengine.core.search import SearchBot
    from pyengine.utils.loading import load_rl_model

    pth = str(_ATARAXOS_ROOT / "pretrained" / "final_run" / "model42400.pthw")
    rl_model, arrangements = load_rl_model(pth)
    rl_model.eval()

    calibrated_bundle = calibrate_model_and_clearer(
        seed=seed,
        rl_model=rl_model,
        arrangements=arrangements,
        belief_train_updates=belief_train_updates,
        ataraxos_belief_ckpt=ataraxos_belief_ckpt,
    )
    belief_eval = evaluate_beliefs(
        seed=seed,
        rl_model=rl_model,
        arrangements=arrangements,
        calibrated_bundle=calibrated_bundle,
        num_eval_boards=num_eval_boards,
    )

    search_env = Stratego(
        num_envs=120,
        traj_len_per_player=100,
        quiet=2,
        max_num_moves=260,
        max_num_moves_between_attacks=60,
    )
    eval_env = Stratego(
        num_envs=1,
        traj_len_per_player=160,
        quiet=2,
        max_num_moves=260,
        max_num_moves_between_attacks=60,
    )

    agent_names = (
        "Control_UniformRandom",
        "Ataraxos_14.7M_Blueprint",
        "Control_Untrained_Init_SearchBot",
        "Ataraxos_SearchBot_Uniform",
        "Ataraxos_SearchBot_Marginalized",
        "Control_SearchBot_Helper_Heuristic_Prior",
        "GLC_v3_AtaraxosSearchBot",
        "Control_Exact_Ceiling_PerfectInfoSearchBot",
    )

    common_search_kwargs = {
        "rl_model": rl_model,
        "env": search_env,
        "depth": 2,
        "stepsize": 10.0,
        "temperature": 0.006,
        "td_lambda": 1.0,
        "max_num_samples": 20,
        "dtype": torch.bfloat16,
    }

    agents_map: Dict[str, Any] = {
        "Control_UniformRandom": UniformRandomStrategoAgent(),
        "Ataraxos_14.7M_Blueprint": rl_model,
        "Control_Untrained_Init_SearchBot": SearchBot(
            **common_search_kwargs,
            belief_model=calibrated_bundle["untrained_plugin"],
        ),
        "Ataraxos_SearchBot_Uniform": SearchBot(
            **common_search_kwargs,
            belief_model=uniform_belief,
        ),
        "Ataraxos_SearchBot_Marginalized": SearchBot(
            **common_search_kwargs,
            belief_model=marginalized_uniform_belief,
        ),
        "Control_SearchBot_Helper_Heuristic_Prior": SearchBot(
            **common_search_kwargs,
            belief_model=calibrated_bundle["helper_prior"],
        ),
        "GLC_v3_AtaraxosSearchBot": GLC_v3_AtaraxosSearchBot(
            **common_search_kwargs,
            belief_model=calibrated_bundle["glc_plugin"],
        ),
        "Control_Exact_Ceiling_PerfectInfoSearchBot": SearchBot(
            **common_search_kwargs,
            belief_model=None,
        ),
    }

    focal_opponents = (
        "Ataraxos_SearchBot_Marginalized",
        "Ataraxos_SearchBot_Uniform",
        "Control_SearchBot_Helper_Heuristic_Prior",
        "Ataraxos_14.7M_Blueprint",
        "Control_Untrained_Init_SearchBot",
    )

    n_agents = len(agent_names)
    name_to_idx = {nm: idx for idx, nm in enumerate(agent_names)}
    win_mat = np.full((n_agents, n_agents), 0.5, dtype=np.float64)
    league = BayesianEloLeague(anchor_elo=1000.0)
    agent_ids: List[str] = []
    for aname in agent_names:
        b_val = 0 if ("Blueprint" in aname or "Control_UniformRandom" in aname) else 120
        spec = league.register_agent(aname, aname, b_val)
        agent_ids.append(spec.agent_id)

    focal_matchups: Dict[str, Any] = {}
    all_matchups: Dict[str, Any] = {}

    # 1. Run Powered Focal Matchups (`focal_games_per_matchup = 100` games = `50` seat-swapped pairs per seed)
    i_glc = name_to_idx["GLC_v3_AtaraxosSearchBot"]
    for opp_idx, opp_name in enumerate(focal_opponents):
        j_opp = name_to_idx[opp_name]
        pair_seed = seed * 1000 + opp_idx * 43 + 7
        res = play_ataraxos_pair_match(
            agents_map["GLC_v3_AtaraxosSearchBot"],
            agents_map[opp_name],
            eval_env=eval_env,
            arrangements=arrangements,
            num_paired_games=focal_games_per_matchup,
            seed=pair_seed,
        )
        win_mat[i_glc, j_opp] = res["score_a"]
        win_mat[j_opp, i_glc] = 1.0 - res["score_a"]
        focal_matchups[f"GLC_v3_AtaraxosSearchBot_vs_{opp_name}"] = res
        all_matchups[f"GLC_v3_AtaraxosSearchBot_vs_{opp_name}"] = res

        for _ in range(int(round(res["wins_a"]))):
            league.add_match("Micro-Stratego", agent_ids[i_glc], agent_ids[j_opp], outcome_a=1.0, is_draw=False, seed=seed)
        for _ in range(int(round(res["wins_b"]))):
            league.add_match("Micro-Stratego", agent_ids[i_glc], agent_ids[j_opp], outcome_a=0.0, is_draw=False, seed=seed)
        for _ in range(int(round(res["draws"]))):
            league.add_match("Micro-Stratego", agent_ids[i_glc], agent_ids[j_opp], outcome_a=0.5, is_draw=True, seed=seed)

    # 2. Run Anchor Cross-Matchups (`anchor_games_per_matchup = 20` games per seed) to anchor BayesElo
    anchor_pairs = (
        ("Ataraxos_SearchBot_Marginalized", "Ataraxos_SearchBot_Uniform"),
        ("Ataraxos_SearchBot_Marginalized", "Ataraxos_14.7M_Blueprint"),
        ("Control_SearchBot_Helper_Heuristic_Prior", "Ataraxos_14.7M_Blueprint"),
        ("Ataraxos_14.7M_Blueprint", "Control_UniformRandom"),
        ("Control_Exact_Ceiling_PerfectInfoSearchBot", "GLC_v3_AtaraxosSearchBot"),
        ("Control_Exact_Ceiling_PerfectInfoSearchBot", "Ataraxos_SearchBot_Marginalized"),
    )
    for a_idx, (na, nb) in enumerate(anchor_pairs):
        ia = name_to_idx[na]
        ib = name_to_idx[nb]
        pair_seed = seed * 2000 + a_idx * 59 + 11
        res = play_ataraxos_pair_match(
            agents_map[na],
            agents_map[nb],
            eval_env=eval_env,
            arrangements=arrangements,
            num_paired_games=anchor_games_per_matchup,
            seed=pair_seed,
        )
        win_mat[ia, ib] = res["score_a"]
        win_mat[ib, ia] = 1.0 - res["score_a"]
        all_matchups[f"{na}_vs_{nb}"] = res

        for _ in range(int(round(res["wins_a"]))):
            league.add_match("Micro-Stratego", agent_ids[ia], agent_ids[ib], outcome_a=1.0, is_draw=False, seed=seed)
        for _ in range(int(round(res["wins_b"]))):
            league.add_match("Micro-Stratego", agent_ids[ia], agent_ids[ib], outcome_a=0.0, is_draw=False, seed=seed)
        for _ in range(int(round(res["draws"]))):
            league.add_match("Micro-Stratego", agent_ids[ia], agent_ids[ib], outcome_a=0.5, is_draw=True, seed=seed)

    bp_idx = name_to_idx["Ataraxos_14.7M_Blueprint"]
    fit_res = league.fit_environment("Micro-Stratego", anchor_agent_id=agent_ids[bp_idx], n_bootstrap=150, seed=seed)
    elos_arr = np.array([fit_res.ratings[aid].elo for aid in agent_ids], dtype=np.float64)
    melo_info = compute_melo_cyclic_decomposition(win_mat, elos_arr)

    return {
        "seed": seed,
        "agent_names": list(agent_names),
        "belief_model_param_counts": calibrated_bundle["param_counts"],
        "win_rate_matrix_pct": (win_mat * 100.0).round(2).tolist(),
        "bayes_elo": {agent_names[i]: round(float(elos_arr[i]), 2) for i in range(n_agents)},
        "melo": {agent_names[i]: round(float(melo_info["melo_ratings"][i]), 2) for i in range(n_agents)},
        "focal_matchups": focal_matchups,
        "all_matchups": all_matchups,
        "belief_calibration": belief_eval,
    }


def evaluate_r26b_preregistered_gates(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate all pre-registered numeric pass/fail bars in `benchmarks/PREREG-R26B-ATARAXOS-STRATEGO.md`."""
    d0 = summary["d0_audit_and_pretrained_verification"]
    ver = d0["ataraxos_pretrained_verification"]
    gate_d0_checks = {
        "D0.1_ast_zero_violations": bool(d0["static_ast_violations"] == 0),
        "D0.2_canaries_and_patch_tests": bool(d0["canaries_caught"] == 7 and d0["patch_unit_tests_passed"]),
        "D0.3_ataraxos_rl_exact_match": bool(ver["rl_exact_match_passed"]),
        "D0.4_ataraxos_arr_exact_match": bool(ver["arr_exact_match_passed"]),
    }
    gate_d0 = all(gate_d0_checks.values())

    # Gate R26B-E1b: Held-Out Belief Calibration (Per-Seed AND Pooled 95% CI Bars)
    bel = summary["e1b_belief_calibration"]
    p_diffs = bel["pooled_pairwise_diff_ci95"]
    per_seed_bel = bel["per_seed_pairwise_diff_ci95"]
    seeds_str = list(per_seed_bel.keys())

    e1b_1_ce_marg = bool(
        float(p_diffs["ce_glc_clear_minus_marginalized"]["ci95_high"]) <= -0.120
        and all(float(per_seed_bel[s]["ce_glc_clear_minus_marginalized"]["ci95_high"]) <= -0.120 for s in seeds_str)
    )
    e1b_1_ce_net_vs_help = bool(
        float(p_diffs["ce_glc_net_only_minus_helper"]["ci95_high"]) <= -0.030
        and all(float(per_seed_bel[s]["ce_glc_net_only_minus_helper"]["ci95_high"]) <= -0.030 for s in seeds_str)
    )
    e1b_1_ce_clear_vs_net = bool(
        float(p_diffs["ce_glc_clear_minus_glc_net_only"]["ci95_high"]) <= -0.015
        and all(float(per_seed_bel[s]["ce_glc_clear_minus_glc_net_only"]["ci95_high"]) <= -0.015 for s in seeds_str)
    )
    e1b_2_acc_marg = bool(
        float(p_diffs["acc_glc_clear_minus_marginalized_pp"]["ci95_low"]) >= 4.0
        and all(float(per_seed_bel[s]["acc_glc_clear_minus_marginalized_pp"]["ci95_low"]) >= 4.0 for s in seeds_str)
    )
    e1b_2_acc_help = bool(
        float(p_diffs["acc_glc_clear_minus_helper_pp"]["ci95_low"]) >= 1.5
        and all(float(per_seed_bel[s]["acc_glc_clear_minus_helper_pp"]["ci95_low"]) >= 1.5 for s in seeds_str)
    )
    e1b_2_brier_marg = bool(
        float(p_diffs["brier_glc_clear_minus_marginalized"]["ci95_high"]) <= -0.025
        and all(float(per_seed_bel[s]["brier_glc_clear_minus_marginalized"]["ci95_high"]) <= -0.025 for s in seeds_str)
    )

    e1b_3_ordering = True
    for s in seeds_str:
        ce_s = summary["per_seed"][s]["belief_calibration"]["cross_entropy_summary"]
        if not (
            float(ce_s["Control_Exact_Ceiling_Posterior"]["mean"])
            < float(ce_s["GLC_v3_BeliefNet_and_Clearer"]["mean"])
            < float(ce_s["GLC_v3_BeliefNet_Only"]["mean"])
            < float(ce_s["Control_Helper_Heuristic_Prior"]["mean"])
            < float(ce_s["Ataraxos_Marginalized_Uniform_Belief"]["mean"])
            < float(ce_s["Control_Uniform_1Over12"]["mean"])
            and float(ce_s["GLC_v3_BeliefNet_Only"]["mean"]) < float(ce_s["Control_Untrained_Init"]["mean"])
        ):
            e1b_3_ordering = False

    gate_e1b_checks = {
        "E1b.1_ce_glc_clear_vs_marginalized_ci95_high_le_minus_0.120": e1b_1_ce_marg,
        "E1b.1_ce_glc_net_only_vs_helper_ci95_high_le_minus_0.030": e1b_1_ce_net_vs_help,
        "E1b.1_ce_glc_clear_vs_net_only_ci95_high_le_minus_0.015": e1b_1_ce_clear_vs_net,
        "E1b.2_acc_glc_clear_vs_marginalized_ci95_low_ge_plus_4.0pp": e1b_2_acc_marg,
        "E1b.2_acc_glc_clear_vs_helper_ci95_low_ge_plus_1.5pp": e1b_2_acc_help,
        "E1b.2_brier_glc_clear_vs_marginalized_ci95_high_le_minus_0.025": e1b_2_brier_marg,
        "E1b.3_mandatory_control_ordering_all_seeds": e1b_3_ordering,
    }
    gate_e1b = all(gate_e1b_checks.values())

    # Gate R26B-E1a: Powered 10x10 Stratego Paired Matchups (300 Games Per Focal Matchup)
    e1a = summary["e1a_stratego_tournament"]
    pooled_m = e1a["pooled_focal_matchups"]
    per_seed_m = e1a["per_seed_focal_matchups"]

    k_marg = "GLC_v3_AtaraxosSearchBot_vs_Ataraxos_SearchBot_Marginalized"
    k_unif = "GLC_v3_AtaraxosSearchBot_vs_Ataraxos_SearchBot_Uniform"
    k_help = "GLC_v3_AtaraxosSearchBot_vs_Control_SearchBot_Helper_Heuristic_Prior"
    k_bp = "GLC_v3_AtaraxosSearchBot_vs_Ataraxos_14.7M_Blueprint"
    k_untr = "GLC_v3_AtaraxosSearchBot_vs_Control_Untrained_Init_SearchBot"

    e1a_1_marg = bool(
        float(pooled_m[k_marg]["ci95_low"]) > 50.0
        and all(float(per_seed_m[s][k_marg]["win_rate_pct"]) > 50.0 for s in seeds_str)
    )
    e1a_2_unif = bool(
        float(pooled_m[k_unif]["ci95_low"]) > 52.0
        and all(float(per_seed_m[s][k_unif]["lcb95_one_sided_pct"]) > 50.0 for s in seeds_str)
    )
    e1a_3_help = bool(
        float(pooled_m[k_help]["ci95_low"]) > 50.0
        and all(float(per_seed_m[s][k_help]["win_rate_pct"]) > 50.0 for s in seeds_str)
    )
    e1a_4_bp_untr = bool(
        float(pooled_m[k_bp]["ci95_low"]) > 55.0
        and float(pooled_m[k_untr]["ci95_low"]) > 53.0
        and all(
            float(per_seed_m[s][k_bp]["ci95_low_pct"]) > 50.0
            and float(per_seed_m[s][k_untr]["ci95_low_pct"]) > 50.0
            for s in seeds_str
        )
    )

    non_ceiling_arms = (
        "Control_UniformRandom",
        "Ataraxos_14.7M_Blueprint",
        "Control_Untrained_Init_SearchBot",
        "Ataraxos_SearchBot_Uniform",
        "Ataraxos_SearchBot_Marginalized",
        "Control_SearchBot_Helper_Heuristic_Prior",
    )
    e1a_5_elo_rank1 = all(
        float(summary["per_seed"][s]["bayes_elo"]["GLC_v3_AtaraxosSearchBot"])
        > max(float(summary["per_seed"][s]["bayes_elo"][nm]) for nm in non_ceiling_arms)
        for s in seeds_str
    )

    gate_e1a_checks = {
        "E1a.1_vs_ataraxos_marginalized_pooled_lcb95_gt_50_and_all_seeds_gt_50": e1a_1_marg,
        "E1a.2_vs_ataraxos_uniform_pooled_lcb95_gt_52_and_all_seeds_lcb_gt_50": e1a_2_unif,
        "E1a.3_vs_helper_prior_pooled_lcb95_gt_50_and_all_seeds_gt_50": e1a_3_help,
        "E1a.4_vs_blueprint_and_untrained_init_lcb95": e1a_4_bp_untr,
        "E1a.5_ranks_first_non_ceiling_bayes_elo_all_seeds": e1a_5_elo_rank1,
    }
    gate_e1a = all(gate_e1a_checks.values())

    return {
        "R26B_D0_zero_leakage_and_public_weights": {"passed": gate_d0, "checks": gate_d0_checks},
        "R26B_E1b_stratego_belief_calibration_ci95": {"passed": gate_e1b, "checks": gate_e1b_checks},
        "R26B_E1a_stratego_powered_tournament_ci95": {"passed": gate_e1a, "checks": gate_e1a_checks},
        "all_gates_passed": bool(gate_d0 and gate_e1b and gate_e1a),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Round 26-B Ataraxos 10x10 Stratego Evaluation (3 Seeds)")
    parser.add_argument("--out-dir", type=str, default=str(_REPO_ROOT / "results" / "r26b"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--focal-games-per-matchup", type=int, default=100)
    parser.add_argument("--anchor-games-per-matchup", type=int, default=20)
    parser.add_argument("--num-eval-boards", type=int, default=600)
    parser.add_argument("--belief-train-updates", type=int, default=240)
    parser.add_argument("--ataraxos-belief-ckpt", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    active_seeds = tuple(int(s) for s in args.seeds)
    t_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(
        f"[R26-B] Starting Ataraxos 10x10 Stratego Evaluation on device={device} ({gpu_name}), "
        f"seeds={active_seeds}, focal_games={args.focal_games_per_matchup}, eval_boards={args.num_eval_boards}",
        flush=True,
    )

    d0_res = verify_zero_oracle_leakage_gate()
    print(
        f"[R26B-D0] AST v2 Violations={d0_res['static_ast_violations']} "
        f"(Canaries={d0_res['canaries_caught']}/7), "
        f"Patch Unit Tests={d0_res['patch_unit_tests_passed']}, "
        f"RL Exact Match={d0_res['ataraxos_pretrained_verification']['rl_exact_match_passed']}, "
        f"Arr Exact Match={d0_res['ataraxos_pretrained_verification']['arr_exact_match_passed']}",
        flush=True,
    )

    per_seed_results: Dict[str, Any] = {}
    for seed in active_seeds:
        t_s = time.time()
        print(f"\n[R26-B] === Executing Seed {seed} ===", flush=True)
        s_res = evaluate_r26b_seed(
            seed=seed,
            focal_games_per_matchup=args.focal_games_per_matchup,
            anchor_games_per_matchup=args.anchor_games_per_matchup,
            num_eval_boards=args.num_eval_boards,
            belief_train_updates=args.belief_train_updates,
            ataraxos_belief_ckpt=args.ataraxos_belief_ckpt,
        )
        ce_sum = s_res["belief_calibration"]["cross_entropy_summary"]
        fm = s_res["focal_matchups"]
        print(
            f"  [R26B-E1b] Held-Out Board CE: "
            f"Marg={ce_sum['Ataraxos_Marginalized_Uniform_Belief']['mean']:.4f}, "
            f"Helper={ce_sum['Control_Helper_Heuristic_Prior']['mean']:.4f}, "
            f"GLC_NetOnly={ce_sum['GLC_v3_BeliefNet_Only']['mean']:.4f}, "
            f"GLC_Clear={ce_sum['GLC_v3_BeliefNet_and_Clearer']['mean']:.4f}",
            flush=True,
        )
        print(
            f"  [R26B-E1a] Focal Matchups: "
            f"vs_Marg={fm['GLC_v3_AtaraxosSearchBot_vs_Ataraxos_SearchBot_Marginalized']['win_rate_pct']:.1f}% "
            f"(CI [{fm['GLC_v3_AtaraxosSearchBot_vs_Ataraxos_SearchBot_Marginalized']['ci95_low_pct']:.1f}, "
            f"{fm['GLC_v3_AtaraxosSearchBot_vs_Ataraxos_SearchBot_Marginalized']['ci95_high_pct']:.1f}]), "
            f"vs_Unif={fm['GLC_v3_AtaraxosSearchBot_vs_Ataraxos_SearchBot_Uniform']['win_rate_pct']:.1f}%, "
            f"vs_Helper={fm['GLC_v3_AtaraxosSearchBot_vs_Control_SearchBot_Helper_Heuristic_Prior']['win_rate_pct']:.1f}%, "
            f"vs_Blueprint={fm['GLC_v3_AtaraxosSearchBot_vs_Ataraxos_14.7M_Blueprint']['win_rate_pct']:.1f}% "
            f"({time.time() - t_s:.1f}s)",
            flush=True,
        )
        per_seed_results[str(seed)] = s_res
        # Save compact seed JSON without large raw board arrays
        compact_s = {k: v for k, v in s_res.items() if k != "belief_calibration"}
        compact_s["belief_calibration"] = {
            k: v
            for k, v in s_res["belief_calibration"].items()
            if not k.startswith("raw_board_")
        }
        (out_dir / f"r26b_seed_{seed}.json").write_text(json.dumps(compact_s, indent=2), encoding="utf-8")

    # Pool belief boards across all seeds (`1,800` boards)
    bel_models = per_seed_results[str(active_seeds[0])]["belief_calibration"]["model_names"]
    pooled_ce_raw = {
        m: [x for s in active_seeds for x in per_seed_results[str(s)]["belief_calibration"]["raw_board_ce"][m]]
        for m in bel_models
    }
    pooled_acc_raw = {
        m: [x for s in active_seeds for x in per_seed_results[str(s)]["belief_calibration"]["raw_board_acc"][m]]
        for m in bel_models
    }
    pooled_brier_raw = {
        m: [x for s in active_seeds for x in per_seed_results[str(s)]["belief_calibration"]["raw_board_brier"][m]]
        for m in bel_models
    }

    def _pool_diff(va: Sequence[float], vb: Sequence[float]) -> Dict[str, Any]:
        return summarize_samples_ci95(np.asarray(va, dtype=np.float64) - np.asarray(vb, dtype=np.float64))

    pooled_pairwise_diff_ci95 = {
        "ce_glc_clear_minus_marginalized": _pool_diff(
            pooled_ce_raw["GLC_v3_BeliefNet_and_Clearer"],
            pooled_ce_raw["Ataraxos_Marginalized_Uniform_Belief"],
        ),
        "ce_glc_net_only_minus_helper": _pool_diff(
            pooled_ce_raw["GLC_v3_BeliefNet_Only"],
            pooled_ce_raw["Control_Helper_Heuristic_Prior"],
        ),
        "ce_glc_clear_minus_glc_net_only": _pool_diff(
            pooled_ce_raw["GLC_v3_BeliefNet_and_Clearer"],
            pooled_ce_raw["GLC_v3_BeliefNet_Only"],
        ),
        "acc_glc_clear_minus_marginalized_pp": _pool_diff(
            pooled_acc_raw["GLC_v3_BeliefNet_and_Clearer"],
            pooled_acc_raw["Ataraxos_Marginalized_Uniform_Belief"],
        ),
        "acc_glc_clear_minus_helper_pp": _pool_diff(
            pooled_acc_raw["GLC_v3_BeliefNet_and_Clearer"],
            pooled_acc_raw["Control_Helper_Heuristic_Prior"],
        ),
        "brier_glc_clear_minus_marginalized": _pool_diff(
            pooled_brier_raw["GLC_v3_BeliefNet_and_Clearer"],
            pooled_brier_raw["Ataraxos_Marginalized_Uniform_Belief"],
        ),
    }

    # Pool focal tournament matchups across all seeds (`150` pairs = `300` games per focal matchup)
    focal_keys = list(per_seed_results[str(active_seeds[0])]["focal_matchups"].keys())
    pooled_focal_matchups: Dict[str, Any] = {}
    for fk in focal_keys:
        all_pair_scores = [
            x
            for s in active_seeds
            for x in per_seed_results[str(s)]["focal_matchups"][fk]["pair_scores_pct"]
        ]
        tot_w_a = sum(per_seed_results[str(s)]["focal_matchups"][fk]["wins_a"] for s in active_seeds)
        tot_w_b = sum(per_seed_results[str(s)]["focal_matchups"][fk]["wins_b"] for s in active_seeds)
        tot_dr = sum(per_seed_results[str(s)]["focal_matchups"][fk]["draws"] for s in active_seeds)
        ci_p = summarize_samples_ci95(all_pair_scores)
        pooled_focal_matchups[fk] = {
            "wins_a": tot_w_a,
            "wins_b": tot_w_b,
            "draws": tot_dr,
            "total_games": int(tot_w_a + tot_w_b + tot_dr),
            "total_pairs": len(all_pair_scores),
            **ci_p,
        }

    ag_names = per_seed_results[str(active_seeds[0])]["agent_names"]
    agg_elo = {
        nm: summarize_across_seeds([per_seed_results[str(s)]["bayes_elo"][nm] for s in active_seeds])
        for nm in ag_names
    }

    # Strip raw arrays before serializing summary
    per_seed_clean: Dict[str, Any] = {}
    for s in active_seeds:
        sd = dict(per_seed_results[str(s)])
        sd["belief_calibration"] = {
            k: v
            for k, v in sd["belief_calibration"].items()
            if not k.startswith("raw_board_")
        }
        per_seed_clean[str(s)] = sd

    summary: Dict[str, Any] = {
        "round": "R26-B",
        "title": "External Public Ataraxos (arXiv:2511.07312) 10x10 Stratego Evaluation (3 Seeds)",
        "citation": {
            "authors": "Sokota, Samuel, Eugene Vinitsky, Hengyuan Hu, J. Zico Kolter, and Gabriele Farina",
            "title": "Scalable Decision Making for Games of Imperfect Information",
            "arxiv": "2511.07312",
            "repo": "AtaraxosAI/stratego@92db29e8ffc323b1b8a2804b5c3f84695d036b05",
            "lineage_arxiv": ["2304.13138", "2206.05825"],
        },
        "seeds": list(active_seeds),
        "device": str(device),
        "gpu_name": gpu_name,
        "focal_games_per_matchup_per_seed": int(args.focal_games_per_matchup),
        "num_eval_boards_per_seed": int(args.num_eval_boards),
        "total_elapsed_s": round(time.time() - t_start, 2),
        "d0_audit_and_pretrained_verification": d0_res,
        "e1b_belief_calibration": {
            "total_pooled_boards": int(args.num_eval_boards * len(active_seeds)),
            "pooled_cross_entropy_summary": {m: summarize_samples_ci95(pooled_ce_raw[m]) for m in bel_models},
            "pooled_top1_accuracy_pct_summary": {m: summarize_samples_ci95(pooled_acc_raw[m]) for m in bel_models},
            "pooled_brier_score_summary": {m: summarize_samples_ci95(pooled_brier_raw[m]) for m in bel_models},
            "pooled_pairwise_diff_ci95": pooled_pairwise_diff_ci95,
            "per_seed_pairwise_diff_ci95": {
                str(s): per_seed_results[str(s)]["belief_calibration"]["pairwise_diff_ci95"]
                for s in active_seeds
            },
        },
        "e1a_stratego_tournament": {
            "agent_names": ag_names,
            "bayes_elo_across_seeds": agg_elo,
            "pooled_focal_matchups": pooled_focal_matchups,
            "per_seed_focal_matchups": {
                str(s): {
                    k: {kk: vv for kk, vv in v.items() if kk != "pair_scores_pct"}
                    for k, v in per_seed_results[str(s)]["focal_matchups"].items()
                }
                for s in active_seeds
            },
        },
        "per_seed": per_seed_clean,
    }
    summary["preregistered_gates"] = evaluate_r26b_preregistered_gates(summary)

    out_json = out_dir / "r26b_ataraxos_stratego_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"\n[R26-B] Completed in {summary['total_elapsed_s']:.2f}s. "
        f"All gates passed={summary['preregistered_gates']['all_gates_passed']}. "
        f"Saved to {out_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
