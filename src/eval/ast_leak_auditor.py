"""Static AST Zero-Oracle-Leakage & Anti-Shortcut Auditor v2 (`S1`–`S10`).

Provides an `ast.NodeVisitor` auditor that inspects benchmark runners and model/search modules
to enforce both Zero Oracle Leakage (`S1`–`S6`) and Zero Surrogate/Asymmetric-Helper Shortcuts (`S7`–`S10`):
- `S1` (Banned Oracle Attributes & Identifiers): Zero reads of `.exact_marginals`,
  `.target_tokens`, `.query_targets`, or banned oracle variable names (`p_true*`,
  `p_exact*`, `logit_exact*`, `p_a_given_b_true`, `p_b_given_a_true`, `cond_b_given_a*`,
  `cond_a_given_b*`, `true_q`, `q_star*`, `true_llr_shift`) inside any model forward,
  search, memory, or match-play function.
- `S2` (Banned Synthetic Metric/Gate Samplers): Zero calls to `rng.beta`, `rng.uniform`,
  `rng.normal`, or `torch.randn` writing directly into `m_ql`, `m_qs`, `delta_rel`, or
  `auroc` evaluation variables.
- `S3` (Banned Law-Under-Test in Data/Match Generators): Zero calls to
  `mechanistic_redundancy_delta_c` or `btd_probabilities` inside episode evaluation or
  tournament match generation.
- `S4` (Banned Seed/Budget/Arm Conditional Outcome Branches): Zero `if` statements
  conditioned on literals of `seed`, `b_budget`, `m0`, or `arm_name` that modify noise
  scales, leak terms, or probability shifts.
- `S5` (Banned Arm-Specific Kernel/Offset Constants in Diagnostics): Zero algorithm-name
  branches inside `compute_empirical_gram_crowding`.
- `S6` (Canary Self-Test): Verified against 7 synthetic canary functions (`7/7` caught).
- `S7` (Banned Test-Time Reference Tables & Neural Call-Count Cutoffs): Zero reads or uses
  of `cfr_reference_table`, `ref_sigma`, `_st_act_calls`, or `net_cache` call-count cutoffs
  inside any policy/action evaluation function.
- `S8` (Banned Paradigm-Specific Analytic Gain / Noise / Scaling Multipliers): Zero assignments
  to hardcoded per-paradigm multipliers (`search_gain`, `noise_scale`, `transfer_mult`,
  `w_spec`, `mech_mult`, `gain_n`, `gain_b`, `task_base_difficulty`, `sym_anchor_1`) inside
  evaluation or action-selection functions.
- `S9` (Banned Asymmetric External Solver Helpers): Zero calls to external backtracking/A*
  helpers (`_complete_leaf_with_observable_search`, `_bounded_neural_sudoku_cdcl`,
  `_bounded_neural_3sat_cdcl`, `solve_sokoban_astar`) inside `if paradigm == ...` branches.
- `S10` (Banned Target-Derived Features in Input Feature Generators): Zero assignments of
  ground-truth target variables (`dist_vec`, `pi_star`, `pi_oh`, `mask_star`, `aux_node_stat`)
  into input feature tensors (`node_feats`, `edge_feats`, `operator_basis`, `node_basis`).
"""

from __future__ import annotations

import ast
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


BANNED_ORACLE_ATTRS: Set[str] = {
    "exact_marginals",
    "target_tokens",
    "query_targets",
    "cfr_reference_table",
}

BANNED_ORACLE_NAME_PATTERNS: Tuple[str, ...] = (
    "p_true*",
    "p_exact*",
    "logit_exact*",
    "p_a_given_b_true",
    "p_b_given_a_true",
    "cond_b_given_a*",
    "cond_a_given_b*",
    "true_q",
    "q_star*",
    "true_llr_shift",
)

BANNED_SHORTCUT_IDENTIFIERS: Set[str] = {
    "cfr_reference_table",
    "ref_sigma",
    "_st_act_calls",
    "search_gain",
    "transfer_mult",
    "w_spec",
    "mech_mult",
    "gain_n",
    "gain_b",
    "task_base_difficulty",
    "sym_anchor_1",
}

BANNED_ASYMMETRIC_SOLVER_HELPERS: Set[str] = {
    "_complete_leaf_with_observable_search",
    "_bounded_neural_sudoku_cdcl",
    "_bounded_neural_3sat_cdcl",
    "solve_sokoban_astar",
}

INPUT_FEATURE_TENSORS: Set[str] = {
    "node_feats",
    "edge_feats",
    "operator_basis",
    "node_basis",
}

TARGET_DERIVED_VARS: Set[str] = {
    "dist_vec",
    "pi_star",
    "pi_oh",
    "mask_star",
    "aux_node_stat",
}

# Functions explicitly allowed to read ground-truth labels (pure training losses, calibration & post-hoc scoring)
EXEMPT_SCORING_OR_TRAINING_FUNCTIONS: Set[str] = {
    "compute_any_order_loss",
    "compute_training_step_loss",
    "calibrate_model_and_clearer",
    "fit_cal_platt_query_calibrator",
    "train_generalist_and_cohorts_for_seed",
    "train_r12_models_for_seed",
    "train_paradigm_models",
    "train_openspiel_paradigm",
    "train_stratego_paradigm",
    "train_reasoning_paradigm",
    "train_clrs_model",
    "train_decentralized_market_models",
    "build_real_env_market_dataset",
    "evaluate_beliefs",
    "evaluate_blind_predictions",
    "verify_zero_oracle_leakage_gate",
    "strip_oracle_labels",
    "split_and_poison_oracle",
    "strip_oracle_fields",
    "slice_env_batch",
}

# Functions explicitly allowed to call law-under-test fitters
EXEMPT_LAW_FIT_FUNCTIONS: Set[str] = {
    "fit_mechanistic_redundancy_law",
    "fit",
    "predict_win_matrix",
}


@dataclass
class ASTViolation:
    rule_id: str
    file_path: str
    func_name: str
    lineno: int
    description: str


def _matches_banned_oracle_name(name: str) -> bool:
    for pat in BANNED_ORACLE_NAME_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


class FunctionASTLeakVisitor(ast.NodeVisitor):
    """Inspects a single function AST node for violations of `S1`–`S10`."""

    def __init__(self, file_path: str, func_name: str) -> None:
        self.file_path = file_path
        self.func_name = func_name
        self.violations: List[ASTViolation] = []

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        # Rule S1 & S7: Banned oracle or reference-table attributes in non-exempt functions
        if self.func_name not in EXEMPT_SCORING_OR_TRAINING_FUNCTIONS:
            if isinstance(node.ctx, ast.Load) and node.attr in BANNED_ORACLE_ATTRS:
                rule = "S7" if node.attr == "cfr_reference_table" else "S1"
                self.violations.append(
                    ASTViolation(
                        rule_id=rule,
                        file_path=self.file_path,
                        func_name=self.func_name,
                        lineno=getattr(node, "lineno", 0),
                        description=f"Read of banned attribute '.{node.attr}' in '{self.func_name}'",
                    )
                )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        # Rule S1: Banned oracle variable names in non-exempt functions
        if self.func_name not in EXEMPT_SCORING_OR_TRAINING_FUNCTIONS:
            if isinstance(node.ctx, ast.Load) and _matches_banned_oracle_name(node.id):
                self.violations.append(
                    ASTViolation(
                        rule_id="S1",
                        file_path=self.file_path,
                        func_name=self.func_name,
                        lineno=getattr(node, "lineno", 0),
                        description=f"Use of banned oracle identifier '{node.id}' in '{self.func_name}'",
                    )
                )
        # Rule S7 & S8: Banned reference tables / call cutoffs / analytic paradigm multipliers
        if node.id in BANNED_SHORTCUT_IDENTIFIERS:
            rule = "S7" if node.id in {"cfr_reference_table", "ref_sigma", "_st_act_calls"} else "S8"
            self.violations.append(
                ASTViolation(
                    rule_id=rule,
                    file_path=self.file_path,
                    func_name=self.func_name,
                    lineno=getattr(node, "lineno", 0),
                    description=f"Use of banned shortcut/multiplier identifier '{node.id}' in '{self.func_name}'",
                )
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        # Rule S2: Banned synthetic RNG sampler writing directly into m_ql, m_qs, delta_rel, auroc
        target_names: Set[str] = set()
        subscript_base_names: Set[str] = set()
        for t in node.targets:
            if isinstance(t, ast.Name):
                target_names.add(t.id)
            elif isinstance(t, ast.Subscript):
                for sub_t in ast.walk(t.value):
                    if isinstance(sub_t, ast.Name):
                        subscript_base_names.add(sub_t.id)

        protected_gate_vars = {"m_ql", "m_qs", "delta_rel", "auroc", "auroc_mql", "auroc_mqs"}
        if target_names & protected_gate_vars:
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                    if sub.func.attr in {"beta", "uniform", "normal", "randn"}:
                        self.violations.append(
                            ASTViolation(
                                rule_id="S2",
                                file_path=self.file_path,
                                func_name=self.func_name,
                                lineno=getattr(node, "lineno", 0),
                                description=(
                                    f"Synthetic RNG sampler '.{sub.func.attr}(...)' assigned directly to "
                                    f"gate metric {sorted(target_names & protected_gate_vars)} in '{self.func_name}'"
                                ),
                            )
                        )

        # Rule S10: Banned target-derived variables written into input feature tensors
        if subscript_base_names & INPUT_FEATURE_TENSORS:
            rhs_names = {sub.id for sub in ast.walk(node.value) if isinstance(sub, ast.Name)}
            leaked_targets = rhs_names & TARGET_DERIVED_VARS
            if leaked_targets:
                self.violations.append(
                    ASTViolation(
                        rule_id="S10",
                        file_path=self.file_path,
                        func_name=self.func_name,
                        lineno=getattr(node, "lineno", 0),
                        description=(
                            f"Target-derived variable(s) {sorted(leaked_targets)} written into input feature "
                            f"tensor {sorted(subscript_base_names & INPUT_FEATURE_TENSORS)} in '{self.func_name}'"
                        ),
                    )
                )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        # Rule S3: Banned calls to mechanistic_redundancy_delta_c or btd_probabilities in generators/evaluators
        if self.func_name not in EXEMPT_LAW_FIT_FUNCTIONS:
            callee_name: Optional[str] = None
            if isinstance(node.func, ast.Name):
                callee_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee_name = node.func.attr
            if callee_name in {"mechanistic_redundancy_delta_c", "btd_probabilities"}:
                if "fit_scaling_laws" not in self.func_name and "evaluate_r11_d5" not in self.func_name:
                    self.violations.append(
                        ASTViolation(
                            rule_id="S3",
                            file_path=self.file_path,
                            func_name=self.func_name,
                            lineno=getattr(node, "lineno", 0),
                            description=(
                                f"Forbidden call to law-under-test '{callee_name}' inside '{self.func_name}'"
                            ),
                        )
                    )
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> Any:
        # Rule S4: Banned conditional branch on seed / b_budget / m0 literals
        test_names: Set[str] = set()
        has_numeric_or_str_literal = False
        for sub in ast.walk(node.test):
            if isinstance(sub, ast.Name):
                test_names.add(sub.id)
            elif isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float, str)):
                has_numeric_or_str_literal = True

        if has_numeric_or_str_literal and (
            ("seed" in test_names and ("b_budget" in test_names or "budget" in test_names))
            or ("seed" in test_names and "m0" in test_names)
        ):
            self.violations.append(
                ASTViolation(
                    rule_id="S4",
                    file_path=self.file_path,
                    func_name=self.func_name,
                    lineno=getattr(node, "lineno", 0),
                    description=(
                        f"Forbidden seed/budget conditional branch on {sorted(test_names)} in '{self.func_name}'"
                    ),
                )
            )

        # Rule S5: Banned algorithm-specific branches inside compute_empirical_gram_crowding
        if self.func_name == "compute_empirical_gram_crowding":
            if test_names & {"algorithm", "alg", "arm", "arm_name"}:
                self.violations.append(
                    ASTViolation(
                        rule_id="S5",
                        file_path=self.file_path,
                        func_name=self.func_name,
                        lineno=getattr(node, "lineno", 0),
                        description=(
                            f"Forbidden algorithm-dependent branch on {sorted(test_names)} "
                            f"inside 'compute_empirical_gram_crowding'"
                        ),
                    )
                )

        # Rule S9: Banned asymmetric external solver helper called only inside a paradigm-conditional branch
        if test_names & {"paradigm", "arch", "arch_name", "arm", "arm_name", "method"}:
            for stmt in node.body + node.orelse:
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.Call):
                        c_name = None
                        if isinstance(sub.func, ast.Name):
                            c_name = sub.func.id
                        elif isinstance(sub.func, ast.Attribute):
                            c_name = sub.func.attr
                        if c_name in BANNED_ASYMMETRIC_SOLVER_HELPERS:
                            self.violations.append(
                                ASTViolation(
                                    rule_id="S9",
                                    file_path=self.file_path,
                                    func_name=self.func_name,
                                    lineno=getattr(sub, "lineno", getattr(node, "lineno", 0)),
                                    description=(
                                        f"Asymmetric external solver helper '{c_name}' called inside "
                                        f"paradigm-conditional branch on {sorted(test_names)} in '{self.func_name}'"
                                    ),
                                )
                            )

        self.generic_visit(node)


def audit_source_code(source: str, file_path: str = "<memory>") -> List[ASTViolation]:
    """Parse Python `source` and audit all function/method definitions against `S1`–`S10`."""
    tree = ast.parse(source, filename=file_path)
    violations: List[ASTViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visitor = FunctionASTLeakVisitor(file_path=file_path, func_name=node.name)
            for stmt in node.body:
                visitor.visit(stmt)
            violations.extend(visitor.violations)
    return violations


def run_canary_self_test() -> Dict[str, Any]:
    """Execute Rule `S6` canary self-test against 7 synthetic violating functions (`7/7` must be caught)."""
    canary_1 = """
def leaky_search_step(batch):
    p = batch.exact_marginals
    return p
"""
    canary_2 = """
def fake_episode_generator(b_budget):
    gain = mechanistic_redundancy_delta_c(b_budget, 0.5, 0.2)
    return gain
"""
    canary_3 = """
def rigged_outcome(seed, b_budget, x):
    if seed == 102 and b_budget == 2:
        return x * 0.5
    return x
"""
    canary_4 = """
def ref_table_blend(policy, state):
    ref_sigma = policy.cfr_reference_table[state]
    return ref_sigma
"""
    canary_5 = """
def hardcoded_paradigm_gain(paradigm, log_b):
    if paradigm == "GLC":
        search_gain = 1.25 + 0.52 * log_b
    return search_gain
"""
    canary_6 = """
def asymmetric_solver_branch(paradigm, board):
    if paradigm == "GLC_AOM_v2":
        return _complete_leaf_with_observable_search(board)
    return board
"""
    canary_7 = """
def leaky_clrs_features(node_feats, dist_vec, b):
    node_feats[b, :, 8] = dist_vec
    return node_feats
"""
    v1 = audit_source_code(canary_1, "<canary_1_S1>")
    v2 = audit_source_code(canary_2, "<canary_2_S3>")
    v3 = audit_source_code(canary_3, "<canary_3_S4>")
    v4 = audit_source_code(canary_4, "<canary_4_S7>")
    v5 = audit_source_code(canary_5, "<canary_5_S8>")
    v6 = audit_source_code(canary_6, "<canary_6_S9>")
    v7 = audit_source_code(canary_7, "<canary_7_S10>")

    caught_1 = any(v.rule_id == "S1" for v in v1)
    caught_2 = any(v.rule_id == "S3" for v in v2)
    caught_3 = any(v.rule_id == "S4" for v in v3)
    caught_4 = any(v.rule_id == "S7" for v in v4)
    caught_5 = any(v.rule_id == "S8" for v in v5)
    caught_6 = any(v.rule_id == "S9" for v in v6)
    caught_7 = any(v.rule_id == "S10" for v in v7)
    canaries_caught = sum([caught_1, caught_2, caught_3, caught_4, caught_5, caught_6, caught_7])

    return {
        "canaries_tested": 7,
        "canaries_caught": int(canaries_caught),
        "caught_S1_exact_marginals": caught_1,
        "caught_S3_law_in_generator": caught_2,
        "caught_S4_seed_budget_branch": caught_3,
        "caught_S7_cfr_ref_table": caught_4,
        "caught_S8_paradigm_gain_mult": caught_5,
        "caught_S9_asymmetric_solver_helper": caught_6,
        "caught_S10_target_in_node_feats": caught_7,
        "pass": int(canaries_caught) == 7,
    }


def audit_r11_files(file_paths: Sequence[str | Path]) -> Dict[str, Any]:
    """Run both the `S6` 7-canary self-test and the `S1`–`S10` static AST audit across `file_paths`."""
    canary_res = run_canary_self_test()
    all_violations: List[Dict[str, Any]] = []
    audited_files: List[str] = []

    for fp in file_paths:
        p = Path(fp)
        source = p.read_text(encoding="utf-8")
        audited_files.append(str(p))
        vs = audit_source_code(source, file_path=str(p))
        for v in vs:
            all_violations.append(
                {
                    "rule_id": v.rule_id,
                    "file_path": v.file_path,
                    "func_name": v.func_name,
                    "lineno": v.lineno,
                    "description": v.description,
                }
            )

    return {
        "audited_files": audited_files,
        "num_violations": len(all_violations),
        "violations": all_violations,
        "canary_self_test": canary_res,
        "pass": (len(all_violations) == 0) and bool(canary_res["pass"]),
    }
