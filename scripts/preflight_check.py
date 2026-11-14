#!/usr/bin/env python3
"""
Deterministic Preflight Verifier for Confirmatory Rounds (docs/PROTOCOL.md §10.2).

Every confirmatory PREREG + runner + smoke JSON must pass this script before handoff
to opus-mats for freeze review:
  1. Verifies Gate-Text Table in PREREG matches the exact AST boolean expressions in
     the runner's evaluate_*_gates() function (Rule 9.2, 0 mismatches), and verifies
     all backticked inequality thresholds in PREREG prose match the runner AST.
  2. Verifies Rule 9.9 Bar Provenance / History: checks `(tightened from X)` and
     `(maintained from X)` claims in PREREG and compares shared metrics against prior
     round PREREGs (`benchmarks/PREREG-R27*.md`) so loosened bars fail immediately.
  3. Verifies zero overlap between train, development, smoke, and test splits and
     disjoint smoke vs. full RNG seeds in both the smoke JSON and runner AST (Rule 9.8 / 9.9).
  4. Verifies Control_Random_Network and Control_Untrained_Init are required in the
     runner AST, gated in every primary empirical gate / method prefix, present in
     the smoke JSON, and fail the main performance bars (Rule 9.1).
  5. Verifies paired CI calls (`paired_ci95`, `paired_diff_stats_df2`) have the
     correct argument order and sign for lower-is-better vs. higher-is-better metrics,
     and that audit counters (`*oracle_calls*`, `*label_access*`, `*overlap*`) are not
     assigned or incremented by literal constants.
  6. Verifies the deterministic scorer refuses the smoke JSON without --allow-smoke
     (exit code != 0) and succeeds with --allow-smoke (Rule 9.8), that smoke gates
     emit `verdict == "SMOKE_NO_VERDICT"` (without requiring `pass: true`), that
     re-evaluating `evaluate_*_gates(smoke_json)` matches the stored `smoke_json["gates"]`,
     and that `runner_sha256` and `prereg_sha256` match the on-disk files.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

LOWER_IS_BETTER_TOKENS = ("nc", "nash_conv", "coll", "collisions", "nll", "loss", "err", "brier", "ece")
HIGHER_IS_BETTER_TOKENS = ("acc", "rate", "v1", "sw", "welfare")

# Frozen prior-round bar registry (from benchmarks/PREREG-R27A*.md and PREREG-R27C*.md)
PRIOR_ROUND_BARS = {
    "R28B": {
        "kuhn_ann_nc": ("<", 0.05, "R27C kuhn_ours_nc <= 0.05 (tightened)"),
        "leduc_ann_nc": ("<", 0.95, "R27C leduc_ours_nc <= 0.95 (tightened)"),
        "liars_ann_nc": ("<=", 0.12, "R27C liars_ours_nc <= 0.12 (maintained/tightened)"),
    },
    "R28A": {
        "mean_clue_acc": (">=", 0.95, "R27A clue_acc >= 0.95"),
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_evaluate_gates_fn(tree: ast.AST) -> Optional[ast.FunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("evaluate_") and node.name.endswith("_gates"):
            return node
    return None


def extract_gate_exprs_from_runner(runner_path: Path) -> Dict[str, Tuple[int, int, str, str]]:
    """Returns {gate_id: (lineno, end_lineno, normalized_ast_expr, var_name)}."""
    src = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    var_exprs: Dict[str, Tuple[int, int, str]] = {}
    gate_id_to_var: Dict[str, str] = {}

    fn_node = _find_evaluate_gates_fn(tree)
    if fn_node is not None:
        for sub in ast.walk(fn_node):
            if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
                vname = sub.targets[0].id
                if vname.endswith("_pass") or vname == "controls_ok":
                    inner = sub.value
                    if isinstance(inner, ast.Call) and getattr(inner.func, "id", "") == "bool" and inner.args:
                        inner = inner.args[0]
                    var_exprs[vname] = (sub.lineno, getattr(sub, "end_lineno", sub.lineno), ast.unparse(inner))
            if isinstance(sub, ast.Dict):
                for k_node, v_node in zip(sub.keys, sub.values):
                    if isinstance(k_node, ast.Constant) and isinstance(k_node.value, str) and "_D" in k_node.value:
                        if isinstance(v_node, ast.Dict):
                            for gk, gv in zip(v_node.keys, v_node.values):
                                if isinstance(gk, ast.Constant) and gk.value == "pass" and isinstance(gv, ast.Name):
                                    gate_id_to_var[k_node.value] = gv.id

    out: Dict[str, Tuple[int, int, str, str]] = {}
    for gate_id, vname in gate_id_to_var.items():
        if vname in var_exprs:
            l0, l1, expr_s = var_exprs[vname]
            out[gate_id] = (l0, l1, expr_s, vname)
    return out


def _op_to_str(op: ast.cmpop) -> str:
    if isinstance(op, ast.LtE):
        return "<="
    if isinstance(op, ast.Lt):
        return "<"
    if isinstance(op, ast.GtE):
        return ">="
    if isinstance(op, ast.Gt):
        return ">"
    if isinstance(op, ast.Eq):
        return "=="
    if isinstance(op, ast.NotEq):
        return "!="
    return ""


def _num_from_ast(node: ast.AST) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        val = float(node.operand.value)
        return -val if isinstance(node.op, ast.USub) else val
    return None


def extract_numeric_thresholds_in_gates(runner_path: Path) -> Tuple[Dict[str, Tuple[str, float]], List[str]]:
    """
    Extracts {lhs_unparsed: (op_str, float_val)} from all `*_pass` boolean expressions
    in `evaluate_*_gates`, and checks that any duplicate comparisons on the same LHS
    inside `evaluate_*_gates` use the exact same threshold.
    """
    errors: List[str] = []
    src = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn_node = _find_evaluate_gates_fn(tree)
    if fn_node is None:
        return {}, [f"No evaluate_*_gates function found in {runner_path}"]

    pass_thresholds: Dict[str, Tuple[str, float]] = {}
    for sub in ast.walk(fn_node):
        if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
            vname = sub.targets[0].id
            if vname.endswith("_pass"):
                for cmp_node in ast.walk(sub.value):
                    if isinstance(cmp_node, ast.Compare) and len(cmp_node.ops) == 1 and len(cmp_node.comparators) == 1:
                        op_s = _op_to_str(cmp_node.ops[0])
                        rval = _num_from_ast(cmp_node.comparators[0])
                        if op_s and rval is not None:
                            lhs_s = ast.unparse(cmp_node.left)
                            if lhs_s in pass_thresholds and pass_thresholds[lhs_s] != (op_s, rval):
                                errors.append(
                                    f"Inconsistent threshold for `{lhs_s}` inside {fn_node.name}: "
                                    f"{pass_thresholds[lhs_s]} vs {(op_s, rval)}"
                                )
                            pass_thresholds[lhs_s] = (op_s, rval)

    # Also verify any other comparison in evaluate_*_gates on a simple Name in pass_thresholds agrees
    for cmp_node in ast.walk(fn_node):
        if isinstance(cmp_node, ast.Compare) and len(cmp_node.ops) == 1 and len(cmp_node.comparators) == 1:
            if isinstance(cmp_node.left, ast.Name) and cmp_node.left.id in pass_thresholds:
                op_s = _op_to_str(cmp_node.ops[0])
                rval = _num_from_ast(cmp_node.comparators[0])
                if op_s and rval is not None:
                    exp_op, exp_val = pass_thresholds[cmp_node.left.id]
                    if op_s != exp_op or abs(rval - exp_val) > 1e-12:
                        errors.append(
                            f"Intra-runner threshold mismatch for `{cmp_node.left.id}` at line {cmp_node.lineno}: "
                            f"`{cmp_node.left.id} {op_s} {rval}` vs gate `{cmp_node.left.id} {exp_op} {exp_val}`"
                        )

    return pass_thresholds, errors


def check_gate_text_table_and_prose(prereg_path: Path, runner_path: Path) -> List[str]:
    errors: List[str] = []
    prereg_text = prereg_path.read_text(encoding="utf-8")
    code_gates = extract_gate_exprs_from_runner(runner_path)
    if not code_gates:
        errors.append(f"No gate expressions found in {runner_path}")
        return errors

    for gate_id, (l0, l1, c_expr, _) in code_gates.items():
        pattern = r"\|\s*`" + re.escape(gate_id) + r"`[^|]*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|"
        m = re.search(pattern, prereg_text)
        if not m:
            errors.append(f"Missing Gate-Text Table row in {prereg_path.name} for `{gate_id}`")
            continue
        loc_str, prereg_expr = m.group(1), m.group(2)
        try:
            p_ast = ast.parse(prereg_expr, mode="eval").body
            if isinstance(p_ast, ast.Call) and getattr(p_ast.func, "id", "") == "bool" and p_ast.args:
                p_ast = p_ast.args[0]
            p_norm = ast.unparse(p_ast)
            c_norm = ast.unparse(ast.parse(c_expr, mode="eval").body)
            if p_norm != c_norm:
                errors.append(f"AST mismatch on `{gate_id}`:\n  PREREG: {p_norm}\n  CODE:   {c_norm}")
        except SyntaxError as exc:
            errors.append(f"SyntaxError parsing PREREG expression for `{gate_id}`: {exc}")

        expected_loc = f"{l0}-{l1}" if l0 != l1 else f"{l0}"
        expected_loc_en = f"{l0}–{l1}" if l0 != l1 else f"{l0}"
        if expected_loc not in loc_str and expected_loc_en not in loc_str and str(l0) not in loc_str:
            errors.append(f"Stale line reference on `{gate_id}`: PREREG says `{loc_str}`, actual lines `{l0}-{l1}`")

    # Check prose backticked inequalities against AST thresholds (catches M2_prose_only)
    pass_thresholds, intra_errs = extract_numeric_thresholds_in_gates(runner_path)
    errors.extend(intra_errs)

    backtick_spans = re.findall(r"`([^`\n]+)`", prereg_text)
    cmp_regex = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(<=|<|>=|>|==)\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)$")
    for span in backtick_spans:
        m_cmp = cmp_regex.match(span.strip())
        if m_cmp:
            var_name, p_op, p_val_s = m_cmp.group(1), m_cmp.group(2), m_cmp.group(3)
            if var_name in pass_thresholds:
                c_op, c_val = pass_thresholds[var_name]
                p_val = float(p_val_s)
                if p_op != c_op or abs(p_val - c_val) > 1e-12:
                    errors.append(
                        f"Prose-to-AST bar mismatch for `{var_name}` in {prereg_path.name}: "
                        f"prose has `{span.strip()}`, runner AST has `{var_name} {c_op} {c_val}`"
                    )

    return errors


def check_bar_history(prereg_path: Path, runner_path: Path) -> List[str]:
    """Rule 9.9 Bar-History check: verifies `(tightened from ...)` claims and prior round PREREG bars."""
    errors: List[str] = []
    prereg_text = prereg_path.read_text(encoding="utf-8")
    pass_thresholds, _ = extract_numeric_thresholds_in_gates(runner_path)

    # 1. Verify explicit `(tightened from X)` / `(maintained from X)` annotations in PREREG
    ann_pattern = re.compile(
        r"`([A-Za-z_][A-Za-z0-9_]*)\s*(<=|<|>=|>|==)\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)`\s*"
        r"\((tightened|maintained|unchanged)\s+from\s+`?([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)`?",
        re.IGNORECASE,
    )
    for m in ann_pattern.finditer(prereg_text):
        var_name, op_s, new_s, claim_word, old_s = (
            m.group(1),
            m.group(2),
            m.group(3),
            m.group(4).lower(),
            m.group(5),
        )
        new_val = float(new_s)
        old_val = float(old_s)
        if claim_word == "tightened":
            if op_s in ("<=", "<") and not (new_val < old_val):
                errors.append(
                    f"Rule 9.9 bar-history violation on `{var_name}`: claims tightened from {old_val} "
                    f"but new bar `{var_name} {op_s} {new_val}` is not strictly tighter (< {old_val})"
                )
            elif op_s in (">=", ">") and not (new_val > old_val):
                errors.append(
                    f"Rule 9.9 bar-history violation on `{var_name}`: claims tightened from {old_val} "
                    f"but new bar `{var_name} {op_s} {new_val}` is not strictly tighter (> {old_val})"
                )
        elif claim_word in ("maintained", "unchanged"):
            if abs(new_val - old_val) > 1e-12:
                errors.append(
                    f"Rule 9.9 bar-history violation on `{var_name}`: claims {claim_word} from {old_val} "
                    f"but new bar is {new_val}"
                )

    # 2. Compare against prior round PREREG bars (R27A / R27C -> R28A / R28B)
    for rnd_key, bar_map in PRIOR_ROUND_BARS.items():
        if rnd_key in prereg_path.name.upper() or rnd_key.lower() in runner_path.name.lower():
            for var_name, (req_rel, old_bar, desc) in bar_map.items():
                if var_name in pass_thresholds:
                    _, cur_val = pass_thresholds[var_name]
                    if req_rel == "<" and not (cur_val < old_bar):
                        errors.append(
                            f"Rule 9.9 bar-history violation on `{var_name}`: current bar {cur_val} "
                            f"is not tighter than prior round ({desc})"
                        )
                    elif req_rel == "<=" and not (cur_val <= old_bar):
                        errors.append(
                            f"Rule 9.9 bar-history violation on `{var_name}`: current bar {cur_val} "
                            f"loosens prior round bar ({desc})"
                        )
                    elif req_rel == ">" and not (cur_val > old_bar):
                        errors.append(
                            f"Rule 9.9 bar-history violation on `{var_name}`: current bar {cur_val} "
                            f"is not tighter than prior round ({desc})"
                        )
                    elif req_rel == ">=" and not (cur_val >= old_bar):
                        errors.append(
                            f"Rule 9.9 bar-history violation on `{var_name}`: current bar {cur_val} "
                            f"loosens prior round bar ({desc})"
                        )

    return errors


def check_runner_ast_integrity(runner_path: Path) -> List[str]:
    """
    AST-level integrity checks on the runner:
    - M3 / M3b: Every primary gate (`d1_pass`, `d2_pass`) and every gated method prefix
      (`ci_<method>_vs_*`) must gate against `Control_Random_Network` (`random_net` / `rand`),
      and every defined `*random_net*` comparison in `evaluate_*_gates` must be used in a `*_pass` expr.
    - M4: Paired CI calls (`paired_ci95`, `paired_diff_stats_df2`) must have the right argument
      order for lower-is-better vs. higher-is-better metrics.
    - M5: `rng_offset` must not be assigned a literal constant `0`.
    - M6 / M6b: Every required-arms list (`*arms*`) in `evaluate_*_gates` must include both
      `Control_Random_Network` and `Control_Untrained_Init`.
    - M7: Audit counters (`*oracle_calls*`, `*label_access*`, `*overlap*`, or any `+= 0`) must
      not be assigned or augmented by a literal constant `0`.
    """
    errors: List[str] = []
    src = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # 1. Check all functions for M5 (rng_offset = 0) and M7 (counter += 0 or overlap = 0)
    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign):
            if isinstance(node.value, ast.Constant) and node.value.value == 0:
                errors.append(
                    f"Line {node.lineno}: suspicious constant `+= 0` on `{ast.unparse(node.target)}` "
                    f"(audit counters must be computed dynamically)"
                )
            t_str = ast.unparse(node.target)
            if any(tok in t_str for tok in ("oracle_calls", "label_access", "overlap")):
                if isinstance(node.value, ast.Constant):
                    errors.append(
                        f"Line {node.lineno}: audit counter `{t_str}` augmented by literal constant "
                        f"`{ast.unparse(node.value)}`"
                    )
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            vname = node.targets[0].id
            if vname == "rng_offset":
                if isinstance(node.value, ast.Constant) and node.value.value == 0:
                    errors.append(
                        f"Line {node.lineno}: `rng_offset` is assigned constant `0` "
                        f"(smoke runs must use a non-zero RNG seed offset)"
                    )
            if "overlap" in vname and isinstance(node.value, ast.Constant):
                errors.append(
                    f"Line {node.lineno}: overlap check `{vname}` assigned literal constant `{node.value.value}`"
                )

    fn_node = _find_evaluate_gates_fn(tree)
    if fn_node is None:
        errors.append(f"No evaluate_*_gates function found in {runner_path}")
        return errors

    # 2. Check M6 / M6b: required arm lists in evaluate_*_gates must include Control_Random_Network & Control_Untrained_Init
    arm_lists_found = 0
    for sub in ast.walk(fn_node):
        if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
            vname = sub.targets[0].id
            if "arms" in vname and isinstance(sub.value, (ast.List, ast.Tuple, ast.Set)):
                arm_lists_found += 1
                elts = [e.value for e in sub.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                has_rand = any("Control_Random_Network" in s for s in elts)
                has_untr = any("Control_Untrained_Init" in s for s in elts)
                if not has_rand:
                    errors.append(
                        f"Rule 9.1 violation in `{vname}` (line {sub.lineno}): missing `Control_Random_Network`"
                    )
                if not has_untr:
                    errors.append(
                        f"Rule 9.1 violation in `{vname}` (line {sub.lineno}): missing `Control_Untrained_Init`"
                    )
    if arm_lists_found == 0:
        errors.append(f"Rule 9.1 violation: no required-arms list (`*arms*`) found in `{fn_node.name}`")

    # 3. Collect all `*_pass` expressions and check M3 / M3b (random-net gate clauses)
    pass_unparsed: Dict[str, str] = {}
    for sub in ast.walk(fn_node):
        if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
            vname = sub.targets[0].id
            if vname.endswith("_pass"):
                pass_unparsed[vname] = ast.unparse(sub.value)

    all_pass_concat = "\n".join(pass_unparsed.values())

    # 3a. Any `ci_*random_net*` or `*_vs_rand*` key/variable defined in evaluate_*_gates must appear in a `*_pass` expr
    for sub in ast.walk(fn_node):
        if isinstance(sub, ast.Dict):
            for k_node in sub.keys:
                if isinstance(k_node, ast.Constant) and isinstance(k_node.value, str):
                    k_str = k_node.value
                    if ("random_net" in k_str or "_vs_rand" in k_str) and k_str.startswith("ci_"):
                        if k_str not in all_pass_concat:
                            errors.append(
                                f"Rule 9.1 violation: random-network CI key `{k_str}` (line {k_node.lineno}) "
                                f"is computed in `{fn_node.name}` but omitted from gate `*_pass` expressions"
                            )
        if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
            vname = sub.targets[0].id
            if ("_vs_rand" in vname or "random_net" in vname) and vname not in all_pass_concat:
                errors.append(
                    f"Rule 9.1 violation: random-network variable `{vname}` (line {sub.lineno}) "
                    f"is computed in `{fn_node.name}` but omitted from gate `*_pass` expressions"
                )

    # 3b. For every `ci_<method>_vs_<target>` family in any `*_pass` expression, `<method>_vs_random_net` must be present
    ci_family_pattern = re.compile(r"\b(ci_[A-Za-z0-9_]+?)_vs_([A-Za-z0-9_]+)\b")
    for p_name, p_expr in pass_unparsed.items():
        families: Dict[str, Set[str]] = {}
        for m in ci_family_pattern.finditer(p_expr):
            prefix, target = m.group(1), m.group(2)
            families.setdefault(prefix, set()).add(target)
        for prefix, targets in families.items():
            if "random_net" not in targets and "rand" not in targets:
                errors.append(
                    f"Rule 9.1 violation in `{p_name}`: method prefix `{prefix}` gates against "
                    f"{sorted(targets)} but drops `{prefix}_vs_random_net`"
                )

    # 4. Check M4: Paired CI argument order & sign in evaluate_*_gates
    # Map variable names assigned inside evaluate_*_gates to their unparsed RHS so we can inspect metric & arm
    local_var_rhs: Dict[str, str] = {}
    for sub in ast.walk(fn_node):
        if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
            local_var_rhs[sub.targets[0].id] = ast.unparse(sub.value)

    def _check_paired_call(ci_label: str, call_node: ast.Call, lineno: int) -> None:
        if len(call_node.args) < 2:
            return
        arg0_s = ast.unparse(call_node.args[0])
        arg1_s = ast.unparse(call_node.args[1])
        arg0_full = f"{arg0_s} {local_var_rhs.get(arg0_s, '')}".lower()
        arg1_full = f"{arg1_s} {local_var_rhs.get(arg1_s, '')}".lower()
        combined = f"{ci_label.lower()} {arg0_full} {arg1_full}"

        tokens = set(re.findall(r"[a-z0-9]+", combined))
        is_lower_better = bool(tokens.intersection(LOWER_IS_BETTER_TOKENS)) or "nash_conv" in combined
        is_higher_better = bool(tokens.intersection(HIGHER_IS_BETTER_TOKENS))

        # Determine how ci_label is gated in all_pass_concat (`ci95_low > 0` vs `ci95_high < 0`)
        uses_low_gt0 = (
            f"'{ci_label}']['ci95_low'] > 0" in all_pass_concat
            or f'"{ci_label}"]["ci95_low"] > 0' in all_pass_concat
            or f"{ci_label}['ci95_low'] > 0" in all_pass_concat
            or f'{ci_label}["ci95_low"] > 0' in all_pass_concat
        )
        uses_high_lt0 = (
            f"'{ci_label}']['ci95_high'] < 0" in all_pass_concat
            or f'"{ci_label}"]["ci95_high"] < 0' in all_pass_concat
            or f"{ci_label}['ci95_high'] < 0" in all_pass_concat
            or f'{ci_label}["ci95_high"] < 0' in all_pass_concat
        )

        # Identify which argument is the primary method ("ours" / "annealed" / "sampled_b16")
        primary_tokens = ("ours", "ann_nc", "samp_b16_nc", "inconsistency_augmented", "inconsistency_annealed", "inconsistency_sampled")
        arg0_is_primary = any(t in arg0_full for t in primary_tokens)
        arg1_is_primary = any(t in arg1_full for t in primary_tokens)

        if is_lower_better and uses_low_gt0:
            # Gate checks ci95_low > 0 on a lower-is-better metric (e.g. NashConv), so diff = baseline - ours > 0.
            # Therefore arg0 must be baseline and arg1 must be primary (ours).
            if arg0_is_primary and not arg1_is_primary:
                errors.append(
                    f"Line {lineno}: inverted paired CI direction on `{ci_label}`: lower-is-better metric "
                    f"gated with `ci95_low > 0` requires `paired_ci(baseline, ours)`, got `({arg0_s}, {arg1_s})`"
                )
        elif is_lower_better and uses_high_lt0:
            # Gate checks ci95_high < 0 on a lower-is-better metric, so diff = ours - baseline < 0.
            if arg1_is_primary and not arg0_is_primary:
                errors.append(
                    f"Line {lineno}: inverted paired CI direction on `{ci_label}`: lower-is-better metric "
                    f"gated with `ci95_high < 0` requires `paired_ci(ours, baseline)`, got `({arg0_s}, {arg1_s})`"
                )
        elif is_higher_better and not is_lower_better and uses_low_gt0:
            # Gate checks ci95_low > 0 on a higher-is-better metric, so diff = ours - baseline > 0.
            if arg1_is_primary and not arg0_is_primary:
                errors.append(
                    f"Line {lineno}: inverted paired CI direction on `{ci_label}`: higher-is-better metric "
                    f"gated with `ci95_low > 0` requires `paired_ci(ours, baseline)`, got `({arg0_s}, {arg1_s})`"
                )

    for sub in ast.walk(fn_node):
        if isinstance(sub, ast.Dict):
            for k_node, v_node in zip(sub.keys, sub.values):
                if (
                    isinstance(k_node, ast.Constant)
                    and isinstance(k_node.value, str)
                    and isinstance(v_node, ast.Call)
                    and getattr(v_node.func, "id", "") in ("paired_ci95", "paired_diff_stats_df2")
                ):
                    _check_paired_call(k_node.value, v_node, v_node.lineno)
        if (
            isinstance(sub, ast.Assign)
            and len(sub.targets) == 1
            and isinstance(sub.targets[0], ast.Name)
            and isinstance(sub.value, ast.Call)
            and getattr(sub.value.func, "id", "") in ("paired_ci95", "paired_diff_stats_df2")
        ):
            _check_paired_call(sub.targets[0].id, sub.value, sub.lineno)

    return errors


def _recompute_runner_gates(runner_path: Path, smoke_data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Dynamically executes `evaluate_*_gates` from `runner_path` and checks it against `smoke_data['gates']`."""
    errors: List[str] = []
    try:
        import math as _math
        import numpy as _np

        src = runner_path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(runner_path))
        keep_nodes: List[ast.stmt] = []
        eval_fn_name: Optional[str] = None
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                val_node = node.value
                if val_node is not None and not any(isinstance(sub, ast.Call) for sub in ast.walk(val_node)):
                    keep_nodes.append(node)
            elif isinstance(node, ast.FunctionDef):
                if (
                    node.name.startswith("evaluate_")
                    and node.name.endswith("_gates")
                    or node.name in ("paired_ci95", "paired_diff_stats_df2", "summary_stats_df2")
                ):
                    keep_nodes.append(node)
                    if node.name.startswith("evaluate_") and node.name.endswith("_gates"):
                        eval_fn_name = node.name

        if eval_fn_name is None:
            return None, [f"No evaluate_*_gates function found in {runner_path}"]

        mod_ast = ast.Module(body=keep_nodes, type_ignores=[])
        ast.fix_missing_locations(mod_ast)
        ns: Dict[str, Any] = {
            "math": _math,
            "np": _np,
            "Any": Any,
            "Dict": Dict,
            "List": List,
            "Optional": Optional,
            "Set": Set,
            "Tuple": Tuple,
        }
        exec(compile(mod_ast, str(runner_path), "exec"), ns)
        eval_fn = ns[eval_fn_name]
        recomputed = eval_fn(smoke_data)
        stored = smoke_data.get("gates", {})
        if recomputed != stored:
            errors.append(
                f"Recomputed `evaluate_*_gates(smoke_json)` does not match stored `smoke_json['gates']` "
                f"(runner code or gate logic was modified after smoke JSON was generated)"
            )
        return recomputed, errors
    except Exception as exc:
        return None, [f"Failed to execute `evaluate_*_gates` from {runner_path.name} on smoke JSON: {exc}"]


def check_smoke_json(
    smoke_path: Path, prereg_path: Path, runner_path: Path, scorer_path: Path
) -> List[str]:
    errors: List[str] = []
    if not smoke_path.exists():
        errors.append(f"Smoke JSON does not exist: {smoke_path}")
        return errors

    data = json.loads(smoke_path.read_text(encoding="utf-8"))
    if data.get("smoke_cpu") is not True:
        errors.append("Smoke JSON must have `smoke_cpu: true`")

    expected_r_sha = sha256_file(runner_path)
    expected_p_sha = sha256_file(prereg_path)
    if data.get("runner_sha256") != expected_r_sha:
        errors.append(f"runner_sha256 mismatch: JSON={data.get('runner_sha256')} vs disk={expected_r_sha}")
    if data.get("prereg_sha256") != expected_p_sha:
        errors.append(f"prereg_sha256 mismatch: JSON={data.get('prereg_sha256')} vs disk={expected_p_sha}")

    # Check split and RNG seed disjointness (Rule 9.8 / 9.9, catches M5_smoke_seeds_equal_full)
    if "effective_rng_seeds" in data and "seeds" in data:
        eff_seeds = set(data["effective_rng_seeds"])
        full_seeds = set(data["seeds"])
        if not eff_seeds.isdisjoint(full_seeds):
            errors.append(
                f"Rule 9.9 violation: smoke `effective_rng_seeds` {sorted(eff_seeds)} "
                f"overlap with full `seeds` {sorted(full_seeds)}"
            )

    overlap_items: Dict[str, Any] = {}
    for k, v in data.items():
        if "overlap" in k:
            overlap_items[k] = v
    disj = data.get("disjointness_checks", {})
    if isinstance(disj, dict):
        for k, v in disj.items():
            if "overlap" in k:
                overlap_items[f"disjointness_checks.{k}"] = v
            elif k.startswith("min_completions_") and isinstance(v, (int, float)) and v < 2:
                errors.append(f"Multi-solution violation: disjointness_checks[{k}] = {v} (< 2)")

    if not overlap_items:
        errors.append("Smoke JSON contains no split/seed `*overlap*` checks")
    for k, v in overlap_items.items():
        if v != 0:
            errors.append(f"Non-zero split/seed overlap in `{k}` = {v}")

    # Check Control_Random_Network and Control_Untrained_Init presence in per_seed (Rule 9.1)
    per_seed = data.get("per_seed", {})
    if not isinstance(per_seed, dict) or not per_seed:
        errors.append("Smoke JSON missing non-empty `per_seed` dictionary")
    else:
        for s_key, s_val in per_seed.items():
            if isinstance(s_val, dict):
                for dom_k, dom_v in s_val.items():
                    if isinstance(dom_v, dict):
                        # Either dom_v is a dict of arms (e.g. sudoku_extreme, nqueens) or a dict of games -> arms (d1_games, d2_games)
                        if "Control_Random_Network" in dom_v or "Control_Untrained_Init" in dom_v or "Control_Uniform" in dom_v:
                            for req_c in ("Control_Random_Network", "Control_Untrained_Init"):
                                if req_c not in dom_v:
                                    errors.append(f"Rule 9.1: missing `{req_c}` in per_seed[{s_key}][{dom_k}]")
                        else:
                            for g_k, g_v in dom_v.items():
                                if isinstance(g_v, dict):
                                    for req_c in ("Control_Random_Network", "Control_Untrained_Init"):
                                        if req_c not in g_v:
                                            errors.append(
                                                f"Rule 9.1: missing `{req_c}` in per_seed[{s_key}][{dom_k}][{g_k}]"
                                            )

    # Check smoke gates have verdict == SMOKE_NO_VERDICT (do NOT require pass == True on smoke runs!)
    gates = data.get("gates", {})
    if not gates:
        errors.append("No `gates` dictionary in smoke JSON")
    for gid, gval in gates.items():
        if gval.get("verdict") != "SMOKE_NO_VERDICT":
            errors.append(f"Smoke gate `{gid}` must have verdict == 'SMOKE_NO_VERDICT', got {gval.get('verdict')}")
        if not isinstance(gval.get("pass"), bool):
            errors.append(f"Smoke gate `{gid}` must have boolean `pass` field, got {gval.get('pass')}")

    # Recompute evaluate_*_gates(data) and verify it matches stored `data["gates"]`
    _, recompute_errs = _recompute_runner_gates(runner_path, data)
    errors.extend(recompute_errs)

    # Check scorer refuses smoke output without --allow-smoke and accepts with --allow-smoke
    res_refuse = subprocess.run(
        [sys.executable, str(scorer_path), "--json", str(smoke_path)],
        capture_output=True,
        text=True,
    )
    if res_refuse.returncode == 0:
        errors.append(f"Scorer {scorer_path.name} failed to refuse smoke_cpu=true JSON without --allow-smoke")

    res_allow = subprocess.run(
        [sys.executable, str(scorer_path), "--json", str(smoke_path), "--allow-smoke"],
        capture_output=True,
        text=True,
    )
    if res_allow.returncode != 0:
        errors.append(f"Scorer {scorer_path.name} failed with --allow-smoke:\n{res_allow.stderr}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="PROTOCOL.md §10.2 Preflight Verifier")
    parser.add_argument("--prereg", type=str, required=True, help="Path to PREREG-*.md")
    parser.add_argument("--runner", type=str, required=True, help="Path to benchmarks/run_r*.py")
    parser.add_argument("--scorer", type=str, required=True, help="Path to scripts/score_r*.py")
    parser.add_argument("--smoke-json", type=str, required=True, help="Path to smoke JSON output")
    args = parser.parse_args()

    prereg_path = Path(args.prereg).resolve()
    runner_path = Path(args.runner).resolve()
    scorer_path = Path(args.scorer).resolve()
    smoke_path = Path(args.smoke_json).resolve()

    errors: List[str] = []
    errors.extend(check_gate_text_table_and_prose(prereg_path, runner_path))
    errors.extend(check_bar_history(prereg_path, runner_path))
    errors.extend(check_runner_ast_integrity(runner_path))
    errors.extend(check_smoke_json(smoke_path, prereg_path, runner_path, scorer_path))

    if errors:
        print(f"[PREFLIGHT FAIL] {len(errors)} issue(s) found:", file=sys.stderr)
        for idx, err in enumerate(errors, 1):
            print(f"  {idx}. {err}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[PREFLIGHT PASS] {prereg_path.name} + {runner_path.name} verified "
        f"(0 AST/prose/bar-history/control/CI/counter/split/scorer issues)."
    )


if __name__ == "__main__":
    main()
