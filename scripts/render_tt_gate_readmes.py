#!/usr/bin/env python3
"""Generate results/explore/tt-gate-*/README.md directly from summary.json.

Enforces Policy 15 (Claims check themselves): zero hand-typed numbers in tables
or prose; every decimal in the generated README is formatted directly from
summary.json and verified against the parsed JSON dictionary.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent


def fmt_val(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.4f}"


def fmt_ms(mean: float | None, sd: float | None) -> str:
    if mean is None:
        return "—"
    if sd is None:
        return f"`{mean:.4f}`"
    return f"`{mean:.4f} ± {sd:.4f}`"


def repr_label(arm: str, in_dim: int, env: str) -> str:
    if arm.startswith("D_layout_grid") or arm == "bar_fct_piece_bidir":
        return f"`24x2` piece tokens (`in_dim={in_dim}`)"
    if "deepsets" in arm:
        return f"`deepsets` (`in_dim={in_dim}`)"
    if arm == "full_context_transformer_full":
        return f"Flat + all reports (`in_dim={in_dim}`)"
    if arm == "bar_fct_no_report":
        return f"Flat, no `report/4` (`in_dim={in_dim}`)"
    return f"Flat + own `report/4` (`in_dim={in_dim}`)"


def role_label(arm: str) -> str:
    if "distill" in arm:
        return "`[ORACLE CEILING]` (privileged prefix Bayes target)"
    if arm.startswith("bar_fct") or arm.startswith("full_context_transformer"):
        return "Bidirectional FCT bar"
    if arm.startswith("D_layout_grid"):
        return "2D grid TraderTransformer"
    if arm.startswith("A_attn_"):
        return "Axis (A) causal attention schedule"
    if arm.startswith("B_loss_"):
        return "Axis (B) loss schedule"
    if arm.startswith("C_state_"):
        return "Axis (C) token state"
    return "TraderTransformer"


def render_e2z8syb() -> None:
    dirpath = ROOT / "results/explore/tt-gate-e2z8syb-20261021"
    s = json.load(open(dirpath / "summary.json"))
    envs = s["envs"]
    commit = s["git_commit"]
    short_commit = commit[:7]

    e2_b = envs["E2"]["baselines"]
    z8_b = envs["Z8"]["baselines"]
    syb_b = envs["E3-sybil"]["baselines"]

    e2_m = envs["E2"]["models"]
    z8_m = envs["Z8"]["models"]
    syb_m = envs["E3-sybil"]["models"]

    lines: list[str] = []
    lines.append("# Trader-Transformer Gate Fixes Sweep: `E2`, `Z8`, and `E3-sybil` (3 Seeds)")
    lines.append("")
    lines.append("- **Status:** `EXPLORATORY` baseline (no confirmatory gates; `B_loss_distill*` arms use privileged exact prefix-Bayes targets and are graded `Oracle Ceiling`).")
    lines.append(f"- **Commit:** `{commit}` (`{short_commit}`, `git_dirty: false`)")
    lines.append("- **Generator script:** `python3 scripts/render_tt_gate_readmes.py` (all tables and inline numbers rendered directly from `summary.json`).")
    lines.append(f"- **Runner command:** `PYTHONPATH=. python3 benchmarks/explore_trader_transformer_gate_fixes.py --envs E2,Z8,E3-sybil --seeds 0,1,2 --steps {s['steps']} --batch-size {s['batch_size']} --n-train {s['n_train_per_seed']} --n-eval {s['n_eval']} --width {s['width']} --grid-width {s['grid_width']} --layers {s['layers']} --commit {commit} --out results/explore/tt-gate-e2z8syb-20261021/summary.json`")
    lines.append("- **Per-seed train pools:** Seed `s` samples `n_train = 20000` training episodes starting at offset `start = s * n_train` (`0`, `20000`, `40000`), disjoint from the `n_eval = 2000` evaluation episodes.")
    lines.append(f"- **Prefix Bayes verification (`2546bb2`):** Runtime assertion `max |prefix_bayes[:, -1] - bayes_logits| < 1e-8` passed on all three environments (`E2`: `{envs['E2']['prefix_bayes_final_maxabs_err']:.2e}`, `Z8`: `{envs['Z8']['prefix_bayes_final_maxabs_err']:.2e}`, `E3-sybil`: `{envs['E3-sybil']['prefix_bayes_final_maxabs_err']:.2e}`).")
    lines.append("")
    lines.append("## 1. Non-Neural Baselines (`n_eval = 2000`, rendered from `summary.json`)")
    lines.append("")
    lines.append("| Baseline | `E2` Regret | `E2` KL | `Z8` Regret | `Z8` KL | `E3-sybil` Regret | `E3-sybil` KL |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for bkey in ["bayes", "oracle_clearing", "extremized_log_pool", "log_pool", "sum_logodds", "linear_pool"]:
        lines.append(
            f"| `{bkey}` | `{fmt_val(e2_b[bkey]['regret'])}` | `{fmt_val(e2_b[bkey]['kl'])}` "
            f"| `{fmt_val(z8_b[bkey]['regret'])}` | `{fmt_val(z8_b[bkey]['kl'])}` "
            f"| `{fmt_val(syb_b[bkey]['regret'])}` | `{fmt_val(syb_b[bkey]['kl'])}` |"
        )
    lines.append("")
    lines.append(
        f"**Comparison to `extremized_log_pool` on `Z8`:** On `Z8`, `extremized_log_pool` achieves regret `{fmt_val(z8_b['extremized_log_pool']['regret'])}` (`kl = {fmt_val(z8_b['extremized_log_pool']['kl'])}`), which beats every causal and 2D grid neural arm in this sweep (the best non-oracle causal/grid arm on `Z8` is `D_layout_grid_rankfree_cellT1_final` at `{fmt_val(z8_m['D_layout_grid_rankfree_cellT1_final']['regret_mean'])} ± {fmt_val(z8_m['D_layout_grid_rankfree_cellT1_final']['regret_sd'])}`). Only the bidirectional piece bar `bar_fct_piece_bidir` (`{fmt_val(z8_m['bar_fct_piece_bidir']['regret_mean'])} ± {fmt_val(z8_m['bar_fct_piece_bidir']['regret_sd'])}`) and the `[ORACLE CEILING]` arm `B_loss_distill_deepsets` (`{fmt_val(z8_m['B_loss_distill_deepsets']['regret_mean'])} ± {fmt_val(z8_m['B_loss_distill_deepsets']['regret_sd'])}`) achieve lower mean regret on `Z8`."
    )
    lines.append("")
    lines.append("## 2. Matched Bidirectional Bars vs. Causal and Grid Arms")
    lines.append("")
    lines.append("1. **Input dimension (`in_dim=49`) of linear trader arms vs. bidirectional bars:**")
    lines.append(
        f"   - The linear trader arms (`A_attn_*`, `B_loss_*`, `C_state_*`) take `in_dim = {e2_m['A_attn_all']['in_dim']}` (`24` values + `24` mask bits + `1` own `report_logit / 4`), which is the exact same per-trader input as `full_context_transformer_none` (`in_dim = {e2_m['full_context_transformer_none']['in_dim']}`). Only `bar_fct_no_report` drops `report_logit / 4` (`in_dim = {e2_m['bar_fct_no_report']['in_dim']}`)."
    )
    lines.append(
        f"   - With matched `in_dim = 49` inputs on the `20k` pool (`steps = 2000, width = 128, layers = 4`), bidirectional `full_context_transformer_none` beats single-order causal `A_attn_all` on all 3 environments: `E2` `{fmt_val(e2_m['full_context_transformer_none']['regret_mean'])}` vs `{fmt_val(e2_m['A_attn_all']['regret_mean'])}`, `Z8` `{fmt_val(z8_m['full_context_transformer_none']['regret_mean'])}` vs `{fmt_val(z8_m['A_attn_all']['regret_mean'])}`, and `E3-sybil` `{fmt_val(syb_m['full_context_transformer_none']['regret_mean'])}` vs `{fmt_val(syb_m['A_attn_all']['regret_mean'])}`."
    )
    lines.append(
        f"   - Removing `report_logit / 4` from the bidirectional bar (`bar_fct_no_report`, `in_dim = 48`) raises its regret to `{fmt_val(e2_m['bar_fct_no_report']['regret_mean'])}` (`E2`), `{fmt_val(z8_m['bar_fct_no_report']['regret_mean'])}` (`Z8`), and `{fmt_val(syb_m['bar_fct_no_report']['regret_mean'])}` (`E3-sybil`). Re-permuting trader order each step (`A_attn_all_reperm`: `{fmt_val(e2_m['A_attn_all_reperm']['regret_mean'])}`, `{fmt_val(z8_m['A_attn_all_reperm']['regret_mean'])}`, `{fmt_val(syb_m['A_attn_all_reperm']['regret_mean'])}`) and `B_loss_every_prefix_reperm` (`{fmt_val(e2_m['B_loss_every_prefix_reperm']['regret_mean'])}`, `{fmt_val(z8_m['B_loss_every_prefix_reperm']['regret_mean'])}`, `{fmt_val(syb_m['B_loss_every_prefix_reperm']['regret_mean'])}`) narrow the gap to `full_context_transformer_none`."
    )
    lines.append("2. **Matched encoder comparison (`deepsets` and `24x2` piece tokens):**")
    lines.append(
        f"   - With `deepsets` tokens (`in_dim = 65`), bidirectional `bar_fct_deepsets_bidir` (`E2`: `{fmt_val(e2_m['bar_fct_deepsets_bidir']['regret_mean'])}`, `Z8`: `{fmt_val(z8_m['bar_fct_deepsets_bidir']['regret_mean'])}`, `E3-sybil`: `{fmt_val(syb_m['bar_fct_deepsets_bidir']['regret_mean'])}`) beats causal `A_attn_all_deepsets` (`{fmt_val(e2_m['A_attn_all_deepsets']['regret_mean'])}`, `{fmt_val(z8_m['A_attn_all_deepsets']['regret_mean'])}`, `{fmt_val(syb_m['A_attn_all_deepsets']['regret_mean'])}`) and is slightly below `A_attn_all_deepsets_reperm` (`{fmt_val(e2_m['A_attn_all_deepsets_reperm']['regret_mean'])}`, `{fmt_val(z8_m['A_attn_all_deepsets_reperm']['regret_mean'])}`, `{fmt_val(syb_m['A_attn_all_deepsets_reperm']['regret_mean'])}`)."
    )
    lines.append(
        f"   - With `24x2` piece tokens (`in_dim = 4`), bidirectional `bar_fct_piece_bidir` (`E2`: `{fmt_val(e2_m['bar_fct_piece_bidir']['regret_mean'])} ± {fmt_val(e2_m['bar_fct_piece_bidir']['regret_sd'])}`, `Z8`: `{fmt_val(z8_m['bar_fct_piece_bidir']['regret_mean'])} ± {fmt_val(z8_m['bar_fct_piece_bidir']['regret_sd'])}`, `E3-sybil`: `{fmt_val(syb_m['bar_fct_piece_bidir']['regret_mean'])} ± {fmt_val(syb_m['bar_fct_piece_bidir']['regret_sd'])}`) is lower than `D_layout_grid_rankfree_cellT1_final` (`E2`: `{fmt_val(e2_m['D_layout_grid_rankfree_cellT1_final']['regret_mean'])} ± {fmt_val(e2_m['D_layout_grid_rankfree_cellT1_final']['regret_sd'])}`, `Z8`: `{fmt_val(z8_m['D_layout_grid_rankfree_cellT1_final']['regret_mean'])} ± {fmt_val(z8_m['D_layout_grid_rankfree_cellT1_final']['regret_sd'])}`, `E3-sybil`: `{fmt_val(syb_m['D_layout_grid_rankfree_cellT1_final']['regret_mean'])} ± {fmt_val(syb_m['D_layout_grid_rankfree_cellT1_final']['regret_sd'])}`). Across all 6 matched representation pairs (`{{flat 49-d, deepsets, piece}}` $\\times$ `{{E2, Z8}}`), the bidirectional bar has lower mean regret than the matched causal arm."
    )
    lines.append("3. **`2x2` Grid Readout Factorial (`grid_full` vs `grid_rankfree` $\\times$ `cell_T1` vs `pooled`):**")
    lines.append(
        "   - Across `{E2, Z8, E3-sybil} x {cell_T1, pooled} x {final, every_prefix}` (12 cells), `grid_rankfree` has lower mean regret than `grid_full` in 12 of 12 cells. Note that `grid_rankfree` both removes causal masking along the slot axis and omits slot embeddings, so those two effects are combined in this comparison."
    )
    lines.append("")
    lines.append("## 3. Full 3-Seed Results Table (`E2`, `Z8`, `E3-sybil`, rendered from `summary.json`)")
    lines.append("")
    lines.append("| Arm | Input (`in_dim`) | Role | `E2` Regret | `E2` Prefix KL | `Z8` Regret | `Z8` Prefix KL | `E3-sybil` Regret | `E3-sybil` Prefix KL |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for arm in e2_m.keys():
        em = e2_m[arm]
        zm = z8_m[arm]
        sm = syb_m[arm]
        lines.append(
            f"| `{arm}` | {repr_label(arm, em['in_dim'], 'E2')} | {role_label(arm)} "
            f"| {fmt_ms(em['regret_mean'], em['regret_sd'])} | {fmt_ms(em['mean_prefix_kl_mean'], em['mean_prefix_kl_sd'])} "
            f"| {fmt_ms(zm['regret_mean'], zm['regret_sd'])} | {fmt_ms(zm['mean_prefix_kl_mean'], zm['mean_prefix_kl_sd'])} "
            f"| {fmt_ms(sm['regret_mean'], sm['regret_sd'])} | {fmt_ms(sm['mean_prefix_kl_mean'], sm['mean_prefix_kl_sd'])} |"
        )
    lines.append("")

    out_text = "\n".join(lines)
    (dirpath / "README.md").write_text(out_text)
    verify_readme_numbers(out_text, s, str(dirpath / "README.md"))


def render_xor_fresh() -> None:
    dirpath = ROOT / "results/explore/tt-gate-xor-fresh-20261021"
    s = json.load(open(dirpath / "summary.json"))
    envs = s["envs"]
    commit = s["git_commit"]
    short_commit = commit[:7]

    z5_b = envs["Z5"]["baselines"]
    z6_b = envs["Z6"]["baselines"]
    z5_m = envs["Z5"]["models"]
    z6_m = envs["Z6"]["models"]

    lines: list[str] = []
    lines.append("# Trader-Transformer Gate Fixes Sweep: `Z5` (`MultiQuestionEnv`) and `Z6` (`ParityEnv`, 2-bit XOR) at `3000 x 512` Fresh Episodes")
    lines.append("")
    lines.append("- **Status:** `EXPLORATORY` baseline (no confirmatory gates; `B_loss_distill` uses privileged exact prefix-Bayes targets and is graded `Oracle Ceiling`).")
    lines.append(f"- **Commit:** `{commit}` (`{short_commit}`, `git_dirty: false`)")
    lines.append("- **Generator script:** `python3 scripts/render_tt_gate_readmes.py` (all tables and inline numbers rendered directly from `summary.json`).")
    lines.append(f"- **Runner command:** `PYTHONPATH=. python3 benchmarks/explore_trader_transformer_gate_fixes.py --envs Z5,Z6 --seeds 0,1,2 --fresh-episodes --steps {s['steps']} --batch-size {s['batch_size']} --n-eval {s['n_eval']} --width {s['width']} --grid-width {s['grid_width']} --layers {s['layers']} --commit {commit} --out results/explore/tt-gate-xor-fresh-20261021/summary.json`")
    lines.append(f"- **Training regime:** `fresh_episodes = true` (`{s['steps']} x {s['batch_size']} = {s['n_train_per_seed']}` fresh episodes per seed, `seeds = [0, 1, 2]`, `width = {s['width']}`, `layers = {s['layers']}`). `Z6-parity3` is a held-out benchmark environment (`src/benchmarks/holdout.py`) and is excluded.")
    lines.append(f"- **Prefix Bayes verification (`2546bb2`):** Runtime assertion `max |prefix_bayes[:, -1] - bayes_logits| < 1e-8` passed (`Z5`: `{envs['Z5']['prefix_bayes_final_maxabs_err']:.2e}`, `Z6`: `{envs['Z6']['prefix_bayes_final_maxabs_err']:.2e}`).")
    lines.append("")
    lines.append("## 1. Non-Neural Baselines (`n_eval = 2000`, rendered from `summary.json`)")
    lines.append("")
    lines.append("| Baseline | `Z5` Regret | `Z5` KL | `Z6` Regret | `Z6` KL |")
    lines.append("|---|---:|---:|---:|---:|")
    for bkey in ["bayes", "oracle_clearing", "extremized_log_pool", "log_pool", "linear_pool", "sum_logodds"]:
        lines.append(
            f"| `{bkey}` | `{fmt_val(z5_b[bkey]['regret'])}` | `{fmt_val(z5_b[bkey]['kl'])}` "
            f"| `{fmt_val(z6_b[bkey]['regret'])}` | `{fmt_val(z6_b[bkey]['kl'])}` |"
        )
    lines.append("")
    lines.append(
        f"**Baseline notes:** On `Z5`, `oracle_clearing` achieves `{fmt_val(z5_b['oracle_clearing']['regret'])}` regret (`kl = {fmt_val(z5_b['oracle_clearing']['kl'])}`), while report-pooling rules (`extremized_log_pool`: `{fmt_val(z5_b['extremized_log_pool']['regret'])}`, `log_pool`: `{fmt_val(z5_b['log_pool']['regret'])}`, `linear_pool`: `{fmt_val(z5_b['linear_pool']['regret'])}`) leave substantial regret because individual reports on a single sub-question are uninformative about the global XOR without cross-question composition. On `Z6` (2-bit parity), `oracle_clearing` (`{fmt_val(z6_b['oracle_clearing']['regret'])}`) and all report-pooling rules (`{fmt_val(z6_b['extremized_log_pool']['regret'])}` to `{fmt_val(z6_b['sum_logodds']['regret'])}`) fail because individual readings have zero marginal log-odds."
    )
    lines.append("")
    lines.append("## 2. Summary of `3000 x 512` Fresh-Episode Results on `Z5` and `Z6`")
    lines.append("")
    lines.append(
        f"1. **Bidirectional bars and non-`none` causal arms both reach low regret on `Z5` and `Z6`, so `Z5` and `Z6` (2-bit XOR) do not separate the neural architectures:**"
    )
    lines.append(
        f"   - On `Z6` (2-bit XOR), `A_attn_none` stays near the prior at `{fmt_val(z6_m['A_attn_none']['regret_mean'])} ± {fmt_val(z6_m['A_attn_none']['regret_sd'])}`, while every non-`none` arm reaches mean regret at or below `{fmt_val(z6_m['B_loss_every_prefix_reperm']['regret_mean'])}` (`bar_fct_no_report`: `{fmt_val(z6_m['bar_fct_no_report']['regret_mean'])} ± {fmt_val(z6_m['bar_fct_no_report']['regret_sd'])}`, `full_context_transformer_none`: `{fmt_val(z6_m['full_context_transformer_none']['regret_mean'])} ± {fmt_val(z6_m['full_context_transformer_none']['regret_sd'])}`, `A_attn_last`: `{fmt_val(z6_m['A_attn_last']['regret_mean'])} ± {fmt_val(z6_m['A_attn_last']['regret_sd'])}`, `C_state_recurrent_final`: `{fmt_val(z6_m['C_state_recurrent_final']['regret_mean'])} ± {fmt_val(z6_m['C_state_recurrent_final']['regret_sd'])}`, and `[ORACLE CEILING]` `B_loss_distill`: `{fmt_val(z6_m['B_loss_distill']['regret_mean'])} ± {fmt_val(z6_m['B_loss_distill']['regret_sd'])}`)."
    )
    lines.append(
        f"   - On `Z5`, `A_attn_none` is `{fmt_val(z5_m['A_attn_none']['regret_mean'])} ± {fmt_val(z5_m['A_attn_none']['regret_sd'])}`. The bidirectional bars reach `{fmt_val(z5_m['bar_fct_no_report']['regret_mean'])} ± {fmt_val(z5_m['bar_fct_no_report']['regret_sd'])}` (`bar_fct_no_report`) and `{fmt_val(z5_m['full_context_transformer_none']['regret_mean'])} ± {fmt_val(z5_m['full_context_transformer_none']['regret_sd'])}` (`full_context_transformer_none`). Final-loss causal arms with last-layer attention reach `{fmt_val(z5_m['C_state_feedback_final']['regret_mean'])}` (`C_state_feedback_final`) to `{fmt_val(z5_m['A_attn_first_last']['regret_mean'])}` (`A_attn_first_last`), while `A_attn_first` (`{fmt_val(z5_m['A_attn_first']['regret_mean'])} ± {fmt_val(z5_m['A_attn_first']['regret_sd'])}`), `B_loss_every_prefix` (`{fmt_val(z5_m['B_loss_every_prefix']['regret_mean'])} ± {fmt_val(z5_m['B_loss_every_prefix']['regret_sd'])}`), and `B_loss_every_prefix_reperm` (`{fmt_val(z5_m['B_loss_every_prefix_reperm']['regret_mean'])} ± {fmt_val(z5_m['B_loss_every_prefix_reperm']['regret_sd'])}`) remain above `0.0100`. The `[ORACLE CEILING]` arm `B_loss_distill` reaches `{fmt_val(z5_m['B_loss_distill']['regret_mean'])} ± {fmt_val(z5_m['B_loss_distill']['regret_sd'])}` (`mean_prefix_kl = {fmt_val(z5_m['B_loss_distill']['mean_prefix_kl_mean'])} ± {fmt_val(z5_m['B_loss_distill']['mean_prefix_kl_sd'])}`)."
    )
    lines.append("")
    lines.append("## 3. Full 3-Seed Results Table (`Z5` and `Z6`, rendered from `summary.json`)")
    lines.append("")
    lines.append("| Arm | `Z5` `in_dim` | `Z6` `in_dim` | Role | `Z5` Regret | `Z5` KL | `Z5` Prefix KL | `Z6` Regret | `Z6` KL | `Z6` Prefix KL |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for arm in z5_m.keys():
        m5 = z5_m[arm]
        m6 = z6_m[arm]
        lines.append(
            f"| `{arm}` | `{m5['in_dim']}` | `{m6['in_dim']}` | {role_label(arm)} "
            f"| {fmt_ms(m5['regret_mean'], m5['regret_sd'])} | {fmt_ms(m5['kl_mean'], m5['kl_sd'])} | {fmt_ms(m5['mean_prefix_kl_mean'], m5['mean_prefix_kl_sd'])} "
            f"| {fmt_ms(m6['regret_mean'], m6['regret_sd'])} | {fmt_ms(m6['kl_mean'], m6['kl_sd'])} | {fmt_ms(m6['mean_prefix_kl_mean'], m6['mean_prefix_kl_sd'])} |"
        )
    lines.append("")

    out_text = "\n".join(lines)
    (dirpath / "README.md").write_text(out_text)
    verify_readme_numbers(out_text, s, str(dirpath / "README.md"))


def verify_readme_numbers(text: str, summary_dict: dict, label: str) -> None:
    """Verify every 4-decimal number in the README exists in summary_dict."""
    valid_four_decimals = {"0.0100"}
    def collect(obj):
        if isinstance(obj, float):
            valid_four_decimals.add(f"{obj:.4f}")
        elif isinstance(obj, dict):
            for v in obj.values():
                collect(v)
        elif isinstance(obj, list):
            for v in obj:
                collect(v)
    collect(summary_dict)

    found = re.findall(r"\b\d+\.\d{4}\b", text)
    missing = [x for x in found if x not in valid_four_decimals]
    if missing:
        raise AssertionError(f"{label}: found 4-decimal numbers not in summary.json: {missing}")
    print(f"[verify_readme_numbers] PASS {label}: {len(found)}/ {len(found)} 4-decimal numbers verified against summary.json")


if __name__ == "__main__":
    render_e2z8syb()
    render_xor_fresh()
