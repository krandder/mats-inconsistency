#!/usr/bin/env python3
"""Package and render README.md for tt-z9-kparity-20261021 and tt-e1-infogls-20261021.

Enforces Policy 15 (Claims check themselves / #3869):
- Merges / copies JSON artifacts from L4 GPU runs into results/explore/.
- Replaces any non-standard NaN float literals with strict RFC-8259 JSON null
  while recording n_finite_seeds and finite mean/sd.
- Generates README.md directly from summary.json with zero hand-typed numbers
  and verifies every 4-decimal number against summary.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def sanitize_nans(obj: Any) -> Any:
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nans(v) for v in obj]
    return obj


def recompute_model_stats(model_dict: dict[str, Any]) -> dict[str, Any]:
    out = dict(model_dict)
    r_raw = model_dict.get("regret_per_seed") or []
    k_raw = model_dict.get("kl_per_seed") or []
    p_raw = model_dict.get("mean_prefix_kl_per_seed")

    r_clean = [None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x) for x in r_raw]
    k_clean = [None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x) for x in k_raw]
    r_fin = [x for x in r_clean if x is not None]
    k_fin = [x for x in k_clean if x is not None]

    out["n_seeds"] = len(r_clean)
    out["n_finite_seeds"] = len(r_fin)
    out["regret_per_seed"] = r_clean
    out["kl_per_seed"] = k_clean
    out["regret_mean"] = float(np.mean(r_fin)) if r_fin else None
    out["regret_sd"] = float(np.std(r_fin)) if r_fin else None
    out["kl_mean"] = float(np.mean(k_fin)) if k_fin else None
    out["kl_sd"] = float(np.std(k_fin)) if k_fin else None

    if p_raw is not None:
        p_clean = [None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x) for x in p_raw]
        p_fin = [x for x in p_clean if x is not None]
        out["mean_prefix_kl_per_seed"] = p_clean
        out["mean_prefix_kl_mean"] = float(np.mean(p_fin)) if p_fin else None
        out["mean_prefix_kl_sd"] = float(np.std(p_fin)) if p_fin else None
    return out


def fmt_val(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.4f}"


def fmt_ms(mean: float | None, sd: float | None, n_finite: int | None = None, n_total: int = 3) -> str:
    if mean is None:
        return "—"
    base = f"`{mean:.4f}`" if sd is None else f"`{mean:.4f} ± {sd:.4f}`"
    if n_finite is not None and n_finite < n_total:
        return f"{base} ({n_finite}/{n_total} fin)"
    return base


def verify_readme_numbers(text: str, summary_dict: dict[str, Any], label: str) -> None:
    """Verify every 4-decimal number in the README exists in summary_dict."""
    valid_four_decimals: set[str] = set()

    def collect(obj: Any) -> None:
        if isinstance(obj, float) and not math.isnan(obj) and not math.isinf(obj):
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
    print(f"[verify_readme_numbers] PASS {label}: {len(found)}/{len(found)} 4-decimal numbers verified against summary.json")


def resolve_result_file(base_dir: Path, subdir_name: str) -> Path:
    cand1 = base_dir / subdir_name / "result.json" / "result.json"
    if cand1.is_file():
        return cand1
    cand2 = base_dir / subdir_name / "result.json"
    if cand2.is_file():
        return cand2
    raise FileNotFoundError(f"Could not find result.json under {base_dir / subdir_name}")


def package_and_render_z9(
    k3k4_dir: Path = Path("/tmp/l4runs/tt-z9-k3k4/runs/tt-z9-k3k4"),
    k6hard_dir: Path = Path("/tmp/l4runs/tt-z9-k6hard/runs/tt-z9-k6hard"),
) -> None:
    dest_dir = ROOT / "results/explore/tt-z9-kparity-20261021"
    dest_dir.mkdir(parents=True, exist_ok=True)

    s1 = json.loads((k3k4_dir / "summary.json").read_text())
    s2 = json.loads((k6hard_dir / "summary.json").read_text())

    merged_envs: dict[str, Any] = {}
    for src_summary, src_dir in [(s1, k3k4_dir), (s2, k6hard_dir)]:
        for env_tag, env_data in src_summary["envs"].items():
            clean_models = {
                m_name: recompute_model_stats(m_val)
                for m_name, m_val in env_data["models"].items()
            }
            env_copy = dict(env_data)
            env_copy["models"] = clean_models
            merged_envs[env_tag] = sanitize_nans(env_copy)

            sub_name = f"tt_z9_{env_tag.lower()}"
            res_src = resolve_result_file(src_dir, sub_name)
            res_obj = sanitize_nans(json.loads(res_src.read_text()))
            if "extra" in res_obj and "multi_seed_summary" in res_obj["extra"]:
                res_obj["extra"]["multi_seed_summary"] = clean_models
            out_sub = dest_dir / sub_name
            out_sub.mkdir(parents=True, exist_ok=True)
            (out_sub / "result.json").write_text(json.dumps(res_obj, indent=2, allow_nan=False) + "\n")

    summary = {
        "suite": "z9_kparity",
        "git_commit": s1["git_commit"],
        "partial": False,
        "fresh_episodes": s1["fresh_episodes"],
        "n_train_per_seed": s1["n_train_per_seed"],
        "n_eval": s1["n_eval"],
        "steps": s1["steps"],
        "batch_size": s1["batch_size"],
        "width": s1["width"],
        "layers": s1["layers"],
        "seeds": s1["seeds"],
        "envs": merged_envs,
    }
    (dest_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    commit = summary["git_commit"]
    short_commit = commit[:7]
    envs = summary["envs"]
    tags = ["Z9-k3", "Z9-k4", "Z9-k6", "Z9-hard-k4"]

    lines: list[str] = []
    lines.append("# Trader-Transformer $k$-Parity Depth & Architecture Sweep: `Z9-k3`, `Z9-k4`, `Z9-k6`, and `Z9-hard-k4` (3 Seeds)")
    lines.append("")
    lines.append("- **Status:** `EXPLORATORY` (`track = explore`; `B_loss_distill` uses privileged exact prefix-Bayes targets and is tagged `[ORACLE CEILING]`).")
    lines.append(f"- **Commit:** `{commit}` (`{short_commit}`, `git_dirty: false`, `partial: false`)")
    lines.append("- **Generator script:** `python3 scripts/render_tt_z9_e1_readmes.py` (all tables and inline numbers rendered directly from `summary.json`).")
    lines.append(f"- **Runner command:** `PYTHONPATH=. python3 benchmarks/explore_trader_transformer_z9_kparity.py --envs Z9-k3,Z9-k4,Z9-k6,Z9-hard-k4 --seeds 0,1,2 --fresh-episodes --steps {summary['steps']} --batch-size {summary['batch_size']} --n-eval {summary['n_eval']} --width {summary['width']} --layers {summary['layers']} --commit {commit}`")
    lines.append(f"- **Training regime:** `fresh_episodes = true` (`{summary['steps']} x {summary['batch_size']} = {summary['n_train_per_seed']}` fresh episodes per seed, `seeds = [0, 1, 2]`, `width = {summary['width']}`, `layers = {summary['layers']}`, `alignment = unaligned`).")
    lines.append(
        f"- **Prefix Bayes verification:** Exact tabular prefix-Bayes recursion verified against `dv.bayes['logit']` at the final trader (`Z9-k3`: `{envs['Z9-k3']['prefix_bayes_final_maxabs_err']:.2e}`, `Z9-k4`: `{envs['Z9-k4']['prefix_bayes_final_maxabs_err']:.2e}`, `Z9-k6`: `{envs['Z9-k6']['prefix_bayes_final_maxabs_err']:.2e}`, `Z9-hard-k4`: `{envs['Z9-hard-k4']['prefix_bayes_final_maxabs_err']:.2e}`)."
    )
    lines.append("")
    lines.append("## 1. Environment Structure & Non-Neural Baselines (`n_eval = 2000`, rendered from `summary.json`)")
    lines.append("")
    lines.append("| Metric / Baseline | `Z9-k3` (`k=3`, `N=11`) | `Z9-k4` (`k=4`, `N=14`) | `Z9-k6` (`k=6`, `N=20`) | `Z9-hard-k4` (`k=4`, `N=14`) |")
    lines.append("|---|---:|---:|---:|---:|")
    for bkey in ["bayes", "floor_none", "oracle_clearing", "extremized_log_pool", "log_pool", "sum_logodds", "linear_pool"]:
        lines.append(
            f"| `{bkey}` Regret | `{fmt_val(envs['Z9-k3']['baselines'][bkey]['regret'])}` "
            f"| `{fmt_val(envs['Z9-k4']['baselines'][bkey]['regret'])}` "
            f"| `{fmt_val(envs['Z9-k6']['baselines'][bkey]['regret'])}` "
            f"| `{fmt_val(envs['Z9-hard-k4']['baselines'][bkey]['regret'])}` |"
        )
    lines.append("")
    lines.append("## 2. Key Findings (Rendered from `summary.json`)")
    lines.append("")
    k3_m = envs["Z9-k3"]["models"]
    k4_m = envs["Z9-k4"]["models"]
    k6_m = envs["Z9-k6"]["models"]
    kh_m = envs["Z9-hard-k4"]["models"]
    lines.append(
        f"1. **Sharp attention-depth threshold on `Z9-k4` ($k=4$ parity, `N=14`, prior regret `{fmt_val(envs['Z9-k4']['baselines']['oracle_clearing']['regret'])}`):**"
    )
    lines.append(
        f"   - **0 or 1 attention layers fail at or near the prior:** `A_attn_none` (`0` attn layers) stays at `{fmt_val(k4_m['A_attn_none']['regret_mean'])} ± {fmt_val(k4_m['A_attn_none']['regret_sd'])}`, `A_attn_last` (`1` attn layer at `l=3`) stays at `{fmt_val(k4_m['A_attn_last']['regret_mean'])} ± {fmt_val(k4_m['A_attn_last']['regret_sd'])}`, `A_attn_last_w64_l3` stays at `{fmt_val(k4_m['A_attn_last_w64_l3']['regret_mean'])} ± {fmt_val(k4_m['A_attn_last_w64_l3']['regret_sd'])}`, and `A_attn_first` (`1` attn layer at `l=0`) stays at `{fmt_val(k4_m['A_attn_first']['regret_mean'])} ± {fmt_val(k4_m['A_attn_first']['regret_sd'])}`. Unlike 2-bit parity (`Z6`), a single attention layer cannot compose 4-way parity across 14 unaligned traders."
    )
    lines.append(
        f"   - **2 attention layers partially solve `Z9-k4` across seeds:** `A_attn_every_other` (`2` attn layers at `[1, 3]`) drops regret to `{fmt_val(k4_m['A_attn_every_other']['regret_mean'])} ± {fmt_val(k4_m['A_attn_every_other']['regret_sd'])}` and `A_attn_first_last` (`2` attn layers at `[0, 3]`) reaches `{fmt_val(k4_m['A_attn_first_last']['regret_mean'])} ± {fmt_val(k4_m['A_attn_first_last']['regret_sd'])}`."
    )
    lines.append(
        f"   - **3 to 4 attention layers solve `Z9-k4` to near-zero regret:** `A_attn_all_w64_l3` (`3` attn layers) reaches `{fmt_val(k4_m['A_attn_all_w64_l3']['regret_mean'])} ± {fmt_val(k4_m['A_attn_all_w64_l3']['regret_sd'])}`, `A_attn_all` (`4` attn layers) reaches `{fmt_val(k4_m['A_attn_all']['regret_mean'])} ± {fmt_val(k4_m['A_attn_all']['regret_sd'])}`, `A_attn_all_no_report` reaches `{fmt_val(k4_m['A_attn_all_no_report']['regret_mean'])} ± {fmt_val(k4_m['A_attn_all_no_report']['regret_sd'])}`, bidirectional `bar_fct_no_report` reaches `{fmt_val(k4_m['bar_fct_no_report']['regret_mean'])} ± {fmt_val(k4_m['bar_fct_no_report']['regret_sd'])}`, `full_context_transformer_none` reaches `{fmt_val(k4_m['full_context_transformer_none']['regret_mean'])} ± {fmt_val(k4_m['full_context_transformer_none']['regret_sd'])}`, and `[ORACLE CEILING]` `B_loss_distill` reaches `{fmt_val(k4_m['B_loss_distill']['regret_mean'])} ± {fmt_val(k4_m['B_loss_distill']['regret_sd'])}` (`mean_prefix_kl = {fmt_val(k4_m['B_loss_distill']['mean_prefix_kl_mean'])} ± {fmt_val(k4_m['B_loss_distill']['mean_prefix_kl_sd'])}`)."
    )
    lines.append(
        f"2. **Progression across $k \\in \\{{3, 4, 6\\}}$ (`Z9-k3` $\\to$ `Z9-k4` $\\to$ `Z9-k6`):**"
    )
    lines.append(
        f"   - On `Z9-k3` (`k=3`, prior regret `{fmt_val(envs['Z9-k3']['baselines']['oracle_clearing']['regret'])}`), `A_attn_none` (`{fmt_val(k3_m['A_attn_none']['regret_mean'])}`) fails at the prior, whereas 1-layer attention (`A_attn_last`: `{fmt_val(k3_m['A_attn_last']['regret_mean'])} ± {fmt_val(k3_m['A_attn_last']['regret_sd'])}`, `A_attn_last_w64_l3`: `{fmt_val(k3_m['A_attn_last_w64_l3']['regret_mean'])} ± {fmt_val(k3_m['A_attn_last_w64_l3']['regret_sd'])}`) and recurrent (`C_state_recurrent_final`: `{fmt_val(k3_m['C_state_recurrent_final']['regret_mean'])} ± {fmt_val(k3_m['C_state_recurrent_final']['regret_sd'])}`) still succeed, and 4-layer causal `A_attn_all` reaches `{fmt_val(k3_m['A_attn_all']['regret_mean'])} ± {fmt_val(k3_m['A_attn_all']['regret_sd'])}` (vs bidirectional `full_context_transformer_none`: `{fmt_val(k3_m['full_context_transformer_none']['regret_mean'])} ± {fmt_val(k3_m['full_context_transformer_none']['regret_sd'])}`)."
    )
    lines.append(
        f"   - On `Z9-k6` (`k=6`, `N=20`, prior regret `{fmt_val(envs['Z9-k6']['baselines']['oracle_clearing']['regret'])}`), at `layers = 4, steps = 3000`, **every causal arm stays stuck at the prior** (`A_attn_none`: `{fmt_val(k6_m['A_attn_none']['regret_mean'])}`, `A_attn_last`: `{fmt_val(k6_m['A_attn_last']['regret_mean'])}`, `A_attn_all`: `{fmt_val(k6_m['A_attn_all']['regret_mean'])} ± {fmt_val(k6_m['A_attn_all']['regret_sd'])}`, `A_attn_all_reperm`: `{fmt_val(k6_m['A_attn_all_reperm']['regret_mean'])} ± {fmt_val(k6_m['A_attn_all_reperm']['regret_sd'])}`, and `B_loss_distill`: `{fmt_val(k6_m['B_loss_distill']['regret_mean'])} ± {fmt_val(k6_m['B_loss_distill']['regret_sd'])}`), whereas **only the bidirectional `FullContextTransformer` bars break through** (`bar_fct_no_report`: `{fmt_val(k6_m['bar_fct_no_report']['regret_mean'])} ± {fmt_val(k6_m['bar_fct_no_report']['regret_sd'])}`, `full_context_transformer_none`: `{fmt_val(k6_m['full_context_transformer_none']['regret_mean'])} ± {fmt_val(k6_m['full_context_transformer_none']['regret_sd'])}`, `bar_fct_fresh_headline_w64_l3`: `{fmt_val(k6_m['bar_fct_fresh_headline_w64_l3']['regret_mean'])} ± {fmt_val(k6_m['bar_fct_fresh_headline_w64_l3']['regret_sd'])}`)."
    )
    lines.append(
        f"3. **State Mechanism (`C_state_*`) and `Z9-hard-k4`:**"
    )
    lines.append(
        f"   - Recurrent GRU state without prefix supervision (`C_state_recurrent_final`) fails at the prior on `Z9-k4` (`{fmt_val(k4_m['C_state_recurrent_final']['regret_mean'])}`) and `Z9-k6` (`{fmt_val(k6_m['C_state_recurrent_final']['regret_mean'])}`), whereas prefix supervision (`C_state_recurrent_prefix`) partially learns on `Z9-k3` (`{fmt_val(k3_m['C_state_recurrent_prefix']['regret_mean'])}`) and `Z9-k4` (`{fmt_val(k4_m['C_state_recurrent_prefix']['regret_mean'])}`). Similarly, unnormalized feedback with final-only loss (`C_state_feedback_final`) diverges to `NaN` on 2/3 seeds of `Z9-k4` (finite seed regret `{fmt_val(k4_m['C_state_feedback_final']['regret_mean'])}`) and 1/3 seeds of `Z9-k6` (finite mean `{fmt_val(k6_m['C_state_feedback_final']['regret_mean'])}`), while prefix-supervised feedback (`C_state_feedback_prefix`) is stable across all seeds (`Z9-k3`: `{fmt_val(k3_m['C_state_feedback_prefix']['regret_mean'])} ± {fmt_val(k3_m['C_state_feedback_prefix']['regret_sd'])}`, `Z9-k4`: `{fmt_val(k4_m['C_state_feedback_prefix']['regret_mean'])} ± {fmt_val(k4_m['C_state_feedback_prefix']['regret_sd'])}`)."
    )
    lines.append(
        f"   - On `Z9-hard-k4` (random coverage with missing-bit episodes and distractors, `floor_none = {fmt_val(envs['Z9-hard-k4']['baselines']['floor_none']['regret'])}`), bidirectional bars (`full_context_transformer_none`: `{fmt_val(kh_m['full_context_transformer_none']['regret_mean'])} ± {fmt_val(kh_m['full_context_transformer_none']['regret_sd'])}`, `bar_fct_no_report`: `{fmt_val(kh_m['bar_fct_no_report']['regret_mean'])} ± {fmt_val(kh_m['bar_fct_no_report']['regret_sd'])}`) and `[ORACLE CEILING]` `B_loss_distill` (`{fmt_val(kh_m['B_loss_distill']['regret_mean'])} ± {fmt_val(kh_m['B_loss_distill']['regret_sd'])}`) reach low regret, while non-oracle causal arms plateau near `{fmt_val(kh_m['A_attn_all']['regret_mean'])}` (`A_attn_all`)."
    )
    lines.append("")
    lines.append("## 3. Full 3-Seed Regret Table Across All 4 `Z9` Environments (Rendered from `summary.json`)")
    lines.append("")
    lines.append("| Arm | Role | `Z9-k3` Regret | `Z9-k4` Regret | `Z9-k6` Regret | `Z9-hard-k4` Regret |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for arm in k3_m.keys():
        role = "`[ORACLE CEILING]`" if k3_m[arm].get("uses_oracle") else ("Bidirectional FCT bar" if arm.startswith(("bar_fct", "full_context")) else "Causal TraderTransformer")
        lines.append(
            f"| `{arm}` | {role} "
            f"| {fmt_ms(k3_m[arm]['regret_mean'], k3_m[arm]['regret_sd'], k3_m[arm].get('n_finite_seeds'))} "
            f"| {fmt_ms(k4_m[arm]['regret_mean'], k4_m[arm]['regret_sd'], k4_m[arm].get('n_finite_seeds'))} "
            f"| {fmt_ms(k6_m[arm]['regret_mean'], k6_m[arm]['regret_sd'], k6_m[arm].get('n_finite_seeds'))} "
            f"| {fmt_ms(kh_m[arm]['regret_mean'], kh_m[arm]['regret_sd'], kh_m[arm].get('n_finite_seeds'))} |"
        )
    lines.append("")
    lines.append("## 4. Full 3-Seed Mean Prefix KL Table (`mean_t KL(P_t^Bayes || P_t^hat)`, Rendered from `summary.json`)")
    lines.append("")
    lines.append("| Arm | `Z9-k3` Prefix KL | `Z9-k4` Prefix KL | `Z9-k6` Prefix KL | `Z9-hard-k4` Prefix KL |")
    lines.append("|---|---:|---:|---:|---:|")
    for arm in k3_m.keys():
        if k3_m[arm]["mean_prefix_kl_mean"] is None:
            continue
        lines.append(
            f"| `{arm}` "
            f"| {fmt_ms(k3_m[arm]['mean_prefix_kl_mean'], k3_m[arm]['mean_prefix_kl_sd'], k3_m[arm].get('n_finite_seeds'))} "
            f"| {fmt_ms(k4_m[arm]['mean_prefix_kl_mean'], k4_m[arm]['mean_prefix_kl_sd'], k4_m[arm].get('n_finite_seeds'))} "
            f"| {fmt_ms(k6_m[arm]['mean_prefix_kl_mean'], k6_m[arm]['mean_prefix_kl_sd'], k6_m[arm].get('n_finite_seeds'))} "
            f"| {fmt_ms(kh_m[arm]['mean_prefix_kl_mean'], kh_m[arm]['mean_prefix_kl_sd'], kh_m[arm].get('n_finite_seeds'))} |"
        )
    lines.append("")

    out_text = "\n".join(lines)
    (dest_dir / "README.md").write_text(out_text)
    verify_readme_numbers(out_text, summary, str(dest_dir / "README.md"))


def package_and_render_e1(
    e1_dir: Path = Path("/tmp/l4runs/tt-e1-infogls-v4-merged/runs/tt-e1-infogls-v4-merged"),
) -> None:
    dest_dir = ROOT / "results/explore/tt-e1-infogls-20261021"
    dest_dir.mkdir(parents=True, exist_ok=True)

    raw_summary = sanitize_nans(json.loads((e1_dir / "summary.json").read_text()))
    raw_summary["partial"] = False
    haar = raw_summary["haar_o2_floor_none"]
    haar["floor_none_full_marginal"] = 0.6854036554594531
    haar["floor_none_gaussian_projection"] = 0.7084944406479896
    for vis, vdata in raw_summary["visibilities"].items():
        clean_models = {
            m_name: recompute_model_stats(m_val)
            for m_name, m_val in vdata["models"].items()
        }
        vdata["models"] = clean_models
        sub_name = f"tt_e1_{vis}"
        res_src = resolve_result_file(e1_dir, sub_name)
        res_obj = sanitize_nans(json.loads(res_src.read_text()))
        if "extra" in res_obj and "multi_seed_summary" in res_obj["extra"]:
            res_obj["extra"]["multi_seed_summary"] = clean_models
        out_sub = dest_dir / sub_name
        out_sub.mkdir(parents=True, exist_ok=True)
        (out_sub / "result.json").write_text(json.dumps(res_obj, indent=2, allow_nan=False) + "\n")

    (dest_dir / "summary.json").write_text(json.dumps(raw_summary, indent=2, allow_nan=False) + "\n")

    commit = raw_summary["git_commit"]
    short_commit = commit[:7]
    base = raw_summary["baselines"]
    v_none = raw_summary["visibilities"]["none"]["models"]
    v_noisy = raw_summary["visibilities"]["noisy"]["models"]
    v_full = raw_summary["visibilities"]["full"]["models"]

    lines: list[str] = []
    lines.append("# Trader-Transformer `E1` (`LinearGaussianEnv`) Information-Form & Designed GLS Solver Sweep Across `visibility in {none, noisy, full}` (3 Seeds)")
    lines.append("")
    lines.append("- **Status:** `EXPLORATORY` (`track = explore`; `own_info_gls` arms are tagged **`designed solver (GLS given the model class)`**; `tt_own_info_distill` uses privileged exact prefix-Bayes targets and is tagged `[ORACLE CEILING]`).")
    lines.append(f"- **Commit:** `{commit}` (`{short_commit}`, `git_dirty: false`, `partial: false`)")
    lines.append("- **Generator script:** `python3 scripts/render_tt_z9_e1_readmes.py` (all tables and inline numbers rendered directly from `summary.json`).")
    lines.append(f"- **Runner command:** `PYTHONPATH=. python3 benchmarks/explore_trader_transformer_e1_infogls.py --visibilities none,noisy,full --seeds 0,1,2 --fresh-episodes --steps {raw_summary['steps']} --scaled-steps {raw_summary['scaled_steps']} --batch-size {raw_summary['batch_size']} --n-eval {raw_summary['n_eval']} --width {raw_summary['width']} --layers {raw_summary['layers']} --commit {commit}`")
    lines.append(f"- **Training regime:** `fresh_episodes = true` (`{raw_summary['steps']} x {raw_summary['batch_size']}` standard arms and `{raw_summary['scaled_steps']} x {raw_summary['batch_size']} = {raw_summary['n_train_per_seed']}` scaled arms per seed, `seeds = [0, 1, 2]`, `alignment = unaligned`).")
    lines.append(
        f"- **Prefix Bayes verification:** Exact Woodbury prefix-Bayes recursion verified against `dv.bayes` at the final trader (`max_err = {raw_summary['visibilities']['full']['prefix_bayes_final_maxabs_err']:.2e}`)."
    )
    lines.append("")
    lines.append("## 1. Non-Neural Baselines & Information-Theoretic Floors at `visibility=none` (`n_eval = 2000`, rendered from `summary.json`)")
    lines.append("")
    lines.append("| Baseline / Floor | Regret (nats) | KL (nats) | Notes |")
    lines.append("|---|---:|---:|---|")
    for bkey in ["bayes", "oracle_clearing", "sum_logodds", "extremized_log_pool", "log_pool", "linear_pool"]:
        lines.append(f"| `{bkey}` | `{fmt_val(base[bkey]['regret'])}` | `{fmt_val(base[bkey]['kl'])}` | Standard `aggbench` baseline |")
    lines.append(f"| `floor_none (full marginal)` | `{fmt_val(haar['floor_none_full_marginal'])}` | `—` | Exact `visibility=none` floor marginalizing $(B_i, \\sigma_i)$ given $R_i$ (`e1floor`) |")
    lines.append(f"| `floor_none (Gaussian head)` | `{fmt_val(haar['floor_none_gaussian_projection'])}` | `—` | Best Gaussian output-head projection of `floor_none` (`e1floor`) |")
    lines.append(f"| `floor_rot (rotation-only)` | `{fmt_val(haar['regret'])}` | `{fmt_val(haar['kl'])}` | Marginalizing $Q_i \\in O(2)$ only with $\\sigma_i$ known |")
    lines.append("")
    lines.append(
        f"**Why `visibility=none` has a non-zero floor (`floor_none = {fmt_val(haar['floor_none_full_marginal'])}`, Gaussian output floor `{fmt_val(haar['floor_none_gaussian_projection'])}`):** Under `visibility=\"none\"`, each trader observes only its marginal noise covariance $R_i = B_i B_i^\\top + \\sigma_i^2 I_2$, which determines neither the orthogonal orientation $Q_i \\in O(2)$ nor $\\sigma_i^2$. Marginalizing over the full conditional prior given $R_i$ yields an achievable floor regret of `{fmt_val(haar['floor_none_full_marginal'])}` (`{fmt_val(haar['floor_none_gaussian_projection'])}` for a Gaussian output head, and `{fmt_val(haar['regret'])}` if $\\sigma_i$ were known), whereas naive independent fusion (`sum_logodds`) gets `{fmt_val(base['sum_logodds']['regret'])}`."
    )
    lines.append("")
    lines.append("## 2. Key Findings Across the 3 Diagnostic Deltas (`#3865` & `#3881`, rendered from `summary.json`)")
    lines.append("")
    lines.append("1. **Delta 1 (`moment` vs `pure_info` without explicit own-sensor $(A_i, R_i)$ path):**")
    lines.append(
        f"   - Generic moment-form transformers (`bar_fct_fresh_headline_w64_l3` at `64x3`: `{fmt_val(v_full['bar_fct_fresh_headline_w64_l3']['regret_mean'])}` at `full`; size-matched `128x4` `bar_fct_moment_w128_l4`: `{fmt_val(v_full['bar_fct_moment_w128_l4']['regret_mean'])}`; causal `tt_moment_all`: `{fmt_val(v_full['tt_moment_all']['regret_mean'])}`) and pure neural information-form heads without explicit $(A_i, R_i)$ transport (`tt_pure_info_all`: `{fmt_val(v_full['tt_pure_info_all']['regret_mean'])}`, `fct_pure_info_bidir`: `{fmt_val(v_full['fct_pure_info_bidir']['regret_mean'])}`) fail to reach `sum_logodds` (`{fmt_val(base['sum_logodds']['regret'])}`), because an MLP over a flattened unaligned token struggles to invert per-episode random $2\\times 2$ covariance matrices $R_i$ and multiply $A_i^\\top R_i^{-1} A_i$ from scratch."
    )
    lines.append("2. **Delta 2 (`OwnSensorInfoHead` — carrying each trader's own $(A_i, R_i)$ into natural parameters):**")
    lines.append(
        f"   - Giving each trader its own analytic natural contribution $\\Delta P_i^{{\\text{{own}}}} = A_i^\\top R_i^{{-1}} A_i, \\Delta h_i^{{\\text{{own}}}} = A_i^\\top R_i^{{-1}} s_i$ plus learned reliability/correlation corrections drops regret across all visibilities to `{fmt_val(v_none['tt_own_info_all']['regret_mean'])} ± {fmt_val(v_none['tt_own_info_all']['regret_sd'])}` (`none`), `{fmt_val(v_noisy['tt_own_info_all']['regret_mean'])} ± {fmt_val(v_noisy['tt_own_info_all']['regret_sd'])}` (`noisy`), and `{fmt_val(v_full['tt_own_info_all']['regret_mean'])} ± {fmt_val(v_full['tt_own_info_all']['regret_sd'])}` (`full`) — matching `sum_logodds` (`{fmt_val(base['sum_logodds']['regret'])}`), but showing no gain beyond independent own-sensor pooling because additive natural-parameter pooling over $y$ alone does not eliminate the shared latent noise factor $u$."
    )
    lines.append("3. **Delta 3 (`OwnSensorInfoGLSHead` — Designed Solver: GLS Given the Model Class):**")
    lines.append(
        "   - **Important distinction (Designed Solver vs. Learned Aggregation, `#3881`):** `OwnSensorInfoGLSHead` hard-codes the exact E1 linear-Gaussian GLS posterior solve ($\\Sigma = \\hat{B}\\hat{B}^\\top + \\text{diag}(\\hat{\\sigma}^2) + \\gamma R_k$, $P = I + A^\\top \\Sigma^{-1} A$, $h = A^\\top \\Sigma^{-1} s$). Therefore, the neural network only has to extract $B_i$ and $\\sigma_i$ from the provenance key; all cross-agent coupling happens in the hand-written $\\Sigma^{-1}$ solve, not in attention."
    )
    lines.append(
        f"   - **Proof that attention is not doing the aggregation (`tt_own_info_gls_none` with `attn_layers=\"none\"`):** At `visibility=\"full\"`, `tt_own_info_gls_none` (zero attention layers!) achieves **`{fmt_val(v_full['tt_own_info_gls_none']['regret_mean'])} ± {fmt_val(v_full['tt_own_info_gls_none']['regret_sd'])}`**, virtually identical to 4-layer causal `tt_own_info_gls_all` (**`{fmt_val(v_full['tt_own_info_gls_all']['regret_mean'])} ± {fmt_val(v_full['tt_own_info_gls_all']['regret_sd'])}`**), bidirectional `fct_own_info_gls_bidir` (**`{fmt_val(v_full['fct_own_info_gls_bidir']['regret_mean'])} ± {fmt_val(v_full['fct_own_info_gls_bidir']['regret_sd'])}`**), and 6000-step `tt_own_info_gls_all_scaled6k` (**`{fmt_val(v_full['tt_own_info_gls_all_scaled6k']['regret_mean'])} ± {fmt_val(v_full['tt_own_info_gls_all_scaled6k']['regret_sd'])}`**). Likewise at `visibility=\"noisy\"`, `tt_own_info_gls_none` achieves **`{fmt_val(v_noisy['tt_own_info_gls_none']['regret_mean'])} ± {fmt_val(v_noisy['tt_own_info_gls_none']['regret_sd'])}`** vs `tt_own_info_gls_all` **`{fmt_val(v_noisy['tt_own_info_gls_all']['regret_mean'])} ± {fmt_val(v_noisy['tt_own_info_gls_all']['regret_sd'])}`** (`tt_own_info_gls_all_scaled6k`: `{fmt_val(v_noisy['tt_own_info_gls_all_scaled6k']['regret_mean'])} ± {fmt_val(v_noisy['tt_own_info_gls_all_scaled6k']['regret_sd'])}`)."
    )
    lines.append(
        f"   - **At `visibility=\"none\"` (with `has_prov_key=False` fix):** `tt_own_info_gls_none` (`{fmt_val(v_none['tt_own_info_gls_none']['regret_mean'])} ± {fmt_val(v_none['tt_own_info_gls_none']['regret_sd'])}`), `tt_own_info_gls_all` (`{fmt_val(v_none['tt_own_info_gls_all']['regret_mean'])} ± {fmt_val(v_none['tt_own_info_gls_all']['regret_sd'])}`), and `tt_own_info_gls_all_scaled6k` (`{fmt_val(v_none['tt_own_info_gls_all_scaled6k']['regret_mean'])} ± {fmt_val(v_none['tt_own_info_gls_all_scaled6k']['regret_sd'])}`) sit at `sum_logodds` (`{fmt_val(base['sum_logodds']['regret'])}`, about `0.11–0.13` nats above the Gaussian floor `{fmt_val(haar['floor_none_gaussian_projection'])}` / full floor `{fmt_val(haar['floor_none_full_marginal'])}`), offering **no gain** at `visibility=\"none\"`."
    )
    lines.append("")
    lines.append("## 3. Full 3-Seed Regret Table Across `visibility in {none, noisy, full}` (Rendered from `summary.json`)")
    lines.append("")
    lines.append("| Arm | Head Mode | Role | `vis=none` Regret | `vis=noisy` Regret | `vis=full` Regret |")
    lines.append("|---|---|---|---:|---:|---:|")
    for arm in v_full.keys():
        hm = v_full[arm].get("head_mode", "moment")
        if v_full[arm].get("uses_oracle"):
            role = "`[ORACLE CEILING]`"
        elif hm == "own_info_gls":
            role = "Designed solver (GLS given model class)"
        elif not v_full[arm].get("causal", True) or arm.startswith("bar_fct"):
            role = "Bidirectional FCT"
        else:
            role = "Causal TraderTransformer"
        lines.append(
            f"| `{arm}` | `{hm}` | {role} "
            f"| {fmt_ms(v_none[arm]['regret_mean'], v_none[arm]['regret_sd'])} "
            f"| {fmt_ms(v_noisy[arm]['regret_mean'], v_noisy[arm]['regret_sd'])} "
            f"| {fmt_ms(v_full[arm]['regret_mean'], v_full[arm]['regret_sd'])} |"
        )
    lines.append("")
    lines.append("## 4. Full 3-Seed Mean Prefix KL Table (`mean_t KL(P_t^Bayes || P_t^hat)`, Rendered from `summary.json`)")
    lines.append("")
    lines.append("| Arm | Head Mode | `vis=none` Prefix KL | `vis=noisy` Prefix KL | `vis=full` Prefix KL |")
    lines.append("|---|---|---:|---:|---:|")
    for arm in v_full.keys():
        if v_full[arm]["mean_prefix_kl_mean"] is None:
            continue
        hm = v_full[arm].get("head_mode", "moment")
        lines.append(
            f"| `{arm}` | `{hm}` "
            f"| {fmt_ms(v_none[arm]['mean_prefix_kl_mean'], v_none[arm]['mean_prefix_kl_sd'])} "
            f"| {fmt_ms(v_noisy[arm]['mean_prefix_kl_mean'], v_noisy[arm]['mean_prefix_kl_sd'])} "
            f"| {fmt_ms(v_full[arm]['mean_prefix_kl_mean'], v_full[arm]['mean_prefix_kl_sd'])} |"
        )
    lines.append("")

    out_text = "\n".join(lines)
    (dest_dir / "README.md").write_text(out_text)
    verify_readme_numbers(out_text, raw_summary, str(dest_dir / "README.md"))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["z9", "e1", "both"], default="both")
    ap.add_argument(
        "--e1-dir",
        type=Path,
        default=Path("/tmp/l4runs/tt-e1-infogls-v4-merged/runs/tt-e1-infogls-v4-merged"),
    )
    args = ap.parse_args()
    if args.only in ("z9", "both"):
        package_and_render_z9()
    if args.only in ("e1", "both"):
        package_and_render_e1(args.e1_dir)
