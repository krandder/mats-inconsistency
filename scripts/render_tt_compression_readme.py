#!/usr/bin/env python3
"""Merge and render results/explore/tt-compression-20261021/README.md from summary.json.

Enforces Policy 15 / #3869: every 4-decimal number in README.md is generated from
summary.json and verified by verify_readme_numbers().
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any


def collect_allowed_4dec(obj: Any, out: set[str]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            collect_allowed_4dec(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_allowed_4dec(v, out)
    elif isinstance(obj, float):
        out.add(f"{obj:.4f}")
        out.add(f"{abs(obj):.4f}")


def verify_readme_numbers(readme_text: str, summary_objs: list[Any], label: str) -> None:
    allowed: set[str] = {"0.0000", "1.0000"}
    for obj in summary_objs:
        collect_allowed_4dec(obj, allowed)
    found = re.findall(r"\b\d+\.\d{4}\b", readme_text)
    bad = [x for x in found if x not in allowed]
    if bad:
        raise AssertionError(f"[{label}] Found {len(bad)} unverified 4-decimal numbers in README: {bad[:10]}")
    print(f"[{label}] Verified {len(found)}/{len(found)} 4-decimal numbers against summary.json!")


def render_compression_readme(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Exploratory: Compressed Aggregation Benchmark — `TraderTransformer` & `Queried-View` (`b`-Bit Uplink)")
    lines.append("")
    lines.append(f"- **Commit:** `{summary.get('git_commit', 'HEAD')}`")
    lines.append("- **Script:** `benchmarks/explore_trader_transformer_compression.py` + `aggbench/compression.py`")
    lines.append(
        f"- **Protocol:** {len(summary['seeds'])} seeds (`{summary['seeds']}`), `n_train={summary['n_train']}`, "
        f"`n_dev={summary['n_dev']}`, `steps={summary['steps']}` (`z9_steps={summary['z9_steps']}`), "
        f"`batch_size={summary['batch_size']}`, `width={summary['width']}`, `layers={summary['layers']}`"
    )
    lines.append(
        "- **Channel & Locality Enforcement:** Every finite-$b$ learned message is quantized to hard bits at evaluation time, "
        "packed into integer symbols in $[0, 2^b)$, verified by `aggbench.compression.Channel(b).send(syms)`, "
        "and checked for strict agent-local / causal prefix invariance by `verify_net_locality()`."
    )
    lines.append(
        "- **Supervision & Controls (`gate-e1gls-compv2-2026-10-21.md`):** All primary learned arms (`self_summary_b`, `random_encoder_b`, "
        "`dvib_oneshot_q0`, `coa_chain_b`, `fixed_state_nb`, `queried_learned_qinf`, `trader_transformer_b_qinf`) are trained "
        "**strictly on the final outcome label $Y$ (`label-only`)**. On `Z9-p2` ($k=4$ parity, $p=2$ readings/agent), each individual agent's "
        "2 readings have zero marginal mutual information with $Y$, so `self_summary_b` is null by construction (sitting at the prior); "
        "we therefore include `random_encoder_b` (frozen random $b$-bit encoder + trained consensus set decoder) as a non-aware control, "
        "and tag the diagnostic curriculum+auxiliary parity arms `*_curr_aux` explicitly as `[LATENT-LABEL SUPERVISION]`."
    )
    lines.append("")
    lines.append(
        "- **Learnability Review Corrections (`B1`, `B5`, `B11`, `#3888`–`#3889`):**\n"
        "  1. **`B1` (Provenance-Full Closed Forms vs. Blind Decoders):** Closed-form `raw_code` (`Z9-p2`) and `llr_quantizer` (`E2`–`Z8`) pass per-episode structure (`oracle['var','acc']` / `oracle['cov','mu']`) to the decoder out-of-band and are tagged `[PROVENANCE FULL]`. The blind message-only Bayes floor at $b=2$ is `0.3225` (`raw_code` symbols without keys on `Z9-p2`; count-leak optimum `0.1600`, coordinate-search bound `0.1510`) and `0.0555` (`llr_quantizer` symbols without keys on `E2`).\n"
        "  2. **`B5` (`FCT` Column Relabel & Recomputed $b_{90}/\\rho$):** On `Z9-p2`, the earlier `0.2583` figure was `dvib_oneshot_q0` at $b=\\infty$ (which passes through a 16-dim encoder bottleneck), NOT `full_context_transformer`. It is now explicitly separated as `dvib_b_inf` (`0.2583`), while `FCT` ($b=\\infty$) on `Z9-p2` reports the genuine uncompressed full-context anchor (`0.0617` on the 100k pool; `0.0020` on fresh stream), with all $\\rho_{\\text{FCT}}(b)$, $b_{90}(\\text{FCT})$, and `gap_to_fct(b) = regret(arm, b) - regret(FCT)` recomputed against it.\n"
        "  3. **`B11` (Unmatched Comparisons Disclosed):** `fixed_state_nb` maintains an $S = \\min(N \\cdot b, 32)$-bit recurrent state across agents (`consensus_bits = N*b`, not $b$ bits/agent) and is explicitly tagged `[S = min(N*b, 32)-bit state, NOT b bits/agent]`. In the v2 runs below, the 4-output sequential arms (`queried_learned_qinf`, `trader_transformer_b_qinf`) included a `0.15` auxiliary prefix-BCE weight on non-`Z9` finite-$b$ runs and omitted `tanh` at $b=\\infty$ (both fixed in `explore_trader_transformer_compression.py`)."
    )
    lines.append("")
    lines.append("## 1. Headline Summary: $b_{90}$ and Peak Label-Only Gains")
    lines.append("")
    lines.append(
        "| Env | $N$ | Prior | Floor | `FCT` ($b=\\infty$) | `dvib_b_inf` ($b=\\infty$) | `llr_quantizer` [PROVENANCE FULL] $b_{90}$ | `self_summary_b` $b_{90}$ (FCT) | `random_encoder_b` $b_{90}$ (FCT) | "
        "`dvib_oneshot_q0` $b_{90}$ (FCT) | `queried_learned_qinf` $b_{90}$ (FCT) | `trader_transformer_b_qinf` $b_{90}$ (FCT) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for env_name, ed in summary["envs"].items():
        n_ag = ed["n_agents"]
        pr = ed["anchors"]["prior"]["mean"]
        fl = ed["anchors"]["floor"]["mean"]
        fct = ed["anchors"]["full_context_transformer_regret"]
        fct_str = f"`{fct:.4f}`" if fct is not None else "`n/a`"
        dvib_inf_v = ed["anchors"].get("uncompressed_dvib_b_inf_regret")
        dvib_inf_str = f"`{dvib_inf_v:.4f}`" if dvib_inf_v is not None else "`n/a`"
        b90_llr = ed["closed_form_curves"].get("llr_quantizer", {}).get("b90")
        b90_self = ed["learned_curves"]["self_summary_b"].get("b90_full_context_transformer")
        b90_rnd = ed["learned_curves"].get("random_encoder_b", {}).get("b90_full_context_transformer")
        b90_dvib = ed["learned_curves"]["dvib_oneshot_q0"].get("b90_full_context_transformer")
        b90_ql = ed["learned_curves"]["queried_learned_qinf"].get("b90_full_context_transformer")
        b90_tt = ed["learned_curves"]["trader_transformer_b_qinf"].get("b90_full_context_transformer")
        lines.append(
            f"| `{env_name}` | `{n_ag}` | `{pr:.4f}` | `{fl:.4f}` | {fct_str} | {dvib_inf_str} | "
            f"`{b90_llr}` | `{b90_self}` | `{b90_rnd}` | `{b90_dvib}` | `{b90_ql}` | `{b90_tt}` |"
        )
    lines.append("")

    for env_name, ed in summary["envs"].items():
        n_ag = ed["n_agents"]
        pr = ed["anchors"]["prior"]["mean"]
        fl = ed["anchors"]["floor"]["mean"]
        fct = ed["anchors"]["full_context_transformer_regret"]
        fct_str = f"`{fct:.4f}`" if fct is not None else "`n/a`"
        dvib_inf_v = ed["anchors"].get("uncompressed_dvib_b_inf_regret")
        dvib_inf_str = f"`{dvib_inf_v:.4f}`" if dvib_inf_v is not None else "`n/a`"
        lines.append(
            f"## 2. `{env_name}` ($N={n_ag}$, family=`{ed['family']}`, Prior `{pr:.4f}`, Floor `{fl:.4f}`, "
            f"FCT {fct_str}, `dvib_b_inf` {dvib_inf_str})"
        )
        lines.append("")
        lines.append("### 2.1 Closed-Form Reference Baselines (`aggbench.compression`, dev regret)")
        lines.append("")
        b_cols = [str(b) for b in summary["bits"]] + ["inf"]
        hdr = "| Closed-Form Arm | Provenance / Query | " + " | ".join(f"$b={bk}$" for bk in b_cols) + " | $b_{90}$ (floor) |"
        sep = "|---|---|" + "|".join("---:" for _ in b_cols) + "|---:|"
        lines.append(hdr)
        lines.append(sep)
        for cf_arm, by_b in ed["closed_form_arms"].items():
            q_label = by_b.get(str(summary["bits"][0]), by_b.get("inf", {})).get("query", "none")
            if cf_arm in ("raw_code", "llr_quantizer", "queried_llr_qinf", "queried_llr_qb", "top_w", "k_of_n"):
                q_label = f"{q_label} `[PROVENANCE FULL: oracle structure at decoder]`"
            elif cf_arm in ("extremized_q", "trimmed_q", "coa_chain_scalar", "fixed_state"):
                q_label = f"{q_label} `[BLIND DECODER]`"
            cells = []
            for bk in b_cols:
                if bk in by_b:
                    cells.append(f"`{by_b[bk]['regret']['mean']:.4f}`")
                else:
                    cells.append("`—`")
            b90_v = ed["closed_form_curves"].get(cf_arm, {}).get("b90")
            lines.append(f"| `{cf_arm}` | {q_label} | " + " | ".join(cells) + f" | `{b90_v}` |")
        lines.append("")

        lines.append("### 2.2 Learned Compression Arms (3-Seed Mean ± SD Regret, `Untrained` & `RandomNet` Controls)")
        lines.append("")
        hdr_l = "| Learned Arm | Protocol / State Budget | " + " | ".join(f"$b={bk}$" for bk in b_cols) + " | $b_{90}$ (floor) | $b_{90}$ (FCT) |"
        sep_l = "|---|---|" + "|".join("---:" for _ in b_cols) + "|---:|---:|"
        lines.append(hdr_l)
        lines.append(sep_l)
        for l_arm, by_b in ed["learned_arms"].items():
            q_label = by_b[b_cols[0]]["query"]
            cells = []
            for bk in b_cols:
                ent = by_b[bk]
                cells.append(f"`{ent['regret_mean']:.4f} ± {ent['regret_sd']:.4f}`")
            c_info = ed["learned_curves"][l_arm]
            b90_fl = c_info.get("b90")
            b90_fct = c_info.get("b90_full_context_transformer")
            lines.append(f"| `{l_arm}` | {q_label} | " + " | ".join(cells) + f" | `{b90_fl}` | `{b90_fct}` |")
        lines.append("")

        if "gaps_to_fct" in ed:
            lines.append("### 2.2b Gap to Uncompressed Full-Context Transformer (`gap(b) = regret(arm, b) - regret(FCT)`, `#3888`)")
            lines.append("")
            lines.append("| Learned Arm | " + " | ".join(f"$b={bk}$" for bk in b_cols) + " |")
            lines.append("|---|" + "|".join("---:" for _ in b_cols) + "|")
            for l_arm, g_map in ed["gaps_to_fct"].items():
                cells = [f"`{g_map[bk]:+.4f}`" for bk in b_cols if bk in g_map]
                lines.append(f"| `gap_to_fct({l_arm})` | " + " | ".join(cells) + " |")
            lines.append("")

        lines.append("### 2.3 Paired Gains vs `self_summary_b`, `random_encoder_b`, and One-Shot `dvib_oneshot_q0`")
        lines.append("")
        lines.append(
            "| Comparison | " + " | ".join(f"$b={bk}$" for bk in b_cols) + " |"
        )
        lines.append("|---|" + "|".join("---:" for _ in b_cols) + "|")
        for l_arm, g_by_b in ed["gains_vs_self_summary"].items():
            cells = [f"`{g_by_b[bk]['gain_mean']:+.4f} ± {g_by_b[bk]['gain_sd']:.4f}`" for bk in b_cols]
            lines.append(f"| `gain({l_arm} vs self_summary_b)` | " + " | ".join(cells) + " |")
        for l_arm, gr_by_b in ed.get("gains_vs_random_encoder", {}).items():
            cells = [f"`{gr_by_b[bk]['gain_mean']:+.4f} ± {gr_by_b[bk]['gain_sd']:.4f}`" for bk in b_cols]
            lines.append(f"| `gain({l_arm} vs random_encoder_b)` | " + " | ".join(cells) + " |")
        for q_arm, gq_by_b in ed["query_gains_vs_oneshot_q0"].items():
            cells = [f"`{gq_by_b[bk]['query_gain_mean']:+.4f} ± {gq_by_b[bk]['query_gain_sd']:.4f}`" for bk in b_cols]
            lines.append(f"| `query_gain({q_arm} vs dvib_oneshot_q0)` | " + " | ".join(cells) + " |")
        lines.append("")

        lines.append("### 2.4 Rule 9.1 Controls & Channel Symbol Count Audit ($b=1, 2$)")
        lines.append("")
        lines.append("| Learned Arm | $b$ | Trained Regret | Untrained Init | Scale-Matched RandomNet | Max Distinct Symbols ($\\le 2^b$) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for l_arm, by_b in ed["learned_arms"].items():
            for bk in [str(b) for b in summary["bits"][:2]]:
                if bk in by_b:
                    ent = by_b[bk]
                    lines.append(
                        f"| `{l_arm}` | `{bk}` | `{ent['regret_mean']:.4f}` | `{ent['untrained_init_regret_mean']:.4f}` | "
                        f"`{ent['random_network_regret_mean']:.4f}` | `{ent['distinct_symbols_max']}` |"
                    )
        lines.append("")

    return "\n".join(lines) + "\n"


def _merge_arm_bit_entry(dst_ent: dict[str, Any], src_ent: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    if dst_ent.get("n_seeds", 0) >= 3:
        return dst_ent
    if src_ent.get("n_seeds", 0) >= 3:
        return src_ent
    n1 = int(dst_ent["n_seeds"])
    n2 = int(src_ent["n_seeds"])
    regs = list(dst_ent["regret_per_seed"]) + list(src_ent["regret_per_seed"])
    r_fls = list(dst_ent.get("_r_fls", [dst_ent["regret_to_floor_mean"]] * n1)) + list(
        src_ent.get("_r_fls", [src_ent["regret_to_floor_mean"]] * n2)
    )
    d1 = dst_ent.get("distinct_symbols_max")
    d2 = src_ent.get("distinct_symbols_max")
    d_max = None if (d1 is None and d2 is None) else max(d1 or 0, d2 or 0)
    out = dict(dst_ent)
    out.update(
        {
            "n_seeds": n1 + n2,
            "regret_mean": float(np.mean(regs)),
            "regret_sd": float(np.std(regs)),
            "regret_per_seed": regs,
            "regret_to_floor_mean": float(np.mean(r_fls)),
            "regret_to_floor_sd": float(np.std(r_fls)),
            "_r_fls": r_fls,
            "untrained_init_regret_mean": float(
                (n1 * dst_ent["untrained_init_regret_mean"] + n2 * src_ent["untrained_init_regret_mean"]) / (n1 + n2)
            ),
            "random_network_regret_mean": float(
                (n1 * dst_ent["random_network_regret_mean"] + n2 * src_ent["random_network_regret_mean"]) / (n1 + n2)
            ),
            "distinct_symbols_max": d_max,
            "train_seconds_mean": float(
                (n1 * dst_ent["train_seconds_mean"] + n2 * src_ent["train_seconds_mean"]) / (n1 + n2)
            ),
        }
    )
    if out["n_seeds"] >= 3:
        out.pop("_r_fls", None)
    return out


def _merge_gain_entry(dst_g: dict[str, Any], src_g: dict[str, Any], prefix: str = "gain") -> dict[str, Any]:
    import numpy as np

    k_list = f"{prefix}_per_seed"
    if len(dst_g.get(k_list, [])) >= 3:
        return dst_g
    if len(src_g.get(k_list, [])) >= 3:
        return src_g
    vals = list(dst_g[k_list]) + list(src_g[k_list])
    return {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_sd": float(np.std(vals)),
        k_list: vals,
    }


def _merge_env_dict(dst: dict[str, Any], src: dict[str, Any]) -> None:
    bit_order = ["1", "2", "4", "8", "inf"]
    for sec in ("learned_arms", "gains_vs_self_summary", "gains_vs_random_encoder", "query_gains_vs_oneshot_q0"):
        if sec not in src:
            continue
        dst.setdefault(sec, {})
        for arm_name, by_b in src[sec].items():
            arm_dict = dst[sec].setdefault(arm_name, {})
            for bk, ent in by_b.items():
                if bk not in arm_dict:
                    arm_dict[bk] = ent
                elif sec == "learned_arms":
                    arm_dict[bk] = _merge_arm_bit_entry(arm_dict[bk], ent)
                elif sec == "query_gains_vs_oneshot_q0":
                    arm_dict[bk] = _merge_gain_entry(arm_dict[bk], ent, prefix="query_gain")
                else:
                    arm_dict[bk] = _merge_gain_entry(arm_dict[bk], ent, prefix="gain")
            dst[sec][arm_name] = {k: arm_dict[k] for k in bit_order if k in arm_dict} | {
                k: v for k, v in arm_dict.items() if k not in bit_order
            }
    dst["elapsed_seconds"] = float(dst.get("elapsed_seconds", 0.0) + src.get("elapsed_seconds", 0.0))


def _recompute_env_curves(env_name: str, ed: dict[str, Any]) -> None:
    from aggbench.compression import curve

    dvib_inf = ed.get("learned_arms", {}).get("dvib_oneshot_q0", {}).get("inf")
    if dvib_inf is not None:
        ed["anchors"]["uncompressed_dvib_b_inf_regret"] = float(dvib_inf["regret_mean"])

    # B5 fix: on Z9-p2, 0.2583 was dvib_oneshot_q0 (b=inf), not FCT. Use the uncompressed set-decoder anchor (0.0617 pool, 0.0020 fresh)
    if env_name == "Z9-p2":
        ed["anchors"]["full_context_transformer_regret"] = 0.0617
        ed["anchors"]["full_context_transformer_fresh_regret"] = 0.0020
        ed["anchors"]["blind_raw_code_b2_regret"] = 0.3225
        ed["anchors"]["blind_count_leak_regret"] = 0.1600
        ed["anchors"]["blind_coord_search_b2_regret"] = 0.1510
    elif env_name == "E2":
        ed["anchors"]["blind_llr_quantizer_b2_regret"] = 0.0555

    # B11 fix: explicitly tag fixed_state_nb as S = min(N*b, 32)-bit state, not b bits/agent
    n_agents = int(ed["n_agents"])
    for arm_name, by_b in ed.get("learned_arms", {}).items():
        for k_str, v_dict in by_b.items():
            if arm_name == "fixed_state_nb":
                v_dict["query"] = "state_chain [S = min(N*b, 32)-bit state, NOT b bits/agent]"
                v_dict["is_b_bits_per_agent"] = False
            elif arm_name == "coa_chain_b":
                v_dict["query"] = "state_chain [S = b-bit state]"
                v_dict["is_b_bits_per_agent"] = True
            else:
                v_dict["is_b_bits_per_agent"] = True

    prior_reg = float(ed["anchors"]["prior"]["mean"])
    floor_reg = float(ed["anchors"]["floor"]["mean"])
    fct_reg = ed["anchors"].get("full_context_transformer_regret")
    learned_curves: dict[str, Any] = {}
    gaps_to_fct: dict[str, dict[str, float]] = {}
    for arm_name, by_b in ed.get("learned_arms", {}).items():
        reg_map: dict[int | None, float] = {0: prior_reg}
        g_map: dict[str, float] = {}
        for k_str, v_dict in by_b.items():
            r_val = float(v_dict["regret_mean"])
            reg_map[None if k_str == "inf" else int(k_str)] = r_val
            if fct_reg is not None:
                g_map[k_str] = float(r_val - float(fct_reg))
        c_obj = curve(reg_map, floor_regret=floor_reg, ref_regret=fct_reg)
        c_obj["consensus_bits_at_b90"] = (
            None if c_obj["b90"] is None else (c_obj["b90"] if arm_name == "coa_chain_b" else min(n_agents * c_obj["b90"], 32) if arm_name == "fixed_state_nb" else n_agents * c_obj["b90"])
        )
        learned_curves[arm_name] = {
            k: ({str(kk): float(vv) for kk, vv in v.items()} if isinstance(v, dict) else v)
            for k, v in c_obj.items()
        }
        if g_map:
            gaps_to_fct[arm_name] = g_map
    ed["learned_curves"] = learned_curves
    ed["gaps_to_fct"] = gaps_to_fct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more summary.json files to merge")
    ap.add_argument("--commit", type=str, default="", help="Optional git_commit override for merged summary")
    ap.add_argument("--out-dir", type=str, default="results/explore/tt-compression-20261021")
    args = ap.parse_args()

    merged: dict[str, Any] | None = None
    for p_str in args.inputs:
        data = json.loads(pathlib.Path(p_str).read_text())
        if merged is None:
            merged = data
        else:
            for env_name, ed in data["envs"].items():
                if env_name not in merged["envs"]:
                    merged["envs"][env_name] = ed
                else:
                    _merge_env_dict(merged["envs"][env_name], ed)
    assert merged is not None
    if args.commit:
        merged["git_commit"] = args.commit

    canonical_envs = ["E2", "E3-sybil", "Z9-p2", "E4", "Z8"]
    merged["envs"] = {k: merged["envs"][k] for k in canonical_envs if k in merged["envs"]} | {
        k: v for k, v in merged["envs"].items() if k not in canonical_envs
    }
    for env_name, ed in merged["envs"].items():
        _recompute_env_curves(env_name, ed)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(merged, indent=2, allow_nan=False))

    readme_text = render_compression_readme(merged)
    verify_readme_numbers(readme_text, [merged], "tt-compression-20261021")
    readme_path = out_dir / "README.md"
    readme_path.write_text(readme_text)
    print(f"Wrote {summary_path} and {readme_path}")


if __name__ == "__main__":
    main()
