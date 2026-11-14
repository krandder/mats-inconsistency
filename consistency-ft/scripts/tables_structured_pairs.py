"""Markdown tables from /home/agent/data/structured-pairs-20260924/scores.json (score_structured_pairs.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

S = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "/home/agent/data/structured-pairs-20260924/scores.json").read_text())
TYPES = ("threshold", "date", "mc_excl", "control", "all")
SHORT = {"qwen3-32b": "32B", "qwen3-235b-a22b": "235B"}


def ci(v, k="log_loss"):
    if not v:
        return "n/a"
    d = v[k]
    return f"{d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"


def main():
    out = []
    for g in ("joint", "single"):
        out.append(f"\n### {g} forms: log loss (Δ vs market + independence, cluster-bootstrap 95% CI)\n")
        out.append("| type | model | arm | pairs | market indep | market + known relation | raw Δ | projected Δ | hybrid Δ | hybrid_and3 Δ |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for t in TYPES:
            for name, j in sorted(S["jobs"].items()):
                e = j["by_type"].get(t)
                if not e:
                    continue
                mi, mo = e["market_indep"][g]["log_loss"], e["market_oracle"][g]["log_loss"]
                row = [t, SHORT[j["model"]], j["arm"], str(e["scores"]["raw"][g]["pairs"]), f"{mi:.3f}", f"{mo:.3f}"]
                row += [f"{e['scores'][m][g]['log_loss']:.3f} ({ci(e['vs_market_indep'][m][g])})" for m in ("raw", "projected", "hybrid", "hybrid_and3")]
                out.append("| " + " | ".join(row) + " |")
    out.append("\n### joint forms: Δ vs market + KNOWN relation (oracle structure)\n")
    out.append("| type | model | arm | raw | projected | hybrid | hybrid_and3 |")
    out.append("|---|---|---|---|---|---|---|")
    for t in TYPES:
        for name, j in sorted(S["jobs"].items()):
            e = j["by_type"].get(t)
            if e:
                out.append(f"| {t} | {SHORT[j['model']]} | {j['arm']} | " +
                           " | ".join(ci(e["vs_market_oracle"][m]["joint"]) for m in ("raw", "projected", "hybrid", "hybrid_and3")) + " |")
    out.append("\n### oracle structure vs independence, market prices (joint / all 12)\n")
    out.append("| type | market indep joint | oracle joint | Δ joint [CI] | Δ all12 [CI] |")
    out.append("|---|---|---|---|---|")
    for t in TYPES[:-1]:
        b = S["baselines"][t]
        out.append(f"| {t} | {b['market_indep']['joint']['log_loss']:.3f} | {b['market_oracle']['joint']['log_loss']:.3f} | "
                   f"{ci(b['oracle_vs_indep']['joint'])} | {ci(b['oracle_vs_indep']['all12'])} |")
    out.append("\n### Does the model respect the known relation? (share of pairs; D = dependence captured, 1 = full)\n")
    out.append("| type | model | arm | form | order p(A)≤p(B) | zero cell ≤0.05 | AND≈p(A) | D median | D≥0.5 (n) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for t in ("threshold", "date"):
        for name, j in sorted(S["jobs"].items()):
            if t not in j["by_type"]:
                continue
            r = j["by_type"][t]["respect"]
            for k in ("raw", "projected"):
                x = r[k]
                out.append(f"| {t} | {SHORT[j['model']]} | {j['arm']} | {k} | {x['order']:.2f} | {x['zero_cell']:.2f} | "
                           f"{x['and_eq_antecedent']:.2f} | {x['dep_median']:.2f} | {x['dep_share_ge_half']:.2f} ({x['dep_n']}) |")
    out.append("\n| type | model | arm | form | p(A and B)≤0.05 | p(A)+p(B)≤1.01 | D median | D≥0.5 (n) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for name, j in sorted(S["jobs"].items()):
        if "mc_excl" not in j["by_type"]:
            continue
        r = j["by_type"]["mc_excl"]["respect"]
        for k in ("raw", "projected"):
            x = r[k]
            out.append(f"| mc_excl | {SHORT[j['model']]} | {j['arm']} | {k} | {x['and_zero']:.2f} | {x['sum_le_1']:.2f} | "
                       f"{x['dep_median']:.2f} | {x['dep_share_ge_half']:.2f} ({x['dep_n']}) |")
    out.append("\n### control pairs: projected log odds ratio\n")
    out.append("| model | arm | median | share abs > 1 |")
    out.append("|---|---|---|---|")
    for name, j in sorted(S["jobs"].items()):
        if "control" not in j["by_type"]:
            continue
        x = j["by_type"]["control"]["log_odds_ratio"]
        out.append(f"| {SHORT[j['model']]} | {j['arm']} | {x['median']:+.2f} | {x['share_abs_gt_1']:.2f} |")
    out.append("\n### pairs where the market violates the relation at t (joint log loss)\n")
    out.append("| type | model | arm | pairs | market indep | market oracle | raw | projected | hybrid | hybrid_and3 | projected − indep [CI] |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in ("threshold", "date", "mc_excl"):
        for name, j in sorted(S["jobs"].items()):
            v = j["by_type"].get(t, {}).get("market_violation_subset")
            if v:
                out.append(f"| {t} | {SHORT[j['model']]} | {j['arm']} | {v['pairs']} | {v['market_indep']['joint']['log_loss']:.3f} | "
                           f"{v['market_oracle']['joint']['log_loss']:.3f} | {v['raw']['joint']['log_loss']:.3f} | "
                           f"{v['projected']['joint']['log_loss']:.3f} | {v['hybrid']['joint']['log_loss']:.3f} | {v['hybrid_and3']['joint']['log_loss']:.3f} | "
                           f"{ci(v['projected_vs_market_indep']['joint'])} |")
    out.append("\n### projected − raw (joint)\n")
    out.append("| type | model | arm | Δ [CI] |")
    out.append("|---|---|---|---|")
    for t in TYPES:
        for name, j in sorted(S["jobs"].items()):
            if t in j["by_type"]:
                out.append(f"| {t} | {SHORT[j['model']]} | {j['arm']} | {ci(j['by_type'][t]['projected_vs_raw']['joint'])} |")
    out.append("\n### paired comparisons, joint forms (x − y on common pairs; negative = x better)\n")
    out.append("| x | y | type | pairs | raw | projected | hybrid | hybrid_and3 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for c in S.get("comparisons", []):
        for t, v in c["by_type"].items():
            out.append(f"| {c['x']} | {c['y']} | {t} | {v['pairs']} | " + " | ".join(ci(v[m]) for m in ("raw", "projected", "hybrid", "hybrid_and3")) + " |")
    out.append("\n### jobs\n")
    out.append("| model | arm | calls | missing | cost | served |")
    out.append("|---|---|---|---|---|---|")
    for name, j in sorted(S["jobs"].items()):
        out.append(f"| {j['model']} | {j['arm']} | {j['calls']} | {j['missing']} | ${j['cost_usd']:.2f} | {j['served_models']} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
