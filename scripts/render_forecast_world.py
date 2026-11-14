"""Render results/explore/forecast-world-20261022/{summary.json, figure-data.json, README.md} from cells/*.json.

    python scripts/render_forecast_world.py results/explore/forecast-world-20261022
"""
import json
import sys
from pathlib import Path

import numpy as np

BASE = ["bayes", "bayes_oracle_keys", "prior", "single_forecaster", "mean_prob", "mean_logodds", "median",
        "sum_logodds", "extremized_mean_logodds"]
ARMS = ["nokey", "key0", "key0.5", "key1", "key3", "oracle"]
LABEL = {"bayes": "(A) full-info Bayes", "bayes_oracle_keys": "(B) Bayes given forecasts + exact source sets (GLS)",
         "prior": "prior (0.5)", "single_forecaster": "one forecaster", "mean_prob": "mean of probabilities",
         "mean_logodds": "mean of log-odds", "median": "median", "sum_logodds": "sum of log-odds (naive Bayes)",
         "extremized_mean_logodds": "extremized mean log-odds (factor fit on train)",
         "attn_skey0": "attention + standardized key (own projection), sigma 0",
         "attn_skey1": "attention + standardized key (own projection), sigma 1",
         "attn_skey3": "attention + standardized key (own projection), sigma 3",
         "attn_nokey": "attention, forecasts only", "attn_key0": "attention + key, sigma 0",
         "attn_key0.5": "attention + key, sigma 0.5", "attn_key1": "attention + key, sigma 1",
         "attn_key3": "attention + key, sigma 3", "attn_oracle": "attention + exact source sets (multi-hot)"}


LEARNED = ["fc", "fckey", "fckeyv", "wide", "fairbar", "bar", "fckey_rand", "wide_rand"]
LLABEL = {"fc": "learned forecasters: forecasts only", "fckey": "learned: forecasts + learned key",
          "wide": "learned: wide free-form message (64)",
          "fckeyv": "learned: forecasts + learned key, readout matched to wide",
          "fairbar": "fair bar: one transformer over all forecaster x source readings (duplicates kept)",
          "bar": "union bar: one transformer over each distinct source once (handed deduplication)",
          "fckey_rand": "learned: forecasts + key, random descriptions (control)",
          "wide_rand": "learned: wide message, random descriptions (control)"}


def merge(r, s):
    """Merge part s (other learned arms of the same cell) into r; the shared baselines must agree exactly."""
    for b in BASE:
        if b not in r["methods"]:
            continue
        if abs(r["methods"][b]["kl"]["mean"] - s["methods"][b]["kl"]["mean"]) > 1e-6:   # GPU rounding only
            raise ValueError(f"parts of cell {r['extra']['cell']} disagree on {b}")
    r["methods"].update(s["methods"])
    r["controls"].update(s["controls"])
    r["learned_arms"] += s["learned_arms"]
    for k in ("seed_kl", "paired_vs_refs", "training", "local_forecast_kl_vs_exact"):
        if k in s["extra"]:
            r["extra"][k].update(s["extra"][k])
    r["extra"]["heldout_test"]["methods"].update(s["extra"]["heldout_test"]["methods"])
    r.setdefault("parts", [r["run_id"]]).append(s["run_id"])
    r.setdefault("commits", {r["harness"]["git_commit"]}).add(s["harness"]["git_commit"])


def load(d, sub="cells", prefix="cell_"):
    cells = {}
    for p in sorted((Path(d) / sub).glob(prefix + "*.json")):
        if p.name.endswith("_smoke.json"):
            continue
        r = json.loads(p.read_text())
        c = r["extra"]["cell"]
        key = (c["alpha"], c["N"], c["k"])
        if key in cells:
            merge(cells[key], r)
        else:
            cells[key] = r
    return cells


def row(r, held=False):
    """method -> {kl, kl_sd (seed sd, learned only), brier, seeds}; learned arms averaged over seeds."""
    meth = r["extra"]["heldout_test"]["methods"] if held else r["methods"]
    out = {}
    for b in BASE:
        if b in meth:
            out[b] = {"kl": meth[b]["kl"]["mean"], "ci": [meth[b]["kl"]["ci_low"], meth[b]["kl"]["ci_high"]],
                      "brier": meth[b]["brier"]["mean"], "kl_sd": 0.0, "seeds": 1}
    for a in ARMS:
        names = [m for m in meth if m.startswith(f"attn_{a}/s") and ":" not in m]
        if not names:
            continue
        kls = [meth[m]["kl"]["mean"] for m in names]
        out[f"attn_{a}"] = {"kl": float(np.mean(kls)), "kl_sd": float(np.std(kls, ddof=1)) if len(kls) > 1 else 0.0,
                            "brier": float(np.mean([meth[m]["brier"]["mean"] for m in names])),
                            "per_seed": kls, "seeds": len(kls),
                            "ci": [float(np.mean([meth[m]["kl"]["ci_low"] for m in names])),
                                   float(np.mean([meth[m]["kl"]["ci_high"] for m in names]))]}
    return out


def row_learned(r, held=False):
    meth = r["extra"]["heldout_test"]["methods"] if held else r["methods"]
    out = {}
    for a in LEARNED:
        names = [m for m in meth if m.startswith(f"learned_{a}/s") and ":" not in m]
        if names:
            kls = [meth[m]["kl"]["mean"] for m in names]
            loc = [r["extra"]["local_forecast_kl_vs_exact"][m]["heldout" if held else "test"] for m in names]
            out[a] = {"kl": float(np.mean(kls)), "kl_sd": float(np.std(kls, ddof=1)) if len(kls) > 1 else 0.0,
                      "per_seed": kls, "brier": float(np.mean([meth[m]["brier"]["mean"] for m in names])),
                      "local_kl": None if loc[0] is None else float(np.mean(loc))}
    return out


def fmt(v, sd=None):
    return f"{v:.4f}" + (f" ± {sd:.4f}" if sd else "")


def main(d):
    d = Path(d)
    cells = load(d)
    summary = {"cells": {}}
    for key, r in cells.items():
        summary["cells"][f"a{key[0]:g}_N{key[1]}_k{key[2]}"] = {
            "cell": r["extra"]["cell"], "top3_share": r["extra"]["top3_share"],
            "distinct_sources_mean": r["extra"]["distinct_sources_mean"],
            "extremize_factor": r["extra"]["extremize_factor"], "test": row(r), "heldout_test": row(r, True),
            "heldout_source_share": r["extra"]["heldout_test"]["heldout_source_share"],
            "controls": {m: {"trained": r["methods"][m]["kl"]["mean"],
                             "random_network": r["methods"][c["random_network"]]["kl"]["mean"],
                             "untrained_init": r["methods"][c["untrained_init"]]["kl"]["mean"]}
                         for m, c in r["controls"].items()},
            "harness": r["harness"], "oracle_check": r["oracle_check"], "n_test": r["splits"]["eval"]["n"],
            "train_cfg": r["extra"]["train_cfg"]}
    (d / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    def curve(keys, var):
        xs = [k[var] for k in keys]
        methods = {}
        for k in keys:
            for m, v in row(cells[k]).items():
                methods.setdefault(m, {"kl": [], "kl_sd": [], "ci": [], "brier": []})
                for f in ("kl", "kl_sd", "ci", "brier"):
                    methods[m][f].append(v[f])
        out = {"x": xs, "methods": methods}
        if var == 0:
            out["top3_share"] = [cells[k]["extra"]["top3_share"] for k in keys]
            out["distinct_sources_mean"] = [cells[k]["extra"]["distinct_sources_mean"] for k in keys]
        return out
    ov = sorted(k for k in cells if k[1] == 10 and k[2] == 3)
    nk = sorted(k for k in cells if k[0] == 1.0 and k[2] == 3)
    kk = sorted(k for k in cells if k[0] == 1.0 and k[1] == 10)
    fig = {"metric": "mean per-question KL(A || method), nats; A = full-info Bayes; normalize by methods.prior.kl "
                     "for the share of the prior's KL left",
           "overlap_curve_alpha": curve(ov, 0), "N_curve": curve(nk, 1), "k_curve": curve(kk, 2),
           "labels": LABEL, "labels_learned": LLABEL}
    if (1.0, 10, 3) in cells:
        fig["medium_cell"] = {"test": row(cells[(1.0, 10, 3)]), "heldout_test": row(cells[(1.0, 10, 3)], True)}
    (d / "figure-data.json").write_text(json.dumps(fig, indent=1) + "\n")

    L = ["# ForecastWorld: forecasts plus source keys, aggregated by attention (2026-10-22)", "",
         "Stage 2 of the Deger plan without LLMs: forecasters output calibrated forecasts on all 20 questions plus a key "
         "describing which news-like sources they read; an aggregator combines them. Everything has an exact Bayes "
         "ceiling. Environment and ceilings: `aggbench/forecast_world.py` (docstring). Runner: "
         "`scripts/forecast_world_run.py`. This README is generated by `scripts/render_forecast_world.py` from "
         "`cells/*.json`; plot data is `figure-data.json`.", "",
         "Metric: mean per-question KL from (A) full-info Bayes (nats; its expectation is the log-loss excess over A). "
         "Learned arms: mean ± sd over 3 training seeds, best-val checkpoint on a dev split, fresh episodes every step, "
         "test on 4096 test-split episodes. Held-out: forecasters may also read 10 sources never seen in training.", ""]
    if (1.0, 10, 3) in cells:
        r = cells[(1.0, 10, 3)]
        t, h = row(r), row(r, True)
        kb = t["bayes_oracle_keys"]["kl"]
        L += [f"## Medium cell: alpha = 1, N = 10, k = 3 (top-3 share {r['extra']['top3_share']:.2f})", "",
              "Gap closed = share of the distance from extremizing to (B) that a method closes.", "",
              "| method | KL vs A | gap to B | gap closed | Brier | KL vs A, held-out sources |",
              "|---|---:|---:|---:|---:|---:|"]
        ke = t["extremized_mean_logodds"]["kl"]
        for m, v in t.items():
            hv = h.get(m, {}).get("kl")
            L.append(f"| {LABEL.get(m, m)} | {fmt(v['kl'], v['kl_sd'])} | {v['kl'] - kb:+.4f} | "
                     f"{'—' if m.startswith('bayes') else format((ke - v['kl']) / (ke - kb), '+.0%')} | {v['brier']:.4f} | "
                     f"{fmt(hv, h[m]['kl_sd']) if hv is not None else '—'} |")
        L.append("")
    if ov:
        cols = [m for m in ["mean_prob", "mean_logodds", "median", "extremized_mean_logodds", "attn_nokey",
                            "attn_key0.5", "attn_key3", "attn_oracle", "bayes_oracle_keys", "prior"]
                if all(m in row(cells[k]) for k in ov)]
        L += ["## Overlap curve (N = 10, k = 3): KL vs A by Zipf exponent alpha", "",
              "top-3 share = share of forecasters who read at least one of the 3 most popular sources.", "",
              "| alpha | top-3 share | distinct sources | " + " | ".join(LABEL.get(c, c) for c in cols) + " |",
              "|---:|---:|---:|" + "---:|" * len(cols)]
        for k in ov:
            rw = row(cells[k])
            L.append(f"| {k[0]:g} | {cells[k]['extra']['top3_share']:.2f} | "
                     f"{cells[k]['extra']['distinct_sources_mean']:.1f} | " +
                     " | ".join(fmt(rw[c]['kl']) for c in cols) + " |")
        L.append("")
        L += ["Same curve normalized: KL vs A as a share of the prior's KL (the information there is to aggregate shrinks "
              "as overlap grows, because fewer distinct sources are read).", "",
              "| alpha | " + " | ".join(LABEL.get(c, c) for c in cols[:-1]) + " |", "|---:|" + "---:|" * (len(cols) - 1)]
        for k in ov:
            rw = row(cells[k])
            L.append(f"| {k[0]:g} | " + " | ".join(f"{rw[c]['kl'] / rw['prior']['kl']:.1%}" for c in cols[:-1]) + " |")
        L.append("")
        keycols = [f"attn_{a}" for a in ARMS if all(f"attn_{a}" in row(cells[k]) for k in ov)]
        L += ["## Key informativeness (sigma) by overlap: KL vs A, mean ± seed sd", "",
              "| alpha | " + " | ".join(LABEL[c] for c in keycols) + " |", "|---:|" + "---:|" * len(keycols)]
        for k in ov:
            rw = row(cells[k])
            L.append(f"| {k[0]:g} | " + " | ".join(fmt(rw[c]['kl'], rw[c]['kl_sd']) for c in keycols) + " |")
        L.append("")
        L += ["## Held-out sources by overlap: KL vs A", "",
              "| alpha | share of forecasters reading a held-out source | " +
              " | ".join(LABEL[c] for c in keycols + ['extremized_mean_logodds', 'bayes_oracle_keys'] if c in LABEL) + " |",
              "|---:|---:|" + "---:|" * (len(keycols) + 2)]
        for k in ov:
            rw = row(cells[k], True)
            L.append(f"| {k[0]:g} | {cells[k]['extra']['heldout_test']['heldout_source_share']:.2f} | " +
                     " | ".join(fmt(rw[c]['kl'], rw[c]['kl_sd']) for c in keycols +
                                ['extremized_mean_logodds', 'bayes_oracle_keys']) + " |")
        L.append("")
    for title, keys, idx in [("Number of forecasters N (alpha = 1, k = 3)", nk, 1),
                             ("Sources per forecaster k (alpha = 1, N = 10)", kk, 2)]:
        if len(keys) < 2:
            continue
        cols = [m for m in ["mean_logodds", "extremized_mean_logodds", "attn_nokey", "attn_key0", "attn_key1",
                            "attn_oracle", "bayes_oracle_keys", "prior"] if all(m in row(cells[k]) for k in keys)]
        L += [f"## {title}: KL vs A", "", "| value | " + " | ".join(LABEL.get(c, c) for c in cols) + " |",
              "|---:|" + "---:|" * len(cols)]
        for k in keys:
            rw = row(cells[k])
            L.append(f"| {k[idx]} | " + " | ".join(fmt(rw[c]['kl'], rw[c]['kl_sd']) for c in cols) + " |")
        L.append("")
    strong = load(d, "strong")
    if strong:
        sk = sorted(strong)
        cols = [c for c in ["attn_nokey", "attn_key0", "attn_oracle"] if all(c in row(strong[k]) for k in sk)]
        L += ["## Stronger learner on the overlap curve (full Q x Q readout, 24000 steps, one seed; N = 10, k = 3)", "",
              "Same environment and test sets as above; only the aggregator's readout and training budget differ. "
              "Extremizing, (B) and the prior come from the main cells.", "",
              "| alpha | top-3 share | extremized | " + " | ".join(LABEL[c] + " (strong)" for c in cols) +
              " | (B) | held-out: " + " | held-out: ".join(LABEL[c] + " (strong)" for c in cols) + " |",
              "|---:|---:|---:|" + "---:|" * (2 * len(cols) + 1)]
        fig["overlap_curve_strong"] = {"x": [k[0] for k in sk], "methods": {}, "heldout": {}}
        for k in sk:
            rw, hw = row(strong[k]), row(strong[k], True)
            base = row(cells[k]) if k in cells else {}
            for c in cols:
                fig["overlap_curve_strong"]["methods"].setdefault(c, []).append(rw[c]["kl"])
                fig["overlap_curve_strong"]["heldout"].setdefault(c, []).append(hw[c]["kl"])
            L.append(f"| {k[0]:g} | {strong[k]['extra']['top3_share']:.2f} | "
                     f"{fmt(base['extremized_mean_logodds']['kl']) if base else '—'} | " +
                     " | ".join(fmt(rw[c]['kl']) for c in cols) +
                     f" | {fmt(base['bayes_oracle_keys']['kl']) if base else '—'} | " +
                     " | ".join(fmt(hw[c]['kl']) for c in cols) + " |")
        L.append("")
        (d / "figure-data.json").write_text(json.dumps(fig, indent=1) + "\n")
    scale_lists, extra_commits = {}, set()
    for sub in ("scale", "strong", "learned_disjoint", "learned_probe"):
        for p in (d / sub).glob("*.json"):
            extra_commits.add(json.loads(p.read_text())["harness"]["git_commit"])
    for p in sorted((d / "scale").glob("cell_*.json")):
        r = json.loads(p.read_text())
        c = r["extra"]["train_cfg"]
        for m in r["learned_arms"]:
            scale_lists.setdefault((m.split("/")[0], c["width"], c["layers"], c.get("readout", "diag"), c["steps"]),
                                   []).append((r["methods"][m]["kl"]["mean"],
                                               r["extra"]["heldout_test"]["methods"][m]["kl"]["mean"]))
    scale = {k: (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])), len(v))
             for k, v in scale_lists.items()}
    if scale and (1.0, 10, 3) in cells:
        base = row(cells[(1.0, 10, 3)])
        L += ["## Budget and readout probe, medium cell (one seed each)", "",
              "Is the distance to (B) a budget limit? Baseline rows are the 3-seed main runs (width 128, 3 layers, "
              "diagonal readout, 6000 steps).", "",
              "Standardized key rows (gate 2026-10-23): the key is divided by its sd and gets its own input "
              "projection, so a noisy key cannot swamp the forecasts.", "",
              "| arm | width x layers | readout | steps | seeds | KL vs A | KL vs A, held-out sources |",
              "|---|---|---|---:|---:|---:|---:|"]
        for a in ["attn_nokey", "attn_key0", "attn_skey0", "attn_key1", "attn_skey1", "attn_key3", "attn_skey3",
                  "attn_oracle"]:
            if a in base:
                L.append(f"| {LABEL[a]} | 128 x 3 | diag | 6000 | {base[a]['seeds']} | "
                         f"{fmt(base[a]['kl'], base[a]['kl_sd'])} | {fmt(row(cells[(1.0, 10, 3)], True)[a]['kl'])} |")
            for (arm, w, nl, ro, st), (kl, hk, ns) in sorted(scale.items()):
                if arm == a:
                    L.append(f"| {LABEL[a]} | {w} x {nl} | {ro} | {st} | {ns} | {fmt(kl)} | {fmt(hk)} |")
        L.append("")
        fig["budget_probe_medium"] = [{"arm": k[0], "width": k[1], "layers": k[2], "readout": k[3], "steps": k[4],
                                       "kl": v[0], "kl_heldout": v[1], "seeds": v[2]} for k, v in sorted(scale.items())]
        (d / "figure-data.json").write_text(json.dumps(fig, indent=1) + "\n")
    learned = load(d, "learned", "learned_")
    if learned:
        lk = sorted(learned)
        L += ["## Learned forecasters (2026-10-22 redirect; N = 10, k = 3): KL vs A by overlap", "",
              "Forecasters are one shared network that reads the raw observations of its k sources (each source's "
              "8 readings mixed with 24 nuisance features by a fixed rotation, plus a 32-d source description: which "
              "factors it loads on and its reliability). The aggregator is trained jointly with them on outcomes "
              "only; 10000 steps of 256 fresh episodes, best-val, mean ± sd over seeds. Reference columns use the "
              "exact local posteriors as forecasts (main cells above, same test episodes).", ""]
        cols = [a for a in LEARNED if any(a in row_learned(learned[k]) for k in lk)]
        L += ["| alpha | top-3 share | " + " | ".join(LLABEL[a] for a in cols) +
              " | ref: exact forecasts, extremized | ref: exact forecasts, attention + exact key | (B) |",
              "|---:|---:|" + "---:|" * (len(cols) + 3)]
        fig["learned_overlap_curve"] = {"x": [k[0] for k in lk], "methods": {}, "heldout": {}, "local_kl": {}}
        for k in lk:
            rw = row_learned(learned[k])
            base = row(cells[k]) if k in cells else {}
            for a in cols:
                for f, src in (("methods", rw), ("heldout", row_learned(learned[k], True))):
                    fig["learned_overlap_curve"][f].setdefault(a, []).append(src.get(a, {}).get("kl"))
                fig["learned_overlap_curve"]["local_kl"].setdefault(a, []).append(rw.get(a, {}).get("local_kl"))
            L.append(f"| {k[0]:g} | {learned[k]['extra']['top3_share']:.2f} | " +
                     " | ".join(fmt(rw[a]['kl'], rw[a]['kl_sd']) if a in rw else "—" for a in cols) + " | " +
                     " | ".join(fmt(base[m]['kl']) if m in base else "—"
                                for m in ("extremized_mean_logodds", "attn_key0", "bayes_oracle_keys")) + " |")
        L.append("")
        L += ["Held-out sources (forecasters may also read 10 sources never seen in training; their descriptions "
              "are meaningful in the main arms and random in the controls):", "",
              "| alpha | " + " | ".join(LLABEL[a] for a in cols) + " | ref: exact forecasts, extremized (knows the new "
              "sources) | fair: extremized, ignoring new sources | fair: Bayes ignoring new sources |",
              "|---:|" + "---:|" * (len(cols) + 3)]
        fair = json.loads((d / "heldout_fair.json").read_text())["cells"] if (d / "heldout_fair.json").exists() else {}
        for k in lk:
            hw = row_learned(learned[k], True)
            base = row(cells[k], True) if k in cells else {}
            fc_ = fair.get(f"a{k[0]:g}", {})
            L.append(f"| {k[0]:g} | " + " | ".join(fmt(hw[a]['kl'], hw[a]['kl_sd']) if a in hw else "—" for a in cols)
                     + f" | {fmt(base['extremized_mean_logodds']['kl']) if base else '—'} | "
                     + " | ".join(fmt(fc_[m]['mean']) if m in fc_ else "—"
                                  for m in ("extremized_ignoring_heldout", "bayes_ignoring_heldout")) + " |")
        L += ["", "The source descriptions give coverage and reliability, not the signed loadings, and observations are "
              "rotated: even an exact Bayesian cannot read a new source from its description, so the fair comparators "
              "ignore new sources (`scripts/forecast_world_heldout_fair.py`, `heldout_fair.json`)."]
        L.append("")
        L += ["Quality of the learned forecasters' own forecasts: mean KL(exact local posterior || learned local "
              "forecast) per question, test set.", "", "| alpha | " +
              " | ".join(LLABEL[a] for a in cols if a.startswith("fc")) + " |",
              "|---:|" + "---:|" * len([a for a in cols if a.startswith("fc")])]
        for k in lk:
            rw = row_learned(learned[k])
            L.append(f"| {k[0]:g} | " + " | ".join(fmt(rw[a]['local_kl']) if a in rw and rw[a]['local_kl'] is not None
                                                      else "—" for a in cols if a.startswith("fc")) + " |")
        L.append("")
        (d / "figure-data.json").write_text(json.dumps(fig, indent=1) + "\n")
        for k in lk:
            r = learned[k]
            summary["cells"].setdefault(f"learned_a{k[0]:g}_N{k[1]}_k{k[2]}", {
                "test": row_learned(r), "heldout_test": row_learned(r, True), "harness": r["harness"],
                "oracle_check": r["oracle_check"], "train_cfg": r["extra"]["train_cfg"],
                "controls": {m: {"trained": r["methods"][m]["kl"]["mean"],
                                 "random_network": r["methods"][c["random_network"]]["kl"]["mean"],
                                 "untrained_init": r["methods"][c["untrained_init"]]["kl"]["mean"]}
                             for m, c in r["controls"].items()}})
        (d / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        cells_for_controls = list(cells.values()) + list(learned.values())
        disj = load(d, "learned_disjoint", "learned_disjoint_")
        if disj and (1.0, 10, 3) in learned:
            ov, dj = row_learned(learned[(1.0, 10, 3)]), row_learned(next(iter(disj.values())))
            r_dj = next(iter(disj.values()))
            L += ["## Fair bar and the no-overlap control (gate 2026-10-23; alpha = 1, N = 10, k = 3, 3 seeds)", "",
                  "Disjoint: the 30 sources read in an episode are all distinct (no overlap at all). If the gap "
                  "between message designs and the fair bar is about overlap, it should shrink there.", "",
                  "| arm | with overlap (main) | disjoint sets |", "|---|---:|---:|"]
            for a in ("fckey", "wide", "fairbar", "bar"):
                if a in ov or a in dj:
                    L.append(f"| {LLABEL[a]} | " + " | ".join(fmt(x[a]['kl'], x[a]['kl_sd']) if a in x else "—"
                                                            for x in (ov, dj)) + " |")
            bo = row(cells[(1.0, 10, 3)]) if (1.0, 10, 3) in cells else {}
            bd = r_dj["methods"]
            L.append(f"| ref: exact forecasts, extremized | {fmt(bo['extremized_mean_logodds']['kl']) if bo else '—'} | "
                     f"{fmt(bd['extremized_mean_logodds']['kl']['mean'])} |")
            L.append(f"| (B) | {fmt(bo['bayes_oracle_keys']['kl']) if bo else '—'} | "
                     f"{fmt(bd['bayes_oracle_keys']['kl']['mean'])} |")
            L.append(f"| prior (scale of the problem) | {fmt(bo['prior']['kl']) if bo else '—'} | "
                     f"{fmt(bd['prior']['kl']['mean'])} |")
            for a in ("wide", "fckey"):
                if a in ov and a in dj and "fairbar" in ov and "fairbar" in dj:
                    L.append(f"| ratio {a} / fair bar | {ov[a]['kl'] / ov['fairbar']['kl']:.2f} | "
                             f"{dj[a]['kl'] / dj['fairbar']['kl']:.2f} |")
            L.append("")
            fig["fairbar_disjoint"] = {"overlap": ov, "disjoint": dj,
                                       "disjoint_refs": {m: bd[m]["kl"]["mean"] for m in
                                                         ("extremized_mean_logodds", "bayes_oracle_keys", "prior")}}
            cells_for_controls += list(disj.values())
        probe = load(d, "learned_probe", "learned_")
        if probe:
            pr = row_learned(next(iter(probe.values())))
            steps = next(iter(probe.values()))["extra"]["train_cfg"]["steps"]
            L += [f"## Learned-forecaster budget probe (alpha = 1; {steps} steps, one seed, vs 10000-step 3-seed runs)",
                  "", "| arm | 10000 steps | " + f"{steps} steps |", "|---|---:|---:|"]
            base10 = row_learned(learned[(1.0, 10, 3)]) if (1.0, 10, 3) in learned else {}
            for a in LEARNED:
                if a in pr:
                    L.append(f"| {LLABEL[a]} | {fmt(base10[a]['kl'], base10[a]['kl_sd']) if a in base10 else '—'} | "
                             f"{fmt(pr[a]['kl'])} |")
            L.append("")
            fig["learned_budget_probe"] = {"steps": steps, "arms": pr}
            cells_for_controls += list(probe.values())
        (d / "figure-data.json").write_text(json.dumps(fig, indent=1) + "\n")
    else:
        cells_for_controls = list(cells.values())
    L += ["## Controls (KL vs A, mean over seeds and cells)", "",
          "| arm | trained | random network (scale-matched) | untrained init |", "|---|---:|---:|---:|"]
    agg = {}
    for r in cells_for_controls:
        for m, c in r["controls"].items():
            a = m.split("/")[0]
            agg.setdefault(a, []).append((r["methods"][m]["kl"]["mean"], r["methods"][c["random_network"]]["kl"]["mean"],
                                          r["methods"][c["untrained_init"]]["kl"]["mean"]))
    for a, v in agg.items():
        v = np.array(v)
        lab = LLABEL.get(a.replace("learned_", ""), a) if a.startswith("learned_") else LABEL.get(a, a)
        L.append(f"| {lab} | {v[:, 0].mean():.4f} | {v[:, 1].mean():.4f} | {v[:, 2].mean():.4f} |")
    L.append("")
    commits = sorted(set().union(*[r.get("commits", {r["harness"]["git_commit"]}) for r in cells_for_controls])
                     | extra_commits)
    L += [f"Harness commit(s): {', '.join(commits)}; oracle checks verified in every cell: "
          f"{all(r['oracle_check']['verified'] for r in cells.values())}. The exact-forecaster code path is the same "
          "at every listed commit (later commits only add the optional full readout and the learned-forecaster "
          "extras, drawn from separate random streams; `test_instance_unchanged_by_learned_extras`).", ""]
    (d / "README.md").write_text("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
