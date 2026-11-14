#!/usr/bin/env python3
"""Render the compact two-panel binding-plateau paper figure."""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator


COLORS = {"stalled": "#D55E00", "successful": "#0072B2"}
MARKERS = {1: "s", 4: "o"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact_label(run):
    return (run.replace("w16_", "16/").replace("w32_", "32/")
            .replace("_L2", "").replace("_mix4", "/mix")
            .replace("_ds4", "/ds4").replace("_", "/"))


def extract(reference_path, intervention_path):
    reference = json.loads(Path(reference_path).read_text())
    intervention = json.loads(Path(intervention_path).read_text())
    if intervention.get("schema") != "tiny-pfn-binding-intervention-v1":
        raise ValueError("unexpected intervention schema")
    if intervention.get("status") != "complete" or intervention.get("training_updates") != 0:
        raise ValueError("intervention is incomplete or includes training")
    selection = intervention["selection"]
    if selection["bayes_symmetric_kl_minimum"] != 0.3:
        raise ValueError("figure requires the preregistered SKL >= 0.3 subset")

    reference_keys = ["query_mixture", "binding_blind_mixture", "full_mixture"]
    panel_a = []
    for label, key in zip(("Q", "Q + U", "Q + S"), reference_keys):
        gap = reference["references"][key]["gap_to_full"]
        panel_a.append({"label": label, "reference_key": key,
                        "mean_nll_gap": float(gap["mean"]), "paired_se": float(gap["se"])})

    panel_b = []
    for model in intervention["models"]:
        status = model["historical_status"]
        group = "stalled" if status == "stalled" else "successful"
        selected = model["selected"]
        panel_b.append({
            "run": model["run"],
            "compact_label": compact_label(model["run"]),
            "historical_status": status,
            "group": group,
            "head_components": int(model["head_components"]),
            "selected_n": int(selected["n"]),
            "mean_response_rms_ratio": float(selected["mean_response_rms_ratio"]),
            "mean_response_percent_of_bayes": 100.0 * float(selected["mean_response_rms_ratio"]),
        })
    stalled = [row for row in panel_b if row["group"] == "stalled"]
    successful = [row for row in panel_b if row["group"] == "successful"]
    if len(stalled) != 6 or len(successful) != 5:
        raise ValueError(f"expected 6 stalled and 5 successful models, got {len(stalled)} and {len(successful)}")
    if {row["selected_n"] for row in panel_b} != {selection["selected_n"]}:
        raise ValueError("model subset sizes disagree with intervention selection metadata")
    return panel_a, panel_b, selection, reference, intervention


def render(panel_a, panel_b, selection, out_pdf, out_png):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8.2,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.1,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.10, 2.62),
                                     gridspec_kw={"width_ratios": [0.84, 1.16]})

    x_a = np.arange(3)
    means = np.array([row["mean_nll_gap"] for row in panel_a])
    errors = np.array([row["paired_se"] for row in panel_a])
    ax_a.errorbar(x_a, means, yerr=errors, fmt="o", color="#222222",
                  markersize=5.0, capsize=3.0, elinewidth=1.0, zorder=3)
    ax_a.vlines(x_a, 0, means, color="#B5B5B5", linewidth=1.2, zorder=1)
    ax_a.axhline(0, color="#666666", linewidth=0.7)
    ax_a.set_xticks(x_a, [row["label"] for row in panel_a])
    ax_a.set_ylabel("NLL gap to full Bayes (nats)")
    ax_a.set_title("A   Information retained", loc="left", fontweight="bold", pad=5)
    ax_a.set_ylim(-0.008, 0.215)
    ax_a.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
    for x, mean, se in zip(x_a, means, errors):
        label = "0" if mean == 0 else f"{mean:.3f} ± {se:.3f}"
        ax_a.annotate(label, (x, mean), xytext=(0, 8), textcoords="offset points",
                      ha="center", va="bottom", fontsize=7.1)
    ax_a.text(0.5, -0.24, "paired episode SE; exact mixtures", transform=ax_a.transAxes,
              ha="center", va="top", fontsize=7.0, color="#444444")

    jitters = {"stalled": np.linspace(-0.14, 0.14, 6),
               "successful": np.linspace(-0.12, 0.12, 5)}
    for group, x_center in (("stalled", 0.0), ("successful", 1.0)):
        rows = sorted((row for row in panel_b if row["group"] == group), key=lambda row: row["run"])
        for jitter, row in zip(jitters[group], rows):
            ax_b.scatter(x_center + jitter, row["mean_response_percent_of_bayes"],
                         s=37, marker=MARKERS[row["head_components"]],
                         facecolor=COLORS[group], edgecolor="white", linewidth=0.65, zorder=3)
    ax_b.set_yscale("log")
    ticks = [0.1, 0.3, 1, 3, 10, 30, 100]
    ax_b.yaxis.set_major_locator(FixedLocator(ticks))
    ax_b.set_yticklabels(["0.1%", "0.3%", "1%", "3%", "10%", "30%", "100%"])
    ax_b.set_ylim(0.09, 150)
    ax_b.set_xlim(-0.34, 1.34)
    ax_b.set_xticks([0, 1], ["Stalled\n6 models", "Successful\n5 models"])
    ax_b.set_ylabel("Predictive-mean RMS response / Bayes")
    ax_b.set_title("B   Response to broken source binding", loc="left", fontweight="bold", pad=5)
    ax_b.axhline(100, color="#666666", linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
    ax_b.text(1.32, 100, "Bayes", ha="right", va="bottom", fontsize=7.0, color="#444444")
    ax_b.text(0.02, 0.80,
              f"Bayes symmetric KL ≥ 0.3; {selection['selected_n']:,} fixed episodes",
              transform=ax_b.transAxes, ha="left", va="top", fontsize=7.0, color="#444444")
    legend = [
        Line2D([0], [0], marker="o", linestyle="none", color="none", markerfacecolor="#777777",
               markeredgecolor="white", markersize=6, label="4-component head"),
        Line2D([0], [0], marker="s", linestyle="none", color="none", markerfacecolor="#777777",
               markeredgecolor="white", markersize=5.5, label="Gaussian head"),
    ]
    ax_b.legend(handles=legend, loc="lower right", frameon=False, borderpad=0.1,
                handletextpad=0.35, labelspacing=0.25)

    for ax in (ax_a, ax_b):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", which="major", color="#E5E5E5", linewidth=0.55, zorder=0)
        ax.tick_params(width=0.65, length=3)

    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.27, wspace=0.36)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding-reference", type=Path, required=True)
    parser.add_argument("--binding-intervention", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel_a, panel_b, selection, _, _ = extract(args.binding_reference, args.binding_intervention)
    pdf = args.out_dir / "binding-summary.pdf"
    png = args.out_dir / "binding-summary.png"
    render(panel_a, panel_b, selection, pdf, png)
    manifest = {
        "schema": "binding-plateau-paper-figure-v1",
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "definition": {
            "panel_a": "exact-mixture Bayes NLL gaps to full Bayes; error bars are paired episode SE",
            "panel_b": "per-model predictive-mean RMS response divided by Bayes response on the fixed Bayes symmetric-KL >= 0.3 subset; no model-level aggregate",
            "successful_group": "acquired_sibling plus plateau_then_escape historical statuses",
        },
        "inputs": {
            str(args.binding_reference.resolve()): sha256(args.binding_reference),
            str(args.binding_intervention.resolve()): sha256(args.binding_intervention),
        },
        "selection": selection,
        "panel_a": panel_a,
        "panel_b": panel_b,
        "outputs": {str(pdf.resolve()): sha256(pdf), str(png.resolve()): sha256(png)},
    }
    manifest_path = args.out_dir / "binding-summary-data.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"pdf": str(pdf), "png": str(png), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
