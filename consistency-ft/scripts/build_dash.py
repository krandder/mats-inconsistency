#!/usr/bin/env python3
"""Build the self-contained multi-model consistency-ft dashboard."""

from __future__ import annotations

import html
import json
import math
import re
import sys
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "data/benchmark-runs"
OUTPUT = ROOT / "dash" / "index.html"
MOCK_MODEL = "mock/sha256"
MODEL_PREFIX = "accounts/fireworks/models/"
HEAT_COLORS = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

sys.path.insert(0, str(ROOT))
from common import CHECK_FIELDS, LOGPROB_PROMPT, VERBAL_PROMPT  # noqa: E402
from scripts.export_benchmark import build_comparison_context, comparison_context, load_snapshot, rebuild_database  # noqa: E402

ORIGINAL_NATIVE_CAMPAIGN = "native-20260912"


def esc(value):
    return html.escape(str(value), quote=True)


def display_name(name):
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)


def short_model(name):
    if name.startswith("typesafe/"):
        return name.removeprefix("typesafe/").replace("jev-", "JEV ").replace("@noul", " · TypeSafe")
    if name.startswith("native/"):
        model, _, effort = name.removeprefix("native/").partition("@")
        return f"{model} · {effort} (native)"
    return name.removeprefix(MODEL_PREFIX)


def read_jsonl(path):
    """Read a stable snapshot, tolerating a writer's incomplete final line."""
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        print(f"Warning: skipped {path.name}: {error}", file=sys.stderr)
        return []
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            print(f"Warning: skipped invalid {path.name}:{index + 1}: {error.msg}", file=sys.stderr)
    return rows


def load_runs():
    """Read the versioned, hash-verified snapshot; no results/ or mtimes needed."""
    return load_snapshot(SNAPSHOTS)[3]


def latest_runs(runs):
    latest = {}
    for run in runs:
        key = (run["kind"], run["model"])
        if key not in latest or (run["recorded_at"] or "", run["file"]) > (latest[key]["recorded_at"] or "", latest[key]["file"]):
            latest[key] = run
    return latest


def load_questions():
    questions = {}
    for check_type in CHECK_FIELDS:
        path = ROOT / "data" / "tuples" / f"{check_type}.jsonl"
        for row in read_jsonl(path):
            questions[row["id"]] = row["questions"]
    return questions


def rounded_bar(x, y, width, height, radius=4):
    end = x + max(width, 0.5)
    radius = min(radius, height / 2, max(width, 0.5) / 2)
    return (
        f"M{x:.1f},{y:.1f} H{end-radius:.1f} Q{end:.1f},{y:.1f} {end:.1f},{y+radius:.1f} "
        f"V{y+height-radius:.1f} Q{end:.1f},{y+height:.1f} {end-radius:.1f},{y+height:.1f} "
        f"H{x:.1f} Z"
    )


def nice_max(value):
    if value <= 0:
        return 1.0
    exponent = 10 ** math.floor(math.log10(value))
    scaled = value / exponent
    step = 1 if scaled <= 1 else 2 if scaled <= 2 else 5 if scaled <= 5 else 10
    return step * exponent


def ranking_chart(values, target=300, unranked_runs=None, *, source_coverage=None, cohort_label="reviewed"):
    unranked_runs = unranked_runs or {}
    width, left, right, top, step, bar_h, bottom = 900, 360, 95, 24, 34, 20, 42
    height = top + (len(values) + len(unranked_runs)) * step + bottom
    plot_w = width - left - right
    maximum = nice_max(max((value for _, value, _ in values), default=0) * 1.18)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Model ranking by mean violation">',
        f'<line class="axis" x1="{left}" y1="{top-8}" x2="{left}" y2="{height-bottom}"/>',
    ]
    for index in range(5):
        tick = maximum * index / 4
        x = left + plot_w * index / 4
        parts.extend(
            [
                f'<line class="grid" x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{height-bottom}"/>',
                f'<text class="tick" x="{x:.1f}" y="{height-15}" text-anchor="middle">{tick:.2f}</text>',
            ]
        )
    for index, (model, value, count) in enumerate(values):
        y = top + index * step
        bar_w = plot_w * value / maximum if maximum else 0
        partial = count < target
        tooltip = f"{model}\nMean violation: {value:.3f}\nn={count}" + ("\nPARTIAL RUN: only parsed tuples scored (survivorship)" if partial else "")
        label = short_model(model) + (f" ({count}/{target})" if partial else "")
        if source_coverage is not None:
            source = source_coverage[model]
            tooltip += f'\nOriginal coverage: {source["n"]}/{source["target_tuples"]}; displayed mean uses the shared set'
            if source["n"] < source["target_tuples"]:
                label += f' ({source["n"]}/{source["target_tuples"]} original)'
        parts.extend(
            [
                f'<text class="category{" partial" if partial else ""}" x="{left-12}" y="{y+bar_h/2+4:.1f}" text-anchor="end">{esc(label)}</text>',
                f'<path class="bar mark" d="{rounded_bar(left, y, bar_w, bar_h)}" data-tooltip="{esc(tooltip)}" tabindex="0"/>',
                f'<text class="value" x="{min(left+bar_w+8, width-right+8):.1f}" y="{y+bar_h/2+4:.1f}">{value:.3f}</text>',
            ]
        )
    for offset, (model, run) in enumerate(sorted(unranked_runs.items(), key=lambda item: short_model(item[0]))):
        y = top + (len(values) + offset) * step + bar_h / 2 + 4
        reason = f'unranked: {run["unranked_reason"]}'
        coverage = f'{run["n"]}/{run["target_tuples"]} {cohort_label} tuples'
        tooltip = f'{model}\n{reason}\n{coverage}\nOriginal coverage: {run["unscoped_n"]}/{run["original_target_tuples"]}'
        parts.append(f'<text class="category partial" x="{left-12}" y="{y:.1f}" text-anchor="end">{esc(short_model(model))} ({run["n"]}/{run["target_tuples"]} {cohort_label})</text>')
        parts.append(f'<text class="value mark" x="{left+8}" y="{y:.1f}" data-tooltip="{esc(tooltip)}" tabindex="0">{esc(reason)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def heatmap(model_order, check_runs, empty_label="No scored tuples"):
    checks = list(CHECK_FIELDS)
    left, top, cell_w, cell_h, right, bottom = 300, 102, 92, 42, 15, 15
    width = left + len(checks) * cell_w + right
    height = top + len(model_order) * cell_h + bottom
    aggregates = {}
    for model in model_order:
        by_type = defaultdict(list)
        for row in check_runs[model]["rows"]:
            by_type[row["check_type"]].append(float(row["violation"]))
        for check in checks:
            values = by_type[check]
            aggregates[(model, check)] = (sum(values) / len(values), len(values)) if values else (None, 0)
    maximum = max((value for value, _ in aggregates.values() if value is not None), default=0)
    parts = [
        f'<svg class="heatmap" viewBox="0 0 {width} {height}" role="img" aria-label="Mean violation by model and check type">'
    ]
    for column, check in enumerate(checks):
        x = left + column * cell_w + cell_w / 2
        parts.append(
            f'<text class="heat-header" transform="translate({x:.1f} {top-12}) rotate(-42)" text-anchor="start">{esc(display_name(check))}</text>'
        )
    for row_index, model in enumerate(model_order):
        y = top + row_index * cell_h
        reason = check_runs[model].get("unranked_reason")
        label = short_model(model) + (" · unranked" if reason else "")
        parts.append(
            f'<text class="heat-row" x="{left-12}" y="{y+cell_h/2+4:.1f}" text-anchor="end">{esc(label)}</text>'
        )
        for column, check in enumerate(checks):
            value, count = aggregates[(model, check)]
            x = left + column * cell_w
            if value is None:
                color, label, text_color = "var(--grid)", "—", "var(--text-primary)"
                tooltip = (f"{model}\nCheck: {display_name(check)}\nunranked: {reason}" if reason else
                           f"{model}\nCheck: {display_name(check)}\nUnavailable: {empty_label}\nn=0")
            else:
                color_index = round(6 * value / maximum) if maximum else 0
                color_index = max(0, min(6, color_index))
                color, label = HEAT_COLORS[color_index], f"{value:.2f}"
                text_color = "var(--text-primary)" if color_index <= 2 else "white"
                tooltip = f"{model}\nCheck: {display_name(check)}\nValue: {value:.3f}\nn={count}"
            parts.extend(
                [
                    f'<rect class="heat-cell mark" x="{x+1}" y="{y+1}" width="{cell_w-2}" height="{cell_h-2}" fill="{color}" data-tooltip="{esc(tooltip)}" tabindex="0"/>',
                    f'<text class="heat-value" x="{x+cell_w/2:.1f}" y="{y+cell_h/2+4:.1f}" text-anchor="middle" style="fill:{text_color}">{label}</text>',
                ]
            )
    parts.append("</svg>")
    return "".join(parts)


def pearson(rows):
    if len(rows) < 2:
        return None
    xs = [float(row["verbalized_probability"]) for row in rows]
    ys = [float(row["logprob_probability"]) for row in rows]
    x_mean, y_mean = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_var * y_var)
    return numerator / denominator if denominator else None


def scatter_chart(rows, model):
    size, left, right, top, bottom = 340, 48, 16, 18, 45
    plot = size - left - right
    bottom_y = top + plot
    parts = [
        f'<svg class="scatter" viewBox="0 0 {size} {size}" role="img" aria-label="Verbalized probability versus logprob probability for {esc(model)}">'
    ]
    for index in range(5):
        value = index / 4
        x = left + value * plot
        y = bottom_y - value * plot
        parts.extend(
            [
                f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom_y}"/>',
                f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot}" y2="{y:.1f}"/>',
                f'<text class="tick" x="{x:.1f}" y="{bottom_y+18}" text-anchor="middle">{value:.2f}</text>',
                f'<text class="tick" x="{left-7}" y="{y+4:.1f}" text-anchor="end">{value:.2f}</text>',
            ]
        )
    parts.extend(
        [
            f'<line class="reference" x1="{left}" y1="{bottom_y}" x2="{left+plot}" y2="{top}"/>',
            f'<line class="axis" x1="{left}" y1="{bottom_y}" x2="{left+plot}" y2="{bottom_y}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom_y}"/>',
            f'<text class="axis-title" x="{left+plot/2:.1f}" y="{size-7}" text-anchor="middle">Verbalized</text>',
            f'<text class="axis-title" transform="translate(12 {top+plot/2:.1f}) rotate(-90)" text-anchor="middle">Logprob</text>',
        ]
    )
    for row in rows:
        verbal = float(row["verbalized_probability"])
        logprob = float(row["logprob_probability"])
        x, y = left + verbal * plot, bottom_y - logprob * plot
        title = row["title"] if len(row["title"]) <= 80 else row["title"][:77] + "..."
        tooltip = f"{title}\nVerbalized: {verbal:.3f}\nLogprob: {logprob:.3f}"
        parts.append(
            f'<circle class="dot mark" cx="{x:.1f}" cy="{y:.1f}" r="4" data-tooltip="{esc(tooltip)}" tabindex="0"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def scoped_elicitation_runs(elicit_runs, propositions, allowed_ids):
    """Restrict archived paired readouts using their original IDs or exact titles."""
    allowed_titles = {
        proposition["text"].split("\n\nResolution criteria:", 1)[0]
        for proposition in propositions if proposition["id"] in allowed_ids
    }
    scoped = {}
    for model, run in elicit_runs.items():
        rows = [row for row in run["rows"]
                if row.get("proposition_id", row.get("id")) in allowed_ids
                or row.get("title") in allowed_titles]
        scoped[model] = {**run, "rows": rows, "n": len(rows),
                         "stat": sum(row["abs_diff"] for row in rows) / len(rows) if rows else None}
    return scoped


def elicitation_table(elicit_runs, model_order):
    rows = []
    ordered = model_order + sorted(set(elicit_runs) - set(model_order), key=short_model)
    for model in ordered:
        if model not in elicit_runs or not elicit_runs[model]["rows"]:
            unavailable = ("Logprobs unavailable from native client" if model.startswith("native/")
                           else "Direct Noul probability; token logprobs unavailable" if model.startswith("typesafe/")
                           else "No eligible paired elicitation data")
            rows.append(
                f"<tr><td>{esc(short_model(model))}</td>"
                f'<td colspan="3" class="empty">{unavailable}</td>'
                '<td class="numeric">0</td></tr>'
            )
            continue
        data = elicit_runs[model]["rows"]
        diffs = [float(row["abs_diff"]) for row in data]
        correlation = pearson(data)
        shown_correlation = f"{correlation:.3f}" if correlation is not None else "—"
        rows.append(
            "<tr>"
            f"<td>{esc(short_model(model))}</td>"
            f'<td class="numeric">{sum(diffs)/len(diffs):.3f}</td>'
            f'<td class="numeric">{100*sum(value > .2 for value in diffs)/len(diffs):.1f}%</td>'
            f'<td class="numeric">{shown_correlation}</td>'
            f'<td class="numeric">{len(data)}</td></tr>'
        )
    if not rows:
        return '<p class="empty">No elicitation runs for evaluated models.</p>'
    return (
        '<div class="table-scroll"><table><thead><tr><th>model</th>'
        '<th class="numeric">mean |verbal−logprob|</th><th class="numeric">disagreement &gt;0.2</th>'
        '<th class="numeric">Pearson</th><th class="numeric">paired n</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def sample_blocks(model_order, check_runs, questions):
    blocks = []
    for model in model_order:
        if check_runs[model].get("unranked_reason"):
            blocks.append('<details class="model-details">'
                          f'<summary>{esc(short_model(model))}<span>unranked: {esc(check_runs[model]["unranked_reason"])}</span></summary>'
                          '<p class="note">Available historical observations remain in the details and database.</p></details>')
            continue
        rows = sorted(
            check_runs[model]["rows"],
            key=lambda row: (-float(row["violation"]), row["check_type"], row["id"]),
        )[:3]
        top = f'{float(rows[0]["violation"]):.3f}' if rows else "unavailable"
        samples = []
        for row in rows:
            check_type = row["check_type"]
            tuple_questions = questions.get(row["id"], {})
            role_lines = []
            for role in CHECK_FIELDS[check_type]:
                question = tuple_questions.get(role, {})
                title = question.get("title", "Question text unavailable")
                if len(title) > 110:
                    title = title[:107] + "..."
                probability = float(row["probabilities"][role])
                role_lines.append(
                    f'<div class="role-line">{esc(role)}: {esc(title)} -&gt; <span class="prob">{probability:.2f}</span></div>'
                )
            samples.append(
                '<div class="sample">'
                f'<div class="sample-head">{esc(display_name(check_type))} · violation <span class="prob">{float(row["violation"]):.3f}</span></div>'
                f"{''.join(role_lines)}</div>"
            )
        blocks.append(
            '<details class="model-details">'
            f'<summary>{esc(short_model(model))}<span>top violation <span class="prob">{top}</span></span></summary>'
            f"{''.join(samples)}</details>"
        )
    return "".join(blocks)


def headline_runs(ranking, target=300):
    """A headline must cover its stated comparison target; an empty target has none."""
    return [row for row in ranking if row[2] >= target] if target else []


def model_ranking_table(ranking, check_runs, scoped_runs=None, common_target=None, *, corrected=True):
    """Show every scored model/effort in one table, retaining coverage and method."""
    names = {"gpt-5.6-luna": "GPT-5.6 Luna", "gpt-6-astra": "GPT-6 Astra",
             "grok-4.6": "Grok 4.6", "glm-5.3": "GLM-5.3"}
    rows = []
    table_ranking = list(ranking) + [(model, None, 0) for model in sorted(set(check_runs) - {row[0] for row in ranking}, key=short_model)]
    for model, value, count in table_ranking:
        run = check_runs[model]
        unranked_reason = run.get("unranked_reason")
        native = run["transport"] == "subscription-native-cli"
        raw_name = model.removeprefix("native/").partition("@")[0] if native else short_model(model)
        label = names.get(raw_name, raw_name)
        typed_api = run["transport"] == "typesafe-systemone-api"
        effort = "Not exposed" if typed_api else run["reasoning"] if native else "Not recorded*"
        if effort == "native-default":
            effort = "Provider default"
        scoped_run = scoped_runs[model] if scoped_runs is not None else run
        valid = f'{scoped_run["coverage"].get("valid_propositions", 0)}/{scoped_run["target_propositions"]}' if native or typed_api else "Not recorded"
        target = common_target if common_target is not None else run["target_tuples"]
        partial = count < target
        coverage = f'{count}/{target}'
        if partial:
            coverage += ' <span class="coverage-badge">partial</span>'
        if unranked_reason:
            coverage = "—"
        method = "TypeSafe Noul API" if typed_api else "Subscription" if native else "Fireworks API"
        recorded = (run["recorded_at"] or "Not recorded").split("T")[0]
        shown_value = f'{value:.4f}' if value is not None else "—"
        if unranked_reason:
            shown_value = f'— <span class="coverage-badge">unranked: {esc(unranked_reason)}</span>'
        scope_cells = ""
        if scoped_runs is not None:
            scope_mean = f'{scoped_run["stat"]:.4f}' if scoped_run["stat"] is not None else "—"
            scope_cells = (f'<td class="numeric">{scope_mean}</td>'
                           f'<td class="numeric">{scoped_run["n"]}/{scoped_run["target_tuples"]}</td>')
        rows.append(
            f'<tr data-model="{esc(model)}" class="{"native-row" if native else "historical-row"}">'
            f'<th scope="row" title="{esc(model)}">{esc(label)}</th>'
            f'<td>{esc(effort)}</td><td class="numeric score">{shown_value}</td>'
            f'<td class="numeric">{coverage}</td>{scope_cells}<td class="numeric">{esc(valid)}</td>'
            f'<td>{method}</td><td>{esc(recorded)}</td></tr>'
        )
    cohort_label = "Reviewed-scope" if corrected else "Original-cohort"
    cohort_adjective = "reviewed" if corrected else "original"
    caption = (f'Sorted by mean violation on the same shared tuples (lower is better). {cohort_label} means use each model’s available {cohort_adjective} tuples and can have different denominators.'
               if scoped_runs is not None else 'All measured models and reasoning settings, sorted by mean violation (lower is better).')
    score_heading = "Shared mean ↓" if scoped_runs is not None else "Mean violation ↓"
    count_heading = "Shared tuples" if scoped_runs is not None else "Scored tuples"
    scope_headings = (f'<th scope="col" class="numeric">{cohort_label} mean</th><th scope="col" class="numeric">{cohort_adjective.capitalize()} tuples</th>'
                      if scoped_runs is not None else '')
    return ('<div class="table-scroll ranking-scroll" tabindex="0" role="region" aria-label="Complete model ranking table">'
            f'<table class="ranking-table"><caption>{caption}</caption>'
            '<thead><tr><th scope="col">Model</th><th scope="col">Reasoning</th>'
            f'<th scope="col" class="numeric">{score_heading}</th><th scope="col" class="numeric">{count_heading}</th>{scope_headings}'
            '<th scope="col" class="numeric">Valid probabilities</th><th scope="col">Method</th><th scope="col">Run date (UTC)</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def coverage_table(check_runs, comparison_runs=None):
    rows = []
    ordered = sorted(check_runs.values(), key=lambda r: (r["transport"] != "subscription-native-cli", short_model(r["model"])))
    for run in ordered:
        native = run["transport"] == "subscription-native-cli"
        typed_api = run["transport"] == "typesafe-systemone-api"
        coverage = run["coverage"]
        valid = f'{coverage.get("valid_propositions", 0)}/{run["target_propositions"]}' if native or typed_api else "unrecorded"
        completed = f'{coverage.get("completed_propositions", 0)}/{run["target_propositions"]}' if native or typed_api else "unrecorded"
        errors = ", ".join(f"{kind}: {count}" for kind, count in sorted(coverage.get("errors", {}).items())) or "—"
        status = run["status"] if native or typed_api else ("historical full" if run["n"] >= run["target_tuples"] else "historical partial")
        common_cell = ''
        if comparison_runs is not None:
            comparison = comparison_runs[run["model"]]
            common_cell = (f'<td class="notes">unranked: {esc(comparison["unranked_reason"])}</td>' if comparison.get("unranked_reason") else
                           f'<td class="numeric">{comparison["n"]}</td>')
        rows.append(f'<tr><td>{esc(short_model(run["model"]))}</td><td>{"TypeSafe Noul API" if typed_api else "subscription CLI" if native else "Fireworks API"}</td>'
                    f'<td class="numeric">{run["n"]}/{run["target_tuples"]}</td>{common_cell}'
                    f'<td class="numeric">{valid}</td><td class="numeric">{completed}</td>'
                    f'<td>{esc(status)}</td><td class="notes">{esc(errors)}</td></tr>')
    common_heading = '<th class="numeric">shared ranking tuples</th>' if comparison_runs is not None else ''
    return ('<div class="table-scroll"><table><thead><tr><th>model / reasoning</th><th>method</th>'
            f'<th class="numeric">scored tuples</th>{common_heading}<th class="numeric">valid probabilities</th>'
            '<th class="numeric">completed propositions</th><th>status</th><th>missing readouts</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def scope_notes(context, *, corrected=True):
    """Explain the reviewed target separately from selection by shared availability."""
    target = len(context["target_tuple_ids"])
    common = len(context["common_tuple_ids"])
    settings = len(context["participating_checks"])
    limiters = [model for model, run in context["scoped_checks"].items()
                if model in context["participating_checks"] and run["n"] == common and common < target]
    archive_limiters = [model for model in limiters
                       if context["latest_checks"][model]["transport"] != "subscription-native-cli"]
    limiting_note = (f' Archived coverage for {esc(", ".join(short_model(model) for model in sorted(archive_limiters)))} '
                     f'is {common}/{target} and limits the shared set.' if archive_limiters else '')
    setting_label = f'all {settings} settings' if not context["unranked_models"] else f'{settings} participating settings'
    policy_note = ''
    if context["policy"] == "complete-historical-plus-native":
        policy_note = (' All native settings and API runs with complete original coverage participate; '
                       'incomplete historical settings remain visible and unranked.')
    if corrected:
        cohort_note = (f'from {target} reviewed tuples.{limiting_note}{policy_note} '
                       'Selection by shared availability can change the ordering; these results do not represent the full original 300-tuple benchmark. ')
        details_url = "/corrected-details.html#coverage"
    else:
        cohort_note = (f'from the original {target}-tuple benchmark.{limiting_note}{policy_note} '
                       'No topic or semantic subset is applied here. Missing original readouts reduce the shared set; '
                       'selection by shared availability can change the ordering. '
                       f'Native results use the original {ORIGINAL_NATIVE_CAMPAIGN} campaign (numeric-v1). ')
        details_url = "/details.html#coverage"
    note = (f'<strong>{common} shared tuples across {setting_label}</strong>, '
            + cohort_note + 'Native subscription runs and historical API runs differ in dates, prompts and reasoning settings. '
            f'<a href="{details_url}">Coverage and methods →</a>')
    common_types = {row["check_type"] for run in context["comparison_checks"].values() for row in run["rows"]}
    reviewed_types = {check for check, count in context["scope"]["check_type_counts"].items() if count}
    missing = []
    for check in CHECK_FIELDS:
        if check not in reviewed_types:
            missing.append(f'{esc(display_name(check))} unavailable: no eligible reviewed tuples.' if corrected else
                           f'{esc(display_name(check))} unavailable: no original source tuples.')
        elif check not in common_types:
            missing.append(f'{esc(display_name(check))} unavailable: no tuples scored by every participating setting.')
    return note, " ".join(missing)


def usage_table():
    path = SNAPSHOTS / "subscription-usage.json"
    if not path.exists():
        return '<p class="empty">Subscription measurements are pending. Token counts alone do not establish the fraction of a subscription limit used.</p>'
    report = json.loads(path.read_text())
    def shown(value):
        if value is None:
            return "unavailable"
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value)
    rows = []
    for row in report.get("providers", []):
        rows.append('<tr>' + ''.join(f'<td class="notes">{esc(shown(row.get(key)))}</td>'
                    for key in ("provider", "window", "before", "after", "used_delta", "unit", "notes")) + '</tr>')
    before = min((row["before_at"] for row in report.get("providers", []) if row.get("before_at")), default="unrecorded")
    after = max((row["after_at"] for row in report.get("providers", []) if row.get("after_at")), default=report.get("as_of", "unrecorded"))
    cumulative = report.get("cumulative_recorded_native_tokens_including_setup", {})
    cumulative_note = ""
    if cumulative:
        cumulative_note = (f'<p class="note">Across all archived campaigns: <strong>{cumulative["total_tokens"]:,} recorded native tokens</strong> '
                           f'and {report["cumulative_recorded_cli_launches"]:,} CLI launches, including setup and interruptions. '
                           f'{report["cumulative_interrupted_launches_without_receipts"]:,} interrupted launches have no native receipt; '
                           f'native usage is unknown for {shown(report.get("cumulative_native_usage_unknown_count"))} launches in total. '
                           'Unknown consumption is not zero. Imported observations are counted once. Quota percentages are not summed across campaigns or accounts.</p>')
    return (cumulative_note + f'<p class="note">Quota changes below cover {esc(report.get("campaign", "unrecorded"))}: '
            f'{esc(before)} to {esc(after)}. Account-wide deltas can include other fleet sessions; '
            'window resets and provider accounting prevent inference from tokens alone. '
            '<a href="/subscription-usage.json">Download complete usage report</a>.</p>'
            '<div class="table-scroll"><table><thead><tr><th>provider</th><th>window</th><th>before</th>'
            '<th>after</th><th>used delta</th><th>unit</th><th>measurement limits</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def render_page(title, subtitle, content, embedded_json):
    """Keep the overview and supplementary results in the same site shell."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
<style>
:root {{
  --surface-1:#fcfcfb; --page-bg:#f9f9f7; --text-primary:#0b0b0b;
  --text-secondary:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --series-1:#2a78d6; --border:rgba(11,11,11,.10);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --surface-1:#1a1a19; --page-bg:#0d0d0d; --text-primary:#ffffff;
    --text-secondary:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
    --series-1:#3987e5; --border:rgba(255,255,255,.10);
  }}
}}
:root[data-theme="dark"] {{
  --surface-1:#1a1a19; --page-bg:#0d0d0d; --text-primary:#ffffff;
  --text-secondary:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --series-1:#3987e5; --border:rgba(255,255,255,.10);
}}
* {{ box-sizing:border-box; font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }}
body {{ margin:0; background:var(--page-bg); color:var(--text-primary); }}
.page {{ max-width:1180px; margin:auto; padding:42px 24px 64px; }}
h1 {{ margin:0; font-size:32px; letter-spacing:-.03em; }}
.subtitle {{ margin:8px 0 14px; color:var(--text-secondary); }}
.pagelinks {{ margin:0 0 18px; font-size:14px; }}
.pagelinks a {{ color:var(--series-1, #2a78d6); text-decoration:none; }}
.pagelinks a:hover {{ text-decoration:underline; }}
.sitenav {{ display:flex; gap:10px; flex-wrap:wrap; margin:0 0 30px; }}
.sitenav a {{ font:500 12.5px/1 ui-monospace,monospace; letter-spacing:.04em; color:var(--series-1);
  border:1px solid var(--border); border-radius:999px; padding:7px 12px; text-decoration:none; background:var(--card-bg); }}
.sitenav a:hover {{ border-color:var(--series-1); }}
.stats {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:28px; }}
.stat,.card {{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px; }}
.stat {{ padding:20px; min-height:112px; }}
.stat-value {{ font-size:30px; line-height:1.15; font-variant-numeric:tabular-nums; }}
.model-value {{ font-size:20px; overflow-wrap:anywhere; }}
.stat-label {{ margin-top:9px; color:var(--text-secondary); font-size:13px; }}
.card {{ padding:20px; margin-bottom:16px; }}
h2 {{ margin:0 0 16px; font-size:17px; font-weight:600; }}
h3 {{ margin:0 0 8px; font-size:13px; font-weight:600; text-align:center; overflow-wrap:anywhere; }}
.chart-scroll,.table-scroll {{ overflow-x:auto; }}
.bar-chart svg {{ width:100%; min-width:620px; height:auto; display:block; }}
.heatmap {{ display:block; min-width:1145px; width:100%; height:auto; }}
svg text {{ fill:var(--text-secondary); }}
.grid {{ stroke:var(--grid); stroke-width:1; vector-effect:non-scaling-stroke; }}
.axis {{ stroke:var(--axis); stroke-width:1; vector-effect:non-scaling-stroke; }}
.reference {{ stroke:var(--axis); stroke-width:1; stroke-dasharray:5 5; vector-effect:non-scaling-stroke; }}
.tick {{ font-size:11px; fill:var(--muted); }}
.category,.value,.heat-row,.heat-header,.heat-value {{ font-size:12px; }}
.value,.heat-value {{ font-variant-numeric:tabular-nums; }}
.heat-cell {{ stroke:var(--surface-1); stroke-width:2; outline:none; }}
.axis-title {{ font-size:11px; fill:var(--text-secondary); }}
.bar,.dot {{ fill:var(--series-1); outline:none; }}
.dot {{ stroke:var(--surface-1); stroke-width:2; vector-effect:non-scaling-stroke; }}
.mark:focus {{ filter:brightness(1.12); }}
.scatter-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:20px; }}
.scatter {{ display:block; width:100%; max-width:340px; height:auto; margin:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th {{ color:var(--muted); font-weight:500; text-align:left; }}
th,td {{ padding:11px 10px; border-bottom:1px solid var(--border); white-space:nowrap; }}
tbody tr:last-child td {{ border-bottom:0; }}
.numeric {{ text-align:right; font-variant-numeric:tabular-nums; }}
.ranking-table caption {{ text-align:left; color:var(--text-secondary); padding:0 0 16px; font-size:13px; }}
.ranking-table th[scope="row"] {{ color:var(--text-primary); font-weight:600; }}
.ranking-table .native-row th[scope="row"] {{ color:var(--series-1); }}
.ranking-table .score {{ font-weight:650; }}
.ranking-table tbody tr:hover {{ background:var(--page-bg); }}
.coverage-badge {{ display:inline-block; margin-left:5px; font-size:11px; padding:2px 5px;
  border:1px solid var(--axis); border-radius:4px; color:var(--text-secondary); }}
.ranking-scroll:focus-visible {{ outline:2px solid var(--series-1); outline-offset:4px; }}
.ranking-intro {{ max-width:90ch; }}
.ranking-jumps {{ display:flex; flex-wrap:wrap; gap:16px; margin:16px 0 0; font-size:13px; }}
.ranking-jumps a {{ color:var(--series-1); }}
.filename {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }}
.empty {{ color:var(--muted); font-size:13px; }}
.note {{ color:var(--text-secondary); font-size:14px; line-height:1.5; }}
.notes {{ white-space:normal; min-width:140px; max-width:500px; overflow-wrap:anywhere; }}
.model-details {{ border-top:1px solid var(--border); }}
.model-details:last-child {{ border-bottom:1px solid var(--border); }}
summary {{ display:flex; justify-content:space-between; gap:20px; padding:14px 4px; cursor:pointer; font-weight:600; }}
summary span {{ color:var(--text-secondary); font-weight:400; }}
.method {{ margin-top:14px; border-top:1px solid var(--border); font-size:14px; }}
.category.partial {{ fill:var(--muted); font-style:italic; }}
.method summary {{ font-weight:500; color:var(--text-secondary); }}
.method pre {{ background:var(--surface-2, rgba(127,127,127,0.08)); border:1px solid var(--border); border-radius:8px; padding:12px 14px; overflow-x:auto; font-size:13px; line-height:1.5; }}
.method h3 {{ font-size:14px; margin:14px 0 6px; }}
.method p {{ color:var(--text-secondary); margin:6px 0; max-width:72ch; }}
.sample {{ padding:14px; margin:0 0 12px; background:var(--page-bg); border-radius:8px; }}
.sample-head {{ margin-bottom:8px; font-weight:700; font-variant-numeric:tabular-nums; }}
.role-line {{ margin:5px 0; color:var(--text-secondary); font-size:13px; line-height:1.4; }}
.prob {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-variant-numeric:tabular-nums; }}
#tooltip {{ position:fixed; z-index:10; display:none; pointer-events:none; max-width:340px; padding:8px 10px;
  color:var(--text-primary); background:var(--surface-1); border:1px solid var(--border); border-radius:6px;
  box-shadow:0 6px 22px rgba(0,0,0,.14); font-size:12px; line-height:1.45; white-space:pre-line; }}
@media (max-width:760px) {{
  .page {{ padding:28px 16px 48px; }} .stats {{ grid-template-columns:repeat(2,1fr); }}
}}
</style>
</head>
<body>
<main class="page">
  <header><h1>{esc(title)}</h1><p class="subtitle">{esc(subtitle)}</p><nav class="pagelinks"><a href="[removed]">Grafana progress panel</a> · <a href="/benchmark.sqlite">SQLite database</a> · <a href="/benchmark-manifest.json">Data manifest</a> · <a href="/plan.html">Plan DAG</a> · <a href="/proposal.html">Proposal</a> · <a href="https://github.com/futarchy-fi/consistency-ft">Repo</a></nav>
  <nav class="sitenav" aria-label="Site"><a href="/">original benchmark</a><a href="/details.html">original details</a><a href="/corrected.html">corrected comparison</a><a href="/corrected-details.html">corrected details</a><a href="/proposal.html">RLEF proposal</a><a href="/open-agency.html">open agency</a><a href="/compiled-rewards.html">compiled rewards</a><a href="/belief-training.html">explicit beliefs</a><a href="/wave1-results.html">wave one</a><a href="/self-trade-rl.html">self-trade</a><a href="/jev.html">Jev research</a><a href="/bridge.html">futarchy bridge</a><a href="/environments.html">belief gym</a><a href="https://github.com/futarchy-fi/consistency-ft/blob/main/docs/BEADS.md">research tasks</a><a href="https://github.com/futarchy-fi/consistency-ft">repo</a></nav></header>
{content}
</main>
<div id="tooltip" role="tooltip"></div>
<script type="application/json" id="dashboard-data">{embedded_json}</script>
<script>
const tip=document.querySelector('#tooltip');
const move=(event)=>{{
  const pad=14, box=tip.getBoundingClientRect();
  tip.style.left=Math.min(event.clientX+pad,innerWidth-box.width-pad)+'px';
  tip.style.top=Math.min(event.clientY+pad,innerHeight-box.height-pad)+'px';
}};
document.querySelectorAll('[data-tooltip]').forEach(mark=>{{
  mark.addEventListener('pointerenter',event=>{{tip.textContent=mark.dataset.tooltip;tip.style.display='block';move(event);}});
  mark.addEventListener('pointermove',move);
  mark.addEventListener('pointerleave',()=>tip.style.display='none');
  mark.addEventListener('focus',()=>{{tip.textContent=mark.dataset.tooltip;tip.style.display='block';const box=mark.getBoundingClientRect();move({{clientX:box.left+box.width/2,clientY:box.top+box.height/2}});}});
  mark.addEventListener('blur',()=>tip.style.display='none');
}});
</script>
</body>
</html>
"""


def original_benchmark_runs(runs):
    """The original view never advances to a later native campaign."""
    selected = []
    for run in runs:
        if run.get("transport") == "subscription-native-cli":
            if run["run_id"].split("/", 1)[0] != ORIGINAL_NATIVE_CAMPAIGN:
                continue
            if run.get("protocol_id", "numeric-v1") != "numeric-v1" or run.get("scope_id"):
                raise ValueError("original native campaign must use numeric-v1 and the full source cohort")
        selected.append(run)
    return selected


def original_comparison_context(propositions, tuples, runs):
    """Use the complete original catalog without editing the corrected scope."""
    scope_id = "original-benchmark"
    scope = {"scope_id": scope_id,
             "included_tuple_ids": [row["id"] for row in tuples],
             "included_proposition_ids": [row["id"] for row in propositions],
             "check_type_counts": {check: sum(row["check_type"] == check for row in tuples) for check in CHECK_FIELDS}}
    return build_comparison_context(scope_id, scope, runs, "complete-historical-plus-native")


def corrected_benchmark_runs(manifest, runs):
    """Keep the reviewed native cohort separate from original native history."""
    scope_id = manifest.get("comparison_scope_id")
    return [run for run in runs if run.get("transport") != "subscription-native-cli"
            or (scope_id is not None and run.get("scope_id") == scope_id)]


def render_dashboard_pair(manifest, propositions, runs, context, *, corrected):
    """Render one explicitly selected cohort and its supplementary details."""
    latest = latest_runs(runs)
    check_runs = {
        model: run for (kind, model), run in latest.items() if kind == "checks" and model != MOCK_MODEL
    }
    elicit_runs = {
        model: run for (kind, model), run in latest.items() if kind == "elicit" and model != MOCK_MODEL
    }
    details_url = "/corrected-details.html" if corrected else "/details.html"
    overview_url = "/corrected.html" if corrected else "/"
    target = 300
    scope_note = check_note = elicitation_note = ""
    if context:
        check_runs = context["comparison_checks"]
        target = len(context["common_tuple_ids"])
        if corrected:
            elicit_runs = scoped_elicitation_runs(elicit_runs, propositions, context["target_proposition_ids"])
            elicitation_note = (f'Historical pairs are restricted to the {len(context["target_proposition_ids"])} reviewed propositions; '
                                'paired counts can differ by model. ')
        else:
            elicitation_note = 'Original historical paired readouts are retained without the corrected scope filter. '
        scope_note, check_note = scope_notes(context, corrected=corrected)
    if not check_runs:
        raise SystemExit("No non-mock check configurations found")
    ranking = sorted(
        ((model, run["stat"], run["n"]) for model, run in check_runs.items() if run["n"] and run.get("comparison_eligible", True)),
        key=lambda item: (item[1], short_model(item[0])),
    )
    model_order = [model for model, _, _ in ranking]
    model_order += sorted(set(check_runs) - set(model_order), key=short_model)
    full = headline_runs(ranking, target=target)
    counts = [run["n"] for run in check_runs.values()]
    tuple_count = str(target) if context else (str(counts[0]) if len(set(counts)) == 1 else f"{min(counts)}–{max(counts)}")
    timestamp = max((run["recorded_at"] for run in check_runs.values() if run["recorded_at"]), default="unrecorded")
    stats = [
        (str(len(model_order)), f'settings shown · {len(context["participating_checks"])} participating' if context else "model / reasoning settings compared", ""),
        (tuple_count, f'shared tuples · {len(context["target_tuple_ids"])} {"reviewed" if corrected else "original"}' if context else "tuples per setting · target 300", ""),
        (short_model(full[-1][0]) if full else "unavailable", "highest shared-set mean" if context else "highest full-run mean", f"mean {full[-1][1]:.3f}" if full else "no complete comparison"),
        (short_model(full[0][0]) if full else "unavailable", "lowest shared-set mean" if context else "lowest full-run mean", f"mean {full[0][1]:.3f}" if full else "no complete comparison"),
    ]
    stats_html = "".join(
        '<div class="stat">'
        f'<div class="stat-value{(" model-value" if detail else "")}">{esc(value)}</div>'
        f'<div class="stat-label">{esc(label)}{f" · {esc(detail)}" if detail else ""}</div></div>'
        for value, label, detail in stats
    )

    readouts = elicitation_table(elicit_runs, model_order)
    questions = load_questions()
    samples = sample_blocks(model_order, check_runs, questions)
    scatter_models = [model for model in model_order if model in elicit_runs and elicit_runs[model]["rows"]]
    scatter_models += sorted((model for model in elicit_runs if model not in scatter_models and elicit_runs[model]["rows"]), key=short_model)
    scatter_html = ""
    if scatter_models and len(scatter_models) <= 4:
        multiples = "".join(
            f'<div class="small-multiple"><h3>{esc(short_model(model))}</h3>{scatter_chart(elicit_runs[model]["rows"], model)}</div>'
            for model in scatter_models
        )
        scatter_html = (
            '<article class="card"><h2>Verbalized vs logprob probability</h2>'
            f'<div class="scatter-grid">{multiples}</div></article>'
        )

    run_rows = []
    for run in sorted(runs, key=lambda item: (item["recorded_at"] or "", item["file"], item["model"]), reverse=True):
        stat = f'{run["stat"]:.3f}' if run["stat"] is not None else "—"
        run_rows.append(
            "<tr>"
            f'<td class="filename">{esc(run["source_file"])}</td><td>{run["kind"]}</td><td>{esc(run["model"])}</td>'
            f'<td class="numeric">{run["n"]}</td><td class="numeric">{stat}</td>'
            f'<td>{esc(run["recorded_at"] or "pending")}</td><td>{esc(run["timestamp_source"])}</td></tr>'
        )
    public_keys = ("run_id", "file", "kind", "model", "reasoning", "transport", "n", "stat", "recorded_at", "status", "coverage")
    def public_run(run):
        result = {key: run[key] for key in public_keys}
        result.update(protocol_id=run.get("protocol_id"), scope_id=run.get("scope_id"))
        if "comparison_eligible" in run:
            result.update(comparison_eligible=run["comparison_eligible"], unranked_reason=run["unranked_reason"])
            if not run["comparison_eligible"]:
                result["n"] = None
        return result
    embedded = {
        "view": "corrected" if corrected else "original",
        "native_campaigns": sorted({run["run_id"].split("/", 1)[0] for run in runs if run.get("transport") == "subscription-native-cli"}),
        "latest": [public_run(run) for run in
                   (list(check_runs.values()) + list(elicit_runs.values()) if context else latest.values())],
        "ranking": ranking,
        "runs": [public_run(run) for run in runs],
    }
    if context:
        embedded["comparison"] = {
            "scope_id": context["scope_id"], "policy": context["policy"], "cohort": "reviewed" if corrected else "original",
            "reviewed_target" if corrected else "original_target": len(context["target_tuple_ids"]),
            "shared_target": target, "shared_tuple_ids": sorted(context["common_tuple_ids"]),
            "models": [{"model": model, "shared_n": run["n"] if run["comparison_eligible"] else None,
                        "comparison_eligible": run["comparison_eligible"], "unranked_reason": run["unranked_reason"],
                        "reviewed_n" if corrected else "original_n": context["scoped_checks"][model]["n"]} for model, run in check_runs.items()],
        }
    embedded_json = json.dumps(embedded, separators=(",", ":")).replace("<", "\\u003c")
    native_ranking = [row for row in ranking if row[0].startswith("native/")]
    legacy_ranking = [row for row in ranking if check_runs[row[0]]["transport"] == "fireworks-api"]
    empty_chart = ('<p class="empty">No tuples are currently scored by every participating setting; '
                   f'<a href="{details_url}#coverage">see coverage</a>.</p>' if context else
                   '<p class="empty">No tuples scored yet; see coverage below.</p>')
    unranked_runs = ({model: {**context["scoped_checks"][model], "unranked_reason": reason}
                      for model, reason in context["unranked_models"].items()} if context else {})
    chart_options = {"source_coverage": context["scoped_checks"] if context and not corrected else None,
                     "cohort_label": "reviewed" if corrected else "original"}
    overview_chart = ranking_chart(ranking, target=target, unranked_runs=unranked_runs, **chart_options) if ranking or unranked_runs else empty_chart
    native_chart = ranking_chart(native_ranking, target=target, **chart_options) if native_ranking else empty_chart
    legacy_chart = ranking_chart(legacy_ranking, target=target, unranked_runs=unranked_runs, **chart_options) if legacy_ranking or unranked_runs else empty_chart
    ranking_note = scope_note or ('All measured model / reasoning settings are shown. Partial runs are labeled by scored tuples and excluded from full-run headlines. '
                                 'Native subscription and historical API settings differ; these scores measure coherence, not forecasting accuracy. '
                                 '<a href="/details.html">Coverage, methods, ranking table and subscription usage →</a>')
    heatmap_note = f'<p class="note">{check_note}</p>' if check_note else ''
    runs_note = ('<p class="note">Run summaries below retain each run’s own coverage. Their totals are not the shared-set ranking above. '
                 'The database contains both original and corrected campaigns; its <code>comparison_ranking</code> view is the corrected comparison. '
                 '<a href="/benchmark.sqlite">Download the complete database</a>.</p>')

    jev_runs = [run for run in check_runs.values() if run["transport"] == "typesafe-systemone-api"]
    jev_note = ('<p class="note"><strong>Jev uses a distinct probability channel:</strong> '
                'TypeSafe Noul returns a direct yes-probability for each question, independently, with its full resolution criteria. '
                'No shared question context, logical hints, prompt tuning, verbal answers or token logprobs. '
                'Reasoning controls and an explicit abstention field are not exposed. The same algebraic formulas and shared tuple IDs are used; '
                'this does not isolate model quality from elicitation method. '
                '<a href="/jev.html">Research and protocol →</a></p>') if jev_runs else ''
    jev_spotlight = ""
    for position, (model, value, count) in enumerate(ranking, 1):
        if model.startswith("typesafe/"):
            jev_spotlight = (f'<p class="note"><strong>{esc(short_model(model))} is included below: '
                             f'#{position} of {len(ranking)}, mean violation {value:.4f}</strong> '
                             f'on {count} shared tuples. <a href="/jev.html">Jev research →</a></p>')
            break
    home_content = f"""  <section class="stats" aria-label="Summary statistics">{stats_html}</section>
  <article class="card bar-chart" id="model-ranking"><h2>Model ranking</h2>{jev_spotlight}
    <div class="chart-scroll">{overview_chart}</div>
    <p class="note">{ranking_note}</p>{jev_note}
  </article>
  <article class="card" id="check-results"><h2>Mean violation by model × check type</h2><div class="chart-scroll">{heatmap(model_order, check_runs, empty_label="no shared scored tuples" if context else "no scored tuples")}</div>{heatmap_note}</article>
  <article class="card"><h2>Elicitation readouts</h2>{readouts}
  <p class="note">{elicitation_note}Paired statistics require both verbalized and True/False logprob readouts. Native subscription clients do not expose logprobs; their paired metrics are unavailable.</p>
  <details class="method"><summary>Methodology: exact prompts &amp; readout definitions</summary>
    <p>For the historical Fireworks elicitation runs only, each proposition (title + resolution criteria) is asked twice, at temperature 0,
    <code>reasoning_effort: "none"</code> (falling back to <code>"low"</code> where a model rejects "none"),
    <code>max_tokens: 2048</code>.</p>
    <h3>1 · Verbalized probability</h3>
    <pre>{html.escape(VERBAL_PROMPT.format(proposition="{proposition}"))}</pre>
    <p>The reply is parsed as a number in [0,1].</p>
    <h3>2 · True/False logprob probability</h3>
    <pre>{html.escape(LOGPROB_PROMPT.format(proposition="{proposition}"))}</pre>
    <p>Token logprobs (<code>top_logprobs: 5</code>) are scanned <em>from the end</em> of the
    completion (reasoning models mention true/false mid-analysis; the answer token is last).
    At the last position where both appear: P = exp(lp_True) / (exp(lp_True) + exp(lp_False)).</p>
    <p><strong>mean |verbal−logprob|</strong> = mean absolute gap between the two readouts over the
    same propositions; <strong>disagreement &gt;0.2</strong> = fraction with gap above 0.2;
    Pearson over the paired values. Historical consistency-check probabilities elsewhere on this page
    use readout 1 only; Jev uses its direct Noul probability.</p>
  </details></article>
  {scatter_html}
  <article class="card"><h2>Worst violations by model</h2>{samples}</article>
  <article class="card"><h2>Runs</h2>{runs_note}<div class="table-scroll"><table>
    <thead><tr><th>file</th><th>kind</th><th>model</th><th class="numeric">n</th><th class="numeric">key stat</th><th>recorded time (UTC)</th><th>time evidence</th></tr></thead>
    <tbody>{''.join(run_rows)}</tbody>
  </table></div></article>
"""
    details_intro = (scope_note if context else 'Each row shows its own valid tuples; partial coverage and different evaluation methods limit direct comparisons.')
    coverage_note = (f'Original target: {len(context["target_tuple_ids"])} tuples across '
                     f'{sum(bool(n) for n in context["scope"]["check_type_counts"].values())} checks, '
                     f'using {len(context["target_proposition_ids"])} distinct propositions. '
                     f'The ranking compares the same {target} tuples scored by all {len(context["participating_checks"])} participating settings. '
                     'Each model’s original coverage and mean over its own available tuples are preserved below; those means can have different denominators. '
                     'Missing original readouts reduce the shared set. A shared-set headline does not imply completion of the full original benchmark. '
                     f'Native results are fixed to {ORIGINAL_NATIVE_CAMPAIGN} (numeric-v1). '
                     '<a href="/corrected.html">The corrected subset and rerun are reported separately</a>.')
    coverage_runs = check_runs
    if context:
        coverage_runs = context["scoped_checks"]
    if corrected:
        coverage_note = (f'Reviewed target: {len(context["target_tuple_ids"])} whole tuples across '
                         f'{sum(bool(n) for n in context["scope"]["check_type_counts"].values())} checks, '
                         f'using {len(context["target_proposition_ids"])} distinct propositions. '
                         'The review applied the documented topic and semantic criteria to source texts, without consulting model responses. '
                         f'The ranking uses only the {target} tuples scored by every one of the {len(context["participating_checks"])} participating settings. '
                         'The table below retains each model’s coverage on the full reviewed target, including questions outside that shared set. '
                         'Missing readouts and incomplete archived coverage can select easier questions; a shared denominator does not remove this selection effect. '
                         '<a href="/reviewed-scope.json">Reviewed scope and exclusion reasons</a>. '
                         '<a href="https://github.com/futarchy-fi/consistency-ft/blob/main/docs/native-evaluation.md">Native protocol and audit</a>.')
    subgroup_note = ' Both charts retain the same shared tuples across all participating settings.' if context else ''
    details_content = f"""  <article class="card" id="model-ranking"><h2>Model ranking · all {len(model_order)} model / reasoning settings</h2>
    <p class="note ranking-intro">GLM-5.3, Grok 4.6, GPT-6 Astra, and GPT-5.6 Luna at <strong>none, low, medium, high, xhigh, and max</strong> are included alongside every historical model.
    Scores measure probability consistency, not accuracy. {details_intro}</p>
    {jev_note}
    {model_ranking_table(ranking, check_runs, scoped_runs=context["scoped_checks"] if context else None, common_target=target if context else None, corrected=corrected)}
    <p class="note">*Historical API runs requested none, with low as a fallback; the setting actually used was not recorded per proposition.
    Subscription runs were measured in September and historical API runs in August, with different native prompts, reasoning settings, and output limits.</p>
    <nav class="ranking-jumps" aria-label="Detailed results"><a href="#native-ranking">Subscription charts</a><a href="#historical-ranking">Historical charts</a><a href="{overview_url}#check-results">Per-check results</a><a href="#coverage">Full coverage details</a><a href="#subscription-usage">Subscription usage</a></nav>
  </article>
  <article class="card" id="coverage"><h2>Coverage and comparison limits</h2>
    <p class="note">{coverage_note}
    Unanswered probabilities remain missing, never zero. Low violation measures coherence, not forecasting accuracy or calibration.</p>
    {coverage_table(coverage_runs, comparison_runs=check_runs if context else None)}
  </article>
  <article class="card bar-chart" id="native-ranking"><h2>Model ranking · native subscriptions</h2>
    <p class="note">Fresh CLI sessions, one verbalized probability per proposition, tools and browsing disabled. Each reasoning level is a separate setting.
    No logprobs are available. Native agent scaffolds, reasoning settings and output limits differ by provider; this is not a matched API experiment.{subgroup_note}</p>
    <div class="chart-scroll">{native_chart}</div>
  </article>
  <article class="card bar-chart" id="historical-ranking"><h2>Model ranking · historical Fireworks API</h2>
    <p class="note">Original August runs, using verbalized probabilities at temperature 0. Requested reasoning was none, with low as a fallback; actual fallback choices were not recorded per proposition.
    Native and API results use the same tuple target and algebraic violation formulas, with different elicitation settings.{subgroup_note}</p>
    <div class="chart-scroll">{legacy_chart}</div>
  </article>
  <article class="card" id="subscription-usage"><h2>Subscription usage</h2>{usage_table()}</article>

"""
    view_label = "corrected comparison" if corrected else "original benchmark"
    page = render_page(
        f"consistency-ft · {view_label}",
        f"{view_label} · shared-cohort model comparison — selected runs as of {timestamp}",
        home_content, embedded_json,
    )
    details_page = render_page(
        f"consistency-ft · {view_label} details",
        f"{view_label} · full ranking, coverage, methods and subscription usage — selected runs as of {timestamp}",
        details_content, embedded_json,
    )
    return page, details_page


def main():
    manifest, propositions, tuples, runs = load_snapshot(SNAPSHOTS)
    original_runs = original_benchmark_runs(runs)
    original_context = original_comparison_context(propositions, tuples, original_runs)
    page, details_page = render_dashboard_pair(manifest, propositions, original_runs, original_context, corrected=False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(page)
    details_output = OUTPUT.with_name("details.html")
    details_output.write_text(details_page)
    corrected_runs = corrected_benchmark_runs(manifest, runs)
    context = comparison_context(manifest, corrected_runs, SNAPSHOTS)
    if context:
        corrected_page, corrected_details = render_dashboard_pair(manifest, propositions, corrected_runs, context, corrected=True)
        scope_file = manifest["scopes"][context["scope_id"]]["file"]
        shutil.copyfile(SNAPSHOTS / scope_file, OUTPUT.parent / "reviewed-scope.json")
    else:
        pending = '<article class="card"><h2>Corrected comparison pending</h2><p>No corrected cohort has been published. <a href="/">View the original benchmark</a>.</p></article>'
        corrected_page = corrected_details = render_page("consistency-ft · corrected comparison", "Corrected results are not available in this snapshot.", pending, '{"view":"corrected","status":"pending"}')
    OUTPUT.with_name("corrected.html").write_text(corrected_page)
    OUTPUT.with_name("corrected-details.html").write_text(corrected_details)
    rebuild_database(SNAPSHOTS, OUTPUT.parent / "benchmark.sqlite")
    shutil.copyfile(SNAPSHOTS / "manifest.json", OUTPUT.parent / "benchmark-manifest.json")
    if (SNAPSHOTS / "subscription-usage.json").exists():
        shutil.copyfile(SNAPSHOTS / "subscription-usage.json", OUTPUT.parent / "subscription-usage.json")
    open_agency = ROOT / "site" / "open-agency.html"
    if open_agency.exists():
        shutil.copyfile(open_agency, OUTPUT.parent / "open-agency.html")
    compiled_rewards = ROOT / "site" / "compiled-rewards.html"
    if compiled_rewards.exists():
        shutil.copyfile(compiled_rewards, OUTPUT.parent / "compiled-rewards.html")
    belief_training = ROOT / "site" / "belief-training.html"
    if belief_training.exists():
        shutil.copyfile(belief_training, OUTPUT.parent / "belief-training.html")
    jev = ROOT / "site" / "jev.html"
    if jev.exists():
        shutil.copyfile(jev, OUTPUT.parent / "jev.html")
    for name in ("wave1-results.html", "self-trade-rl.html"):
        page = ROOT / "site" / name
        if page.exists():
            shutil.copyfile(page, OUTPUT.parent / name)
    print(f"Wrote original and corrected overview/details pages in {OUTPUT.parent}; database retains {len(runs)} runs")


if __name__ == "__main__":
    main()
