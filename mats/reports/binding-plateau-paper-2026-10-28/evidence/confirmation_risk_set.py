#!/usr/bin/env python3
"""Post-review sensitivity: risk sets use only acquisition confirmed by landmark.

python confirmation_risk_set.py --baselines /tmp/astra-pfn-baselines.json \
    --analyzer /path/to/tiny_pfn_plateau_20261028/analyze.py \
    --out confirmation-risk-set.json

Read-only inputs; stdlib only; no training or core-analyzer edits.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baselines", type=Path, required=True)
    ap.add_argument("--analyzer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    saved = json.loads(args.baselines.read_text())
    spec = importlib.util.spec_from_file_location("registered_analysis", args.analyzer)
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    files = [args.baselines, args.analyzer, Path(__file__)]
    for root in saved["sources"]:
        files.extend(Path(root) / name for name in ("meta.json", "curve.json"))
    hashes = {str(path.resolve()): sha(path) for path in files}
    runs, _ = analysis.load_runs(saved["sources"])
    original = {(g["width"], g["condition"]): g for g in saved["early_predictors"]}
    groups = []
    for key, run in sorted(runs.items()):
        old_landmarks = {item["landmark"]: item for item in original[key]["landmarks"]}
        events = [run["per_seed"][seed]["acquisition"][str(analysis.PRIMARY_THRESHOLD)]
                  for seed in run["seeds"]]
        group = {"width": key[0], "condition": key[1], "landmarks": []}
        by_step = {row["step"]: row for row in run["curve"]}
        for landmark in analysis.LANDMARKS:
            if landmark not in by_step:
                continue
            eligible = [i for i, event in enumerate(events)
                        if event is None or event["detection_step"] > landmark]
            labels = [events[i] is not None for i in eligible]
            n_pos, n_neg = sum(labels), len(labels) - sum(labels)
            old = old_landmarks[landmark]
            item = {
                "landmark": landmark, "eligible": len(eligible),
                "later_confirmed": n_pos, "not_confirmed_by_horizon": n_neg,
                "status": "ok" if min(n_pos, n_neg) >= 5 else "insufficient_outcome_classes",
                "retrospective_onset_eligible": old["eligible"],
                "retained_pending_confirmation": [
                    {"seed": run["seeds"][i], **events[i]} for i in eligible
                    if events[i] is not None and events[i]["event_step"] <= landmark],
                "metrics": {},
            }
            vectors = analysis.metric_vectors(by_step[landmark], len(run["seeds"]))
            for name, direction in analysis.PRIMARY_PREDICTORS.items():
                metric = {"fixed_direction": direction,
                          "retrospective_onset_auc": old["metrics"][name].get("fixed_direction_auc"),
                          "retrospective_onset_bootstrap95": old["metrics"][name].get("stratified_seed_bootstrap95")}
                if item["status"] == "ok":
                    values = [vectors[name][i] for i in eligible]
                    auc = analysis.auc(values, labels)
                    metric.update(
                        confirmation_risk_set_auc=auc if direction == "higher" else 1 - auc,
                        stratified_seed_bootstrap95=analysis.bootstrap_auc(
                            values, labels, direction, f"{key[0]}/{key[1]}/{landmark}/{name}"))
                item["metrics"][name] = metric
            group["landmarks"].append(item)
        groups.append(group)
    if hashes != {str(path.resolve()): sha(path) for path in files}:
        raise RuntimeError("an input or analysis source changed during computation")
    result = {
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "status": "post-review operational risk-set sensitivity; not a replacement of the preregistered analysis",
        "definition": "include a seed iff no acquisition is confirmed at or before the landmark (detection_step > landmark, or none)",
        "outcome": "three-point acquisition confirmed by each condition's observed horizon",
        "method": "same four fixed-direction metrics, five registered landmarks, 2000 stratified seed bootstraps and deterministic seed strings as original analyzer; minimum five seeds per class",
        "uncertainty": "conditional on retained evaluation episodes; no multiplicity or metric-selection adjustment",
        "input_sha256": hashes, "groups": groups,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for group in groups:
        for item in group["landmarks"]:
            if group["width"] == 16 and item["landmark"] == 5000:
                print(json.dumps(item, indent=2))


if __name__ == "__main__":
    main()
