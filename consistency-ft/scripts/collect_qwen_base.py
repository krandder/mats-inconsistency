#!/usr/bin/env python3
"""Plan, collect, inspect or explicitly derive complete Qwen own-base anchors."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.qwen_base_collection import (AttemptLedger, collect, create_plan, export_anchor,
                                           load_plan, owner_lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Pure full-catalog/identity plan; no Torch or weights")
    plan.add_argument("--spec", type=Path, required=True)
    plan.add_argument("--spec-sha256", required=True)
    plan.add_argument("--directory", type=Path, required=True)
    for command in ("preflight", "status", "collect", "export"):
        sub = commands.add_parser(command)
        sub.add_argument("--directory", type=Path, required=True)
        sub.add_argument("--plan-sha256", required=True)
        if command == "collect":
            for name in ("checkpoint", "tokenizer-artifacts"):
                sub.add_argument("--" + name, type=Path, required=True)
            for name in ("max-attempts", "token-budget", "wall-seconds", "cpu-soft-seconds", "cpu-hard-seconds", "address-space-bytes"):
                sub.add_argument("--" + name, type=int, required=True)
        if command == "export":
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--mode", choices=("exact", "symmetric-mixture"), required=True)
            sub.add_argument("--smoothing-epsilon", type=float)
    args = parser.parse_args()
    values = vars(args).copy()
    command = values.pop("command")
    try:
        if command == "plan":
            result = {"status": "planned-no-inference", "plan_sha256": create_plan(
                args.spec, args.spec_sha256, args.directory), "target_propositions": 2926, "target_episodes": 1619}
        elif command in ("preflight", "status"):
            with owner_lock(args.directory, shared=True):
                plan, spec, rows = load_plan(args.directory, args.plan_sha256)
                result = AttemptLedger(args.directory, rows, binding=(plan, spec, args.plan_sha256)).inspect()
                result.pop("states")
                result.update({"status": "preflight-verified-no-inference", "target_episodes": 1619,
                               "plan_sha256": args.plan_sha256, "training_identity_sha256": plan["training_identity_sha256"]})
        elif command == "collect":
            result = collect(**values)
            if "coverage" in result:
                result["coverage"].pop("states", None)
            result.pop("states", None)
        else:
            snapshot = export_anchor(**values)
            result = {"status": snapshot["payload"]["status"], "snapshot_sha256": snapshot["snapshot_sha256"],
                      "treatment": snapshot["payload"]["treatment"], "target_propositions": 2926}
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 1 if result.get("status") == "failed" else 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Collection command failed: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
