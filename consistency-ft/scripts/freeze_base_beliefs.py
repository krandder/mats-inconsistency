#!/usr/bin/env python3
"""Freeze existing published base beliefs offline; never makes provider calls."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchors import freeze_from_manifest, write_snapshot  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/benchmark-runs/manifest.json"))
    parser.add_argument("--run-id", required=True, help="one explicit source run; no base-model default")
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "heldout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true", help="write explicit missing-proposition/episode coverage")
    parser.add_argument("--allowed-ids", type=Path, help="JSON array of permitted source proposition IDs")
    parser.add_argument("--smoothing-epsilon", type=float, help="explicit symmetric-mixture KL treatment, not clipping")
    args = parser.parse_args()
    try:
        allowed = json.loads(args.allowed_ids.read_text()) if args.allowed_ids else None
        if allowed is not None and (not isinstance(allowed, list) or not all(isinstance(x, str) for x in allowed)):
            raise ValueError("--allowed-ids must contain a JSON array of strings")
        snapshot = freeze_from_manifest(args.manifest, args.run_id, args.episodes, args.split,
                                        allow_partial=args.allow_partial, allowed_ids=allowed,
                                        smoothing_epsilon=args.smoothing_epsilon)
        write_snapshot(args.output, snapshot)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Cannot freeze base beliefs: {error}\n")
    payload = snapshot["payload"]
    coverage = payload["coverage"]
    print(json.dumps({"output": str(args.output), "snapshot_sha256": snapshot["snapshot_sha256"],
                      "status": payload["status"], "base": payload["base"], "treatment": payload["treatment"],
                      "dataset_sha256": payload["dataset"]["sha256"],
                      "anchored_propositions": coverage["anchored_propositions"],
                      "target_propositions": coverage["target_propositions"],
                      "complete_episodes": len(coverage["complete_episode_ids"]),
                      "target_episodes": coverage["target_episodes"]}, sort_keys=True))


if __name__ == "__main__":
    main()
