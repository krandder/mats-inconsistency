#!/usr/bin/env python3
"""Run bounded synthetic CPU integration; never download/load candidate models."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from anchors import sha256
from training.synthetic_cpu import SyntheticExperiment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; must not already exist")
    parser.add_argument("--steps", type=int, choices=range(1, 6), default=2)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--model-seed", type=int, default=20260913)
    parser.add_argument("--anchor-seed", type=int, default=101)
    parser.add_argument("--rollout-seed", type=int, default=202)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must name a new directory; existing artifacts are never overwritten")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    # Seeds are declared before generating anything; no success-dependent retry.
    environment = {"synthetic_software_integration_only": True, "python": sys.version.split()[0],
                   "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                   "torch_cuda_build": torch.version.cuda, "torch_threads": torch.get_num_threads(),
                   "torch_interop_threads": torch.get_num_interop_threads(),
                   "model_seed": args.model_seed, "anchor_seed": args.anchor_seed,
                   "rollout_seeds": list(range(args.rollout_seed, args.rollout_seed + args.steps)),
                   "optimizer": {"kind": "SGD", "learning_rate": .01},
                   "lambda_anchor": 2.0,
                   "fixture_hyperparameters_are_not_scientific_selections": True}
    base = Path(__file__).resolve().parents[1]
    environment["source_sha256"] = {name: sha256((base / name).read_bytes()) for name in (
        "scripts/run_synthetic_cpu_training.py",
        "training/torch_loop.py", "training/categorical.py", "training/synthetic_cpu.py", "training/episode_adapter.py", "anchors.py",
        "training/requirements-cpu-py312-linux-x86_64.lock")}
    try:
        experiment = SyntheticExperiment(args.output, model_seed=args.model_seed, anchor_seed=args.anchor_seed)
        write_json(args.output / "environment.json", environment)
        original_anchor = sha256((args.output / "anchor.json").read_bytes())
        optimizer = torch.optim.SGD(experiment.model.parameters(), lr=.01)
        steps = [experiment.step(optimizer, batch_id=f"step-{index}", rollout_seed=args.rollout_seed + index)
                 for index in range(args.steps)]
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report = {"status": "synthetic-integration-complete", "synthetic_only": True,
                  "candidate_training_performed": False, "scientific_base_selected": False,
                  "model_parameters": sum(p.numel() for p in experiment.model.parameters()),
                  "anchor_propositions": len(experiment.snapshot["payload"]["anchors"]),
                  "anchor_unchanged": original_anchor == sha256((args.output / "anchor.json").read_bytes()),
                  "steps": steps, "wall_seconds": time.monotonic() - started,
                  "cpu_seconds": usage.ru_utime + usage.ru_stime,
                  "peak_rss_bytes_linux": usage.ru_maxrss * 1024, "environment": environment}
        write_json(args.output / "report.json", report)
        print(json.dumps({key: report[key] for key in ("status", "model_parameters", "anchor_propositions",
                                                       "anchor_unchanged", "wall_seconds", "peak_rss_bytes_linux")}, indent=2))
    except FileExistsError:
        # Also protect against another invocation claiming this path after the
        # preflight check. Never write a failure receipt into its directory.
        raise
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output / "run-failure.json", {"status": "failed", "synthetic_only": True,
                       "error": str(error), "environment": environment, "resampled": False})
        raise


if __name__ == "__main__":
    main()
