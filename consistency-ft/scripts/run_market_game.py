#!/usr/bin/env python3
"""Run the market game: a frozen model trades a combinatorial LMSR market.

Examples
  # offline smoke (no provider calls, no money)
  python3 scripts/run_market_game.py --provider mock --limit 20 --passes 2 --out runs/market-game/smoke
  # full mock run
  python3 scripts/run_market_game.py --provider mock --out runs/market-game/mock-full
  # print the exact Codex command and the first prompt, execute nothing
  python3 scripts/run_market_game.py --provider astra --effort high --dry-run --out runs/market-game/dry
  # real run (needs Kelvin's go), resumable
  python3 scripts/run_market_game.py --provider astra --effort high --out runs/market-game/astra-1 --resume

Use /home/agent/workspaces/venv-cft/bin/python the first time so the
parquet can be read (pyarrow); afterwards the stdlib interpreter reads the
question cache under runs/market-game/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import market_game as mg  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=["mock", "astra", "luna", "sol"], default="mock")
    parser.add_argument("--effort", default="medium", help="Codex model_reasoning_effort (astra/luna)")
    parser.add_argument("--n-per-session", type=int, default=8)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", required=True, help="run directory (log, snapshots, config)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="skip logged sessions, reload the latest snapshot")
    parser.add_argument("--dry-run", action="store_true", help="print the command and the first prompt; execute nothing")
    parser.add_argument("--limit", type=int, default=None, help="seeded random subset of questions for smoke runs")
    parser.add_argument("--liquidity", type=float, default=mg.LIQUIDITY)
    parser.add_argument("--max-width", type=int, default=mg.MAX_WIDTH)
    parser.add_argument("--budget-factor", type=float, default=mg.BUDGET_FACTOR,
                        help="eps_total = factor * N on passes after the first")
    parser.add_argument("--cap-first-pass", action="store_true", help="apply the KL budget on pass 1 too")
    parser.add_argument("--snapshot-every", type=int, default=50, help="engine snapshot every K sessions (0: only per pass)")
    parser.add_argument("--quota-check-cmd", default=None, help="shell command run before sessions; PAUSE on stdout pauses the pool")
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds per Codex invocation")
    parser.add_argument("--data", default=str(mg.PARQUET_PATH), help="parquet (pyarrow) or JSONL export of the test set")
    parser.add_argument("--download", action="store_true", help="mirror the dataset into the archive if missing")
    parser.add_argument("--table1-json", default=None, help="also write the Table 1 verification JSON here")
    parser.add_argument("--mock-flip-rate", type=float, default=0.2, help="mock: probability of believing the wrong outcome")
    parser.add_argument("--mock-step", type=float, default=0.5, help="mock: fraction of the gap to the believed outcome to move")
    parser.add_argument("--mock-invalid-rate", type=float, default=0.1)
    parser.add_argument("--mock-conditional-rate", type=float, default=0.15)
    parser.add_argument("--mock-garbage-rate", type=float, default=0.02)
    parser.add_argument("--mock-big-jump-rate", type=float, default=0.05)
    parser.add_argument("--mock-latency", type=float, default=0.0, help="mock: seconds to sleep per call")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = Path(args.data)
    if args.download or not data.exists():
        data = mg.ensure_dataset()
    questions = mg.load_questions(data)
    verification = mg.verify_table1(questions)
    if len(questions) == 1265 and not verification["ok"]:
        raise SystemExit("Table 1 verification failed: " + json.dumps(verification["computed"]))
    mock_options = {"flip_rate": args.mock_flip_rate, "step": args.mock_step, "invalid_rate": args.mock_invalid_rate,
                    "conditional_rate": args.mock_conditional_rate, "garbage_rate": args.mock_garbage_rate,
                    "big_jump_rate": args.mock_big_jump_rate, "latency": args.mock_latency}
    config = mg.RunConfig(out=args.out, provider=args.provider, effort=args.effort, n_per_session=args.n_per_session,
                          passes=args.passes, workers=args.workers, seed=args.seed, limit=args.limit,
                          liquidity=args.liquidity, max_width=args.max_width, budget_factor=args.budget_factor,
                          first_pass_uncapped=not args.cap_first_pass, snapshot_every=args.snapshot_every,
                          quota_check_cmd=args.quota_check_cmd, timeout=args.timeout, data=str(data),
                          mock_options=mock_options if args.provider == "mock" else {})
    selected = mg.select_questions(questions, args.limit, args.seed)
    provider = mg.make_provider(args.provider, selected, effort=args.effort, seed=args.seed, timeout=args.timeout,
                                mock_options=mock_options, work_root=Path(args.out) / "codex")
    arena = mg.Arena(config, selected, provider)

    if args.dry_run:
        arena.market = mg.Market.fresh([q.question_id for q in selected], config.liquidity, config.max_width)
        groups = arena.groups_for_pass(1)
        request, _ = arena.build_request(1, 0, groups[0], 1)
        print(f"# dry run: {len(selected)} questions, {len(groups)} sessions per pass, "
              f"{config.passes} passes, provider {provider.describe()}")
        print(f"# developer_instructions (sha256 {mg.sha256_text(mg.DEVELOPER_RULE)}):\n{mg.DEVELOPER_RULE}\n")
        if isinstance(provider, mg.CodexProvider):
            plan = provider.dry_run(request, Path(args.out) / "codex" / "dry-run")
            print("# command line (prompt on stdin):")
            print(plan["command_line"])
            print(f"# env overrides: {json.dumps(plan['env_overrides'])}")
            print(f"# final message file: {plan['output_file']}")
        else:
            print("# mock provider: no external command")
        print(f"# prompt sha256 {mg.sha256_text(request.prompt)}, {len(request.prompt)} chars, "
              f"budget {request.budget}")
        print("# ---- prompt ----")
        print(request.prompt)
        return 0

    arena.prepare(resume=args.resume)
    (Path(args.out) / "table1-verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    if args.table1_json:
        Path(args.table1_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.table1_json).write_text(json.dumps(verification, indent=2) + "\n")
    print(f"table 1: " + ", ".join(f"{k.split('_', 1)[1]} {v['brier']:.4f}" for k, v in verification["computed"].items())
          + f" (ok={verification['ok']})", flush=True)
    summary = arena.run()
    stats = summary["stats"]
    print(f"done: {stats['sessions']} sessions ({stats['failed']} failed, {stats['retries']} retries), "
          f"{stats['executed']} executed / {stats['proposed']} proposed trades, "
          f"{summary['elapsed_seconds']}s; scores: python3 scripts/score_market_game.py {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
