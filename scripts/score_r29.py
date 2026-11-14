#!/usr/bin/env python3
"""
Deterministic Scorer for Round 29 (R29A, R29B) under docs/PROTOCOL.md §2.5 & §9.7-9.8.
Reads the synced JSON output from a completed run and formats RESULTS-R29{A,B}.md
purely from JSON fields (zero hand-typed numbers; verdicts strictly PASS or FAIL).
Hard-fails if:
  - any requested results JSON file is missing (§9.7),
  - 'smoke_cpu' is missing or not strictly False unless --allow-smoke is passed (§9.8),
  - 'runner_sha256' or 'prereg_sha256' in JSON does not match the file on disk (unless --allow-smoke),
  - 'seeds' in JSON does not match the frozen PREREG seed list (unless --allow-smoke),
  - 'gates' key set in JSON is empty, has extra keys, or differs from recomputed gates.
"""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


EXPECTED_SPECS: Dict[str, Tuple[Path, Path, List[int]]] = {
    "R29A": (
        Path(_REPO_ROOT) / "benchmarks" / "run_r29a_decentralized_kyle_futarchy_and_clrs_ood.py",
        Path(_REPO_ROOT) / "benchmarks" / "PREREG-R29A-DECENTRALIZED-KYLE-FUTARCHY-AND-CLRS-OOD.md",
        [101, 102, 103, 104, 105],
    ),
}


def _resolve_round_spec(rnd: str) -> Tuple[Path, Path, List[int]]:
    if rnd in EXPECTED_SPECS:
        return EXPECTED_SPECS[rnd]
    bench_dir = Path(_REPO_ROOT) / "benchmarks"
    runners = sorted(bench_dir.glob(f"run_{rnd.lower()}_*.py"))
    preregs = sorted(bench_dir.glob(f"PREREG-{rnd.upper()}-*.md"))
    if runners and preregs:
        return (runners[0], preregs[0], [101, 102, 103, 104, 105])
    raise RuntimeError(f"Unknown round '{rnd}' and no matching benchmarks/run_{rnd.lower()}_*.py found")


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_and_recompute_gates(json_path: Path, data: dict, allow_smoke: bool = False) -> dict:
    if "smoke_cpu" not in data or not isinstance(data["smoke_cpu"], bool):
        raise RuntimeError(
            f"Refusing to score {json_path}: missing or non-boolean 'smoke_cpu' key (docs/PROTOCOL.md §9.8)"
        )
    if data["smoke_cpu"] is not False and not allow_smoke:
        raise RuntimeError(
            f"Refusing to score {json_path}: smoke_cpu={data.get('smoke_cpu')} (docs/PROTOCOL.md §9.8)"
        )
    rnd = data.get("round", "")
    runner_path, prereg_path, expected_seeds = _resolve_round_spec(rnd)
    if not allow_smoke:
        if data.get("seeds") != expected_seeds:
            raise RuntimeError(
                f"Refusing to score {json_path}: seeds={data.get('seeds')} != frozen {expected_seeds}"
            )
        actual_runner_sha = compute_sha256(runner_path)
        actual_prereg_sha = compute_sha256(prereg_path)
        if data.get("runner_sha256") != actual_runner_sha:
            raise RuntimeError(
                f"Refusing to score {json_path}: runner_sha256={data.get('runner_sha256')} != disk {actual_runner_sha}"
            )
        if data.get("prereg_sha256") != actual_prereg_sha:
            raise RuntimeError(
                f"Refusing to score {json_path}: prereg_sha256={data.get('prereg_sha256')} != disk {actual_prereg_sha}"
            )

    stored_gates = data.get("gates")
    if not isinstance(stored_gates, dict) or len(stored_gates) == 0:
        raise RuntimeError(f"Refusing to score {json_path}: 'gates' must be a non-empty dict")

    if rnd == "R29A":
        from benchmarks.run_r29a_decentralized_kyle_futarchy_and_clrs_ood import evaluate_r29a_gates

        recomputed = evaluate_r29a_gates(data)
    else:
        mod_name = f"benchmarks.{runner_path.stem}"
        mod = importlib.import_module(mod_name)
        eval_fn = getattr(mod, f"evaluate_{rnd.lower()}_gates")
        recomputed = eval_fn(data)

    if set(stored_gates.keys()) != set(recomputed.keys()):
        raise RuntimeError(
            f"Refusing to score {json_path}: stored gate keys {sorted(stored_gates.keys())} != expected {sorted(recomputed.keys())}"
        )
    for gid, gobj in recomputed.items():
        if stored_gates.get(gid, {}).get("verdict") != gobj.get("verdict"):
            raise RuntimeError(
                f"Gate verdict mismatch for {gid}: json={stored_gates.get(gid, {}).get('verdict')} vs recomputed={gobj.get('verdict')}"
            )
    return recomputed


def render_markdown_from_json(json_path: Path, allow_smoke: bool = False) -> str:
    if not json_path.exists():
        raise FileNotFoundError(f"Missing required results file: {json_path} (docs/PROTOCOL.md §9.7)")
    data = json.loads(json_path.read_text())
    rnd = data.get("round", "R29")
    gates = validate_and_recompute_gates(json_path, data, allow_smoke=allow_smoke)
    allowed_verdicts = ("PASS", "FAIL", "SMOKE_NO_VERDICT") if allow_smoke else ("PASS", "FAIL")
    lines = [
        f"# Deterministic Score Report: `{rnd}` (`docs/PROTOCOL.md` §2.5)",
        "",
        f"- **Source JSON:** `{json_path}`",
        f"- **Runner SHA256:** `{data.get('runner_sha256', 'N/A')}`",
        f"- **PREREG SHA256:** `{data.get('prereg_sha256', 'N/A')}`",
        f"- **Device:** `{data.get('device')}`",
        f"- **Smoke CPU Mode:** `{data.get('smoke_cpu')}`",
        f"- **Elapsed Seconds:** `{data.get('elapsed_sec')}`",
        f"- **Seeds:** `{data.get('seeds')}`",
        "",
        "## 1. Pre-Registered Gate Verdicts (Copied Verbatim from JSON `gates`)",
        "",
        "| Gate ID | Verdict | Details (JSON Path `gates.<gate_id>`) |",
        "| :--- | :---: | :--- |",
    ]
    for gid, gobj in gates.items():
        verdict = gobj.get("verdict", "FAIL")
        if verdict not in allowed_verdicts:
            raise RuntimeError(f"Invalid verdict {verdict} for gate {gid}: allowed={allowed_verdicts}")
        summary_kv = {k: v for k, v in gobj.items() if k != "verdict"}
        lines.append(f"| `{gid}` | **`{verdict}`** | `{json.dumps(summary_kv)}` |")

    lines += [
        "",
        "## 2. Full Gate Payload (`gates`)",
        "",
        "```json",
        json.dumps(gates, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="", help="Path to single r29{a,b}_results.json")
    parser.add_argument("--out-md", default="", help="Path to write RESULTS-R29*.md")
    parser.add_argument("--results-dir", default="", help="Directory containing r29{a,b}_results.json")
    parser.add_argument(
        "--subrounds",
        default="r29a",
        help="Comma-separated subrounds to score in --results-dir (e.g. r29a,r29b)",
    )
    parser.add_argument("--out", default="", help="Optional summary JSON output path")
    parser.add_argument("--allow-smoke", action="store_true", help="Dev-only override; default refuses smoke_cpu=true")
    args = parser.parse_args()

    if args.json:
        md = render_markdown_from_json(Path(args.json), allow_smoke=args.allow_smoke)
        if args.out_md:
            out_p = Path(args.out_md)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(md)
            print(f"Wrote {out_p} from {args.json}")
        else:
            print(md)
        return

    if args.results_dir:
        rdir = Path(args.results_dir)
        summary = {}
        subrounds = [s.strip().lower() for s in args.subrounds.split(",") if s.strip()]
        if not subrounds:
            raise ValueError("No subrounds specified in --subrounds")
        for sub in subrounds:
            name = f"{sub}_results.json"
            p = rdir / name
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing required results file {p} for requested subround '{sub}' (docs/PROTOCOL.md §9.7)"
                )
            data = json.loads(p.read_text())
            rnd = data.get("round", sub.upper())
            gates = validate_and_recompute_gates(p, data, allow_smoke=args.allow_smoke)
            md = render_markdown_from_json(p, allow_smoke=args.allow_smoke)
            summary[rnd] = {gid: gobj.get("verdict", "FAIL") for gid, gobj in gates.items()}
            md_path = rdir / f"RESULTS-{rnd}.md"
            md_path.write_text(md)
        print(json.dumps(summary, indent=2))
        if args.out:
            out_p = Path(args.out)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(summary, indent=2) + "\n")
        return

    parser.error("Provide either --json or --results-dir")


if __name__ == "__main__":
    main()
