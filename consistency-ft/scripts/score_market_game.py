#!/usr/bin/env python3
"""Score a market-game run directory from its snapshots and session log.

Per pass (from snapshots/pass-P.json): Brier and log loss of the market
marginals against the resolutions, the Turtel baselines from the file,
per-pass improvement, extremization rate, herding (autocorrelation of
per-question price increments between consecutive passes) and the sibling
coherence report. From sessions.jsonl: trade validity, rejection reasons,
realized LMSR payoff per session and the telescoping check.

Writes <run>/scores.json and <run>/scores.md; prints the markdown.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import market_game as mg  # noqa: E402

EXTREME_LOW, EXTREME_HIGH = 0.1, 0.9


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def log_payoff(pre: float, post: float, outcome: int, eps: float = 1e-12) -> float:
    """log p_new(w) - log p_old(w) for a binary slice priced as P(yes)."""
    if outcome == 1:
        return math.log(max(post, eps)) - math.log(max(pre, eps))
    return math.log(max(1.0 - post, eps)) - math.log(max(1.0 - pre, eps))


_REPLAY_CACHE: dict[str, dict[int, dict]] = {}


def replay_markets(run: Path, passes: int, ids: list[str]) -> dict[int, dict]:
    """End-of-pass market snapshots rebuilt from the session log by fill version.

    The log is the source of truth: resumed or retried runs rewrite pass-N.json,
    but every executed fill carries a global execution version, so the market at
    the end of pass N is the replay of all fills whose version is at most the
    largest version among pass-N sessions. Sessions of an earlier pass re-run
    after a later pass (out of order) are excluded from the boundary and reported.
    """
    key = str(run)
    if key in _REPLAY_CACHE:
        return _REPLAY_CACHE[key]
    cfg = json.loads((run / "config.json").read_text()).get("config", {})
    rows = [json.loads(l) for l in (run / "sessions.jsonl").read_text().splitlines() if l.strip()]
    fills = []
    for r in rows:
        if r.get("status") != "completed":
            continue
        for e in r.get("executed") or []:
            fills.append((int(e["version"]), int(r["pass"]), e, r.get("finished_at") or ""))
    fills.sort(key=lambda x: x[0])
    boundary: dict[int, int] = {}
    # a pass boundary is the last version before any fill of a later pass appears
    seen_later = 0
    for v, p, e, _ in fills:
        if p >= seen_later:
            seen_later = p
            boundary[p] = v
    market = mg.Market.fresh(ids, liquidity=float(cfg.get("liquidity", 100.0)), max_width=int(cfg.get("max_width", 8)))
    out: dict[int, dict] = {0: market.snapshot(meta={"pass": 0})}
    out_of_order = 0
    for v, p, e, _ in fills:
        t = e["trade"]
        trade = mg.Trade(t["variable"], float(t["target"]), tuple((c["variable"], c["outcome"]) for c in t.get("context") or []))
        got, rej = market.execute([trade], None)
        if rej or abs(got[0]["post"] - e["post"]) > 1e-6:
            raise SystemExit(f"replay diverges from the log at fill version {v}")
        if p < seen_later and v > boundary.get(p, 0):
            out_of_order += 1
        for pn, b in boundary.items():
            if v == b and pn <= passes:
                out[pn] = market.snapshot(meta={"pass": pn, "replayed_from_log": True, "at_version": v})
    out["_out_of_order_fills"] = out_of_order  # type: ignore[index]
    _REPLAY_CACHE[key] = out
    return out


def pass_prices(run: Path, passes: int, ids: list[str]) -> dict[int, dict[str, float]]:
    snaps = replay_markets(run, passes, ids)
    return {p: mg.Market.from_snapshot(snaps[p]).prices() for p in range(0, passes + 1) if p in snaps}


def joint_log_loss(run: Path, p: int, resolution: dict[str, int]) -> float | None:
    """Exact -log P(omega)/n under the pass-p joint, by the chain rule: read P(q | conditioned so far),
    then condition the junction tree on q's resolution. Order-invariant (verified 2026-09-22)."""
    snaps = _REPLAY_CACHE.get(str(run)) or {}
    if p not in snaps:
        return None
    market = mg.Market.from_snapshot(snaps[p])
    total = 0.0
    n = 0
    for qid in market.question_ids:
        if qid not in resolution:
            continue
        pyes = market.engine.marginal(qid)["yes"]
        pw = pyes if resolution[qid] else 1.0 - pyes
        total += -math.log(max(1e-12, pw))
        market.engine.condition(qid, "yes" if resolution[qid] else "no")
        n += 1
    return total / n if n else None


def score_prices(prices: dict[str, float], resolution: dict[str, int]) -> dict[str, Any]:
    pairs = [(prices[qid], resolution[qid]) for qid in resolution if qid in prices]
    extreme = sum(1 for p, _ in pairs if p <= EXTREME_LOW or p >= EXTREME_HIGH)
    return {"brier": mg.brier(pairs), "log_loss": mg.log_loss(pairs), "n": len(pairs),
            "extremization_rate": extreme / len(pairs) if pairs else math.nan,
            "mean_price": sum(p for p, _ in pairs) / len(pairs) if pairs else math.nan}


def coherence(prices: dict[str, float], sets: list[dict[str, Any]], resolution: dict[str, int]) -> dict[str, Any]:
    rows = []
    for sibling in sets:
        if not sibling.get("exclusive"):
            continue
        members = [m for m in sibling["members"] if m in prices]
        if len(members) < 2:
            continue
        total = sum(prices[m] for m in members)
        yes = sum(resolution[m] for m in members)
        rows.append({"set_id": sibling["set_id"], "kind": sibling["kind"], "n": len(members), "sum": total,
                     "resolved_yes": yes, "complete_partition": yes == 1 and not sibling.get("duplicate_texts"),
                     "duplicate_texts": sibling.get("duplicate_texts", 0)})
    clean = [r for r in rows if not r["duplicate_texts"]]
    complete = [r for r in rows if r["complete_partition"]]
    return {
        "exclusive_sets": len(rows), "exclusive_sets_distinct_text": len(clean),
        "mean_sum_exclusive": sum(r["sum"] for r in clean) / len(clean) if clean else None,
        "share_sum_above_one": sum(1 for r in clean if r["sum"] > 1.05) / len(clean) if clean else None,
        "complete_partitions": len(complete),
        "mean_abs_sum_minus_one_complete": (sum(abs(r["sum"] - 1.0) for r in complete) / len(complete)
                                            if complete else None),
        "sets": rows,
    }


def herding(prices: dict[int, dict[str, float]], ids: list[str]) -> dict[str, Any]:
    passes = sorted(p for p in prices if p > 0)
    increments = {p: [prices[p][q] - prices[p - 1][q] for q in ids] for p in passes if p - 1 in prices}
    pairs = []
    pooled_x: list[float] = []
    pooled_y: list[float] = []
    for p in passes:
        if p + 1 in increments and p in increments:
            xs, ys = increments[p], increments[p + 1]
            same = [(x > 0) == (y > 0) for x, y in zip(xs, ys) if x != 0 and y != 0]
            pairs.append({"passes": [p, p + 1], "autocorrelation": pearson(xs, ys),
                          "same_direction_share": sum(same) / len(same) if same else None,
                          "mean_abs_increment": [sum(abs(x) for x in xs) / len(xs), sum(abs(y) for y in ys) / len(ys)]})
            pooled_x += xs
            pooled_y += ys
    return {"consecutive_pairs": pairs, "pooled_autocorrelation": pearson(pooled_x, pooled_y) if pooled_x else None,
            "mean_abs_increment_by_pass": {p: sum(abs(x) for x in xs) / len(xs) for p, xs in increments.items()}}


def session_metrics(records: list[dict[str, Any]], resolution: dict[str, int], liquidity: float,
                    final_prices: dict[str, float], links: dict[str, list[str]]) -> dict[str, Any]:
    final = [r for r in records if r.get("status") in {"completed", "failed"}]
    proposed = sum(r.get("proposed", 0) for r in records)
    executed = sum(len(r.get("executed", [])) for r in records)
    reasons = Counter()
    for r in records:
        for rejected in r.get("rejected", []):
            reasons[rejected.get("reason", "unknown")] += 1
    parse_failures = sum(1 for r in records if r.get("status") in {"retry", "failed"})
    missing = sum(len(r.get("missing_questions", [])) for r in final)
    asked = sum(r.get("n", 0) for r in final)
    kl_budget_sessions = sum(1 for r in final if any(x.get("reason") == "kl_budget" for x in r.get("rejected", [])))

    payoffs = []
    telescoping_sum: dict[str, float] = {}
    conditional_trades = 0
    called_off = 0
    for r in final:
        total = 0.0
        for fill in r.get("executed", []):
            spec = fill["trade"]
            outcome = resolution[spec["variable"]]
            context = spec.get("context") or []
            if context:
                conditional_trades += 1
                if any(resolution[c["variable"]] != (1 if c["outcome"] == "yes" else 0) for c in context):
                    called_off += 1
                    continue
                total += liquidity * log_payoff(fill["pre"], fill["post"], outcome)
            else:
                gain = liquidity * log_payoff(fill["pre_marginal"], fill["post_marginal"], outcome)
                total += gain
                telescoping_sum[spec["variable"]] = telescoping_sum.get(spec["variable"], 0.0) + gain
        payoffs.append({"session_id": r["session_id"], "pass": r["pass"], "payoff": total})
    by_pass: dict[int, list[float]] = {}
    for item in payoffs:
        by_pass.setdefault(item["pass"], []).append(item["payoff"])
    residuals = []
    unlinked_max = 0.0
    linked_abs = []
    for qid, outcome in resolution.items():
        if qid not in final_prices:
            continue
        target_total = liquidity * log_payoff(0.5, final_prices[qid], outcome)
        residual = target_total - telescoping_sum.get(qid, 0.0)
        is_linked = bool(links.get(qid))
        residuals.append({"variable": qid, "residual": residual, "linked": is_linked})
        if is_linked:
            linked_abs.append(abs(residual))
        else:
            unlinked_max = max(unlinked_max, abs(residual))
    return {
        "sessions": len(final), "attempts": len(records),
        "failed_sessions": sum(1 for r in final if r["status"] == "failed"),
        "parse_or_empty_failures": parse_failures, "proposed_trades": proposed, "executed_trades": executed,
        "validity_rate": executed / proposed if proposed else None,
        "rejection_reasons": dict(sorted(reasons.items())),
        "missing_question_rate": missing / asked if asked else None,
        "sessions_hitting_kl_budget": kl_budget_sessions,
        "conditional_trades_executed": conditional_trades, "conditional_trades_called_off": called_off,
        "payoff": {"total": sum(p["payoff"] for p in payoffs),
                   "mean_per_session": sum(p["payoff"] for p in payoffs) / len(payoffs) if payoffs else None,
                   "share_positive": sum(1 for p in payoffs if p["payoff"] > 0) / len(payoffs) if payoffs else None,
                   "by_pass": {p: {"total": sum(v), "mean": sum(v) / len(v)} for p, v in sorted(by_pass.items())}},
        "telescoping": {"variables": len(residuals), "linked_variables": len(linked_abs),
                        "max_abs_residual_unlinked": unlinked_max,
                        "mean_abs_residual_linked": sum(linked_abs) / len(linked_abs) if linked_abs else 0.0,
                        "sum_residual_linked": sum(r["residual"] for r in residuals if r["linked"]),
                        "unlinked_ok": unlinked_max < 1e-6},
        "per_session_payoff": payoffs,
    }


def score_run(run: Path) -> dict[str, Any]:
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    questions = read_jsonl(run / "questions.jsonl")
    ids = [q["question_id"] for q in questions]
    resolution = {q["question_id"]: int(q["resolution"]) for q in questions}
    sets = json.loads((run / "sibling-sets.json").read_text()) if (run / "sibling-sets.json").exists() else []
    liquidity = float(config["config"]["liquidity"])
    passes = int(config["config"]["passes"])
    prices = pass_prices(run, passes, ids)
    scored_passes = sorted(p for p in prices if p > 0)

    baselines = {}
    for column in ("prediction_polymarket", "prediction_o1", "prediction_remax"):
        values = {q["question_id"]: q[column] for q in questions if q.get(column) is not None}
        baselines[column] = score_prices(values, {k: resolution[k] for k in values})
    base_rate = sum(resolution.values()) / len(resolution)
    baselines["constant_0.5"] = score_prices({q: 0.5 for q in ids}, resolution)
    baselines["base_rate"] = score_prices({q: base_rate for q in ids}, resolution)

    per_pass = []
    previous = score_prices(prices[0], resolution)
    for p in scored_passes:
        current = score_prices(prices[p], resolution)
        jll = joint_log_loss(run, p, resolution)
        current.update({"pass": p, "brier_improvement": previous["brier"] - current["brier"],
                        "log_loss_improvement": previous["log_loss"] - current["log_loss"],
                        "joint_log_loss": jll,
                        "structure_gain": (current["log_loss"] - jll) if jll is not None else None,
                        "coherence": coherence(prices[p], sets, resolution)})
        per_pass.append(current)
        previous = current

    records = read_jsonl(run / "sessions.jsonl") if (run / "sessions.jsonl").exists() else []
    last = scored_passes[-1] if scored_passes else 0
    snapshot = (_REPLAY_CACHE.get(str(run)) or {}).get(last) or {}
    if snapshot:
        snapshot = dict(snapshot); snapshot.setdefault("meta", {})["engine_stats"] = mg.Market.from_snapshot(snapshot).stats()
    sessions = session_metrics(records, resolution, liquidity, prices[last], snapshot.get("links", {}))
    result = {"run": str(run), "questions": len(ids), "passes_scored": scored_passes, "liquidity": liquidity,
              "provider": config.get("provider"), "baselines": baselines, "per_pass": per_pass,
              "herding": herding(prices, ids), "sessions": sessions, "engine_stats": snapshot.get("meta", {}).get("engine_stats")}
    return result


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown(result: dict[str, Any]) -> str:
    lines = [f"# Market game scores: {result['run']}", "",
             f"{result['questions']} questions, passes scored {result['passes_scored']}, "
             f"provider {json.dumps(result['provider'])}", "",
             "| pass | Brier | log loss | joint log loss | structure gain | dBrier vs prev | extremization | mean price | excl. sets mean sum | complete partitions mean abs(sum-1) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for row in result["per_pass"]:
        c = row["coherence"]
        lines.append(f"| {row['pass']} | {fmt(row['brier'])} | {fmt(row['log_loss'])} | {fmt(row.get('joint_log_loss'))} | {fmt(row.get('structure_gain'))} | {fmt(row['brier_improvement'])} | "
                     f"{fmt(row['extremization_rate'], 3)} | {fmt(row['mean_price'], 3)} | {fmt(c['mean_sum_exclusive'], 3)} "
                     f"({c['exclusive_sets_distinct_text']}) | {fmt(c['mean_abs_sum_minus_one_complete'], 3)} ({c['complete_partitions']}) |")
    lines += ["", "| baseline | Brier | log loss | extremization |", "|---|---|---|---|"]
    for name, row in result["baselines"].items():
        lines.append(f"| {name} | {fmt(row['brier'])} | {fmt(row['log_loss'])} | {fmt(row['extremization_rate'], 3)} |")
    h = result["herding"]
    lines += ["", f"Herding: pooled autocorrelation of consecutive increments {fmt(h['pooled_autocorrelation'])}; "
              + "; ".join(f"passes {p['passes']}: r={fmt(p['autocorrelation'])}, same-direction {fmt(p['same_direction_share'], 3)}"
                          for p in h["consecutive_pairs"])]
    s = result["sessions"]
    lines += ["", f"Sessions: {s['sessions']} ({s['failed_sessions']} failed, {s['parse_or_empty_failures']} parse/empty attempts), "
              f"validity {fmt(s['validity_rate'])} ({s['executed_trades']}/{s['proposed_trades']}), "
              f"missing-question rate {fmt(s['missing_question_rate'])}, KL-budget hits {s['sessions_hitting_kl_budget']}, "
              f"conditional trades {s['conditional_trades_executed']} ({s['conditional_trades_called_off']} called off)",
              f"Rejections: {json.dumps(s['rejection_reasons'])}",
              f"Realized LMSR payoff: total {fmt(s['payoff']['total'], 2)}, mean/session {fmt(s['payoff']['mean_per_session'], 2)}, "
              f"share positive {fmt(s['payoff']['share_positive'], 3)}; by pass "
              + ", ".join(f"{p}: {fmt(v['total'], 1)}" for p, v in s["payoff"]["by_pass"].items()),
              f"Telescoping: unlinked max |residual| {s['telescoping']['max_abs_residual_unlinked']:.2e} "
              f"(ok={s['telescoping']['unlinked_ok']}); linked variables {s['telescoping']['linked_variables']}, "
              f"mean |residual| {fmt(s['telescoping']['mean_abs_residual_linked'], 3)}, "
              f"sum {fmt(s['telescoping']['sum_residual_linked'], 2)}"]
    if result.get("engine_stats"):
        lines += ["", f"Engine: {json.dumps(result['engine_stats'])}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="run directory written by scripts/run_market_game.py")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    run = Path(args.run)
    result = score_run(run)
    text = markdown(result)
    (run / "scores.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    (run / "scores.md").write_text(text)
    if not args.quiet:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
