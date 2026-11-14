#!/usr/bin/env python3
"""Create a public usage report from private native receipts and quota snapshots."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

if __package__:
    from .measure_subscription_usage import RUN_KEYS, campaign_receipt_groups, openai_receipt_sessions, interrupted_launches, interrupted_openai_sessions
else:
    from measure_subscription_usage import RUN_KEYS, campaign_receipt_groups, openai_receipt_sessions, interrupted_launches, interrupted_openai_sessions

ROOT = Path(__file__).resolve().parents[1]
TOKEN_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens", "total_tokens")


def tokens(provider, usage):
    """Return disjoint canonical totals; cache is a subset of input, reasoning of output."""
    if provider == "glm":
        result = Counter({key: 0 for key in TOKEN_KEYS})
        for step in usage or []:
            cache = step.get("cache", {})
            result.update(input_tokens=step.get("input", 0) + cache.get("read", 0) + cache.get("write", 0),
                          cached_input_tokens=cache.get("read", 0),
                          output_tokens=step.get("output", 0) + step.get("reasoning", 0),
                          reasoning_output_tokens=step.get("reasoning", 0))
    else:
        usage = usage or {}
        cached = usage.get("cache_read_input_tokens", 0) if provider == "grok" else usage.get("cached_input_tokens", 0)
        input_tokens = usage.get("input_tokens", 0)
        if provider == "grok":
            input_tokens += cached + usage.get("cache_creation_input_tokens", 0)
        result = Counter(input_tokens=input_tokens, cached_input_tokens=cached,
                         output_tokens=usage.get("output_tokens", 0),
                         reasoning_output_tokens=usage.get("reasoning_tokens", 0) if provider == "grok" else usage.get("reasoning_output_tokens", 0))
    result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return dict(result)


def read_json(path):
    return json.loads(path.read_text())


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat() if isinstance(value, (int, float)) else value


def quota_limits(snapshot):
    data = snapshot.get("response", snapshot.get("data", {}))
    return data.get("data", {}).get("limits", [])


def configuration_scope(row, campaign):
    """Resolve legacy directory labels and explicit receipt campaign metadata."""
    run_key = row["run_key"]
    if run_key.startswith("setup/"):
        _, directory_name, run_key = run_key.split("/", 2)
        campaign = directory_name
    campaign = row.get("campaign", campaign)
    if not isinstance(campaign, str) or not campaign:
        raise ValueError("configuration scopes require a nonempty campaign ID")
    return campaign, run_key


def report_totals(report):
    """Validate independent totals; cache/reasoning remain subsets throughout."""
    totals = {key: report["total_recorded_native_tokens_including_setup"].get(key, 0) for key in TOKEN_KEYS}
    if any(type(value) is not int or value < 0 for value in totals.values()):
        raise ValueError("recorded token totals must be nonnegative integers")
    if totals["total_tokens"] != totals["input_tokens"] + totals["output_tokens"]:
        raise ValueError("total tokens must equal input plus output without adding cache or reasoning subsets")
    if totals["cached_input_tokens"] > totals["input_tokens"] or totals["reasoning_output_tokens"] > totals["output_tokens"]:
        raise ValueError("cache and reasoning token subsets exceed their parent totals")
    counts = [row["request_receipts"] for row in report["configurations"]]
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("request receipt counts must be nonnegative integers")
    for key in TOKEN_KEYS:
        if sum(row.get(key, 0) for row in report["configurations"]) != totals[key]:
            raise ValueError(f"configuration {key} sum differs from the campaign total")
    return totals, sum(counts)


def launch_counts(report, receipt_count):
    missing = report.get("interrupted_invocations", {}).get("missing_receipt_launches", 0)
    if type(missing) is not int or missing < 0:
        raise ValueError("missing-receipt launch counts must be nonnegative integers")
    launches = receipt_count + missing
    if report.get("recorded_cli_launches", launches) != launches:
        raise ValueError("recorded CLI launches must equal receipts plus missing-receipt launches")
    usage_counts = [row.get("receipts_with_usage") for row in report["configurations"]]
    if any(value is not None and (type(value) is not int or not 0 <= value <= row["request_receipts"])
           for row, value in zip(report["configurations"], usage_counts)):
        raise ValueError("receipts with usage must lie within the recorded receipt count")
    unknown = missing + sum(row["request_receipts"] - value for row, value in zip(report["configurations"], usage_counts)) if all(value is not None for value in usage_counts) else None
    if "native_usage_unknown_count" in report and report["native_usage_unknown_count"] != unknown:
        raise ValueError("unknown native usage count differs from receipts plus interrupted launches")
    return launches, missing, unknown


def add_previous_reports(report, previous_reports=()):
    """Combine only each report's own campaign, never its cumulative fields."""
    campaign_ids, scopes = set(), set()
    cumulative = Counter()
    receipt_count = 0
    cli_launches, interrupted_count, unknown_count = 0, 0, 0
    previous_campaigns = []
    sources = [(report, None)]
    for supplied in previous_reports:
        raw = Path(supplied).read_bytes()
        sources.append((json.loads(raw), hashlib.sha256(raw).hexdigest()))
    for source, digest in sources:
        campaign = source["campaign"]
        if not isinstance(campaign, str) or not campaign:
            raise ValueError("reports require a nonempty campaign ID")
        if campaign in campaign_ids:
            raise ValueError(f"duplicate campaign ID in usage reports: {campaign}")
        campaign_ids.add(campaign)
        source_scopes = set()
        for row in source["configurations"]:
            scope = configuration_scope(row, campaign)
            if scope in scopes or scope in source_scopes:
                raise ValueError(f"overlapping configuration scope in usage reports: {scope[0]}/{scope[1]}")
            source_scopes.add(scope)
        for row in source.get("interrupted_invocations", {}).get("configurations", []):
            scope = configuration_scope(row, campaign)
            if scope in scopes:
                raise ValueError(f"overlapping configuration scope in usage reports: {scope[0]}/{scope[1]}")
            source_scopes.add(scope)
        scopes.update(source_scopes)
        totals, count = report_totals(source)
        launches, missing, unknown = launch_counts(source, count)
        cumulative.update(totals)
        receipt_count += count
        cli_launches += launches
        interrupted_count += missing
        unknown_count = unknown_count + unknown if unknown_count is not None and unknown is not None else None
        if digest is not None:
            previous_campaigns.append({"campaign": campaign, "as_of": source["as_of"],
                "total_recorded_native_tokens_including_setup": totals,
                "request_receipts": count, "recorded_cli_launches": launches,
                "interrupted_launches_without_receipts": missing, "native_usage_unknown_count": unknown,
                "source_sha256": digest})
    report.update(previous_campaigns=previous_campaigns,
                  cumulative_recorded_native_tokens_including_setup=dict(cumulative),
                  cumulative_request_receipts=receipt_count,
                  cumulative_recorded_cli_launches=cli_launches,
                  cumulative_interrupted_launches_without_receipts=interrupted_count,
                  cumulative_native_usage_unknown_count=unknown_count)
    return report


def generate(directory, label="after", campaign=None, include_setup_dirs=(), previous_reports=()):
    directory = Path(directory)
    campaign = campaign or directory.name
    all_groups = list(campaign_receipt_groups(directory, include_setup_dirs, include_imports=True))
    groups, imported = [], []
    for run_key, excluded, records in all_groups:
        actual = [record for record in records if "imported_from" not in record]
        if actual:
            groups.append((run_key, excluded, actual))
        count = len(records) - len(actual)
        if count:
            imported.append({"run_key": run_key, "imported_observations": count})
    configurations, sessions = [], openai_receipt_sessions(groups)
    launches, overlaps_excluded, overlap_incomplete = interrupted_launches(directory, include_setup_dirs, groups)
    interrupted_sessions = interrupted_openai_sessions(launches, sessions)
    for run_key, excluded_setup, records in groups:
        provider = records[0]["provider"]
        fallback_campaign = run_key.split("/", 2)[1] if run_key.startswith("setup/") else campaign
        record_campaigns = {record.get("campaign", fallback_campaign) for record in records}
        if len(record_campaigns) != 1:
            raise ValueError(f"mixed receipt campaigns in configuration: {run_key}")
        source_campaign = record_campaigns.pop()
        count = Counter()
        for record in records:
            count.update(tokens(provider, record.get("usage")))
        configurations.append({"run_key": run_key, "campaign": source_campaign, "requested_model": records[0]["requested_model"],
            "reasoning": records[0].get("reasoning", "native-default" if provider == "glm" else "low"),
            "category": ("excluded diagnostic/setup" if run_key.startswith("setup/") else "excluded setup") if excluded_setup else "benchmark (including reusable pilot)",
            "request_receipts": len(records), "receipts_with_usage": sum(bool(r.get("usage")) for r in records),
            "receipts_with_positive_token_totals": sum(tokens(provider, r.get("usage")).get("total_tokens", 0) > 0 for r in records),
            "runtime_variants": dict(Counter(r.get("runtime_variant", "existing-native-wrapper") for r in records)),
            **dict(count), "median_seconds": round(statistics.median(r["elapsed_seconds"] for r in records), 3),
            "started_at": min(r["started_at"] for r in records), "finished_at": max(r["finished_at"] for r in records)})
    proxy_path = directory / f"proxy-usage-{label}.jsonl"
    proxy_rows = [json.loads(line) for line in proxy_path.read_text().splitlines() if line.strip()]
    proxy = [row for row in proxy_rows if row.get("session_id") in sessions]
    interrupted_proxy = [row for row in proxy_rows if row.get("session_id") in interrupted_sessions]
    matched = {row["session_id"] for row in proxy if row.get("session_id") in sessions}
    accounts = sorted({row["account"] for row in proxy + interrupted_proxy})
    # Stable public aliases are unrelated to personal names or provider account IDs.
    aliases = {account: f"subscription {i + 1}" for i, account in enumerate(accounts)}
    providers, account_usage = [], []
    for account in accounts:
        rows = [row for row in proxy if row.get("account") == account and row.get("session_id") in sessions]
        if not rows:
            continue
        per_run = Counter(sessions[row["session_id"]] for row in rows)
        sums = Counter()
        for row in rows:
            sums.update(row.get("tokens") or {})
        account_usage.append({"account": aliases[account], "matched_proxy_requests": len(rows),
                              "requests_by_configuration": dict(sorted(per_run.items())),
                              "proxy_input_tokens": sums.get("input", 0),
                              "proxy_output_tokens": sums.get("output", 0),
                              "cache_details": "Use native receipts; this proxy feed does not expose Codex cache splits."})
    codex_before = read_json(directory / "quota-astra-before.json")
    codex_after = read_json(directory / f"quota-astra-{label}.json")
    before_profiles = {p["profile"]: p for p in codex_before["profiles"]}
    after_profiles = {p["profile"]: p for p in codex_after["profiles"]}
    for account in accounts:
        name = account.removesuffix("-codex")
        before_windows = {w["window"]: w for w in before_profiles.get(name, {}).get("windows", [])}
        after_windows = {w["window"]: w for w in after_profiles.get(name, {}).get("windows", [])}
        for window in sorted(set(before_windows) | set(after_windows)):
            a, b = before_windows.get(window, {}), after_windows.get(window, {})
            av, bv = a.get("used_percent"), b.get("used_percent")
            ta = before_profiles.get(name, {}).get("ts", codex_before["captured_at"])
            tb = after_profiles.get(name, {}).get("ts", codex_after["captured_at"])
            reset_a = ta + a.get("reset_after_seconds", 0)
            reset_b = tb + b.get("reset_after_seconds", 0)
            reset = bool(a and b and abs(reset_b - reset_a) > 120)
            delta = bv - av if av is not None and bv is not None and not reset else None
            providers.append({"provider": f"OpenAI · {aliases[account]}", "window": window,
                "before": av, "after": bv, "used_delta": delta, "unit": "percentage points of this subscription",
                "before_at": iso(codex_before["captured_at"]), "after_at": iso(codex_after["captured_at"]),
                "window_reset_detected": reset,
                "notes": "Shared with other fleet sessions; whole-percent native readings. " +
                         ("Window changed; subtraction is not a usage estimate." if reset else "Attributable request/token counts are separate; the provider exposes no exact marginal quota conversion.")})
    glm_before = read_json(directory / "quota-glm-before.json")
    glm_after = read_json(directory / f"quota-glm-{label}.json")
    before_limits = {(w["type"], w["unit"], w["number"]): w for w in quota_limits(glm_before)}
    after_limits = {(w["type"], w["unit"], w["number"]): w for w in quota_limits(glm_after)}
    for identity in sorted(set(before_limits) | set(after_limits)):
        a, b = before_limits.get(identity, {}), after_limits.get(identity, {})
        av, bv = a.get("percentage"), b.get("percentage")
        reset = a.get("nextResetTime") != b.get("nextResetTime")
        window = ("5-hour token window" if identity == ("TOKENS_LIMIT", 3, 5) else
                  "weekly token window (provider unit 6)" if identity == ("TOKENS_LIMIT", 6, 1) else
                  f"{identity[0]} (provider unit {identity[1]} × {identity[2]})")
        providers.append({"provider": "Z.ai GLM coding plan", "window": window,
            "before": av, "after": bv, "used_delta": bv - av if av is not None and bv is not None and not reset else None,
            "unit": "percentage points", "before_at": iso(glm_before["captured_at"]), "after_at": iso(glm_after["captured_at"]),
            "window_reset_detected": reset,
            "notes": "Native quota endpoint; includes excluded setup within the measurement window and any concurrent account activity. " +
                     ("Reset timestamp changed; no delta attributed." if reset else "Window identifiers are preserved as returned by the provider.")})
    grok_before = read_json(directory / "quota-grok-before.json")
    grok_after = read_json(directory / f"quota-grok-{label}.json")
    same_grok_window = bool(grok_before.get("reset_display")) and grok_before.get("reset_display") == grok_after.get("reset_display")
    providers.append({"provider": "Grok SuperGrok Heavy", "window": "weekly",
        "before": grok_before["used_percent"], "after": grok_after["used_percent"],
        "used_delta": grok_after["used_percent"] - grok_before["used_percent"] if same_grok_window else None, "unit": "percentage points",
        "before_at": iso(grok_before["captured_at"]), "after_at": iso(grok_after["captured_at"]),
        "before_reset_display": grok_before.get("reset_display"), "after_reset_display": grok_after.get("reset_display"),
        "notes": "Native /usage screen, whole-percent resolution; shared-account delta. A zero displayed change does not mean zero consumption. " +
                 ("Same weekly reset verified." if same_grok_window else "Matching reset identity unavailable; no usage delta inferred.")})
    totals = Counter()
    for row in configurations:
        totals.update({key: row[key] for key in TOKEN_KEYS})
    launch_groups = {}
    for row in launches:
        run_key = row["run_key"]
        fallback = run_key.split("/", 2)[1] if run_key.startswith("setup/") else campaign
        source_campaign = row.get("campaign") or fallback
        group = launch_groups.setdefault((source_campaign, run_key), {"campaign": source_campaign, "run_key": run_key,
            "category": "excluded diagnostic/setup" if row["excluded_setup"] else "benchmark",
            "missing_receipt_launches": 0, "native_usage_unknown_count": 0})
        group["missing_receipt_launches"] += 1
        group["native_usage_unknown_count"] += 1
    proxy_only_tokens = Counter()
    for row in interrupted_proxy:
        proxy_only_tokens.update({key: value for key, value in (row.get("tokens") or {}).items() if key in {"input", "output"}})
    proxy_with_usage = {row["session_id"] for row in interrupted_proxy if row.get("tokens") and
                        any(row["tokens"].get(key, 0) > 0 for key in ("input", "output"))}
    interrupted_summary = {"missing_receipt_launches": len(launches), "native_usage_unknown_count": len(launches),
        "configurations": list(launch_groups.values()), "completed_receipt_overlaps_excluded": overlaps_excluded,
        "overlap_check_incomplete_count": overlap_incomplete,
        "proxy_only_usage": {"known_sessions": len(interrupted_sessions),
            "matched_sessions": len({row["session_id"] for row in interrupted_proxy}),
            "matched_proxy_requests": len(interrupted_proxy), "sessions_with_positive_recorded_usage": len(proxy_with_usage),
            "launches_without_positive_recorded_proxy_usage": sum(row.get("session_id") not in proxy_with_usage for row in launches),
            "requests_by_configuration": dict(Counter(interrupted_sessions[row["session_id"]] for row in interrupted_proxy)),
            "proxy_input_tokens": proxy_only_tokens["input"], "proxy_output_tokens": proxy_only_tokens["output"],
            "proxy_total_tokens": proxy_only_tokens["input"] + proxy_only_tokens["output"],
            "notes": "Only interrupted sessions without native receipts; separate from all native token totals. Missing proxy usage is unknown consumption, not zero."}}
    receipt_count = sum(row["request_receipts"] for row in configurations)
    receipts_without_usage = sum(row["request_receipts"] - row["receipts_with_usage"] for row in configurations)
    report = {"schema_version": 1, "campaign": campaign, "as_of": iso(codex_after["captured_at"]),
        "billing": "Existing subscriptions; no separately billed inference API calls. Provider dollar estimates are not charges.",
        "provider_limit_documentation": {"glm": "https://docs.z.ai/devpack/faq"},
        "providers": providers, "configurations": configurations, "total_recorded_native_tokens_including_setup": dict(totals),
        "recorded_cli_launches": receipt_count + len(launches), "interrupted_invocations": interrupted_summary,
        "receipts_without_native_usage": receipts_without_usage, "native_usage_unknown_count": receipts_without_usage + len(launches),
        "imported_observations": sum(row["imported_observations"] for row in imported), "imported_configurations": imported,
        "openai_proxy_attribution": {"receipt_sessions": len(sessions), "matched_sessions": len(matched),
            "unmatched_sessions": len(set(sessions) - matched), "matched_proxy_requests": len(proxy),
            "accounts": account_usage, "scope": "Exact experiment session IDs; no personal account names or session IDs published."},
        "limitations": ["Quota changes include other users of each subscription and are rounded by providers.",
            "No supported conversion from token counts to exact percentage of a subscription limit is available.",
            "Input totals include cached tokens; output totals include reasoning. These subsets must not be added again.",
            "Native calls with no returned token usage may have consumed unreported tokens; receipt coverage is explicit for each configuration.",
            "Interrupted launches without receipts have unknown native usage and no inferred latency. Any proxy-only usage is reported separately and never added to native totals.",
            "Imported observations are counted separately; their source receipt, token, latency and session usage is not charged again to this campaign.",
            "Setup receipts are excluded from benchmark scores but included in subscription usage.",
            "Included setup receipts may predate the main campaign baseline; token totals cover all included receipts, while quota deltas cover only the recorded measurement windows.",
            "Cumulative totals add only each supplied report's own campaign and included setup, never its cumulative fields. Quota percentages are not aggregated.",
            "Overlap checks use reported campaign/configuration identities; historical reports without those identities cannot reveal renamed or copied receipt groups."]}
    return add_previous_reports(report, previous_reports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/native-20260912")
    parser.add_argument("--campaign", help="public campaign name (defaults to the run directory basename)")
    parser.add_argument("--include-setup-dir", type=Path, action="append", default=[],
                        help="include all receipts and attempts from an excluded diagnostic/setup campaign (repeatable)")
    parser.add_argument("--previous-report", type=Path, action="append", default=[],
                        help="add a previous report's own campaign totals to cumulative accounting (repeatable)")
    parser.add_argument("--label", default="after")
    parser.add_argument("--output", type=Path, default=ROOT / "data/benchmark-runs/subscription-usage.json")
    args = parser.parse_args()
    report = generate(args.run_dir, args.label, args.campaign, args.include_setup_dir, args.previous_report)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}: {len(report['configurations'])} configurations/setup groups")
