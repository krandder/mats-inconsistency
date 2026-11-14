#!/usr/bin/env python3
"""Read native subscription limits and match Codex receipts to proxy usage.

This performs no inference and never writes credentials. Outputs are private in
ignored runs/. Grok's percentage is transcribed from the native /usage screen.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUN_KEYS = ["astra", "grok", "glm"] + ["luna-" + effort for effort in ("none", "low", "medium", "high", "xhigh", "max")]


def campaign_roots(directory, include_setup_dirs=()):
    """Resolve campaign directories once, with unambiguous public setup labels."""
    directory = Path(directory)
    roots = [(directory, None)]
    seen_paths = {directory.resolve()}
    setup_names = set()
    for supplied in include_setup_dirs:
        setup = Path(supplied)
        resolved = setup.resolve()
        if resolved == directory.resolve():
            raise ValueError("the main campaign cannot also be an excluded setup campaign")
        if resolved in seen_paths:
            continue
        if not setup.is_dir():
            raise ValueError(f"setup campaign directory does not exist: {setup}")
        name = resolved.name
        if name in setup_names:
            raise ValueError(f"setup campaign directory names must be unique: {name}")
        seen_paths.add(resolved)
        setup_names.add(name)
        roots.append((setup, "setup/" + name))
    return roots


def campaign_receipt_groups(directory, include_setup_dirs=(), include_imports=False):
    """Yield native receipt groups, including archived attempts but no configs."""
    for root, prefix in campaign_roots(directory, include_setup_dirs):
        run_keys = sorted(p.name for p in root.iterdir() if p.is_dir()) if prefix else RUN_KEYS + ["glm-preflight-steps1"]
        for run_key in run_keys:
            records = []
            for path in sorted((root / run_key).glob("*.json")):
                record = json.loads(path.read_text())
                if (isinstance(record, dict) and record.get("provider") in ("astra", "luna", "grok", "glm")
                        and "requested_model" in record):
                    if "imported_from" in record:
                        origin = record["imported_from"]
                        if not isinstance(origin, dict) or not all(origin.get(key) for key in ("campaign", "run_key", "receipt_sha256", "receipt_id")):
                            raise ValueError("imported receipts require source campaign, run_key, receipt hash and ID")
                        if not include_imports:
                            continue
                    records.append(record)
            if records:
                yield (f"{prefix}/{run_key}" if prefix else run_key,
                       bool(prefix) or run_key == "glm-preflight-steps1", records)


def interrupted_launches(directory, include_setup_dirs=(), groups=()):
    """Read private launch evidence; remove only identifiable receipt overlaps.

    No receipt, token count or latency is synthesized. Ambiguous proposition
    matches require resolving the evidence before totals can be produced.
    """
    receipts = {run_key: records for run_key, _, records in groups}
    launches, seen, excluded, incomplete = [], set(), 0, 0
    for root, prefix in campaign_roots(directory, include_setup_dirs):
        path = root / "interrupted-invocations.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        rows = data.get("invocations") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError("interrupted invocation evidence requires an invocations list")
        for raw in rows:
            run_key = raw["run_key"]
            if not isinstance(run_key, str) or not run_key or "/" in run_key or "\\" in run_key:
                raise ValueError("interrupted run_key must be a configuration name")
            public_key = f"{prefix}/{run_key}" if prefix else run_key
            sid, invocation_id = raw.get("session_id"), raw.get("invocation_id")
            proposition = raw.get("proposition_id") or raw.get("prompt_hash_prefix")
            started = raw.get("started_at")
            identity = (public_key, invocation_id or sid or (proposition, started))
            if identity in seen:
                continue
            seen.add(identity)
            candidates = receipts.get(public_key, [])
            if any((sid and sid == row.get("session_id")) or
                   (invocation_id and invocation_id == row.get("invocation_id")) or
                   (proposition and started and row.get("id", "").startswith(proposition)
                    and started == row.get("started_at")) for row in candidates):
                excluded += 1
                continue
            if proposition and not started and any(row.get("id", "").startswith(proposition) for row in candidates):
                raise ValueError(f"ambiguous completed-receipt overlap for interrupted configuration: {public_key}")
            if not (sid or proposition):
                incomplete += 1
            campaign = raw.get("campaign", data.get("campaign") if isinstance(data, dict) else None)
            launches.append({**raw, "run_key": public_key, "campaign": campaign,
                             "excluded_setup": bool(prefix) or run_key == "glm-preflight-steps1"})
    return launches, excluded, incomplete


def interrupted_openai_sessions(launches, receipt_sessions):
    sessions = {}
    for row in launches:
        sid, run_key = row.get("session_id"), row["run_key"]
        native_key = run_key.split("/", 2)[-1] if run_key.startswith("setup/") else run_key
        if sid and (native_key == "astra" or native_key.startswith("luna-")):
            if sid in receipt_sessions or (sid in sessions and sessions[sid] != run_key):
                raise ValueError("an interrupted session has ambiguous receipt/configuration attribution")
            sessions[sid] = run_key
    return sessions


def openai_receipt_sessions(groups):
    """Map sessions to their configuration; reject ambiguous attribution."""
    sessions = {}
    for run_key, _, records in groups:
        for record in records:
            sid = record.get("session_id")
            if sid and record["provider"] in {"astra", "luna"}:
                if sid in sessions and sessions[sid] != run_key:
                    raise ValueError("a native session ID occurs in multiple configuration groups")
                sessions[sid] = run_key
    return sessions


def remote_python(host, code, stdin=None, sudo=False):
    command = ("sudo -n " if sudo else "") + "python3 -c " + shlex.quote(code)
    result = subprocess.run(["ssh", "-o", "ConnectTimeout=15", host, command],
                            input=stdin, text=True, capture_output=True, timeout=120, check=True)
    return result.stdout


def capture(directory, label, grok_percent=None, grok_reset_display=None, include_setup_dirs=()):
    directory = Path(directory)
    groups = list(campaign_receipt_groups(directory, include_setup_dirs))
    sessions = openai_receipt_sessions(groups)
    launches, _, _ = interrupted_launches(directory, include_setup_dirs, groups)
    interrupted_sessions = interrupted_openai_sessions(launches, sessions)
    directory.mkdir(parents=True, exist_ok=True)
    captured_at = datetime.now(timezone.utc).isoformat()
    auth_path = Path("/home/agent/.local/share/opencode/auth.json")
    key = json.loads(auth_path.read_text())["zai-coding-plan"]["key"]
    request = urllib.request.Request("https://api.z.ai/api/monitor/usage/quota/limit",
                                     headers={"Authorization": key})
    with urllib.request.urlopen(request, timeout=30) as response:
        quota = json.load(response)
    (directory / f"quota-glm-{label}.json").write_text(json.dumps({
        "captured_at": captured_at, "source": "native coding-plan quota endpoint", "response": quota}, indent=2))
    codex_code = '''
import sys,json,time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,"/home/kelvin/fleet/apps/tokens-dashboard/usage")
from codex_limits import auth_paths
from codex_usage import poll_usage_limits
def one(pair):
 name,path=pair
 try:
  result=poll_usage_limits(path);result["profile"]=name;return result
 except Exception as error: return {"profile":name,"error":type(error).__name__+": "+str(error)}
with ThreadPoolExecutor(max_workers=10) as pool:
 rows=list(pool.map(one,auth_paths().items()))
print(json.dumps({"captured_at":time.time(),"profiles":rows}))
'''
    (directory / f"quota-astra-{label}.json").write_text(remote_python("deploy-host", codex_code))
    usage_code = '''
import sys,json,glob
ids=set(json.load(sys.stdin))
for path in sorted(glob.glob("/var/lib/primeproxy/unified-claude/usage/usage*.jsonl")):
 for line in open(path):
  try: row=json.loads(line)
  except ValueError: continue
  if row.get("session_id") in ids: print(json.dumps(row))
'''
    usage = remote_python("prime-any-proxy-1", usage_code, json.dumps(sorted(set(sessions) | set(interrupted_sessions))), sudo=True)
    (directory / f"proxy-usage-{label}.jsonl").write_text(usage)
    if grok_percent is not None:
        (directory / f"quota-grok-{label}.json").write_text(json.dumps({
            "source": "General native Grok Build 1.0.13 /usage (interactive terminal)",
            "measurement": "Weekly limit (SuperGrok Heavy)", "used_percent": grok_percent,
            "resolution": "whole percent", "reset_display": grok_reset_display,
            "captured_at": captured_at}, indent=2))
    print(json.dumps({"captured_at": captured_at, "receipt_sessions": len(sessions),
                      "interrupted_sessions": len(interrupted_sessions),
                      "proxy_usage_rows": len(usage.splitlines()), "saved": str(directory)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label", help="before, during, or after")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/native-20260912")
    parser.add_argument("--include-setup-dir", type=Path, action="append", default=[],
                        help="excluded diagnostic/setup campaign whose sessions are also attributed (repeatable)")
    parser.add_argument("--grok-percent", type=float)
    parser.add_argument("--grok-reset-display", help="exact reset date/time shown beside the native percentage")
    args = parser.parse_args()
    capture(args.run_dir, args.label, args.grok_percent, args.grok_reset_display, args.include_setup_dir)
