#!/usr/bin/env python3
"""R9-E1's own GPU ledger (ceiling $0.60 inside budget group r9) from logical-soup/GPU-LOG.md (soup-gpu-r9e1-* VMs only).

Per VM: the DELETE or COST-AUDIT line's cost_usd when present; otherwise, for a VM still alive, rate x age so far plus the
rest of its reservation window (worst case). Rates are the UNVERIFIED on-demand list-price ceiling ($0.87/h per L4).
  ledger.py                     print the table and the total
  ledger.py --check-launch MIN  exit 1 if spent + this launch's worst case (0.87 x (MIN/60 + 0.05)) > CEILING. MIN is the
                                gpu-run.sh timeout; gpu-run.sh deletes the VM on exit (trap) and 3 min covers VM start
                                and delete (R8-E1: start-to-delete overhead about 1 min);
                                the r9 guard deletes any r9 VM older than 90 min as the backstop.
  ledger.py --markdown          the table as markdown (for RESULTS-R9-E1.md section GPU-LOG)
"""
import re
import sys
from datetime import datetime, timezone

GPULOG = "/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md"
CEILING = 0.60
RATE = 0.87


def parse():
    vms = {}
    for line in open(GPULOG):
        m = re.search(r"vm=(soup-gpu-r9e1-[a-z0-9-]+)", line)
        if not m:
            continue
        v = vms.setdefault(m.group(1), {"lines": []})
        v["lines"].append(line.rstrip())
        ts = line[2:19]
        if " CREATE vm=" in line:
            v["created"] = ts
            z = re.search(r"zone=(\S+)", line)
            v["zone"] = z.group(1) if z else ""
            p = re.search(r"model=(\S+)", line)
            v["prov"] = p.group(1) if p else ""
            r = re.search(r"reserve_usd=([0-9.]+)", line)
            v["reserve"] = float(r.group(1)) if r else 0.0
        if " PENDING-CREATE " in line:
            r = re.search(r"reserve_usd=([0-9.]+)", line)
            v.setdefault("reserve", float(r.group(1)) if r else 0.0)
        if " CREATE-FAILED " in line:
            v["failed"] = True
        if re.search(r" (DELETE|COST-AUDIT) ", line):
            c = re.search(r"cost_usd=([0-9.]+)", line)
            if c:
                v["cost"] = float(c.group(1))
                v["deleted"] = ts
    return vms


def cost_of(v, now):
    if "cost" in v:
        return v["cost"], "DELETE/COST-AUDIT line"
    if v.get("failed"):
        return 0.0, "create failed (no instance)"
    if "created" in v:
        c = datetime.strptime(v["created"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
        age = (now - c).total_seconds() / 3600
        return max(v.get("reserve", 0.0), RATE * age), "alive: reservation (worst case)"
    return v.get("reserve", 0.0), "pending: reservation"


def main():
    now = datetime.now(timezone.utc)
    vms = parse()
    rows, tot = [], 0.0
    for n, v in sorted(vms.items(), key=lambda kv: kv[1].get("created", "")):
        c, basis = cost_of(v, now)
        tot += c
        rows.append((n, v.get("zone", ""), v.get("prov", ""), v.get("created", ""), v.get("deleted", ""), c, basis))
    if "--check-launch" in sys.argv:
        mins = float(sys.argv[sys.argv.index("--check-launch") + 1])
        worst = RATE * (mins / 60 + 0.05)
        ok = tot + worst <= CEILING
        print(f"R9-E1 ledger: spent/reserved ${tot:.2f} + this launch worst case ${worst:.2f} "
              f"{'<=' if ok else '>'} ceiling ${CEILING:.2f}")
        sys.exit(0 if ok else 1)
    if "--markdown" in sys.argv:
        print("| VM | zone | provisioning | created (UTC) | deleted (UTC) | cost USD (ceiling) | basis |")
        print("|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]:.2f} | {r[6]} |")
        print(f"\n**R9-E1 total (ceiling): ${tot:.2f}** of its ${CEILING:.2f} ceiling ({len(rows)} VM entries).")
        return
    for r in rows:
        print(*r, sep="\t")
    print(f"TOTAL\t{tot:.2f}\tceiling\t{CEILING:.2f}")


if __name__ == "__main__":
    main()
