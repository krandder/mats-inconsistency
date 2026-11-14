"""wm2 audit: GPU cost of the wm2 VMs recomputed from logical-soup/GPU-LOG.md (own parser; not gpu/cost.py).

  python3 audit/gpucost.py [--now 2026-10-11T19:50Z]

Per VM: CREATE time, provisioning model, rate; end = DELETE (cost_usd as logged) or COST-AUDIT; VMs with neither are
still running and are priced at the ceiling rate from CREATE to --now (and to their max timeout as an upper bound).
Also recomputes each logged DELETE cost as rate x (DELETE - CREATE) to check the logged numbers.
"""
import argparse
import re
from datetime import datetime, timezone

LOG = "/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md"


def ts(s):
    s = s.rstrip("Z")
    for f in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(s, f).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", default=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"))
    a = ap.parse_args()
    now = ts(a.now)
    vms = {}
    for line in open(LOG):
        m = re.match(r"- (\S+) (\S+) vm=(soup-gpu-wm2\S*)", line)
        if not m:
            continue
        t, ev, vm = ts(m.group(1)), m.group(2), m.group(3)
        v = vms.setdefault(vm, {"events": [], "create": None, "model": None, "rate": None, "cost": None, "end": None,
                                "audit": None, "timeout_min": None, "reserve": None})
        v["events"].append(ev)
        if ev == "CREATE":
            v["create"] = t
            mo = re.search(r"model=(\S+)", line)
            v["model"] = mo.group(1) if mo else None
            r = re.search(r"rate_usd_h=([\d.]+)", line)
            v["rate"] = float(r.group(1)) if r else None
            r = re.search(r"reserve_usd=([\d.]+)", line)
            v["reserve"] = float(r.group(1)) if r else None
        if ev == "RUN":
            r = re.search(r"timeout=(\d+)m", line)
            v["timeout_min"] = int(r.group(1)) if r else None
        if ev in ("DELETE", "COST-AUDIT"):
            c = re.search(r"cost_usd=([\d.]+)", line)
            v["cost"] = float(c.group(1)) if c else None
            v["end"] = t
            v["audit"] = ev
    tot_logged, tot_recomp, tot_open_now, tot_open_max = 0.0, 0.0, 0.0, 0.0
    rows = []
    for vm, v in vms.items():
        if v["cost"] is not None:
            tot_logged += v["cost"]
            rec = None
            if v["create"] and v["audit"] == "DELETE":
                rec = v["rate"] * (v["end"] - v["create"]).total_seconds() / 3600
                tot_recomp += rec
            else:
                tot_recomp += v["cost"]
            rows.append((vm, v["model"], v["audit"], v["cost"], rec))
        elif v["create"] is not None:
            h = (now - v["create"]).total_seconds() / 3600
            hmax = ((v["timeout_min"] or 232) + 15) / 60
            tot_open_now += v["rate"] * h
            tot_open_max += v["rate"] * max(h, hmax)
            rows.append((vm, v["model"], "OPEN", v["rate"] * h, v["rate"] * max(h, hmax)))
        else:
            rows.append((vm, v["model"], "never-created", 0.0, None))
    for r in sorted(rows, key=lambda r: r[0]):
        print(f"  {r[0][13:55]:44s} {str(r[1]):9s} {r[2]:13s} {r[3]:6.2f} {'' if r[4] is None else f'{r[4]:6.2f}'}")
    n_created = sum(1 for v in vms.values() if v["create"])
    spot = sum(1 for v in vms.values() if v["model"] == "SPOT")
    print(f"wm2 VMs in the log: {len(vms)} names, {n_created} created ({spot} spot), "
          f"closed: logged cost ${tot_logged:.2f} (recomputed from timestamps ${tot_recomp:.2f}); "
          f"open at {a.now}: ${tot_open_now:.2f} so far, at most ${tot_open_max:.2f} at their timeouts; "
          f"total so far ${tot_logged + tot_open_now:.2f}, upper bound ${tot_logged + tot_open_max:.2f}")


if __name__ == "__main__":
    main()
