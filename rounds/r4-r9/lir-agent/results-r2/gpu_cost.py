"""GPU spend of group r4-e3a from logical-soup/GPU-LOG.md: per VM the DELETE/COST-AUDIT cost_usd (ceiling rate x age),
else the CREATE reservation (worst case). Prints one line per VM and the total. Read-only."""
import re
import sys

LOG = "/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md"
res, cost, zone, first = {}, {}, {}, None
for line in open(LOG):
    if "soup-gpu-r4e3a" not in line:
        continue
    m = re.search(r"vm=(soup-gpu-r4e3a[\w-]+)", line)
    if not m:
        continue
    vm = m.group(1)
    ts = line.split()[1]
    if first is None:
        first = (ts, line.strip()[:160])
    if " CREATE " in line or "PENDING-CREATE" in line:
        r = re.search(r"reserve_usd=([\d.]+)", line)
        if r:
            res[vm] = float(r.group(1))
        z = re.search(r"zone=([\w-]+)", line)
        if z:
            zone[vm] = z.group(1)
    if " DELETE " in line or "COST-AUDIT" in line:
        c = re.search(r"cost_usd=([\d.]+)", line)
        if c:
            cost[vm] = float(c.group(1))
tot = 0.0
for vm in sorted(set(res) | set(cost)):
    v = cost.get(vm, res.get(vm, 0.0))
    tot += v
    print(f"{vm} zone={zone.get(vm, '?')} {'cost' if vm in cost else 'RESERVED(no delete line)'}={v:.2f}")
print(f"first_r4e3a_line {first}")
print(f"TOTAL_r4e3a_usd {tot:.2f} (ceiling rate, Billing API disabled; spot/standard real price lower or equal)")
