"""Assemble RESULTS-R3.md (round 3) from RESULTS-R3-head.md, RESULTS-R3-static.md, RESULTS-R3-mid.md, results/r3/tables.md
and the GPU ledger. Usage: python3 build_results_r3.py"""
import os
import re

E = os.path.dirname(os.path.abspath(__file__))
rd = lambda p: open(os.path.join(E, p)).read()
ledger = rd("GPU-LOG.md")
m = re.search(r"\*\*Total \(ceiling\): \$([\d.]+)\*\* over (\d+) VMs", ledger)
total = float(m.group(1)) if m else float("nan")
r3 = sum(float(l.split("|")[7]) for l in ledger.splitlines() if l.startswith("| soup-gpu-eca3"))
n3 = sum(1 for l in ledger.splitlines() if l.startswith("| soup-gpu-eca3") and l.split("|")[5].strip())   # created VMs only
live = re.search(r"Live eca VMs at rebuild time \(gcloud\): (.*)", ledger)
live = live.group(1).strip() if live else "?"
cost = (f"- Round 3: **${r3:.2f}** at the $0.87/h ceiling for {n3} on-demand L4 VMs (1 smoke + 9 main; cap $40); night total for "
        f"E-CA **${total:.2f}** (`GPU-LOG.md`, rebuilt by `gpu/ledger.py`).\n- Live eca VMs at the final check: {live}. MATS cluster: not used.")
out = (rd("RESULTS-R3-head.md").replace("R3_COST_SHORT", f"${r3:.2f}").replace("R3_TOTAL", f"${total:.2f}") + "\n"
       + rd("RESULTS-R3-static.md") + "\n" + rd("RESULTS-R3-mid.md").replace("R3_COST", cost) + "\n## 12. Full tables\n\n"
       + rd("results/r3/tables.md"))
out = out.replace("\n## 12. Full tables\n", """
## 11. Audit and post-audit corrections

The independent audit (`AUDIT-R3.md`, Opus 5.5, 12:00-12:18Z) re-derived every verdict and table line from the raw files
(2,154 summary numbers within 3.9e-7), regenerated the data (0 overlap, shipped data identical), checked the exact posteriors
against its own dense 4,096-state code (9.7e-15), confirmed the code diff and that nothing leaks, and the GPU total. It found
no high-severity issue. Corrections applied after it (12:20Z): the R3-D3 null is close to degenerate and plain confidence
tracks the information gain better than the head (M1); the gap is not attributed to the overlap removal alone (M2); the dd
pool's offset, not the VoI chains, is what is worse than one pass (M3); and L1-L12 (hybrid branching counts and the
"frontier" wording, rounding, the rule-90 outlier seed, the VMs count, the round-2 one-layer wording, the spec revision at
09:35:51Z, the unverifiable draft time, stale docstrings, the unused round-1 gate field).

## 12. Full tables
""")
open(os.path.join(E, "RESULTS-R3.md"), "w").write(out)
print("RESULTS-R3.md", len(out), "round-3 $", round(r3, 2), "night total $", total)
