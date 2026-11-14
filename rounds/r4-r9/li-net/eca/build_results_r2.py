"""Assemble RESULTS-R2.md (round 2) from RESULTS-R2-head.md, RESULTS-R2-mid.md, results/r2/tables.md and the GPU ledger."""
import os
import re

E = os.path.dirname(os.path.abspath(__file__))
rd = lambda p: open(os.path.join(E, p)).read()
head, mid = rd("RESULTS-R2-head.md"), rd("RESULTS-R2-mid.md")
ledger = rd("GPU-LOG.md")
m = re.search(r"\*\*Total \(ceiling\): \$([\d.]+)\*\* over (\d+) VMs", ledger)
total = float(m.group(1)) if m else float("nan")
r2 = 0.0
for line in ledger.splitlines():
    if line.startswith("| soup-gpu-eca-r2"):
        r2 += float(line.split("|")[7])
live = re.search(r"Live eca VMs at rebuild time \(gcloud\): (.*)", ledger)
live = live.group(1).strip() if live else "?"
tldr = """- **R2-D2a (depth gap with a working one-step law, 1 layer): FAIL for both rules.** One layer opens a large one-pass gap
  (validation KL rule 110: F 0.34, B1 0.45, B2 0.26; rule 90: F 0.10, B1 0.36, B2 0.049 nats) but also fails the one-step
  check (S1 KL 0.037 and 0.077 against a 0.025 bar, vs about 0.008 at 4 layers): the network no longer computes the local
  rule reliably.
- **R2-D2b (stiff-first search wins under the bottleneck): FAIL for both rules.** Best case, rule 110 under sw_mix at M = 16:
  stiffness-first chains cut KL by 24% on forward queries (bar 25%) and beat random order there; on bridges they cut 21-29%
  with CIs including 0 and no edge over random order. Under the plain mixture they gain nothing.
- **R2-D3a (key separation): PASS as pre-registered, but by construction.** The rebuilt key's cosine between forward and
  backward chains is -0.10 / 0.07 (B1 / B2; round 1: 0.95), redundant views keep 0.91, drift ratio 0.05. The audit showed the
  separation comes from restricting the fingerprint to the newly materialised cells, which are disjoint for the two chains:
  the same key over all revealed cells gives 0.73 / 0.81 (it would fail), and two disjoint same-direction chains also score
  about 0.08. So the key separates disjoint supports, not directions.
- **R2-D3b (variance-gated VoI): FAIL.** VoI over uncertain cells beats VoI over stiff cells on B1 (M = 8, mixture: 0.099 ->
  0.068, cut 0.31 [0.15, 0.47]) and beats one pass by 25-31% there, short of the 35% bar, and does not beat stiff-gated VoI
  on B2. Random order remains the best selector on bridges.
- **Secondary, an anchored mixture repairs round 1's pooling:** on the 4-layer rule-110 models, sw_mix (anchored at the
  one-pass belief) improves on one pass at M = 16 for every selector and family except VoI-uncertain on F (14 of 20 cuts
  with paired CIs above 0, bridges -20% to -46%; on F only confidence-first's CI is above 0), where round 1's pool made
  beliefs worse. The fitted stiffness exponent is 0 in 89 of 90 of these fits, so this is an anchored uniform mixture, not
  stiffness weighting. Added after the results were read; secondary.
- **Cost:** ${r2:.2f} for round 2 (5 L4 VMs; ceiling rate), ${total:.2f} for the night so far; zero eca VMs left.""".format(r2=r2, total=total)
cost = f"""- Round 2: **${r2:.2f}** at the $0.87/h ceiling for 5 on-demand L4 VMs (round-2 cap $25); night total **${total:.2f}**
  for E-CA (`GPU-LOG.md`, rebuilt by `gpu/ledger.py`). One R2a VM landed in us-west1-a after stockouts in us-central1.
- Live eca VMs at the final check: {live}. MATS cluster: not used."""
out = head.replace("R2_TLDR", tldr) + "\n" + mid.replace("R2_COST", cost).replace("R2_TABLES", rd("results/r2/tables.md"))
open(os.path.join(E, "RESULTS-R2.md"), "w").write(out)
print("RESULTS-R2.md", len(out), "round-2 $", round(r2, 2), "total $", total)
