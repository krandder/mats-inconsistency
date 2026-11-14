"""Assemble RESULTS.md from the hand-written fragments and the generated tables (eurisko E-CA, 2026-10-15).
  python3 build_results.py
Inputs: RESULTS-head.md (title + TL;DR), RESULTS-static.md (Sec. 2 body), RESULTS-mid.md (Sec. 3-8; VERDICTS_TABLE is
replaced by results/verdicts.md), results/replication/verdicts.md, results/t8/verdicts.md, RESULTS-tail.md (Sec. 10),
GPU-LOG.md (ledger), results/tables.md (Sec. 13)."""
import json
import os
import re

E = os.path.dirname(os.path.abspath(__file__))
rd = lambda p: open(os.path.join(E, p)).read()
head, static, mid, tail = rd("RESULTS-head.md"), rd("RESULTS-static.md"), rd("RESULTS-mid.md"), rd("RESULTS-tail.md")
mid = mid.replace("VERDICTS_TABLE", rd("results/verdicts.md").strip())
S = json.load(open(os.path.join(E, "results/summary.json")))
src = S.get("sources", {})
n_b = sum(1 for v in src.values() if "/main2/" in v)
rep_v = rd("results/replication/verdicts.md").strip()
t8_v = rd("results/t8/verdicts.md").strip()
ledger = rd("GPU-LOG.md")
m = re.search(r"\*\*Total \(ceiling\): \$([\d.]+)\*\* over (\d+) VMs", ledger)
total, nvm = (m.group(1), m.group(2)) if m else ("?", "?")
live = re.search(r"Live eca VMs at rebuild time \(gcloud\): (.*)", ledger)
live = live.group(1).strip() if live else "?"
sec9 = f"""## 9. Secondary results: the replication batch and the rule-90 T = 8 hedge

**Replication (the original 01:35Z runs; float32 pooling, probabilities capped to [6e-8, 1 - 6e-8], PREREG Sec. 12).**
Same seeds and configuration as the primary runs; GPU nondeterminism makes them a second draw of every training run.
Statistics that are non-finite in their result.json (the float32 bug) are "not evaluable".

{rep_v}

The replication reproduces every primary verdict that it can evaluate, except rule-90 T2's D1 (AUC 0.848 here vs 0.857 in
the primary runs: the 0.85 bar sits inside the run-to-run spread). D3b is not evaluable for rule 90 here because of the
float32 bug.

**Rule 90 at T = 8 (secondary hedge, PREREG Sec. 11; also float32, capped).** A deeper grid does not create a gap for
rule 90: one pass stays near-exact (validation KL 0.001-0.011 on F, B1 and B2 for T1), so D2 has nothing to win; D1's AUC
drops to 0.82.

{t8_v}

Full tables: `results/replication/tables.md`, `results/t8/tables.md`; figures `results/replication/*.png`, `results/t8/*.png`.
"""
sec11 = f"""## 11. Cost and clean-up

- GPU: **${total}** at the on-demand ceiling rate ($0.87/h per g2-standard-8 + L4; Cloud Billing API disabled, so this is
  an upper bound) for {nvm} VMs: the priced smoke, 12 original runs on 6 VMs, the rule-90 T = 8 hedge on 3 VMs and the
  12-run eca-b rerun on 6 VMs. Per-VM ledger: `GPU-LOG.md` (rebuilt from `logical-soup/GPU-LOG.md` by `gpu/ledger.py`).
  Group cap $110 (overnight-ca); `guard-eca.sh` (4.2 h age kill, $105 stop) never fired.
- Live eca VMs at the final check: {live}. MATS cluster: no jobs submitted (cluster full).
- Primary runs from eca-b: {n_b} of 12 (the rest fall back to the original runs, capped, per PREREG Sec. 12).
"""
sec12 = """## 12. Files

- Code: `ecadata.py` (CA simulator, fixed permutation, exact forward-backward inference, brute-force check, query families),
  `ecanet.py` (network + stiffness and key heads; smallnet's `netdef.Block` unchanged; edited for round 2 at
  04:26Z, the as-run round-1 file is `results/raw/ecanet-as-run-r1.py`), `ecarun.py` (one run: training,
  heads, P0/D0, D1, chains, pools, D3; float64 version, the as-run 01:35Z version is `results/raw/ecarun-as-run-0135Z.py`),
  `analyze.py` (verdicts, tables, figures), `build_results.py` (this file), `render.sh`, `gpu/` (launch, driver, remote,
  ledger, zone-flexible copies of the shared gpu-up/gpu-run).
- Primary per-run outputs: `results/raw/main2/r{rule}-{arm}-s{seed}/` with `result.json` (all metrics, pool scales,
  training history summary), `test-estimates.csv` (per-instance pooled logits for every selector x pool x M, P0, the exact
  posterior and the outcome), `d1-pairs.csv` (the 2,000 stiffness pairs), `train-history.csv`.
- Replication: `results/raw/main/`; T = 8 hedge: `results/raw/t8/`.
- Derived: `results/summary.json`, `results/tables.md`, `results/verdicts.md`, figures `results/*.png`; the same for
  `results/replication/` and `results/t8/`.
- Pre-registration and change log: `PREREG.md`. Audit: `AUDIT.md`. GPU ledger: `GPU-LOG.md`.
"""
sec14 = """## 14. Audit and post-audit corrections

The independent audit (`AUDIT.md`, Opus 5.5, separate pass, 04:27Z) re-derived every pre-registered verdict and every number
in `results/summary.json` from the raw files with its own code (0 differences above 1e-6 relative), confirmed the float32
bug story, the identical training code, the fixed permutation, the stop rules and the GPU total, and found no leak of
exact posteriors or rule knowledge into training or selection. It flagged four prose overstatements (M1-M4) and thirteen
minor items (L1-L13). This report was corrected after the audit (04:35Z): the P1 headline is labelled as a secondary,
post hoc readout (M1); stiffness-first chains' smaller gain on rule-110 B2 is stated (M2); "every search arm makes things
worse" is replaced by the measured small-M gains (M3); the D1 label caveat and the "ceiling" wording are fixed (M4); and
L1-L8, L12-L13 are folded into Sec. 3, 7, 8, 10 and 12. No verdict changed. The audit's line numbers refer to the 04:10Z
version of this file.
"""
sec13 = "## 13. Full tables (primary runs; 3-seed means with 95% t-intervals)\n\n" + rd("results/tables.md")
out = (head.rstrip() + "\n\n" + static.split("## 2. What ran", 1)[-1].strip() + "\n\n" + mid.strip() + "\n\n" + sec9 + "\n"
       + tail.strip() + "\n\n" + sec11 + "\n" + sec12 + "\n" + sec14 + "\n" + sec13)
open(os.path.join(E, "RESULTS.md"), "w").write(out)
print("RESULTS.md", len(out), "chars; eca-b primary runs:", n_b, "; GPU total $", total)
