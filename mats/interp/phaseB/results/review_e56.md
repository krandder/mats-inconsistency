Model: GPT-6 (Codex)

Checked 2026-10-30 by reading code, metadata, JSON, tables, logs and hashes; no Python or training/evaluation reruns. Binary checkpoint/NPZ contents were not independently recomputed.

1. HIGH: Causal exclusion and equivalence are overstated. `update70_draft.md:18,46,48` and item 51 say “no effect at all” and “ruled out.” `results/e56.json:pairs.sep/ctrl.rom_ratio_ci` spans 0.61–1.40; ten acquired pairs, differential stalls and CPU/GPU confounding limit inference. Uniform-pool similarity supports a narrower conclusion, not exact equivalence (`pairs.unifsep/unif.kl_ratio_ci`: 0.90–1.15). Fix: “This intervention did not produce the registered median-c increase or mean-departure reduction; uniform-pool results remained close.” Apply this qualification to the plain-words addition too.

2. MEDIUM: The sharper split is a post hoc sample description, not an established format effect. Evidence: `results/e56_tables.md` seed rows; `e56_summ.py:71` defines A by negative rom_middle. Thus type is itself departure-derived, and comparisons condition on acquisition. Fix: explicitly say “post hoc, among these ten acquired nets”; distinguish the c association from partly definitional type/departure association. Avoid attributing increased separation to format without a matched-device replication.

3. MEDIUM: “Weakest caps seen at this width” (`update70_draft.md:47`; item 51's “any usual network”) exceeds the comparison. `results/e55_geom.json:e44_inp_r6_d9.c` is 5.396, above two sep A nets. Fix: “All four exceed the twelve E37 ctrl nets,” rather than a width-wide superlative.

4. LOW: Summary precision. Line 14 juxtaposes 11-net ctrl arm summaries with ten-pair ratios (`e56_summ.py:64,80`); label those denominators separately. Line 49's “geometric median” should be “median of tokenwise A/U” (`lngeom.py:32–33`).

Survive: registered sep routing, shared-parameter initialization and episode stream, readings-column evaluator, comparable labelled-token c, and P219/P220 failures plus P221 success. Code hashes match; metadata supports 24 CPU/90,000-step runs; tables match JSON. Need rewording: causal conclusions, equivalence, post hoc split and comparison scope. Fail: categorical exclusion of the shared-column explanation and the claim that all four caps exceed prior width-32 nets.
