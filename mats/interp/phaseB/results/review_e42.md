Model: GPT-6 (Codex)

Checked 2026-10-30, read-only; no Python or model execution.

1. **High: “nine new lineages” overstates independent end-state replication.** Draft lines 10, 13, 23 contradict `PREREG-B.md:2422-2425`: seeds 0–2, 4 and 7 reuse initialisations/streams whose endpoints were already measured; only their trajectories are new. Only seeds 8–11 are new lineages, all B. Fix the title and plain-words addition: nine trajectories beyond E16, including four new B lineages; no new A endpoint sample. Restrict “end states are general” to the observed sample.

2. **Medium: endpoint increases do not establish post-acquisition timing.** Draft lines 5, 57 infer growth after learning source weights from P181's 3k-versus-90k comparison (`results/e42.json:predictions.P181`). No acquisition-aligned test is performed. Similarly, “before 10,000” (line 6) should be “by the 10,000-step checkpoint”; no intermediate samples locate the fall. Fix: report 5/6 endpoint increases and early attention differences descriptively. Failed window thresholds reject the registered population prediction, not every late-refinement account. Specify three checkpoints per window, versus E31's six/eleven; “10 steps” at line 59 means ten checkpoints.

3. **Medium: “types … set early” needs qualification.** Draft lines 23, 58 exceed E43's evidence: Update 50 records one A-to-B switch among eleven br0 copies, specifically seed 2, and one replacement stream per parent. E42's 3 A/6 B observational trajectories cannot establish commitment. Fix: “consistent with early attention differences and strong type persistence under the tested stream replacements”; retain the exception.

4. **Medium: cost ratio is not an attributable share.** Draft line 57 calls negation/flip the “share of the sign effect.” `e26_run.py:87-97` computes separate finite-ensemble geometric-averaging costs. Update 40 explicitly disclaims an exact parity share. Fix: call it the negation/flip cost ratio.

**Survive:** registered families, selection/typing, windows, table/ranges, coverage and P179–P182 outcomes match; six code hashes match. Saved plumbing reproduces E31 exactly and E35 within 2e-8. **Reword:** novelty, population generality and timing/commitment interpretations. **Fail:** P179, P180, P182; P181 survives. Descriptive endpoint separation survives.
