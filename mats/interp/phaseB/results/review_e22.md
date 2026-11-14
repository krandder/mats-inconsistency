Model: GPT-6 (Codex)

Read-only review; checked 2026-10-29 17:53 UTC. No Python or model execution.

1. **High: decreases confused with sign reversals.** Draft lines 6–7 say 46/48 penalise a big miss. `results/e22.json:per_net[*].large_p1` is negative in **38/48**; 46/48 merely decrease relative to `large_p0`. All 48 start positive. Fix: “The score fell in 46/48 and reversed sign in 38/48”; qualify lines 70–71 similarly. The mean reversal is correct.

2. **High: mechanistic conclusion exceeds the intervention.** Draft lines 72–75 (“works toward”, “runs through”) and plain-words lines 8–10 infer the natural mechanism from an off-distribution intervention. Equalisation removes all within-group routing differences, including dependencies on identity/labels, not specifically error weighting; downstream disruption remains an alternative (`PREREG-B.md:1594–1596`). Fix: “The departure is sensitive to this intervention; the toy’s directional prediction fails. This does not identify the natural error-weighting mechanism.” A small pooled Spearman correlation does not establish mild *direct* dependence or exclude nonlinear/head-specific dependence (`e22_post.py:31–35`).

3. **Medium: E23 completion and blinding overstatement.** Draft line 78 says “tested”; no local `results/e23.json` or `results/e21/*.npz` exists at review time. Amendment 30 says registered before measurement, after E22 discovery (`PREREG-B.md:1601–1614`). That is a sound prospective test on fresh nets if E21 outcomes remain unseen; prior training-log inspection is disclosed. Fix: “registered for testing”; substantiate outcome blinding with dated measurement logs when available. Local absence cannot prove remote nonmeasurement.

4. **Medium: plain-language A/B claim hides exceptions.** Draft line 12 implies every B base leans toward larger errors; `e22_post.json:w32_h2_L2_mix4_ws_s2.spearman` is −0.061. Fix: explicitly say 6/6 A and 5/6 B bases, post hoc, limited to this cohort. Technical 11/12, 24/30 two-head, 9/9 positive-rho, and 18/18 one-head exceptions check out; retain those limits.

Static implementation matches registration: head reshape, rows 4/9/14, source groups, per-head mass preservation and remaining forward operations (`e22_run.py:18–52`; `pfn.py:37–77`; `nnoise.py:64–70`). No undisclosed implementation deviation found. P128/P129, means, configuration table, post-hoc summary and −0.431 correlation match JSON. Post-hoc analyses are labelled.

Hashes match current code/list; E22 NPZ mtimes 17:48:57–17:49:20 follow the 17:48:55 hash. Summary/post-hoc files precede E23’s 17:51:46 hash. These support chronology, not independent timestamp certification.

**Survive:** intervention sensitivity, mean reversal, descriptive signature. **Reword:** mechanism, plain story, E23 status. **Fail:** P128/P129 and 46/48 sign reversals.
