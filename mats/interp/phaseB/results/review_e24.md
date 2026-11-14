Model: GPT-6 (Codex)

Checked 2026-10-29 18:40Z by file reads, jq, sha256sum and stat; no Python or model execution. Draft references below mean update33_draft.md.

1. **High: exclusive localisation is unsupported.** Draft:13–17,55–56 says “produced entirely in block 1” and dismisses other routes. e24_run.py:30 detaches the reference attention row, excluding its reading-dependent derivative. Moreover, :34 replaces values globally before a@v, affecting every receiving token, including paths through other tokens into block 2. This implements the registration but does not isolate one gathering step. Fix: say these block-1 interventions largely remove the *detached-row weight contrast*; do not conclude that other routes contribute nothing.

2. **High: “whole difference, in all 48” is false as stated.** Draft:5–6 upgrades a pooled signed result to universal elimination. results/e24.json:counts.both establishes only >50% removal in 48/48; per_net.w32_h2_L2_mix4_s7_nb.large.both is 1.619. The pooled 1.0586 overshoots zero; cancellation can conceal remaining differences. Fix: “The joint swap removes 106% of the pooled contrast and exceeds 50% in every network,” with the derivative qualification above.

3. **Medium: nonadditivity does not identify the bias's origin.** Draft:19,57–62; JSON:pooled.large gives joint 1.0586 versus singles summing to 0.7180. This supports a descriptive intervention interaction, not exclusive generation by attention–value interaction. Joint replacement reconstructs the spread pooled vector at the query token; its own residual is already unchanged across E12 pairs (e12_data.py:26–32). Downstream nonlinear processing can also produce nonadditivity. Fix: report the 0.3406 excess joint removal and distinguish reconstruction/sufficiency from mechanistic origin.

4. **Medium: natural donors do not guarantee on-distribution hybrids.** Draft:14–15; e24_run.py:60–61 mixes spread components with big-miss components. Fix: “matched natural-activation transplants”; retain a hybrid-state distribution caveat, especially for single swaps.

5. **Medium: denominator caveat is selective and insufficiently evidenced.** e24_summ.py:20–22 defines a signed ratio, undefined at zero and unbounded otherwise. JSON:pooled.middle.ref_diff_mean is 0.01528 versus large 0.01634; this alone does not establish “nearly cancel.” All three ratios share the denominator, including the unbracketed 90%. Fix: show subgroup denominators and cancellation diagnostics; qualify both as well as singles, and explain overshoot.

Verification: correct tokens, all heads, detached rows/donors, and post-LayerNorm/projected values before pooling; remaining operations match pfn.py:37–77. JSON numbers, counts and prediction decisions match. SHA256 matches prereg_hashes.txt:230–232; hash-file mtime 18:37:40.614 precedes log births .615–.617, supporting post-hash launch. Self-identity evidence is limited to the disclosed 50-row check.

Survive: registered implementation, P133 pass, P134 fail. Reword: interaction, percentages, distribution and middle-range claims. Fail: exclusive localisation and universal whole-difference removal.
