Model: GPT-6

Checked 2026-10-29: source, hashes, JSON, tables, log and prior reviews; no Python or numerical reruns. “Draft” = update41_draft.md.

1. **High: causal exclusion is unjustified.** Draft:5,16,19,45 says “not a route” and infers unused values. P156 rejects the registered halving prediction, not every contribution. Redundant paths, compensation from renormalisation, cancellation in pooled costs, and small removed mass remain possible; final-net ablation cannot exclude a training-time role. **Fix:** “This ablation leaves pooled sign dependence nearly unchanged.” Retain the marker association; restore the registered off-distribution caveat (`PREREG-B.md:2030`).

2. **Medium: plain words confuse metric stability with output stability.** Draft:3-4,18 says “changes nothing,” “answers barely moved,” and “1% or less.” JSON `groups.B.kl_base.ratio=1.010987` means 1.10%; stable KL-to-Bayes does not establish similar predictions. “All nets” also obscures aggregation: `per_net.w32_h2_L2_long_s2` has roughly 7.5%/10.3% flip/negation increases. **Fix:** report pooled metrics explicitly, use “about 1.1%,” and replace “lost” with “greatly reduced.”

3. **Medium: next test needs a precise intervention and narrower inference.** Draft:46's paired average removes activation oddness under that source's negation, but includes cross-source interactions and, for labelled tokens, the residual stream. It does not isolate linear own-source pooling. **Fix:** specify post-block versus attention-output replacement, hold other sources fixed, patch each source separately, include identity/sham controls, and report base KL alongside negation cost. Treat reductions as evidence about those activations; nonlinear downstream effects prevent an additive attribution.

4. **Medium: numerical coverage is unverified for the patch.** `e32_run.py:105` discards returned mass; `e26_run.py:94-96` uses finite-grid quadrature. **Fix:** retain patched/unpatched coverage and qualify “KL-exact” pending coverage validation.

**Survive:** matching hashes; CPU forward matches `../tiny-pfn/pfn.py:37-77` structurally; mask correctly targets 12 labelled rows, two columns each, all heads, block 1 only, with renormalisation (`e32_run.py:24-64`). xq divides by 12 and averages equal batches. P156/P157 match registration and JSON: 1.017157, 2/6, both fail (`e32_summ.py:30-43`). Numerical forward equivalence was not rerun. **Reword:** plain words and proposed test. **Fail:** categorical route exclusion and “changes nothing.”
