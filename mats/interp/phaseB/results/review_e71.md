Model: GPT-6 (Codex)

Checked 2026-10-31, read-only; no Python or model execution. Source inspection supports the architectures, paired initialisation, episode streams and E71 loaders. Metadata confirms all 36 runs. E70's incorrect block-2 head split cannot affect these block-1-only head/trajectory measures. Saved tables and JSON agree.

1. **High: causal exclusivity exceeds the design.** `update83_draft.md:3,22,71` says count “decides” and “sets” KL. Amendment 79 (`PREREG-B.md:4034`) also identifies changing total attention dimension, parameter count and output-map initial scale; CPU/GPU is another confound, only checked on four ctrl seeds. `results/e71.json:contrasts` gives the two-head width interval [-0.227,0.704], compatible with substantial effects. Fix: say the tested layouts support the count account; width equivalence and a uniquely causal head budget remain unproved. Three heads are not necessary for A: two-head A nets exist.

2. **High: “unused” is contradicted by measured attention.** Draft lines 6,22,72 promote missing classifications into missing function. `results/e71_heads/e71_h3w16_s{0,2}.json:olab_h[1]` is 0.771/0.764: the unclassified head attends strongly to other sources' labelled tokens. Fix: say “meets neither registered criterion”; attention mass does not establish computational roles, redundancy or inactivity.

3. **Medium: failure interpretations become universal mechanisms.** Draft lines 7,22,74 retain costs “following the count” and exclude competition from timing. `results/e71.json:predictions.P278/P280` fail; `PREREG-B.md:4028-4033` specifies the corresponding failed account predictions. Fix: acknowledge the cost prediction's partial failure and reject the registered early-pair prediction specifically. Equal pooled checkpoint medians do not exclude any competition effect; h4w16's median is 5,000.

4. **Low: missing undecided branch.** `e71_summ.py:98-103` always assigns Boolean P275-P277 verdicts, contrary to `PREREG-B.md:4012`. Fix: return undecided below six acquired. Current verdicts are unaffected. Step zero is also omitted from the search, harmless here because all 48 initial trajectories have zero pairs.

Survive: P275-P277/P279, numerical tables, P278/P280 failures, and preregistered new-arm testing with disclosed plumbing; this still uses validation-selected models (`train_sw.py:143`). Reword causal headlines, functional roles and timing. Fail: unused-head and uniquely count-determined claims.
