Model: GPT-6 (Codex)

Checked 2026-10-30: read-only source, metadata, NPZ spot-checks, JSON and SHA-256 checks; no Python or recomputation. All six amendment-55 hashes match. Marker formulas, ctrl/width/depth filtering, acquisition cutoff, AUC ties and 2,000-resample bootstrap implement the registration, except below. Tables and prediction flags match JSON; aggregate NPZ results were not independently recomputed.

1. **High: step-zero duplicates violate registration.** `e47_summ.py:50` explicitly exempts `br0`, assigning K=-1; amendment 55 (`PREREG-B.md:2639`) excludes brK through K. `train_br.py:32` and `:58` preserve the parent's initialisation; the stored lab@0 values for `e43_br0_s2` and `e37_ctrl_s2` both equal 0.7575643658638. This changes weighting and potentially P192. **Fix:** exclude br0 at step 0, regenerate step-zero estimates/P192, disclose correction.

2. **High: temporal and anatomical precedence are unsupported.** JSON `auc.kl@3000` already gives 0.65 [0.53,0.76]; lab exceeds xq at 2,000/3,000, and sign share is an output measure, not block-local. Five checkpoints and unequal univariate markers cannot establish “first in block 1,” “only after reliability,” or block-2 causality (`update57_draft.md:45-47`). **Fix:** report earliest *sampled* AUC≥0.8 at 5,000 (sign share), and strongest discrimination at 10,000; remove causal ordering.

3. **High: “fixed at initialisation” contradicts prior evidence.** Update 50 reports one A→B change under the same initialisation and explicitly rejects “fixed” (`cm-interp-phaseB-2026-10-28.md:5141,5147`). **Fix:** replace with “strongly associated with initialisation under tested stream replacements,” including the exception, throughout both drafts.

4. **Medium: registered clusters miss donor dependence.** `e47_summ.py:26-27,62-67` clusters hybrids only by recipient; reciprocal hybrids share donor/recipient weights (`gpu/jobs_e44a.txt:1-4`). Registration compliance does not establish independent clusters or population generality. Branch exclusions also change the cohort across steps. **Fix:** qualify intervals; report per-step counts, experiment-stratified/common-cohort and donor-aware sensitivity analyses.

5. **Medium: P192 is not evidence of functional invisibility.** JSON step-zero intervals extend beyond 0.35–0.65; four near-chance directed AUCs cannot exclude nonlinear/multivariate information. P191's interval includes 0.75. **Fix:** describe registered point-estimate passes/failures, not absence of signal.

**Survive:** JSON transcription, P191 scoring, descriptive later discrimination. **Reword:** plain-words timing, generality and P192 interpretation. **Fail:** complete deduplication, deterministic initialisation and claimed block-first/causal ordering.
