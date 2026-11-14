# Independent audit of cm-interp Phase A

Completed 2026-10-28; independent final review is recorded in
`checks/final-independent-review.md` and `checks/review-response.md`.
The full checkpoint run finished at 19:43:52 UTC and all 90 original/frozen
source hashes matched at 19:44:55 UTC (`verify_source.py`).

Artifact paths below are relative to
`/home/agent/workspaces/inconsistency-search-astra-teacher-graphs-20261024/experiments/cm_interp_audit_20261028/`.

The reported direct gaps reproduce. Their interpretation needs narrowing:
they show the cost of replacing selected activations with these fitted
description families. They do not establish an irreducible information gap,
a causal share due to binding, or absence of hidden binding information in
stalled models.

The source snapshot and planned checkpoint extension are committed in
`5176c8bb`. This audit has not modified original source or Round 2 files. No new PFN
training or cloud resource has been started for this audit.

## Findings that affect conclusions

1. **The 19–81× headline is a mean of seed ratios.** The ten selected main-site
   means over five near-Bayes seeds give direct gaps 0.083021–0.388708 nats.
   Averaging each seed's gap/model-KL ratio gives 18.860857–80.803651;
   dividing the two seed means gives 17.908252–77.203502. Individual
   model/site gaps span 0.047846–0.545866 nats. The stalled checkpoint is
   excluded from those ranges. All 408 saved direct-rung summaries satisfy
   Delta=Lq-Lp to 8.88e-16. Raw arrays were not saved in Phase A, so its
   bootstrap intervals cannot be reconstructed from the JSON summaries.

2. **q0U does not remove binding at LAB or ALL.** Its labelled-token features
   retain source and question IDs, readings and observed answers. Together
   they reconstruct every source-associated residual sum; the ALL features
   reconstruct the complete observed input. At QQ, the collective feature map
   omits the norm-to-source assignment, but the real complement remains informative. q0 and
   q0U also use different, non-nested nonlinear feature bases. A comparison
   of their fitted losses is not an isolated binding intervention.

3. **The covariance diagnostic has a finite-M bias.** The within-cell
   calculation in `source/lib/r1est.py:86` uses denominator M. Even with an
   exact fixed density ratio, its expectation is `(M-1)/M Cov_q(G,w)`.
   At M=8 this is 7/8 of the desired covariance. An independent discrete
   known-answer test with true covariance -0.9 yields -0.789978, consistent
   with the predicted -0.7875; multiplying by 8/7 gives -0.902832.
   `cmlib._cov_point` shares this issue for small groups. Correct the
   covariance estimate and its SE, then reassess that diagnostic. The
   correction does not repair learned-ratio errors, self-normalization, or
   the bridge; direct mean and RMS gap estimates are unaffected.

4. **Bridge endpoint agreement does not validate the interior path.** An
   exact Gaussian control has ratio AUC 0.976 and a known path. A monotone
   distortion of the log ratio preserves AUC and reproduces the endpoint
   gap to 2.4%, yet gives 1.621 nats of interior error. With the true ratio,
   M=8 itself leaves up to 0.120 nats of interior bias in this control.
   Increasing M reduced that bias in this control. In Phase A, 60/64 high-AUC cases pass
   endpoint checks while 0/64 pass weight normalization. The derivative
   integral computed from the same estimated bridge mainly checks numerical
   integration of that curve. Neither test establishes correct density
   ratios or the shape of the population path.

5. **The infinite-variance claim in the estimator review is false here.**
   Final LayerNorm bounds the head means for fixed finite weights, and
   log-scales are clipped to [-6,3]. For a constant C depending on the
   checkpoint, `0 <= G <= C(1+||Q||^2)`. Query readings have a finite Gaussian
   mixture distribution, so exact direct gaps have every polynomial moment.
   The finite-grid score also has a polynomial envelope. Low empirical Hill
   estimates warn about finite-sample tail sensitivity; they do not prove
   infinite variance or justify discarding the direct estimator. The proof
   and checkpoint weight bounds are in `estimator_review.md`. This argument
   does not apply automatically to importance-weighted estimators.

6. **The mechanism claims exceed their controls.** Real-state noise addition
   does not isolate an off-manifold share: it changes perturbation size and
   dependence, and can break useful correlations. Four directions that
   recover the measured G-gap can identify readout sensitivity without
   identifying four missing computational variables. A pushforward through
   the actual block uses the block's computation; an advantage over a
   separately fitted Gaussian does not itself compress or explain that
   computation. These concerns agree with the two original reviews.

7. **Some proposed Round 2 controls need the same care.** Real donor tokens
   lie on their marginal activation manifold, but combining them with a
   recipient's other tokens can leave the joint manifold. Differences of
   RMS gaps are differences of norms, not an additive decomposition into
   form and binding. A full-cut zero control requires a joint donor
   preserving c and the target convention. If only unordered reliabilities
   are preserved, scoring against the recipient's full-Bayes target answers
   a different question from recomputing the donor's target.

8. **Output-space ratio existence still needs assumptions.** Having 141
   nuisance dimensions and 11 free outputs does not prove a full-rank
   nuisance-to-output map or absolute continuity of the real output law
   with respect to the replacement law. Scale clipping can also create
   singular faces; it need not invalidate a ratio if both distributions
   share them appropriately. METHOD's claim that the output ratio always
   exists is not established by the supplied dimension argument or tests.
   Direct estimates require neither density ratios nor those assumptions.

One further qualification affects feature naming. At scale-normalized sites,
`QTok.sample` normalizes the Gaussian draw, but `logpdf_proj` scores a
projection of the Gaussian before normalization. This is a surrogate score,
not the exact projected log density of the replacement distribution. It
does not alter the direct loss measured from actual replacement samples.

## What the checks support

The paired direct estimator is correct for the implemented experiment.
The split-half estimator is unbiased for the squared conditional mean gap,
given independent inner halves and a fixed fitted q. Its truncated square
root is not unbiased and is nonregular near zero. Cell bootstraps estimate
evaluation-episode uncertainty conditional on q and the checkpoint; they
omit q-fit and training-seed uncertainty. Their nominal coverage at these
finite sample sizes has not been established by the algebraic checks.

The conditional sampler is correct for observed inputs given the source-
indexed sufficient statistics c=(Q,S). It copies latent dictionary entries
such as true v and the hidden query answer, so it should not be described
as resampling the entire latent world. Those fields do not enter the
observed-input features or the integrated Bayes score in this experiment.

All 18 existing tests pass in an isolated one-thread run. Independent checks
also exercise the actual per-cell r1est formulas, which the original suite
does not cover at M=8. Bridge implementations agree within 1.8e-15; direct
and RMS known-answer tests pass. Exact commands and imported-file hashes
are in `commands.md` and the before/after source manifests. Imported live
files were not modified and hash-matched to the frozen evidence; the historical
checkpoint extension uses its own frozen-source adapter.

## Historical stalled/acquired checkpoint extension

All eleven previously selected checkpoints were evaluated at h1-QQ, with
20,000 fresh fit episodes and 3,000 fresh cells with eight conditional draws
each. Seeds, exact feature maps and source hashes were fixed before this
evaluation in `PROTOCOL.md` and `checks/RUNNER_FREEZE.md`. Both baseline
feature matrices are exact subsets of the union, including their original
duplicated reading columns. No checkpoint was dropped or reselected.

The table shows RMS of the conditional mean increase in KL to Bayes, in
nats. Smaller is better for this particular replacement experiment. The
model's real KL is a mean, so it is shown for scale rather than as the same
quantity as the RMS gap.

| Selected checkpoint | Status | Real mean KL | q0 RMS | q0U RMS | Union RMS |
|---|---|---:|---:|---:|---:|
| w16_h1_L2_mix4_s3 | stalled | 0.1442 | 0.4274 | 0.1936 | 0.1861 |
| w16_h1_L2_mix4_s4 | stalled | 0.1443 | 0.4967 | 0.1968 | 0.1897 |
| w16_h2_L2_ds4_mix4_s1 | stalled | 0.1425 | 0.3543 | 0.2327 | 0.2129 |
| w32_h2_L2_mix4_s3 | stalled | 0.1418 | 0.4168 | 0.1924 | 0.1825 |
| w16_h2_L2_s3 | stalled | 0.1621 | 0.5534 | 0.2182 | 0.1993 |
| w16_h2_L2_s4 | stalled | 0.1624 | 0.5164 | 0.1706 | 0.1544 |
| w16_h1_L2_mix4_s0 | acquired | 0.0343 | 0.6915 | 0.7829 | 0.6871 |
| w16_h2_L2_ds4_mix4_s0 | acquired | 0.0194 | 0.8724 | 1.0754 | 0.8628 |
| w32_h2_L2_mix4_s0 | acquired | 0.0068 | 0.8823 | 1.1291 | 0.8698 |
| w16_h2_L2_s0 | acquired | 0.0367 | 0.6038 | 0.6483 | 0.4300 |
| w16_h2_L2_s2 | late escape | 0.0459 | 0.6445 | 0.4918 | 0.2854 |

q0U has the smaller RMS gap in all six stalled checkpoints. Four acquired
siblings prefer q0. The late-escape checkpoint also prefers q0U, so this
ordering is not a universal marker of stalled versus acquired behavior.
These are descriptive point-estimate comparisons on this selected panel,
not independent tests of a population-level classifier.

Adding the seven source-specific q0 columns to q0U reduces the RMS gap,
relative to q0U RMS, by 3.6–9.5% in the six stalled checkpoints,
12.2–33.7% in the four acquired siblings and 42.0% in the late-escape
checkpoint. This includes a large improvement for the late-escape model
that the raw q0/q0U ordering misses. The percentages describe reductions
in a norm, not additive causal shares of binding. Feature form, covariance
refitting, the hidden-state regression objective and the real recipient
context still confound a mechanistic interpretation.

No family closes by METHOD's criterion. Even the stalled q0U RMS gaps are
0.171–0.233 nats, compared with epsilon=0.1 times real KL, about 0.014–0.016.
The results therefore do not support “no structure beyond q0U” or “binding-
free features nearly suffice.” They support a smaller incremental benefit
from these source-specific features in these stalled checkpoints.

`CHECKPOINT_PANEL.md` gives pointwise paired cell-bootstrap intervals,
signed mean gaps and squared-gap estimates. All intervals condition on the
fitted descriptions and these historically selected checkpoints; they do
not cover checkpoint selection or training-seed variation, and are not
simultaneous confidence bands. No inference about success rates of new
training runs is warranted.

The independent summary script checked 66 file hashes and recomputed 462
numerical entries from the saved losses, with maximum discrepancy zero.
It also checked shared cell indices, noise prefixes, CV folds, feature
subset maps and recorded validation gates. It did not refit descriptions
or rerun network forwards. For the acquired Gaussian s0 checkpoint, the
q0U-minus-q0 RMS interval includes zero despite its positive point estimate.
Two acquired mixture checkpoints likewise have union-minus-q0 intervals
that include zero. The table should not be read as universal significance.

The full run took 177 seconds on one CPU thread. All eleven adapters matched
native outputs exactly, including ds4 and Gaussian heads. On the adapter
validation cells, quotient normalization changed G by at most 3.31e-5 nats.
The largest full-run grid discrepancy on the checked real/synthetic subset
was 7.54e-8 nats, below the prespecified 1e-6 threshold. This is a checked
subset, not a certified integration-error bound for every draw.

Raw losses, predictions, noise, fits and inputs are retained locally under
`results/historical-panel-v1/`, with SHA256 links in its manifests. The raw
NPZ files follow the repository's existing binary exclusions; source code,
JSON summaries and the report are tracked. This avoids relying on the
summary-only evidence available for the original Phase A run.

## A stronger follow-up test

For each fresh episode, retain Q, sorted source residual norms U, all IDs,
observed answers and each source's residual direction. Enumerate all six
assignments of norms to sources. Rebuild complete inputs and weight them
by the exact posterior assignment probabilities conditional on Q and U.
This avoids stitching hidden activations and needs no fitted q.

For predictive densities f_a and assignment weights w_a, measure
`sum_a w_a KL(f_a || sum_b w_b f_b)`. It is nonnegative and vanishes exactly
when predictions agree across every positive-probability assignment for
that retained context. The unordered Bayes predictor is a null control.
The full Bayes predictor is a positive control: its weighted mixture must
equal the unordered Bayes predictor, and its average divergence is the
conditional mutual information lost by forgetting the assignment. A small
finite-sample estimate describes sensitivity on the evaluated contexts;
population bounds need valid uncertainty and numerical error control.
Neither establishes absence of unused hidden information.

The exact assignment weights and proof are in `interpretation_review.md`.
This proposal has been sent to the Round 2 owner to avoid duplicate work.

## Evidence

- `source-manifest.json`: 90 SHA256-frozen Phase A files.
- `estimator_review.md`, `audit_direct_estimators.py`,
  `direct-estimator-audit.json`: aggregation, algebra and finite-moment proof.
- `bridge_review.md`, `exact_bridge_audit.py`, `exact_bridge_results.json`,
  `direct_rms_audit.py`, `direct_rms_results.json`: independent known answers.
- `interpretation_review.md`: feature reconstruction, sampler qualifications,
  density-score caveat and exact conditional assignment proposal.
- `pytest-18.log`, `commands.md`, `source-sha-before.txt`,
  `source-sha-after.txt`: bounded execution and source integrity checks.
- `CHECKPOINT_PANEL.md`, `checkpoint-panel-summary.json`,
  `summarize_checkpoint_panel.py`: independent raw-array recomputation for
  all eleven historical checkpoints.

Recommendations: keep direct gaps as primary measurements; correct the
covariance factor and headline aggregation; retract the information-ablation,
off-manifold-share and infinite-variance interpretations; validate any
tempering path with exact-ratio/inner-draw controls before using its shape.
