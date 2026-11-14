# Response to the paper review

Checked 2026-10-28 at 14:03 UTC against the revised source and compiled PDF.
The paper remains local. No training or new checkpoint evaluation was needed.

1. Evidence manifest: `evidence.md` is present and maps E1–E5 to exact paths,
   JSON selectors and hashes. The added GPU conformance records document their
   small batched engine-test scope and the softmax-invariant key-bias exception.
2. AUC selection: the highlighted results are now “selected summaries of
   prespecified analyses.” All 20 width-16 baseline results, fixed directions,
   eligibility counts and intervals are exposed in `evidence/auc-all-landmarks.md`.
   The abstract describes an association in this cohort. The operational-risk-set
   sensitivity remains separate and is not presented as resolving selection bias.
3. Context use: the abstract now says “nonquery-context use”; the paper does
   not infer label dependence from beating query-only Bayes.
4. Scope: the abstract explicitly says “one fixed 3-cycle,” and the title limits
   the result to selected stalled models. The limitations also state that the
   intervention changes the joint distribution despite staying on support.
5. Uncertainty and conformance: paired SE ranges were added for scalar-total
   comparisons and intervention distribution-change ratios. Acquisition is a
   three-evaluation criterion spanning 2k updates, not indefinite persistence.
   The source states that softmax-invariant key-bias updates differ numerically.

All four rendered pages are clean and the build warning scan passes; see
`BUILD.md`. Revised PDF SHA-256:
`47061e608cdfaac9ddac26de90434b47c82f1064d581aa7f2c44deb78d5eb1ee`.
