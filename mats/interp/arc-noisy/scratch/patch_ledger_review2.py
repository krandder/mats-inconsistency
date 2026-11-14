"""One-off: ledger, anet docstring and report review-section fixes after the claims review."""
L = '/home/agent/workspaces/mats/work/arc-noisy/LEDGER.md'
s = open(L).read()
def rep(a, b):
    global s
    if a not in s:
        raise SystemExit('MISSING: ' + a[:80])
    s = s.replace(a, b)
rep("""   Died: 10% gain. Closure errors at depth are mostly orthogonal to the mean direction (69% for K=2, 84% for K=3f).""",
    """   Died: 9% better than the exact Gaussian closure but 11% worse than kprop K=2 at depth 32 (claims-review correction).
   Closure errors at depth are mostly orthogonal to the mean direction (69% for K=2, 84% for K=3f).""")
rep("""Died on arithmetic: the shift is 10-1000x the closure error and extrapolation
   weights (1.8, -0.8 for 0.02/0.03) amplify closure errors by about 2x.""",
    """I rejected it "on arithmetic" (the shift is 10-1000x the closure error; extrapolation weights amplify closure errors).
   WITHDRAWN after the claims review: a large shift is what extrapolation is meant to cancel, and correlated closure
   biases need not be amplified. The reviewer's exploratory check (not preregistered, evaluation nets): K=3f-augmented at
   sigma 0.03 and 0.1, weights 1.0988 / -0.0988, gives depth-8 nMSE 1.83e-6 vs 1.92e-6 for the best single entry (+4%, CI
   above 1, cost 2.1e11); no tested depth-32 pair passed. Status: open, untested by me.""")
rep("""   Died: per-neuron marginals are enough for a one-step readout (item 5) but not for 8 more layers; the depth-32 means
   depend on the joint non-Gaussian dependence of the intermediate layers.""",
    """   Died: per-neuron marginals of the NEXT layer's pre-activations are enough for a one-step readout (item 5), but the
   empirical per-neuron marginals of h_24 (2^18 paths) with Gaussian dependence are not enough for 8 more layers; the
   depth-32 means depend on non-Gaussian dependence at the intermediate layers (for the replacements tested).""")
rep("""   layer 32 from l = 4 / 8 / 16 / 24 / 28: 0.009 / 0.10 / 0.11 / 0.25 / 0.74 at sigma = 0.001 (predicted by
   prod(1 - arccos(c_k)/pi): 0.012 / 0.032 / 0.12 / 0.37 / 0.62); unchanged within 15% up to sigma = 0.03; 2-4x smaller at
   0.1.""", """   layer 32 from l = 4 / 8 / 16 / 24 / 28: 0.009 / 0.10 / 0.11 / 0.25 / 0.74 at sigma = 0.001 (predicted by
   prod(1 - arccos(c_k)/pi): 0.012 / 0.032 / 0.12 / 0.37 / 0.62); unchanged within 16% up to sigma = 0.03; 1.4-3.9x smaller
   at 0.1.""")
s += """15. Claims review (GPT-6, 04:39-04:50Z): H2 also wins narrowly at depth 32 at sigma = 0.03 (+1-6% over 6.8e10-2.72e11,
   both CIs above 1) and at sigma = 0.01 at 6.8e10 only (+2%); report headline narrowed to the primary noise levels.
"""
open(L, 'w').write(s)

A = '/home/agent/workspaces/mats/work/arc-noisy/anet.py'
a = open(A).read()
a = a.replace("""Weights come from numpy's legacy RandomState (stream stable across versions), so the GPU truth runs rebuild them
bit for bit; weights_checksum() is compared on both sides.""", """Weights come from numpy's legacy RandomState (stream stable across versions), so the GPU truth runs rebuild the same
float64 weights (and then run them in float32; largest difference 1.5e-8); weights_checksum() of the float64 weights is
stored with each truth and checked by analyze.py.""")
open(A, 'w').write(a)

P = '/home/agent/workspaces/mats/outbox/arc-noisy-deep-2026-10-29.md'
r = open(P).read()
r = r.replace("""[CLAIMS REVIEW]""", """Claims review (GPT-6 via codex, high effort, 04:39-04:50Z; it regenerated every table and rebuilt H2 independently):
- Confirmed: the depth-8 H2 range 1e9-2.72e11 at sigma 0.001, 0.003 and 0.01 with both CI tests; no dev/eval leak; Kerdock is
  the best plain sampler across those ranges; C6 at depth 32 and sigma 0.1 passes at every budget; H1's numbers; the main
  competitor tables; ARC's counter is about 1.3x cheaper than full-cost matmuls for K=2 (1.32-1.34x) but not for factorized
  K=3 (1.01x), so the 1.3x sensitivity is a check, not a repricing.
- High: my depth-32 headline ("no win up to sigma 0.03") was false: H2 wins by 1-6% at sigma 0.03 over 6.8e10-2.72e11 and by
  2% at 6.8e10 at sigma 0.01. Fixed: headline restricted to the primary levels, wins reported.
- Medium: my rejection of noise extrapolation was not a test; its exploratory check found +4% at depth 8 (section 4).
  The multilevel argument assumed a sampled coarse level; reworded. The oracle numbers share truth paths with the target
  (slightly optimistic; stated). "Exact" marginals in the copula oracle are empirical quantiles from 2^18 paths (stated).
  The sampler is unbiased but not exact for degree-5 polynomials once radii are randomized and blocks are partial (stated).
- Low: HSC is 9% better than the Gaussian closure but 11% worse than K=2 (fixed); survival is within 16%, and 1.4-3.9x at
  sigma 0.1 (fixed); the mean-shift table now uses all 24 nets; H2's prediction at 3e11 just missed (0.97-0.99x; fixed);
  Sobol and chi-quantile costs are stipulated allowances (100 FLOPs per coordinate / per point), H1's final control matvec
  is uncharged (2n^2 per estimate, negligible), and H2's checkpoint grid leaves up to about 13% of the budget unused at 3e11.
- Noted here: the PREREG's "what was seen" noise-share line came from an early exploration net with a pre-final weight
  generator; the final weights give the same numbers (section 2).""")
open(P, 'w').write(r)
print('ok')
