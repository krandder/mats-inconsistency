You are a rigorous critic reviewing a research design (read-only; do not modify any file). Write your critique as your
final answer (it will be saved to a file). Be concrete and terse; number your findings and mark each critical / major /
minor.

Context: MATS noisy-channel interpretability program. Target "M1": a sample-free (deductive, no sampling) estimator that
matches or beats sampling at equal compute on ARC's deep random He-ReLU MLPs at depths that grow with width (256 x 32,
x ~ N(0, I), W iid N(0, 2/n), no bias, small activation noise sigma = 0.001 relative). Estimands: per-neuron means of
the last post-ReLU layer, the first four moments of the last pre-activation, and tail probabilities 1e-4 / 1e-6.
Bars: ARC's factorized K=3 cumulant propagation reaches nMSE 2.2e-4 at depth 32 for 2.06e11 FLOPs; a Kerdock
mutually-unbiased-bases sampler reaches 6.7e-6 at 2.72e11 FLOPs. Prior lane (moments: ARC's descriptions, Gaussian,
copula, scale-mixture closures) found no sample-free win at depth 32.

Files (read them):
- /home/agent/workspaces/mats/work/boundary-family/DESIGN.md  (the design under review; section 1 lists the
  exploration measurements on development nets)
- /home/agent/workspaces/mats/work/boundary-family/lib/lrg.py   (log-radial Gaussian push: tilting, Stein, delta method)
- /home/agent/workspaces/mats/work/boundary-family/lib/gmprop.py (Gaussian-mixture propagation, splits, merges)
- /home/agent/workspaces/mats/work/boundary-family/lib/gauss_relu.py (exact Gaussian ReLU moments)
- /home/agent/workspaces/mats/work/boundary-family/scratch/diag1.py, diag2.py, diag3.py, diag4.py (oracles)
- Prior lane summary: /home/agent/workspaces/mats/outbox/arc-noisy-deep-2026-10-29.md (top 100 lines, and
  sections 4-5)

Questions:
1. Is the math right? (Gaussian tilting readout E[exp(rho) f(d)] = E[exp(rho)] E[f(d + Cov(d, rho))]; second moments
   with shift 2c; Stein's lemma for Cov(rho, ReLU(u_i + d_i)); the delta-method update of (rho', d'); the moment-
   preserving 3-way split; boundary split with forced gates.) Point to concrete errors in lrg.py / gmprop.py.
2. Is the diagnosis in section 1 (F1-F6) supported by what was measured, or are there confounds (e.g. sampling floors,
   oracle cells fitted on the same samples used as the reference, width-64 not representative)?
3. Will the proposed HBC family plausibly beat K3f at 256 x 32 at equal cost, and can its error-vs-compute curve track a
   sampler's 1/N? What is the strongest reason it will fail? Is the two-regime idea (cumulants until a switch layer,
   cells after) sound, and how should cumulants be converted to cells?
4. Are the early-kill measurements K1-K4 the right ones, with the right thresholds? What cheaper or more decisive
   measurement is missing?
5. Candidate B (backward / dual-weighted refinement): is it well defined and worth doing for means and for tails?
6. Anything in the cost model that is wrong or unfair against the sampler?
7. Any better non-moment family for this problem (that represents decision boundaries correctly) that the design misses?
