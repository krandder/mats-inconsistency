You are a rigorous critic reviewing a research design (read-only; do not modify any file). Write your critique as your
final answer (it will be saved to a file). Be concrete and terse; number your findings and mark each critical / major /
minor. Check the math yourself where you can (you may run small read-only Python checks with
/home/agent/.venvs/arc-noisy/bin/python if useful; do not write files outside /tmp).

Context: MATS noisy-channel interpretability program. ARC's random He-ReLU MLPs: x ~ N(0, I_n), z_l = W_l h_{l-1},
a_l = ReLU(z_l), h_l = a_l + sigma r_l eps (r_l = RMS of a_l, sigma = 0.001), W iid N(0, 2/n), no bias. Estimand:
per-neuron E[a_L]; metric nMSE = mean_j (est_j - truth_j)^2 / mean_j Var(a_L,j). Target M1: a sample-free estimator that
matches sampling at depths growing with width (256 x 32: Kerdock sampler 6.7e-6 at 2.72e11 FLOPs; ARC's factorized
K=3 cumulant propagation 2.2e-4). Prior lanes: Gaussian closures, cumulants, Gaussian cells, latent-conditional
Gaussians all lose information at every re-projection, and at the critical point the errors are not damped
(Gaussianizing the true state at layer 24 gives 5.8e-5 one layer later but 2.6e-4 at layer 32). Kelvin (PI): "Let us
seek a family that is fully non-gaussian, even if its bad at the input, and hope that it at least scales on depth."

Files (read them):
- /home/agent/workspaces/mats/work/nongauss-family/DESIGN.md (the design under review; section 1 has
  exploration numbers from one development net)
- /home/agent/workspaces/mats/work/nongauss-family/scratch/explore_shape.py (how those numbers were computed)
- /home/agent/workspaces/mats/outbox/boundary-family-2026-10-29.md (prior lane; section 6 is the handoff that
  defines the criterion and candidates (a)-(d))
- /home/agent/workspaces/mats/outbox/random-net-phases-2026-10-29.md (critical-phase structure; summary)

Questions:
1. Is the homogeneity reduction right (E[f(z)] = E[rho] E_nu[f(u + delta)] for degree-1 homogeneous f, nu the
   rho-weighted law of delta; sampling rho' independently from its unweighted law)? What does the noise break, and by
   how much at sigma = 0.001? Is the gnomonic chart well defined (rho > 0) and is u = E z / |E z| the right centre?
2. Are the families well defined and do they test what they claim (PI, PR, PC3I, PC3R, ATTk, and the Gaussian
   controls)? Is "fully non-Gaussian" honestly satisfied (e.g. independent components with exact empirical marginals;
   two-point components)? Is ATTk's principal-component matching meaningful, and what better attractor test exists?
3. Is the criterion (single-cut R_F(l; L) growth over L, composition from layer 32, two-cut coherence) the right
   operationalization of "scales on depth"? Propose concrete numeric pass/fail thresholds that account for the
   fit-set floor (2^19 paths, about 2e-6), the 2^20-sample propagation noise, the 2^22-path truth, and the bootstrap
   noise that composition injects at every layer.
4. What is the strongest reason this will fail, and what cheaper or more decisive measurement is missing?
5. Is there a better fully non-Gaussian family for this structure (collapse direction, remembered scale, 15-40
   gate-relevant neurons per deep layer, platykurtic skewed top components) that the design misses? Be specific.
6. Does any candidate have a plausible deductive (sample-free) push whose cost scales to 10M+ parameter models? Which
   part of the proposed push sketch is wrong or infeasible?
