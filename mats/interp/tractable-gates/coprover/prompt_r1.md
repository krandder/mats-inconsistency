You are the co-prover and checker for a research lane (read-only sandbox; the lane directory is your working dir).
Question (from Kelvin): which activation and gating functions keep the benefits of gating while making the value function
and fronts smooth enough that one-way linear summaries (the cut-rank sense of ../cut-rank: the eps-rank r(eps) of the
front-to-back cut operator, against Monte Carlo's 1/eps^2) and deductive estimators beat sampling?

Read notes/theory.md (the claims to check), then as needed lib/acts.py (exact Hermite spectra), scratch/ray_family.py and
logs/ray_family_vR1_partial.log (exact ray-front ranks), scratch/hermite_classes.py and logs/hermite_classes.log,
scratch/depth_radius.py and logs/depth_radius.log, and the cut-rank lane's statement ../cut-rank/FORMAL.md and its report
/home/agent/workspaces/mats/outbox/cut-rank-lower-bound-2026-10-30.md (sections 1-4) for the setting.

Tasks, in order. Be concrete; give corrected statements where something is wrong.
1. Lemma P, Corollaries P1/P2, Lemma R, Corollary R1: check each line (hypotheses, measurability, the contraction,
   the scale-mixture reduction, the density-ratio bound). Is Corollary R1's handling of S near 0 right? Give the weakest
   clean hypothesis under which Theorem A(i) (polylog r(eps) for super-polynomial Hermite decay, any front law, one back
   layer) is PROVEN for Gaussian-tailed S (unbounded scale), or say why it fails.
2. Section 2's formulas: gsrelu (r^k S' c_k^relu), GELU = gsrelu(1) - npdf, the logistic normal scale mixture (scale
   2K, K Kolmogorov), and the resulting softplus/SiLU/tanh representations; the Laplace asymptotic
   log c_k^2 ~ -2 pi sqrt(k)/(beta s) for softplus (check the constant against Hille's strip theorem in the probabilists'
   normalisation).
3. Section 3: is the ray-model computation right (kt(t) = sum_k (E_R c_k)^2 t^k, Vbar = sum_k [E c_k^2 - (E c_k)^2], the
   Gegenbauer energy shares, rank counting)? Is Theorem A(iii)'s Laplace heuristic (E_l ~ l^{1 - 2 alpha} for
   b_k ~ k^{-alpha}; exp(-c' l^{2 gamma/(1+gamma)}) for b_k ~ exp(-c k^gamma)) right? Can it be made rigorous for the
   power class at least?
4. Section 5 (depth): check R_d = min(R_1, f^{-1}(R_{d-1})) for compositions of a normalised positive power series and
   the phase statements (ordered / chaotic / critical). Is "smoothness survives depth only in the ordered phase" a fair
   summary? What changes at finite width?
5. Sections 6-7: GLU kernel shift; bilinear finiteness; the scale instability of plain GLU stacks; the density claim for
   strictly monotone C^1 gates; the GELU/SiLU fold singularity and the failure of the L^2-density condition; the
   fuzzy-ray transfer for softplus. Which are right, which need repair?
6. Section 8's practical prediction: is the argument that soft (beta s ~ 1) bias-free gates are ordered, so fronts are
   tight and the relative budget eps^2 Vbar shrinks, sound? What is the strongest correct statement?
7. New results, if you see them: (a) a lower bound for smooth gates on genuine Gaussian fronts (an analogue of the
   lane's Theorem 4(c), or a proof that none exists at eps above the fuzzy-ray spread); (b) a clean statement of when a
   C-infinity gate beats sampling at fixed eps as a function of (n, beta s, depth); (c) anything on whether a gate can be
   smooth enough for summaries and still sharp enough to learn kinked targets (sorting) at matched parameters.
For every claim give a label: PROVEN, PROVEN UNDER (state it), HEURISTIC, CONJECTURE, WRONG (with the fix).
Keep the answer under about 2,500 words; lead with a verdict table (claim, label, one-line reason).
