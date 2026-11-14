# Literature and design choice (leaf pool-uwmcvo.2, 31 October 2026, 11:00Z)

Status marks: [V] checked this session (search result or abstract page), [P] read by an earlier lane of this project
(outbox reports cite them as verified), [M] from memory, to be treated as unverified in the report.

## Sources
- Typically-correct derandomization, input as the randomness source:
  - Goldreich, Wigderson, "Derandomization that is rarely wrong from short advice that is typically good",
    RANDOM 2002, LNCS 2483, 209-223 [V: bibliographic record; idea attributed to them by later papers].
  - Hoza, "Typically-correct derandomization for small time and space", CCC 2019 [V: "treating the input as a source
    of randomness ... pioneered by Goldreich and Wigderson"].
  - Kinne, van Melkebeek, Shaltiel, "Pseudorandom generators, typically-correct derandomization, and circuit lower
    bounds", Comput. Complexity 2012 [V: record].
- Region counts: Hanin, Rolnick, "Deep ReLU networks have surprisingly few activation patterns", NeurIPS 2019,
  arXiv 1906.00904 [V: abstract: expected number of activation patterns at initialisation bounded by
  (#neurons)^(input dim), depth-independent]. Hypotheses (bias densities) [M]; our nets are bias-free, so the
  theorem is not applied, only compared with measured counts.
- Spin glasses: Gamarnik, Kizildag, "Computing the partition function of the Sherrington-Kirkpatrick model is hard on
  average", Ann. Appl. Probab. 31(3) 2021, arXiv 1810.05907 [V: abstract: exact computation hard on average unless
  P = #P, via random and downward self-reducibility; finite-precision model]. Huang, Sellke, "Tight Lipschitz
  hardness for optimizing mean field spin glasses" (FOCS 2022) [M]; El Alaoui, Montanari, Sellke, sampling SK via
  algorithmic stochastic localisation, with stable-algorithm hardness at low temperature (FOCS 2022) [M].
- Spherical designs / discrepancy: Beck 1984 (Mathematika), L2 cap discrepancy order N^{-1/2-1/(2d)} on S^d, squared
  N^{-1-1/d}; Stolarsky invariance; Brauchart-Dick (Proc. AMS 2013): worst-case integration error in a Sobolev RKHS
  equals the L2 cap discrepancy [V: search snippets]. QMC designs for H^s(S^d), order N^{-s/d}: Brauchart, Saff,
  Sloan, Womersley 2014 [M].
- Average-case IBC: Ritter, "Average-Case Analysis of Numerical Problems", LNM 1733, Springer 2000 (chapter
  "Nonlinear methods for linear problems": adaptive and nonlinear methods for linear problems under Gaussian
  measures) [V: table of contents and keywords]. Wasilkowski, Wozniakowski, "Can adaption help on the average?",
  Numer. Math. 44 (1984); Traub, Wasilkowski, Wozniakowski, "Information-Based Complexity" (1988) [M].
- Deterministic vs randomized volume: Elekes 1986; Barany, Furedi 1987 (deterministic oracle algorithms cannot
  approximate volume within (c d / log d)^d); Dyer, Frieze, Kannan 1991 (randomized FPRAS) [M].
- Random circuits: Bouland, Fefferman, Nirkhe, Vazirani 2019 (average-case #P-hardness of exact output
  probabilities) [M]. Aaronson, Arkhipov 2011 (Gaussian permanents) [M].
- Fine-grained average case: Ball, Rosen, Sabin, Vasudevan, STOC 2017; Goldreich, Rothblum FOCS 2018;
  Boix-Adsera, Brennan, Bresler FOCS 2019 (all via random self-reducibility of low-degree polynomials) [M].
- IKW 2002, Chen-Tell 2021, Kabanets-Impagliazzo 2004 [P, hardness lane read IKW and Chen-Tell in full].
- Verification bounds: symbolic interval propagation (ReluVal / Neurify, Wang et al. 2018), DeepPoly (Singh et al.
  2019), CROWN (Zhang et al. 2018), input-splitting branch and bound [M].
- In-house: hardness-mean-estimation-2026-10-29 (Prop 3.5 nonuniform determinism, rotation equivalence of designs),
  cut-rank-lower-bound-2026-10-30 (eps-rank; Bietti-Bach spectrum l^{-(n+2)}), narrow-deep-m1-2026-10-30 (width-8
  adaptive exact pieces: GHM floors at 1e-3; region histories explode by layer 8; BO quadrature at the input needs
  6-7 of 8 directions), tractable-gates-2026-10-31 (smooth gates, closures).

## Options for the progress prong (an adaptive whole-network deterministic estimator)
1. Certified branch-and-bound over input cones. Cells: cones over boxes in gnomonic coordinates on the 2n faces of
   the cube (x_k = s t, x_i = t y_i, y in a box). The Gaussian mass and first moment of a cell are 1-D integrals in t
   of products of normal cdf differences, so they are exact up to a 1-D quadrature. Bounds: forward symbolic linear
   bounds (ReluVal/Neurify style) with the actual weights, exact when every unit is stable on the cell; by positive
   homogeneity they extend from the slice to the cone. Cells where every unit is stable are integrated exactly;
   elsewhere the estimate is the local linear map at the cell centre applied to the cell's first moment, and the
   certificate is the integral of the bounds. Refinement splits the cell with the largest certified gap along the
   coordinate that explains most of the unstable units' ranges.
2. Exact region enumeration (arrangement traversal). The exact limit of option 1; only for n = 2-3.
3. Deterministic low-discrepancy evaluation: unscrambled Sobol on the sphere with exact radius, compared with
   scrambled Sobol. Not sample-free (it is an average of f over a point set).
4. Sensitivity-guided adaptive cubature without certificates (embedded-rule error indicators): cheaper per cell
   but no certificate.
5. Gate-pattern mixtures in hidden space (narrow-deep-m1's GHM): already measured, floors at 1e-3 to 1e-2 at width 8.
6. Reachable-set (zonotope / star-set) refinement in hidden space: tighter but costs O(n^3 L^2) or more per cell.

Chosen: option 1 (it is genuinely adaptive, uses the whole net and the actual downstream weights, gives
certificates, and contains option 2 as its exact limit), with option 3 as the deterministic-sampler reference and
plain MC / scrambled-Sobol RQMC (both with exact radius) as baselines. Option 4 is dropped: without certificates it
is a deterministic quadrature like option 3 with extra cost. Options 5-6 are covered or too costly at depth.
