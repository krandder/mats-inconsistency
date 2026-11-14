# Scale-free theory: candidate definitions, assumptions and checks (2026-10-29, leaf pool-yw7kwn.9)

Each item: the claim or assumption, and how it gets checked (proof, citation, numerics, or reviewer).

## Definitions to compare
D1. Fixed hierarchy T, exact cut complexity: K_f(S_v) = number of Nerode classes of x_{S_v} (residual functions).
    Check: prove the canonical tree computation is minimal and exists (Myhill-Nerode argument); log K = one-way CC.
D1e. Tolerance version: epsilon-covers of residual functions; error compounds additively over nodes.
    Check: write the substitution argument; numerics not needed.
D2. Input-dependent hierarchy: per-item labels of b bits + a restricted rule (group-by-label, sort, contiguous chunks);
    content = Nerode classes given public labels. Assumption to state: the rule must be restricted, or routing absorbs
    the whole computation. Check: give the loophole example and the restricted definition.
D3. Bond dimension: rank of the matricisation across each cut (real rank; nonnegative rank for laws = latent states).
    Check: separations rank << K (generic linear sum), and expectation vs tail (knapsack #P-hard at rank 2).
D4. Treewidth of (a) the circuit DAG, (b) the activation law's graphical model, (c) its non-Gaussian part.
    Checks: Amarilli et al. (treewidth circuit -> structured d-DNNF); Chandrasekaran et al. 2008; Lauritzen 1992 /
    Lauritzen-Jensen 2001 (conditional Gaussian, strong junction tree); Lerner-Parr 2001 (CLG polytrees NP-hard).
D5. MI across scales / RG: block MI laws of data from latent trees (log law), power-law correlations, KS threshold;
    RG = merge map on message laws, fixed points (CLT, stable, extreme value). Check: separator bound proof; cite KS.
D6. Hierarchical low rank (H-matrix / FMM) for interaction kernels; attention = sparse routing + smooth low-rank part.
    Checks: Alman-Song 2023 bounded-entries dichotomy (exact statement), FMMformer, H-Transformer-1D on arXiv.

## Sharp claims to establish (status to be marked)
C1. Exact output law along T under a latent-tree input law costs sum_v R_v R_a R_b K_a K_b. (proof)
C2. Misaligned hierarchies: union treewidth of two random trees grows with N. (numerics, tanglegram; literature?)
C3. Conditionally independent sources: additive messages = log-likelihood functions; fixed dim for all N iff
    exponential family (Pitman-Koopman-Darmois, regularity); analytic families: dim O(polylog(N/delta)).
    Check: PKD citation; numerics on the tiny-PFN world (eps-rank of per-source log-likelihood of y*, N = 3..256).
C4. Rounds: one-round messages must carry the global likelihood function (eps-rank grows fast with latent dim G);
    two rounds (broadcast a pilot = "messages written after reading the consensus") need O(G^2) local statistics,
    asymptotically exact (LAN/BvM, Le Cam one-step). Check: numerics on a G-group location mixture; cite BvM/one-step.
C5. Routing separation: key-routed aggregation is Omega(N) bits at a balanced node of every fixed hierarchy (reduction
    from disjointness), O(1) content bits + log N label bits adaptively. (proof; Razborov / Kalyanasundaram-Schnitger)
C6. Routing cost in attention = sign-rank (exact) / margin complexity (robust) of the routing relation: equality and
    order O(1) exact, O(log N) robust; Hadamard-type >= sqrt(N) (Forster 2002). Check: citations; equality construction.
C7. Correlation-based routing: per-item sketch bits Theta(min(T, 1/rho^2)) via gap-Hamming (Chakrabarti-Regev 2011)
    + padding reduction. Check: citation and the padding argument.
C8. Pointer chasing: one-pass hierarchies Omega(n); k rounds O(k log n) (Nisan-Wigderson 1993); transformers
    k-hop in O(log k) layers (Sanford-Hsu-Telgarsky 2024). Check citations.
C9. Parity: 1-bit cut complexity under every hierarchy, yet SQ-hard to learn (and World 3 chains failed on parity).
    Separation of untangled vs learnable. Check citations (Blum et al. 1994 / Kearns 1998).
C10. Pairwise GF(2) functions: best-hierarchy message = rank-width (+1); complete graph rw = 1 vs treewidth n-1;
     random graphs linear rank-width (Lee-Lee-Oum 2012). Check: brute-force Nerode counts for small n; citation.
C11. Softmax attention with a sink = exact natural-parameter merge of Gaussian messages (logit = log precision,
     sink weight = tau_0 x posterior variance). Without the sink the merge is not closed. (algebra; numeric check)
C12. Noise: normalized (bounded) mean pooling with noise sigma cannot resolve the posterior beyond
     N* ~ (R/sigma)^2; needed dimension >= log N / log(1 + SNR) (rate-distortion + capacity). (proof)
C13. Algebraic escapes: Gaussian, GF(2)-quadratic (Dickson), planar zero-field Ising (FKT) are tractable at high
     treewidth: tangled != intractable. Check citations.

## Transformer measurements to specify (for the experimental agent)
T1. Interaction profile: double differences of log p(y|x) across source-aligned vs source-splitting cuts (Bayes = 0
    across source-aligned cuts in the tiny-PFN world). Check: verify the Bayes identity numerically.
T2. Routing labels: fit attention logits from label-equality features (source id, question id, type); routing bits.
T3. Frozen routing vs static routing vs label routing substitutions (loss deltas).
T4. Bottleneck probe across cuts: minimal k for an encoder-decoder to reproduce the net's output (vs Bayes's k).
T5. Sink / posterior-variance check (C11).
T6. 2 x 2 design for hierarchy-vs-dense: routing oracle vs learned x dense vs tree content.

## Open questions (not expected to be settled today)
Q1. Is there a converse to C3 for non-regular families (monoid statistics like max)? (literature check only)
Q2. Does training align the net's hierarchy with the data's (C2 as a learning claim)? (experiment, later)
Q3. Lower bound for one-round eps-rank in G (C4) beyond numerics. (try Gaussian-kernel eigenvalue counts)
