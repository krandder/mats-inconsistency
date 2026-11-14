### 6.5 Specification for the hierarchy-vs-dense experiment (for the experimental agent)

The comparison is only informative if it separates routing from combination, and rounds from hierarchy shape.

**Design: 2 x 2, plus rounds.**
- Routing: *oracle* (tokens arrive grouped by source, or a mask from the true grouping) vs *learned* (flat tokens with
  per-episode ID embeddings, as in the phase-B and World FA formats, with token order permuted per episode).
- Combination: *dense* (the phase-B transformer, matched parameters and FLOPs) vs *SFU tree*: per-source encoder (a
  set network over that source's tokens) to k numbers; merge = sum; readout = small MLP to the mixture-density head.
  For W2 use a two-level tree (tokens -> sources -> shared-error groups -> root), or the source and group boundaries
  collapse.
- Rounds, tree only: one pass, or two passes (pass 1 sends a few numbers per source; a learned readout broadcasts a
  pilot; pass 2's encoder also sees the pilot and sends k numbers). Match the total message size and parameter count
  between one-pass and two-pass trees, or rounds are confounded with capacity.
- Learned routing for the tree: cluster tokens by pairwise dot products of their ID embeddings (the same
  rotation-invariant information dense attention uses). Not hashing (tokens of a source share an identical ID vector,
  so hashing is oracle routing in disguise) and not learned prototypes (they break the rotation symmetry of random
  per-episode IDs).

**Worlds.**
- W1: tiny-PFN prior with N in {3, 8, 24, 64, 256} sources (same generator; exact Bayes from the per-source
  log-likelihood sum on a grid, `work/scale-free-theory/n1c_rounds.py: ell_grid`).
- W2: W1 plus G shared-error groups (one shared error per group and question), membership visible as a label or
  hidden; G in {1, 2, 4, 8}. With hidden membership, break the label symmetry (for example order groups by their
  shared-error variance) or the posterior has G! modes and local two-round messages fail.
- W3: World FA, binding-given vs flat.

**Read-outs.** KL to Bayes against k; smallest k reaching KL <= 0.01 and 0.001 (95th percentile over episodes) per N;
M1 (with deviation shares), M2 and an internal M4 on the dense nets; FLOPs.

**Predictions (to pre-register).**
- P1 (W1, oracle routing): a two-pass tree at small k (about 4-8) reaches the KL that a one-pass tree reaches only with
  larger k, once N >= 64; at N = 3 the two are similar. Calibration (N1d, N1e): a 2-number one-round code stays at KL
  about 0.2 at every N, while 4 numbers over two rounds reach 2e-4 at N = 64 and 5e-7 at N = 1024; an 8-number
  one-round Chebyshev code is far worse than an 8-number two-round code (N1e). Learned nonlinear encoders may beat our
  linear codes, so the numbers are calibration, not bounds. Refuted if the two-pass tree is no better than the
  one-pass tree at matched size for N >= 256.
- P2 (W2, visible membership): a group-aligned tree needs the same k as W1; a tree whose groups split the shared-error
  groups needs k growing with the number of straddling groups.
- P3 (W2, hidden membership): one-pass k grows quickly with the number of open latents (N2: empirical linear one-round
  codes need about 19, 330 and 1,650 numbers for 1, 2 and 3 open latents at RMS error 1e-3); two passes with
  re-routing after the pilot stay far smaller.
- P4 (routing): with learned routing on flat tokens, both architectures fail to learn binding without help (World FA:
  0 of 15 flat runs), and both succeed with oracle routing. Refuted for "density pays for routing" if dense with
  learned routing matches dense with oracle routing, or if the tree with oracle routing falls far short of dense with
  oracle routing at matched compute.
- P5 (M3 payoff): for a one-pass SFU tree, given y* and the latents, the summed message is a sum of independent
  per-source terms; for the additive log-odds message (before any nonlinear readout), Chernoff bounds certify tails and
  a grid convolution is exact when the message has 1-3 dimensions. For a two-pass tree, condition on a discretised
  pilot first (the pilot couples the sources). Compare against sampling at equal compute (the M3 bar).
