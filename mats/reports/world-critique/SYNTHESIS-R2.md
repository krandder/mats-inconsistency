# Synthesis after round 2 (opus-mats, 2026-10-28 ~02:00Z) — input to round 3

Sources: r1-*.md, r2-*.md (GPT-6 via Codex; Gemini 3.8 Flash in r1, Gemini 4 Argon in r2; Claude Fable 5.1).
Kelvin's goal: priors that make small transformers show interesting in-context (not prior-level) dedup and
market-like aggregation; mechanistic study; architectures that scale to more observations / more reasoning; sharp
cruxes that he and gemini would accept.

## Agreed by all three
- Ignorability must be explicit: policies may depend only on visible keys, recorded observations and independent
  randomness (r2-codex caught that a "cell-concentrated" policy using hidden R invalidates the oracle).
- Trivial versions are rejected: exact event-id copies alone (hash match, already measured), 8-bit junta (item
  space saturates), GP-ARD (done by PFNs4BO, costly oracle), multi-index g(Ux) (no exact Bayes), max-backup trees.
- Contribution, if any, is narrow: gap to exact Bayes under policy shift and N >> N_train, decomposed against
  references that separate prior-level from in-context, plus a causally identified mechanism. "A transformer
  approximates Bayes on bandit data" is not new (PFNs4BO, DPT, Bai et al. 2023).

## Portfolio (build order)
W1 Similarity world (junta family). Items x in {0,1}^16; hidden R, |R|=2 (120 hypotheses); g(cell) ~ N(0,1);
   optional item effect b_x ~ N(0,tau^2); fresh measurements; copies as exact (event id) or relays (shared measurement
   + small relay noise, no id). Exact Bayes O(120 N). References: prior-level, copy-naive, item-only, R-oracle, full.
   Disputed: whether tau>0 adds in-context content (Fable: prior-level table; Codex/Gemini: separates measurements
   from items). Pilot both. [Being built now: work/worlds/.]
W2 Market world (merge of Fable P2 + Gemini rung 2). K items v_k ~ N(0,1); S sources with bias b_s ~ N(0,tau^2) and
   noise level sigma_s in a discrete set (2^S hypotheses); quote types SPOT (v_k), DIFF (v_k - v_j), BASKET (mean of a
   subset); exogenous policy over (source, quote) with arbitrary concentration; K, S grow with N. Exact Bayes: mixture
   over noise-level hypotheses of linear-Gaussian posteriors over (v, b). Source bias is identifiable only through
   the overlap graph, so prior-level shrinkage fails. Market-like: heterogeneous, biased, overlapping contracts
   combined into one price. Predicted mechanism: alternating item<->source residual passing (~1 Gauss-Seidel round per
   layer pair).
W3 Constraint/chain world (Fable P3, Gemini rung 3, Codex world 3). Linear-Gaussian version: SPOT measurements,
   DIFF relations, hard equalities; the path length l from query to nearest measurement is a dial. Discrete version
   (later): latent bits with unary, XOR and majority messages ("rich messages" that individual scalars lack).
W4 Tree / transposition world (Fable CTW with Codex's recursion correction Z_v = 1/2 L_v + 1/2 Z_l Z_r; Gemini's
   transposition DAG). Build only if W1-W3 show template-menu shortcuts that W4 forbids.

## Cruxes (merged; thresholds to be attacked in round 3)
C1 Policy-invariant aggregation extrapolates (W1). Train N<=64 on a policy mixture; test N=4096 on held-out
   concentrated policies; contamination slope = regression of (model mean - Bayes mean) on (dominant-group mean -
   query-group Bayes mean); Bayes slope 0. Refuted if slope > 0.5 on >= 8/10 seeds at N=4096 while < 0.1 at N=64.
C2 Fixed depth limits chain length (W3). l in 1..6, depth L in {2,3,4,6}, plus weight-tied looped net (2 layers x 8).
   Refutes "depth is the bottleneck" if depth-2 is within 0.05 nats of Bayes at l=6 on >= 8/10 seeds; supports the
   looped-architecture motive if fixed depth collapses for l >= L+1 while the looped net holds KL < 0.05 for all l <= 6.
C3 Bounded consensus. Codex: universal lossless bounded consensus is impossible analytically (B-bit state, K revealed
   bits, query later: error >= h2^-1(1 - B/K)). So the empirical claim must be narrower: a query-agnostic k-slot state
   matches full attention when sufficient statistics are bounded (fixed K in W2) and fails when they grow (K ~ N/4).
   Refuted (in the strong form) if the k=32 bottleneck stays within 0.03 nats of full attention at N=2048 with K=N/4.
C4 In-context vs prior-level (W2). Append 8 readings of the query item from the dominant source vs 8 from 8 new
   sources; compare delta_rep/delta_fresh with Bayes; refuted if predictions are closer to a training-rate shrinkage
   reference than to in-context Bayes on > 80% of episodes at every depth.
C5 Optimisation, not expressivity (W2/W4). Query NLL alone stalls at the prior-level plateau in >= 8/10 seeds at 10x
   oracle steps, while warm-start from known structure or an auxiliary consistency loss reaches KL < 0.03.
C6 Inconsistency minimisation sufficiency (conceptual). Exhibit two equally consistent optima differing by >= 0.2 in a
   query probability when Bayes is unique; then likelihood/prior terms are essential to the program's claim.

## Architecture motives (only if the matching crux fails cleanly)
- C1 fails by count drift -> count/precision-normalised attention (log-multiplicity bias or kernel-density normaliser).
- C2 shows a depth cliff -> weight-tied iterative layer run until residual inconsistency stops falling (Kelvin's
  minimised-inconsistency search as architecture).
- C3 bounded fails only when sufficient statistics grow -> consensus state = ledger of sufficient statistics plus
  query-conditioned retrieval.
- If all hold at 64x N, no new architecture is justified; move to the reasoning axis.
