Model: Gemini 4 Argon

1. CRUX CRITIQUES AND CORRECTED CRITERIA

C1 (Policy-invariant aggregation extrapolates, W1) — REJECT AS WRITTEN.
Flaws:
(a) Confounds length OOD (64x context length, RoPE/softmax entropy shift) with multiplicity concentration.
(b) Confounds latent structure identification (finding R) with within-cell count drift. Two random items in {0,1}^16 share 8 coordinates on average, so under a uniform prior over the 120 pairs they fall in the same cell with probability C(8,2)/C(16,2) = 28/120 = 0.233. If attention dilution at N=4096 makes the model lose R and fall back toward a uniform mixture over R, the dominant group automatically leaks into the query with slope ~0.23 even with zero within-cell count drift.
(c) A threshold of 0.5 is far too lax (40% contamination is already catastrophic failure).
Corrected C1: Fix clean informative context at N_clean=48 balanced across cells, and append M in {0, 16, 64, 512, 4048} duplicates/relays of one dominant item x_dom at Hamming distance 8 from x_q with R(x_dom) != R(x_q). Run alongside a matched-length balanced control (N = 48 + M balanced) and an R-known oracle baseline. Refuted if contamination slope exceeds 0.10 (or grows > 0.03 per doubling of M) at any M where the matched-length balanced control maintains KL < 0.05.

C2 (Fixed depth limits chain length, W3) — REJECT AS WRITTEN.
Flaws:
(a) Wrong horizon math and small-K shortcut: pointer doubling composes paths in O(log l) layers (up to 2^L hops, not L+1). Worse, if total nodes K <= 32, Layer 1 attention can dump every edge in the context into the query token's residual stream (d_model >= 256) and Layer 2's MLP solves the 6-hop chain in one vector operation, passing l=6 at L=2 without sequential message passing.
(b) Comparing L=2 (2 passes) to looped 2x8 (16 passes) confounds weight tying with 8x compute.
Corrected C2: Put M >= 8 disjoint chains of length l in each episode (K >= 64 randomly permuted node IDs) with randomly oriented edges ((u,v,+d) vs (v,u,-d)) so the graph cannot fit in one token's head state. Train untied L in {2,4,6} and looped 2x3 on l in {1..4}; evaluate on l in {1..8} running the looped net for k in {3,6,9} iterations. Refuted if L=2 achieves KL < 0.05 at l=6 with M=8 distractors; looped motive holds only if test-time iterations k=6,9 beat untied L=6 on OOD lengths l in {5..8}.

C3 (Bounded consensus) — REJECT AS WRITTEN.
Flaws:
(a) Testing k=32 slots on K=512 independent items is a trivial pigeonhole test, yet if d_model=256, k=32 slots hold 8,192 floats—enough to pack 512 scalar means and precisions via linear superposition. Capacity must be measured in total state floats D_state = k * d_slot relative to sufficient-statistic dimension.
(b) Misses the real consensus bottleneck in W2: retroactive source reweighting.
Corrected C3: Test two regimes at fixed N=2048:
(i) Retroactive calibration (fixed K=16, S=16, two 1024-token chunks): Chunk 1 contains SPOT/DIFF quotes from all 16 sources; Chunk 2 reveals which 4 sources are low-noise vs high-noise/biased. A chunk-1 state that compresses to O(K) item summaries instead of the O(K*S) source-item incidence ledger must fail (> 0.10 nats gap to full attention).
(ii) Fixed float budget D_state vs growing active item count K.

C4 (In-context vs prior-level, W2) — REJECT AS WRITTEN.
Flaws:
(a) 1-hop token-ID counting beats the test: a model that simply counts how often `source_id` appears in the context (one induction head) will discount 8 repeats from a dominant source vs 8 fresh sources without inferring source bias b_s or noise sigma_s from the overlap graph.
(b) If tau^2 is large, 8 uncalibrated fresh sources actually have higher bias uncertainty than 1 source whose bias was already calibrated on 50 other items in the context.
Corrected C4: Hold source frequency equal. Target item k* gets 1 quote from s_A and 1 quote from s_B, where both s_A and s_B appear 16 times on other items in the episode. On those 15 other items, s_A's residuals against the overlap graph are tight while s_B's are noisy/biased. Refuted unless the model's relative weight on s_A vs s_B beats a "source-frequency-only" reference toward full graph-calibrated Bayes on >= 80% of episodes (closing >= 50% of the MSE/KL gap).

C5 (Optimisation, not expressivity, W2/W4) — REJECT AS WRITTEN.
Flaw: Single-query NLL gives O(1) scalar gradient per length-N sequence, whereas auxiliary consistency losses provide O(N) labels per sequence. Winning with auxiliary loss confounds supervision density with architecture/optimization plateaus.
Corrected C5: Compare single-query NLL against all-token next-observation predictive NLL (zero latent oracle labels) plus an N/l curriculum. An optimization barrier beyond label sparsity is shown only if all-token predictive NLL + curriculum stalls for 10x the gradient-SNR horizon while self-consistency regularization or weight warm-start reaches KL < 0.03.

C6 (Inconsistency minimisation sufficiency) — REJECT AS WRITTEN.
Flaw: W2 already has a trivial shift gauge (v_k -> v_k + c, b_s -> b_s - c leaves all residuals zero), which any energy minimizer fixes by adding the prior norm ||v||^2 + ||b||^2/tau^2. The real gap between minimum-inconsistency (MAP / free-energy point estimation) and Bayes is volume marginalization over discrete hypotheses (R in W1, sigma_s in W2).
Corrected C6: Construct episodes in W1/W2 where 1 sharp hypothesis H_1 has slightly lower total residual+prior energy and predicts y_q = -1, while 50 slightly higher-energy hypotheses H_2..H_51 agree on y_q = +1. Point-estimate inconsistency minimization picks H_1; exact Bayes marginalizes the volume of H_2..H_51.

2. FATAL FLAWS IN W1-W4 AND FIXES

W1 Fatal Flaw: Without event IDs, distinguishing fresh measurements of item x from noisy relays of x requires marginalizing over all clusterings/partitions of observations of x (Bell number B_n), breaking the O(120 N) linear-Gaussian oracle. Conversely, if x is measured only once and all repeats of x are relays, relay measurement error is mathematically identical to the item random effect b_x ~ N(0, tau^2).
Fix: Define W1 as a clean 3-level hierarchical Gaussian: Cell g(x_R) ~ N(0,1) -> Item v_x ~ N(g(x_R), tau^2) -> Observation y_i ~ N(v_{x_i}, sigma^2). When tau > 0, repeating item x saturates at variance tau^2 (item dedup), while sampling distinct items in the same cell averages down tau^2 (cell aggregation). Exact Bayes is strictly O(120 N).

W2 Fatal Flaw: "K, S grow with N" with "2^S noise-level hypotheses" makes the exact Bayes oracle O(2^S (K+S)^3)—completely intractable for S > 16 (at N=2048, S=512 gives 2^512).
Fix: Keep continuous bias b_s ~ N(0, tau^2) on all S sources (solved in one sparse Cholesky solve), but restrict unknown binary noise sigma_s in {sigma_L, sigma_H} to at most S_uncertain <= 8 active sources per episode (or factor into disconnected market clusters of size <= 8).

W3 Fatal Flaw: Single-chain episodes or canonically oriented edges allow bag-of-edges summing shortcuts without path tracing.
Fix: Pack M >= 8 disjoint chains per episode with random 50/50 edge sign flips ((u,v,+d) vs (v,u,-d)).

W4 Fatal Flaw: On a transposition DAG where children l and r share a common descendant, Z_l and Z_r are correlated, so the recursive formula Z_v = 1/2 L_v + 1/2 Z_l Z_r is mathematically wrong (it double-counts shared evidence).
Fix: Use exact cutset conditioning over shared DAG nodes as the Bayes oracle, turning W4 into a direct test of whether the model avoids loopy double-counting.

3. WHAT IS MISSING

Missing Crux C7 (Topological Dedup / Diamond Double-Counting in W2/W3):
W1 tests item-level dedup; W2 tests source bias. Neither tests correlated-path dedup in a network. Compare a Diamond Graph (1 noisy SPOT quote on v_0 split into two independent DIFF chains v_1 - v_0 and v_2 - v_0 that reconverge at query v_q) against a Two-Anchor Graph (2 independent SPOT quotes on v_{0a}, v_{0b} feeding v_q with identical path lengths and edge noises). Naive message passing double-counts v_0's precision on the Diamond; exact Bayes discounts the shared root.

4. RANKING BY EXPECTED INFORMATION PER DOLLAR (NEXT 12H, $100 L4) & FIRST 3 EXPERIMENTS

Ranking:
1. W1 (3-level hierarchical junta: Cell -> Item -> Obs) — Fastest oracle O(120 N), cleanest test of C1 and item vs cell dedup.
2. W3 (Multi-chain randomly-oriented linear-Gaussian constraints) — Instant sparse-linear oracle, isolates C2 (depth vs looped compute) without optimization confounds.
3. W2 (Market world with S_uncertain <= 8) — Highest relevance to market aggregation (C3, C4, C7), slightly heavier oracle.
4. W4 (Transposition DAG) — Defer past 12h; C7 Diamond test in W2/W3 captures its best crux cheaply.

First Three Experiments:
1. Exp 1 (W1 — C1 + tau>0 resolution): Train L=4 transformer on N<=64 (tau=1.0, sigma=0.5). Test at N in {64..4096} across (a) balanced length control, (b) single-item repeat (tests tau^2 floor), and (c) dominant-cell repeat M in {16..4048} (Corrected C1).
2. Exp 2 (W3 — Corrected C2): Train untied L in {2,4,6} and looped 2x3 on M=8 disjoint chains of length l in {1..4}; test zero-shot on l in {1..8} with k in {3,6,9} loop iterations.
3. Exp 3 (W2 — Corrected C4 + C7 Diamond): Train L=4 on K=16, S=8 market graphs; evaluate frequency-matched graph calibration (Corrected C4) and Diamond vs Two-Anchor posterior variance (C7).
