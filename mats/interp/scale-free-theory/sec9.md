## 9. Claim register

Statuses after the GPT-6 and Gemini reviews (findings and our responses: section 10).

| ID | Claim | Status | Section |
|---|---|---|---|
| T1 | Minimal exact tree message at v = number of Nerode classes; ceil(log2) of it = one-way CC; internal epsilon-nets add at most epsilon per node | PROVEN | 2 |
| L1 | Unrestricted routing rules absorb the whole computation | PROVEN | 2 |
| T2 | Messages are likelihoods of the open latents; merges add log-likelihoods and sum out closing latents | PROVEN (standard) | 5.1 |
| T2m | Minimality for the joint posterior of the open latents | PROVEN UNDER richness; smaller for a single target | 5.1 |
| T2a | Tiny-PFN: only y* is open between sources | PROVEN | 5.1 |
| T2b | Inverse-Wishart errors: predictive message across a trader cut = O(T^2) additive numbers (Sylvester); full posterior of Sigma needs T \|S\| | PROVEN UNDER the matrix-normal / IW model, prior scale block-diagonal or diagonal plus low rank | 5.1 |
| T3 | Fixed-dimension sufficient statistics for all N only in exponential families | PROVEN UNDER regularity (Koopman-Pitman-Darmois) | 5.2 |
| T3a | Analytic families: O(log(MN/delta)^d) additive numbers per source for KL <= delta | PROVEN UNDER compact prior and a uniform complex-analytic bound | 5.2 |
| T4 | Two rounds (pilot, then local degree-p statistics) give KL about N^{1-p} up to logs | PROVEN UNDER smoothness, posterior concentration, a pilot inside the window w.h.p., unimodality (one-step estimators; cited) | 5.3 |
| T4a | One-round size grows exponentially in the number of open latents for location-type families | HEURISTIC; MEASURED (N2) | 5.3 |
| T5 | Keyed aggregation: Omega(N) bits at a node of every fixed hierarchy; O(1) + log N label bits routed | PROVEN | 5.4 |
| T5b | Bayes analogue (World FA) and "learning, not representation" | HEURISTIC | 5.4 |
| T6 | Attention routing cost = sign-rank of the routing relation; equality 3, order 2, Hadamard >= sqrt(N) | PROVEN | 5.4 |
| T6a | Sharp selection needs logit range ~ log N, Alman-Song's worst-case hard regime; label routing is O(N log N) | PROVEN (arithmetic + cited theorem) | 5.4 |
| T6b | Per query, softmax attention is an associative merge with d + 2 numbers | PROVEN | 5.4 |
| T7 | Correlation-based grouping needs Omega(min(T, 1/rho^2)) bits per pair; keys of O(min(T, log(N/delta)/rho^2)) bits suffice | PROVEN (promise problem, sign strings); HEURISTIC (iid histories) | 5.4 |
| T8 | 3+ hop lookups need Omega(n) bits at any cut separating the tables in one pass; k rounds suffice | PROVEN (for such cuts; cited constants unverified in primary text) | 5.5 |
| T9 | Parity: untangled under every definition, SQ-hard; gradient methods only in a coarse-precision regime | PROVEN (cited); MEASURED (World 3) | 5.5 |
| T10 | GF(2) pairwise functions: worst message of the best hierarchy = rank-width (+1 bit) | PROVEN | 5.6 |
| T10a | ReLU gates couple through Gaussian pre-activations, so the non-Gaussian part is the gate-pattern entropy | HEURISTIC | 5.6 |
| T11a | Exact output law in O(R K (R + K)) per node along a shared hierarchy (sufficient, not necessary) | PROVEN | 5.7 |
| T11b | Misaligned random trees: union treewidth grows with N | MEASURED (N3a); linear growth CONJECTURE | 5.7 |
| T11c | Rank-2 functions can have #P-complete tails (integer weights) | PROVEN | 5.7 |
| T11d | Bounded noisy messages: m >= (log2(1 + N I) - 2 delta log2 e) / log2(1 + R^2/sigma^2) | PROVEN UNDER the Gaussian setting stated | 5.7 |
| T12 | Latent-tree data: block MI O(log L); correlations at most power-law; KS threshold (strict inequalities) | PROVEN | 5.8 |
| A1 | Softmax attention with a sink = merge of Gaussian messages; sink weight = tau_0 x posterior variance | PROVEN (identity) | 6.1 |
| A2 | Transformers implement one-level R-round SFU computations in O(R) layers | PROVEN UNDER one-level routing and bounded group size; HEURISTIC otherwise | 6.2 |
| A3 | Flat-token binding stalls at a routing-content saddle | HEURISTIC | 6.2 |
| C1 | Every trained net's error is about 30-48% cross-source interaction; raw interaction tracks error size; one-layer and stuck nets lack within-source interaction | MEASURED (52 nets, N5-N6) | 6.4 |
| C1b | 2-layer nets: layer 1 routes by labels, layer 2 weights sources by log precision | MEASURED (15 clean nets, N7; fixed layout) | 6.4 |
| C2 | Each departure from Bayes is the error of the merge family at some scale | CONJECTURE | 7 |
