# Overnight experiments, 2026-10-15: any-order network + stiffness + search on a permuted CA grid (and Sudoku)

Coordinator: opus-mats. Authorised by Kelvin 2026-10-15 00:3xZ (verbatim): "Please run the experiments it recomments.
You can have up to $200 in budget. You can try to use the MATS cluster as well. Try to run it overnight (tonight).
I like the cellular automata experiments. You can give the designs for it to review before launching."
"It" = the Gemini 4 Argon agent (fleet name `gemini`): its synthesis `argon-pdg-search-synthesis-2026-10-14.md`
(blueprint, section 4) and its chat with Kelvin on 10-14/15 (Candidates 1-3; Mechanisms A/B for stiffness; key-value
insight fingerprints). Reviewer before launch: `gemini`. Executors: eurisko (E-CA), calibrated-ai (E-SUD).
Budget: $200 total of Kelvin's money on GCP (eurisko's gpu-up.sh budget groups), plus free MATS-cluster L40s where the
queue allows. Results due by 08:00Z, on mats.ekelvin.com/experiments in the usual format, with an audit.

## The thesis being tested (Kelvin, 10-14, via the gemini chat)
One any-order network over homogeneous variables, with no direction labelled "forward" or "inverse". Can a stiffness
head beta(i | S) learn which of the network's own conditionals are stiff one-step laws and which are soft multi-step
leaps, and can a search policy (LIR Refocus) route through the stiff arcs so that beliefs about long-range queries
improve with compute, where the one-pass conditional fails? Secondary: do key-value fingerprints deduplicate
redundant arcs independently of how often the policy queries them?

Rules (Kelvin, standing): training uses outcomes only (never the exact posterior; it is for scoring); nothing
hand-coded about the domain (the model is not told the CA rule, the topology, or the noise level); every arm must
be something that could run on real-world questions. Pre-register metrics and decision rules before launch.

## E-CA: permuted 1-D cellular-automaton spacetime grid (eurisko; cap $110)

Data. Elementary CA with noise: x[t+1, c] = rule(x[t, c-1], x[t, c], x[t, c+1]) XOR Bernoulli(eps), periodic
boundary, random initial row. Grid T x W binary variables with a FIXED random permutation of the T*W indices
(the model sees permuted position ids only). Rules: 110 (gliders, complex) and 90 (XOR light-cones, scale-2^k
shortcuts) as main; 30 as a hard control if time. eps = 0.05. Sizes: W = 8, T = 6 (48 variables); exact posteriors
for any evidence mask come from forward-backward over the row chain (2^W = 256 row states), so ground truth is
cheap. Also W = 6, T = 6 for a brute-force cross-check of the exact-inference code (2^36 is too many to enumerate;
cross-check on W = 4, T = 4 instead).
Episodes: a grid plus an evidence mask. Evidence = revealed true cell values; mask rate uniform in [0.2, 0.8] during
training. Test query families: (F) forward: evidence concentrated in early rows, query late rows; (B) bridge: evidence
in row 0 and row T-1, query middle rows (no single causal direction works); (R) random masks. N_train = 50k grids per
rule; validation 3k; test 3k (held-out grids).

Model. One any-order transformer (reuse smallnet's netdef: pre-LN, d = 128, 4 layers, 4 heads, learned position
embeddings over the permuted ids, 48 tokens; input per token = revealed value or MASK; output = Bernoulli for every
masked cell). Heads added: (1) stiffness beta_int(i | S) (Mechanism A: a head trained to predict the arc's held-out
surprise -log q(x_i | x_S); beta = 1 / predicted surprise); (2) key k(i | S) in R^16, unit norm, from the last-layer
hidden state of cell i (used by the dedup arm). Training arms: T1 plain masked (any-order) loss; T2 reveal-chain TD
(rbtd8 as in smallnet, which is the LIR consolidation of search into one-pass arcs). 3 seeds each. Early stopping on
validation outcome log loss, never on the posterior.

Inference arms (frozen network; the search spends compute on a query cell Y given evidence E):
P0 direct: q(Y | E), one pass.
P1 random-order rollouts: K rollouts sampling masked cells in random order from the chain rule, SNIS mixture on Y.
P2 Refocus-VoI chain: repeat M times: pick the masked cell z maximising the estimated value of computation
    E_z[(mu(Y | E, z) - mu(Y | E))^2] (two extra forward passes per candidate; candidates = a random subset of 12
    masked cells), branch on z exactly (both values, weights q(z | E, ...)), contract the branch into a claim on Y.
    Claims pooled with the prior-corrected gamma = 0 pool: logit mu(Y) = logit p0 + sum_a w_a * (logit q_a(Y) - logit p0),
    w_a = beta_int,a / beta_0 learned on validation outcomes with the closed-form gradient (STUDY-LIR 3c/3d:
    dL/dw_a = -(l_a(y) - E_mu[l_a])), with the prior edge carrying the residual weight.
P3 same as P2 with confidence-first selection (pick the most peaked masked cell). P4 same with random selection.
P5 P2 plus key-value dedup (Gram crowding: w_a_eff = w_a / sum_b max(0, <k_a, k_b>)) versus scalar w_a.
P6 (aggregator comparison on the same claims, the blueprint's step 1): SNIS mixture vs plain gamma = 0 pool vs
    prior-corrected learned-w pool vs gamma = 1 product, on held-out log loss and KL.
Budget sweep M in {0, 1, 2, 4, 8, 16} materialised arcs (P1 uses K = M rollouts).

Metrics (pre-registered). Primary: KL(exact posterior of Y || estimate), averaged over query cells, per query family
(F, B, R), as a function of M. Secondary: held-out outcome log loss of the same estimates; calibration (ECE).
Stiffness recovery: Spearman correlation between beta_int(i | S) and the exact -H(X_i | X_S) over 2k sampled (i, S)
pairs, and AUC of beta for "S contains i's three parents" versus not. Dedup: with the policy forced to query m in
{1, 2, 5} near-duplicate conditioning sets of one arc, the pooled confidence must not grow with m under P5 (report the
growth under scalar w for contrast).
Decision rules. (D1) Search helps: P2 at M = 8 beats P0 on bridge queries by >= 30% KL with paired 95% CI over
3 seeds; if P0 is already within 1e-3 nats of exact on bridge queries, record "no depth bottleneck at this size" and
stop (that is the gap check the papers study asked for first). (D2) Stiffness is real: Spearman >= 0.5. (D3) VoI beats
confidence-first and random at equal M on bridge queries. (D4) Dedup: pooled logit growth with m under P5 is < 20% of
the growth under scalar w. Report every rule as pass/fail with numbers; a failed rule is a result, not a reason to tune.

Compute. GCP spot/on-demand L4 via logical-soup/infra/gpu-up.sh, budget group `overnight-ca`, cap $110 at the
$0.87/h ceiling. Plan: smoke on one VM (price it), then 2 rules x 2 training arms x 3 seeds = 12 training runs
(~30-45 min each at 50k grids), inference sweeps on the same VMs. Also submit the same 12 runs to the MATS cluster
(`compute` partition, 1 GPU each, time limit 2 h) as k.santos; whichever finishes first is used, the other cancelled.
Stretch, only if the CA pipeline is done by 04:00Z: Candidate 2, modular-arithmetic constraint nets (N = 15 variables
in Z_5, sparse X_i + X_j = X_k mod 5), same arms; tests synergistic stiffness.

## E-SUD: 4x4 and 6x6 any-order Sudoku with ambiguous and unsolvable boards (calibrated-ai; cap $50)
Extend the existing ARDM any-order Sudoku code (fu-b2): boards with multiple solutions (so cell marginals are real
probabilities, exact by enumeration at 4x4 and by a solver at 6x6) and boards with no solution. Query: cell
marginals, and Solvable in {0, 1}. Arms: P0 direct, P1 random-order rollouts, P2 VoI chain with branch-and-contract,
P3 confidence-first, and computation-as-evidence for Solvable: logit P(solvable | null search of M branches) =
logit p0 + log(1 - eta(C_M)), C_M = searched proposal mass, eta a one-parameter recall curve fitted on validation
outcomes (Argon synthesis, insight 5). Metrics: KL to exact marginals vs M; log loss and calibration of
P(solvable) on positive and negative boards vs M. Decision rule: null-search updating beats "ignore null search" on
negative boards by >= 20% log loss at M = 8 (paired CI, 3 seeds). Compute: GCP L4s, budget group `overnight-sud`,
cap $50; MATS cluster as a second lane.

## Process and timeline (UTC)
00:50 design sent to gemini for review (this file) and to the executors to start building.
01:20 review deadline; executors apply accepted changes; smoke runs priced.
02:00 launch (no launch without a priced smoke); fleet-now change line with the plan and ETA.
every ~2 h: executors post progress (fleet-now change + agent mail to opus-mats); opus-mats checks in.
07:30 RESULTS.md + audit per experiment; entries on /experiments; opus-mats summary to Kelvin by 08:00Z.
Spend ledger: GPU-LOG.md per budget group; hard caps in gpu-up.sh; total <= $200 (reserve $40 unallocated).
