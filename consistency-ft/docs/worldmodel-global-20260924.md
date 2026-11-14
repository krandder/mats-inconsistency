# How much structure is in Qwen3-32B's beliefs over 400 events at one date? (2026-09-24)

Question (Kelvin, 24 Sep): take one date t and many open events, and ask Qwen3-32B for every
marginal, for pairs and for 3-way ANDs. Then fit one global joint distribution to its answers,
as a post-hoc analysis with no training. How much structure is in the fitted world model, does
it predict answers it never saw, is it more accurate, and are its dependencies right?

**Answer: almost none of the structure survives the fit.** The global fit predicts held-out
answers much better than independence of the stated marginals: KL 0.033 against 0.071. But a
fit with **no pairwise terms at all** does as well (0.0326). Nearly all of that gain is one
uniform bias: the model's stated ANDs sit far above the product of its marginals. Nearly half
(46%) of its "A and B" answers are exactly 0.15. The mean stated AND is 0.216, while the
realised both-YES rate is 0.037. Against outcomes, plain independence of the stated marginals
beats every fit on the joints. The weak pairwise signal that remains is directionally right
for similar pairs. It correlates with 60-day price co-movement (Spearman 0.18 for stated log
odds ratios, 0.13 for fitted ones; random pairs about 0). The fitted pair tables also predict
realised joint outcomes slightly better than independence, by 0.001 nats, which is significant
but tiny.

## Setup

- **Date t = 2025-09-15 00:00 UTC.** It falls 4.5 months after the Qwen3 release
  (2025-04-29), so no outcome after t can be in the weights, and the as-of line then only sets
  tense. The number of eligible test-split parents grows as t moves later: 124 on 15 Jun, 193 on
  15 Jul, 266 on 15 Aug and 322 on 15 Sep. On 15 Sep there are 424 eligible propositions, so
  400 events could come from the test split alone, with no "unused" parents needed. That date
  also leaves a 60-day price window (from 17 Jul) and puts most resolutions in Q4 2025.
- **Events** (`scripts/worldmodel_global.py build`):
  - 400 propositions: 271 binary markets and 129 multiple-choice options (at most 4 per parent,
    chosen by most bets before t), from 306 parents, all from the TEST split of
    jev-disjoint-v1-filtered-20260919. The code asserts that none is a train or validation
    parent (of jev-disjoint-v1, of -filtered or of jev-pairings-v1) or one of the 178 excluded
    parents.
  - Each event is public, created before t and open at t: not closed, not resolved, and no
    resolution logged at or before t. Each final resolution is YES or NO, strictly after t.
  - YES rate 17%. The question text is the as-of-t version.
- **Similarity**: BGE-small (bge-small-en-v1.5) cosine on "question — option".
- **Queries** (one call each; the formula of each is recorded in `queries.jsonl`):

  | kind | n |
  |---|---|
  | marginals | 400 |
  | "A and B", similar pairs (union of each event's top-10 neighbours) | 2,894 |
  | "A and B", random non-neighbour pairs (same number) | 2,894 |
  | "A and not B" (10% of each pair set, random direction) | 289 + 289 |
  | 3-way ANDs on triangles of the neighbour graph (highest mean cosine; each event in at most 2) | 200 |
  | total | 6,966 |

- **Prompt**: v6_direct, the same as in `worldmodel_clusters.py`. It has BASE_SYSTEM, the line
  "Assume today is 2025-09-15", the question blocks, the statement, ANSWER_DIRECT and
  " /no_think". There is no market data. Settings: qwen3-32b on DashScope, thinking off,
  temperature 0, max_tokens 20, and one retry on a parse failure. Results: 6,966 calls, all
  served by qwen3-32b, 0 retries, 0 missing.
- **Cost**: **$2.61** at the conservative cap price (0.7/2.8 $/M). That covers 3.54M prompt
  tokens and 49k output tokens (cap $5.5, 12 workers).

## Global fit (`scripts/worldmodel_global_fit.py`)

- **Model**: a pairwise binary MRF (Ising in {0,1} coordinates),
  log p(x) = Σ h_i x_i + Σ_(i,j)∈E J_ij x_i x_j − log Z. E is the 5,788 queried pairs.
- **Objective**: minimise Σ_q KL_Bernoulli(stated_q ‖ implied_q) + λ·ΣJ², by L-BFGS
  (autograd). Stated values are clipped to [0.01, 0.99].
- **Inference**:
  - Junction tree: not possible. The queried graph has treewidth ≤ 287 (min-degree bound), and
    the similar-pair graph alone has treewidth ≤ 103 (min-fill bound).
  - So it uses damped loopy BP (Bethe), run to convergence (message change < 1e-10), plus 30
    differentiable iterations from the fixed point.
  - Pair beliefs give the pairs. A Kirkwood closure gives the triples.
  - Sequential clamping gives any query without an edge (for example, held-out pairs).
- **Checking inference**: a chromatic Gibbs sampler checks it.
  - At the final fit, BP and Gibbs agree to 0.0006 on marginals and 0.0005 on P(A∧B).
  - Every held-out and accuracy number was recomputed by sampling the fitted MRF directly,
    and they agree to 3 decimals.
- **Bayes-market's junction-tree code**
  (`/home/agent/workspaces/bayes-engine/inference/factored_market.py`, an LMSR with a width
  budget) was not reused, because the queried graph's treewidth is far beyond any width
  budget.
- **Penalty choice**: a grid over L2 {0.01, 0.03, 0.1, 0.3} and L1 {0.01, 0.03} on a 15%
  validation split. Validation KL was 0.0322 at L2 0.03, 0.0336 at 0.1, 0.0338 at 0.3 and
  0.0332 at L1 0.03. BP did not converge at 0.01.
  - The rule takes the largest λ within 5% of the best, which gives **L2 0.3**.
  - L2 0.03 was the best on validation, but on the full query set its BP stops converging
    (residual 3e-5 after 3,000 iterations), and Gibbs then disagrees with it by 0.066 on
    marginals.
  - An earlier run unrolled only 80 BP iterations, and the optimiser fitted a transient that
    was not the BP fixed point. The Gibbs check caught it, and those numbers were discarded.
    The logs are kept in the data directory (`fit-grid1.log`, `fit-grid2-unconverged-bp.log`,
    `fit-run3-lam003-unconverged.log`).
- **Controls**:
  - independence of the stated marginals;
  - a **marginals-only fit** (J ≡ 0, with h fitted to the same queries);
  - a local "pairwise" projection: the 2×2 or 2³ table fitted to each pair's or triple's own
    queries, as in `coherence_projection.py`. For a marginal it takes the mean of that event's
    pair projections.

## (a) Structure

| | similar pairs | random pairs |
|---|---|---|
| stated 2×2 (A, B, A∧B) incoherent (a negative cell) | 64% | 52% |
| stated \|log OR\| > 0.5 (all; coherent only) | 92.5%; 80% | 87.1%; 73% |
| stated median \|log OR\| | 3.8 | 2.5 |
| local 2×2 projection \|log OR\| > 0.5 | 91.7% | 87.3% |
| global fit \|log OR\| > 0.5, chosen L2 0.3 | 0.03% (1 edge) | 0% |
| global fit, L2 0.1 (converges; validation KL +4%) | 0.9% (32 edges in total) | 0.2% |
| global fit, L2 0.03 (best validation; BP unreliable on the full set) | 12.9% | 6.1% |
| fitted \|J\| < 0.05, chosen fit | 72% | 80% |
| corr(stated log OR, fitted log OR) | 0.63 | 0.62 |

Pair by pair, the stated answers look like dense structure: |log OR| > 0.5 on nearly every
pair, and similar pairs are only slightly more "dependent" than random ones. Most of this is
incoherence, not dependence. The stated ANDs are too high for the stated marginals: an "A and
B" answer of 0.15 against marginals of 0.15 and 0.25 gives a huge log odds ratio. The global
fit explains most of this by raising the marginals: the mean fitted marginal is 0.458 against
0.407 stated. Once that is done, very little pairwise coupling is needed.

Graphs:

- **Queried graph**: 400 nodes, 5,788 edges, one component, treewidth ≤ 287.
- **Similar-pair graph**: treewidth ≤ 103, max clique 11, 11 Louvain communities of 13–65
  events, modularity 0.61. These are the topic clusters of the embedding, and their contents
  were not inspected here.
- **The dependence network itself is sparse and tree-like.**
  - At L2 0.1, |log OR| > 0.5 gives 32 edges on 57 events in 25 components. The largest
    component has 5 events, and the treewidth is 1 (a forest).
  - At the chosen fit, |log OR| > 0.1 gives 275 edges on 271 events in 25 components (largest
    204), with treewidth ≤ 3. Of these edges, 73% are similar pairs and 66% are positive.
  - The strongest L2-0.1 edges make sense: near-duplicate "Ukraine recognises Russian
    sovereignty" markets (+1.0); Bitcoin $150k with $140k/$130k (+0.7 to +0.8); US recession
    with an NBER-announced recession (+0.7); "Trump talks to Putin" in two markets (+0.6).
    Some are noise, such as Intel CEO vs Cracker Barrel CEO (−0.8).

## (b) Held-out prediction (20% of pair units and 20% of triples: 1,198 units, 1,307 queries, never in the fit)

| held out | KL(stated ‖ ·): fit | marginals-only fit | independence of stated marginals | log loss vs outcomes: fit | marginals-only | independence | raw answer |
|---|---|---|---|---|---|---|---|
| similar pairs (579) | 0.0338 | 0.0339 | 0.0846 | 0.330 | 0.331 | **0.292** | 0.365 |
| random pairs (579) | 0.0307 | 0.0309 | 0.0561 | 0.282 | 0.283 | **0.248** | 0.268 |
| triples (40) | 0.0381 | 0.0393 | 0.0968 | 0.239 | 0.239 | **0.198** | 0.284 |
| all | **0.0325** | 0.0326 | 0.0712 | 0.304 | 0.305 | **0.267** | 0.316 |

- Fit − independence, KL: −0.039 [−0.045, −0.033] (95% bootstrap over units).
- Fit − marginals-only fit, KL: **−0.0002 [−0.0003, −0.0001]**.
- Fit − raw, log loss: −0.012 [−0.025, +0.001].
- Correlation of stated with predicted: 0.76 (fit) vs 0.68 (independence).

So the fitted world model does predict answers it never saw: KL is more than halved relative
to independence. But almost all of that comes from learning the model's uniform AND inflation,
not from which events it links. Against outcomes, the fit's prediction of an unseen joint is a
little better than the model's own answer to that query, but worse than plain independence of
its stated marginals.

## (c) Accuracy vs outcomes (every query, log loss / Brier, clip 0.01)

| queries | n | YES | raw | local projection | global fit | marginals-only fit | independence of stated marginals |
|---|---|---|---|---|---|---|---|
| marginals | 400 | 17.2% | 0.627 / 0.214 | **0.618** / 0.212 | 0.660 / 0.227 | 0.661 / 0.228 | 0.627 / 0.214 |
| A∧B, similar | 2,894 | 4.2% | 0.350 / 0.103 | 0.323 / 0.092 | 0.310 / 0.086 | 0.310 / 0.087 | **0.270 / 0.079** |
| A∧B, random | 2,894 | 3.2% | 0.278 / 0.075 | 0.262 / 0.068 | 0.280 / 0.072 | 0.281 / 0.073 | **0.244 / 0.066** |
| A∧¬B | 578 | 13.5% | 0.519 / 0.173 | 0.453 / 0.145 | **0.444** / 0.138 | 0.445 / 0.138 | 0.452 / 0.140 |
| A∧B∧C | 200 | 2.0% | 0.250 / 0.065 | 0.245 / 0.062 | 0.189 / 0.045 | 0.187 / 0.045 | **0.161 / 0.043** |
| all joints | 6,566 | 4.5% | 0.330 / 0.095 | 0.305 / 0.085 | 0.305 / 0.083 | 0.306 / 0.084 | **0.271 / 0.077** |
| all | 6,966 | 5.2% | 0.347 / 0.102 | 0.323 / 0.093 | 0.325 / 0.092 | 0.326 / 0.092 | **0.292 / 0.085** |

- **Global vs raw** on all joints: −0.025 [−0.031, −0.020]. On the marginals it is **+0.033
  [+0.001, +0.065]**: the fit drags the marginals up toward the inflated ANDs.
- **Global vs local** on all queries: ≈ 0 (95% CI −0.002 to +0.006).
- **Both coherence fits beat the raw answers on joints.** The best forecast of every joint is
  still the product of the model's own stated marginals. Its joint answers carry negative
  information about outcomes.

## (d) Are its dependencies right?

**Prices.** The window is the 60 days before t, using daily log-odds changes. Of the 400
events, 113 moved on at least 5 days, and 525 queried pairs have both events moving.

| pairs | n | Spearman(price co-movement, stated log OR) | Spearman(·, fitted log OR, chosen fit) |
|---|---|---|---|
| similar | 298 | **0.18 [0.06, 0.29]** | 0.13 [0.01, 0.24] |
| random | 227 | 0.01 [−0.12, 0.14] | −0.06 [−0.19, 0.07] |

- Among similar pairs with |price corr| > 0.3 (77 pairs), the stated log OR has the same sign
  in 77% of them. Chance is about 61%, since 70% of stated log ORs and 78% of the strong price
  correlations are positive. The chosen fit's log OR has the same sign in 71%.

**Realised outcomes.** The score is the 4-cell log loss of the realised (A, B) outcome, fitted
pair table minus the product of fitted marginals:

- similar pairs: **−0.0013 [−0.0017, −0.0008]**, slightly better;
- random pairs: +0.0007 [+0.0003, +0.0010], slightly worse.

The mean fitted P(A∧B) is 0.215 against a realised both-YES rate of 0.037. That miss is the
level bias again, not the dependence.

The dependencies the model states are directionally right for related events and null for
random ones. The effect is weak, and it is swamped by the model's conjunction inflation.

## Caveats

- The stated answers are coarse. 46% of pair ANDs are exactly 0.15, and the marginals sit
  mostly on 0.15/0.25/0.35/0.65/0.75. Much of the "structure" in a single 2×2 is this
  default, not a belief about the pair.
- Structure in the fit depends on λ. The penalty that validation prefers (0.03) keeps 12.9% of
  similar pairs strong, but its BP is unreliable on the full query set. The chosen 0.3 keeps
  almost none. Held-out KL separates these by only 5%, so the data cannot pin the pairwise
  terms.
- Price co-movement is available for only 113 of the 400 events (many Manifold markets rarely
  trade). BGE cosine above about 0.6 is often topical, not causal: the similar set includes
  some unrelated pairs.
- Triples are fitted and read through a Kirkwood closure of the Bethe pair beliefs when all
  three edges are in the fit, and through clamped BP otherwise. Direct Gibbs sampling of the
  fitted MRF gives the same held-out and accuracy numbers.

## Files

Code:

- `scripts/worldmodel_global.py`: selection, queries and the run.
- `scripts/worldmodel_global_fit.py`: the MRF fit, held-out, accuracy, structure and prices.
- `scripts/worldmodel_global_network.py`: the network chart data, the λ=0.1 refit and the
  independence rows.

Data in `/home/agent/data/worldmodel-global-20260924/`:

- `events.jsonl`, `queries.jsonl` (prompts and formulas), `log.jsonl` (raw replies) and
  `graph.json` (pairs, triples, cosines);
- `fit-per-query.jsonl` (raw, local, global, Gibbs, held-out flag), `fit-pairs.jsonl` (stated,
  local and fitted log OR, J) and `fit-params*.pt`;
- the logs.

Numbers:

- `docs/worldmodel-global-20260924.json`: the full results.
- `docs/worldmodel-global-20260924-charts.json`: chart-ready data, including
  `dependence_network_lambda_0.1` and `dependence_network_chosen_top` (nodes = events with
  stated/fitted marginal, outcome, Louvain community and spring-layout x/y; edges = fitted log
  OR, similar or random, cosine, same parent), plus log-OR and price scatters and the
  accuracy and held-out tables.
