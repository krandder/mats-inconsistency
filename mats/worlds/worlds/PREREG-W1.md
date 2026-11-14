# W1 junta pilot: preregistration (2026-10-28, before any training)

Code: `junta.py` (world, exact Bayes, references), `harness.py` (world-agnostic PFN harness), `diag_junta.py`
(constructed-context diagnostics), `test_junta.py`, `run_pilot.sh`. SHA-256 in `code_sha256.txt`.
Spec source: `../world-critique/DESIGN-v1.md` (W1) plus the coordinator's round-2/3 and literature amendments.

## World (final W1 spec)
- Items x in {0,1}^16. Hidden R uniform over the 120 coordinate pairs. Cell values g(c) ~ N(0,1) (4 cells).
- Item value v_x = g(x_R) + b_x, b_x ~ N(0, tau2); pilot both tau2 = 0 (plain junta) and tau2 = 0.25.
- Fresh reading y ~ N(v_x, sigma2), sigma2 uniform on {0.1, 0.5} per episode.
- Provenance: each fresh reading carries a random event-id vector in R^8. Token t is, with per-episode probability
  p_copy, an exact copy of a uniformly chosen earlier token (same item, y and id). p_copy = 0 with prob 1/3, else
  U(0, 0.6). Hidden-ancestry relays are dropped (their exact oracle needs a sum over partitions; and the
  fresh-vs-copy odds would then involve the policy's item probabilities, so Bayes would stop being policy-free).
- Policies, all ignorable (visible bits, recorded observations, independent randomness only): uniform; Zipf(a),
  a ~ U(0.8, 1.5), ranks hashed to items; pattern (prob U(0.5, 0.95) of matching a random bit pattern on 2-6 random
  coordinates chosen independently of R); repeat (prob U(0.5, 0.95) of one of K ~ U{1..16} items); UCB on running
  item means (eps = 0.2). Train on the first four; UCB is held out. Every structural count is randomised per
  episode (copy rate, Zipf exponent, pattern rate and width, repeat pool and rate), and the training N is log-uniform
  in [8, 64].
- Query: with prob 0.5 a uniformly chosen distinct context item, else a uniform item. Target v_q.
- Tokens: 16 item bits (+-1), y, event id (8), query flag. The query token is last; no positional encoding.

## Exact Bayes and references
Bayes: collapse tokens by event id; per item n_x, mean and within-item SS; for each (R, sigma2), item means are
N(g_c, tau2 + sigma2/n_x) given g_c and the within SS informs sigma2; g_c is integrated analytically; the result is a
mixture of 240 Gaussians for v_q. It is verified by tests against dense joint-Gaussian conditioning (to 1e-9),
against ABC on the generative spec, by reductions (no copies: copy-naive = Bayes; copies removed: unchanged;
one hypothesis: oracle = Bayes), by replay ignorability (same policy randomness and observed values under a
different hidden world give identical items, provenance and query), and by calibration of Bayes z-scores under
every policy.
References: `prior` N(0, 1+tau2); `r0` fixed shrinkage of the query's raw readings (copy-blind; mean n*ybar/(n+k),
var a + b/(n+k)), fitted on training episodes (prior-level); `shortcut` Gaussian regression of the target on the
query's counts only (no values), fitted on training episodes; `item` item-only Bayes (pools repeats of one item,
no sharing across items); `count` copy-naive Bayes (copies treated as fresh); `oracle` Bayes with R known;
`bayes` full.
Abilities: similarity = fraction of the item->Bayes NLL gain the model achieves; dedup = fraction of the
count->Bayes gain on episodes with copies; decomposition = (NLL_r0 - NLL_ref) / (NLL_r0 - NLL_model) per reference.

## Pilot arms and evaluation
24 arms: tau2 {0, 0.25} x depth {1, 2, 4} x attention sink {0, 1} x seed {0, 1}; width 64, 4 heads, MLP 4x,
4-component Gaussian-mixture head, AdamW lr 5e-4 (1k warmup, cosine to 10%), batch 256, 40k steps, best-val
checkpoint (val = 2048 training-distribution episodes). The sink is a learned key every token sees (Barbero et al.
2024 ablation; crossed with depth 1 vs 2+).
Eval: N in {16, 32, 64, 128, 256, 512, 1024, 2048, 4096} x all five policies; 1000 episodes per cell up to N=512,
then 500/250/125. Metrics: NLL gap to Bayes (mean, se), KL(Bayes || model) by quadrature, gap of every reference,
bias and log variance ratio of the model predictive vs Bayes, splits (query observed / unobserved, copies / none).
Diagnostics (`diag_junta.py`, 100 constructed bases each): C1 contamination (M in {0,16,64,512,4048} readings of
one dominant item at Hamming distance 8 from the query, in another cell; fresh vs copy variants; matched-length
balanced control; Bayes and R-known-Bayes contamination simulated as anchors; query attention mass on the dominant
item vs its share); D1 R-evidence pairs (same items and value multiset, values permuted among non-query reports);
D2 k copies vs k agreeing fresh readings of the query's single reading (the arXiv 2610.16801 control; k in
{1,4,16,64}); D4 whole context x k as copies (Bayes unchanged) vs as agreeing fresh readings (Bayes sharper);
ridge probes of token states for log item count (fresh, raw) and log true-cell count.
C1 rule (DESIGN-v1): refuted if at any M where the matched-length control has KL < 0.05 the model's KL on the
fresh-dominant context exceeds the control's by > 0.10, or its KL grows > 0.03 per doubling of M. If the control
is never below 0.05 the rule is vacuous and is reported as such.

## Predictions (before training)
P1 Similarity. At N=64 in distribution, depth 4 achieves >= 50% of the item->Bayes gain and depth 2 >= 25%, for
   both tau2; depth 1 < 25% (it stays near item/prior level, as in the tiny-world runs).
P2 Dedup. At N=64 on copy episodes, depth >= 2 achieves >= 70% of the count->Bayes gain (ids make copies easy).
P3 Length. Gap to Bayes grows with N past the training range for every arm: gap(N=1024) > 2 x gap(N=64), and at
   N=4096 at least one policy has the model worse than `item`. The transversal crux is refuted for this
   architecture (gap grows).
P4 Held-out policy. At N <= 64 the UCB gap is within 0.05 nats of the mean in-distribution gap.
P5 Sink. Sink arms have smaller gap growth with N than no-sink arms of the same depth (by >= 20% at N=1024).
P6 C1. Without a sink, layer-1 query attention mass on the dominant item tracks its share M/(48+M) within 0.1;
   model contamination exceeds the Bayes anchor at M >= 512 for both fresh and copy variants; the rule is vacuous
   or refuted (control KL above 0.05 at large M).
P7 D4. Without a sink, x k copies leave the model exactly unchanged (token multiset scaled; a check, not a
   finding). For x k agreeing fresh readings the model gets < 50% of the Bayes log-variance drop at k = 8.
P8 D2. Depth >= 2: |log variance change| < 0.1 for k <= 16 copies; for agreeing fresh readings >= 50% of the
   Bayes log-variance drop at k = 4 and < 50% at k = 64.
P9 D1. Depth 4 at N=64: slope of model shift on Bayes shift >= 0.5; lower at N=256.
P10 Probes. Log item count (fresh) is linearly decodable (R^2 > 0.5) after layer 1 in depth >= 2 arms.

## Related work to cite
arXiv 2610.16801 (production PFNs over-count repeated measurements under an IID-rows prior; proposes a
copies-vs-fresh hierarchical prior, which this world implements; D2 is their missing control). Barbero et al.
2024 (softmax without a sink sees only proportions; D4 and the sink arms). Golkar et al. (fixed training counts
become biases; hence randomised counts).

# Amendment W1b (2026-10-28 ~06:30Z, before any W1b training): does precision scale with evidence?

Pilot finding being tested: past the training range the model's predictive variance does not shrink as Bayes's does
(tau2 = 0, depth 4: variance about 200x Bayes's at N = 4096). Question: is this (a) length extrapolation (it would
be fixed by training on larger N) or (b) representational (normalised softmax attention sees proportions, not counts),
and does a minimal precision channel fix it?

World: W1 with tau2 = 0 (the world where precision should grow with the cell count), otherwise unchanged.
Arms (depth 4, width 64, 4 heads, MLP 4x, 4-component mixture head, lr 5e-4 cosine, 100k steps, 3 seeds each;
training N drawn once per batch, log-uniform, batch size min(256, 16384 / N), so all arms see ~16k context tokens per
step):
  A  baseline, N_train in [8, 64]
  B  baseline, N_train in [8, 512]
  C  B + one unnormalised 'sum' head per layer (width 16): S_i = sum_j sigmoid(q_i.k_j / 4) v_j over context keys,
     output sign(S) log(1 + |S|) (log-counts and log-precisions become linear features)
  D  B + learned attention-sink token
  E  C with N_train in [8, 64] (does the precision channel extrapolate without seeing large N?)
Convergence rule (reported): best val NLL in the last quarter improves on the earlier best by < 0.003 nats.
Eval: N = 16..4096 x 5 policies (UCB held out), same seeds as the pilot eval set; gap to Bayes split into a
variance part 0.5(log r + 1/r - 1), r = v_model/v_Bayes, a mean part 0.5 (m_model - m_Bayes)^2 / v_model (moment
matched) and the remaining shape part; D2 copy/fresh reaction ratio (log-variance change for k copies divided by that
for k agreeing fresh readings; Bayes 0) at k in {4, 16, 64, 256}; C1 rule only at M where the matched control has
KL < 0.05 (reported as vacuous if never).

Pre-registered reading:
  - "(a) length extrapolation": B's gap at N = 256 and 512 is < 0.5 x A's, and B's gap grows again past 512
    (gap(4096) > 2 x gap(512)). "(b) representational": B's gap at N = 512 is >= 0.5 x A's.
  - The sharp architecture result ("a precision channel is needed for scaling") is claimed iff E extrapolates and
    A does not, in >= 2 of 3 seeds each: "extrapolates" = gap(4096) <= 1.5 x gap(64) on the training policies AND the
    variance part at N = 4096 <= 0.25 x A's seed-mean variance part.
Predictions:
  Q1 A: the variance part is >= 50% of KL at N = 4096.
  Q2 (a) holds: B is close to A's N=64 level up to 512 and degrades past it.
  Q3 C's variance part at N = 4096 is <= 0.5 x B's.
  Q4 E does not meet the strict extrapolation definition (my probability it does: 0.25), but its variance part at
     N = 4096 is < 0.5 x A's.
  Q5 D's variance part at 4096 is within +-30% of B's (a sink gives a saturating 1/(1+N) signal, not a count).
  Q6 Copy/fresh ratio at k = 16: about 0.3 for A and B; >= 0.5 for C and E (a sum counts copies unless it learns
     to weight by 1/multiplicity).
  Q7 C1 is vacuous for all arms (control KL never < 0.05 at tau2 = 0).

## W1b continuation (2026-10-28 ~08:45Z, after the 100k-step results, before any continued training)
No run met the convergence rule at 100k steps (last-quarter improvement 0.014-0.052 nats); the 100k results are
kept as runs_w1b/tables_100k.md. Every run is continued for 100k more steps from its final weights (fresh AdamW,
lr 2e-4 cosine to 2e-5, no warmup, same data settings); the convergence rule is applied to the whole 200k curve
(last quarter = 150k-200k). Readings and predictions above are applied unchanged to the 200k checkpoints (best val).
Deviation note: the readings were preregistered for "100k steps or until convergence"; the 100k-step tables are
also reported, so the reader can see whether any verdict changes with the extra training.
