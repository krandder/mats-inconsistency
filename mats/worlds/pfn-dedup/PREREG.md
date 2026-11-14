# PREREG: how does a set-style PFN deduplicate, as a function of depth? (opus-mats, 2026-10-26)

Written before any learned run. Kelvin: "Let us run this experiment. I somehow expected attention to be responsible
for much of the deduplication; this will show."

## World (linear-Gaussian, exact answers)
- Episode: Q = 9 questions (8 labelled, 1 query), S = 6 sources.
- y_q ~ N(0, 1) i.i.d.
- Partition pi of the 6 sources into groups. Prior: 0.3 on all-singletons, 0.7 on the Ewens/CRP(alpha=1) law over
  all 203 set partitions (the all-singletons partition also gets its Ewens mass).
- Reading o_{s,q} = y_q + z_{g(s),q} + e_{s,q}, with z ~ N(0, sigma_g^2 = 1) shared by a group (per question) and
  e ~ N(0, sigma_e^2 = 0.25) idiosyncratic. Variances fixed and known.
- Duplicates: with probability 0.3, k ~ U{1,2,3} distinct sources are copied: an extra row of tokens with the SAME
  source identity and identical values, for all 9 questions. A copy carries no information.
- Labelled questions show y_q; the query question shows only the readings.

## Tokens (set-style, no positional encoding)
One token per (source row, question): [source-identity vector (random N(0,1)/4 in R^16, fresh per episode),
question-identity vector (same, R^16), value o, label y (0 if missing), missing flag, 0/1 query-question flag].
One query token: [question-identity of q*, query flag]. Context tokens attend to all valid context tokens; the query
token attends to valid context tokens and itself; nothing attends to the query token. Output from the query token:
Gaussian mean and log-std; loss = NLL of y_{q*}.
KEYS arm: each token also gets a group-identity vector (fresh random per episode, R^16), so the partition is visible.

## Exact references for y_{q*} (computed in numpy on the test set)
- (a) count-all-iid: every reading of the query question (copies included) treated as independent, variance
  sigma_g^2 + sigma_e^2. Over-counts duplicates and shared errors.
- (b) iid-distinct: distinct sources independent (copies removed). Correct for all-singleton episodes.
- (c) GLS-oracle: true partition known. Equals Bayes in the KEYS arm.
- (d) exact Bayes (no-keys arm): posterior over all 203 partitions from the 8 labelled residual vectors and the
  query readings' marginal, then the mixture of GLS predictives.
Unit tests: (d) with a prior concentrated on one partition equals (c); (d) matches an independent Monte-Carlo
computation (prior sampling + generic joint-Gaussian conditioning) at small S; references reduce correctly when
there is no sharing.

## Arms
depth L in {1, 2, 4, 8}, width d = 96, 4 heads, MLP 4d, pre-norm; KEYS absent / present; seeds 0-2. 24 runs.
Fresh episodes every batch (batch 256), never repeated. AdamW lr 3e-4, warmup 1000, then constant.
MAX_STEPS = 40000 (extend to 80000 once for any arm that fails the convergence rule, if budget allows).
Validation: 4096 fixed episodes every 1000 steps; report the best checkpoint.
Convergence rule: converged if best validation NLL improved by < 0.003 nats over the last 25% of steps; otherwise
UNCONVERGED (reported, not hidden).

## Metrics (fixed test set, n = 8000, disjoint seed)
- NLL gap to the arm's Bayes ((d) without keys, (c) with keys), mean +- se, paired; also the gaps of (a), (b), (c).
- Split by redundancy type: duplicates present / absent; partition all-singletons / max group size 2 / 3 / >= 4.
- Levels (each scored per depth, mean over seeds):
  - L1 handles duplicates: on duplicate episodes with all-singleton partitions, the model closes >= 75% of the NLL
    gap between (a) and Bayes.
  - L2 handles independent repeats: on all-singleton episodes without duplicates, NLL gap to Bayes <= 0.02 nats.
  - L3 handles shared errors: on episodes with a group of size >= 2, the model closes >= 75% of the gap between (b)
    and Bayes.
- Mechanism (best model per keys arm; cheap):
  - linear probes per layer on context tokens: residual r = o - y (labelled tokens), and source group size;
  - per-layer ablation: context tokens attend only to themselves at that layer; measure NLL and which reference
    the prediction moves toward;
  - final-layer attention of the query token, summed per source over the query question's tokens, versus the GLS
    per-source weight (proportional to (1/m) / (sigma_g^2 + sigma_e^2/m) for a source in a group of size m).

## Predictions
1. Depth 1 behaves like (a): it over-counts duplicates and shared errors.
2. Depth >= 2 reaches L1 and L2.
3. Without keys, L3 needs more depth than with keys (the correlation must be inferred from labelled residuals).
4. With keys, L3 at depth 2.
5. Kelvin's hypothesis to test: if attention is responsible for the deduplication, the final-layer query attention
   per source should track the GLS weights (smaller per source in larger groups); if earlier layers do it by
   rewriting values, the final attention can stay roughly uniform while the values carry the correction.

## Budget and stop
At most $6 of L4 spot GPU in total; VMs deleted after each job. No result is reported as a "finding" without the
adversarial review appended to the report.

## Amendment (2026-10-26 21:45Z, before any learned result was seen)
Run pfndedup-1026 (spot L4) was preempted after ~38 min of training with no arm finished and nothing synced
($0.63). Relaunching unchanged on an on-demand L4 (GPU_PROVISIONING=STANDARD), 8 arms in parallel.
Relaunch pfndedup-1026b failed before VM creation (on-demand L4 stockout in us-central1-a/b/c; $0). Retrying the same
code across more zones (us-east4, us-east1, us-west1, us-central1), on-demand.

## Amendment: eval-only duplicate sweep (coordinator request, written before any learned result)
mech2.py, on all saved models (24 trained + matched untrained initialisations, same seeds), 600 episodes without
duplicates (seed 40000), one source s0 per episode copied to m = 1..8 rows (same identity, identical values):
- per layer and head: query attention mass on the m copies' query-question tokens (total, and per copy);
- predictive pull toward the duplicated value: OLS slope of (mu_m - mu_1) on (o_dup - mu_1); log-precision shift;
  references: true duplicates (Bayes) = 0 for all m; independent copies (GLS + m-1 independent readings) rises in m;
- count decodability: retrained ridge probes (episode-level split, lambda by inner CV) for log m from the copies'
  hidden states per layer; within-episode version from (h_m - h_1); shuffled-target baseline; untrained-network
  baseline; mean ||h_m - h_1|| per m (count may enter multiplicatively with content, invisible to a linear probe);
- intervention: patch the m=4 copies' residual after layer l with the m=1 run's state of the original source token;
  if the patched model's pull rises toward the independent-copies reference, count information at layer l was used.
Prediction: trained models with depth >= 2 keep the pull near 0 (true-duplicate Bayes); if dedup happens in the
readout attention, total query mass on the copies stays roughly constant in m (per-copy mass ~ 1/m).

## Amendment (2026-10-27 01:25Z, opus-mats; operational only, no design change)
Run pfndedup-1026c hit its 3.5 h VM timeout (6 arms in parallel on one L4 was slower than estimated; cost $3.08).
Complete: all L1 and L2 arms (12). Incomplete: L4 (stopped at 12k-36k steps), L8 (2 arms at 6k steps, 4 never started).
Relaunch L4 and L8 (all 12 arms, fresh, identical settings, 40000 steps) on 4 L4 VMs, 3 arms each (ARMS env added to
run_all.sh/launch.sh to select arms; nothing else changed). L1/L2 checkpoints from pfndedup-1026c are evaluated as is.
Budget: Kelvin's $200 total for MATS compute until MATS funding arrives.

## Post-hoc additions (2026-10-27 ~02:20Z, written AFTER results were seen; exploratory, not pre-registered)
- analysis2.py: MSE of predictive mean to the Bayes mean per split (dedup separated from variance calibration), and
  probes/attention for all seeds with matched untrained baselines.
- mech3.py: readout score deduplication beyond softmax arithmetic (m*q1/(1+(m-1)*P1)).
- mech4.py (requested by the adversarial reviewer): patching controls after layer 0: copies vs rest-of-context vs all
  vs norm-matched random.
- Spot relaunch 1027a/b (depth 4) and 1027d (depth 8 keys) were preempted with nothing synced; depth 4 not relaunched
  (the $6 cap). 1027c (depth 8, no keys, 3 seeds, 40000 steps) continues.
- 2026-10-27 03:50Z: relaunch VMs a (L4 k0), b (L4 k1), d (L8 k1) lost to spot preemption (ssh gone, no sync); c (L8 k0) completed. Relaunching a/b/d on on-demand L4s (GPU_PROVISIONING=STANDARD), same settings.
- 2026-10-27 ~06:30Z: the coordinator relaunched depth 4 (both arms) and depth 8 with keys on demand (pfndedup-1027e/f/g,
  $4.04, VMs deleted). These replace the timed-out depth-4 models in the report; all arms now 40k steps. Total cost ≈ $10.9.

## Amendment: prior-level vs in-context split (2026-10-27, written BEFORE computing any of it; eval-only, CPU)
Kelvin: "prior-level discounting is very different from the kind of deduplication and info aggregation that I'm
interested in. Its good to have these two things separated."
References that use no within-episode comparison of tokens (all ignore keys, labelled residuals and source identity):
- R1, best additive pool: mu = a_n + sum over rows r of g_n(o_r), o_r = the query-question reading of row r, n = number
  of rows (6..9). g_n is a ridge fit on a basis of o (linear + 11 hat functions, knots -5..5), separately per n,
  target y*, on 300k fresh training-distribution episodes (seed 50000). It knows the row count and each token's own
  value, never compares tokens: an exact copy just adds g_n(o) again. (Labelled tokens add nothing to an additive
  predictor of y*, whose residuals have mean zero, so they are omitted.) This is the "prior-level" reference.
- R2, duplicate-blind Bayes on the query readings: exact posterior mean of y* given the n query readings only, with
  identities removed and no labelled residuals, under a model in which all n rows are distinct sources (prior 0.3
  all-singletons + 0.7 Ewens(1) over partitions of n rows, Bell(n) <= 21147 partitions). It cannot exploit exact value
  equality as such: the likelihood is a continuous Gaussian density, so two equal readings are scored exactly like two
  readings 0 apart from different sources (likelihood ratio for same-group vs not at distance 0 is sqrt(5)). It does
  compare query values (clustering by proximity), so R2 = "prior + query-value comparison", an upper bracket on what
  needs no identities or labels.
- R2eq (reported only as a leak bracket): drop exact duplicate values, then R2 on the distinct readings.
Metric: MSE of the predictive mean to the arm's Bayes mean ((d) no keys, (c) keys) on the fixed test set (8000,
seed 20000); equivalent to excess MSE for y*. Splits: dup_singletons (duplicates, all-singleton partition),
shared_nodup (a group of size >= 2, no duplicates), singletons_nodup.
Split of each model's improvement over count-all (a), per split: prior-level share = (E_a - E_R1) / (E_a - E_model),
in-context share = (E_R1 - E_model) / (E_a - E_model); the same with R2 in place of R1. Shares are reported only
where E_a - E_model > 0.005 (on singletons_nodup count-all ~= Bayes, so there I report MSEs only).
Prediction: depth 1 ~= prior level (its MSE close to R1's on duplicates and shared errors); deeper models and keys add
in-context aggregation, mostly on shared errors.
- Addendum (written before computing it): the preregistered "improvement over count-all" share is ill-posed on
  dup_singletons, because count-all's iid assumption is exactly right on that subset except for the copy (R1/R2 are
  worse than count-all there). Two additions, labelled post-hoc in the report:
  (1) duplicate excess = MSE(dup_singletons) - MSE(singletons_nodup) for models and references; in-context share =
      (excess_R1 - excess_model) / (excess_a - excess_model).
  (2) decisive identity test on the 600 copy-sweep episodes: add one row that is (i) a true copy of s0 (same identity,
      same values), (ii) same values on all 9 questions but a new identity, (iv) a new independent source (new identity,
      fresh labelled readings y + z_new + e) whose query reading equals o_s0. Pull toward o_s0 vs the 6-row episode.
      R1 and R2 give identical pulls for (i), (ii), (iv) (same query values, same row count). pull(iv) - pull(i) > 0
      is in-context deduplication; (ii) shows whether identity or value equality is used. (iv) is out of distribution
      (7 distinct sources never occur in training); stated as a caveat. Prediction: depth 1 pull(i) ~= pull(iv).

## Amendment A2 (2026-10-28, written before any A2 training): is the no-keys dedup ceiling a weak-signal effect?
Observation: without keys, depth >= 2 discounts a copy by 44-49% relative to an independent source with the same
query reading, uses value equality rather than identity (a new-identity copy is discounted the same), and depth 4/8
add nothing over depth 2. In the original world copies are rare (30% of episodes, one extra row per copied source),
so the training signal for deduplication is small (count-all's NLL gap on duplicate episodes is 0.074 nats).
Variant world (PFN_WORLD=costly, world.py): with prob 0.8 an episode copies k ~ U{1,2,3} distinct sources, each
c ~ U{1,2,3} times (up to 4 rows per source, up to 15 rows). Everything else unchanged: no keys, depth 1/2/4,
seeds 0-2, 40000 steps, same optimiser, validation and convergence rule. Control: the existing original-world
no-keys depth 1/2/4 models (same seeds and settings), not retrained.
Evaluation (CPU, identical episodes for variant and control models): the copy-vs-independent identity test
(split_copy.py: (i) copy, (ii) same values new identity, (iv) independent source with the same query reading) and
the copy sweep (mech2.py, m = 1..8; m <= 4 is now in distribution), plus each model's own test NLL gap.
Predictions if weak signal is the cause:
 P1. depth >= 2 in-context dedup 1 - pull(i)/pull(iv) rises well above 0.44-0.49 (>= 0.70);
 P2. identity is used: pull(ii) exceeds pull(i) by >= 25% of [pull(iv) - pull(i)];
 P3. depth helps beyond 2: depth 4 dedup exceeds depth 2 by >= 0.10, or its m = 4 pull is <= 2/3 of depth 2's;
 P4. pull at in-distribution m = 4 stays <= 0.05 for depth >= 2.
If instead the ceiling is architectural/optimisation-bound: dedup stays near 0.5, (ii) ~= (i), depth 4 ~= depth 2.
Depth 1 is predicted to stay prior-level (dedup <= 0.2) in both worlds.
Compute: on-demand L4 (GPU_PROVISIONING=STANDARD), 3 VMs x 3 arms, 5.5 h cap each; hard spend cap $15.

## Post-review-2 correction (2026-10-28 ~02:00Z; written after review 2 and a 100-episode smoke on L1_k0_s0 only)
Review 2 found that table A's "independent source" kept an exact tie with o_s0 at the query question, which in
training only ever means a copy. Table A is withdrawn as the headline test. Replacement (split_indist.py, seed 78000,
600 all-singleton episodes): the in-distribution weight ratio rho = w(s0 with its copy)/w(s1) (rho = 2: no within-
episode dedup, rho = 1: full), following the reviewer's method in an independent implementation, plus cue-removal
variants of the copy (query tie removed; labelled equality removed; identity and group key changed), which are out of
distribution and labelled as such. Shares are re-reported in three tiers: prior (R1), query-value comparison (R2 - R1),
labelled/identity/key evidence (beyond R2), with both the count-all-anchored share (preregistered) and the share of
R's remaining error removed. The split prediction "depth 1 ~= prior level" is scored against R1, as preregistered.
- A2 evaluation change (2026-10-28 ~02:05Z, before any A2 model exists): after review 2, the copy-vs-independent test
  with a tied query reading (split_copy.py) is confounded. P1-P3 are scored on the in-distribution test
  (split_indist.py: 2 - rho for one copy, plus cue removal), and also on the tied test for continuity. The
  original-world controls score 0.36 / 0.26 / 0.25 (depth 1/2/4) in distribution, against 0.09 / 0.44 / 0.49 on the
  tied test. P1 becomes: depth >= 2 in-distribution dedup >= 0.55 (more than double the control's 0.25-0.26). P3:
  depth 4 exceeds depth 2 by >= 0.10. P2 (identity use): the new-identity-and-key variant loses >= 0.10 of the dedup.
  Added: four-row copies (s0 plus 3 copies, in distribution only in the variant world): dedup = (4 - rho)/3.
- A2 results (2026-10-28 03:40Z): P1 true (0.61 / 0.77), P2 true at depth >= 2, P3 true, P4 true, depth-1 stays
  prior-level false (0.36 control, 0.61 costly). Report section "Track A2". Cost $4.29; VMs deleted.
