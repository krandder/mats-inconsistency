# Adversarial review 2: prior-level vs in-context split (pfn-dedup-depth-2026-10-26.md, v4 headlines)

Reviewer: independent subagent, 2026-10-28. Scope: the v4 "Answer" section, section 0 (tables A–C), §3 item 7.
The first review (appended to the report) is not repeated. Code read: world.py, pfn.py, split.py, split_copy.py,
split_tables.py, mech2.py, PREREG.md (incl. the split amendment and its addendum).

## Spot-checks against result files (all match)

| headline number | file | recomputed |
|---|---|---|
| depth 1, no keys: copy 0.049, indep 0.054, 9% [0.07..0.10] | runs_final/split_copy.json | 0.0493, 0.0542; per seed 0.071 / 0.096 / 0.099 |
| depths 2–8 no keys 44–49%; keys 69%, 80–83% | split_copy.json | 0.44 / 0.49 / 0.49; 0.69, 0.81 / 0.83 / 0.80 |
| shared errors: 60–79% by R1, 75–99% by R2; beyond R2 0.01 / 0.11–0.14 / 0.19 / 0.24–0.25 | runs_final/split.json | per seed: depth 1 k0 0.021 / 0.002 / 0.010; k1 0.18–0.19; deeper k0 0.08–0.14; k1 0.24–0.25 |
| R1 0.076, R2 0.061 (table A refs) | split_copy.json | 0.0763, 0.0608 |
| table C depth 1: 0.68 | split.json | (0.0152 − 0.0084)/(0.0184 − 0.0084) = 0.68 |
| best copy pull 0.026 | split_copy.json | L4 k1 mean 0.0261 |

Arithmetic is right. The problems are in what the numbers are said to mean.

## Reviewer's own check (small, CPU, eval only)

Code and output: `/home/agent/workspaces/mats/work/pfn-dedup/review2_check/indist.py`, `indist.json` (all 24 models, 600 fresh
episodes with all-singleton partitions, seed 77000; bootstrap se over episodes). Two tests:

- **In-distribution weight ratio ρ.** Training-distribution episodes: 6 singleton sources plus one copy of s0, so 7 rows
  and 6 identities. ρ = (dμ/do* for s0, both rows moved together) / (dμ/do* for another source s1 in the same episode),
  by central differences (h = 0.05). Both sources sit in the same episode, so the row count and the softmax normaliser
  are shared. Any reader that does not know which rows are copies has ρ = 2 (checked: R1 1.99). Full deduplication has ρ = 1.
  2 − ρ is the share of the copy removed using within-episode evidence.
- **Untied independent source.** Table A's (iv) again, but the 7th source's query reading is drawn fresh (y* + z_new + e)
  instead of being set equal to o_s0. Pull = OLS slope on its own reading. Bayes on these episodes: 0.121.

| depth | keys | conv | ρ (in-dist.) [seed range] | 2 − ρ | copy pull | untied indep pull | 1 − copy/indep (untied) | table A (tied) |
|---|---|---|---|---|---|---|---|---|
| R1 | | | 1.99 | 0.01 | | | | |
| R2 | | | 1.66 | 0.34 | | | | |
| 1 | 0 | 3/3 | 1.56 [1.49..1.62] | **0.44** | 0.059 | 0.105 | **0.44** [0.38..0.50] | 0.09 |
| 2 | 0 | 1/3 | 1.74 [1.70..1.78] | 0.26 | 0.083 | 0.115 | 0.27 [0.24..0.33] | 0.44 |
| 4 | 0 | 3/3 | 1.74 [1.71..1.76] | 0.26 | 0.089 | 0.118 | 0.25 [0.23..0.27] | 0.49 |
| 8 | 0 | 1/3 | 1.68 [1.66..1.71] | 0.32 | 0.082 | 0.118 | 0.30 [0.27..0.32] | 0.49 |
| 1 | 1 | 2/3 | 1.51 [1.43..1.60] | 0.49 | 0.060 | 0.118 | 0.49 [0.40..0.55] | 0.69 |
| 2 | 1 | 2/3 | 1.32 [1.32..1.33] | 0.68 | 0.041 | 0.119 | 0.66 | 0.81 |
| 4 | 1 | 2/3 | 1.27 [1.23..1.30] | 0.73 | 0.035 | 0.119 | 0.71 | 0.83 |
| 8 | 1 | 1/3 | 1.36 [1.32..1.41] | 0.64 | 0.045 | 0.119 | 0.62 | 0.80 |

Bootstrap se: ρ 0.01–0.06 per model, pulls 0.001–0.004. The two independent methods agree to within 0.03 in every arm.
The depth-1 vs deeper no-keys gap holds in every seed (depth 1 max ρ 1.62 < deeper min 1.66).

## Problems, most severe first

### 1. CRITICAL: Table A's "independent source" carries copy evidence, the exact query tie. Without keys the result reverses.
- **Evidence.** In split_copy.py, `fresh[:, 0, W.QL] = o[ar, s0, W.QL]`: variant (iv) has a query reading exactly
  equal to o_s0. In training, two rows tie exactly only if one is a copy. The tie is therefore within-episode copy
  evidence, and (iv) keeps it. 1 − (i)/(iv) measures only the use of *labelled-question and identity* evidence. Use of
  the query tie scores as 0.
  - R2's lower pull (0.061 vs R1 0.076) already shows that proximity of the query readings alone discounts the row.
  - Depth 1 discounts tied rows: (iv) 0.054 tied vs 0.105 untied.
- **In distribution the ordering reverses** (table above):
  - Depth 1 without keys removes 0.44 of a copy, more than R2 (0.34) and far more than R1 (0.01).
  - Depths 2–8 without keys remove only 0.26–0.32.
  - With keys, depth 1 is about the same as without keys (0.49 vs 0.44), not 69% vs 9%. Keys help mainly at depth ≥ 2
    (0.64–0.73).
- **This matches table C,** which the report set aside. Depth 1's duplicate excess is 0.0084, against 0.0103–0.0117
  at depths 2–8 without keys. The reason given for dropping C, that softmax normalisation is a prior-level row
  discount R1 cannot express, does not hold. In a fixed 7-row episode the normaliser is shared by s0 and s1, so a
  softmax pool whose scores depend only on each token's own content gives ρ = 2. Something in depth 1 tells the
  duplicated source apart. The query token's post-attention MLP and LayerNorm act on the pooled vector, which makes depth 1
  a Deep Sets reader ρ(Σφ). That can pick up value and identity coincidences, for example through second-order terms
  in the pooled source-identity × value features.
- **Fixes.**
  - Make the in-distribution ρ test (or an untied (iv)) the headline test.
  - Report tied and untied (iv) side by side; their difference measures how much each model uses the query tie.
  - Optionally break down depth 1's 0.44: a copy whose query reading is perturbed by ε (tie removed, rest equal) vs
    an exact copy. This is OOD, so label it.
- **Headlines that fall:** "Depth 1 without keys is prior-level only" and "prediction confirmed"; "Depths 2–8 without
  keys deduplicate in context, about halfway"; "69% at depth 1" with keys; §3 item 7; and "At depth 1 without keys,
  readout attention does only prior-level work".

### 2. MAJOR: R2 is used as "prior-level", but by the prereg's own definition it is not, and the depth-1 prediction was rescored
- The prereg amendment calls R1 "the prior-level reference". It calls R2 "prior + query-value comparison, an upper
  bracket". The headline describes R2 as "Bayes on the query values alone with no identities or labels". It then uses
  "at the level of R2" to call depth 1 prior-level, and "75–99% reproduced by R2" to call shared errors prior-level.
  Comparing query values across rows is within-episode evidence, and clustering of the readings is exactly how shared
  errors show up on the query question.
- The preregistered prediction was "depth 1 ≈ prior level (its MSE close to R1's on duplicates and shared errors)".
  Depth 1 (no keys) beats R1 on all three splits: shared 0.0289 vs 0.0467, dup 0.0350 vs 0.0445, singletons 0.0266 vs
  0.0294. As preregistered, the prediction **failed**. The report scores it "confirmed" using a different reference (R2)
  and a test added later (table A).
- **Fix.**
  - Score prediction 7 as failed against R1.
  - Use three tiers: prior (R1), query-value comparison (R2 − R1), and labelled/identity/key evidence (beyond R2).
  - Depth 1 then does tiers 1 and 2 on shared errors (0.21 of its gain is beyond R1) and some of tier 3 on duplicates
    (problem 1).

### 3. MAJOR: "Shared errors are mostly prior-level" is an artifact of anchoring the share on count-all
- On shared_nodup, count-all's MSE (0.113–0.117) is 3–4× R1's, so most of any trained model's "improvement over
  count-all" is plain shrinkage, which every reader learns. The same numbers, read as the share of R's remaining error
  that the model removes:
  - keys, depths 2–8: 85–88% of R2's error removed in context (0.032 → 0.0039–0.0048), 90–92% of R1's;
  - no keys, depths 2–8: 36–44% of R2's error removed (0.0298 → 0.0166–0.0191), 59–64% of R1's;
  - depth 1 without keys: 3% beyond R2, 38% beyond R1.
- The prereg defined the count-all-anchored share, so it should be reported. The headline's verbal reading
  ("mostly prior-level") is not robust to the choice of denominator.
- **Fix:** give both normalisations. A defensible headline: "with keys, deeper models remove almost all the
  shared-error loss that a no-comparison reader leaves; without keys, about 40%."

### 4. MODERATE: "Value equality across labelled questions, not the identity vector" goes beyond what (ii) shows
- (ii) = (i) shows only that, once the values match on all 9 questions, a changed identity vector makes no difference.
  Two things are not tested:
  - whether identity is used at all: there is no identity-only copy (same identity, fresh values);
  - whether the cue is *exact equality* or *residual correlation*: there is no near-copy or same-group sibling (z shared,
    e independent, residual correlation 0.8).
  The deeper no-key models do use residual correlation on shared_nodup (0.11–0.14 beyond R2), so "value equality"
  specifically is not established.
- With keys, (ii) keeps s0's group key (`lab_ii = lab[ar, s0]`), so the (ii) column says nothing about identity in the
  keys arm.
- "The copy shares its source's group key, which is direct in-context evidence" is not isolated either. That needs a
  copy with a new group key, and an independent source with s0's key.
- **Fix:** add those four variants (cheap, same harness). Until then, phrase it as "identity is not needed when values
  match".

### 5. MODERATE: The references leak the row count, i.e. the copy count
- R1 is fitted per n, and n = 6 + number of copies in this world, so R1 knows exactly how many copies an episode has
  (not which rows). R2 also conditions on n.
- In table A, (i), (ii) and (iv) share n, so table A is unaffected. In tables B and C, "prior-level" includes
  episode-specific count information. Treating it as prior-level is defensible, but it should be stated.
- Exact value equality: R2 is scored without exploiting ties (likelihood ratio √5 at distance 0), as stated. R2eq leaks
  by design and is only a bracket. Identity vectors: none of R1, R2 or R2eq reads them. No other leak found.

### 6. MINOR: Statistics and convergence
- **Uncertainty.** Table A's brackets are min..max over 3 seeds on the *same* 600 episodes, not CIs, and they leave out
  episode sampling. My bootstrap se on pulls is 0.001–0.004, so the table A contrasts are not noise-limited. They are
  design-limited (problem 1).
- **Convergence is not shown in tables A–C.** 15/24 runs converged; depth 8 with keys is 1/3. Eight runs have
  best_step = 40000. L8_k0_s2 improved by 0.013 nats in its last quarter (rule: < 0.003); L2_k0_s1/s2 by 0.006.
  In my check, converged and unconverged seeds of an arm agree (for example depth 2 without keys: ρ 1.75 converged vs
  1.70 / 1.78), so convergence does not appear to drive the split results.
- "Depth beyond 2 adds nothing" and every depth comparison without keys rest on arms that are 1/3 converged at
  depths 2 and 8. Add a conv column to tables A–C.

### 7. MINOR: Stale or unsupported wording
- "The adversarial reviewer has not seen section 0": update.
- "Attention is necessarily the carrier, since MLPs act per token": the query token's MLP acts on the attention-pooled
  vector. The non-additive post-pool readout is a plausible carrier of depth 1's in-distribution discount (problem 1),
  so the bullet needs qualifying.
- Bayes "independent 0.176" (table A) averages over all partitions. On singleton episodes the matching number is
  0.121, and the trained models' untied independent pulls (0.105–0.119) sit near it. So without keys, the low (iv) pulls
  in table A (0.054 at depth 1, 0.087–0.093 at depths 2–8) mostly reflect the tie, not discounting of independent sources.

## Which headlines survive as stated

| headline (v4) | verdict |
|---|---|
| Depth 1 without keys is prior-level only (9%); prediction confirmed | **Falls.** In distribution, depth 1 removes 0.44 of a copy (beyond R2's 0.34). Its shared-error MSE is at R2 level, which the prereg calls query-value comparison. The prediction against R1 failed. |
| Depths 2–8 without keys deduplicate in context about halfway (44–49%) | **Falls as stated.** In distribution 0.26–0.32, *less* than depth 1. The 44–49% comes from the tied (iv). |
| …use labelled value equality, not identity; depth beyond 2 adds nothing | **Weakened.** "Identity not needed when values match" survives. "Value equality" vs residual correlation is untested. "Adds nothing beyond 2" survives (0.26 / 0.26 / 0.32), on arms that are 1/3 converged. |
| With keys, strong in-context dedup at every depth (69%, 80–83%) | **Partly.** Depths 2–8 survive at 0.64–0.73 (lower than stated). Depth 1 with keys is 0.49, no better than without keys, so "keys make depth 1 strong" falls. Key vs values not isolated. |
| Shared errors mostly prior-level (60–79% R1, 75–99% R2) | **Numbers right, reading falls.** R2 is not prior-level. Under the residual-error normalisation, deeper keyed models remove 85–88% of R2's error. |
| Beyond-R2 residuals 0.01 / 0.11–0.14 / 0.19 / 0.24–0.25 | **Survives** as numbers (spot-checked), with the R2 label fixed. |
| No model deduplicates fully; best copy pull 0.026 | **Survives** (table A). In distribution the best ρ is 1.23 (depth 4, keys), against Bayes 1. |
| Attention necessarily the carrier; depth-1 readout attention does only prior-level work | **First half survives** (the earlier review's point). **Second half falls** (problem 1). |
| Caveat that table A is the test to trust and C is confounded | **Falls.** C agrees with the in-distribution test, and A has the tie confound. |

**Net:** the prior-level vs in-context split needs redoing on an in-distribution test. In-context deduplication exists
at every depth. Without keys it is largest at depth 1 (about 0.44 of a copy) and smaller at depths 2–8 (about 0.3).
With keys it grows with depth (about 0.5 → 0.65–0.73). What depth 1 uses (the query tie, or pooled identity × value
statistics) is open and cheap to test.
