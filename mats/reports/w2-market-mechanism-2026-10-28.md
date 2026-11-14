# W2 market world: how the W2b models do C4 (in-context source calibration), 2026-10-28

## Summary
- **Answer.** Each quote token works out its own source's reliability from the other quotes of that source. The query token then weights the quotes on the query item by that reliability.
- **Where it happens.** In D4 and Q6 the step that moves calibration evidence sits in block 3. In Q4 it sits in block 2.
- **Causal evidence.** Block 3 has heads that match on source identity. In that block we cut one edge: the query-item quote of A stops attending to A's other quotes, and the same for B. This removes the C4 effect:

  | Net | Shift kept after the cut | C4 closure before → after |
  |---|---|---|
  | D4 | 0.01 | 94% → 28% |
  | Q6 | 0.02 | 84% → 21% |

  Cutting 12 random keys instead keeps the full shift (1.00).
- **Status.** Locate (probes and attention) and intervene (patching and edge ablation) are done for D4 and Q6. Patching and ablation were also run on D6 and Q4, described only.
  - The algorithm below is a hypothesis, for the replacement tests that come later.
  - Replacement tests were not run.
- **Caveats.**
  - There is one training seed per arm.
  - All four networks start from the same seed-0 initialisation (for the blocks they share), so they are not independent replications.
- Cost: CPU only, $0. Prereg: `work/w2-mech/PREREG.md`, with amendments 1 and 2, all hashed in `PREREG.sha256` before each run.

## Setup
- **Networks.** D4_s0 and Q6_s0 `best.pt` (the best D arm and the best Q arm on C4 at N=96). Extras, read descriptively only: D6_s0 and Q4_s0.
- **Untrained control.** The same architecture right after `torch.manual_seed(0)`.
- **Probe set.** 2000 natural W2 episodes at N=96, policy `dir`. Episodes 0-1499 are used for training the probes, 1500-1999 are held out.
- **C4 set.** The pilot evalset: 200 frequency-matched pairs at N=96.
  - The two members of a pair differ only in the y values of the 24 other-item quotes of the two unknown-noise sources A and B. This was checked, and the token positions line up across the two members.
- **Exact Bayes, extended** (`mech.bayes_ext`). It gives per-source posteriors (bias mean, P(σ = high), effective precision) and the linear Bayes weights β_t on each quote, such that Σ β_t y_t equals the Bayes mean.
  - Checked against `predictive()` to within 1e-6. Σβy matches the Bayes mean to 2e-15.
  - The re-implemented forward pass matches `harness.predict` to within 3e-6.

## 1. Locate: ridge probes
Held-out R², with the untrained network in brackets. The shuffled baseline is about 0 everywhere. A probe counts as positive if R² ≥ 0.5 and it beats max(untrained, shuffled) by ≥ 0.10. Sites: `L1.a` is after the block-1 attention, `L1` is after the block-1 MLP.

| Target (token type) | D4: first positive site | D4: final R² | Q6: first positive site | Q6: final R² |
|---|---|---|---|---|
| item posterior mean E[a·v] (all quotes) | L1.a 0.82 (0.64), strong | 0.91 | L1.a 0.82 (0.64), strong | 0.92 |
| quote residual y − E[a·v] (SPOT/BASKET) | L1.a 0.84 (0.56), strong | 0.93 | L1.a 0.83 (0.56), strong | 0.94 |
| residual² (SPOT/BASKET) | L1 0.74 (0.14) | 0.75 | L1 0.64 (0.14) | 0.70 |
| source bias posterior mean (SPOT/BASKET) | L2.a 0.60 (0.21) | 0.70 | L2.a 0.50 (0.21) | 0.66 |
| P(σ_s = high) (unknown-noise sources' quotes) | L3.a 0.65 (0.00); L2 = 0.46 | 0.71 | L3.a 0.54 (0.00); L2 = 0.19 | 0.65 |
| log effective precision (unknown-noise quotes) | L3.a 0.65 (0.00) | 0.72 | L3.a 0.53 (0.00) | 0.65 |
| Bayes predictive mean (query token) | L1.a 0.50; strong at L4.a | 0.87 | L1; strong at L4 | 0.85 |

- **Probes moved to C4 tokens.** The P(σ = high) probe was fitted on the natural set. Applied to the query-item quotes Aq and Bq of the C4 set, it gives:

  | | Before block 3 (L2) | After block 3 (L3) |
  |---|---|---|
  | D4, r with Bayes | 0.57 | 0.95 |
  | Q6, r with Bayes | 0.07 | 0.88 |

  - At L3 the member-0 − member-1 difference in decoded P(high) for A is −0.93 for D4 and −0.81 for Q6. Bayes gives −1.00.
  - Residual² on A's and B's other quotes is decodable from L1 onwards (r 0.85-0.92).
- The noise-level probes are positive but not strong on natural episodes (R² 0.65-0.72).

## 2. Attention
Untrained values are 0.97-1.01 for every ratio below.

- **Overlap graph** (a quote attending to other sources' quotes on the same items). Pooled per-key ratio, overlapping keys against non-overlapping keys:
  - D4: block 1 head 2 (4.0); block 3 head 2 (14.0).
  - Q6: block 1 head 1 (5.6); block 3 head 2 (8.9); block 4 head 2 (5.4).
- **Source-identity matching** (a quote attending to the other quotes of its own source). Pooled per-key ratio, same source against other sources:
  - D4: block 2 heads 1 and 3 (11.3, 3.2); block 3 heads 0 and 3 (6.5, 8.4).
  - Q6: block 2 head 2 (4.3); block 3 heads 0 and 3 (8.3, 7.3).
  - These are the candidate calibration circuit; the causal test is in section 3.
  - On C4, from Aq, the per-key attention ratio to A's other quotes in block 3 is 100-5000 (Bq to B's quotes: the same).
- **Query attention against the Bayes weights |β_t|** (Pearson r over episode-token pairs):

  | Net | Block | r, head mean | Best single head |
  |---|---|---|---|
  | D4 | 4 | 0.57 | 0.47 |
  | Q6 | 4 | 0.53 | 0.43 |
  | Q6 | 5 | 0.50 | not reported |

  - Untrained: |r| ≤ 0.04. This passes the preregistered rule.
- **C4 paired attention.** In the read block, the query token's attention log-ratio between Aq and Bq shifts toward the more reliable source on 97-100% of pairs. Untrained: about 50%.
  - D4: block 4 head 3, which puts 0.89 of its mass on the two quotes.
  - Q6: block 4 heads 1-3, and block 5 heads 0, 2 and 3.

## 3. Intervene (C4 pairs; median flip fraction, 0 = no change, 1 = the other member's answer)
Residual patching from the other member, after block l:

| Net | Evidence leaves A/B's other quotes (EV patch: before → after) | Calibration arrives on Aq/Bq (QQ patch: before → after) | Query token reads (QT patch: before → after) |
|---|---|---|---|
| D4 | 0.93 → 0.01 across block 3 | 0.07 → 0.94 across block 3 | 0.06 → 1.00 across block 4 |
| Q6 | 0.99 → 0.02 across block 3 | 0.02 → 0.97 across block 3 | 0.00 → 0.79 → 1.04 across blocks 4 and 5 |
| D6 (extra) | 0.96 → 0.03 across block 3 | 0.04 → 0.97 across block 3 | 0.00 → 0.33 → 1.04 across blocks 4 and 5 |
| Q4 (extra) | 1.00 → 0.03 across block 2 | 0.00 → 0.97 across block 2 | 0.00 → 1.01 across block 3 |

- Patching Aq alone or Bq alone after the transfer block gives about 0.45 each. Each query-item quote carries its own source's reliability.

Edge ablation. In the transfer block, Aq's attention to A's other quotes and Bq's attention to B's other quotes is set to zero. Each cell is shift kept / C4 closure.

| Net | Edge cut in transfer block | Control (12 random keys) | Same edge in the block before | Overlap edge cut, block 1 |
|---|---|---|---|---|
| D4 | 0.01 / 28% | 1.00 / 94% | b2: 0.36 / 32% | 0.74 / 52% (control 0.99) |
| Q6 | 0.02 / 21% | 1.01 / 84% | b2: 0.40 / 34% | 0.70 / 45% (control 0.96) |
| D6 | 0.03 / 25% | 1.00 / 92% | b2: 0.58 / 49% | not run |
| Q4 | b2: 0.00 / 17% | 1.00 / 80% | b3 (after transfer): 0.97 | not run |

Notes on the table:
- The overlap edge is cut on the 24 other-item quotes, removing their attention to other sources' quotes on the same item. It is partly causal: it does not meet the ≤ 0.3 threshold, but closure drops by about 40 points. A likely reason is a fallback signal: the raw spread of a source's quotes also reveals its noise level, without any consensus.
- Cutting the same edge one block earlier (block 2) has a partial effect, even though the member-specific content of Aq at block 2 does not matter (QQ patch about 0.05). Block 2 probably supplies something the block-3 matching needs that is the same in both members, for example source-pooled identity features. This is open.

Read-out mechanism. The query token's attention to Aq and Bq is set to their mean in the read block(s). The table gives the shift that is kept.

| Net | Read block(s) equalised | Shift kept |
|---|---|---|
| Q6 | b4 | 0.51 |
| Q6 | b5 | 0.75 |
| Q6 | b4 + b5 | 0.05 |
| D6 | b4 + b5 | 0.14 |
| Q4 | b3 | 0.40 |
| D4 | b4 | 0.65 |
| D4 | b3 + b4 | 0.64 |

In the Q nets and D6, the attention pattern carries the reliability weighting. In D4, about 35% of the weighting is in the attention pattern and the rest is carried in the content of the Aq and Bq values.

## 4. Hypothesis of the algorithm (for replacement tests; not yet tested by replacement)
1. **Block 1, item consensus and residuals.** Overlap heads let each quote gather other sources' quotes on its items. This gives a consensus E[a·v]. The block-1 MLP forms the quote's residual against that consensus and the residual's square.
2. **Block 2, start of per-source pooling.** Source-matching heads pool signed residuals across a source's quotes into a bias estimate (the bias probe turns positive at L2.a). In D4, some noise information is pooled here too.
3. **Block 3, noise level** (block 2 in Q4). Source-identity heads let every quote token pool residual² from the other quotes of its own source. The result is the source's noise-level posterior and effective precision, held on each quote token. The quotes on the query item therefore carry their own reliability. This edge is causal.
   - This runs in parallel for every quote, because context tokens never see the query.
4. **Read block(s)** (block 4; blocks 4-5 in Q6 and D6; block 3 in Q4). The query token attends to the quotes on the query item. It weights them by reliability, mostly through the attention pattern (D4 partly through the values), and outputs a bias-corrected, precision-weighted mean.

What this does not cover:
- Why C4 fails at N=512. To test: whether source-matching attention gets diluted as S grows.
- C7. The mechanism found is keyed on source identity and item overlap. No event-id or provenance-matching head was looked for. That would be the first check for the C7 failure.

## Deviations from the prereg
- **Shuffled baseline.** Targets were permuted across training-sample rows, not as episode blocks, because token counts per episode vary.
- **Attention ratios.** The per-token "ratio of means" statistic is dominated by near-zero denominators (values up to 1e8). The pooled ratio is reported as the primary number. Both are in the JSON.
- **D6 and Q4.** These were added as descriptive extras with the same scripts. For Q4, the block-4 equalisation in `intervene.py` is the wrong block; the correct block-3 run is in `intervene3_Q4_s0.json`.
- **Untrained nets.** They show no C4 shift at all (|Δm| < 0.05 on every pair), so patching on them is undefined.

## Files
- Code: `work/w2-mech/` (`mech.py`, `prep.py`, `locate.py`, `c4.py`, `intervene.py`, `intervene2.py`, `intervene_lib.py`, `show_locate.py`).
- Results: `work/w2-mech/out/` (`locate_*.json`, `c4_*.json`, `intervene*_*.json`, logs).
- W2c files and VMs were not touched.
