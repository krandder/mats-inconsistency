# Untangled (tree) nets against dense transformers on the PFN worlds (29 October 2026)

Experimental lane for Kelvin's question: is tangledness necessary, or can a "scale-free" structure (the same simple
combination at every scale, with dimensions allowed to grow with scale) learn as well? Work dir
`work/scale-free-exp/`: pre-registration `PREREG.md` (amendments 1-11, each before the results it affects; hashes in
`prereg_hashes.txt`), GPU log `GPU-LOG.md`, GPT-6 reviews `results/review1.md` and `results/review2.md`. Every number below is a test KL to
exact Bayes in nats unless stated, after 10.24M training episodes per net, 2-3 seeds.

## Summary

**In short.** On these worlds, combining evidence does not need tangling. Suppose a net is told which tokens belong to
which source. Then a tree whose sources interact only through a sum, plus one broadcast round, matches the dense
transformer at equal data, and with 4x the data reaches KL 0.001, about 3x below the dense net at the same budget (0.004; N = 12). What dense attention is needed for,
and is slow to learn, is routing: finding which tokens belong together, and comparing sources to find hidden shared
errors.

1. **Within our training budget, the dense token transformer fails at routing, not at combining.**
   - Setup: in the phase-B token format, each source's five tokens must be grouped by matching random per-episode
     ids. The net first sits on a plateau whose predictions are those of history-blind Bayes, which uses the
     readings but not each source's track record.
   - The plateau gets longer with N. It ends after about 1.5M episodes at N = 3 and 3-4M at N = 6. At N = 12, two of
     three seeds escaped by 8-10M. At N = 24 none of three escaped within 10.24M: their predictions are 0.005-0.006
     nats from history-blind Bayes and 0.21 from Bayes.
   - Given the routing graph as an attention mask, the same transformer reaches 0.013 at N = 12, against 0.194 when it
     must learn the routing; learning the routing from the ids alone is just as stuck (0.195).
   - This is World FA's binding failure, reproduced inside the PFN world. It is a fixed-budget result: longer
     training or other recipes might escape.
2. **Given the routing, untangled nets come close to the dense net, and the two-pass trees we tested match it.**
   - On W0, Bayes adds per-source evidence.
     - A one-pass tree ends at 1.4-2.2x the KL of a dense transformer that gets one token per source (0.007-0.033
       vs 0.005-0.015 for N = 3-64). In it, each source sends m numbers, the numbers are summed, and a small network
       reads the sum.
     - More numbers per source helped little: from m = 4 to 32 the KL falls by at most 0.002-0.005 at N = 3-64.
   - A two-pass tree matches or beats the dense net up to N = 64 (N = 3: 0.0033 vs 0.0049; N = 24: 0.007 vs 0.010;
     N = 64: 0.013 vs 0.015).
     - Its first pass sends 3 numbers per source and gives a pilot estimate that is sent back to every source.
     - The second pass sends 2-8 numbers relative to it.
   - The advantage survives the controls added after the review:
     - One-pass trees with slightly more parameters (22-23k vs 20-21k) and as many numbers per source (7 or 11) get
       0.0057-0.0061 at N = 3, 0.016-0.017 at 24 and 0.027 at 64, against the two-pass trees' 0.0033-0.0041,
       0.0066-0.0078 and 0.013-0.015.
     - Dropping the two-pass tree's auxiliary pilot loss does not hurt (0.0055-0.0059 at N = 24, 0.012-0.013 at 64).
     - What is not separated from the second round is that pass 2 sees its reading standardised by the pilot.
       That is the design itself: "write relative to the consensus".
   - So, at this budget, a second round rather than more message dimensions closes the gap, as the theory note
     predicted. It helps at every N we tried, including N = 3, not only from N of about 64. At N = 256 the gap narrows
     (0.061-0.082 vs 0.074-0.087).
   - With more data the picture sharpens. This was an exploratory check at N = 12 with 4x the episodes (41M),
     designed after the main results:
     - the one-pass tree reaches 0.0051-0.0052 (2 seeds);
     - the dense set net reaches 0.0039 (1 seed);
     - the two-pass tree reaches 0.0011-0.0012 (2 seeds).

     So part of the one-pass tree's gap was learning speed: it is 1.8x the dense net's KL at 1x and 1.3x at 4x. But the
     second round does more than speed learning up: at 4x the two-pass tree's KL is about 3x below the dense net's.
     Budget and cooldown length changed together, and the set net has one seed.
3. **Shared errors.**
   - With known groups (W1), a hierarchy told the grouping stays at 0.02-0.05 at every N. The dense set net, which
     must match group ids, goes from 0.010 to 0.068 and is behind the hierarchy from N = 16.
   - With hidden pairs (W2), every sum-pooled tree we trained (up to 128 numbers per source) recovered only 14-29% of
     what modelling the pairs is worth; dense attention across sources recovered 69-77%. Comparing sources is where
     dense attention clearly did better here. That does not prove a tree cannot do it, only that ours did not
     learn to.
4. **The dense nets that learned the task look like a tree inside.** Both 2-layer token nets at N = 3 and the best
   3-layer one show the same pattern, which is consistent with source-local processing followed by a
   precision-weighted merge:
   - layer 1 routes by labels (label R^2 0.75-0.78);
   - the last layer's weights on each source's reading track its log precision (r 0.86-0.95 in the main head);
   - after layer 1, a source's query token barely depends on the other sources (2.5-3%);
   - in some nets, one head's attention to the query token correlates with the predictive variance (R^2 0.61, 0.72).

   Coordinator note (00:34Z, 30 October): the first two readings are correlational. On the phase-B nets, the theory
   lane's pre-registered causal test of the same reading failed ("layer 1 routes, layer 2 weights", P-M3b, 4 and 6 of
   9 nets; `scale-free-theory-2026-10-29.md`, Addendum). Both layers were very sensitive to label scrambling (0.4-4.4
   nats) and to episode-specific attention (0.2-1.05 nats). Read "routes by labels" and "track log precision" as
   descriptions of attention and weights, not as the mechanism. The cut results below are causal.

   Every measured net that learned the task combines sources nearly additively: its cross-source interaction share is
   0.002-0.019, where Bayes has exactly 0. Bottlenecks at cuts on the N = 3 nets agree:
   - Removing every attention edge between sources costs nothing (-0.002 nats, fine-tuned).
   - Rank-2 keys and values on every edge leaving a source keep the frozen net within KL about 0.005 of itself (rank 1: 0.016-0.040).
   - What does not compress is the path out of a source's labelled tokens (+0.05 to +0.09 at rank 1), which also carries the
     id matching used for routing.
   - On the nets stuck at history-blind, cutting the cross-source edges rescues them (KL 0.20 to 0.02-0.03 in 2.56M
     episodes): imposing the routing lets them bind.
5. **Deduction works but did not pay at this size.**
   - Setup: for a tree with 2-number messages, the law of its output given five shared variables is a 12-fold
     convolution of one source's message law. Computed numerically, it matches an 8M-episode sampling reference
     from the mean down to tail probabilities of about 1e-6.
   - It needed a post-hoc fix to the integration rule for y* before it did.
   - Sampling this tiny net is so cheap that deduction was not cheaper at equal accuracy.
   - The best tree (16-number messages) is out of reach for exact deduction.
6. **World FA (E4).** Within 2M episodes, no arm recovered any of the value of the traders' history (all at -0.04 to -0.02 of it), even with the trader grouping given as an attention mask. A likely but untested reason: comparing two traders' histories also requires pairing their rounds, which carry only random ids, and that routing was not given. World FA's earlier success came from putting each trader's errors in fixed positions.

## Setup

**Worlds** (`sfworld.py`; exact Bayes on an adaptive grid, tested against tiny-pfn's enumeration and brute-force
enumeration to 1e-6): y* ~ N(0, 1), 4 labelled questions and 1 query, each source's error variance drawn from
{0.1, 0.5, 2}.
- W0: N independent sources (N = 3 to 256). Bayes adds per-source log-likelihoods of y*.
- W1: known groups of 4 sources share an error factor (rho 0.5); group ids shown to the dense nets. Bayes adds
  per-group terms. Coupling-blind Bayes (treating sources as independent) loses 0.35-0.40.
- W2: hidden pairs (rho 0.9): Bayes must find which sources share an error. Coupling-blind Bayes loses 0.28-0.37.

**Nets** (mixture-of-4-Gaussians head as in phase B):
- tree (10-14k parameters): per-source MLP on the source's 9 numbers (4 readings, 4 labels, query reading) -> m numbers
  -> sum over sources -> MLP readout. No cross-source interaction.
- two-pass tree (about 20k): pass 1 sends 3 numbers per source; their sum gives a pilot mean and log-sd (with an
  auxiliary NLL loss); pass 2's per-source MLP also sees the pilot and its reading standardised by the pilot.
- hier (W1): per source -> sum inside the known group (or one attention block inside the group) -> group message
  -> sum over groups.
- set (18k): dense transformer (2 layers, width 32) over one token per source (the same 9 numbers) plus a query
  token; "set, no cross-source" lets source tokens attend only to themselves.
- tok (19k): the phase-B token transformer, one token per (source, question) with random 16-number ids, no
  positional encoding; it must group a source's tokens by matching ids.

**Training**: 10.24M fresh episodes per net, batches of 1024 for 10,000 steps (lr 1e-3, warmup 250, cooldown over
the last 2,500), TF32 matmuls in training, FP32 evaluation. Test: 4096 fixed episodes per world.
The E5 trees, trained on CPU, used the original recipe: batches of 256 for 40,000 steps, the same number of episodes.

## E1: tree vs dense

**W0, independent sources.** Test KL, mean [min, max] over seeds.

| N | tree1 | tree2 | tree4 | tree8 | tree16 | tree32 | set | setnc | tok | two-pass 3+2 / 3+4 / 3+8 | one-pass h96 m 7 / 11 | history-blind Bayes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.2234 [0.2222, 0.2253] | 0.0195 [0.0194, 0.0196] | 0.0078 [0.0075, 0.0082] | 0.0074 [0.0071, 0.0077] | 0.0072 [0.0070, 0.0073] | 0.0071 [0.0066, 0.0076] | 0.0049 [0.0048, 0.0049] | 0.0064 [0.0061, 0.0068] | 0.0243 [0.0116, 0.0494] | 0.0041 / 0.0036 / 0.0033 | 0.0061 / 0.0057 | 0.174 |
| 6 | 0.2954 [0.2949, 0.2956] | 0.0334 [0.0327, 0.0345] | 0.0127 [0.0124, 0.0129] | 0.0126 [0.0124, 0.0129] | 0.0113 [0.0103, 0.0125] | 0.0113 [0.0105, 0.0128] | 0.0064 [0.0063, 0.0066] | 0.0107 [0.0106, 0.0108] | 0.0268 [0.0187, 0.0356] | - / - / - | - / - | 0.212 |
| 12 | 0.3361 [0.3352, 0.3376] | 0.0397 [0.0387, 0.0405] | 0.0177 [0.0171, 0.0188] | 0.0156 [0.0143, 0.0173] | 0.0154 [0.0147, 0.0166] | 0.0140 [0.0122, 0.0157] | 0.0076 [0.0069, 0.0083] | 0.0138 [0.0128, 0.0146] | 0.1157 [0.0527, 0.2005] | - / - / - | - / - | 0.211 |
| 24 | 0.3552 [0.3535, 0.3577] | 0.0448 [0.0438, 0.0460] | 0.0232 [0.0225, 0.0237] | 0.0206 [0.0186, 0.0234] | 0.0219 [0.0210, 0.0237] | 0.0205 [0.0174, 0.0234] | 0.0102 [0.0092, 0.0113] | 0.0140 [0.0135, 0.0143] | 0.2060 [0.2050, 0.2067] | 0.0066 / 0.0078 / 0.0073 | 0.0174 / 0.0163 | 0.212 |
| 64 | - | - | 0.0374 [0.0331, 0.0417] | 0.0345 [0.0316, 0.0374] | 0.0341 [0.0313, 0.0370] | 0.0332 [0.0303, 0.0361] | 0.0154 [0.0152, 0.0156] | - | - | 0.0142 / 0.0153 / 0.0126 | 0.0272 / 0.0268 | 0.187 |
| 256 | - | - | 0.1243 [0.1240, 0.1247] | 0.1135 [0.0972, 0.1297] | 0.0972 [0.0871, 0.1073] | 0.1016 [0.0917, 0.1116] | - | - | - | 0.0817 / 0.0695 / 0.0612 | 0.0865 / 0.0740 | 0.211 |

(tree m = 1: 0.22-0.36. Parameters: trees 10-14k, two-pass trees about 20-21k, set 18k, tok 19k.)

Steps (of 10,000 at batch 1024) to reach val gap <= 0.05: tree (m 16) 2-3k (N = 3), 3k (6), 3-4k (12), 8k (24);
set 0.75-1k, 0.75-1.5k, 1-2k, 2-4k; tok 4-5k and once never (N = 3), 5-10k (6), 10k once / never (12), never (24).
No loss spikes in any run.

What W0 shows:
- **The token net's failure is a plateau, and the plateau lengthens with N.**
  - Before escaping, its predictions are those of history-blind Bayes. Measured directly at N = 24, they are 0.005-0.006
    nats from history-blind Bayes and 0.205-0.207 from Bayes; at N = 12 the stuck seed is 0.012 from blind and 0.200
    from Bayes (`blind_cmp.py`).
  - Escape times: about 1.5k steps (N = 3), 3-4k (6), 8-10k for two of three seeds (12), and more than 10k for all
    three (24).
  - With routing given (set net), the same budget gives 0.005-0.015 at every N.
- **With routing given, the dense net beats a one-pass tree by 1.4-2.2x, and message size is not the reason.**
  The one-pass tree's KL is flat from m = 4 to 32, while the Bayes audit says a linear additive code would need about
  6-16 numbers (`audit/rank_post.json`).
- **The two-pass trees match or beat the dense net up to N = 64, and beat capacity-matched one-pass trees.** The
  matched trees have slightly more parameters (22-23k vs 20-21k) and as many numbers per source as the two-pass tree in total
  (7 or 11): column "one-pass h96".
  - Without its auxiliary pilot loss, the two-pass tree is no worse (arm means 0.0055-0.0059 at N = 24 and 0.012-0.013 at 64)
    (Amendment 10).
  - What remains confounded with the second round is pass 2's reading standardised by the pilot, which is part of
    the design.
- **Exploratory 4x-budget check at N = 12** (Amendment 11; 40.96M episodes; trees seeds 0 and 1, set net seed 0; written after the main
  results):
  - one-pass tree (m 32): 0.0051, 0.0052, against 0.0140 at 1x;
  - two-pass tree (3 + 8): 0.0012, 0.0011 (it was not run at N = 12 with 1x; at 1x it reached 0.0033 at N = 3 and 0.0073 at N = 24);
  - set net: 0.0039 (one seed), against 0.0076 at 1x.
  - Part of the one-pass tree's 1x gap is learning speed: 1.8x the set net's KL at 1x, 1.3x at 4x. Seed 0's validation gap barely moved during the constant-rate
    phase (0.019 at step 30k) and fell during the final cooldown (0.008 at 35k).
- **N = 256 is hard for every architecture at this budget** (0.06-0.12).

**W1, known groups of 4 sharing an error.**

| N | coupling-blind | tree32 | hiersum_m4 | hiersum_m16 | hierattn_m4 | hierattn_m16 | setg | tokg |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.349 | 0.030 [0.029, 0.032] | 0.031 [0.031, 0.031] | 0.030 [0.030, 0.030] | 0.025 [0.025, 0.025] | 0.022 [0.021, 0.024] | 0.010 [0.009, 0.011] | 0.061 [0.044, 0.079] |
| 8 | 0.363 | 0.104 [0.103, 0.106] | 0.040 [0.040, 0.040] | 0.039 [0.039, 0.039] | 0.035 [0.034, 0.036] | 0.033 [0.032, 0.034] | 0.032 [0.031, 0.033] | 0.209 [0.133, 0.284] |
| 16 | 0.395 | 0.142 [0.141, 0.143] | 0.047 [0.045, 0.048] | 0.046 [0.044, 0.048] | 0.038 [0.037, 0.039] | 0.039 [0.038, 0.039] | 0.050 [0.049, 0.051] | 0.279 [0.278, 0.280] |
| 24 | 0.376 | 0.151 [0.149, 0.152] | 0.049 [0.048, 0.050] | 0.048 [0.046, 0.050] | 0.042 [0.039, 0.046] | 0.042 [0.041, 0.043] | 0.068 [0.065, 0.072] | 0.272 [0.271, 0.273] |

The hierarchy told the grouping stays near 0.02-0.05; the dense set net with group ids degrades with N. The flat,
group-blind tree recovers much more than I predicted (P6 failed): group size and coupling are the same in every
episode, so a group-blind net can learn the average loss of independence without knowing who is grouped with whom.

**W2, hidden pairs.** KL (range of the recovered share of the coupling value).

| N | coupling-blind | tree8 | tree32 | tree128 | set | tok |
|---|---|---|---|---|---|---|
| 4 | 0.279 | 0.206 (26%-26%) | 0.204 (27%-27%) | 0.201 (28%-29%) | 0.065 (77%-77%) | 0.205 (26%-27%) |
| 6 | 0.309 | 0.264 (14%-15%) | 0.263 (15%-15%) | 0.261 (15%-16%) | 0.092 (70%-70%) | 0.281 (5%-13%) |
| 8 | 0.366 | 0.298 (18%-19%) | 0.297 (18%-19%) | 0.296 (19%-19%) | 0.113 (69%-69%) | 0.323 (9%-15%) |

The sum-pooled trees do not find the pairs at any message size we tried (8 to 128); the dense set net recovers about
70% of the value; the token net, which must also learn routing, recovers 5-27%.

## E2 and the theory note's measurements: what the dense nets compute across cuts

**Measurements M1-M6** (`m1prof.py`, `tokan.py`; W0; 2,000-3,000 fresh episodes per net). M1's share is the
interaction variance divided by the sum of the two main-effect variances (not a fraction of total variance).

| measurement | 2-layer tok, N = 3 (KL 0.012) | 3-layer tok (E3 A), N = 3 (KL 0.009) | 2-layer tok N = 12 stuck (KL 0.20) | 2-layer tok N = 12, partly escaped (KL 0.09) | reference |
|---|---|---|---|---|---|
| context order permuted per episode: max output change | 1.4e-6, 1.7e-6 | 1.9e-6 | 1.4e-6 | 1.9e-6 | 0 (no positional encoding) |
| M2 layer 1: routing classes among context rows (TV <= 0.1) | 4.8-6.7 of 15 | 4.1-5.5 of 15 | 2.2-9.6 of 60 | 27-28 of 60 | - |
| M2 layer 1: label R^2 of the pair-specific attention (held-out) | 0.75-0.77 | 0.77-0.78 | 0.01 | 0.24-0.27 | - |
| M2 last layer: per-key (weighting) share | 0.73-0.87 | 0.83-0.93 | 0.84-0.89 | 0.93-0.94 | - |
| M2: query's log mass on each source's reading vs log E[1/v given residuals], r (both heads) | 0.92, 0.86; 0.95, 0.50 | 0.94, 0.77 | 0.40, 0.43 | 0.71, 0.68 | GLS: 1 |
| M5: query self-attention vs predictive variance, R^2 (best head; Bayes's variance) | 0.61 (0.58); 0.72 (0.70) | 0.33 (0.33) | 0.03 | 0.30 | sink identity |
| M6: change in a source's query token after layer 1 from other sources' errors / from its own | 0.025, 0.029 | 0.031 | 0.32 | 0.32 | one round: 0 |
| M1 cross-source interaction share, net (net minus Bayes) | 0.019, 0.017 (0.61, 0.58) | - | 0.013 (0.03) | 0.019 (0.08) | Bayes: 0 exactly |
| M1 within-source share, net (Bayes) | 0.043 (0.039) | - | 0.0008 (0.011) | 0.008 (0.011) | - |

M1 for the other nets, at the same cuts:
- One-pass trees, N = 12: cross-source share 0.002-0.003; within-source 0.011-0.016 (Bayes 0.011).
- Set nets, N = 3: cross 0.010-0.014 (deviation 0.62-0.82); within 0.041-0.043 (Bayes 0.039).
- Set nets, N = 12: cross 0.003-0.007 (deviation 0.12-0.17); within 0.011.

M3 (2-layer N = 3 nets, after a per-episode permutation). These are KL distances from the original net after
replacing one layer's attention:

| replacement | layer 1 | layer 2 |
|---|---|---|
| per-episode pair-class mean of the logits | 0.26-0.30 | 0.14-0.15 |
| the batch-averaged attention probabilities | 0.44-0.72 | 0.61-0.67 |
| another episode's attention | 0.94-1.15 | 0.77-1.60 |

Two caveats on M2 and M3:
- The label regression uses 7 pair indicators (same source, same question, query-reading endpoints, self and two
  interactions), not a saturated pair-class model.
- The batch average is over probabilities, not logits.

What the measurements suggest:
- The N = 3 nets that learned the task route by labels in layer 1, then merge by a weighting that tracks precision.
- In some nets, one head's attention correlates with the predictive variance, as in the theory note's "prior as sink" identity.
- Their first layer is close to source-local, and they combine sources nearly additively.
- Keeping label routing but removing content-dependent weighting costs 0.14-0.30 nats, so content weighting
  matters within the routing classes.
- These are correlations plus three crude interventions, not a causal decomposition.
- The stuck net shows none of the routing signatures. Its within-source interaction is a tenth of Bayes's, as
  expected if it never applies a per-source nonlinearity to the labelled errors.

**E2: k-dimensional bottlenecks at cuts, whole net fine-tuned** (2.56M episodes). Each entry is KL(cut) minus
KL(control), where the control is the same fine-tune without a cut. The cut applies in every layer:
- `xsrc`: edges between different sources' tokens.
- `toq`: edges into the query token.
- `labout`: edges out of a source's labelled-question tokens to any other token.

k is the rank of the keys and values per head on cut edges; k = 0 removes the edges.

- w0 N=3 s0: base 0.0120, control 0.0108; xsrc0 -0.0020, xsrc1 -0.0021, xsrc2 -0.0021, xsrc4 -0.0021, toq1 +0.0069, toq2 +0.0008, toq4 +0.0003, toq8 +0.0001, labout0 +0.1739, labout1 +0.0949, labout2 +0.0686, labout4 +0.0178
- w0 N=3 s1: base 0.0116, control 0.0105; xsrc0 -0.0030, xsrc1 -0.0030, xsrc2 -0.0030, xsrc4 -0.0030, toq1 +0.0086, toq2 +0.0010, toq4 +0.0002, toq8 +0.0000, labout0 +0.1728, labout1 +0.0541, labout2 +0.0363, labout4 +0.0167
- w0 N=12 s0: base 0.2005, control 0.1989; xsrc0 -0.1739, xsrc1 -0.1748, xsrc2 -0.1768, xsrc4 -0.1650, toq1 +0.0043, toq2 +0.0008, toq4 +0.0001, toq8 -0.0001, labout0 +0.0073, labout1 -0.0002, labout2 -0.0002, labout4 -0.0003
- w0 N=12 s1: base 0.0938, control 0.0786; xsrc0 -0.0586, xsrc1 -0.0600, xsrc2 -0.0606, xsrc4 -0.0611, toq1 +0.0101, toq2 +0.0038, toq4 +0.0012, toq8 -0.0000, labout0 +0.1295, labout1 +0.1207, labout2 +0.1025, labout4 +0.0139
- w0 N=24 s0: base 0.2067, control 0.2062; xsrc0 -0.1792, xsrc1 -0.1831, xsrc2 -0.1772, xsrc4 -0.0021, toq1 +0.0024, toq2 +0.0001, toq4 -0.0000, toq8 -0.0001, labout0 +0.0015, labout1 +0.0003, labout2 -0.0001, labout4 -0.0001
- w0 N=24 s1: base 0.2062, control 0.2053; xsrc0 -0.1837, xsrc1 -, xsrc2 -, xsrc4 -, toq1 -, toq2 -, toq4 -, toq8 -, labout0 +0.0021, labout1 +0.0025, labout2 -, labout4 -
- w1 N=8 s0: base 0.2844, control 0.2852; xsrc0 -0.1745, xsrc2 -0.1673, xgrp0 -, xgrp2 -

Reading E2:
- **Only the N = 3 base nets had fully learned the task, so only they measure what crosses each cut.**
  - Between sources, nothing is needed: removing every cross-source edge slightly helps (-0.002 to -0.003).
  - Into the query token, 2 numbers per head per token suffice (+0.001); 1 costs +0.007 to +0.009.
  - Out of a source's labelled tokens, the cut is expensive: +0.17 removed, +0.054 and +0.095 at k = 1 (two seeds), +0.018 at k = 4. Bayes
    needs only one number per source here (its sum of squared errors). But this cut also squeezes the keys that
    other tokens use to find the source by its id, so it measures routing as well as content.
- **On the base nets stuck at or near history-blind (W0 N = 12 seed 0 and N = 24; W1 N = 8 seed 0), cutting the cross-source edges rescues them.** In W0 the KL
  falls from about 0.20 to 0.02-0.03 within 2.56M episodes (at rank k <= 2; at k = 4 the N = 24 seed-0 net was not rescued), and in W1 from 0.285 to 0.11. Removing those edges imposes the routing, and the net then
  binds quickly.

**M4: the same cut on every edge out of a source, base net frozen, only the bottleneck maps trained** (2.56M episodes,
lr 3e-3). Each entry is KL to Bayes, then KL(base net || bottlenecked net).

- N=3 s0 base KL 0.0120: k=1: 0.0240 / 0.0160; k=2: 0.0132 / 0.0051; k=4: 0.0104 / 0.0026; k=8: 0.0103 / 0.0026
- N=3 s1 base KL 0.0116: k=1: 0.0457 / 0.0396; k=2: 0.0110 / 0.0051; k=4: 0.0094 / 0.0038; k=8: 0.0091 / 0.0035
- N=12 s0 base KL 0.2005: k=1: 0.1532 / 0.1429; k=2: 0.0823 / 0.1657; k=4: 0.0689 / 0.1870; k=8: 0.0636 / 0.1980
- N=12 s1 base KL 0.0938: k=1: 0.0774 / 0.0715; k=2: 0.0516 / 0.0590; k=4: 0.0473 / 0.0552; k=8: 0.0448 / 0.0536
- N=24 s0 base KL 0.2067: k=1: 0.2279 / 0.0304; k=2: 0.1481 / 0.0631; k=4: 0.2026 / 0.0026; k=8: 0.1987 / 0.0045
- N=24 s1 base KL 0.2062: k=1: 0.1315 / 0.1167; k=2: 0.0952 / 0.1350; k=4: 0.0821 / 0.1592; k=8: 0.0632 / 0.1961

Reading M4:
- **N = 3 net that learned the task.** Rank-2 keys and values on every edge leaving a source keep the frozen net
  within KL about 0.005 of itself; rank 1 costs 0.016-0.040, and rank 4-8 is within 0.003-0.004.
  - With rank >= 4 the trained maps even lower its KL to Bayes slightly (0.012 to 0.010).
  - Our k counts per head, token and layer: at k = 2, about 40 numbers per source per layer cross the cut, in keys
    and values. So the theory note's "2-4 dimensions per source" is not directly comparable. What we can say is that
    the information leaving a source is low-rank in every head.
- **Stuck nets (N = 12, 24).** The maps, trained on frozen nets, mostly lower KL to Bayes from about 0.20 to 0.06-0.15 (not for N = 24 seed 0 at ranks 1, 4 and 8: 0.20-0.23). They
  change which tokens attend to which, imposing some routing, so here M4 does not measure message size.

## E3: routing vs content

The same 3-layer token transformer (width 32, 2 heads), W0, 10.24M episodes. Only the attention changes.

| variant | attention per layer | N = 3 | N = 12 |
|---|---|---|---|
| A: dense, routing learned from the ids | dense x3 | 0.016 [0.009, 0.023] | 0.194 [0.193, 0.195] |
| D: routing from the ids only (no content in the scores) | id-routed x3 | 0.028 [0.023, 0.033] | 0.195 [0.195, 0.196] |
| C: true routing graph, content attention inside it | self, within-source, all | 0.0054 [0.0053, 0.0056] | 0.013 [0.011, 0.015] |
| B': true routing graph, uniform pooling within source, dense query | self, uniform within-source, all | 0.0058 [0.0056, 0.0060] | 0.013 [0.012, 0.014] |
| B: true routing graph, fixed uniform combiners (a tree in token form) | self, uniform within-source, uniform all | 0.0091 [0.0086, 0.0096] | 0.017 [0.016, 0.018] |
| history-blind Bayes | | 0.174 | 0.211 |

W1, 4 layers with group ids: dense with learned routing (A4) 0.075 [0.055, 0.096] at N = 8 and 0.256 [0.255, 0.256] at N = 16; given the routing graph of sources and groups (self, uniform within source, content attention within group, uniform over all; G4) 0.031 and 0.039, and with uniform pooling within the group as well (G4u) 0.032 and 0.039, level with the hierarchies of E1 (0.033-0.047).

At N = 12, routing decides almost everything: given the routing graph, the transformer reaches 0.013-0.017, and left
to learn it, it stays at the history-blind solution for the whole budget. Content-dependent attention inside the
given graph is worth about 0.004 nats over fixed uniform combiners.

## E4: World FA binding with routing structure

World FA's prior f with one anti-correlated pair, N = 4 traders, T = 32 past rounds, report tokens (133 per episode).
The history value (history-blind Bayes NLL minus Bayes NLL) is 0.22 nats on the test set. The net is a token
transformer, width 64, 4 layers, 4 heads, 8,000 steps at batch 256 (Amendment 7). Read-out: the fraction of the
history value recovered.

| arm | attention per layer | fraction recovered, seed 0 / seed 1 | gap to Bayes |
|---|---|---|---|
| flat | dense x4 | -0.035 / -0.040 | 0.232 / 0.233 |
| flat1 | self, dense x3 | -0.033 / -0.023 | 0.232 / 0.230 |
| told (trader routing graph in layer 2) | self, within-trader, dense, dense | -0.035 / -0.023 | 0.232 / 0.230 |
| toldu (uniform pooling within trader) | self, uniform within-trader, dense, dense | -0.038 / -0.021 | 0.233 / 0.229 |
| learned (routing from ids) | self, id-routed, dense, dense | -0.021 / -0.020 | 0.229 / 0.229 |

**No arm recovered any of the history value within 2M episodes.** That includes the two that were told which tokens belong to which
trader, so all sit at the history-blind level. P11 held; P12 and P13 failed.

A likely reason, not tested: knowing which tokens are one trader's is not enough here. To see that two traders' errors
are anti-correlated, the net must also pair round t of one trader with round t of the other, and rounds carry only
random per-episode ids. That is a second routing problem, by round, which the trader mask does not solve. World FA's
arm that worked (0.95 of the history value) put each trader's errors in fixed column positions inside one token, which
solves both. The theory note's T7 says grouping by correlated histories needs keys of about min(T, 1/rho^2) bits per
item. So in World FA, unlike the PFN world, the routing that must be learned is by content, not only by labels.
A longer budget (World FA's binding-given arms acquired at 4-8k steps, but they had the columns) might change this.

## E5: deduction vs sampling for a tree net

**Why a tree allows deduction.** Given the five shared variables (the four labels and y*), a one-pass tree's sources
are independent. So its root state is a sum of N independent messages, and its law is the N-fold convolution of one
source's message law. One source's law is a 5-dimensional integral (its errors) for each of 3 variance levels; the
shared variables are another 5-dimensional integral. Both integrals are done by deterministic rules (scrambled-Sobol
points with fixed seeds; Gauss-Hermite in y* for the corrected rule). The convolution is computed exactly by FFT on a
grid, so the result is exact for that discretised law, not for the net's true law.

**Nets.** W0, N = 12, seed 0: m = 2 (test KL 0.038) and m = 16 (0.009).

**Results** (`results/e5.jsonl`, `e5summ.py`). The reference is plain Monte Carlo with 8M episodes (314 s on one core);
event counts are in brackets.

| method (m = 2 net) | seconds | E[NLL] error | P(NLL>8) | P(NLL>10) | P(NLL>12) | P(NLL>16) | P(NLL>20) |
|---|---|---|---|---|---|---|---|
| Monte Carlo, 8M episodes (reference) | 314 | 0 (SE 0.0003) | 4.5e-5 (358) | 8.3e-6 (66) | 2.1e-6 (17) | 1.3e-7 (1) | 1.3e-7 (1) |
| convolution, pre-registered rule for the shared variables | 653 | +0.0007 | 5.4e-5 | 1.5e-5 | 5.6e-6 | 1.3e-6 | 3.8e-7 |
| same, finer grid and more points (pre-registered refinement) | 1037 | +0.0021 | 5.8e-5 | 1.5e-5 | 5.8e-6 | 1.3e-6 | 3.8e-7 |
| convolution, y* by 40-point Gauss-Hermite (post-hoc fix) | 529 | +0.0008 | 4.4e-5 | 7.8e-6 | 1.7e-6 | 2.7e-7 | 9.8e-8 |
| same, refined (56 points in y*, 2^14 per source level) | 648 | +0.0025 | 4.8e-5 | 8.6e-6 | 2.0e-6 | 2.8e-7 | 9.6e-8 |
| Gaussian root (CLT), post-hoc rule | 22 | +0.013 | 7.2e-5 | 6.2e-6 | 1.2e-6 | 1.8e-7 | 8.6e-8 |

For the m = 16 net, exact convolution is out of reach because the grid grows like (cells per axis)^16. Its Gaussian
root gets E[NLL] within 0.001-0.003 nats in 18-25 s, but its tails are off by 1.3-5x (P(NLL>10): 2.2e-5 to 3.4e-5,
against 7.0e-6).

What this says:
- **The factorisation works.** Conditioning on five shared variables turns the 12-source computation into a
  convolution. After the fix, it matches the reference wherever the reference has events (within 1-18% for
  L = 8-12), and stays stable under refinement below that (L = 16, 20). Those deepest values rest on one reference
  event each, so they are not validated.
- **It did not beat sampling here.** One core samples this tiny net at 25,000 episodes a second (forward pass alone,
  278,000). In the deduction's 529 s, sampling would draw about 13M episodes, with SE 0.0002 on E[NLL] and, at the
  reference rates, about 29 events at L = 12 and about 2 at L = 16 and L = 20. These equal-time numbers are
  extrapolated from the reference run's rate, not measured. Sampling the dense token net costs about 57x more per
  episode (4,900 episodes a second, forward pass only), and it has no such factorisation.
- **The bug the reference caught.** The pre-registered rule for the shared variables under-resolved extreme y*
  (|y*| > 4), which drives about a tenth of the deep-tail events. Tails below 1e-5 came out 2-3x too high (10x at L = 16, where the reference has one event), and the
  pre-registered refinement (grid, per-source points) did not touch this. A dense rule in y* fixed it, after seeing
  the comparison.
- **Exact deduction needs small messages.** Its cost grows exponentially with the message dimension and only
  log-linearly with N. The more accurate 16-number tree is out of reach, and the Gaussian shortcut gets its mean but
  not its tails. This is the theory note's "expectations need rank; tails need dimension" in practice.

## Predictions scorecard

Pre-registered in `PREREG.md`: P1-P10 before any run; the rest in amendments, each before its results. Each row is
scored clause by clause against the registered wording.

| # | prediction (short) | outcome |
|---|---|---|
| P1 | W0: best tree (m >= 8) within 0.01 of the best dense net at N = 3, 6 and within 0.02 at N = 12, 24 | held, narrowly at N = 24 (0.0071 vs 0.0049; 0.0113 vs 0.0064; 0.0140 vs 0.0076; 0.0205 vs 0.0102) |
| P2 | W0: smallest m within 0.005 of the m = 32 tree grows with N, <= 16 at N = 24 | mixed: m = 4 at every N from 3 to 64 (flat, not growing), 16 at N = 256 |
| P3 | W0: tree (m 16) reaches val gap 0.05 sooner than tok at every N | held |
| P4 | W0: tree (m 16) seed range <= tok's at every N | held at N = 3-12; failed at N = 24, where every tok seed sits on the same plateau |
| P5 | W0: tok's KL grows from N = 3 to 24 more than the tree's | held (tok +0.18, tree m 16 +0.015) |
| P6 | W1: flat tree >= 0.8 x coupling-blind gap; hier (attention, m2 16) within 0.03 of the best dense net or below | first part failed (flat tree 0.09-0.40 of the gap); second held |
| P7 | W2: set recovers >= 50%; tree m 32 recovers less at N = 8 than at 4; tree m 8 <= 20% at every N | first two held (69-77%; 18-19% vs 27%); third failed at N = 4 (26%) |
| P8 | E3: B within 0.02 of A at N = 3 and below A at N = 12; D within 0.02 of A at N = 3 | held (B 0.009 vs A 0.016; 0.017 vs 0.194; D 0.028 vs 0.016) |
| P9 | E2 (W0): cross-source cut free at every N; labout k = 1 free; toq k = 2 costs <= 0.02 at N = 3, and the k needed for <= 0.01 non-decreasing in N | mixed: cross-source cut free (-0.002 to -0.003 at N = 3; at N = 12, 24 it rescues stuck nets); toq k = 2 +0.001 at N = 3, k = 2 also suffices at N = 12; labout k = 1 failed (+0.054 and +0.095 at N = 3) |
| P10 | E2 (W1): cut between groups free; cut between sources costly | not tested: 7 of the 10 W1 fine-tunes were lost (their base nets sat on another VM; reruns did not all finish before it shut down). The seed-0 base (0.284, stuck) was rescued by removing cross-source edges (-0.17), as in W0 |
| P11-P13 | E4: flat arms <= 10%; told >= 30% (both seeds); learned >= 30% (one seed) | P11 held (flat -0.04 to -0.02); P12 failed (told -0.035, -0.023); P13 failed (learned -0.021, -0.020) |
| P14 | E5: deduction within 0.003 on E[NLL] and within 25% on tails with >= 20 reference events | failed with the pre-registered rule (P(NLL>10) +78%); held only after the post-hoc fix (-6% to +6% where the reference has >= 20 events) |
| P15 | E5: Gaussian root underestimates P(NLL>8) by >= 2x (m = 2 and m = 16) | mixed: with the pre-registered rule, m = 2 under by 2.3x (held), m = 16 over by 2x (failed); with the corrected rule, m = 2 is over by 1.6x |
| P16 | E5: equal-time sampling better for E[NLL]; deep tails stable under refinement where equal-time sampling sees no event | first clause held; second failed: extrapolated equal-time sampling would see about 2 events at L = 16 and 20, and the "stable" pre-registered values were 4-5x off |
| P17 | rounds: at N >= 64 the best two-pass tree (m2 <= 8) <= the one-pass m = 32; at N = 3 within 0.01 of the one-pass m = 8 | held (0.013 vs 0.033; 0.061 vs 0.102; N = 3: 0.0033 vs 0.0074) |
| P18 | the one-pass smallest-m does not decrease from N = 24 to 256 | held (4 to 16) |
| P19 | set at N = 256 worse than the best tree | not tested (the set arm at N = 256 was dropped for memory) |
| P20 | M1: dense nets' cross-source share below their within-source share and below 0.1 | held for the N = 3 token nets and all set nets; failed for the N = 12 token nets stuck at history-blind |
| P21 | two-pass tree without the auxiliary loss within 25% of with it and below the capacity-matched one-pass tree | mixed: below the matched one-pass trees held (N = 24: 0.0055-0.0059 vs 0.016-0.017; N = 64: 0.012-0.013 vs 0.027); "within 25%" failed literally at N = 24, m2 = 4, where the no-auxiliary tree was 29% better (0.0055 vs 0.0078) |
| A11 (exploratory, written after E1) | at 4x the budget (N = 12) the one-pass tree stays above 1.3x the set net's KL | held, barely (0.0051-0.0052 vs 0.0039: 1.31-1.34x); the two-pass tree at 4x is 0.0011-0.0012 |
| theory M1 | nets without a per-source nonlinearity before mixing show less within-source interaction than Bayes | consistent (stuck net 0.0008 vs 0.011; nets that bind 0.043 vs 0.039) |
| theory M2 | few routing classes in layer 1, labels explain most; query mass tracks log precision | held for the nets that learned the task; not for the stuck ones |
| theory M3 | (iii), (iv) worse than (ii); refuted if (iv) as good as the net | held ((ii) 0.14-0.30, (iv) 0.44-0.72, (iii) 0.77-1.60); our (iv) averages probabilities |
| theory M4 | nets near Bayes pass 2-4 dimensions per source | consistent but not comparable: rank 2 per head, token and layer keeps the N = 3 net within KL 0.005 (rank 1: 0.016-0.040); that is about 40 numbers per source per layer |
| theory M5 | sink weight vs posterior variance R^2 >= 0.5 in some nets | held (0.61, 0.72) |
| theory 6.5 P1 | at matched size, a two-pass tree beats a one-pass one once N >= 64, similar at N = 3 | first part held (N = 64: 0.013 vs 0.027); second failed: two passes also win at N = 3 (0.0036 vs 0.0057-0.0061) and 24 |
| theory 6.5 P4 | dense with learned routing does not match dense with routing given | held for the dense nets (0.194 vs 0.013 at N = 12); the tree with learned routing was not run |

## What this says, and what it does not

On Kelvin's question, for these worlds and this budget:
- **Is tangledness necessary? Not for combining evidence.**
  - When Bayes is a sum over sources (W0) or over known groups (W1), nets with no cross-source interaction before a
    sum (trees) or before a group sum (hierarchies) come within about 2x of the dense transformer's KL.
  - The two-pass trees we tested match it at equal data. The controls point to the second round itself rather than parameters or the auxiliary loss. At 4x the data (N = 12) they beat the dense net by about 3x (0.0011-0.0012 vs 0.0039), while the one-pass tree stays 1.3x behind it.
  - The dense nets that learned the task already compute something close to this shape inside.
- **Where dense attention did better:**
  - finding which sources belong together when nothing labels them (W2's hidden pairs);
  - in the token format, grouping a source's tokens by ids at all. This it learns slowly, with a plateau that
    lengthens with N.
- **"Scale-free" in Kelvin's sense** (the same merge at every scale, dimension allowed to grow) fits W0-W1 here. The
  message size learned trees needed did not grow with N from 3 to 64. At 256 every net struggled.

What this does not show:
- **That the token net cannot learn routing at large N.** Its failure is a budget result: two of three seeds had begun to escape
  at N = 12 near the end of training. Longer runs, other initialisations or curricula might escape; World FA tried
  some, without success.
- **That trees cannot find hidden pairs.** We showed only that sum-pooled trees trained this way did not.
- **That rounds would help on richer data or larger nets.** Here the controls separate the second round from parameter count and the auxiliary loss, but not from the pilot-standardised input that defines it.
- **Results beyond this setting.** These are small nets (10-21k parameters), one budget, 2-3 seeds, and simple
  synthetic worlds with fixed structure (Gaussian errors, three variance levels, fixed group size).
- **That deduction pays.** It was exact (for a discretised law) only for 2-number messages, and was not cheaper than
  sampling for such small nets.

## Reviews (GPT-6, `results/review1.md`, `results/review2.md`) and what changed

The review was adversarial (gpt-6-astra, checked in its log). Changes made:
- **Stale numbers.** Every table is regenerated from one snapshot (`mktables.py`, `results/snapshot.json`).
- **"History-blind".** The claim is now backed by a direct comparison of predictions (`blind_cmp.py`), not by
  matching average gaps.
- **Budget qualifier.** "Within 10.24M episodes" is added to every routing claim, with escape times.
- **Two-pass confounds.** The two-pass trees' extra parameters, auxiliary loss and standardised feature are now
  stated, and an ablation was added (Amendment 10, before its results).
- **W2.** The conclusion is restricted to the architectures we trained.
- **Mechanism wording.**
  - M1's ratio is defined, M5 reports the net's and Bayes's variance separately, and the r = 0.50 head is included.
  - M2's regression design and M3's probability averaging are disclosed.
- **Scorecard.** P15, P16 and the theory note's 6.5 P1 and P4 are now scored clause by clause.
- **E5.** "Exact" is qualified, equal-time numbers are marked as extrapolated, and the E5 nets' recipe is stated.
- **Amendment times.** Headings are corrected to the hash-log times (they had been written up to 30 min late).

A second pass (`results/review2.md`) confirmed that the regenerated tables match the snapshot and that the first
round's numeric errors are fixed. It led to these further changes:
- E2, M4 and P21 ranges are now stated per seed, with exceptions.
- "Carries the variance" became "correlates with".
- The learning-speed claim is limited to one seed.
- The capacity-matched trees' parameter counts are given exactly.
- E5's original-rule error at L = 16 (10x) is stated.
- "Two of three" seeds had begun to escape at N = 12.

Still partly open: the pilot-standardised input remains confounded with the second round, M3's logit-average
intervention was not run (we averaged probabilities), and the chronology rests on the hash log, not on versioned
snapshots.

## Files and cost

Work dir `/home/agent/workspaces/mats/work/scale-free-exp/`:
- `PREREG.md` and `prereg_hashes.txt`.
- Code:
  - `sfworld.py`: worlds and exact Bayes; tests in `test_sfworld.py`.
  - `sfgpu.py`: on-device sampler, tested equal in law; tests in `test_sfgpu.py`.
  - `sfmodels.py`, `sftrain.py`.
  - `est5.py`: E5.
  - `m1prof.py`, `tokan.py`, `klpair.py`, `blind_cmp.py`: measurements.
  - `e4fa.py` with `fa_core_copy.py`: an unchanged copy of World FA's core.
  - `mktables.py`, `summ.py`, `e5summ.py`, `tokan_summ.py`: tables.
- Runs: `runs/sfx-b{3,4,5}/` (final syncs; result files written earlier name `local/sfx-b*`, copies of the same runs, now in trash) and `runs_local/` (E5 and Amendment 11 trees, CPU). Results: `results/` (tables: `results/tables.md`, from `mktables.py runs/sfx-b3 runs/sfx-b4 runs/sfx-b5`).
- GPU: on-demand L4s logged in `GPU-LOG.md`.
  - Two failed starts cost $0.28: one too slow, one missing a file.
  - Three VMs ran in asia-southeast1-a after US and EU stockouts: $2.20, $3.13 and $1.88.
  - Total $7.49 at the list-price ceiling (Billing API disabled), under the $10 cap. All five VMs are deleted (instance list checked 23:45Z).
- CPU work ran at the lowest priority, at most 3 processes.
