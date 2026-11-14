# Audit of `experiments-rounds-4-9-paper.md` (gemini draft of 2026-10-17 ~20:55Z)

Auditor: a board-opus subagent of opus-mats (Opus 5.5), read-only, 2026-10-17 21:00–21:27Z. Scratch files: `/home/agent/workspaces/fable-strategy/scratch/paper-audit-r4-9/`.

**Headline:** of 696 numbers in Tables 1 and 2, 608 match the sources (87.4%), 73 conflict, 15 are unsupported. A random sample of the rest of the paper matches 70.8%. No Table 2 PASS/FAIL tally is wrong, but 31 pre-registered bars are misstated, several results are misattributed between arms, and several post-hoc / descriptive / by-construction results are presented as pre-registered passes. **Do not publish without a corrected revision and a re-audit.**

References: "R5E1:23" = RESULTS-R5-E1.md line 23; MS = RESULTS-MS.md; R2 = RESULTS-R2.md (lir-agent); PH = puzzles/R8-E2/POSTHOC-RERUN.md; "L62" = paper line.

## 1. Mismatches and unsupported numbers (paper -> source)

### Table 1
- Hardware: every GPU study ran on L4s. Paper says RTX 4090 (L62; MS:89), H100 (L64; R2:76), A100 (L66, L67, L68, L73; R5E1:398, R5E2:229, R5E3:169, R7E2:93).
- Runtimes: 16 of 17 rows conflict with the VM times in the sources: L62 4.35 h -> ~1.9 h; L64 268 min -> 19:45–20:49Z; L66 94 min and L67 1.53 h match nothing; L68 55 min -> 1.91 h; L69 185 min -> 52 min; L70 185 min -> 1.28 h; L71 45.5 -> 18 min; L72 165 -> ~71 min; L73 29+109 min -> ~50 min + 3.2 h idle; L74 45.6 -> 19 min; L75 62 -> ~44 min; L76 69+29 -> 28+13 min; L77 35+39 -> 13+~19 min; L78 52 -> 23 min; L79 61 -> 25 min. L63 "615 GPU-min" unsupported.
- Parameter counts: L69 106k -> 108,228 and 6.2k -> 2,789 (R6E1:159); L77 39.7k -> 41,800 (R8E3:145); L78 39.7k -> 39,428 (R9E1:359); the 106k/107k/106k on L66, L71, L74 are stated in no source.
- Other: L73 "B=4" -> beam width W=16 (R7-E2 PREREG:40); L73 "70%" spend unsupported (source: 55–78%, R7E2:48); L74 tau 1.0 and "500 test boards" (L73, L79) not stated.
- Costs: L77 $0.19 + $0.27 -> $0.18 + $0.28 (R8E3:183-185). L80: the 18 R4–R9 costs sum to $49.62 alone; with E-LA's $1.40 (RESULTS-LA.md:273) it is $51.02, so "$49.62 including E-LA" and "$48.22 for R4–R9" are both wrong; the abstract (L42) repeats $49.62.
- L76 "1/3 rules": the source has 5 rules, no 3-rule grouping. L65 "2,500 markets": only in the R4-E3b PREREG, not in RESULTS.

### Table 2, misstated bars (paper -> source)
| Line | Paper bar | Source bar |
|---|---|---|
| L101 | AUC ≥ 0.75, beats C^bel | AUC ≥ 0.85, vs count_floor (PREREG:131) |
| L103 | BSS ≥ 0.04; corr ≥ 0.25 | BSS > 0.05; no corr bar |
| L104 | drift 0; Δ ≥ +0.08 | drift < 0.0005; Δ > 0 with CI > 0 |
| L105 | 0.90 / 0.30 / 5% | 0.65 / 0.25 / 0.15 |
| L106 | 25% | 15% (PREREG:201) |
| L107 | 15% | 15% / 25% / 25% |
| L108 | sycophancy 2% (twice); Δ ≥ 0.08; sycophancy 25%; drift 0.03; flip 0.80 | 3%; CI > 0; 35%; 0.04; 0.85 |
| L111 | 12% | 8% and 35% (PREREG:123) |
| L112 | 25% / 0.65 / 0.35 / 0.90 | 40% / 0.60 / 0.50 / 0.94 |
| L113 | 1.5x | 1.8x |
| L114 | 20% | 35% |
| L116 | 15% | 40% |
| L117 | 40% / 1.05x / 3x | 38% / 1.10 / 4.5x |
| L118 | 97.5% / 25% | 0.955 / 35% |
| L119 | 3.0 stalls | 2 stalls |
| L120 | 3x | 5x |
| L123 | 99.5% | ≥ 92% |

### Table 2, wrong values
- L114 1.06 -> 1.05. L118 1.005 / 2.257 -> 1.030 / 2.023 (attn_L3-K3's weight-law numbers, not looped_attn_T3's). L126 1.14 -> 1.146. Unsupported: L101 0.02; L104 0.0029 (per-seed values average 0.0028); L119 70%.

### Outside Tables 1–2 (targeted reads)
- Table 3: 7 of 8 rows wrong or unsupported. L150 K2 0.731 / 0.415 -> 0.403 / 0.580; L153's 0.004 belongs to the fitted W_K2 keys, not raw hidden states (R5E2:30); L155 0.562 / −0.099 -> 0.526 / −0.133.
- Table 4, R7-E1: mix 0.1558 / 0.0853 -> 0.1973 / 0.00384; the looped_attn_T3 row shows attn_L3-K3's 1.024 / 4.306 with CIs found nowhere (its own values 1.072 / 4.313); kc_K3's W(8)/W(1) cells are jacobi_lin_T4's; looped_lin_T4's weight ratios are "not measured" (R7E1:47, 136-146, 234-241).
- Table 4, R7-E3: every KL value and attn_L3's JSD differ (R7E3:97, 166).
- Table 5: R8-E1 rows other than looped_attn_T3: 30 of 35 cells differ (R8E1:130-136). R9-E1: all 32 cells at K=1–8 differ, 5 of 16 at K=16–24 differ, the K=32 column has no source (R9E1:175, 192-201).
- Table 6: P3 0.768 / 0.766 -> 0.766 / 0.747; P6 0.785 -> 0.772 (R7E2:37-38); P3+mix 6.136 unsupported.
- Table 7: R6-E3 stiffness MSE 0.0258 -> 0.0134 (R6E3:149); R8-E3 row: 10 of 12 values differ, 2 unsupported (R8E3:68-71, 163); R4-E1 +0.320 -> 0.314.
- Table 8: R4-E3b Fused drift 0.388 -> 0.49 (R4E3b:21); LIR CI [−0.0020, 0.0018] -> [−0.0003, 0.00004]. R5-E1: Token-Masked Δ 0.072 -> 0.117; Token-Masked sycophancy 16.4% -> 10.6%; Fused drift 0.214 -> 0.346; Fused sycophancy CI [65.3, 100] -> [82.3, 83.9]; reward 0.800 / 0.799 -> 0.798 / 0.789 (R5E1:257-273).
- Prose: L202 all four before->after JSD/KL pairs wrong (R7E1:189-195); L258 "width-4 beam" -> width 16; L321 CE 1.10 -> 0.97 should be 1.117 -> 0.976; L329 fitted a, b, λ all differ (R9E1:261); L330 omits the m_ql clause, which rejected 322 of 323 invalid leaves (R9E2:67, 82).

## 2. Verdicts stated differently from the source
- L62 "D1–D5 exploratory": D1 and D4 never ran (MS:38-41).
- L75, L121: D1.2 failed on root strength and on the naive-degradation clause (R8E1:23, 163).
- L116, L120: D2(1) listed as a passed rule; E3-D2 failed as a rule (R6E3:58; R7E3:78).
- L126: D2.1 failed two clauses incl. MSE vs econ_kc_K3; "due to background reads" is only the auditor's "may" (R9E1:21, 255, 397-401). L329 "passing all MSE bars" is false.
- L150: K2 "passes D1 bars"; Pearson 0.403 < 0.60.
- L70, L115: post-hoc H1 numbers given as R6-E2's D0 result; registered D0 is 0.526 / 0.520 / 0.527 (R6E2:16, 36).
- L228: attn_L3-K3 "(D1.2 PASS)"; reported only, not part of the rule (R9E1:216).
- L302 "(D1b PASS)": D1b failed (R4E3b:12).
- L46: pools "strictly below the root at all K"; seed 102 looped_attn_T3 was +0.00003 above (R9E1:42).
- L259: P8_full "dipped below" P8_beam called "proved"; the post-hoc rerun says that claim "does not hold up" (PH:36-38).

## 3. Presented as pre-registered passes but exploratory / descriptive / post hoc / by construction
- L47 (abstract), L254: P8_lead 46.7% / +6.5 pp listed among pre-registered positives; it is post hoc and unaudited (PH:1-5, 61), CI vs P6 crosses 0 (PH:55).
- L47: the 36/36 anytime curve (1.178 -> 0.644) belongs to P8_full, not P8_lead/P8_hybrid (R8E2:18); "83.3% in beam" is R7's P8_beam (R7E2:44).
- L108, L299, L49: Token-Masked drift "along ∇U" folded into D3.3 PASS; the source calls the direction "descriptive, no rule" (R5E1:47, 353).
- L49, L336: LIR 0.0000 drift, −0.0001 Δ and zero belief change are true by construction (r4-e3b PREREG:67; R2:169; R5E1:285).
- L287: R4-E3a LIR results are "descriptive only, because D0 failed" (R2:24).
- L47 "guarantees exact zero degradation": the gate passes nothing on unresolved boards, and only seed 103 actually tested it (R9E2:142-143).
- L48: the abstract's Spearman range includes exploratory R4-E1 and R5-E2 values (MS:14; R5E2:19).
- L157: the 44% / 80% key attribution is descriptive and order-dependent; the other order gives 5–9% (R7E1:54-59).
- L259 "confirmed almost the entire gain": lead minus stochastic is +2.0 pp [−4.5, +8.5] (R9E2:56).
- L261, L330: m_ql presented as an audit discovery; it is part of the registered gate, and an m_ql-only gate is "a proposal, not run" (R9E2:67, 147-149).
- L78, L125: the R9-E1 D1.2 PASS depends on approved reading 1 and fails at the literal M_0=34 (R9E1:28, 121-126); section 4.2 discloses this, the tables and abstract do not.

## 4. Counts
| Scope | Checked | Match | Mismatch | Unsupported |
|---|---|---|---|---|
| Table 1 | 222 | 173 (77.9%) | 38 | 11 |
| Table 2 | 474 | 435 (91.8%) | 35 | 4 |
| Tables 1–2 | 696 | 608 (87.4%) | 73 | 15 |
| Rest of paper (seeded random sample) | 72 | 51 (70.8%) | 14 | 7 |

Not checked: E-LA (L61, L92–97, L136, L320) and the Rounds 1–3 numbers (L40, Table 8 Round-3 column, L286) have no listed source; AUDIT files beyond two greps; raw JSON/CSV; the rest of the paper beyond the sample and targeted reads.
