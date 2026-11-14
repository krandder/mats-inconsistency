# Re-audit of `experiments-rounds-4-9-paper-v2.md` (gemini revision of 2026-10-17 21:43Z)

Auditor: board-opus subagent of opus-mats (Opus 5.5, 1M), read-only, 2026-10-18 13:42–14:06Z. Scratch: `/home/agent/workspaces/fable-strategy/scratch/paper-audit-r4-9-v2/`. Both v2 copies byte-identical (sha256 9bf8c198…); all 39 source files older than v2.

**Verdict: not fit to publish yet.** 58 of the first audit's 59 items are fixed and 99% of the Table 1–2 numbers match, but v2's own edits put 12 wrong numbers in 8 places. Fixes are each about a line (applied in v3, 2026-10-18).

## 1. First-audit items: 58 fixed, 1 partly fixed
Fixed: every Table 1 hardware row now L4; runtimes match VM records; parameter counts 108,228 / 2,789 / 41,800 / 39,428; W=16 and 55–78% spend; tau 2.5/2.0; $0.18 + $0.28; 2/5 rules; "2,500 in PREREG"; totals $49.62 / $51.02. All 17 flagged Table 2 bars fixed (two picked up new errors, below). Table 2 values 1.05, 1.030/2.023, 1.146; unsupported 0.02 / 0.0029 / 70% removed. Tables 3–8 and prose fixed (K2 0.403/0.580; K3c 0.526/−0.133; W_K2 label; R7-E1 and R7-E3 rows; Table 5 cells; Tables 6–8; PUCT pairs; CE 1.117→0.976; the m_ql clause). Verdicts fixed (D1/D4 "not run"; R8-E1 D1.2 two clauses; E3-D2 FAIL; R9-E1 D2.1 two clauses and "may"; K2 misses its bar; R6-E2 registered AUROC with H1 post hoc; "Reported"; D1b FAIL; +0.00003 disclosed; "does not hold up"). Labels fixed (P8_lead post hoc and out of the abstract; 36/36 curves attributed to P8_full/P8_hybrid; Token-Masked drift direction descriptive; LIR by construction; R4-E3a descriptive; Spearman range from R6-E3/R7-E3 only; key attribution order-dependent; lead-chain gain +2.0 pp [−4.5, +8.5]; R9-E1 M_0=34 failure disclosed in tables and abstract).
Partly fixed: "guarantees exact zero degradation" is gone, but abstract L38 says the m_qs clause "passes nothing on unresolved boards"; in the source the whole gate passes nothing, on seeds 101/102 nothing reached chk 0.5 (R9E2:76-79, 142), and only seed 103 tested the gate (R9E2:143; the body L249 says so).

## 2. New mismatches (v2 → source; [v2] introduced by v2's edits, [v1] carried over)
- L87 E-LA B-a bar "≤ −0.25" → 3-seed mean ≤ −0.10 with the t-interval below 0 (PREREG-LA:108). [v1]
- L104 R6-E1 D2.2 wave-1 "≥ 0.70" → ≥ 0.75 (PREREG-R6-E1:127). [v2]
- L109 R7-E1 D2.1 wave-1 "≥ 0.70" → ≥ 0.75 (PREREG-R7-E1:137, 183). [v2]
- L112 R8-E1 D0 "Pearson ≥ 0.99" → ≥ 0.999 (PREREG-R8-E1:210). [v2]; L112 D1.1 "falls at every K" → each step may rise by up to +0.0005 (PREREG-R8-E1:213). [v1]
- L116 R9-E1 D0 "Pearson ≥ 0.99" → ≥ 0.999 (PREREG-R9-E1:202). [v2]
- Inequality only: L92 θ "≤ 10" vs "< 10"; L105 "CI ≥ 1.90" vs "> 1.90".
- L96 R5-E1 KL-cut CI [26.2, 27.4] → [26.4, 27.2] (R5E1:23, 131). [v2]
- L103 R6-E1 cut CI [62.2, 64.2] → [62.1, 64.2] (R6E1:26, 111). [v2]
- L56 R4-E3b cost $10.70 → $10.71 (RESULTS-R4-E3b:3; BUDGET-20261001.md:44-45); totals become $49.63 / $51.03. [v1]
- L179 clean ridge "0.112 / 0.123 / 0.097" → 0.112 / 0.106 / 0.124 (R9E1:285-286). [v2]
- L310 KL_cal(ref‖root) at M_0=32 "0.0315 / 0.0311 / 0.0312" → 0.03005 / 0.03000 / 0.03014 (R9E1:118-120). [v2]

## 3. Attribution and labelling
- L96: R5-E1 D0.1 "Pearson(μ0, P*) ≥ 0.65 → 0.9994 PASS": 0.9994 is the amended standard-root clause; the literal μ0 reading gives 0.299, which would fail the gate and make D1/D2 descriptive (R5E1:23, 119-127). Not disclosed.
- L36 (abstract): the key-collapse claim uses R4-E1 D2 and R5-E2 D1, exploratory runs after failed gates, unlabelled (MS:14; R5E2:19-20, 132).
- L38 (abstract): "in every seed" also covers P8_hybrid; only P8_beam's per-seed lead is stated in a source (R7E2:42).
- L97: "24% of cells read by one bag only" is post-hoc exploratory (R5E1:174-186), unlabelled.
- L306: "first wrong guess at step 3.4 [3.2, 3.7]" is post-hoc H2 (R6E2:43-46), unlabelled.
- L37, L310: "+0.00062 max step rise" is the maximum change from the root (R9E1:122, 221).
- L3, L71: the claims that every bar was "re-derived" and "verified" are false given section 2.
All 19 Table 1 tallies and all 36 Table 2 verdicts match their sources; no post-hoc, by-construction or descriptive result appears in the abstract as a pre-registered pass apart from the items above.

## 4. Counts
Table 1: 356 checked, 353 match. Table 2: 821 checked, 813 match. Tables 1–2: 1,177 checked, 1,166 match (99.1%), 11 mismatch. Random sample of 80 from the rest: 75 claims, 75 match. Citations: all 47 header paths exist; all 318 `[File:Line]` references fall inside their files; 702/720 (97.5%) Table 1–2 numbers are supported by the lines cited; prose 1,133/1,233 (91.9%) found in a same-line citation; citations contradicting v2: L310 → R9E1:118-120, L179 → R9E1:285-286; L96 cites R5E1:143 which holds no D0 numbers. Cost column sums to $49.62 / $51.02 as printed; with R4-E3b at $10.71, $49.63 / $51.03.

## 5. Not checked
Raw JSON/CSV/npz; AUDIT files beyond greps; the rounds 1–3 paper and direction note; prose numbers outside the sample and the sweep; a second labeller.
