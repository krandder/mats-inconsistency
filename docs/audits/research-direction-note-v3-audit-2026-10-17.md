# Audit of `research-direction-note-v3-2026-10-17.md`, section 5 (Evidence So Far, Rounds 1–9) and the disclosures list

Auditor: the same board-opus subagent (Opus 5.5), read-only, 2026-10-17 21:29–21:36Z; lines 105–128 of the note. Scratch: `/home/agent/workspaces/fable-strategy/scratch/paper-audit-r4-9/` (`extract_note.py`, `labels-note.tsv`, `note_stats.py`).

**Headline:** numbers are mostly right (348 of 364 match, 95.6%; 12 conflict, 4 unsupported). The problems are attributions and verdicts: one header labels four failed rules as PASS, and several post-hoc or exploratory results sit under "Holds" as if pre-registered.

References: "R5E1:27" = RESULTS-R5-E1.md line 27; PH = puzzles/R8-E2/POSTHOC-RERUN.md; OVN = the rounds 1–3 paper; "L109" = note line.

## 1. Mismatched or unsupported numbers (note -> source)
- L109 "0.9999 battle accuracy on 4×4 and 6×6": 4×4 is 1.0000 (MS:34).
- L111 "null search cuts deep-contradiction loss by 95–99.6%": source range 99.38–99.64% (OVN:99, 240).
- L112 "R5-E1 raw K1 0.382": R5-E1 is 0.375 (R5E1:27); 0.382 is R6-E1's (R6E1:121).
- L116 "53.7% vs mix": source 53.1% (R5E1:236) or 47% (R6E1:196).
- L116 looped_attn_T3 W(8)/W(1) "1.06 < 1.99 < 3.44": source 1.025 / not reported / 3.657 (R7E1:240).
- L117 CIs [0.320, 0.375] and [0.346, 0.391] -> [0.2751, 0.4195] and [0.3096, 0.4287] (R7E3:68-69); "63.2–63.9%": trust_pool's cut is 63.4% (R7E3:74).
- L119 "restored 90.2% of the exact-Bayes reward ceiling": 0.902 is the normalised gain from pre-RL reward to the ceiling; LIR's reward is 0.798/0.830, 96% of the ceiling (R5E1:24, 257).
- Unsupported: L119 "KL ≈ 0"; L119 "+0.014 in R5-E1 B0" (no B0/B1 arms, R5E1:109-112); L116's 1.99; L120's "at K=1" for the +0.00062 rise.

## 2. Verdicts or attributions stated differently
- **L116 header** marks "R5-E1 D2.2, R6-E1 D3, R7-E1 D1.1/D1.2, R8-E1 D2.1, R9-E1 D2.2 PASS"; four failed: R5E1:28, R6E1:30, R7E1:31, R8E1:32.
- L119 "R5-E2 (D1/D2 FAIL)": D1–D3 were not rule tests because the D0 gate failed (R5E2:12, 47).
- L119 "R6-E2 (D1/D2 FAIL): P6 lost to P3, 38.9% vs 40.2%": R6-E2 never ran P6; those are R7-E2's numbers (R7E2:19); the P6-vs-P3 rule was R5-E3's D1.
- L110 "decreases KL at every step" (R9-E1): seed 101 rose +0.00016 (R9E1:152). L109 "holds across every step and seed": seed 102 looped_attn_T3 sat +0.00003 above the strong root (R9E1:42). Both passed within tolerance, not strictly.
- L111 "spending 99.5% of budget" also covers P8_beam, which spent 55–78% (R7E2:48). "beats P6 with 95% CIs > 0" omits P8_lead vs P6 = +7.8 [−0.9, 16.5] (PH:55).
- L112 "K3 at 0.234 due to 0.56 cone overlap": R7-E3 says "presumably ... not measured" (R7E3:88); 0.56 is R5-E2's post-hoc figure (R5E2:74).
- L120: R8-E1 D1.2 also failed its naive-degradation clause (R8E1:23, 163); the R9-E1 strong-root pass fails at M_0=34 (R9E1:121-126), not said; "because the 8 background reads raise the intercept" is only the auditor's "may" (R9E1:397-401), and D2.1's second failing clause (MSE vs econ_kc_K3) is omitted (R9E1:21, 255).
- L121 "dropping R7's greedy lead chain dropped witnesses to 42.8%" stated as a cause; R8-E2 changed several things at once ("no arm here isolates that", R8E2:41-44); the post-hoc rerun put P8_full at 44.3%, level with P8_beam (PH:36-38).
- L122 "due to 4-ply horizon bias" is untested in R8-E3 (R8E3:111-113); the decomposition is from R6-E3/R7-E3 post-hoc analysis.

Post hoc / exploratory / descriptive / by construction listed under "Holds":
- L111, L119: P8_lead's 46.7% and +6.5 pp are post hoc and unaudited (PH:1-5, 61).
- L109: the early-stopped looped_attn_T3-K3c "never degrades" result is a post-hoc rerun (R8E3:25, 119).
- L117: R4-E1 (+0.418) and R5-E2 trust results are exploratory runs after failed gates (MS:14; R5E2:19).
- L115: the R8-E1 2×2 PUCT factorial is descriptive under the failed D2.2 (R8E1:33, 212); R7-E3's policy independence comes from imitating a reference that is policy-independent by construction (R7E3:99-100).
- L111: D2.2's 0.0000 holds because the gate passes nothing on unresolved boards, and only seed 103 had invalid leaves to reject (R9E2:142-143).

## 3. Counts
| Scope | Checked | Match | Mismatch | Unsupported |
|---|---|---|---|---|
| Section 5 total | 364 | 348 (95.6%) | 12 | 4 |
| "Holds" (L109–117) | 276 | 264 (95.7%) | 11 | 1 |
| Disclosures (L118–122) | 88 | 84 (95.5%) | 1 | 3 |

Not checked: L107's prior-work figures (external paper) and the paradigm table (L124–128).
