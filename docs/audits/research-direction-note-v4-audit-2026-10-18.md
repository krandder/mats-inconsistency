# Re-audit of `research-direction-note-v4-2026-10-17.md`, section 5 (evidence), 2026-10-18 13:42–14:01Z

Auditor: board-opus subagent (Opus 5.5, 1M), read-only. Scratch: `/home/agent/workspaces/fable-strategy/scratch/note-v4-reaudit/`. v4 sha256 c621dd8f… identical in outbox and `inconsistency-search/docs/direction/`. Rounds 4–9 sources unchanged since the v3 audit. v4 cites no rounds 10–12.

**Verdict: not fit to be the live page.** Numbers are 97.2% right (528 of 543 checked), 24 of yesterday's 25 items are fixed, but six FAIL verdicts are shown as PASS, one post-hoc result is listed as pre-registered, and the R7-E2 CIs and bar are wrong. All need only line edits.

## 1. Yesterday's 25 items: 24 fixed, 1 not fixed
- Not fixed (and worsened): L116 "47.3% vs mix" at tau=0.3 (R6-E1). 0.4724 is the tau=2.5 cut; the tau=0.3 cut is 0.5366 (results.json `msecut_vs_mix_t0.3`), so yesterday's flag on 53.7% was itself wrong and the edit introduced an error.
- Fixed: 4×4 accuracy (L109), null-search range (L111), K1 values (L112), looped_attn_T3 W ratios (L116), R7-E3 CIs and cuts (L117), 90.2% → "96.1% of the ceiling; 90.2% of the available gain" (L124), removal of "KL ≈ 0", "+0.014 B0", "1.99", "at K=1"; L116 header verdicts; R5-E2 "exploratory diagnostics" (L120); R6-E2 vs P6 (L124); R9-E1 strictness (L110); 99.5% budget and P8_lead CI (L111, L119); 0.234/0.56 (L112); the four L120 omissions (L125); 42.8% causal claim (L126); horizon bias (L127); post-hoc labelling of P8_lead, early stopping, R4-E1/R5-E2 trust, the 2×2 factorial with R7-E3 imitation, D2.2 0.0000 (L119–122, L111).
- Six fixes (15, 16, 18, 19, 22, 24) added new errors, below.

## 2. New mismatches and unsupported numbers
- L110: the R9-E1 +0.00016 rise is "from K=16 to K=24"; source K=24→32 (RESULTS-R9-E1:152).
- L111: R7-E2 P8_beam "+4.33 pp [0.73, 7.94] vs P3, +5.53 pp [1.46, 9.61] vs P6"; source +4.3 [2.0, 6.5] and +5.5 [0.06, 11.0] (RESULTS-R7-E2:19; ANALYSIS.json). The "+6.0 pp bar" is R8-E2's; R7-E2's bar was +8 pp.
- L111: mix "+0.4533"; source 0.4526 (+0.453 in the audit, +0.45 in RESULTS).
- L116: "47.3% vs mix" at tau=0.3 (see above; 53.66%).
- L119: R8-E3 early stopping on "1,000 held-out" positions; source 500 (RESULTS-R8-E3:26,121).
- L121: "R8-E1 D2.2 FAIL against the tightened KL ≤ 0.032 bar"; no such bar (PREREG-R8-E1:229-233); D2.2 failed on W(16)/W(1) = 1.417 (RESULTS-R8-E1:33,208), as v4's own L125 says.
- L122: seeds 101/102 "0.0000 … under plain chk_mix"; in R9-E2 chk_mix gives +0.0038/−0.0004 (ANALYSIS.json); the 0.0000 comes from the chk ≥ 0.80 gate (RESULTS-R9-E2:78).
- L125: attn_L3-K3 "+0.00066 (seed 101), +0.00105 (seed 102)"; source +0.00070 and +0.00037 (RESULTS-R9-E1:222); attn_L3-K3 is reported only, not a D1.2 part.
- L126: R8-E2 "drawing fresh test boards"; R8-E2 reused R7's boards (RESULTS-R8-E2:92-93); the source lists different changes (:41-42).

## 3. Verdicts, attributions, post hoc
- L109 header "R4-E2 D0/D1/D2 PASS", inline "D0 PASS" and "D1 PASS": all three FAILED, "no rule passed" (RESULTS-R4.md:13-15,20). New in v4.
- L109 "R5-E1 D1.1 PASS": FAIL (RESULTS-R5-E1:25).
- L113 "E-LA PASS": A1–A3, B-main and B-b FAIL; only B-a passes (RESULTS-LA.md:15,41-48), including A2, the preconditioning mechanism behind the Vladymyrov claim cited there.
- L113 "R6-E1 D2.2 PASS": FAIL, coverage 0.9494 (RESULTS-R6-E1:29).
- L111: R7-E2 "failing E2-D1 only on the +6.0 pp point-estimate bar vs P3"; P8_beam missed +8 pp vs both P3 and P6, and D1.2 also failed (RESULTS-R7-E2:19-20); contradicts v4's own L124.
- L112: R7-E3 D1(1) sits under "salience keys track graded overlap"; the source says pure salience failed (0.234) and K3c's gap "is a property of centring within a skewed bag, not evidence that the keys learned which rollouts are duplicates" (RESULTS-R7-E3:19,92-93).
- Post hoc shown as pre-registered: L109 "46.3% (6×6 Sudoku, Rounds 1–3)" is Round 1b, labelled post hoc (overnight paper :95,182-188,250).
- L108 "Audited PASS Rules Only" still covers clauses of failed rules: L116 (disclosed), R8-E3 D1(1) (L109), R7-E2/R9-E2 D1.1 (L111), D2(1) (L117). L117's 63.4/63.9% are R7-E3 only; in R6-E3 trust_pool was 26.8% worse (RESULTS-R6-E3:56).
- Minor: L124's −0.266 is P7_stitched vs flat P3, both under global chk_mix (RESULTS-R6-E2:18). L109's 41.1% is R9-E2's D1.2 cut-vs-P0 clause, labelled D2.1.
- Section 6 spot check: L144/L145 "D0–D3 PASS" and "LIR D1/D2/D3 PASS" (R5-E1 D3.1 failed, RESULTS-R5-E1:29); L148 "R4-E3b D0/D1/D2 PASS" (3 of 7 rules passed; D1a/b/c and D2 failed, RESULTS-R4-E3b:11-15,17).

## 4. Counts
Tokens 598; not checkable 55; checked 543; matched 528 (97.2%); mismatched 14; unsupported 1. Holds (L108–117) 314/324; post-hoc list (L118–122) 73/76; disclosures (L123–127) 140/142. PASS/FAIL mentions: 96 checked, 90 correct, 6 wrong.

Not checked: L107 external figures; paradigm table (L129–133); raw arrays beyond ANALYSIS.json / results.json / numbers.txt.
