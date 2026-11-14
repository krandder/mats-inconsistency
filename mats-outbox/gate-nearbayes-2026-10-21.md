# Gate: nearbayes push (inconsistency-search, branch nearbayes @ 66aedaa), 2026-10-21

Adversarial gate for opus-mats. Bar: Kelvin's "comparable to what a transformer would learn looking at everything" and "we have to beat extremizing".
Checked: README, 198 cell records (runs/ JSON + logs/ RESULT lines, loaded with nearbayes_table.load), aggbench/examples/nearbayes.py, git ancestry of every recorded `commit`.

**Verdict: usable as an EXPLORATORY result: yes, once the must-fixes below are done. It is not a parity result for E3 or Z8.**

## Holds (evidence)
1. Inputs: the flat token is `full_context(b, vis, alignment, own_model=True)` (nearbayes.py:87). Piece cells come from `agent_inputs(..., own_model=True)` (nearbayes.py:126). Both are unaligned by default (`cfg.get("alignment","unaligned")`, :471).
   At keys none, the key width is 0 in E2/Z6/Z8/Z9/Z1 (checked on 50 dev episodes). Under unaligned inputs, keys full == own in E2/Z6/Z9, so "full" there is a duplicate-feature control, as the README says.
   The one aligned cell (E2-full-...-alignmentaligned, 0.0029) is labelled. The summary does not select it.
   The flat token carries the trader's report (logit/4, transformer.py:50) next to its raw evidence. The report is a function of the trader's own inputs, so it leaks nothing. Piece cells carry no report.
2. Fixes: 176 of 198 cells ran on commits that contain both 20ee6a0 and eb3cd7d. The other 22 ran at 92cb6e3/46d5563 (E2/Z6 S1-S2 small pools; E2 S2 4M flat, pos=learned, features=exact, aligned).
   20ee6a0 does not change E2 or Z6 own models: `own_model_from_full` returns the full key for E2, and Z6 has no provenance block. So those 22 are not contaminated, but they are unmarked.
   The Z8 cells at 055bc23 predate 2546bb2. They are still correct: final Bayes check maxabs 6.6e-7, and the script used its own closed-form Z8 prefix Bayes (055bc23:nearbayes.py:182-189).
3. Bayes: final Bayes is recomputed from the oracle arrays and asserted against batch.bayes (raises if the difference is > 1e-3, nearbayes.py:493-494). Max over all cells: 1.9e-6.
   The prefix Bayes last position is recorded against batch.bayes (~1.6e-6) but not asserted.
   Splits are disjoint by seed base (splits.py:10, train 0.., dev 10M..; the size guard is at :17). There is no checkpoint selection on dev: the final weights are scored.
4. E2 matched controls: grid_full, grid_tc, fct-pieces, settf and fct-agents are identical in env, pool (4M), steps (48k), batch (1024), lr (7e-4), width x depth (128x4), heads, amp and dev set. Parameter counts are 794k (settf 423k).
   grid_tc and fct-pieces seed1 ran at the same commit (43593ab).
5. Memorisation: in the headline cells, train/dev KL agree. Train loss_ema is close to dev log loss (grid_full 0.276/0.282, fct-pieces 0.275/0.277, Bayes 0.274). The memorised small-pool cells are reported as such.
6. The "All cells" table is generated: rerunning nearbayes_table.py over runs/ and logs/ reproduces all 200 lines byte-identically. Spot checks, all equal to the JSON:
   grid_tc 0.0048 [0.0046,0.0050]; fct seed2 0.0021; E2 S3 flat 0.0137; Z8 full grid_full 0.0060; E3-collude DeepSets 0.0089; Z9-hard 0.0007; E4 S2 0.0258 (README summary 0.0232 is every_prefix, labelled); ISAB Z9 N128 m4 0.2778; E2 mixed-N@64 0.0049; Z6 mixed-N@32 0.0028.
7. Distill and features=exact carry "USES ORACLE" in the table and the summary. "Below" is computed in one place (ci_high < 0.01 and < 0.05*gap, nearbayes.py:599) and used consistently (Z9-hard target 0.0040).

## Must-fix
M1. E3-sybil, E3-collude and Z8-full have no per-piece fct bar. Their "trader below target, bar above" rows compare per-piece trader tokens with per-agent bar tokens.
    That is the confound the README itself measures on E2 (fct per-agent 0.0191 vs per-piece 0.0026). Run fct-pieces on E3x2 and Z8 (none and full) at 48k, 3 seeds, or drop the trader-vs-bar framing for those rows.
M2. Add a parity column (trader minus matched bar, CIs) to the summary. Under Kelvin's criterion:
    - E2 per-piece: the trader misses. grid_full is 0.0082+-0.0005 vs bar 0.0026+-0.0005, 3 seeds each, about 3x.
    - Z9 k=6 noisy: the trader misses. It stays at the prior, 0.371, vs bar 0.0003.
    - Z6 with mixed N, tested out of range: the trader is about 4x worse (0.0028 vs 0.0007).
    Today the "(below)" marks read as success.
M3. The summary picks the best single cell of each arm, by seed and lr, on the same 20k dev set it reports. Examples: bar fct seed 2 (0.0021 of 0.0021-0.0030); grid_tc, which has 1 seed; Z9 k=6 trader seed 1 at lr 3e-4.
    Report seed means and say the selection is on dev. grid_tc needs 2 more seeds before "causality cost 0.0048 vs 0.0026" is claimed.
    That gap also mixes causality with readout: fct reads a CLS token, grid_tc reads the last trader's last slot, often a padded cell (trader_transformer.py:232).
M4. The Summary, ISAB, NSCALE and BC tables say "(generated)", but no generator is committed (grep finds none). Commit the generator or drop the word.
    Hand-typed prose disagrees with the tables:
    - 0.0026 "+-0.0004" vs the BC table's +-0.0005 (population vs sample sd);
    - feedback +-0.0002 vs +-0.0001;
    - recurrent +-0.0006 vs +-0.0005;
    - E2 N=8 fct "0.0188" vs 0.0187.
M5. Diagnosis 10, "E2 per-piece: m = 4 matches full attention at every N", is false:
    - at N=8, ISAB m4 0.0143 vs fct 0.0077;
    - at N=32, ISAB m4 0.0116 [0.0112,0.0120] vs SAB 0.0095 [0.0091,0.0098].
    It holds only at N >= 128.
M6. The README claims the torch prefix Bayes matches the harness's prefix_bayes_logits within 5e-6 at every position. No committed code or log shows this check. Commit it as an assert or a test.
    Distill targets and the prefix-KL diagnostics depend on it.

## Should-fix
- Tag the 22 pre-20ee6a0 cells with their commit in the table.
- Fix the title (it omits Z9, E3, E4).
- Fix "Z1's (pre-eb3cd7d)": the current Z1 full key (6 dims) differs from own (3 dims).
- Soften verdict-like headers: "Stuck arms are optimisation failures" (Z9 k=6 noisy is unresolved); "Tokens are what matter".
- Extremizing does not appear anywhere. These cells have no compression, so it is not the binding bar here. Still, add the extremized log-odds pool as one closed-form row per env, so the reader sees what the learned models beat.
- The "fresh bar" column comes from a different regime and a 2000-episode dev set: keep it out of any comparison.
