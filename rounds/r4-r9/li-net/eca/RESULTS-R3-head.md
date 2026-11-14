# E-CA round 3: a zero-overlap cellular-automaton grid, a depth gate, and stiff-first versus random search: results

eurisko study, 2026-10-15 (round 3, 09:27-13:xxZ). Written by an Opus 5.5 subagent for eurisko. Asked by Kelvin ("Please run
new experiments with gemini; we didn't like the previous results"); spec by gemini
(`/home/agent/workspaces/mats/outbox/overnight-followups-round-3.md`, Rank 2), dispatched by opus-mats (mail #3649).
Pre-registration: `PREREG-R3.md` (drafted 09:46Z; its final version, after a header-line edit, was saved at 09:47:03.48Z, 82 ms
before the smoke's source copy and 18 minutes before the main launches; a copy, `/tmp/eca/PREREG-R3-prelaunch.md`, md5
da122736, was taken in between; unchanged since. The 09:46Z draft time cannot be verified: the edit created a new file).
Numbers: `results/r3/summary.json`, `results/r3/tables.md`, re-derivable from `results/raw/r3/`. Audit: `AUDIT-R3.md` (no
high-severity finding; every verdict reproduced; its wording findings are applied below, Sec. 11).

## 1. TL;DR

- **The overlap is gone, and there is a real gap.** At W = 12, T = 10 no validation or test grid shares its row 0 with
  training (0.000), and the one-pass network has a real multi-step gap (round 3 changed the overlap, W, T and the depth at
  once, so the gap cannot be attributed to the overlap removal alone): rule 110 one-pass KL on validation is F 0.13, B1 0.29, B2 0.24 nats at 4 layers,
  while its one-step check is 0.006. **R3-D0 PASSES for rule 110 (L = 4 selected).**
- **Rule 90 fails the gate at every depth.** Its seed-0 run at L = 4 misses the one-step bar by 0.0003 (S1 KL 0.0153 vs
  0.015; that seed stopped early on patience); L = 2 and 3 miss it by more. Per the pre-registration, rule 90's search rules
  are reported as descriptive only. (Its 3-seed mean at L = 4 would have passed, 0.0108: a near miss, verdict unchanged.)
- **R3-D2a FAILS for rule 110.** On forward queries (rows 6-9 from row 0) under the anchored pool at M = 16, stiffness-first,
  confidence-first and hybrid search each cut KL by 13-15% (bar 25%), though each beats random order by 15-17% with CIs above
  0 (random order gains nothing on F). **R3-D2b FAILS:** the hybrid cuts F, B1, B2 by 12-14% and almost never branched (rule
  110: one branch step in all test chains; some unrevealed cell always had H < 0.20 nats), so it behaved like
  confidence-first.
- **R3-D3 FAILS for rule 110** (and rule 90, descriptive): the stiffness head (Spearman 0.700 with -KL(P* || q)), plain
  confidence (0.697) and 1-step self-consistency (0.697) rank the network's own errors equally well; best minus confidence
  +0.002 (bar 0.05). The comparison is close to degenerate: the head ranks pairs almost exactly like plain confidence
  (Spearman 0.97-0.99 between them), and plain confidence even tracks the exact information gain slightly better (0.937 vs
  0.926).
- **Secondary (not a rule):** VoI among the 6 stiffest cells is the strongest forward search (rule 110 F -31%, paired CI
  above 0; rule 90 F -73%, descriptive, CI includes 0); random order is the strongest on rule-110 bridges (B1 -24%, B2 -23%, CIs above 0);
  VoI among the most uncertain cells is harmful under the plain mixture and neutral under the anchored pool (0%). The arms the
  rules name (stiff, conf, hyb) improve on one pass, but by less than the pre-registered bars; VoI over stiff cells (31%) is
  above the 25% bar but is not one of them.
- **Cost:** R3_COST_SHORT for round 3 (cap $40); night total for E-CA R3_TOTAL; zero eca VMs left.
