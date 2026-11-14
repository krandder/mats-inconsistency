# R27B audit (board-opus, Opus 5.5, read-only, 2026-10-20 15:09-15:26Z)

Scripts and outputs: this directory. No GitHub calls.

**1 Provenance (15:09Z).**
- freeze-R27B = 749a4a3. All three sha256 match at the tag, main 367ce7d, /tmp/r27b-run (HEAD 749a4a3, clean) and the JSON.
- Not checkable: the full command (GPU-LOG truncates it). n_test=80 fits the 1/80 granularity of top-1.

**2 Numbers (mdcheck.py).**
- 169 checked, 0 mismatches.
- Missing: a cost line; the generator is outside the repo.

**3 Gates (regate.py, own code).**
- 100 conjuncts. Result D0 P, D1 F, D2 F, D3 P, the same as the frozen code and scorer.
- Failing conjuncts:
  - D1 rule-KL margin vs Helper: overall 0.223 and shiphunt 0.101, bar 0.25.
  - D2 minesweeper vs Untrained: ci_low −0.017.

**4 Oracle.**
- The in-code check is narrow (own arrays, one mask, shiphunt rules 0-1 only).
- Independent enumeration (802/360, sets equal), 80 episodes, all rules: max |dPz| 2.9e-8, TV 1.5e-8. Exact.

**5 Leakage / by construction.**
- The rule label never enters a loss (data flow traced).
- WatchedLabelArray is never read, so its count of 0 is vacuous.
- Disclosed: z supervision, one labelled board per rule.
- **"Learned VoI" = hand formula** (it regresses a fixed mix of its own inputs). Rule entropy, seeds 101-102:

  | Probing by | shiphunt | minesweeper |
  |---|---|---|
  | head | 0.0496 | 0.0962 |
  | formula | 0.0552 | 0.0964 |
  | random | 0.216 | 0.220 |

- **D3 JSD = 0 by construction:** U enters only the policy head; with .detach() removed the belief diff is still 0, so the detach is never tested.
- **Policy ≈ ranking by mu_z.** Seek: 0.515 vs 0.513, 0.448 vs 0.425. The reversal comes from the hand feature (mu_z−0.5)·U.
- No-firewall ablations equal ours on every task metric. NoRuleSlots rule-KL bars duplicate Uniform's (rule_logits = 0).
- CIs cover training seeds only; ours' kl_rule rose from 0.20/0.32 on smoke (where the bars were set) to 0.34/0.52 on test.

**6 Substance.**
- **D1:** ours beats every control on state KL in both environments. Its rule beliefs beat the true-stencil helper by only 0.10 on shiphunt and stay far from Bayes.
- **D2:** gain over random probing is real but formula-driven; fragile on minesweeper (untrained net 0.179, ours 0.108, ceiling 0.000).

**7 CPU re-run (unmodified tag, all 5 seeds, 15:12-15:24Z).**
- Same verdicts, with the same D1 conjuncts (0.213, 0.101).
- D2 also fails minesweeper vs NoRuleSlots (ci_low −0.0015).
- Not bitwise equal to GPU (CUDA RNG).

**8 Kelvin's question.**
- **Shown:** on two synthetic 6×6 grids, a generator-shaped model trained on true hidden state holds better-than-control beliefs over state and rules (rule top-1 0.94/0.84 after 5 cells). Formula probing beats random.
- **Not shown:**
  - beliefs learned from observations alone
  - Bayes-level rule inference
  - learned information gathering: a "most-uncertain cell" heuristic scores 0.012 vs ours 0.050 on shiphunt (it loses on minesweeper)
  - any learned separation of goals from beliefs under utility swaps
- **Unchecked:** split disjointness (in-code counts only).

| Gate | Verdict |
|---|---|
| D0 | supported |
| D1 (FAIL) | supported as a fail |
| D2 (FAIL) | supported as a fail; "learned VoI" is the hand formula |
| D3 (PASS) | artefact (by construction) |

Round: **partly supported**.

The run is genuine and faithfully reported. The model forms better-than-control beliefs over hidden state and rules, but it misses the pre-registered margin over a true-stencil helper and does not probe robustly on minesweeper. Its only passing behaviour claim, unchanged beliefs under utility swaps, holds by design and shows no learning.
