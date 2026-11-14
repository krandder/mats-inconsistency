# P19 results: does a fast reader's prequential gain rank candidate intermediates?

**Status: Killed (K1 and K2), by the frozen rule in `PREREG.md`.** All three gates passed on all 5 test seeds.

- **K1 fires on tau.** The fast reader (FR) ranks the 44 maze items at Kendall tau 0.576 [0.526, 0.627]. The kill
  line is 0.70. Its compute is under the line: 0.040 of one wrapper replicate per item, against a limit of 0.05.
- **K2 fires, barely.** FR beats the first-order price (FO) by 0.046 [0.011, 0.081], paired over seeds. The gap is
  real but below the pre-registered 0.05 margin.
- **Neither cheap price is close to the wrapper.** R7b is P8's 80/20 wrapper: 0.866 with 4 replicates, 0.836 with 1.
  Plain importance (R1, a fully trained reader) reaches 0.674, which beats FR by 0.097. It costs about 27 times as
  much per item.

THEORY-3's reading of K1 is "proposing must be done by search". Its reading of K2 is "the fast reader is
unnecessary; boosting's price suffices". The first half holds only for this fast reader. THEORY-3's reader is
meta-initialised or in-context, and this one is neither (see Scope). The second half needs care: first-order pricing
does not *suffice*. It only matches this fast reader to within 0.05, and both are far below 0.70.

**Scope.** P19 bounds *this* fast reader: a warm-start copy of the current reader with a zero-initialised new
channel, trained with plain Adam for 64 chunks. A meta-learned or in-context reader was not tested; P16 is that
direction.

## Setup (details in `PREREG.md`)

**Task and items.**

- The task is P8's maze ladder: 11×11 mazes, Y = [BFS distance ≤ 8], N = 1024, and two contexts: empty, and
  g2g4 (g2 and g4 already deployed).
- There are 44 ranked items: P8's pool, plus x2, x4 and x6 (g_k XOR key, where key is an input), xb4 (g4 XOR a key
  that is never an input), pg4 (the next maze's g4) and fg4 (1 − g4).

**Truth.** The deployed gain over base (P8/item3 `job_truthrep`): predict z from X by cross-fitting, then train the
Y reader on X, Ŝ and ẑ. It is the mean over 2 inits × 2 replicates.

**Rules.**

| rule | what it is | replicates | compute per item (× one R7b replicate) |
|---|---|---|---|
| FR | fast reader: prequential gain of adding the true z to a warm-start copy of the current reader; 64 chunks of 16 mazes, Adam at 3e-3; mean of 2 stream orders | 2 orders | 0.040 |
| FO | first-order price: ridge logistic offset on the current reader's logit, using a 3×3 neighbourhood of the centred z; 2-fold cross-fit | 1 | 0.024 |
| R4 | P8's price: L_Y − L_YZ − L_Z | 2 (Deviation 1) | 1.06 (shared with R1) |
| R1 | importance alone: L_Y − L_YZ | 2 (Deviation 1) | 1.06 (shared with R4) |
| R7b | P8's 80/20 wrapper, with a predicted z | 4 (Deviation 1) | 4.0 |

- FR, FO, R1 and R4 all price the *true* z. R7b and the truth use a z predicted from X.
- The pilot rule fixed LR\* = 3e-3 and (K\*, T\*) = (2, 64), truth-free on seed 99, before the freeze.
- Compute is forward MACs, P8's convention. FR's figure includes training both current readers (0.018 by
  themselves).

**Statistic.** Kendall tau-b against the truth over the 44 items, for each seed. Aggregates are the mean and 95%
t-interval over seeds 0 to 4. The freeze was at 01:30:06Z (`PREREG.sha256`).

## Primary results (5 seeds)

| rule | tau (95% CI) | FR − rule, paired (95% CI) | per-seed tau |
|---|---|---|---|
| **FR** | **0.576 ± 0.051** | | 0.550 0.575 0.558 0.552 0.647 |
| FO | 0.530 ± 0.040 | **+0.046 ± 0.035** | 0.521 0.557 0.526 0.483 0.564 |
| R1 | 0.674 ± 0.033 | −0.097 ± 0.032 | 0.649 0.700 0.651 0.664 0.704 |
| R4 | 0.293 ± 0.036 | +0.284 ± 0.081 | 0.315 0.324 0.288 0.288 0.250 |
| R7b | 0.866 ± 0.037 | −0.289 ± 0.068 | 0.884 0.816 0.867 0.892 0.869 |

**Gates.** Every gate passed on every seed.

- **G1.** Truth A/A Spearman ranged from 0.914 to 0.955.
- **G2.** FR(null) and FO(null) are exactly 0.
- **G3.** FR(copyY) beats every ranked FR in the empty context: 48,616 to 51,472 bits across seeds, against
  44,848 to 47,616 for the next highest item (nz05) on the same seed.

**Verdict inputs** (`results/p19_summary.json`, `verdict`): tau_FR 0.5763, FR compute ratio 0.0397, tau_FR −
tau_FO 0.0460, so K1 = true and K2 = true.

## Secondary results (pre-registered, not decisive)

**More orders or more steps for FR.**

| FR variant | tau | compute ratio |
|---|---|---|
| T = 16, 4 orders | 0.505 | 0.029 |
| T = 32, 4 orders | 0.549 | 0.040 |
| T = 64, 4 orders | 0.577 | 0.061 |
| T = 64, 1 order | 0.573 | 0.029 |

- More stream orders do not help: 1, 2 and 4 orders give 0.573, 0.576 and 0.577.
- **Longer prefixes rank better.** Budget T means the gain summed over the first T chunks: 256, 512 and 1,024
  mazes for T = 16, 32 and 64. The gains per doubling shrink: +0.044, then +0.028. T = 64 is the whole stream, so
  P19 cannot extend the trend at N = 1024.

**The forward subset** drops nz05 and nz20, which are functions of Y. The picture is the same:

- FR 0.577, FO 0.529, R1 0.667, R4 0.349, R7b 0.853.
- FR − FO: 0.048 ± 0.042.

**Against the control truth** (gain against the free-latent control, P8's other truth), FR and FO tie: FR 0.551, FO
0.552, R1 0.653, R4 0.333, R7b 0.823.

**Per context** (tau within the context):

| context | FR | FO | R1 | R4 | R7b |
|---|---|---|---|---|---|
| empty | 0.538 | 0.528 | 0.586 | 0.379 | 0.916 |
| g2g4 | 0.501 | 0.471 | 0.678 | 0.183 | 0.811 |

**XOR-type candidates.** The truth is gain in bits per held maze, the mean over seeds. For scale, empty:g4 is 21.3.

| item | truth | FR | FO | R7b |
|---|---|---|---|---|
| empty:x2 | 5.4 | −145 | −2 | 4,185 |
| empty:x4 | 12.8 | −46 | +2 | 14,093 |
| empty:x6 | 10.7 | +465 | −14 | 10,813 |
| g2g4:x2 | −0.1 | −1,424 | +1 | −153 |
| g2g4:x4 | −0.2 | −1,424 | +2 | −418 |
| g2g4:x6 | 14.0 | −1,409 | +1 | 14,962 |

- **Mean absolute rank error on the XOR items (out of 44):** R7b 2.5, R1 8.0, R4 9.3, FR 9.5, FO 11.5.
- **THEORY-3's prediction for FO holds.** FO prices every XOR item at about 0 bits, against about 15,000 for g4.
  Seed means are at most 14 in absolute value; single seeds reach 40. FO has zero first-order value there by construction.
- **This FR does not fix it.** In 64 warm-start chunks it does not learn the XOR with the key, so it prices the
  valuable XOR items at about 0 or below. By mean rank, g2g4:x6 is 16th by truth and 41st by FR; empty:x4 is 18th
  by truth and 30th by FR.
- **Only the wrapper prices the XOR items well.** Like the truth, R7b trains a predictor of z from X (which
  includes key) and deploys its output.

**fg4 against g4** (the relative gap (fg4 − g4) / |g4|):

- The truth sees them as nearly equal: −0.04.
- FO is exactly invariant, 0.00, as designed.
- **FR is not invariant: −0.47.** It prices 1 − g4 at about half of g4. A zero-initialised channel on an uncentred
  0/1 input depends on the coding, which is a design flaw of this FR.
- R1 −0.18, R7b −0.07.

**Near-zero controls.** xb4 and pg4 have truth ≤ 0.24. Every rule gives each of them a mean rank of 30 or lower
(out of 44).

## Exploratory (not pre-registered; `code/explore19.py` → `results/p19_items.json`)

**Largest FR misrankings** (mean rank, truth → FR):

- g2g4:x6 16 → 41
- empty:x4 18 → 30
- empty:g3 3 → 14
- g2g4:g6 1 → 12
- empty:reach 27 → 16
- g2g4:m2 38 → 28
- g2g4:nz05 19 → 9
- empty:nz20 14 → 4

**FR, FO and R1 share most of these errors; R7b does not.** On g3, g6, reach, m2 and the nz items, FO and R1 each
rank every item within about 6 places of FR. R7b is within 2 places of the truth on every one of them.

**A hypothesis, untested.** FR, FO and R1 price the true z, while the truth deploys a z predicted from X. The
remaining gap to 0.70 may be about learnability and deployment (how well z can be predicted, and where), not
about a slow reader. P8's learnability charge (R4 = R1 − L_Z) does not capture it; it lowers tau from 0.67 to 0.29.
A cheap rule priced on a predicted z (FR or FO applied to ẑ) is the obvious next test.

## Disclosures

- **The brief's section references.** The brief cites "THEORY-3 §6(b) and §4", but THEORY-3 has only §§1 to 5. The
  experiment is §3 option 3 plus its "Cheapest decisive experiment" (see PREREG, Deviations item 3).
- **Bornschein, Li & Hutter 2022** (arXiv 2210.07931) is real. It recommends online learning with rehearsal for
  prequential MDL. It does not speak of "fast learners". FR here uses replay, which follows that paper.
- **Pilot sanity.** After the freeze I ran `code/pilot_sanity.py` on the pilot seed 99, which is never a test seed.
  - Its pilot taus: FR 0.611, FO 0.536, R7b with 1 replicate 0.774, and truth A/A 0.936.
  - It changed nothing that was frozen.
- **Deviations** (dated, in `DEVIATIONS.md`). None changes the verdict inputs (FR, FO, the compute ratio) or a kill
  rule. Deviation 1 does lower the replicate counts behind the comparator taus (R1, R4 and R7b):
  1. Fewer replicates for the comparators because of cost: R7b 12 → 4 and pricerep 8 → 2. R1, R4 and R7b are
     therefore noisier than in P8. R7b comes out somewhat below P8's value: 0.866 here (CI 0.829 to 0.902, which
     excludes P8's 0.91).
  2. A JSON serialization bug fix in `analyze19.py` (numpy bool), found in a dry run on synthetic data before any
     test result existed.
  3. The spot VM was preempted at 01:59Z before any test-seed job finished. The rerun used an on-demand VM (same
     ceiling rate) with the 45 needed jobs at 2 threads each. `main2.txt` also lists the jobs in a different order
     from `main.txt`, which has no effect on any output.
- **What was read from item3.** Before the freeze I read item3's PREREG, DEVIATION-1, code, VM ledger and job
  start and finish times (`vmout/logs/queue.log`), to reuse its code and estimate CPU time. I read no item3 result
  file, then or later. PREREG's list of what was seen before the freeze omits this.
- **Compute prose.** PREREG says FR compute counts "all members". The frozen code counts the base plus the ranked
  members, not the 5 sanity members (copyY, random, null, open, startZ). Counting them gives a ratio of 0.042,
  still under 0.05.

- **Independent verification** (04:01Z). A separate read-only check recomputed the primary numbers with its own
  script: tau_FR 0.5763, tau_FO 0.5303, FR − FO 0.0460, compute ratio 0.0397. Nothing differed from the summary by
  more than 0.001. It confirmed the freeze (only the Deviation 2 change) and agreed with "Killed (K1 and K2)". Its 9
  minor findings (wording, one omitted misranking, and the two disclosures above on item3 and compute) are fixed.

## Cost

$2.85 at the $1.20/h ceiling: 2.38 VM-h in total.

- trader-p19-1 (spot, preempted): 0.71 h, $0.85.
- trader-p19-2 (on-demand): 1.67 h, $2.00.

Both VMs and their disks are deleted. At 03:49Z, gcloud's instance and disk lists for `trader-p19` were empty.
The rows are in `../../GPU-LEDGER.md` (P19 section) and `calibrated-ai/BUDGET-24H-20260929.md`.

## Files

- `PREREG.md`, `PREREG.sha256` (the freeze), `DEVIATIONS.md`
- `results/p19_summary.json`: every statistic above
- `results/p19_chart.json`: tau against compute per rule, the FR budget points, per-seed taus, replicate curves
  and the kill lines
- `results/p19_items.json`: exploratory per-item means and ranks
- `results/pilot/`: the pilot rule and pilot sanity
- `results/vmout/res/`: raw job outputs, sha256-verified
- `code/p19.py`, `code/analyze19.py`, `code/pilot_rule.py`, `code/pilot_sanity.py`, `code/explore19.py`
- `vm.sh`, `vm/LEDGER.md`
- `queues/main.txt` (frozen) and `queues/main2.txt` (what ran)

Reproduce with `cd code && P19_KREP_W=4 P19_KREP_P=2 python analyze19.py ../results/vmout/res`. That writes to
`results/vmout/`; the copies in `results/` also carry the compute fields added to the chart after the run.
