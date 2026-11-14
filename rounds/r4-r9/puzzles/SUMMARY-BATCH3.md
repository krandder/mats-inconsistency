# Trader-economy puzzles, batch 3 (P13-P16): does the economy do any work?

For Kelvin. Sources, read 2026-09-30: `CATALOGUE-PUZZLES.md` entries P13-P16 (including the verification notes for P14 and P16) and each experiment's `RESULTS.md`. Two independent verifiers checked each experiment and agreed with every status below. See `SUMMARY.md` for P0-P7 and `SUMMARY-BATCH2.md` for P8-P12. P17-P19 and the batch-2 follow-ups are catalogued separately.

Spend: about $64 estimated (P13 $37.52, P14 $7.69, P15 $9.26, P16 $9.82). At the published or ceiling rates it is about $80. P13 comes to about $47 at the real spot price, which is over its $40 internal cap. Nothing was read from billing, because the Billing API is off.

## Bottom line

Four experiments, zero gains. Across the three batches: sixteen experiments, zero gains.

| Outcome | Experiments |
|---|---|
| No gain | P13 (market ResNet on mazes), P14 (value-priced event publishing), P15 (local EM on a tied automaton) |
| Problem (inconclusive, undertrained) | P16 (meta-trained reader values a new bit) |

In every experiment the economic part was either inert or never tested. P13's wealth cap never binds once the net is any good. In P14 the market only adds churn, and the gradient estimator underneath it is the worse one. P15 tests EM, not a market. P16 tests a valuation tool, not a mechanism. The external bar has not moved: the released DT-recall checkpoint solves 94% of 99x99 mazes, and every trader model solves 0%.

## Per experiment

- **P13, market ResNet on mazes: 0.00 at 99x99 in 9 of 9 trader models.**
  - DT-recall retrained at equal compute scores 0.21, with seeds 0 / 0.83 / 0 / 0.
  - Difference -0.21, Welch 95% CI [-0.87, 0.45].
  - The traders learn 9x9 and then do not extrapolate at all: at most 0.04 at 33x33.
  - The cap scales 0% of trades after the first 10% of training, with 0 respawns, and cap-on equals cap-off. The economy reduces to local per-step losses.
  - The runaway fix bounds the logits, but calibration is no better than a post-hoc temperature.
- **P14, value-priced event publishing: V finds a sufficient bit in 3/20 seeds against 17/20 for tuned sparse backprop.**
  - Difference -0.70 [-0.84, -0.40].
  - The one significant effect runs against our estimator (LEG worse, Holm p = 0.0002).
  - The fee grid saturates, so the market was only ever tested at maximum churn. Wealth ("money") fails as a regulator (S5).
- **P15, local EM: ties backprop at depth 16 but falls behind at depth 32.**
  - At depth 32 it scores 0.734 [0.50, 0.97] and is exact in 1/5 seeds, against 4/5 for backprop (Fisher p about 0.10).
  - The log-loss gap grows: 0.10, then 0.39, then 0.41 nats per cell.
  - It never recovers the exact rule (0/10), yet it extrapolates to 1024 bits as well as backprop does.
  - It needs about 93x the memory of checkpointed BPTT.
  - The maze half is void because a gate failed.
- **P16, meta-trained reader: undertrained.**
  - ID AUROC is 0.711 against a 0.80 bar.
  - Read past that, it loses to a tuned Bayes baseline: -0.040 [-0.061, -0.020].
  - The runner's one positive claim, that "FR's value is in the right units", was an artifact: C4's forward gain alone has slope 1.18.

## Process, harshly

- **Failed gates still did not stop the runs, now in a third batch.**
  - P15 ran its maze arm after G3 failed.
  - P16's precondition failed.
  - P13's active-economy ladder found no qualifying setting, and the run went on without it.
- **An arm was cut without disclosure.** P13's pre-registered rule for a >2x FLOPs gap fired: the traders got 0.30x the training FLOPs of DT-recall. The DTR-eqflop arm was then dropped silently.
- **The runners misreported facts, and the verifiers caught them.**
  - P13: the poorest trader's wealth (about 10k at the end, not 503-559), and CIs pasted from the wrong column.
  - P15: memory 93x, not 20x; the G3 explanation was wrong; the decay setting and test-set sizes were wrong.
  - P16: 13 failed steps, not 1.
  - P14: its checksum file covers only half the evals.
  - Verification stays mandatory.
- **The cost caps are soft.** P13 was checked against a prescribed rate, not the real one, and the real price broke its cap.
- **The nulls are mostly weak.** DT-recall's seed spread makes P13's CI span 1.3. P15's decisive depth-32 cell has 5 seeds.

## What next

1. **Stop using the market as a training mechanism for depth or extrapolation.** In sixteen experiments it has never bound where it mattered or beaten a known method. Keep prices as diagnostics only.
2. **Any further economy run needs a binding gate that stops the run automatically.** The cap must bind on at least 10% of trades at the trained operating point before any comparison is made. Seeds must come from a power calculation, and the cost must be checked at the real price.
3. **One cheap lead, all of it post hoc.** Averaging P16's reader with C4's reverse term gives 0.758 against 0.745. A reader that also amortizes the reverse direction would be a new pre-registration at about $10. It is a valuation tool, not an economy.

**Decision needed from Kelvin:** close the trader-economy-as-training line, or fund one binding-gated rerun of P13 (about $40-50).
