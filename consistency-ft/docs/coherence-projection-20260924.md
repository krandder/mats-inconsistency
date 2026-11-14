# Is the 235B all-in-one gain due to coherence? (2026-09-24)

Question (Kelvin): asking all 12 forms in one call improved Qwen3-235B test log loss
from 0.553 (one form per call) to 0.530 (all in one call), both with thinking off.
Is that gain due to coherence?

Test: project each pair's 12 stated probabilities onto the nearest coherent set: fit a
distribution q over the 4 atoms (P∧Q, P∧¬Q, ¬P∧Q, ¬P∧¬Q), then read all 12 forms from q.
Three fits: least squares on probabilities (ls_prob), least squares on log-odds
(ls_logit), and minimum sum of Bernoulli KL(stated‖implied) per form (kl). The fit uses
no labels. Scoring: `scripts/score_verbalized_baseline.score` (clip 0.01, missing imputed
at 0.5). CIs: paired bootstrap over the 1,066 test pairs, 2,000 resamples, seed 20260924.
No API calls. Code: `scripts/coherence_projection.py`; full numbers (observed LL and Brier
CIs too): `docs/coherence-projection-20260924.json`.

Inputs: 235B per-form `precutoff-baseline-20260923/per-form/predictions-test.jsonl`;
235B all-in-one thinking off `precutoff-baseline-20260923/qwen3-235b-a22b-nothink/predictions-test.jsonl`
(26 missing forms, imputed 0.5 in scoring, left out of the fit); 32B
`qwen32b-baseline-20260923/remote-out/predictions-test-v6_direct.jsonl`; 0.6B
`qwen06b-fixed-20260923/results/<arm>/predictions-test-best.jsonl` (five best arms).

## Results (test)

| run | fit | all-12 LL | observed LL | all-12 Brier | gain vs raw, all-12 LL (95% CI) | vs 235B all-in-one raw, all-12 LL (95% CI) |
|---|---|---|---|---|---|---|
| qwen3-235b per-form (thinking off) | raw | 0.5529 | 0.5484 | 0.1852 | — | +0.0232 [+0.0152, +0.0317] |
| | ls_prob | 0.5142 | 0.5232 | 0.1705 | -0.0387 [-0.0409, -0.0366] | -0.0155 [-0.0237, -0.0068] |
| | ls_logit | 0.5165 | 0.5258 | 0.1713 | -0.0364 [-0.0386, -0.0342] | -0.0132 [-0.0214, -0.0045] |
| | kl | 0.5158 | 0.5249 | 0.1711 | -0.0371 [-0.0394, -0.0348] | -0.0139 [-0.0221, -0.0052] |
| qwen3-32b v6_direct per-form | raw | 0.6114 | 0.6155 | 0.2063 | — | +0.0817 [+0.0701, +0.0944] |
| | ls_prob | 0.5457 | 0.5558 | 0.1824 | -0.0657 [-0.0711, -0.0609] | +0.0160 [+0.0047, +0.0279] |
| | ls_logit | 0.5399 | 0.5511 | 0.1810 | -0.0714 [-0.0765, -0.0674] | +0.0102 [-0.0006, +0.0218] |
| | kl | 0.5409 | 0.5513 | 0.1814 | -0.0705 [-0.0758, -0.0660] | +0.0112 [+0.0003, +0.0230] |
| 0.6B A8_full_consistency_l1 | raw | 0.5876 | 0.5909 | 0.2008 | — | +0.0579 [+0.0435, +0.0720] |
| | ls_prob | 0.5835 | 0.5842 | 0.1987 | -0.0041 [-0.0045, -0.0038] | +0.0538 [+0.0393, +0.0679] |
| | ls_logit | 0.5815 | 0.5824 | 0.1978 | -0.0061 [-0.0065, -0.0057] | +0.0518 [+0.0375, +0.0659] |
| | kl | 0.5827 | 0.5835 | 0.1984 | -0.0049 [-0.0052, -0.0047] | +0.0530 [+0.0386, +0.0671] |
| 0.6B A7_full_accuracy | raw | 0.5874 | 0.5917 | 0.2005 | — | +0.0577 [+0.0432, +0.0718] |
| | ls_prob | 0.5823 | 0.5833 | 0.1982 | -0.0051 [-0.0055, -0.0047] | +0.0526 [+0.0381, +0.0669] |
| | ls_logit | 0.5809 | 0.5819 | 0.1975 | -0.0065 [-0.0069, -0.0061] | +0.0512 [+0.0367, +0.0654] |
| | kl | 0.5817 | 0.5827 | 0.1979 | -0.0056 [-0.0059, -0.0054] | +0.0520 [+0.0375, +0.0663] |
| 0.6B A4_one_consistency_l3 | raw | 0.5901 | 0.5956 | 0.2013 | — | +0.0604 [+0.0452, +0.0749] |
| | ls_prob | 0.5788 | 0.5805 | 0.1966 | -0.0113 [-0.0119, -0.0108] | +0.0491 [+0.0339, +0.0638] |
| | ls_logit | 0.5802 | 0.5823 | 0.1972 | -0.0099 [-0.0105, -0.0093] | +0.0505 [+0.0351, +0.0653] |
| | kl | 0.5797 | 0.5816 | 0.1970 | -0.0103 [-0.0109, -0.0098] | +0.0500 [+0.0347, +0.0648] |
| 0.6B A3_one_consistency_l1 | raw | 0.5957 | 0.6010 | 0.2038 | — | +0.0659 [+0.0508, +0.0807] |
| | ls_prob | 0.5796 | 0.5814 | 0.1969 | -0.0161 [-0.0169, -0.0153] | +0.0499 [+0.0347, +0.0645] |
| | ls_logit | 0.5809 | 0.5832 | 0.1975 | -0.0148 [-0.0156, -0.0140] | +0.0512 [+0.0360, +0.0659] |
| | kl | 0.5806 | 0.5827 | 0.1974 | -0.0150 [-0.0158, -0.0143] | +0.0509 [+0.0357, +0.0655] |
| 0.6B A1_one_accuracy | raw | 0.5967 | 0.6027 | 0.2042 | — | +0.0670 [+0.0518, +0.0817] |
| | ls_prob | 0.5796 | 0.5822 | 0.1970 | -0.0171 [-0.0180, -0.0162] | +0.0499 [+0.0345, +0.0647] |
| | ls_logit | 0.5806 | 0.5835 | 0.1974 | -0.0161 [-0.0170, -0.0152] | +0.0509 [+0.0356, +0.0656] |
| | kl | 0.5805 | 0.5831 | 0.1974 | -0.0161 [-0.0170, -0.0153] | +0.0508 [+0.0356, +0.0655] |
| qwen3-235b all-12-in-one-call (thinking off) | raw | 0.5297 | 0.5325 | 0.1765 | — | (reference) |
| | ls_prob | 0.5266 | 0.5269 | 0.1755 | -0.0031 [-0.0040, -0.0023] | -0.0031 [-0.0040, -0.0023] |
| | ls_logit | 0.5265 | 0.5275 | 0.1756 | -0.0032 [-0.0041, -0.0024] | -0.0032 [-0.0041, -0.0024] |
| | kl | 0.5264 | 0.5270 | 0.1755 | -0.0033 [-0.0043, -0.0025] | -0.0033 [-0.0043, -0.0025] |

Negative = lower loss. Per-form projected vs all-in-one projected (same fit on both), all-12 LL:
ls_prob -0.0124 [-0.0204, -0.0036], ls_logit -0.0100 [-0.0181, -0.0012], kl -0.0106
[-0.0187, -0.0021]; on observed forms only: -0.004 to -0.002, CIs cross zero.

## Reading

- Yes, coherence explains the gain, with room to spare. Projecting the 235B one-form-per-call
  predictions lowers all-12 log loss by 0.036 to 0.039 (CIs within ±0.002), more than the
  0.023 gain from asking all 12 at once. The projected per-form run beats the raw all-in-one
  run by 0.013 to 0.016 (CI excludes zero) on all 12 forms; on the observed form alone it is
  0.007 to 0.009 better, CI crosses zero.
- The all-in-one answers are already nearly coherent: projecting them gains only 0.003. Even
  after both are projected, per-form stays 0.010 to 0.012 better on all 12 forms. So asking
  all 12 at once buys coherence, but costs something else (probably the independent noise of
  12 separate readings, which projection averages: each atom is read from several forms).
- The three fits agree within 0.003; no fit is clearly best (ls_prob best for 235B, ls_logit for 32B).
- 32B: projection gains 0.066 to 0.071 (0.611 to 0.540), closing most of its gap to the 235B
  all-in-one run (remaining +0.010 to +0.016).
- 0.6B: gains 0.004 to 0.017; smaller for the arms trained on all 12 labels (A7, A8), which are
  already more coherent. After projection all 0.6B arms land at 0.579 to 0.584.
- Practical: a free post-hoc projection is worth more than the all-in-one prompt for 235B
  thinking off. Not tested here: 235B thinking (0.513 all-in-one) and whether the per-form
  advantage after projection holds on validation.
