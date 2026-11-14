# RESULTS-R27C (frozen `freeze-R27C`, run by opus-mats on General CPU)

> **Audit (docs/audits/R27C-audit.md, board-opus, 2026-10-20 11:12Z):** D0 supported (a regression check against OpenSpiel's own output only); D1 partly supported; D2 artefact. Provenance passes, 0 number mismatches, seed 101 re-run bit-identical.
>
> **What the arm is:** exact-tree counterfactual Q (true payoffs, all infosets) distilled into an MLP by proximal MMD steps. It has no world model, sampling or search. It tracks tabular MMD at the same number of stages (difference +0.005 / +0.019 / +0.019) and adds nothing beyond it. The informative comparison is the proximity edge: NoProx is much worse on all three games. The prior (tau) edge is never ablated.
>
> **D2 follows from its objective:** each arm wins against the opponent it was trained on. The alpha=1 payoff against the reader (-0.0633) is near the Kuhn game value (-1/18). Nothing about the value of commitment is shown (the game is zero-sum).

- Runner sha256 `7dbb67339ad9b482cc1a7fc960279455db9b3790abd153828545cf77f7077f59`, PREREG sha256 `a28f4bae4379631ec129811cfe5fc879a991f83a55ef37178b1a7e55060e154b`, OpenSpiel `2.0.2`, device `cpu`, seeds `[101, 102, 103, 104, 105]`, elapsed 485 s.
- Source of every number: `results/r27/r27c_results.json` (path given per row); verdicts: `results/r27/r27c_verdicts.json` (scripts/score_r27.py).

## Verdicts (as computed by the frozen gate code)

| Gate | Verdict |
|---|---|
| `R27C_D0_upstream_openspiel_exact_curve_reproduction` | **PASS** |
| `R27C_D1_proximity_edge_prevents_cycling_and_beats_baselines` | **FAIL** |
| `R27C_D2_neural_policy_commitment_readability_and_stability` | **FAIL** |

## D1: NashConv by game (5-seed means)

| Game | Ours (prox+prior) | Tabular same stages | NoProx ablation | Untrained | Random net | Uniform | MCCFR (our run) | PREREG bar on ours | Bar met |
|---|---|---|---|---|---|---|---|---|---|
| kuhn_poker | 0.0428 | 0.0381 | 0.3352 | 0.9215 | 0.9096 | 0.9167 | 0.2296 | <= 0.05 | yes |
| leduc_poker | 1.0809 | 1.0615 | 2.2349 | 4.7491 | 4.7460 | 4.7472 | 4.5774 | <= 0.95 | no |
| liars_dice(numdice=1,dice_sides=4) | 0.4546 | 0.4354 | 1.1659 | 1.3119 | 1.3135 | 1.3101 | 1.2290 | <= 0.12 | no |

Paths: `gates.R27C_D1_proximity_edge_prevents_cycling_and_beats_baselines.by_game.<game>.mean_nash_conv_*`. All paired 95% CIs against every control and against NoProx exclude zero on all three games (`all_games_beat_controls_and_noprox = True`); D1 fails on the absolute Leduc and Liar's Dice bars. The PREREG states that the Liar's Dice bar fails by construction: the exact tabular QRE target at tau_base itself has NashConv 0.485.

## D2: policy commitment against a reader (Kuhn, labelled `objective_alignment_verification`)

| Quantity | alpha_read = 0 | alpha_read = 1 |
|---|---|---|
| Payoff vs fixed biased opponent | 0.3621 | 0.1601 |
| Payoff vs reader (soft best response to own policy) | -0.1507 | -0.0633 |
| NashConv | 0.2572 | 0.0691 |

Max fixed-point L1 drift at alpha 1: 0.0197 (bar <= 0.01). NashConv at alpha 1 bar <= 0.05. Paths: `gates.R27C_D2_neural_policy_commitment_readability_and_stability.*`. The three paired comparisons have CI lower bounds above zero; D2 fails on the absolute NashConv and drift bars.

## D0

CFR/CFR+ curves match OpenSpiel 2.0.2 (max error 0.0); tabular MMD inconsistency <= 1e-8 on all games; oracle vs OpenSpiel value error 1.29e-11. D0 is a regression check against OpenSpiel's own output, not a published curve.
