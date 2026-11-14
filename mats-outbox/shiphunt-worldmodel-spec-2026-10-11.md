# ShipHunt round 2: learning a world model for data-efficient forecasting

Owner: eurisko. Agreed with Kelvin 2026-10-11 (opus-mats conversation). Builds on smallnet round 1 (RESULTS.md, champion rbtd8).

## Question
What is the best way to learn a world model from N training episodes and use it to train the forecaster,
measured by single-cell KL to the exact posterior vs number of boards (same sweep and TEST protocol as round 1)?

## Rules (Kelvin)
- **L0 only.** The world model knows only the data format (a probe names a cell and gives a reading). It learns
  everything else from the N episodes: the board distribution, the probe readings, and where probes land.
  No hand-coded noise model, no known flip rate, no ship-shape program class.
- Outcomes only: training uses realised episodes (true board + the probes that happened) and the model's own outputs.
  The exact posterior is for scoring only, as in round 1.
- Anything that would not work on real-world questions is not allowed.

## Arms (starting set; the loop may invent more within the rules)
1. **rbtd8** (round-1 champion) and **k0**: baselines, rerun on the same harness.
2. **Generic joint world model:** a transformer trained on whole episodes (board + probe sequence: locations, order,
   readings) as one joint distribution, with the 8 symmetries.
3. **Dreaming:** sample unlimited episodes (board and probes together) from the world model in arm 2 and train the forecaster
   on them; target = the dreamed board's 0/1 cells. Variants: dream-only, real+dream mix, dream ratio.
4. **Dreaming with the model's own posterior as target:** the world model's posterior for the dreamed evidence
   (importance sampling or another estimator; enumeration over 2^25 boards is too large). NOTE: Kelvin has not yet ruled whether
   this is allowed; run it but report it separately and flag it.
5. **rbtd + dreams:** rbtd8 applied to dreamed episodes as well as real ones (rbtd already learns the board conditionals).
6. **Shared vs separate:** one network that is both world model and forecaster vs two networks.

## Tasks
- **T1:** round-1 ShipHunt (random probes).
- **T2: board-dependent probe selection.** Probes are more likely near ships (pick and pre-register a rule).
  Question: does the L0 world model learn the selection effect, so that where a probe lands counts as evidence? Score as in T1.

## Evaluation
As round 1: single-cell KL to the exact posterior on held-out TEST at 100/300/1k/3k/10k boards, 3 seeds in the loop and
5 seeds for the final TEST, pre-registered selection on validation outcomes only, plus compute per board. For T2 the exact
posterior must include the selection rule.

## Budget
Kelvin approved up to $200 for round 1 and round 1 spent about $35. Round 2 cap is $100 unless Kelvin raises it. Measure a smoke run first.
Report RESULTS.md plus an audit, as in round 1.
