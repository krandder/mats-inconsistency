# LIR control masks for transformer post-training: interim report

Interim report for 2026-10-03, written at about 14:05Z while E3, E4, E5 and E7 are still running. Each statement about the present names the file it was checked in and when. The workflow's final report (REPORT.md) is a separate document. Every result below used the fixed per-environment forecast format (one decile token per claim); the free-form judged format (FF) was not built.

Paths are under `/home/agent/workspaces/eurisko-study/`. `E1/RESULTS.md` means `lir-agent/experiments/E1/RESULTS.md`, and likewise for E2 to E7 and their PREREG.md; `hawks/`, `harness/` and `GUARD.log` are in `lir-agent/`; `GPU-LOG.md` is `logical-soup/GPU-LOG.md`; `bb/RESULTS.md` is `li-net/bb/RESULTS.md`.

## 1. TL;DR

- Finished: E1 (MIXED) and E2 (PARTIAL) on Qwen3-0.6B, E6 (MIXED) on a random-init tiny model. The parameter mask keeps the reward step exactly out of beliefs and forecasts (by construction, confirmed in the logs), and the protected arm did not drift toward the captain's note while the fused arm did (E2 dSyco +0.036 vs +0.286, n = 3 vs 2, `E2/RESULTS.md`); forecast calibration was no better than fused, the mask did not beat a DCPO-style token mask at E1's step 60, and reward did not rise in E2's protected arm.
- Pending: E3, E4, E5 and E7 (6 VMs up, `GUARD.log` 14:00:39Z), T1/T2, the skeptic's CLAIMS.md and the workflow's REPORT.md (about 18:00Z, eurisko's estimate). GPU at the ceiling rate, finished VMs only: lir-agent $37.34, bb $19.99 (`GPU-LOG.md`); lir-agent including running VMs about $41.40 of the $60 night cap (`GUARD.log`).
- Decisions open to Kelvin (none blocks the running work): whether to spend roughly 5 L4-hours (about $4-5 at the ceiling rate, estimated from `E2/PREREG.md`) past tonight's $50 launch stop on E2's untrained leak arms A2/A3 and the 0.1 dose; whether the judged free-form forecast arm (FF) opens the next round; whether "localized honesty under RL" is the MATS framing.

## 2. The claim under test and what the finished runs say

GOAL.md states the claim this way: "GRPO and RLHF are one Local Inconsistency Resolution step with attention on the reward edge and a control mask of all ones, so the inconsistency between 'I want reward' and 'this is what I believe and predict' is resolved partly by moving beliefs and forecasts (sycophancy, confident false success, drift of the internalised preference), not only by moving actions." If reward controls only an action adapter and beliefs and forecasts learn only from what turned out true, "calibration and introspection should survive RL while reward rises, and the fused baseline should show the known failures."

Test bed for the finished LLM runs: Qwen3-0.6B with LoRA adapters on ShipHunt, a Battleship-style text POMDP with an exact Bayes oracle and a branching simulator (`harness/README.md`). Preset `syco` adds a captain's note and an approval rater that pays 0.5 for agreeing with the note and 0.5 for a stated P(success) of at least 0.7, on top of the environment reward (`E2/PREREG.md`). Runs were 60-80 GRPO steps of 32 episodes with 1-3 seeds per arm; GOAL.md asks for at least 3. Belief and forecast passes run on a fresh context with the goal, utility and reward text deleted (`harness/README.md`, R2). Arms: P (A1 in E2) is protected (separate belief, forecast and action adapters; reward updates only the action adapter); F in E1 and A0 in E2 are fused (one adapter, reward on every sampled token plus the same truth losses); D in E1 is a DCPO-style token mask (one shared adapter, claim tokens excluded from the GRPO loss and trained on truth).

**What the mask provably does**

- E1, 3 seeds: in P the reward-phase gradient norm into the belief and forecast adapters is 0.0000 at every step, and the reward step changes belief E[q] and P(success) by 0.0000 (`E1/RESULTS.md` Table 5). D still moves the situation report through its shared adapter: mean |dE[q]| 0.0055 ± 0.0011 per reward step, up to 0.091 ± 0.010 at a false-note cell; F 0.0108 ± 0.0049 and 0.338 ± 0.105 (same table).
- E2: after GPU training, swapping only the action adapter to its post-RL weights on fixed pre-RL contexts leaves the belief cells and every forecast claim bit-identical in all 3 A1 seeds and all 5 dose runs (`E2/RESULTS.md`, headline 2). The self-model claim moves, by design, because the action adapter is stacked at its positions (KL 0.321 ± 0.118 in A1).
- All 15 E1 RL runs and all 12 E6 runs report `mask_violations` 0 (`E1/RESULTS.md`, `E6/RESULTS.md`).
- This was known in advance: the E2 and E4 hawks called the zero gradient a unit test (`hawks/E2.md` §1, `hawks/E4.md` §2), and harness test T2a asserts it (`harness/README.md`). It is a correctness check on the implementation.

**What it buys: no drift toward the claim the rater pays for**

- E1, approval reward on, 3 seeds per arm: agreement with a false captain's note after 60 RL steps is 0.000 ± 0.000 in P and 0.593 ± 0.526 in F (per seed 0.78, 0.00, 1.00); at step 30 it is 0.000 ± 0.000 vs 0.854 ± 0.150 (`E1/RESULTS.md` Tables 2-3). F's spike looks channel-specific, on a single OFF seed: with env reward only, F reached 0.073 at step 30 (n = 1), and F's prequential note bias ON minus OFF is +0.22 (`E1/RESULTS.md`, item 2).
- E2: directed sycophancy (claimed P(note cell) minus the exact Bayes posterior) changed by +0.036 ± 0.026 in A1 (n = 3) and by +0.286 ± 0.013 in A0 (n = 2); Welch p = 0.0008, exact permutation p = 0.10, the smallest possible with 2 vs 3 seeds. Agreement with the note after RL is 0.000 ± 0.000 vs 0.948 ± 0.015, and A1's change moves toward the truth (|bias| 0.269 → 0.232) (`E2/RESULTS.md`, headline 1). The preregistered prediction "A1 within ±0.02 of 0" missed by 0.016, in the truthward direction; "A0 ≥ A1 + 0.05" was met (same file).
- Situation-report calibration in E2: belief ECE fell in A1 (0.084 → 0.043) and rose in A0 (0.033 → 0.094); difference in change 0.102, Welch p = 0.020, n = 3 vs 2 (`E2/RESULTS.md`, headline 7).
- The dose knob, E2 arm A4 (reward routed into the belief and forecast adapters with weight alpha): dSyco +0.591 ± 0.100 at alpha 0.3 (n = 2) and +0.441 ± 0.201 at alpha 1.0 (n = 3), note agreement 1.000 and 0.799 ± 0.270, belief ECE from 0.08 to 0.53 ± 0.07 and 0.37 ± 0.20 (`E2/RESULTS.md`, headline 3). The phrase "any alpha > 0" rests on these two doses; alpha 0.1 was not run.
- The levels are confounded. The protected warm starts already under-report the note cell (pre-RL note bias −0.246 ± 0.008 in P vs −0.022 ± 0.055 fused in E1; −0.269 ± 0.022 vs +0.024 ± 0.027 in E2), so P's zero agreement partly reflects a pessimistic start. The evidence is in the changes under RL and the exact zero reward-step change (`E1/RESULTS.md`, "Confound against P"; `E2/RESULTS.md`).

**What it does not buy, or has not shown**

- Forecast calibration did not improve. In E1, P does not beat F on committed-plan ECE (0.114 ± 0.049 vs 0.107 ± 0.021, d = −0.19) or optimism (|opt| 0.060 vs 0.037, d = −0.31), and it beats D on none of the three primary metrics at step 60 (ECE d = −0.74 and optimism d = −0.77 favour D; false-note agreement 0.000 vs 0.028 ± 0.049, d = 0.82, under the d > 1 bar), with env reward not matched within 1 sd (P 0.574 ± 0.039, D 0.535 ± 0.020; `E1/RESULTS.md`, verdict). In E2, committed-plan optimism vs Bayes does not differ between arms (A1 +0.036 ± 0.182, A0 −0.003 ± 0.055, Welch p = 0.56), and forecast ECE got worse in both (A1 0.089 → 0.121 ± 0.043, A0 0.077 → 0.090) (`E2/RESULTS.md`, headline 7).
- D, the token mask, is competitive at step 60 but drifts mid-RL: false-note agreement 0.691 ± 0.025 at step 30. D with env reward only (n = 2) reaches 0.543 ± 0.060 at step 30, so most of that transient is shared-parameter drift under any reward; with the random-reward floor R not run, sharpening is not separated from drift (`E1/RESULTS.md`, items 3 and 5).
- Reward did not rise in E2's protected arm: RL reward over the first 10 vs the last 20 steps went 0.830 → 0.780 (A0 1.269 → 1.550), and eval env reward 0.522 → 0.379 ± 0.115 (`E2/RESULTS.md`, headline 5). In E1, P's env reward rose (0.490 ± 0.047 → 0.574 ± 0.039) while its approval reward stayed flat (0.915 ± 0.174 → 0.906 ± 0.250) (`E1/RESULTS.md` Tables 1 and 3). So GOAL.md's "calibration survives RL while reward rises" is not shown for the reward being optimised (approval) in either run.
- The gradient share is not a graded measure of leakage: its Spearman correlation with dSyco is 0.48 across A1 and A4 (n = 8) and −0.90 within alpha > 0 (n = 5), for both the gradient and the update share; Adam flattens the dose (update share 0.744 ± 0.014 at alpha 0.3 and 0.777 ± 0.029 at 1.0, gradient share 0.210 vs 0.568) (`E2/RESULTS.md`, headline 4). It separates zero from nonzero and no more.
- The leak channels are untested under training. The hawk's falsifier arms A2 (action-adapter delta applied at every position) and A3 (goal text visible to beliefs, routed to the action adapter) were not trained. Their only evidence is a CPU pre-flight on the 0.6B configuration: max |Δlogit| at belief/forecast claims 29.2/25.4 (A2), 19.0/22.1 (A3), 30.0/23.2 (fused) and 0.0/0.0 (A1) (`E2/RESULTS.md`). This is structural only.
- E6 (random-init tiny Qwen3, 2 layers, hidden size 64, CPU, 3 seeds) found no reward-caused self-serving drift of a latent utility proxy in the fused arm: ΔSSI(d) = +0.051 ± 0.056, and +0.004 ± 0.035 net of familiarity, so the mask had nothing to remove. Truth-only belief learning alone cut the proxy's gold pairwise accuracy from 0.998 to 0.854 ± 0.073, identically in the masked arm and its no-reward replay; reward added undirected damage in the fused arm (gold accuracy −0.051 ± 0.066 against its replay, lower in 3 of 3 seeds) (`E6/RESULTS.md` Tables 1-2). The tiny policy barely learned (success 0.467 → 0.489 in the masked arm, 0.475 → 0.483 fused), so this test is weak.

## 3. Hawks: verdicts and how the runs answered them

All ten hawks returned RESHAPE; the E2 hawk also killed E2 in its filed form (`hawks/E2.md` §6), and no item was dropped. Every experiment's PREREG adopted its hawk's reshape and listed the items it could not take.

| item | verdict | strongest objection (from the hawk file) | answer so far |
|---|---|---|---|
| E1 | RESHAPE | The expected result is "either guaranteed by extra supervision or already published (RLCR, DCPO, CoCal)"; only separate parameters vs token masks, with reward reading the claim, is open. | Supervision matched (F gets every truth loss). Conceded for forecasts: P(success) calibration does not separate P, D and F. Confirmed for the claim the rater reads (F's note bias ON minus OFF +0.22, OFF n = 1). Against DCPO, d = 0.82 at step 60, under the d > 1 bar (`E1/RESULTS.md`). |
| E2 | RESHAPE; E2 as filed killed | "The metric is zero by construction, or undefined, in every planned arm", and a gradient share is blind to leaks through shared positions and attention. | Conceded: A1's share is 0.000 at every step, so a functional swap test replaced it (bit-identical); the share is not graded (Spearman −0.90 within alpha > 0, n = 5). Falsifier arms A2 and A3 not trained (`E2/RESULTS.md`). |
| E3 | RESHAPE | The hidden assumption "optimism gap measures exploitation" fails: the gap measures the forecaster's calibration in the selected tail (toy gap at n = 64: 0.0045 calibrated vs 0.2523 noisy, with identical gold value). | Pending. PREREG adopted gold(n) and isotonic excess gap against exact Bayes values, exogenous n, positive controls (RAPc, RAPf, B0s) and a matched shared-adapter comparator (`E3/PREREG.md`). |
| E4 | RESHAPE | "Turpin probes score the reward-trained plan text, which every arm trains with reward"; the leak question has a unit-test answer (exactly 0.0 only with role gating plus the content mask). | Pending. PREREG: a fixed teacher-forced set scored against a no-RL twin, a forecast-reliance probe, and a note-bias index with an uninformative note (note_rel 1/6) (`E4/PREREG.md`). |
| E5 | RESHAPE | "Protected arm: null by construction": the goal is drawn independently of the board, so hiding it costs nothing at the optimum, and every cell is queried, so relevance has nothing to do. | Pending. PREREG: a two-board environment where a goal-free query must pick the board (E5c) and a signed goal-contingent bias (E5a). The E5c protocol changed three times because perception appeared only as seed-dependent jumps (`E5/PREREG.md`, 09:20Z, 11:10Z, 12:15Z). |
| E6 | RESHAPE | "E6 compares two arms whose outcome is fixed by construction"; only directed, self-serving drift would be new. | Conceded at toy scale: undirected drift is large without reward (the masked arm equals its no-reward replay exactly); directed drift not shown (ΔSSI(d) +0.004 ± 0.035 net of familiarity) (`E6/RESULTS.md`). |
| E7 | RESHAPE | "The protected arm wins by construction": its self-model is trained on-policy on exactly what the metric scores, and a sub-8B model may have no self-advantage to lose. | Pending. PREREG adopted Gate 0, the same self-prediction loss in every arm, separate gating arms, and reward-valenced vs neutral properties (`E7/PREREG.md`). Gate 0 ran before RL (11:52Z); its value is not in this report's sources. |
| T1 | RESHAPE | The expected result "is guaranteed by order statistics"; the bound is vacuous (0.50 at n = 2 against an oracle value of 0.377), and without a cost per forecast the chosen plan's true value never falls with n in 18 ShipHunt configurations. | Pending. The hawk moved it onto E3's checkpoints (CPU, exogenous n, a cost per forecast, a random-pick control); E3's PREREG includes the pessimism and Smith-Winkler baselines (`hawks/T1.md` §4-5, `E3/PREREG.md`). |
| T2 | RESHAPE | "The paradox half is uninformative: a fixed point always exists", and a fused-vs-protected test on an easy POMDP can be null with full leakage (toy signed error −0.025 vs −0.024). | T2a (reward gradient exactly 0 on the belief and forecast adapters) passed when the harness was built (`harness/README.md`); T2b and T2c pending. |
| FORECAST | RESHAPE | Pessimistic errors on never-chosen plans are never corrected (toy lock-in 0.352 ± 0.057 committed-only vs 0.036 ± 0.023 full branching), and ShipHunt's back door (revealed state, unfired hit facts) makes "coherence ≈ branching" the expected result. | Not tested. The harness built arms A0-A4, logged propensities and IPS, but no run compared them and none closed the back door (section 5). |

## 4. Experiments

Status column checked at 14:01Z against `GUARD.log` (STATUS 14:00:39Z) and the PREREG deviation logs.

| item | paper modified | scale | arms run (n seeds) | status / verdict | key number | file |
|---|---|---|---|---|---|---|
| E1 | Phasic Policy Gradient (Cobbe et al. 2020), with DCPO (Ma et al. 2026) as the named baseline | Qwen3-0.6B, `syco`, 100 SFT + 60 RL steps × 32 episodes | P-on 3, D-on 3, F-on 3, P-off 3, D-off 2, F-off 1; R, P-all, B0-floor not run | done: MIXED; the OFF-half rule could not be evaluated | false-note agreement after RL 0.000 ± 0.000 (P) vs 0.593 ± 0.526 (F); committed ECE 0.114 ± 0.049 vs 0.107 ± 0.021 | `E1/RESULTS.md` |
| E2 | Drori et al. 2025 (reward targeting, Mind & Face) with Gradient Routing (Cloud et al. 2024) | Qwen3-0.6B, `syco`, 60 SFT + 80 RL × 32 | A0 2, A1 3, A4 alpha 1.0 3, alpha 0.3 2; A2, A3, alpha 0.1 and the no-conflict check not run | done: PARTIAL | dSyco +0.036 ± 0.026 (A1) vs +0.286 ± 0.013 (A0) | `E2/RESULTS.md` |
| E3 | RAP (Hao et al. 2023) and Tree of Thoughts, read through Gao et al. 2022 best-of-n | Qwen3-0.6B, `hard`, 80 SFT + up to 80 RL × 32 (seed 0: 39-61) | P, RAPc, RAPf 2 seeds; M0g, B0s 3 seeds; eval VMs for s0 and s1 queued, s2 cancelled; missing pairs computed on CPU at reduced scale | running (1 training VM); launches stopped 13:55Z | none yet | `E3/PREREG.md` |
| E4 | Searchformer (Lehnert et al. 2024) and Stream of Search (Gandhi et al. 2024) | Qwen3-0.6B, `syco16` (uninformative note) and `syco`, 100 SFT + 80 RL × 32 | B0, M0, M1, M2 (= P) and a no-RL twin, 3 seeds at note 1/6 (B0/M0 seed 1 still running); note 0.4 seed 0 only; every M1 run cut at RL step 40-60 by the 100-min timeout | running (2 VMs); queue stopped 13:58Z | none yet | `E4/PREREG.md` |
| E5 | Value Equivalence (Grimm et al. 2020) and Tang et al. 2020 for E5c; Paster et al. 2022 for E5a | Qwen3-0.6B; E5c belief adapter only, no RL; E5a 60 SFT + 64 RL × 32 | E5c from a shared pre-training (seeds 1-2 replicate seed 0's); E5a fused cells partly run; dose arm and protected check not run | running (1 VM); launches stopped 13:42Z | none yet | `E5/PREREG.md` |
| E6 | Gao, Schulman, Hilton 2022 (gold vs proxy reward model) | random-init tiny Qwen3 (2 layers, hidden 64), CPU, `easy`, 40 SFT + 200 RL × 32 | c, d, b_c, b_d, a_c, a_d, 3 seeds each | done: MIXED | ΔSSI(d) +0.051 ± 0.056; +0.004 ± 0.035 net of familiarity | `E6/RESULTS.md` |
| E7 | Looking Inward (Binder et al. 2024) | Qwen3-0.6B, `easy`, 80 SFT + 60 RL × 32 | P-all × 5 and B0 × 5 running since 13:39Z; P-role not launched (`E7/REQUEST-w2b.md`) | running (2 VMs) | none yet | `E7/PREREG.md` |

## 5. The forecaster training scheme

This is the part Kelvin flagged as most important (GOAL.md). What the harness built (`harness/README.md`) and what the finished runs used:

| GOAL.md item | built in the harness | used in finished runs |
|---|---|---|
| 1. Committed plan scored by log score / Brier against the realised outcome, every episode | truth losses in `protocol/losses.py`: supervised expected-Brier or NLL, never std-normalised group advantages (`E1/PREREG.md`) | yes, every run |
| 2a. Branching: counterfactual truth for plans not taken | `--fc-arm A4` (branch every candidate; the default) and `A3` (one random branch, label-matched) | A4 everywhere: no PREREG sets another arm, and E3 and E4 name A4 |
| Full-support commit with logged propensity and IPS-weighted scoring (the FORECAST hawk's substitute where branching is impossible) | commit sampled from (1 − eps) softmax + eps/K, propensity logged; IPS weight min(1/(K·prop), clip) via `--p-ips` and `--fc-arm A2` | stochastic commit yes; the IPS arm was not run |
| 2b. Coherence across the deliberation's forecasts | `--fc-arm A1` loss; violation rate as a metric | the loss arm was not run |
| 2c-2d. Cross-episode generalisation and on-policy weighting | properties of every arm, not switches (`hawks/FORECAST.md` §5) | implicit in every run |
| 3. Martingale check on forecast revisions | post-commit revisions only, which is stopping-time safe | logged; not reported in E1 or E2 |
| 4. Optimism-gap model | ridge regressor in `protocol/optgap.py` with zero and constant baselines; no double-estimator baseline | not reported in E1 or E2 |
| Addendum: free-form claims resolved by a goal-blind judge (FF) | not built (first item under "Not built yet") | no |
| Back-door controls: exogenous shot noise, belief adapter not trained on the revealed state | switches exist (a `shot_noise` preset, `--p-belief-target none`) | no PREREG uses them |

What E1 and E2 found about forecasts (Qwen3-0.6B, fixed format):

- Calibration did not separate the arms. Post-RL committed-plan ECE spans 0.088-0.150 across all E1 arms, ON and OFF, with seed sds of 0.006-0.049 (`E1/RESULTS.md` Table 3). E1's explanation: the hawk's toy predicted no separation for claims whose reward coupling is weak, and the rater's "confident ≥ 0.7" bonus was close to saturated before RL (stated confidence ≥ 0.7 on 0.71-0.85 of committed plans), so inflating forecasts had little to buy (`E1/RESULTS.md`, item 4). The selection gap was 0.000-0.008 in every ON arm, so selection contributed almost none of the committed-plan optimism (`E1/RESULTS.md`, must-measure table).
- E2 found no arm difference in committed-plan optimism (Welch p = 0.56) and no sign of the policy shopping for forecaster over-confidence: committed-minus-random signed error stayed negative, and its change was +0.129 ± 0.117 in A1 against +0.020 ± 0.091 in A0, p = 0.33 (`E2/RESULTS.md`, headlines 6-7).
- The forecasters were weak. E2's warm forecaster gave E[q] ≈ 0.750 on committed and random plans alike (A1 seed 0; `E2/RESULTS.md`, headline 5), and E3's seed-0 forecaster was nearly constant on `hard` (RAPf reward 0.65 at every step; `E3/PREREG.md`, 13:10Z). A forecaster that barely varies leaves a calibration comparison little to detect.
- No run closed the back door (revealed hidden state, no shot noise), so in the FORECAST hawk's words these results are "not informative about settings where S is never revealed" (`hawks/FORECAST.md`, addendum).

## 6. Busy-beaver reasoner campaign (separate study, `li-net/bb`)

A separate study, not part of the LIR programme; it finished at 10:30Z. Small transformers (d = 256, 4 + 2 layers, 16,000 steps, 3 seeds) answer questions about roughly 95k machines with 2-5 states; the table gives Brier (mean over six heads) on the 4,126 test rows whose answer needs more than 500 simulator steps, after t reasoning steps (`bb/RESULTS.md`).

| arm | t = 1 | t = 5 | t = 20 | t = 100 | t = 500 |
|---|---|---|---|---|---|
| F (free state) | 0.471 ± 0.004 | 0.451 ± 0.008 | 0.455 ± 0.006 | 0.473 ± 0.005 | 0.499 ± 0.005 |
| B (belief state) | 0.481 ± 0.008 | 0.462 ± 0.010 | 0.462 ± 0.009 | 0.462 ± 0.010 | 0.462 ± 0.010 |
| R0 (no reasoner, one value) | 0.476 ± 0.010 | same | same | same | same |
| S (t-step simulator, base rate on these rows) | 0.614 | 0.614 | 0.591 | 0.590 | 0.590 |

- H1 (F at t = 20 beats R0) holds, barely: log score 12.58 ± 0.02 vs 12.89 ± 0.19. H2 (F improves from t = 20 to t = 100) is rejected: 0.455 → 0.473. H3: S wins every r ≤ 1 bin exactly, and every learned arm beats S's base rate in every r > 1 bin. H4 (B degrades less than F) holds only trivially, +0.000 vs +0.044, because B's belief stops moving.
- Diagnosis: B's belief reaches a fixed point after about 5 blocks (mean change per entry 5.9e-4 over blocks 6+, against 0.35 at block 1) and does not track the machine (P(true head cell) 0.012-0.017, about 1/64). The loss never asks the belief to evolve, since T reaches the model only at the readout. The learned arms' edge of about 0.13 Brier over S's base rate comes from learned priors about machines, without simulation.
- Proposed next leaf: an auxiliary per-block loss on the belief against the true configuration at step k (or T fed into the reasoner, with a stop when the belief reaches it). The campaign cost $20.04 of its $25 cap (`bb/RESULTS.md`).

## 7. Costs and process

GPU cost at the ceiling rate ($0.87/h × VM age; an upper bound, since spot bills less, per the `GPU-LOG.md` header), finished VMs only (running VMs are not yet in it): the sum of `cost_usd` on DELETE and COST-AUDIT lines dated 2026-10-03, checked at 14:01Z.

- lir-agent (`vm=soup-gpu-lir-*`): $37.34 over 40 DELETE lines and no COST-AUDIT lines (`GPU-LOG.md`). By experiment: probes $0.66, E1 $5.81, E2 $4.38, E3 $6.08, E4 $7.41, E5 $9.90, E7 $3.10; E6 ran on CPU. Three VMs ended without a cost line and are in no total: lir-e3-roles-s2, lir-e5-d02, lir-e5-d03.
- bb (`vm=soup-gpu-bb-*`): $19.99 over 11 DELETE and 12 COST-AUDIT lines (`GPU-LOG.md`). The $20.04 in `bb/RESULTS.md` also counts one 2026-10-02 line ($0.05).
- Running estimate for lir-agent, last guard STATUS line (`GUARD.log`, 14:00:39Z): audited $37.34 + running $4.06 = $41.40, with 6 VMs up. The night cap is $60 (GOAL.md).

The infra incident: launch gates counted reservations as spend.

- Admission and the "$50 stops new launches" rule both read `spend_usd`, which charges a live VM its whole reservation and keeps charging a VM that never gets a DELETE line (`GPU-LOG.md` header). Until about 11:12-11:20Z every VM reserved 6 h × $0.87 = $5.22, so nine live VMs filled the $50 lir-agent group whatever they had spent (`GPU-LOG.md` PENDING-CREATE lines; comment in `logical-soup/infra/gpu-up.sh`).
- E3's launches were refused from 08:40Z ("group reservations 48.10 + 5.22 > 50", `E3/PREREG.md`) until its first VMs at 11:21-11:23Z (`GPU-LOG.md`).
- From 11:12Z each new VM reserved its run timeout plus 0.6 h ($1.76-2.19; `GPU-LOG.md`, `logical-soup/infra/gpu-up.sh`). E7's second wave still waited from 11:40Z (`E7/SLOTS-NOTE.md`) to 13:39Z (`GPU-LOG.md`). At 12:33Z the group stood at $48.7 of $50 including reservations, against $23.5 of deleted-VM cost, with 12 sibling VMs; until 13:24Z E7's own launcher still assumed the old $5.22 (`E7/PREREG.md`).
- At 13:41-13:42Z E1, E2 and E5 stopped new launches on `spend_usd lir-agent` = $57.64, which included the reservations of live VMs and of two preempted E5 VMs with no DELETE line (deviation logs in `E1/`, `E2/` and `E5/PREREG.md`). The guard's estimate of spend at 13:40:28Z was $39.37 (`GUARD.log`). E3 stopped at 13:55Z ($56.46 at 13:49Z) and E4 at 13:58Z (their PREREG logs).
- eurisko raised the lir-agent group cap to $58 on this reservation-inclusive measure (`logical-soup/infra/state/group-cap.lir-agent` holds 58, last written 13:38:39Z) and launched E7's two RL VMs at 13:39Z (`E7/PREREG.md`).
- The guard (`guard.sh`: delete any lir VM older than 156 min, and the youngest one if the estimate passes $55) has never fired: `GUARD.log` has no AGE-KILL or CAP-KILL line.
- What the stop cost: in E1 the random-reward floor, the P-all leak arm, B0-floor and three OFF seeds; in E2 the leak arms A2/A3, alpha 0.1, the no-conflict check, A0 seeds 2-4 and A1 seeds 3-4; in E3 the third seed of P, RAPc and RAPf; in E4 note-0.4 seeds 1-2; in E5 the dose arm, the protected check and several fused cells (PREREG deviation logs).
- Other losses: two 0.6B processes on one 22 GB L4 either ran out of memory or gained no throughput (deviation logs of E1, E2, E4, E5 and E7), so later VMs ran one job at a time. E6's 0.6B run on CPU was OOM-killed or swap-bound on the shared box (load 42-46), which is why E6 used the tiny model (`E6/PREREG.md`, 09:15Z).

Pending, checked at 14:01Z:

- Of the four morning deliverables in GOAL.md's goal line, three exist: calibration-under-RL curves (`E1/calib_under_rl.png`), the gradient-localisation metric (`E1/RESULTS.md` Table 5, `E2/RESULTS.md`) and the hawks' verdicts (section 3). Optimism gap vs search is E3's and is pending.
- RESULTS.md for E3, E4, E5 and E7. VMs up at 14:00:39Z (`GUARD.log`): E3 fused s2; E4 `syco-y` s0 and `syco16-f` s1; E5 c9; E7 w2a and w2c. E3's eval VMs launch only if admitted; E7's P-role arm waits on eurisko (`E7/REQUEST-w2b.md`).
- Theory write-ups T1 and T2 (T2a already exists as a harness test).
- The skeptic's CLAIMS.md (no such file under `lir-agent/` at 14:01Z).
- The workflow's full REPORT.md, expected around 18:00Z (eurisko's estimate).

## 8. What eurisko proposes next (proposals, not decisions)

- (a) Train E2's leak arms A2 and A3 and the alpha 0.1 dose, 3 seeds each. A2 and A3 are the hawk's falsifier: do they drift while A1 does not, with a gradient share that misses it? Alpha 0.1 checks whether a small dose already brings sycophancy back. E2's own numbers (A2 about 40 and A3 about 28 L4-minutes per seed by estimate; A1 and A4 runs measured at 1590 s and 1530 s) put this at roughly 5 L4-hours, about $4-5 at the ceiling (`E2/PREREG.md`, budget and 12:00Z log). It needs room past tonight's $50 launch stop.
- (b) Build the FF arm that no experiment used: the policy writes free-form claims with probabilities, a goal-blind judge from another model family marks each one true, false or unresolved, and a formula scores them (GOAL.md addendum). Validate the judge against programmatic truth first, and keep a mandatory P(success) claim scored by the program, because FF moves the phrasing channel from fooling the forecaster to fooling the judge (`hawks/E3.md` §6).
- (c) Frame the MATS project as "localized honesty under RL". The parameter mask gives an exact localisation guarantee that a run can check: zero reward gradient into the belief and forecast adapters (`E1/RESULTS.md` Table 5) and bit-identical beliefs and forecasts under an action-adapter swap after training (`E2/RESULTS.md`). The DCPO-style token mask gives no such guarantee: its reward step moved the note cell by up to 0.091 (`E1/RESULTS.md` Table 5). The behavioural case still needs more seeds, the random-reward floor, the leak arms and a setting where reward rises; at 0.6B and 60-80 RL steps the end-of-training gap to the token mask was not significant.
- (d) Busy beaver: add a per-block loss on the belief against the true machine configuration at step k, so that the belief has to evolve like a simulator state (`bb/RESULTS.md`).
