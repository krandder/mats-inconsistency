# LIR control masks for transformer post-training: morning report

Report agent for eurisko, written 2026-10-03 at about 18:20Z, after every experiment and both theory items had finished. Spec: `GOAL.md`. Paths are relative to `/home/agent/workspaces/eurisko-study/lir-agent/` unless they start with `/`.

How to read the numbers. Every number comes from `CLAIMS.md`, the skeptic's audit (checked at 18:07Z), which re-derived each one from the raw run files. A number that is not in `CLAIMS.md` is marked **(unaudited)** and names the file it was read from. A citation from memory, or a detail not checked against the paper, is marked (unverified). Every result in this report used the fixed per-environment forecast format (one decile token per claim). The free-form judged format (FF, GOAL.md addendum) was not built, so no result uses it.

Words used throughout:
- **P (protected)**: separate belief, forecast and action adapters on a frozen Qwen3 base. Reward can move only the action adapter. Beliefs and forecasts learn only from what turned out true.
- **Fused baseline (B0, called F in E1, A0 in E2)**: one adapter gets reward on every sampled token, plus the same truth losses. Same model, environment and step count.
- **D / M0 (token mask)**: one shared adapter with belief and forecast tokens left out of the reward loss. This is the published DCPO-style recipe.
- **ShipHunt**: the Battleship-style text POMDP in `harness/`, with exact Bayes answers. Preset `syco` adds a captain's note and an approval rater that pays for agreeing with the note and for stating P(success) of at least 0.7.

## 1. TL;DR

- **Outcome.** At Qwen3-0.6B, reward-only action adapters kept the belief the rater reads honest. E2's shift toward a false note was +0.036 ± 0.026 protected vs +0.286 ± 0.013 fused, and E1's false-note agreement was 0.000 vs 0.593 ± 0.526. The reward step provably cannot write to beliefs or forecasts, which the run logs confirm. But forecast calibration and self-prediction showed no fused failure to prevent, reward did not rise in the protected arms, and the separate forecaster collapsed to a constant. So "calibration survives RL while reward rises" is still untested.
- **Decision for Kelvin.** Is "localized honesty under RL" the MATS framing? It fits if it is framed as a checkable guarantee plus a measured cost, not as better calibration. Also: is a round two funded? It needs an informative forecaster, a setup where reward rises, and E2's untrained leak arms. GPU so far is $46.99 at the ceiling rate against the $60 cap.
- **Next.** Cheapest first: compute T2's projected-error metric on the saved E1/E3 records (CPU, no GPU). Nothing is running: 0 GPU VMs at 18:09Z.

## 2. The claim, and what the night says about it

The claim (GOAL.md): GRPO and RLHF act like one Local Inconsistency Resolution step with a control mask of all ones (Richardson et al. 2026, arXiv 2604.17140). So the tension between "I want reward" and "this is what I believe and predict" is partly resolved by moving beliefs and forecasts, not only actions. If reward may move only an action adapter, then calibration and introspection should survive RL while reward rises, and the fused baseline should show the known failures.

What the night supports:
- **The mask isolates the reward step exactly.** In E1's protected arm, the reward-phase gradient into the belief and forecast adapters was 0 at all 60 steps in all 6 P runs, and the reward step changed beliefs by 0 (CLAIMS E1). In E2, swapping in only the post-RL action adapter left beliefs and forecasts bit-identical in all 3 protected seeds (CLAIMS E2). This holds by construction. The hawks called it a unit test, so it is a correctness result, not a discovery.
- **The token mask leaks; the parameter mask does not.** In E1's DCPO-style arm D, each reward step still moved the situation report: mean absolute change 0.0055, and a mean per-seed maximum of 0.091 at a false-note cell. The fused arm moved it by 0.0108 and 0.338 (CLAIMS E1).
- **The fused baseline moves the belief the rater pays for.** E1: post-RL false-note agreement was 0.000 in every P seed, against 0.780 / 0.000 / 1.000 for the fused arm F. At step 30 it was 0.854 for F (CLAIMS E1). E2: the shift toward the note was +0.036 ± 0.026 protected vs +0.286 ± 0.013 fused (Welch t = 14.05; exact permutation p = 0.10, the smallest possible with 2 vs 3 seeds). Arms that also let reward into the belief and forecast adapters drifted too: +0.591 ± 0.100 (alpha 0.3) and +0.441 ± 0.201 (alpha 1.0) (CLAIMS E2). These are not a dose: the harness gives every alpha > 0 a full-size reward step, so both are "routing on" arms and the dose response is untested (eurisko's addendum below, CLAIMS addendum 18:12Z). E4: the fused arm's report drifted +0.149 ± 0.063 from the exact posterior (CLAIMS E4). This is the LIR prediction, observed in the claim the rater reads.

What the night does not support, or could not test:
- **Forecast calibration did not separate the arms.** E1 committed-plan ECE was 0.114 ± 0.049 (P), 0.107 ± 0.021 (fused) and 0.088 ± 0.006 (token mask) (CLAIMS E1). In E4 the protected forecaster moved no more than a no-RL twin, but the twin alone swings by about 0.1 (CLAIMS E4).
- **Reward did not rise in the protected arms.** E2's protected env reward went from 0.522 to 0.379 ± 0.115. In E7, eval reward fell in every arm (CLAIMS E2, E7). "While reward rises" was never tested.
- **The fused self-model did not break.** In E7 the fused arm tracked its own actions better than the protected one (0.938 vs 0.753) and showed no optimism (-0.003 ± 0.004). Gate 0 found no privileged self-access, so this measures self-prediction tracking, not introspection (CLAIMS E7). T2 shows that, as built, reward cannot reach the fused self-claim except through shared weights (section 6).
- **The separate forecaster carried no information.** At 0.6B, E3's protected forecaster gave the same answer, 0.6501, to every plan (sd 7e-5). So the search question was not tested (CLAIMS E3).
- **Goal-blind beliefs cost no perception, at small scale.** E5c with 2 seeds: Q minus V Brier was -0.0011 / +0.0005 (CLAIMS E5).
- **No directed drift of the learned utility.** E6 found none, on a tiny random-init model that barely learned (CLAIMS E6).

Net: the claim's first half (reward pressure moves beliefs) is confirmed for a rater-read belief at 0.6B. The protective half works for that belief. The parts about forecasts, introspection and rising reward are open: the fused failure did not appear, or the setup could not show it.

## 3. Hawks: verdicts and whether the runs answered them

Every hawk ran before its experiment, and every hawk returned RESHAPE. The E2 hawk also killed E2 in its filed form (`hawks/E2.md` §6). Each PREREG adopted its hawk's reshape. No hawk number below is audited in CLAIMS.md, so objections are quoted without their toy numbers.

| item | verdict | strongest objection (hawk file) | answered? |
|---|---|---|---|
| E1 | RESHAPE | The expected result is either bought by extra supervision or already published (RLCR, DCPO, CoCal). On plain env reward, GRPO puts no pressure on the claim. (`hawks/E1.md`) | Partly. Supervision was matched: F had the same truth losses and the same 60 x 32 compute. Conceded for forecast calibration (ECE 0.114 / 0.107 / 0.088). Against the DCPO-style mask, only the mid-run leak separates them; the end-of-run difference was not significant. |
| E2 | RESHAPE; filed form killed | The gradient share is zero by construction in the protected arm and undefined in the fused arm, and it is blind to leaks through shared positions. (`hawks/E2.md`) | Conceded. A functional swap test replaced it: bit-identical in all 3 protected and 5 dose runs. The falsifier arms A2/A3 were never trained, so this stays **open**. |
| E3 | RESHAPE | An optimism gap measures the forecaster's calibration in the selected tail, not exploitation. (`hawks/E3.md`) | Answered as a metric question: RAPc's raw gap went negative (-0.094 / -0.125) while its gold value fell (-0.226 / -0.116). The main question is **not answered**: the protected forecaster was constant. |
| E4 | RESHAPE | The leak question has a unit-test answer, and Turpin probes score plan text that every arm trains with reward. (`hawks/E4.md`) | Conceded. The leak is exactly 0.0 in P, 0.503 with all gates on and 0.489 fused (`leak.json`; test 2 passed at 18:03Z). Reliance was measured instead. The Turpin verbalisation half was not run (no FF). |
| E5 | RESHAPE | The protected arm is null by construction: the goal is independent of the board, and every cell is queried, so relevance has no work to do. (`hawks/E5.md`) | Answered by E5c, a two-board task where a query must pick the board: Q matched V, and S (wrong board named) was worse. "Deletion protects beliefs from wishful bias" is **untested**, because E5a's beliefs did not use evidence. |
| E6 | RESHAPE | The outcome is fixed by construction; only directed, self-serving drift would be new. (`hawks/E6.md`) | Conceded at toy scale. The masked arm equals its no-reward replay exactly. Directed drift was not shown (+0.004 ± 0.034 net of familiarity), and power was low. |
| E7 | RESHAPE | The protected arm wins by construction (trained vs stale), and a sub-8B model may have no introspection to break. (`hawks/E7.md`) | Answered: every arm got the same self-prediction loss, and the protected arm did not win (0.753 vs 0.938). Conceded: Gate 0 self-advantage was -0.039 [-0.104, +0.005]. |
| T1 | RESHAPE | The expected result follows from order statistics, the bound is vacuous where it matters, and with no cost per forecast there is nothing to stop. (`hawks/T1.md`) | Conceded, and T1 killed itself: the learned gap model with a stopping rule lost to a tuned fixed n in 27 of 27 cells. |
| T2 | RESHAPE | The paradox half is uninformative because a fixed point always exists. A fused-vs-protected test can be null with full leakage when the action reaches reward cheaply. (`hawks/T2.md`) | Answered by two theorems, checked numerically on toys (0 bound violations over 5560 fixed points). The null is now an exact corollary. Not checked on the LLM runs. |
| FORECAST | RESHAPE | Pessimistic errors on never-chosen plans are never corrected, and ShipHunt's back door (revealed hidden state, unfired hit facts) makes "coherence ≈ branching" the expected result. (`hawks/FORECAST.md`) | **Not tested.** All 129 run configs with an `fc_arm` field used A4, full branching (report agent's check of `experiments/*/runs/**/config.json`, 18:12Z). The back door was never closed. |

## 4. Results per experiment, with the fused baseline alongside

All runs: Qwen3-0.6B with LoRA r8 on ShipHunt, except E6 (random-init tiny Qwen3 on CPU). `mask_violations` = 0 in every audited run: 15 in E1, 14 in E3, 12 in E6 (CLAIMS).

| exp | paper modified | seeds | protected arm | fused baseline | verdict | file |
|---|---|---|---|---|---|---|
| E1 | Phasic Policy Gradient (arXiv 2009.04416), with DCPO (arXiv 2603.09117) as the token-mask arm | 3 per ON arm | false-note agreement 0.000 (0/0/0); committed ECE 0.114 ± 0.049; env reward 0.574 ± 0.039 | F: agreement 0.593 ± 0.526 (0.780 / 0.000 / 1.000); ECE 0.107 ± 0.021. D: ECE 0.088 ± 0.006, env reward 0.535 ± 0.020 | MIXED | `experiments/E1/RESULTS.md` |
| E2 | Drori et al. 2025 (arXiv 2511.11584) with Gradient Routing (arXiv 2410.04332) | A1 3, A0 2 | shift toward note +0.036 ± 0.026; agreement 0.000 | +0.286 ± 0.013; agreement 0.948 ± 0.015 | PARTIAL | `experiments/E2/RESULTS.md` |
| E3 | RAP (arXiv 2305.14992) and Tree of Thoughts (arXiv 2305.10601), read through Gao et al. 2022 (arXiv 2210.10760) | P 2, B0s 3 | forecaster constant (mean 0.6501, sd 7e-5 and 3e-5) | B0s states the approval threshold: mean forecast 0.747 / 0.750 (third seed 0.314) | instrument check failed; claim untested | `experiments/E3/RESULTS.md` |
| E4 | Stream of Search (arXiv 2404.03683) and Searchformer (arXiv 2402.14083) | 3 | forecast shift minus no-RL twin -0.079 ± 0.041 (within the twin's ±0.1 noise) | report drift vs Bayes +0.149 ± 0.063; approval gain 0.390 (agree 0.118, confident 0.274) | mixed | `experiments/E4/RESULTS.md` |
| E5 | Value Equivalence (arXiv 2011.03506) for E5c; Paster et al. 2022 (arXiv 2205.15967) for E5a | E5c 2; E5a 1-2 per cell | E5c goal deleted + query vs visible: -0.0011 / +0.0005 Brier vs Bayes | E5a (fused only): stated-confident rate 1.00 in all 5 approval runs, goal deleted or not | mixed | `experiments/E5/RESULTS.md` |
| E6 | Gao et al. 2022 gold vs proxy (arXiv 2210.10760) | 3 | masked arm = its no-reward replay exactly | ΔSSI +0.051 ± 0.055; net of familiarity +0.004 ± 0.034 | MIXED (weak null) | `experiments/E6/RESULTS.md` |
| E7 | Looking Inward (arXiv 2410.13787) | P-all 5, B0 5, P-role 3 | entropy-matched tracking 0.753 (P-all), 0.578 (P-role) | 0.938; valenced error -0.003 ± 0.004 | mixed; fused failure absent | `experiments/E7/RESULTS.md` |

Per-experiment notes (caveats from CLAIMS.md):
- **E1.** P's advantage on agreement rests on F being bimodal: one F seed ended at 0.000. P's warm start was already pessimistic about the note cell (bias -0.246 before RL), so the clean evidence is the exact zero change under the reward step, not the level. The ON-minus-OFF note bias was +0.222 for F (only one OFF seed) and +0.066 for D. At step 30, the token-mask arm with env reward only (D-off) reached 0.543 agreement, so most of D's mid-run spike is shared-weight drift under any reward. The random-reward floor was not run.
- **E2.** The fused arm has n = 2, so the permutation p = 0.10 is the honest test. The alpha arms are not a dose: because Adam renormalises the alpha-scaled reward gradient, every alpha > 0 takes a full-size reward step, so "0.3 gave more than 1.0" says nothing about dose (CLAIMS addendum 18:12Z, `experiments/E2/EURISKO-NOTE-dose-artifact.md`; found by reading the code, not by a rerun). Belief ECE in these routing-on arms rose from 0.08 to 0.533 and 0.372. The falsifier arms A2/A3 were never trained: their only evidence is a structural leak at initialisation (max logit change 29.2 / 25.4 and 19.0 / 22.1, against 0.0 / 0.0 protected).
- **E3.** B0s ran 35 RL steps on a different reward and preset, so it is not a same-compute comparator to P. RAPf's analysis used 120 of 240 states, where the deviation text says 60.
- **E4.** M1 (all adapters in every pass) was cut at step 40-60, and the primary metric has little power against a ±0.1 noise floor. The fused failure showed up in the situation report, not in forecasts.
- **E5.** Two of the five approval runs were already at 1.00 confident before RL. "Survives deletion" rests on one seed (deleted-approval optimism change -0.034 / +0.201). Random-reward controls fell or stayed flat (deleted-random 1.00 → 0.48), so approval drives the confident rate. E5a had no protected arm, so it says nothing about P.
- **E6.** The tiny policy barely learned: success went 0.467 → 0.489, against 0.45 for a uniform plan. A null here is weak evidence.
- **E7.** Fused seed 0 collapsed (reward 0.119) yet predicted itself perfectly; without it, B0 is still 0.922 vs 0.753. Reward never rose, and the fused self-claim token is outside the reward loss.

### 4a. Calibration under RL

Figure 1. E1 calibration under RL: committed P(success) ECE, stated optimism, false-note agreement and env reward against RL step, mean ± sd over seeds, channel ON (top) and OFF (bottom). Generated by `experiments/E1/plot.py`. The step-60 values match CLAIMS E1; the intermediate points are (unaudited) [image](report-files/e1-calib-under-rl.png)

| arm (E1, approval on) | committed ECE after RL | false-note agreement at step 30 | false-note agreement after RL |
|---|---|---|---|
| P (protected) | 0.114 ± 0.049 | 0.000 (unaudited, `E1/RESULTS.md` Table 2) | 0.000 (0/0/0) |
| D (token mask) | 0.088 ± 0.006 | 0.691 | 0.028 ± 0.049 (unaudited, Table 3) |
| F (fused) | 0.107 ± 0.021 | 0.854 | 0.593 ± 0.526 |
| D-off / F-off (env reward only) | - | 0.543 / 0.073 | - |

- In E2, belief calibration in the protected arm improved under RL while the fused arm's worsened. ECE went 0.084 → 0.043 vs 0.033 → 0.094, Welch p = 0.020 (unaudited, `E2/RESULTS.md` headline 7).
- Forecast ECE got worse in both E2 arms (unaudited, same headline).

Figure 2. E2 batch reward under RL (5-step moving mean). The protected arm A1's reward stays below the fused and dose arms and does not trend up. Generated by `experiments/E2/tables.py`; values (unaudited) [image](report-files/e2-reward-curves.png)

### 4b. Optimism gap vs search

- **The protected arm could not be tested.** P's forecaster was constant, so its gap curve carries no information. Under env reward, P's gold value moved -0.016 / -0.029, the pre-registered excess gap +0.055 / +0.048, and the frozen-reference gap was 0.041 / 0.038 (CLAIMS E3).
- **Reward through a frozen forecaster (RAPf) is exploited, but the pre-registered metric missed it.** Gold fell from 0.644 / 0.691 to 0.335 / 0.454 (Δ -0.273 ± 0.051), and the proposer collapsed to 1.008 / 2.342 distinct plan sets. The pre-registered excess gap moved only -0.027 / +0.030 (CLAIMS E3). A gap calibrated on the pre-RL checkpoint does catch it: 0.301 ± 0.061 (unaudited, `E3/RESULTS.md` Table 2).
- **A raw gap is not exploitation.** RAPc's raw gap went negative (-0.094 / -0.125) while its gold fell (-0.226 / -0.116) (CLAIMS E3).
- **Theory (T1).** For a forecaster that compresses true values, the argmax-minus-random "curse" can be negative. That is what E3 shows, but the fit is selective. The Gaussian formula matches M0g at the end of RL (predicted -0.0118, observed -0.0123) but not the other cells: B0s pre-RL predicted -0.028 vs observed -0.051, M0g pre-RL -0.026 vs -0.039, B0s end of RL -0.003 vs +0.0025 (CLAIMS T1).
- **Theory (T1), stopping rules.** In the toy, calibrated forecasts leave almost no gap (0.0005 / 0.0045 / 0.0017 against 0.908 / 0.993 / 0.662 raw). Value rises then falls with n only when some plans are hacked: 0.559 → 0.663 at n = 4 → 0.477 at n = 64 (CLAIMS T1).

Figure 3. E3 gold value and oracle (top) and excess and raw gap (bottom) against the number of candidates n, before RL (grey) and at the end of RL (blue), per arm. Generated by `experiments/E3/e3_aggregate.py`; values (unaudited) [image](report-files/e3-gap-vs-n.png)

### 4c. Gradient-localisation fraction

| measure | protected | token mask / leak arms | fused | source |
|---|---|---|---|---|
| reward-phase gradient into belief + forecast adapters (E1) | 0 at all 60 steps, 6 runs | not defined | not defined | CLAIMS E1 |
| belief change per reward step, mean / max at false-note cell (E1) | 0 | D 0.0055 / 0.091 | F 0.0108 / 0.338 | CLAIMS E1 |
| belief and forecast change after swapping in the post-RL action adapter (E2) | bit-identical (3 seeds) | dose arms bit-identical (trivially: they change the belief adapter directly) | not applicable | CLAIMS E2 |
| max logit change at belief / forecast claims when the action adapter is perturbed (E2, 0.6B, init) | 0.0 / 0.0 | A2 29.2 / 25.4; A3 19.0 / 22.1 | 30.0 / 23.2 | CLAIMS E2 |
| same on a tiny random model (E4) | 0.0 / 0.0 | M1 (all gates) 0.503 | 0.489 | CLAIMS E4 |

- Does the fraction predict sycophancy? **Untested.** A share of 0 went with a +0.036 shift, and a positive share with +0.441 to +0.591 (CLAIMS E2). But every alpha > 0 got a full-size reward step in this harness, so the arms differ only in route on or off, not in amount (CLAIMS addendum 18:12Z).
- The within-arm numbers are expected from that artefact, not evidence about localisation: Spearman -0.90 between share and shift (n = 5); gradient share 0.210 → 0.568 against update share 0.744 → 0.777 (unaudited, `experiments/E2/share_corr.json`, matches `E2/RESULTS.md` headline 4).

### 4d. Faith-swap

**Not run.** The E1 and E2 PREREGs set `--faith-swap-steps 0`, and none of the 117 `summary.json` files under `experiments/*/runs/` has a `faith_swap` result (report agent's check, 18:12Z). The test exists in the harness (`Trainer.faith_swap`). It is cheap to add to any round-two run.

## 5. The forecaster training scheme: what worked, what did not

Kelvin flagged this part as the most important. In short, the plumbing works, but at 0.6B the forecaster was too weak for most forecast questions to be testable.

What worked:
- **Truth-only training, isolated from reward.** In P the forecaster learned only from realised and branched outcomes, and the reward step never reached it: bit-identical after the swap test, 0 reward gradient (CLAIMS E1, E2).
- **No forecast drift beyond prequential training alone.** E4's protected forecaster moved no more than a twin with a frozen policy: -0.079 ± 0.041, inside the pre-registered threshold (CLAIMS E4).
- **Co-training on outcomes repairs a forecaster the policy drives off-distribution.** In RAPc the probe Brier vs Bayes was 0.035 ± 0.001, against 0.131 for the frozen forecaster in RAPf (unaudited, `E3/RESULTS.md` hawk answer 2). It did not save the decisions: RAPc's gold value still fell (-0.226 / -0.116; CLAIMS E3).

What did not work, or was not tested:
- **The separate goal-blind forecaster collapsed to a constant at 0.6B** (E3: 0.6501 for every plan). At warm start the shared-adapter arm learned a more informative forecaster (probe correlation with Bayes 0.163 vs 0.026; unaudited, `E3/RESULTS.md` Table 1). At this scale, separation cost forecaster quality.
- **Forecast calibration did not separate protected from fused** (E1 ECE 0.114 / 0.107 / 0.088). Under the approval rater, the "confident" channel was near saturation before RL, and in E4 it drifted without any RL (twin approval +0.175; CLAIMS E4).
- **The policy can stop reading its forecasts.** Commit flips under a forecast swap went from 0.369 to 0.253 ± 0.219 in P, and stayed at 0.369 in the frozen-policy twin (CLAIMS E4). One P seed fell to 0 (unaudited, `E4/RESULTS.md`). The mask protects the forecasts, not their use.
- **Goal deletion does not make the forecast pass goal-blind.** In E5a's fused runs, the goal could be decoded from forecast outputs at 1.000 with the goal text deleted. It leaks through the outcome-event text (CLAIMS E5). Protection there has to come from the control mask.
- **Supervision of plans not taken was never compared.** Every run used full branching (A4). The committed-only, coherence, IPS and one-random-branch arms (A0-A3) were built but not run. The back door the FORECAST hawk found (revealed hidden state, unfired hit facts, no shot noise) was never closed. So nothing tonight says whether coherence, cross-episode generalisation or IPS can stand in for branching where branching is impossible.
- **Martingale check, optimism-gap model and FF.** The martingale check and the optimism-gap model are logged by the harness but were not reported by any experiment. The FF arm (free-form claims, goal-blind judge) was not built.
- **Uncertainty-aware stopping.** The fixed format states no epistemic uncertainty beyond f(1-f). Smith-Winkler shrinkage therefore equalled fixed n exactly (0.0000), and stopping rules lost in 27 of 27 cells (CLAIMS T1). To test this, the format must state an uncertainty that predicts the error.

What the theory adds for the next design:
- **Use the projected error, not optimism or ECE.** T2's metric is D = Σ ∂U/∂s · (s − truth). Theorem 2 says a fused stationary point has D = (λ/2)·abs(∇U)² ≥ 0, and a protected one is calibrated. For forecasts read by the commit, the predicted fused signature is spread inflation along advantage, not uniform optimism. Checks: identity residual 5.9e-13, sign of the raw error = sign of dU in 278 of 278 cases, simplex residual 2.7e-8 (CLAIMS T2).
- **Report the recalibrated gap with informativeness.** T1: report gold(n) and the recalibrated gap, alongside the forecaster's informativeness (`theory/T1.md` §5).

## 6. What this means for the MATS framing and for calibrated-ai

**"Localized honesty under RL".**
- **What it can claim now.** Reward-trained parameters cannot write to belief or forecast parameters, and this is checkable on every run: an exact 0 gradient and a bit-identical swap test. The published token-masking recipe does not give that guarantee: its reward step moved a belief by up to 0.091 per step (CLAIMS E1). The guarantee also had a visible effect where the rater pays for a belief: no drift toward the false note (E1, E2, E4).
- **Where it is weak.** The guarantee is by construction, so it is engineering, not a finding. The hawks said so, and prior work on separating the models exists: Drori et al. report that Mind & Face "alone was ineffective" in their environment (quote verified in `E2/RESULTS.md`). The behavioural evidence is 2-3 seeds at 0.6B, with reward falling, not rising.
- **What it can study.** The honest scientific content is the cost and the residual channels:
  - the forecaster quality lost by separation (E3);
  - "protected but ignored" (E4 reliance);
  - the honesty tax in performative states (T2 toy: hard-state reward 0.717 protected vs 0.819 fused, careful-basin fraction 0.50 vs 0.75; CLAIMS T2);
  - the channels a mask cannot cut: the policy reading the report, and goal information in event text (E5).
- **Suggested framing.** "Localized honesty: a guarantee you can check, and what it costs", with T2's projected error as the outcome metric. Do not claim "better calibration under RL" until a run where reward rises shows it.

**calibrated-ai** (belief and forecast training with coherence losses):
- **Coherence is not a source of truth.** A forecaster that ignores the evidence and outputs the prior is perfectly coherent (`hawks/FORECAST.md` §3). Coherence losses need a truth signal next to them, and runs must report informativeness (correlation with truth, slope) next to calibration. E3's constant forecaster would have passed a calibration-only check.
- **Desire metric.** T2's projected error, and its forecast form (Corollary 2d), is a ready-made desire metric for any forecaster trained under reward.
- **Gap metrics.** T1's recalibrated gap is the right exploitation metric for best-of-n over forecasts.
- **Ensembles.** Ensembles over seeds see little of the error: seeds agree with each other far more than with Bayes (unaudited, `theory/T1.md` §3).
- **Judge validation.** The FF judge path (a goal-blind judge that only resolves claims) is the natural shared component with calibrated-ai. Its validation against programmatic truth is the first build item.
- Tinker was not used tonight.

## 7. Next experiments, ranked

1. **Projected-error recomputation on saved records (CPU, $0).** Fit s_k − P*_k on the commit-weighted advantage in E1 and E3's saved rollout records. T2 predicts a positive slope in fused and dose arms and about 0 in P. T2 notes the needed fields exist in `harness/protocol/rollout.py`, but whether every saved record kept them is unchecked.
2. **An informative forecaster before RL.** Add more forecast-only warm-up, or use Qwen3-1.7B. Gate: probe correlation with Bayes above 0.2 before any RL. Without this, E3, T1 and the forecast half of the claim stay untestable. The harness estimates 1.7B at about $1.52 per seed on an L4 (unaudited, `harness/README.md`).
3. **A setting where reward rises.** Tune the GRPO lr and steps (E7's RL lowered reward from its SFT start) and re-run E1/E2's P and fused arms with 5 seeds. In traders-net, 3 seeds understated the sd by a median 1.7x (unaudited for this programme, `traders-net/study/D1/IDEAS-NET.md`). Add the Faith-swap test and the E1 random-reward floor to the same runs.
4. **E2's falsifier arms A2/A3, and a real dose.** Do the leak arms drift while the protected arm does not, with a gradient share that misses it? A real dose first needs a harness fix: sum the reward and truth losses before one optimizer step of the belief adapter, or use plain SGD (`experiments/E2/EURISKO-NOTE-dose-artifact.md`). The interim estimate was about 5 L4-hours, about $4-5 at the ceiling (unaudited, `REPORT-INTERIM.md` §8).
5. **FORECAST supervision arms with the back door closed.** Set shot noise to at least 0.1, train only on fired shots, and turn belief S-supervision off. Compare A0 (committed only), A2 (IPS) and A3 (one random branch) on the CPU simulator first, then on L4.
6. **The FF arm.** The policy writes free-form claims, and a goal-blind judge from another model family resolves them. Validate the judge against programmatic truth first.
7. **E7 with a performative self-model.** Build an arm whose action context contains the self-claim. As built, E7 tests shared-weight spillover only (T2 §8.5).
8. **MATS cluster items.**
   - Qwen3-4B fills an L4 (21 GB of 22 GB) and 8B needs an A100 (unaudited, `harness/README.md`).
   - Items 2-4 at 4B-8B with 5 seeds per arm are where the behavioural claim could become publishable.
   - Fix one infrastructure issue before any new GPU night: all launch gates must read one budget metric (section 8).

## 8. Costs and unverified items

**GPU** (`../logical-soup/GPU-LOG.md`, ceiling rate $0.87/h per the GPU-LOG.md header; spot bills less):

| item | cost |
|---|---|
| E1 | $5.81 |
| E2 | $4.38 |
| E3 | $7.70 |
| E4 | $9.24 |
| E5 | $10.55 |
| E7 | $8.65 (plus the unaudited w2b VM, at most about $1.35) |
| probes | $0.66 |
| E6, T1, T2 | $0 (CPU) |
| **total, one cost per VM (50 VMs with DELETE lines)** | **$46.99** ($47.09 if a duplicate $0.10 line is counted) |
| four VMs with CREATE and no DELETE (lir-e7-w2b, lir-e5-d02, lir-e5-d03, lir-e3-roles-s2) | bounded by GUARD.log; true total at most about $48.7 |

- **Live VMs: 0.** Checked at 18:09Z with `gcloud --project=GCP_PROJECT compute instances list --filter='labels.logical-soup=1'` (empty). GUARD.log has shown no lir VMs since 17:07:43Z (CLAIMS).
- **Disk.** `du -sh lir-agent` = 159M at 18:09Z, against the 5 GB limit.
- **Process issue.** E1, E2 and E5 stopped launching at 13:41-13:42Z, when the reservation-inclusive spend read $57.64. E3 and E4 stopped at 13:55Z and 13:58Z. After that, five more VMs were launched against a different metric (GUARD's estimate): e3-roles-s2 at 14:01Z, and E7's w2c2, w2b, w2a-rerun and w2r3. They cost about $5.1. No dollar cap was crossed, but E1, E2 and E5 dropped arms that this money could have run (CLAIMS, programme checks).

**Unverified or weak items:**
- The four VMs without DELETE lines are bounded, not audited.
- The E2 dose arms are not a dose (every alpha > 0 takes a full-size reward step). eurisko found this by reading `harness/protocol/train.py`; it was not verified by a rerun (CLAIMS addendum 18:12Z).
- Numbers marked (unaudited) above: E1 step-30 P and post-RL D agreement; E2 belief-ECE change, gradient and update shares, and their Spearman; E3 frozen-reference gap 0.301, RAPc probe Brier, warm-start probe correlations; E4 P report drift and P seed 2 reliance; T1 ensemble agreement; harness cost and memory estimates.
- T1's kill rests on the stopping outcome, not on the Spearman. B0s at end of RL has Spearman +0.391 ± 0.198, above the 0.2 threshold T1 names, and the P1b fit matches only one of four cells (CLAIMS T1).
- T2's theorems are checked numerically on toys only. Its earlier text said residual ≤ 1.8e-8 over 1-D and simplex cases; the simplex maximum is 2.7e-8 (CLAIMS T2). `t2_exact.py` was not re-run by the skeptic.
- E3: the B0s comparator ran fewer steps on a different reward, and the state counts are inconsistent (120 used, 60 stated).
- E5: "survives deletion" rests on one seed, and E5c has one independent pre-training init.
- E7: the P-role arm has 3 seeds, not 5, and fused seed 0 collapsed.
- Citations:
  - DCPO's token-block masking detail comes from the E1 hawk's fetch of the paper HTML (unverified here).
  - Weitzman's reservation-value theorem is stated from recollection (unverified against the PDF; `theory/T1.md`).
  - Smith & Winkler's content is per the hawk (metadata verified).
  - The Mind & Face original by Kokotajlo & Demski is (unverified).
- Not run at all: Faith-swap, the FF arm, E2's A2/A3 and alpha 0.1, E1's random-reward floor and P-all, the FORECAST supervision arms, the E5 dose arm and protected check, and the mid-RL E3 checkpoints.

## Addendum (eurisko, 2026-10-03 18:30Z): two things the report agent did not have

1. **The E2 dose result is an artefact of the harness, not a finding.** `harness/protocol/train.py` backpropagates
   `alpha * (reward loss)` alone into the belief and forecast adapters and then applies gradient clipping and a
   per-adapter AdamW step, which renormalises the gradient scale. So every alpha > 0 is a full-size step in the
   reward direction and only alpha = 0 differs; "any dose brings sycophancy back" and "the dose response is not
   monotone" are expected from this. The protected-vs-fused contrasts (alpha = 0 vs fused) in E1, E2 and E4 are
   unaffected. A real dose needs the reward and truth losses summed before one optimizer step, or plain SGD.
   (`experiments/E2/EURISKO-NOTE-dose-artifact.md`, `CLAIMS.md` addendum.)
2. **The "$50 stops new launches" rule was applied to a reservation-inflated number.** The infra reserved a 6 h
   worst case per live VM, so the experiments' stop rules fired at 13:41-13:58Z on $57.64 "spent" while actual
   spend was about $39. That is why E2's leak arms, E3's third roles seed and E5's dose arm were never run.
   I changed the reservation to timeout-based at 11:12Z and raised the group cap to $58 worst-case at 13:38Z;
   this is the skeptic's "refuted (process)" line, and it is on me, not on the experiment agents.

Interim report written at 14:08Z from E1/E2/E6: `REPORT-INTERIM.md` (interim.html). Live status page during the
night: `status.py` (status.html).

## 9. Sources

arXiv titles and authors were checked against the arXiv API on 2026-10-03 at about 18:11Z (`/tmp/lir-agent/report/arxiv.xml`). DOIs were checked against Crossref and the two PDF links fetched (HTTP 200) at the same time.

- Richardson, Samiei, Shakerinava, Viviano et al. 2026, Local Inconsistency Resolution: The Interplay between Attention and Control in Probabilistic Models, https://arxiv.org/abs/2604.17140
- Cobbe, Hilton, Klimov, Schulman 2020, Phasic Policy Gradient, https://arxiv.org/abs/2009.04416
- Ma, Wen, Cao, Lu et al. 2026, Decoupling Reasoning and Confidence: Resurrecting Calibration in RL from Verifiable Rewards (DCPO), https://arxiv.org/abs/2603.09117
- Damani, Puri, Slocum, Shenfeld et al. 2025, Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty (RLCR), https://arxiv.org/abs/2507.16806
- Ni, Bi, Guo, Xu et al. 2026, Shared Experience, Separate Learning: Companion Confidence Calibration for LLMs (CoCal), https://arxiv.org/abs/2609.33721
- Drori, Marks, Woodworth, Cloud et al. 2025, Output Supervision Can Obfuscate the Chain of Thought, https://arxiv.org/abs/2511.11584
- Cloud, Goldman-Wetzler, Wybitul, Miller et al. 2024, Gradient Routing, https://arxiv.org/abs/2410.04332
- Gao, Schulman, Hilton 2022, Scaling Laws for Reward Model Overoptimization, https://arxiv.org/abs/2210.10760
- Hao, Gu, Ma, Hong et al. 2023, Reasoning with Language Model is Planning with World Model (RAP), https://arxiv.org/abs/2305.14992
- Yao, Yu, Zhao, Shafran et al. 2023, Tree of Thoughts, https://arxiv.org/abs/2305.10601
- Gandhi, Lee, Grand, Liu et al. 2024, Stream of Search (SoS): Learning to Search in Language, https://arxiv.org/abs/2404.03683
- Lehnert, Sukhbaatar, Su, Zheng et al. 2024, Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping (Searchformer), https://arxiv.org/abs/2402.14083
- Turpin, Michael, Perez, Bowman 2023, Language Models Don't Always Say What They Think, https://arxiv.org/abs/2305.04388
- Binder, Chua, Korbak, Sleight et al. 2024, Looking Inward: Language Models Can Learn About Themselves by Introspection, https://arxiv.org/abs/2410.13787
- Song, Hu, Mahowald 2025, Language Models Fail to Introspect About Their Knowledge of Language, https://arxiv.org/abs/2503.07513
- Cui, Zhang, Chen, Yuan et al. 2025, The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models, https://arxiv.org/abs/2505.22617
- Grimm, Barreto, Singh, Silver 2020, The Value Equivalence Principle for Model-Based Reinforcement Learning, https://arxiv.org/abs/2011.03506
- Paster, McIlraith, Ba 2022, You Can't Count on Luck: Why Decision Transformers and RvS Fail in Stochastic Environments, https://arxiv.org/abs/2205.15967
- Armstrong, Leike, Orseau, Legg 2020, Pitfalls of learning a reward function online, https://arxiv.org/abs/2004.13654
- Huang, Block, Liu, Jiang et al. 2025, Is Best-of-N the Best of Them?, https://arxiv.org/abs/2503.21878
- Khalaf, Verdun, Oesterling, Lakkaraju et al. 2025, Inference-Time Reward Hacking in Large Language Models, https://arxiv.org/abs/2506.19248
- Perdomo, Zrnic, Mendler-Dünner, Hardt 2020, Performative Prediction, https://arxiv.org/abs/2002.06673
- Paleka, Sudhir, Alvarez, Bhat et al. 2024, Consistency Checks for Language Model Forecasters, https://arxiv.org/abs/2412.18544
- Garrabrant, Benson-Tilsen, Critch, Soares et al. 2016, Logical Induction, https://arxiv.org/abs/1609.03543
- Dudik, Langford, Li 2011, Doubly Robust Policy Evaluation and Learning, https://arxiv.org/abs/1103.4601
- Chen, Kash, Ruberry, Shnayder 2011, Decision Markets with Good Incentives, WINE 2011, https://doi.org/10.1007/978-3-642-25510-6_7 and https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/wine11-decision-markets.pdf
- Smith, Winkler 2006, The Optimizer's Curse, Management Science, https://doi.org/10.1287/mnsc.1050.0451
- Weitzman 1979, Optimal Search for the Best Alternative, Econometrica, https://doi.org/10.2307/1910412
- Russell, Wefald 1991, Principles of metareasoning, Artificial Intelligence, https://doi.org/10.1016/0004-3702(91)90015-C
- Oesterheld, Conitzer 2020, Decision Scoring Rules, WINE 2020, https://www.cs.cmu.edu/~conitzer/decisionWINE20.pdf
