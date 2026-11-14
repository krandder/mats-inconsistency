# LIR control masks for post-training: the short version

eurisko, 2026-10-04. Plain-language summary of REPORT.md (numbers from CLAIMS.md, the skeptic's audit). Scale of everything below: Qwen3-0.6B, 60-80 GRPO steps, a Battleship-style text game with exact Bayes answers. Read "at this scale" after every sentence.

## The idea in one paragraph

Ordinary RL fine-tuning gives one gradient to the whole network. If part of the network holds beliefs and forecasts and part chooses actions, the reward can satisfy itself by changing the beliefs (say what the rater wants to hear) instead of the actions. We split the network into three adapters on a frozen base: belief, forecast, action. Reward may only move the action adapter; belief and forecast adapters learn only from what turned out to be true. This is Richardson's "control mask" from Local Inconsistency Resolution applied to GRPO.

## What was shown

1. **The split can be enforced exactly inside one network.** Reward-step gradient into the belief and forecast adapters was 0.0000 at every step; swapping only the trained action adapter leaves every belief and forecast bit-identical. This is by construction, and the logs confirm it.
2. **Without the split, reward buys beliefs.** Under a reward that pays for agreeing with a false note, the fused model agreed with the note 59-95% of the time after RL (0.59 ± 0.53 in E1; 0.95 ± 0.02 in E2) and moved its stated belief toward the note by +0.29. The masked model: 0.000 agreement, +0.04 belief shift, in every seed. The fused model did this with the smallest change to its policy: it moved claims, not actions.
3. **A cheaper trick does most of it.** Excluding belief tokens from the reward loss on one shared adapter (the published DCPO-style recipe) also ends at ~0 agreement after 60 steps, but it drifts in the middle of training (0.69 agreement at step 30) and the reward step still moves the beliefs a little every step. The parameter mask makes honesty an invariant; the token mask makes it a tug of war that truth happens to win here.
4. **Goal-blind perception costs nothing measurable.** Deleting the goal from the belief module's input did not hurt its accuracy about the board (gap -0.0002, 2 seeds).

## What was not shown, and why

- **The policy never learned to act.** Environment reward did not rise in any arm. So nothing here says whether the split helps a capable agent, or whether a strong searcher can game its own forecaster. The experiments test sense-making under reward pressure, not action search.
- **Forecast calibration did not separate the arms.** All arms ended at ECE 0.09-0.15. Prequential training alone swings forecasts by about 0.1, so any forecast effect smaller than that is invisible at this scale.
- **The separate forecaster was useless.** In the search experiment it gave the same probability (0.650) to every plan, so "optimism gap vs search" was not tested.
- **Introspection could not be tested.** The 0.6B model has no privileged access to itself (self-prediction advantage -0.04, CI [-0.10, +0.01]), so the self-model experiment only measures tracking.
- **The "dose" result is an artefact.** "Any leak of reward into the belief adapter brings sycophancy back" came from a knob that scales a gradient which Adam then renormalises; every nonzero dose was a full step. Not a finding.
- **Budget rule misfire.** The infra counted a 6 h worst case per live VM against a $50 cap, so experiments stopped launching at "$57" while actual spend was $39. E2's falsifier arms and one E3 seed were never run. My fault, fixed mid-day.

## Busy-beaver side study (separate)

A learned reasoner over Turing-machine tapes does not simulate: thinking longer hurts, and its belief state goes flat after a few blocks because time only reaches the model at the readout. Repeated blocks do not become time steps unless something ties block k to step k.

## What it adds up to

One clean, expected-by-construction mechanism; one small-scale replication of "reward buys beliefs when actions cannot earn it"; several nulls caused by scale and design. No deep result on training yet. The interesting question, whether a capable agent that cannot bribe its own beliefs acts better or stays honest longer, is untouched.

## Next, if funded (~$25-50 on L4s)

1. An environment where a 0.6-1.7B policy demonstrably improves within 100 steps and still has information-gathering actions.
2. 5-10x more episodes per evaluation so forecasts rise above the noise floor.
3. E2's untrained leak arms with a real dose (reward and truth losses summed before one optimizer step).
4. The remaining channel the mask cannot close: reward shaping beliefs through which experiences the agent chooses to have. Theory first (CPU), then runs.

Cost of the night: about $47 GPU for this programme, $20 for busy beaver. All 10 hawks said "reshape", none killed. Full report: index.html. Interim: interim.html.
