# Gate: rich-observation agents (2026-10-23, adversarial check of rich-agents-2026-10-22.md)

**Verdict: numbers PASS; two headline readings FAIL.** Every number I recomputed matches. But the toy is built so that a 12-float message loses nothing, and the "bar" uses the same per-agent factorization. So claims 4 and 7 come close to holding by construction, and the "nuisance" in claim 3 is signal that the ceiling uses.

How I checked: I read rich_env/rich_learn/rich_summary; their sha256 matches results/code_sha256_sweep.txt. I reran test_ceilings.py (9 passed, 125 s, venv aggext) and recomputed everything from results/*.json (378 sweep runs, no duplicates). The -r2 files are reruns after infra failures (the first attempts exited 100 and 1 with no JSON), not reruns until a run came out good.

Claims:
1. **Ceilings exact: PASS.** They match brute force to 1e-9, are calibrated, satisfy sigma=0 equality, and hit the clean limit 0.3378.
2. **Sanity cell: PASS.** 0.0971±0.0003 / 0.0970±0.0004 / 0.0910 / 0.2218. sanity_30k.json is identical to the sweep cell (same deterministic runs), so it is not an independent replicate.
3. **Probes: the numbers PASS (0.769 vs 0.778), but "drops most of the nuisance" is the wrong reading.** g is shared across the agents of an episode, so V^T x_i is part of the sufficient statistic, and exact Bayes pools it (the M·z term). A naive per-agent product that ignores the shared g costs +0.008 / +0.018 / +0.048 / +0.016 nats over the joint ceiling (D64 nu1 a1.5, D16 nu1 a1.5, D16 nu3 a2, D64 nu3 a2; 2000 episodes; script /tmp/pool_check.py). The optimal message keeps g at the from-x R² (0.48). g R² 0.19 means the message uses part of g, not that it wastes capacity on it.
4. **Wide = bar: holds only conditionally.** The 71 = 35 one-shot + 36 queried pairs where both runs escaped give +0.0013±0.0003 (max 0.0094), correctly computed. Unconditional over all 108: **+0.0117±0.0048**. Conditional on the bar escaping (80 runs): +0.019±0.006. Discordant pairs: bar ok and wide stuck 9, the reverse 3 (sign test p≈0.15).
   Two structural facts make the match expected:
   - (a) The likelihood depends on x_i only through U^T x_i (4 numbers) and V^T x_i (8 numbers), so 12 floats are lossless and wide is 32.
   - (b) The "bar" is per-agent MLP([key, x_i]) → a 64-float token → the same transformer. It is a local encoder with a 64-float message, not a network that mixes raw x across agents.
5. **Stalls: PASS.** Stuck counts by nu: bar 0/3/11, wide one-shot 1/5/12. Wording fix: "2/3/9 by D" is over all nu; within nu=3 it is 2/3/6. A parity plateau shared with the bar.
6. **b-bit cost: PASS.** 13 environments where the bar always escapes: median +0.071 (b4) / +0.136 (b2); 10/39 and 20/39 runs stuck. This is not only stalls: non-stuck runs give a median of +0.065 / +0.134. Unconditional over all 54 runs: +0.066±0.008 / +0.105±0.010. There is no b-bit ceiling (disclosed).
7. **Queried = one-shot: the numbers PASS, the claim is overstated.** Conditional: −0.0002±0.0004 (34 pairs). Unconditional: wide −0.006±0.007, b4 **+0.016±0.006 worse**, and b4 queried is stuck 34/54 vs 25/54 one-shot.
8. **Pretraining reduces stalls: PASS but thin.** Only one environment (D64 a1.5), 9 runs per arm. Wide stuck: 8/9 without pretraining vs 3/9 with (queried 8/9 vs 4/9). Bar: 3/9 in both.

Checks asked for:
- **Locality: PASS.** enc/head are per-token linear layers on the agent's own x. The FSQ usage penalty pools batch statistics in the loss only. The aggregator sees only [key, m].
- **Queried causal: PASS. Reads the consensus: NO.** Agent i attends over [empty, kv(m_1..m_{i-1})], built in order. It reads raw earlier messages, without their keys, and never the aggregator's state. Gradients flow back through earlier messages during training only. With wide messages already sufficient, it cannot add information. The informative test, b-bit queried, came out worse.
- **Matched: PASS.**
  - Same 2000 pretraining + 30000 Y steps, batch 512, lr and schedule.
  - Same generator seed (1000+seed) with an identical draw order, so the training episodes are identical.
  - Same validation set and selection rule.
  - Parameters: 138.6k for the bar vs 136.0k for wide (D64).
- **Selection on val Y only: PASS.** Val is 20k label-only episodes (seed [9000001,D]); dev is [9000002,D] and is used for scoring only. But the amplitudes {1.5, 2} and the 30k steps were chosen after dev-scored pilots (a=1.0 and a=1.25). This is disclosed.
- **Fresh roles: PASS.** Roles, bits, g and noise are drawn every step. U, V and W are fixed, as designed.
- **Deviations hide the hard part: YES.**
  - With U and V fixed, "finding what matters" is one fixed linear projection onto 12 directions, the same for every role.
  - A per-agent random V_i would mostly add near-isotropic noise, which is just as easy.
  - The amplitude choice drops the regime where everything stalls.
  - When training escapes, the probes reach local Bayes (0.813 vs 0.823 mean). So the difficulty is the parity plateau, not extraction under a binding message.
- **Forecast world's 2.5x gap: consistent.** That gap is 0.024 vs 0.0097 (30k steps, 1 seed, not converged). There, each forecaster reads 3 sources × 8 readings with loadings that depend on the description, about 50 factors and 20 questions. The sufficient statistic is large and depends on context. The fair bar gives each (forecaster, source) reading its own token, so attention mixes the readings within a forecaster, while the message arm compresses them with an MLP first. Here neither holds: 12 < 32, and the bar shares the per-agent token. Neither result separates "messages are fundamentally lossy" from "the learned message is poor".

**Fundamental:** this toy does not yet contain the property it was built to test, an agent compressing large private evidence through a message that is a real bottleneck. What is real: the parity plateau, which hits the bar too and is the main obstacle, and the b-bit cost.

Must-fix:
- M1: Put the unconditional means and the 9-vs-3 discordant counts next to "+0.0013 (71)".
- M2: Call the bar "per-agent encoder, 64-float token", and state the 12-float sufficiency. Either add a bar that mixes raw x across agents, or drop the "one network reading all raw x" wording.
- M3: Rewrite the g probe. With a shared g, pooling g is worth 0.008-0.048 nats, so carrying g is not waste.
- M4: Queried arm: say it "reads earlier messages, not the consensus". To test point 2, feed it the aggregator's running posterior and judge it on b-bit messages.
- M5: Next toy: make the sufficient statistic exceed the message. Options: relevant directions redrawn each episode and given by context; nonlinear or sparse bit codes; more relevant dimensions than the message width. Add a b-bit ceiling.
- M6: Fix the wording of the D breakdown (2/3/9 is over all nu).
