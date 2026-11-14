# Rich-observation agents: first results (opus-mats builder, 2026-10-23; design: rich-agents-design-2026-10-22.md)

**Short answer.** When training gets off the ground, agents that each see only their own 64-dim vector learn to pull out their bit and send it in 32 floats. The result is as good as one network that reads every raw vector (within 0.001 nats), and 0.004-0.006 nats above the exact ceiling. Two things fail. Small messages (4 or 2 bits) lose a lot. With strong nuisance and many dimensions, training often never leaves the prior, and that is true of the one-network bar as well.

**Setup.** Parity of 3 bits plus a distractor; 6 agents; roles drawn fresh every episode. Each agent sees x = a·U_role·(±1) + nu·V·g + noise. The noise has variance 1 per dimension, U and V are unit-norm, g ~ N(0, I_8) is one per episode, and keys are role + 0.5·noise.
- One deviation from the design: V is the same for all agents rather than a separate V_i. Agents are fresh each episode and share one encoder, so a per-slot V_i would be unknowable to them.
- Agents: one shared MLP encoder, giving a message that is 32 floats or FSQ 4/2 bits. In the queried arm, an agent also reads the earlier messages. The aggregator is a 2-layer transformer over [key, message].
- Bar: the same MLP and transformer over the raw [key, x], with no bottleneck.
- Training: 2k steps of masked-agent pretraining (predict the hidden agent's x), then 30k steps on Y labels only. The checkpoint is chosen on val Y log-loss (20k episodes), so true roles are never used in training or selection. Scored on 4k dev episodes; 3 seeds.
- Regret = KL(true-roles Bayes ‖ model), in nats.

**Exact ceilings** (rich_env.py): enumeration over 4^6 roles × 2^4 bits, with g integrated out exactly (Woodbury). All 9 tests in test_ceilings.py pass:
- matches a brute-force 6D-dimensional Gaussian calculation to 1e-9;
- Monte Carlo calibration (E[log-loss] = E[entropy], binned reliability);
- the keys+content ceiling has strictly more regret than true roles;
- the two ceilings are equal at sigma=0;
- with no nuisance and a=8, both reach the clean-bit parity limit (0.338 nats).

**Sanity cell** (D=64, nu=1, a=1.5, sigma=0.5, wide, one-shot; 3 seeds):

| | regret | Y log-loss |
|---|---|---|
| true-roles Bayes | 0 | 0.479 |
| keys+content ceiling | 0.0910 | 0.561 |
| **learned agents** | **0.0971 ± 0.0003** | 0.567 |
| bar (one network, raw x) | 0.0970 ± 0.0004 | 0.567 |
| content only, no keys (exact) | 0.1818 | 0.649 |
| prior | 0.2218 | 0.693 |

**Probes** (sanity cell): a linear probe reads the agent's own bit from its message at 0.769 accuracy. The best possible from x alone is 0.778, so the bit is extracted almost fully. g is decodable at R² 0.19, against 0.48 from raw x: some nuisance leaks into the message, but most is dropped.

**Sweep** (D {16, 64, 256} × nu {0, 1, 3} × a {1.5, 2} × {wide, b4, b2} × {one-shot, queried}, plus the bar; 378 runs). "Stuck" means a run closed less than 20% of the gap from the prior to the ceiling.
- **Wide messages match the bar where both train:** +0.0013 nats on average (71 paired runs, max 0.009). The bar sits 0.004 above the ceiling.
- **Training stalls, and not only for agents.** Stuck runs for the bar: 0/18 at nu=0, 3/18 at nu=1, 11/18 at nu=3 (2/18 at D=16, 3/18 at D=64, 9/18 at D=256). Wide agents: 1, 5 and 12 of 18 at nu=0, 1, 3. Stuck messages carry g (R² 0.68) and little of the bit (0.57 vs 0.80 possible).
- **Bits are costly.** In the 13 environments where the bar always trains, the median excess over the bar is +0.07 nats for b4 and +0.13 for b2. They also stall more: 10/39 and 20/39 one-shot runs stuck. Symbols are fully used (3.96 of 4 bits; 1.96 of 2).
- **Queried = one-shot.** Wide: −0.0002 ± 0.0004 (34 pairs); b4: −0.0016 ± 0.0005. The queried arm stalls slightly more often.
- **Pretraining helps rather than causes the stalls.** Without it (D=64, a=1.5), wide agents are stuck in 8/9 runs vs 3/9 with it; the bar is unchanged (3/9).

**Not shown.**
- No ceiling for b-bit messages, so the FSQ loss is not split into "a 2-bit code is necessarily lossy" and "the learned code is poor".
- I did not fix the plateau. More steps did not help at a=1.0: 0/6 runs learned in 30k steps, both agents and bar. So a=1.0 was dropped; at a=1.25, 5/6 runs learned.
- Agents do not reach the ceiling. The 0.004-0.006 gap is shared with the bar, so it is the architecture or budget, not the message.
- No real "reading the consensus" benefit appears in the queried arm.
- Nothing yet on point 4 (structure learned through local market signals).

**Files** (work/rich-agents/): rich_env.py, test_ceilings.py, rich_learn.py, rich_summary.py, launch.sh. Results are in results/: sanity_30k.json, sweep_*.json, sweep_summary.txt, nopretrain_*, probe_a*.json.
- Run: `launch.sh <id> "--cells sweep --only 'D==64 and a==1.5' --procs 7"`; 30k steps is the default.
- Compute: about 17 L4-hours, about $15 at the on-demand list ceiling.
