# Causal vs bidirectional one-network bar (qagents, 2026-10-23)

Question (Kelvin): does the one-network bar's lead come from bidirectional access?
Arm `barc`: the same network as the bar (width 128, 4 layers, same pretraining and selection), with a causal mask over
[question, agents in the episode's random order, cls]. The answer is read from cls, which comes last.
a=2.0, seeds 0-2, dev n=4000. Raw results: work/qagents/results/q_causal.json. Amendment in PREREG.md.

| nu | causal bar | bidirectional bar | paired (causal - bidir) |
|---|---|---|---|
| 0 | 0.3504 ± 0.0097 | 0.3475 ± 0.0031 | +0.0029 ± 0.0069 |
| 1 | 0.3458 ± 0.0086 | 0.3382 ± 0.0046 | +0.0076 ± 0.0131 |

For reference (nu=1): query arm 0.4627 (4-bit messages), 0.4148 (wide); question-only 0.4113; Bayes 0.3052.

Reading: causal access costs at most ~0.01 nats, within noise. The gap to our agents (~0.08-0.12) is not about
access direction. Caveats: 3 seeds; both bars were still improving at 30k steps (best step at the end). Not yet gated.
Next: the token-trader ladder (late mixing k, narrow per-layer channel w), running.
