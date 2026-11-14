# Token-trader ladder (qagents, 2026-10-23)

One transformer (width 128, 4 layers, custom pre-norm layer), a=2.0, nu=1, seeds 0-2, dev n=4000. Arm tt_k{K}_w{W}:
the first K layers are per-agent (each token sees only itself and the question); from layer K on, tokens read
other tokens' keys and values only through a W-dim projection of their hidden state. Raw: results/q_ladder.json.

| arm | dev log-loss | vs tt_k0_wfull (paired) |
|---|---|---|
| k0, full channel | 0.3264 ± 0.0011 | 0 |
| k1, full | 0.3273 ± 0.0009 | +0.0009 ± 0.0015 |
| k2, full | 0.3285 ± 0.0011 | +0.0021 ± 0.0020 |
| k3, full (one mixing layer) | 0.3393 ± 0.0035 | +0.0128 ± 0.0025 |
| k0, w=16 | 0.3714 ± 0.0076 | +0.0450 ± 0.0087 |
| k0, w=4 | 0.4544 ± 0.0112 | +0.1280 ± 0.0102 |

References (nu=1): Bayes 0.3052; original bar 0.3382 (this reimplementation is ~0.012 better); query arm b4 0.4627,
wide 0.4148; question-only b4 0.4113.

Reading:
- Late mixing is cheap: even one mixing layer after three per-agent layers costs only ~0.013.
- Channel width is expensive: a 4-dim per-layer channel costs 0.128 even with 4 rounds and end-to-end gradients,
  landing near our 4-bit query arm.
- Our wide-message query arm (0.415) still trails the k3/full transformer (0.339), the closest match to
  "agents, then one consensus round", by ~0.08. Candidates: capacity (MLP encoder + width-64 consensus vs width-128
  transformer layers) and the sequential one-message protocol.
- Confound: the narrow channel here squeezes keys too; our agents send a full key. ladder2 (running) separates them.
Not yet gated.

## Ladder2 (keys full, values narrow), 5 seeds, paired (results/q_ladder2b.json, q_ladder1b.json, ladder2_seeds01_from_log.txt)
| arm | dev log-loss | vs its full-width baseline |
|---|---|---|
| k0 full (5 seeds) | 0.3263 | 0 |
| k0 values 16, keys full | 0.3338 | +0.0075 ± 0.0042 |
| k0 values 4, keys full | 0.3783 | +0.0520 ± 0.0145 |
| k3 full (5 seeds) | 0.3373 | 0 |
| k3 values 16, keys full | 0.3471 | +0.0098 ± 0.0036 |
| k3 values 4, keys full | 0.3580 | +0.0207 ± 0.0088 |
| k3 keys+values 16 | 0.4189 | +0.0816 ± 0.0192 |
| k3 keys+values 4 | 0.4770 | +0.1397 ± 0.0054 |
Reading: squeezing keys is most of the cost; values compress well with full keys. "Agents alone (3 layers), one
consensus layer, full key, 4-dim value" = 0.358, vs our query agents 0.463 (4-bit) / 0.415 (wide): the remaining gap
is our implementation (size, sequential one-message protocol, quantisation), not the architecture. Not yet gated.
