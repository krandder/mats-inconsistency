# P7: Hindsight-posterior settlement of latent markets

**Status: Informative.**

*Checked at:* 2026-09-29 14:45Z, from `results/summary.json` as written by `analyze.py`. *Evidence:* the per-run files `results/{maze,prefix,ablations}/*.json`, with sha256 sums of every result file in `results/SHA256SUMS` (written 2026-09-29 14:45Z, after the last run). Each VM sync was sha256-verified before merging; see the SYNC lines in `vm/LEDGER.md`.

## TL;DR

1. **Mazes: local settlement trains deep chains, in its likelihood form.**
   - Bl passes each trader the likelihood of the evidence through the downstream conditional market. No gradient crosses a trader boundary.
   - It trains depth-2 and depth-4 chains of discrete-latent traders.
   - At depth 2 its log loss is 0.059 ± 0.022 after 2000 steps, against 0.026 ± 0.023 for end-to-end straight-through (A). After 6000 steps Bl reaches 0.006 ± 0.014 with accuracy 1.000 At depth 4 it reaches 0.040 ± 0.065 after 6000 steps (0.125 after 2000). A at matched steps is lower, at 0.002 (depth 2) and 0.010 (depth 4), but the intervals overlap; see [Matched steps](#matched-steps).
   - Its intermediate latents align with the ground-truth sufficient variable: nmi 0.66–0.80, 5–10 states, and about half of the information about g goes beyond the output. A reaches 0.70–0.89.
2. **The literal q-form collapses.**
   - Plain B settles each level against the settled q above it. At depth 4 its intermediate markets collapse to one constant state (nmi 0, 1 state).
   - Depth then hurts: log loss is 0.082 at depth 1 and 0.500 at depth 4.
3. **Arm D collapses to the marginal at depth ≥ 2.** D is B, or Bl, plus a rate cost and different input views.
   - Ablations show the rate cost is the main cause. Even λ = 0.01 leaves 1 seed of 3 at the marginal and the other 2 at 0.26–0.32.
   - Views alone cost about 0.06 nats.
4. **Prefix sums mod 3: only exact joint settlement works.** Bx settles the whole chain by forward-backward EM.
   - It learns the mod-3 counter and generalises from length 16 to length 256 in 2/3 seeds when every output is revealed, 1/3 when every 2nd is revealed, and 0/3 when every 4th is.
   - Every other arm stays at the base rate in 0/3 seeds. That includes A tuned over tau 0.5/1/2, lr 1e-3, and 10000 steps.
5. **KL(q‖p) as value of computation: no gain over simple baselines.**
   - KL(q‖p) predicts where extra depth helps, but no better than the label-free predictive entropy. For A from depth 1 to 4, Spearman is 0.64 for KL against 0.90 for entropy.
   - It does flag the cells where an intermediate latent mis-encodes the sufficient variable (AUROC 0.86–0.90 for Bl at depth 4).

## Setup

**Maze** (`maze.py`). Pixel mazes: random walls at density 0.3 plus one start cell. The target for each open cell is y = 1 if the cell is reachable from the start in at most 4 moves (breadth-first search through open cells). Training uses 9x9 mazes. Tests use 300 mazes of 9x9, 120 of 15x15 and 60 of 21x21 (fixed seeds). The ground-truth sufficient variable at chain level k is g_k = [dist(start, cell) <= k·rho]. The probes also check distance capped at 5, and whether the cell lies on the shortest path to a random goal. The on-path check is a negative control, since it is not needed for y.

**Prefix sums** (`prefix.py`). Input is a string of bits. The state is s_i = (x_1+…+x_i) mod 3, and the output is y_i = [s_i == 0]. Because the output is binary, a latent that only copies the output cannot carry the three-valued state. Only every d-th output is revealed (d = 1, 2, 4), so d−1 latent traders sit between two settled outputs. Training uses length 16; tests use 2000 strings of length 16, 1000 of length 64 and 500 of length 256. The primary metrics score only the positions the training pattern reveals (`_obs`).

**Traders.** Each unit is a discrete latent with K = 4 values; there are U = 2 units per site, and chains have depth 1, 2 or 4.
- In the maze, each cell holds one trader per unit at each level. A trader reads a (2·rho+1)² window of the input and of the latents at the level below, with rho = 4/D. This gives every depth the same receptive field.
- In prefix sums, the trader at position i reads x_i and z_{i−1}, and its weights are shared across positions.
- The output market is a conditional table P(y | z_top).

**Arms**

| arm | how latents learn |
|---|---|
| A | end-to-end backprop through straight-through Gumbel-softmax latents (tau 1, lr 3e-3) |
| B | hindsight-posterior settlement, local only. Sample z from the priors. At the top, q ∝ p·P(y \| z). Each lower level scores the downstream conditional markets against the settled q above it (q-form, target-propagation style): m_k(v) = Σ_t Σ_v' q_t(v') log P_t(v' \| z_k=v, rest realised), and q_k ∝ p_k·e^{m_k}. Each prior is trained by cross-entropy toward its own q; no gradient crosses a trader boundary |
| Bl (added) | same as B, except each level receives the likelihood of the evidence through the downstream conditional market instead of the settled q: lam_k(v) = Σ_t log Σ_v' P_t(v' \| z_k=v, rest)·e^{lam_t(v')}, and q_k ∝ p_k·e^{lam_k}. On a chain this is the exact hindsight posterior given the realised parent; on the grid it is sampled loopy belief propagation. Still local only |
| Bx (added, prefix only) | exact settlement of the whole chain by forward-backward (input-output HMM EM). Each trader is trained on its pairwise posterior; there is still no backprop across positions |
| C | traders frozen at random init; only the output head is trained |
| D | B plus a rate cost 0.05·KL(p ‖ batch-mean p) on every latent market, plus different input views. In the maze, unit 0 sees walls only and unit 1 sees the start only. In prefix sums, unit 1 does not see x |
| Dl (added) | Bl plus the same rate cost and views |

In the maze at depth 1 the pairs B and Bl, and D and Dl, run the same algorithm, so the pairs serve as a duplicate check. In prefix sums the latents also form a chain across positions (z_{i−1} → z_i), so B and Bl differ there at every d. All arms use the same trader shapes, training steps (maze 2000 × batch 32; prefix 3000 × batch 128), Adam at 3e-3 and 3 seeds.

**Measures.**
- Log loss and accuracy on open cells (maze) or revealed positions (prefix), in-distribution and on the larger instances. Prediction uses the prior predictive: an 8-sample average in the maze, the exact forward filter in prefix sums.
- Alignment of z with the ground-truth sufficient variable, measured by greedy z and four quantities:
  - normalised mutual information, `nmi_g`;
  - a lookup-table probe against its majority baseline;
  - I(z; g | y) / H(g | y), `extra_g`, which counts information beyond the output;
  - the number of states used.
- Collapse: I(z; y)/H(y) (`nmi_y`), the number of states, and `extra_g` near 0.
- Value of computation (VoC): does the per-cell hindsight gap KL(q‖p) of a shallow chain predict the cells where a deeper chain gains log loss? It is compared with the shallow chain's own log loss and predictive entropy.
- Intervals are 95% Student-t intervals over 3 seeds (`common.ci`). With 3 seeds they are wide; a ± larger than the mean means "seed-dependent".

## Results: mazes

Base rates of the test sets (majority accuracy, then log loss of the marginal):

| size | majority accuracy | marginal log loss |
|---|---|---|
| 9x9 | 0.719 | 0.594 |
| 15x15 | 0.895 | 0.336 |
| 21x21 | 0.941 | 0.225 |

Larger mazes have a lower share of positive cells, so every arm's log loss falls with size. Compare each arm with the marginal row at the same size.

### Prediction (3 seeds, 2000 steps)

| arm | D | log loss 9x9 | acc 9x9 | log loss 15x15 | acc 15x15 | log loss 21x21 | acc 21x21 |
|---|---|---|---|---|---|---|---|
| A | 1 | 0.095 ± 0.009 | 0.965 ± 0.006 | 0.048 ± 0.005 | 0.983 ± 0.002 | 0.030 ± 0.001 | 0.991 ± 0.001 |
| A | 2 | 0.026 ± 0.023 | 0.996 ± 0.006 | 0.016 ± 0.012 | 0.998 ± 0.003 | 0.012 ± 0.009 | 0.999 ± 0.001 |
| A | 4 | 0.020 ± 0.041 | 0.997 ± 0.011 | 0.012 ± 0.018 | 0.999 ± 0.005 | 0.009 ± 0.010 | 0.999 ± 0.002 |
| B | 1 | 0.082 ± 0.008 | 0.973 ± 0.004 | 0.042 ± 0.005 | 0.987 ± 0.003 | 0.026 ± 0.002 | 0.994 ± 0.000 |
| B | 2 | 0.265 ± 0.157 | 0.916 ± 0.037 | 0.164 ± 0.135 | 0.963 ± 0.018 | 0.134 ± 0.139 | 0.979 ± 0.012 |
| B | 4 | 0.500 ± 0.009 | 0.814 ± 0.001 | 0.328 ± 0.024 | 0.929 ± 0.000 | 0.288 ± 0.035 | 0.960 ± 0.000 |
| Bl | 1 | 0.082 ± 0.008 | 0.973 ± 0.004 | 0.042 ± 0.005 | 0.987 ± 0.003 | 0.026 ± 0.002 | 0.994 ± 0.000 |
| Bl | 2 | 0.059 ± 0.022 | 0.990 ± 0.007 | 0.034 ± 0.010 | 0.996 ± 0.003 | 0.025 ± 0.007 | 0.998 ± 0.002 |
| Bl | 4 | 0.125 ± 0.017 | 0.966 ± 0.014 | 0.074 ± 0.011 | 0.985 ± 0.006 | 0.057 ± 0.013 | 0.992 ± 0.004 |
| C | 1 | 0.619 ± 0.003 | 0.719 ± 0.000 | 0.362 ± 0.003 | 0.895 ± 0.000 | 0.295 ± 0.004 | 0.941 ± 0.000 |
| C | 2 | 0.619 ± 0.003 | 0.719 ± 0.000 | 0.362 ± 0.003 | 0.895 ± 0.000 | 0.295 ± 0.005 | 0.941 ± 0.000 |
| C | 4 | 0.619 ± 0.003 | 0.719 ± 0.000 | 0.362 ± 0.003 | 0.895 ± 0.000 | 0.295 ± 0.004 | 0.941 ± 0.000 |
| D | 1 | 0.281 ± 0.009 | 0.879 ± 0.013 | 0.137 ± 0.003 | 0.950 ± 0.006 | 0.095 ± 0.002 | 0.972 ± 0.002 |
| D | 2 | 0.619 ± 0.003 | 0.719 ± 0.000 | 0.362 ± 0.003 | 0.895 ± 0.000 | 0.295 ± 0.004 | 0.941 ± 0.000 |
| D | 4 | 0.593 ± 0.113 | 0.738 ± 0.080 | 0.339 ± 0.097 | 0.902 ± 0.030 | 0.273 ± 0.093 | 0.944 ± 0.016 |
| Dl | 1 | 0.281 ± 0.009 | 0.879 ± 0.013 | 0.137 ± 0.003 | 0.950 ± 0.006 | 0.095 ± 0.002 | 0.972 ± 0.002 |
| Dl | 2 | 0.619 ± 0.003 | 0.719 ± 0.000 | 0.362 ± 0.003 | 0.895 ± 0.000 | 0.295 ± 0.004 | 0.941 ± 0.000 |
| Dl | 4 | 0.583 ± 0.158 | 0.745 ± 0.110 | 0.344 ± 0.076 | 0.905 ± 0.042 | 0.283 ± 0.053 | 0.946 ± 0.022 |

- **Random encoders (C)** do slightly worse than the marginal (0.619 against 0.594): a random 4-state code of a local window carries almost nothing about 4-step reachability.
- **Larger mazes.** Every arm that learns keeps its accuracy on 15x15 and 21x21 mazes. The task has a fixed radius, so this tests generalisation to larger instances, not to longer reasoning.
- **Bl at depth 4** is worse than at depth 1 after 2000 steps (0.125 against 0.082); see the matched-steps section below.

### Alignment of intermediate latents (levels below the top)

| arm | D | level (radius) | states | nmi(z; g) | probe / majority | extra_g = I(z;g\|y)/H(g\|y) | nmi(z; y) | KL AUROC for z-errors |
|---|---|---|---|---|---|---|---|---|
| A | 2 | 1 (2) | 14.7 ± 5.7 | 0.82 ± 0.17 | 0.98 ± 0.03 / 0.87 | 0.71 ± 0.35 | 0.47 ± 0.06 | 0.79 ± 0.08 |
| A | 4 | 1 (1) | 7.0 ± 2.5 | 0.70 ± 0.79 | 0.98 ± 0.07 / 0.94 | 0.62 ± 0.98 | 0.21 ± 0.05 | 0.71 ± 1.27 |
| A | 4 | 2 (2) | 4.7 ± 3.8 | 0.88 ± 0.39 | 0.98 ± 0.07 / 0.87 | 0.76 ± 0.76 | 0.43 ± 0.24 | 0.70 ± 0.50 |
| A | 4 | 3 (3) | 4.7 ± 3.8 | 0.89 ± 0.42 | 0.98 ± 0.10 / 0.80 | 0.70 ± 1.25 | 0.82 ± 0.12 | 0.81 ± 0.51 |
| B | 2 | 1 (2) | 1.7 ± 2.9 | 0.18 ± 0.79 | 0.88 ± 0.01 / 0.87 | 0.08 ± 0.36 | 0.16 ± 0.67 | 0.42 ± 0.22 |
| B | 4 | 1 (1) | 1.0 ± 0.0 | 0.00 ± 0.00 | 0.94 ± 0.00 / 0.94 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.48 ± 0.05 |
| B | 4 | 2 (2) | 1.0 ± 0.0 | 0.00 ± 0.00 | 0.87 ± 0.00 / 0.87 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.50 ± 0.01 |
| B | 4 | 3 (3) | 1.0 ± 0.0 | 0.00 ± 0.00 | 0.80 ± 0.00 / 0.80 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.45 ± 0.04 |
| Bl | 2 | 1 (2) | 9.0 ± 6.6 | 0.74 ± 0.19 | 0.96 ± 0.05 / 0.87 | 0.53 ± 0.34 | 0.51 ± 0.05 | 0.71 ± 0.16 |
| Bl | 4 | 1 (1) | 9.3 ± 5.2 | 0.66 ± 0.06 | 0.96 ± 0.00 / 0.94 | 0.51 ± 0.07 | 0.23 ± 0.00 | 0.88 ± 0.07 |
| Bl | 4 | 2 (2) | 4.7 ± 2.9 | 0.76 ± 0.13 | 0.96 ± 0.03 / 0.87 | 0.53 ± 0.26 | 0.48 ± 0.01 | 0.86 ± 0.07 |
| Bl | 4 | 3 (3) | 5.7 ± 7.2 | 0.74 ± 0.06 | 0.95 ± 0.02 / 0.80 | 0.28 ± 0.15 | 0.76 ± 0.04 | 0.90 ± 0.02 |
| C | 2 | 1 (2) | 16.0 ± 0.0 | 0.01 ± 0.02 | 0.87 ± 0.00 / 0.87 | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.90 ± 0.03 |
| C | 4 | 1 (1) | 13.7 ± 5.7 | 0.07 ± 0.13 | 0.94 ± 0.00 / 0.94 | 0.08 ± 0.14 | 0.02 ± 0.01 | 0.50 ± 0.05 |
| C | 4 | 2 (2) | 13.7 ± 6.3 | 0.01 ± 0.01 | 0.87 ± 0.00 / 0.87 | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.83 ± 0.25 |
| C | 4 | 3 (3) | 15.0 ± 2.5 | 0.01 ± 0.00 | 0.80 ± 0.00 / 0.80 | 0.00 ± 0.00 | 0.01 ± 0.00 | 0.96 ± 0.01 |
| D | 2 | 1 (2) | 1.3 ± 1.4 | 0.00 ± 0.00 | 0.87 ± 0.00 / 0.87 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.44 ± 0.66 |
| D | 4 | 1–3 | 1.0 ± 0.0 | 0.00 ± 0.00 | = majority | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.50–0.61 |
| Dl | 2 | 1 (2) | 1.3 ± 1.4 | 0.00 ± 0.00 | 0.87 ± 0.00 / 0.87 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.46 ± 0.26 |
| Dl | 4 | 1–3 | 1.0 ± 0.0 | 0.00 ± 0.00 | = majority | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.60–0.65 |

"states" counts joint states of the U = 2 units, up to 16. The "KL AUROC for z-errors" column asks whether the per-cell KL(q‖p) at that level ranks the cells where greedy z disagrees with the probe's prediction of g. It means little for collapsed arms, because a constant z has an error rate equal to the base rate.

Distance probe: nmi(z; min(dist, 5)) is 0.26–0.49 for Bl and 0.29–0.59 for A at matching levels. For collapsed B and D it is 0.00.

On-path negative control: nmi is 0.08–0.11 for every learning arm. The latents do not encode the shortest path, which is not needed for y.

### Top level: collapse onto the output

| arm | D | states | nmi(z; y) | KL(q‖p) per level (bottom → top) |
|---|---|---|---|---|
| A | 1 | 11.7 ± 3.8 | 0.84 ± 0.02 | 0.018 ± 0.004 |
| A | 2 | 5.7 ± 7.2 | 0.97 ± 0.04 | 0.009, 0.008 |
| A | 4 | 4.3 ± 8.0 | 0.98 ± 0.08 | 0.069, 0.011, 0.004, 0.002 |
| B | 1 | 4.0 ± 0.0 | 0.86 ± 0.02 | 0.031 ± 0.008 |
| B | 2 | 4.0 ± 0.0 | 0.55 ± 0.22 | 0.001, 0.050 |
| B | 4 | 3.7 ± 1.4 | 0.21 ± 0.01 | 0.000, 0.000, 0.000, 0.060 |
| Bl | 1 | 4.0 ± 0.0 | 0.86 ± 0.02 | 0.031 ± 0.008 |
| Bl | 2 | 4.0 ± 0.0 | 0.93 ± 0.06 | 0.026, 0.030 |
| Bl | 4 | 4.0 ± 0.0 | 0.79 ± 0.03 | 0.012, 0.026, 0.040, 0.061 |
| D / Dl | 2 | 1.0 | 0.00 | 0.000, 0.000 |
| D / Dl | 4 | 1.3–2.0 | 0.07–0.08 | ≈ 0 at every level |

For A, the KL column is the gap between the Gumbel prior and the hindsight posterior computed from A's own markets; A does not train on it.

The top latent nearly copies the output in every working arm (nmi(z; y) 0.79–0.98). That is what the top latent should do here, because y is a deterministic function of g at the top radius. The collapse question is about the intermediate levels. There the working arms keep information beyond y (extra_g 0.28–0.76), while B and D/Dl at depth ≥ 2 keep none.

### Why the q-form and the rate cost collapse

**q-form (B).** The downstream score m_k(v) = Σ_t Σ_v' q_t(v') log P_t(v' | z_k = v, rest realised) is taken against the q above.
- That q was settled on a sample drawn given the realised z_k, and the conditional market P_t is itself trained toward that q.
- So the message favours whatever value z_k happened to take, and each prior sharpens onto its own samples.
- Once the downstream market ignores z_k, m_k is flat in v, q_k = p_k and KL(q‖p) = 0. No signal remains to break the symmetry.

The data fit this: KL(q‖p) is 0.000 at levels 1–3 of B at depth 4, with 1 state and nmi 0. This is a mechanism consistent with the evidence; we did not test it separately.

Bl passes the likelihood of the evidence. That likelihood depends on z_k whenever the downstream conditional does, including at a random start, so it has no such absorbing state.

**Rate cost (D, Dl).** λ·KL(p ‖ batch-mean p), averaged over a batch, estimates I(z; x) for that market. Early on, the settlement signal at an intermediate level is weak (q ≈ p), so the rate cost pushes p toward the marginal. A constant z is an absorbing state for the same reason as above.

### Maze ablations at depth 2 (3 seeds)

| variant | what changes | log loss 9x9 | acc 9x9 | acc 21x21 | states per level | nmi(z; g) per level | per-seed log loss 9x9 |
|---|---|---|---|---|---|---|---|
| Bl (main) | none | 0.059 ± 0.022 | 0.990 ± 0.007 | 0.998 ± 0.002 | 9.0, 4.0 | 0.74, 0.93 (y) | |
| Bl_long | 6000 steps | 0.006 ± 0.014 | 1.000 ± 0.001 | 1.000 ± 0.000 | 9.7 ± 6.3, 4.0 | 0.80 ± 0.23, 1.00 ± 0.01 | |
| B_long | 6000 steps | 0.219 ± 0.280 | 0.935 ± 0.068 | 0.985 ± 0.017 | 1.3 ± 1.4, 4.0 | 0.18 ± 0.79, 0.65 ± 0.39 | 0.290, 0.278, 0.089 |
| Dl_viewsonly | views, no rate cost | 0.123 ± 0.029 | 0.963 ± 0.014 | 0.991 ± 0.004 | 11.0 ± 4.3, 5.0 ± 2.5 | 0.60 ± 0.09, 0.82 ± 0.08 | |
| Dl_rateonly | rate 0.05, no views | 0.415 ± 0.454 | 0.847 ± 0.276 | 0.966 ± 0.055 | 3.3 ± 2.9, 1.7 ± 1.4 | 0.16 ± 0.67, 0.34 ± 0.74 | 0.269, 0.620, 0.355 |
| Dl_lam001 | rate 0.01 + views | 0.399 ± 0.483 | 0.846 ± 0.276 | 0.966 ± 0.054 | 4.7 ± 7.6, 1.7 ± 1.4 | 0.37 ± 0.80, 0.37 ± 0.79 | 0.321, 0.620, 0.255 |
| Dl (main) | rate 0.05 + views | 0.619 ± 0.003 | 0.719 | 0.941 | 1.3, 1.0 | 0.00, 0.00 | all 0.619 |
| D_viewsonly | q-form + views | 0.348 ± 0.068 | 0.900 ± 0.012 | 0.976 ± 0.002 | 1.0, 3.3 ± 5.7 | 0.00, 0.47 ± 0.04 | |
| D_rateonly | q-form + rate 0.05 | 0.419 ± 0.431 | 0.844 ± 0.268 | 0.965 ± 0.051 | 1.0, 1.7 ± 1.4 | 0.00, 0.32 ± 0.69 | 0.321, 0.620, 0.318 |

- **Rate cost.** The rate cost alone, with no views, raises Bl's depth-2 log loss from 0.059 to 0.27–0.36 in 2 seeds and leaves the third at the marginal (0.620). Cutting λ to 0.01 does not fix it.
- **Views.** Views alone cost about 0.06 nats and keep the latents aligned (nmi 0.60).
- **Together.** Rate cost and views together put all 3 seeds at the marginal.
- **Longer training.** More steps rescue Bl (0.006) but not the q-form (B_long 0.219, still collapsed at level 1).

### Matched steps

Arm A was also run for 6000 steps, so each 6000-step Bl run has an equal-steps comparison:

| arm | D | steps | log loss 9x9 | acc 9x9 | log loss 21x21 | acc 21x21 | nmi(z; g), intermediate levels | extra_g, intermediate levels | wall-clock for 6000 steps |
|---|---|---|---|---|---|---|---|---|---|
| A | 2 | 6000 | 0.002 ± 0.002 | 1.000 ± 0.000 | 0.001 ± 0.001 | 1.000 ± 0.000 | 0.89 ± 0.23 | 0.81 ± 0.43 | 119 s (seed 0) |
| Bl | 2 | 6000 | 0.006 ± 0.014 | 1.000 ± 0.001 | 0.002 ± 0.005 | 1.000 ± 0.000 | 0.80 ± 0.23 | 0.64 ± 0.43 | 1398 s (seed 0) |
| A | 4 | 6000 | 0.010 ± 0.032 | 0.997 ± 0.011 | 0.006 ± 0.020 | 0.999 ± 0.005 | 0.78, 0.96, 0.96 | 0.74, 0.92, 0.91 | 167 s (seed 0) |
| Bl | 4 | 6000 | 0.040 ± 0.065 | 0.995 ± 0.015 | 0.018 ± 0.031 | 0.998 ± 0.005 | 0.65, 0.92, 0.92 | 0.52, 0.84, 0.75 | 1205 s (seed 0) |

- **Matched steps.** A still leads. At depth 2, A reaches 0.002 against 0.006 for Bl; both are at accuracy 1.000, and the intervals overlap. At depth 4, Bl reaches 0.040 ± 0.065 (per seed 0.016, 0.037, 0.068; it was 0.125 after 2000 steps) against 0.010 ± 0.032 for A (0.001, 0.005, 0.025). A is lower in all 3 seeds, but the intervals overlap, and accuracy is 0.995 against 0.997. With more steps, Bl's intermediate latents at depth 4 align about as well as A's (nmi 0.65, 0.92, 0.92 against 0.78, 0.96, 0.96).
- **Wall-clock.** Bl costs much more per step in this implementation. On the same VM, running concurrently, A at depth 4 finished 6000 steps in 167 s, while Bl at depth 4 needed 151 s per 500 steps. The Bl message re-evaluates each downstream market for every value of every latent in its window. The seed-0 timings in the table come from different VMs and loads, so they only give the order of magnitude.

### Value of computation: does KL(q‖p) predict where depth pays?

Per cell, the gain is the shallow chain's log loss minus the deeper chain's, both on the 9x9 test set and matched by seed. A "big gain" is more than 0.1 nats. The three predictors come from the shallow chain only: KL(q‖p) summed over its levels, which needs y; its own log loss, which needs y; and its predictive entropy, which does not.

| pair | mean gain (nats/cell) | cells with gain > 0.1 | Spearman: KL(q‖p) | Spearman: own log loss | Spearman: predictive entropy | AUROC big gain: KL / log loss / entropy |
|---|---|---|---|---|---|---|
| A D1→D4 | 0.075 ± 0.050 | 0.14 ± 0.07 | 0.64 ± 0.09 | 0.90 ± 0.29 | 0.90 ± 0.29 | 0.99 / 1.00 / 0.99 |
| A D1→D2 | 0.069 ± 0.032 | 0.13 ± 0.06 | 0.59 ± 0.13 | 0.88 ± 0.16 | 0.88 ± 0.16 | 0.99 / 1.00 / 0.99 |
| A D2→D4 | 0.006 ± 0.019 | 0.03 ± 0.02 | 0.09 ± 0.20 | 0.51 ± 0.25 | 0.51 ± 0.25 | 0.99 / 1.00 / 1.00 |
| Bl D1→D2 | 0.024 ± 0.014 | 0.09 ± 0.04 | 0.37 ± 0.29 | 0.51 ± 0.24 | 0.50 ± 0.24 | 0.97 / 0.99 / 0.98 |
| Bl D1→D4 | −0.043 ± 0.025 | 0.07 ± 0.01 | 0.03 ± 0.13 | −0.05 ± 0.14 | −0.05 ± 0.15 | 0.97 / 0.99 / 0.97 |
| B D1→D4 | −0.418 ± 0.004 | 0.04 ± 0.03 | −0.32 ± 0.19 | −0.36 ± 0.09 | −0.36 ± 0.08 | 0.96 / 0.98 / 0.96 |

- **Where depth pays, all three signals find the cells.** Where depth helps (A, and Bl from depth 1 to 2), every signal finds the cells with a big gain (AUROC 0.97–1.00). On the full ranking KL is the weakest: 0.64 against 0.90.
- **No gain as a value-of-computation signal.** KL needs the label and does no better than label-free entropy, so this setup gives it no advantage as a compute-allocation signal.
- **Where it is informative.** KL(q‖p) at an intermediate level flags that level's own encoding errors: for Bl at depth 4, the AUROC for cells where z mis-encodes g is 0.86–0.90. Entropy at the output cannot give this per-level signal.

## Results: prefix sums mod 3

Base rate: majority accuracy about 0.667; marginal log loss about 0.636.

| arm | d | log loss (L16, revealed) | acc L16 | acc L64 | acc L256 | seeds solved (acc L256 > 0.99) | nmi(z; s) | states | per-seed acc L256 |
|---|---|---|---|---|---|---|---|---|---|
| A | 1 | 0.560 ± 0.000 | 0.733 ± 0.000 | 0.682 ± 0.000 | 0.670 ± 0.000 | 0/3 | 0.07 ± 0.00 | 2.0 ± 0.0 | [0.6701, 0.6701, 0.6701] |
| A | 2 | 0.578 ± 0.000 | 0.721 ± 0.000 | 0.680 ± 0.000 | 0.672 ± 0.000 | 0/3 | 0.05 ± 0.00 | 2.0 ± 0.0 | [0.6716, 0.6716, 0.6716] |
| A | 4 | 0.583 ± 0.021 | 0.718 ± 0.020 | 0.673 ± 0.005 | 0.672 ± 0.001 | 0/3 | 0.07 ± 0.06 | 5.0 ± 5.0 | [0.6731, 0.6723, 0.6721] |
| B | 1 | 0.558 ± 0.011 | 0.733 ± 0.000 | 0.682 ± 0.000 | 0.670 ± 0.000 | 0/3 | 0.07 ± 0.02 | 2.3 ± 1.4 | [0.6701, 0.6701, 0.6701] |
| B | 2 | 0.628 ± 0.001 | 0.678 ± 0.000 | 0.669 ± 0.000 | 0.669 ± 0.000 | 0/3 | 0.00 ± 0.00 | 1.3 ± 1.4 | [0.669, 0.669, 0.669] |
| B | 4 | 0.631 ± 0.000 | 0.674 ± 0.000 | 0.663 ± 0.000 | 0.669 ± 0.000 | 0/3 | 0.00 ± 0.00 | 1.0 ± 0.0 | [0.6693, 0.6693, 0.6693] |
| Bl | 1 | 0.571 ± 0.032 | 0.731 ± 0.006 | 0.682 ± 0.001 | 0.670 ± 0.000 | 0/3 | 0.07 ± 0.02 | 2.3 ± 1.4 | [0.6701, 0.6701, 0.6699] |
| Bl | 2 | 0.578 ± 0.000 | 0.721 ± 0.000 | 0.680 ± 0.000 | 0.672 ± 0.000 | 0/3 | 0.05 ± 0.00 | 2.0 ± 0.0 | [0.6716, 0.6716, 0.6716] |
| Bl | 4 | 0.527 ± 0.107 | 0.732 ± 0.036 | 0.678 ± 0.011 | 0.673 ± 0.003 | 0/3 | 0.15 ± 0.12 | 9.0 ± 5.0 | [0.6726, 0.6729, 0.6745] |
| Bx | 1 | 0.197 ± 0.816 | 0.911 ± 0.384 | 0.894 ± 0.456 | 0.890 ± 0.473 | 2/3 | 0.69 ± 1.34 | 3.3 ± 2.9 | [1.0, 0.6701, 1.0] |
| Bx | 2 | 0.379 ± 0.780 | 0.809 ± 0.412 | 0.773 ± 0.490 | 0.765 ± 0.508 | 1/3 | 0.37 ± 1.36 | 3.3 ± 2.9 | [1.0, 0.6664, 0.6285] |
| Bx | 4 | 0.464 ± 0.321 | 0.794 ± 0.184 | 0.691 ± 0.043 | 0.655 ± 0.085 | 0/3 | 0.21 ± 0.47 | 8.0 ± 5.0 | [0.6152, 0.6743, 0.6751] |
| C | 1 | 0.635 ± 0.000 | 0.668 ± 0.000 | 0.667 ± 0.000 | 0.666 ± 0.000 | 0/3 | 0.03 ± 0.06 | 6.0 ± 9.0 | [0.6662, 0.6662, 0.6662] |
| C | 2 | 0.628 ± 0.000 | 0.678 ± 0.000 | 0.669 ± 0.000 | 0.669 ± 0.000 | 0/3 | 0.02 ± 0.03 | 6.0 ± 9.0 | [0.669, 0.669, 0.669] |
| C | 4 | 0.631 ± 0.000 | 0.674 ± 0.000 | 0.663 ± 0.000 | 0.669 ± 0.000 | 0/3 | 0.01 ± 0.01 | 6.0 ± 9.0 | [0.6693, 0.6693, 0.6693] |
| D | 1 | 0.586 ± 0.107 | 0.711 ± 0.092 | 0.677 ± 0.023 | 0.669 ± 0.006 | 0/3 | 0.05 ± 0.10 | 1.7 ± 1.4 | [0.6701, 0.6701, 0.6662] |
| D | 2 | 0.628 ± 0.000 | 0.678 ± 0.000 | 0.669 ± 0.000 | 0.669 ± 0.000 | 0/3 | 0.00 ± 0.00 | 1.0 ± 0.0 | [0.669, 0.669, 0.669] |
| D | 4 | 0.631 ± 0.000 | 0.674 ± 0.000 | 0.663 ± 0.000 | 0.669 ± 0.000 | 0/3 | 0.00 ± 0.00 | 1.0 ± 0.0 | [0.6693, 0.6693, 0.6693] |
| Dl | 1 | 0.629 ± 0.000 | 0.668 ± 0.000 | 0.667 ± 0.000 | 0.666 ± 0.000 | 0/3 | 0.03 ± 0.00 | 2.0 ± 0.0 | [0.6662, 0.6662, 0.6662] |
| Dl | 2 | 0.628 ± 0.000 | 0.678 ± 0.000 | 0.669 ± 0.000 | 0.669 ± 0.000 | 0/3 | 0.00 ± 0.00 | 1.0 ± 0.0 | [0.669, 0.669, 0.669] |
| Dl | 4 | 0.631 ± 0.000 | 0.674 ± 0.000 | 0.663 ± 0.000 | 0.669 ± 0.000 | 0/3 | 0.00 ± 0.00 | 1.0 ± 0.0 | [0.6693, 0.6693, 0.6693] |

Tuning and longer training for the arms that stayed at the base rate (d = 1, 3 seeds each):

| variant | log loss (L16, revealed) | acc L256 | seeds solved |
|---|---|---|---|
| A, lr 1e-3, 10000 steps | 0.560 ± 0.000 | 0.670 ± 0.000 | 0/3 |
| A, tau 0.5 | 0.560 ± 0.000 | 0.670 ± 0.000 | 0/3 |
| A, tau 2 | 0.560 ± 0.000 | 0.670 ± 0.000 | 0/3 |
| B, 10000 steps | 0.557 ± 0.011 | 0.670 ± 0.000 | 0/3 |
| Bl, 10000 steps | 0.561 ± 0.005 | 0.669 ± 0.003 | 0/3 |
| Bx, 10000 steps, d = 2 | 0.349 ± 0.751 | 0.749 ± 0.543 | 1/3 (per seed 1.0, 0.604, 0.642) |
| Bx, 10000 steps, d = 4 | 0.457 ± 0.334 | 0.650 ± 0.105 | 0/3 (per seed 0.601, 0.674, 0.675) |

- **Straight-through and one-sample settlement stay at the base rate.** A, B and Bl sit in the same 2-state local optimum: z tracks [s == 0] but not the full 3-valued state, which scores 0.670 at length 256 and 0.733 at length 16. Neither more steps nor another tau gets A out of it in 12 runs. This is a claim about this parameterisation (K = 4, U = 2, one shared trader that reads x_i and z_{i−1}), not about straight-through estimators in general.
- **Bx sometimes finds the counter.** Exact joint settlement marginalises the uncertain previous state instead of sampling it. In 3 of 9 runs this finds the mod-3 counter, which then generalises exactly to length 256.
- **Sparser outputs hurt Bx.** With every 4th output revealed it solves 0/3. It uses more states (8 against 3.3) but encodes s less (nmi 0.21 against 0.69).
- **Longer training does not help Bx.** At 10000 steps instead of 3000, the same seed solves d = 2 and none solves d = 4. The failures look like local optima, not undertraining.

## Checks and caveats

- **Duplicate check.** In the maze at depth 1, B and Bl (and D and Dl) run the same algorithm. Their per-seed 9x9 log losses are identical to 6 decimals in all 3 seeds: 0.081423, 0.085689, 0.079750 for B and Bl, and 0.279468, 0.285674, 0.278922 for D and Dl. In prefix sums there is no such pair, because the latents chain across positions.
- **Hardware nondeterminism.** For the same seed, runs on General and on the spot VM diverge chaotically after a few hundred steps (floating-point order), so a seed is reproducible only on the same machine type. For consistency, all maze runs in the tables ran on the spot VMs. Seed-0 maze runs from an earlier code version were rerun there. The prefix runs at d = 1 with seed 0, and Bx and Bl at d = 2 with seed 0, ran on General with the final code. The per-run JSON records the arguments, not the machine.
- **Is Bx local?** Bx is local in the market sense: each trader trains only on its own pairwise posterior, and no gradient crosses a trader. The posterior itself, however, comes from exact inference over the whole chain, which works only because prefix sums form a chain with small K. On the maze grid, Bl is the tractable analogue (sampled loopy belief propagation).
- **Wide intervals.** Three seeds make the 95% intervals wide. Several rows, including A at depth 4 and all of Bx, are bimodal across seeds; the per-seed columns show this.
- **Scope of the maze output.** Reachability within 4 moves is a local quantity. The size tests show that the local rules transfer to larger grids; they do not show longer reasoning.

## Compute

- **Hardware.** All runs used CPU; no GPU was needed. Smoke tests and some prefix runs ran on General, nice'd and in the heavy slice, with at most 6 cores.
- **Spot VMs.** The main grid ran on spot VMs in GCP GCP_PROJECT:
  - trader-cat-p7-1 (c2d-highcpu-32): main grid and ablations, 0.57 h, at most $0.68 at the on-demand ceiling of $1.20/h.
  - trader-cat-p7-2 (c2d-highcpu-16): matched-steps runs, 0.27 h, at most $0.33. It was spot-preempted at 13:50Z, and the preemption deleted the VM and its disk. The A and Bx runs had already synced, but the three Bl depth-4 runs were lost.
  - trader-cat-p7-3 (c2d-highcpu-8): the rerun of those three Bl depth-4 runs, 0.59 h, at most $0.71.
  - Total estimate: at most $1.72 of the $8 cap. The real cost is lower, since spot prices are below the ceiling rate.
- **Ledger.** All VMs are in `../../GPU-LEDGER.md` and `vm/LEDGER.md`. p7-1 and p7-3 were deleted after a sha256-verified sync. p7-2's results are the ones synced before its preemption. Cleanup was confirmed by `gcloud compute instances list` and `gcloud compute disks list` with `--filter=name~trader-cat-p7`, both empty (checked at 2026-09-29 14:40Z).
- **Preemption fix.** The preemption exposed a gap in `vm.sh`: a VM deleted by GCP leaves no DELETE line. `vm.sh reconcile`, which the watchdog now runs, books such VMs at their preemption time. Ledger rows now go into the P7 table instead of the end of the shared file.

## Files

- `maze.py`, `prefix.py`, `common.py`: the puzzles, traders, arms and metrics. Run `python3 maze.py --arm Bl --D 2 --seed 0`.
- `analyze.py`: reads `results/`, writes `results/summary.json` and `charts/*.json`.
- `vm.sh`, `vm_startup.sh`, `vm_run_queue.sh`: spot-VM runner with watchdog, sync, ledger and delete.
- `results/maze/`, `results/prefix/`, `results/ablations/`: one JSON per run (arguments, training log, evaluation), plus per-cell arrays for the maze.
- `charts/`: maze_logloss, maze_alignment, maze_ablation_D2, maze_voc, prefix_acc256, prefix_alignment (JSON series for plotting).

## References (verified)

- Dempster, Laird & Rubin (1977). Maximum likelihood from incomplete data via the EM algorithm. *JRSS B* 39(1):1–22.
- Bengio & Frasconi (1995). An input output HMM architecture. *NIPS 7*.
- Tishby, Pereira & Bialek (2000). The information bottleneck method. arXiv:physics/0004057.
- Bengio, Léonard & Courville (2013). Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv:1308.3432.
- Bengio (2014). How auto-encoders could provide credit assignment in deep networks via target propagation. arXiv:1407.7906.
- Lee, Zhang, Fischer & Bengio (2015). Difference target propagation. arXiv:1412.7525.
- Jang, Gu & Poole (2016). Categorical reparameterization with Gumbel-softmax. arXiv:1611.01144.
- Maddison, Mnih & Teh (2016). The concrete distribution. arXiv:1611.00712.
- Alemi, Fischer, Dillon & Murphy (2016). Deep variational information bottleneck. arXiv:1612.00410.
- van den Oord, Vinyals & Kavukcuoglu (2017). Neural discrete representation learning. arXiv:1711.00937.
- Schwarzschild et al. (2021). Can you learn an algorithm? Generalizing from easy to hard problems with recurrent networks. arXiv:2106.04537.
