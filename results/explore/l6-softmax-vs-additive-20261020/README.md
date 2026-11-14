Serves: understanding what attention must be to perform Bayesian information aggregation

# L6: softmax (averaging) vs additive attention for Bayesian aggregation (EXPLORATORY)

**EXPLORATORY** (docs/PROTOCOL.md 10.1). Development data only, not a finding. Dev split only; the test split
was never sampled. One run on one spot L4, 2026-10-20.

**One-line result (EXPLORATORY):** softmax attention, whose total weight is fixed at 1, loses more as N grows wherever Bayes needs a total weight near N; one sigmoid-attention model tracks the Bayes total weight across N = 2 to 32 and overlap levels, but in E2 its untrained control, which already adds, gets much of the same benefit.

**Detail.** Softmax keeps the total weight at 1, so its regret grows with N wherever Bayes
needs a total weight near N (E2 low overlap: 0.045 at N=2 to 0.243 at N=32; E1 independent noise: 0.25 to
3.06 nats). One sigmoid-attention model trained on all cells learns a total weight that tracks the Bayes
total weight across N = 2 to 32 and overlap levels (E2 low N=32: 26.9 vs 26.9; high N=32: 6.7 vs 7.0)
and has the lowest (or tied lowest) learned-arm regret in 20 of 30 cells. But its untrained control, which
already adds (total weight about N/2), matches it in 5 of 15 E2 cells (low N=32, mid N>=8, high N=2); learning
helps clearly only where the right total weight is far from N/2 (E2 high overlap, all of E1). No scalar-weight
arm closes the E1 correlated-noise gap; only the exact clearing solve does.

## Question

Softmax attention weights sum to 1, so a softmax readout can only average. Bayes adds independent
log-odds: its total weight exceeds 1 (about N when sources are independent) and shrinks toward 1 when they
overlap. Where does softmax fail, and which attention variant tracks Bayes across overlap levels and
numbers of agents (2 to 32), in E1 and E2?

## Note: why softmax attention averages and Bayes adds

Reports are log-odds l_i with prior l0 (E2). Write any aggregator that weights reports as
q = l0 + sum_i a_i (l_i - l0). Its *total weight* is W = sum_i a_i.

**What Bayes needs.** With full keys (coverage C, piece strengths mu) the best posterior given the
reports is the clearing solve of `aggbench.envs.report_posterior_logit`. Its sum collapses to agent weights:
beta = S^+ diag(S), with S_ij = sum_k C_ik C_jk m_k and m_k = 2 mu_k^2 (diag(S) = C M 1 because C is 0/1).
S is an evidence-weighted overlap matrix: a Gram matrix of phi_i = sqrt(m) * C_i, so it is an
unnormalized attention score matrix. Two limits:
- disjoint evidence: S is diagonal, beta_i = 1, W* = N (add);
- identical evidence: S = s 11^T, beta_i = 1/N, W* = 1 (average).
Partial overlap lies in between. W* is a property of the whole set; it changes with N and with overlap.
E1 (Gaussian) is the same in information form: precision I + sum_i a_i (P_i - I); with independent noise
Bayes is exactly the a_i = 1 sum, W* = N. With shared noise factors Bayes is not a scalar weighting at all
(it needs matrix weights), so every scalar-weight arm has a floor there; `clearing_exact` does the full solve.

**Softmax.** a = softmax(scores) gives a_i >= 0 and W = 1 for every input, so q - l0 lies in the convex
hull of the reports' deviations. Take N agents with disjoint evidence and equal reports l0 + d: Bayes says
l0 + N d, softmax says l0 + d whatever the scores. Its expected log-loss stays at the single-agent level
while Bayes improves with N, so regret grows with N in the low-overlap cells. Exact in the duplicate limit.

**Softmax times a learned scalar T.** W = T for every input: the extremizing pool with per-agent
reweighting. One constant cannot equal W* when W* varies (N = 2 vs 32, low vs high overlap); the best T is a
compromise and errs in opposite directions at the two ends.

**Free total weight.** Sigmoid attention (a_i in (0, 1), W in (0, N)), unnormalized linear attention
(W in (0, inf)) and a softmax with a per-agent learned gain can represent both limits. They still have to
compute beta, which depends on the overlaps with all other agents; one attention layer computes a sum of
pairwise scores, not the inverse S^+. The clearing layer computes beta with one linear solve given the
kernel; with phi read from the keys it is exact by construction (`clearing_exact`), with phi learned it
is `clearing_learned`.

**Limit of the claim.** This is about the attention *weights*. A softmax network whose value path can
multiply by a count (here log N is a public input; a transformer can also estimate N with self-attention)
can escape the convex hull: `softmax_gain` tests that. Normalization is not an impossibility result for
transformers; it is a wrong default for adding independent evidence.

## Design

- **Cells.** 5 agent counts N in {2, 4, 8, 16, 32} x 3 overlap levels, per environment, through `make_env`
  overrides only (recorded in each JSON):
  - E2: n_pieces K = 16N (low), 4N (mid), max(N, 6) (high), so an evidence piece is seen by about
    0.25, 1 or 4 agents on average. Piece strengths are scaled by sqrt(2/N)
    (`strength_range = (0.1, 0.6) * sqrt(2/N)`) so that the Bayes posterior does not saturate at large N:
    Bayes log loss stays between 0.33 and 0.59 nats in every cell. Without this, every arm's regret goes
    to 0 at N = 32 and the comparison says nothing.
  - E1: factor_scale 0 (low: independent noise), 1 (mid, default), 3 (high). With the noise loadings in
    the keys, shared noise is not redundancy: Bayes can cancel it, so its "total weight" exceeds N there.
- **Inputs.** Full visibility. Tokens are the report features, the full key and log(N)/log(32) (N is the
  public `n_agents`). Key widths differ across E2 cells, so coverage keys are zero-padded to K_max = 512
  pieces (padding is lossless).
- **Training.** One model per (env, arm, seed), trained on a mixture of all 15 cells (4000 train
  episodes per cell; each step uses a minibatch from one cell), with log loss against the truth (E2) or
  the Gaussian NLL of theta (E1). No distillation. 6000 AdamW steps, batch 128, lr 1e-3 one-cycle,
  width 128, 2 blocks, 4 heads. 3 seeds (0, 1, 2). Evaluation on 1000 dev episodes per cell.
- **Controls.** Every learned arm goes through `aggbench.controls.run_with_controls` (untrained init and
  scale-matched random weights through the same `predict`). Baselines come from
  `aggbench.baselines.all_baselines`. Its `extremized_log_pool` alpha is fitted **per cell** (1000 train
  episodes), so it knows the cell's average total weight; the learned arms are single models across cells.
  The comparable single-model constant is `softmax_T`.

## Arms

Every learned arm outputs per-agent weights a_i >= 0 and combines the reports with them
(E2: l0 + sum_i a_i (l_i - l0); E1: precision I + sum_i a_i (P_i - I), information sum_i a_i h_i).
Only the way the a_i are produced differs.

| arm | weights | total weight W |
|---|---|---|
| softmax | 2 pre-LN softmax self-attention blocks, softmax readout over agents | 1 |
| softmax_T | as softmax, times one learned scalar T | T (constant) |
| softmax_gain | as softmax, times a per-agent gain softplus(MLP(h_i)) (a free value path) | free |
| sigmoid | sigmoid attention in the blocks and the readout, a_i = sigmoid(q.k_i + b) | (0, N) |
| linear | unnormalized linear attention (phi = elu + 1) in the blocks and the readout | (0, inf) |
| clearing_learned | no blocks; phi_i = MLP(token_i); a = (Phi Phi^T + eps I)^-1 diag(Phi Phi^T) (E1: a clamped >= 0) | free |
| clearing_exact | not learned: the clearing solve with the kernel read from the full keys (E2: the formula above; E1: the full Gaussian solve, matrix weights) | by construction |

## Results

Full per-cell tables (all 15 cells per env, CIs, controls, best arm per cell): `tables.md`. A selection:

**E2: regret to Bayes, nats/episode, mean [95% CI] (learned arms: mean of seeds [0, 1, 2])**

| arm | low N=2 | low N=8 | low N=32 | mid N=32 | high N=8 | high N=32 |
|---|---|---|---|---|---|---|
| softmax | 0.045 [0.032, 0.059] | 0.133 [0.111, 0.155] | 0.243 [0.213, 0.273] | 0.180 [0.156, 0.204] | 0.007 [-0.002, 0.016] | 0.044 [0.025, 0.061] |
| softmax_T | 0.078 [0.057, 0.100] | 0.203 [0.173, 0.233] | 0.305 [0.273, 0.340] | 0.241 [0.214, 0.267] | 0.017 [0.004, 0.029] | 0.085 [0.066, 0.104] |
| softmax_gain | 0.108 [0.077, 0.142] | 1.062 [0.930, 1.209] | 0.764 [0.663, 0.860] | 0.583 [0.505, 0.664] | 0.011 [0.000, 0.022] | 0.057 [0.038, 0.078] |
| sigmoid | 0.014 [0.004, 0.024] | 0.033 [0.020, 0.047] | 0.052 [0.035, 0.072] | 0.085 [0.065, 0.107] | 0.009 [-0.001, 0.018] | 0.018 [0.006, 0.031] |
| linear | 0.055 [0.035, 0.082] | 0.162 [0.130, 0.199] | 0.571 [0.477, 0.663] | 0.193 [0.157, 0.230] | 0.006 [-0.004, 0.015] | 0.013 [0.000, 0.024] |
| clearing_learned | 0.015 [0.006, 0.024] | 0.086 [0.061, 0.115] | 1.134 [0.987, 1.293] | 0.608 [0.519, 0.687] | 0.003 [-0.002, 0.009] | 0.098 [0.074, 0.123] |
| *clearing_exact* | 0.009 [0.002, 0.016] | 0.006 [-0.002, 0.014] | 0.012 [0.004, 0.022] | 0.018 [0.007, 0.028] | 0.002 [0.000, 0.004] | 0.000 [-0.000, 0.001] |
| *oracle_clearing* | 0.009 [0.002, 0.016] | 0.006 [-0.002, 0.014] | 0.012 [0.004, 0.022] | 0.018 [0.007, 0.028] | 0.002 [0.000, 0.004] | 0.000 [-0.000, 0.001] |
| *extremized_log_pool* | 0.010 [-0.000, 0.020] | 0.015 [0.005, 0.026] | 0.030 [0.015, 0.045] | 0.036 [0.021, 0.052] | 0.005 [-0.004, 0.015] | 0.014 [0.002, 0.026] |
| *log_pool* | 0.041 [0.026, 0.057] | 0.219 [0.188, 0.245] | 0.303 [0.271, 0.336] | 0.260 [0.231, 0.289] | 0.019 [0.005, 0.035] | 0.081 [0.054, 0.105] |
| *sum_logodds* | 0.013 [0.001, 0.025] | 0.019 [0.006, 0.033] | 0.029 [0.014, 0.045] | 0.081 [0.047, 0.114] | 0.455 [0.356, 0.550] | 0.531 [0.436, 0.644] |

**E2: total weight on the reports, mean (seed 0; controls of the sigmoid arm included)**

| arm | low N=2 | low N=8 | low N=32 | mid N=32 | high N=8 | high N=32 |
|---|---|---|---|---|---|---|
| bayes | 1.80 | 6.82 | 26.89 | 18.06 | 1.81 | 6.96 |
| softmax:s0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| softmax_T:s0 | 1.99 | 1.99 | 1.99 | 1.99 | 1.99 | 1.99 |
| softmax_gain:s0 | 2.60 | 22.62 | 84.24 | 43.90 | 1.80 | 8.69 |
| sigmoid:s0 | 1.77 | 6.93 | 26.86 | 17.05 | 1.82 | 6.72 |
| linear:s0 | 2.35 | 10.15 | 82.04 | 23.15 | 1.77 | 6.33 |
| clearing_learned:s0 | 1.83 | 9.39 | 92.33 | 41.50 | 1.96 | 6.51 |
| sigmoid:s0:control:untrained_init | 1.01 | 3.96 | 16.30 | 16.25 | 3.98 | 16.04 |
| sigmoid:s0:control:random_network | 0.98 | 3.89 | 15.61 | 15.17 | 3.85 | 14.99 |

**E1: regret to Bayes, nats/episode, mean [95% CI] (learned arms: mean of seeds [0, 1, 2])**

| arm | low N=2 | low N=8 | low N=32 | mid N=32 | high N=8 | high N=32 |
|---|---|---|---|---|---|---|
| softmax | 0.249 [0.214, 0.282] | 1.559 [1.490, 1.630] | 3.063 [2.980, 3.140] | 4.708 [4.615, 4.795] | 3.768 [3.670, 3.871] | 5.943 [5.837, 6.037] |
| softmax_T | 0.381 [0.324, 0.442] | 0.414 [0.370, 0.456] | 1.440 [1.363, 1.506] | 3.149 [3.064, 3.232] | 2.998 [2.891, 3.098] | 4.584 [4.486, 4.678] |
| softmax_gain | 0.036 [0.024, 0.052] | 0.049 [0.035, 0.062] | 0.031 [0.021, 0.042] | 1.591 [1.494, 1.701] | 2.684 [2.566, 2.795] | 3.295 [3.172, 3.425] |
| sigmoid | 0.004 [-0.001, 0.008] | -0.000 [-0.001, -0.000] | -0.000 [-0.000, 0.000] | 1.573 [1.477, 1.676] | 2.617 [2.504, 2.729] | 3.264 [3.137, 3.392] |
| linear | 0.034 [0.022, 0.047] | 0.008 [0.003, 0.013] | 0.003 [0.000, 0.006] | 1.573 [1.478, 1.678] | 2.626 [2.512, 2.735] | 3.261 [3.134, 3.389] |
| clearing_learned | 0.014 [0.001, 0.026] | 0.210 [0.179, 0.241] | 0.755 [0.695, 0.817] | 2.223 [2.115, 2.342] | 2.747 [2.633, 2.864] | 3.906 [3.767, 4.044] |
| *clearing_exact* | 0.000 [0.000, 0.000] | 0.000 [-0.000, 0.000] | -0.000 [-0.000, 0.000] | -0.000 [-0.000, 0.000] | -0.000 [-0.000, 0.000] | 0.000 [-0.000, 0.000] |
| *oracle_clearing* | 0.000 [0.000, 0.000] | 0.000 [-0.000, 0.000] | -0.000 [-0.000, 0.000] | -0.000 [-0.000, 0.000] | -0.000 [-0.000, 0.000] | 0.000 [-0.000, 0.000] |
| *extremized_log_pool* | 0.002 [-0.001, 0.004] | 0.001 [-0.001, 0.003] | 0.175 [0.142, 0.208] | 1.760 [1.679, 1.848] | 2.616 [2.501, 2.729] | 3.453 [3.353, 3.557] |
| *log_pool* | 0.238 [0.201, 0.275] | 2.265 [2.194, 2.347] | 4.888 [4.798, 4.974] | 6.317 [6.222, 6.407] | 4.306 [4.201, 4.411] | 7.565 [7.457, 7.672] |
| *sum_logodds* | 0.000 [-0.000, 0.000] | 0.000 [-0.000, 0.000] | -0.000 [-0.000, 0.000] | 1.573 [1.477, 1.676] | 2.617 [2.503, 2.729] | 3.264 [3.137, 3.392] |

**E1: total weight on the reports, mean (seed 0; controls of the sigmoid arm included)**

| arm | low N=2 | low N=8 | low N=32 | mid N=32 | high N=8 | high N=32 |
|---|---|---|---|---|---|---|
| bayes | 2.00 | 8.00 | 32.00 | 72.03 | 39.94 | 168.88 |
| softmax:s0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| softmax_T:s0 | 3.28 | 3.28 | 3.28 | 3.28 | 3.28 | 3.28 |
| softmax_gain:s0 | 2.13 | 8.46 | 32.03 | 32.85 | 8.99 | 32.24 |
| sigmoid:s0 | 1.97 | 7.99 | 31.96 | 31.96 | 7.98 | 31.96 |
| linear:s0 | 2.09 | 8.06 | 32.05 | 32.20 | 8.25 | 32.69 |
| clearing_learned:s0 | 1.80 | 7.45 | 29.46 | 30.10 | 7.86 | 29.03 |
| sigmoid:s0:control:untrained_init | 0.99 | 3.96 | 15.90 | 15.91 | 3.99 | 15.96 |
| sigmoid:s0:control:random_network | 0.98 | 3.80 | 14.65 | 14.53 | 3.87 | 14.56 |

Controls (seed 0, mean regret over 15 cells: trained / untrained init / random network; cells where trained
is below both controls). From `tables.md`:
- E2: softmax 0.062 / 0.129 / 0.129 (14/15); sigmoid 0.030 / 0.052 / 0.050 (10/15); linear 0.106 / 0.194 / 0.165 (9/15);
  softmax_T 0.106 / 0.129 / 0.096 (5/15); softmax_gain 0.369 / 0.148 / 0.124 (6/15); clearing_learned 0.169 / 0.126 / 0.110 (10/15).
- E1: sigmoid 1.082 / 1.412 / 1.455 (15/15); softmax_gain 15/15; clearing_learned 15/15; softmax 13/15;
  softmax_T 9/15; linear 1.092 / 1.082 / 1.080 (2/15).

## Reading

1. **Softmax fails as predicted, and only where Bayes needs W >> 1.** In E2 low and mid overlap, softmax
   regret rises with N together with the Bayes total weight. At high overlap (Bayes W of 1.2 to 1.8 for
   N <= 8) softmax is within the CI of the best arm. In E1 it is the worst learned arm at every N >= 4. A learned constant
   (softmax_T) does not rescue it: one T cannot match W* from 1.2 to 27 (E2) or 2 to 32 (E1).
2. **Sigmoid attention tracks the Bayes total weight**, across N and overlap, as one model: E2 W within
   10% of Bayes in 14 of 15 cells; E1 independent-noise regret is about 0 at every N. It has the lowest (or tied
   lowest) learned-arm regret in 20 of 30 cells (`tables.md`, best arm per cell; in E1 mid it ties linear and sum_logodds).
3. **Most of that is the inductive bias, not the training.** The untrained sigmoid (W about N/2) already
   matches the trained one in 5 of 15 E2 cells (low N=32, mid N=8/16/32, high N=2), where a control has equal
   or lower regret (`tables.md`). So in those cells the claim "learning found the Bayes weights" is
   void under PROTOCOL 9.1. Training clearly matters where the right W is far from N/2: E2 high overlap
   (N=32: W 16.0 to 6.7, regret 0.133 to 0.011 for seed 0) and every E1 cell.
4. **Unbounded additive weights are hard to train.** linear, softmax_gain and clearing_learned overshoot
   W at N >= 16 in E2 low overlap (N=32: 82 to 92 vs 27) and lose badly there. With a learned generic kernel,
   the clearing layer is best only at small N (E2 mid and high, N <= 8). With the true kernel it is exact
   (clearing_exact = oracle_clearing to every printed digit).
5. **Per-cell extremizing is hard to beat in E2.** A constant total weight fitted to the cell
   (extremized_log_pool) is about as good as the best learned arm. In this E2 configuration, W* varies
   little within a cell, and per-agent weighting from keys adds little. Its E1 low N=32 regret (0.175)
   is an artefact: `fit_extremize` searches alpha only up to 20, and N = 32 needs 32.
6. **E1 with shared noise needs matrix weights.** Bayes uses the known noise loadings to cancel shared
   noise (trace ratio 72 at mid N=32, above N), which no per-agent scalar can express. Every non-softmax
   arm sits at the sum_logodds floor (W = N), and only clearing_exact reaches 0. Overlap in E1 is
   therefore not the redundancy test that E2 overlap is.

## Caveats

Single run on one config family. The models are small and briefly trained (6000 steps), and the training
recipe was not tuned per arm, so the losses of linear, gain and learned clearing at large N may come from
optimization, not capacity. The "total weight" columns are for seed 0. E1 "Bayes total weight" is a trace
ratio, a summary only. Learned arms see log N as an input, which is the easiest case for softmax_gain.

## Provenance and files

- Runner `l6_attention.py` (sha256 recorded in each JSON as `extra.runner_sha256`), launched from commit
  `20ca7bbd08` on a spot L4 (g2-standard-8) through `scripts/infra/gpu-run.sh --up --run-id`
  (`work/run_on_l4.sh` plus `--run-id`, because other lanes' VMs were up). VM age 0.72 h, $0.63 list
  ceiling. An earlier attempt (0.14 h, $0.12) crashed on a device bug, fixed in `20ca7bbd08`, and one spot
  VM was preempted before the start.
- `harness.git_commit` reads "unknown" because the VM copy has no `.git`. `harness.sha256` e50881934b... equals
  `aggbench.results.harness_info()` of commit `20ca7bbd08` (clean), checked locally.
- `json/<env>-N<n>-<overlap>.json`: 30 result files, `aggbench.results.write_result` (schema-validated,
  re-validated locally with jsonschema). Brute-force oracle check verified in each: max abs diff 1e-13.
  `summary.json`, `run.log`: the raw run output. `make_tables.py` writes `tables.md` and `compact.md`
  from the JSONs. The README numbers are copied from them.
- Harness note for its owner: `aggbench.controls.random_weights` draws CPU noise, so a CUDA-resident
  module crashes it; the runner keeps its module on CPU between calls.
