Serves: attention as an information-aggregation mechanism that discounts shared information (the clearing view)

# EXPLORATORY (PROTOCOL 10.1): L3 looped clearing — does depth or iteration let attention discount overlapping sources?

**Track: EXPLORATORY.** Dev split only (train split for fitting, 20000 episodes; 2000 dev episodes, seeds 10^7 ..; the test split was never sampled). Not a finding; guides direction only.

**Result in one line:** a one-layer softmax readout (q . k_r only) fails where duplication dominates (E3-sybil: 0.044 nats above `oracle_clearing` vs 0.017 for an exact learned clearing solve); one self-attention layer already closes most of that gap (0.034), and stacking or looping it to T=8 adds at most ~0.01 nats (significant only in some configs); damped Jacobi needs T=8 to match one attention layer and fails at low overlap; only the exact learned solve gets clearly closer to the ceiling. Without duplicates (E2, E3-collude), every arm ties `extremized_log_pool`.

## Question

Deduplication needs the Gram matrix of keys (k_r . k_s). In E2/E3 the exact posterior given the reports has the form l0 + sum_i alpha_i (l_i - l0) with alpha = G^+ diag(G), G = C M C^T (coverage keys weighted by piece strength). A softmax readout attention layer with a learned query sees only q . k_r per agent, so it cannot discount overlapping sources. Does depth (stacked layers), iteration (one weight-tied layer looped T times) or an iterative solve of the clearing equation fix this, at matched parameter counts? Where does one layer break down as the number of agents and the overlap change?

## Arms (`arch.py`; all ~10.2k-11.7k parameters, matched to `stacked`, width 12, T=8)

Every arm maps keys only to per-agent weights alpha_i, and one shared readout combines reports: E2/E3 `logit = l0 + sum_i alpha_i (l_i - l0)` (the oracle's own form); E1 `prec = I + sum_i w_i (P_i - I)`, `info = sum_i w_i P_i m_i`, w = softplus(alpha).

| arm | what it is |
|---|---|
| `indep` | per-agent MLP, no cross-agent interaction (reference) |
| `readout_softmax` | (a) one standard softmax attention layer: learned query, alpha_i = s softmax_i(q . k_i); sees q . k_r only |
| `stacked_T{1,2,4,8}` | (b) T pre-LN self-attention blocks with independent weights (T=1 is the "one attention layer" that does see k_i . k_j inside softmax) |
| `looped_T{2,4,8}` | (c) one weight-tied self-attention block applied T times (T=1 equals `stacked_T1`) |
| `jacobi_T{1,2,4,8}` | (d) learned clearing system (K^T B K + Lambda) w = u with K rows = MLP(key_i) (p=16), B = diag(b_i), Lambda = diag(softplus(lam)), u = linear(maxpool trunk); T damped-Jacobi steps with learned step sizes from w=0; alpha_i = b_i k_i . w. Equivalent to solving (K^T B K + Lambda) w = K^T B dl and reading out u . w |
| `exact` | (e) the same learned system solved exactly (`torch.linalg.solve`) |

Training: log loss on train-split labels (no distillation), Adam, one-cycle lr 3e-3, 1000 steps x batch 128, train seed 0, CPU, 1 thread per worker. Baselines from `aggbench.baselines.all_baselines` (bayes, oracle_clearing, linear_pool, log_pool, sum_logodds, extremized_log_pool). Every learned arm goes through `aggbench.controls.run_with_controls` (untrained-init and random-network controls through the same `predict`).

Configs: E1, E2, E3-sybil, E3-collude at default settings, keys `full` and `noisy`; sweep on E2 and E3-sybil, `full` keys: n_agents 3 / 5 / 12 and pieces_per_agent 1-3 / 2-6 / 6-10 (overlap low / default / high). 16 result JSONs in `json/`, one per (config, visibility), each written and validated by `aggbench.results.write_result` (harness commit `992bc2bd`, `git_dirty` false, harness sha256 `e50881934b..`, brute-force oracle check on 24 smoke episodes).

## Reading the results

- **Where one layer fails.** The q . k readout is worst against `oracle_clearing` in E3-sybil whenever the copies are a large share of the evidence: N=3 (+0.018 nats vs one self-attention layer, paired CI [0.007, 0.030]) and low overlap, 1-3 pieces per agent (+0.033 [0.017, 0.049]). At N=12 or high overlap, where the union is mostly identified anyway, it ties. `indep` is worse still (it equals `extremized_log_pool`).
- **Depth and looping.** One self-attention layer does most of the de-duplication. T=2..8, stacked or looped, improves on it by at most 0.012 nats; the paired CI excludes 0 in a minority of (arm, config) pairs (e.g. stacked T=8 at N=3, N=12 and with noisy keys; looped T=4/8 at N=12; several arms at high overlap). Looped and stacked are indistinguishable at matched parameters.
- **Jacobi.** T=1..4 is worse than one attention layer on E3-sybil at default settings (+0.011 to +0.019); T=8 catches up at default settings and wins at N=12 (-0.016), but stays +0.037 worse at low overlap. Plain damped Jacobi converges too slowly for T <= 8 on these ill-conditioned systems.
- **Exact solve.** The learned exact clearing has the lowest mean regret in every E3-sybil config and beats one attention layer in each (paired -0.010 to -0.021) and in E2 full (-0.011), yet it stays 0.009-0.032 nats above `oracle_clearing` with full keys (it does not match the ceiling at this training budget).
- **No duplicates.** E2 (noisy keys), E3-collude: all arms within ~0.01 of `extremized_log_pool`; nothing to discount that a fitted global temperature does not already handle.
- **E1 is uninformative here.** The shared per-agent scalar-weight readout cannot represent the exact E1 posterior (correlated noise needs matrix weights), so every arm sits at the `extremized_log_pool` level (~0.83 nats) while `oracle_clearing` = bayes.
- **Noisy keys** shrink every gain (E3-sybil noisy: best arm `exact`, 0.042 above oracle, vs 0.055 for the readout).

## Caveats

One training seed per arm; a short CPU training budget (machine load ~50 on 16 cores); parameter matching within -5%/+10% of the budget (stacked_T4 is the largest, 11.5k). Training on labels, not distilling Bayes, leaves part of the gap to the oracle as training noise. Logits are clipped to [-50, 50] in `predict` for every arm and control alike (needed only by diverging random networks). A first run of the same code (commit 8b39f13) wrote JSONs while other lanes were committing harness changes, so its recorded harness commit was wrong and two files had `git_dirty` true; it was discarded and rerun from a clean worktree pinned at 992bc2bd. Every number in the rerun matched the first run exactly (max |diff| 0).

## Files

`arch.py` (arms), `run.py` (runner; `--only`, `--steps` etc.), `report.py` (writes `summary.json` and `tables.md`; the tables below are its output), `json/*.json` (results, one per config and visibility).

Reproduce: `cd /home/agent/workspaces/inconsistency-search && PYTHONPATH=.:/usr/lib/python3/dist-packages OMP_NUM_THREADS=1 /home/agent/.venvs/trader-cat/bin/python3 results/explore/l3-looped-clearing-20261020/run.py` (4 workers, ~15-25 min), then `PYTHONPATH=/usr/lib/python3/dist-packages python3 results/explore/l3-looped-clearing-20261020/report.py`.

## Tables (generated by report.py)

**Regret to bayes (nats/episode, 95% bootstrap CI, 2000 dev episodes), default configs**

| arm | params | E1 full | E1 noisy | E2 full | E2 noisy | E3-sybil full | E3-sybil noisy | E3-collude full | E3-collude noisy |
|---|---|---|---|---|---|---|---|---|---|
| indep | 10341 | 0.830 [0.774, 0.885] | 0.832 [0.775, 0.887] | 0.044 [0.032, 0.056] | 0.045 [0.033, 0.057] | 0.076 [0.060, 0.090] | 0.077 [0.061, 0.091] | 0.030 [0.020, 0.039] | 0.030 [0.020, 0.038] |
| readout_softmax | 10341 | 0.890 [0.839, 0.942] | 0.887 [0.837, 0.938] | 0.041 [0.030, 0.051] | 0.045 [0.033, 0.056] | 0.061 [0.047, 0.073] | 0.072 [0.057, 0.086] | 0.031 [0.022, 0.040] | 0.030 [0.021, 0.039] |
| stacked_T1 | 10201 | 0.835 [0.778, 0.891] | 0.840 [0.782, 0.897] | 0.047 [0.034, 0.060] | 0.046 [0.033, 0.057] | 0.051 [0.037, 0.064] | 0.073 [0.056, 0.088] | 0.032 [0.022, 0.041] | 0.030 [0.021, 0.039] |
| stacked_T2 | 10153 | 0.838 [0.781, 0.895] | 0.839 [0.782, 0.895] | 0.050 [0.037, 0.064] | 0.046 [0.033, 0.057] | 0.051 [0.038, 0.064] | 0.074 [0.059, 0.091] | 0.033 [0.023, 0.043] | 0.029 [0.020, 0.039] |
| stacked_T4 | 11467 | 0.829 [0.773, 0.885] | 0.830 [0.775, 0.886] | 0.045 [0.033, 0.058] | 0.045 [0.033, 0.056] | 0.046 [0.033, 0.059] | 0.066 [0.050, 0.080] | 0.031 [0.021, 0.040] | 0.030 [0.020, 0.039] |
| stacked_T8 | 10477 | 0.830 [0.775, 0.885] | 0.830 [0.773, 0.885] | 0.046 [0.034, 0.059] | 0.046 [0.034, 0.058] | 0.049 [0.034, 0.061] | 0.061 [0.045, 0.076] | 0.031 [0.022, 0.040] | 0.031 [0.022, 0.040] |
| looped_T2 | 10201 | 0.833 [0.776, 0.888] | 0.837 [0.779, 0.893] | 0.049 [0.037, 0.062] | 0.046 [0.034, 0.058] | 0.047 [0.034, 0.058] | 0.070 [0.054, 0.087] | 0.031 [0.021, 0.040] | 0.030 [0.020, 0.039] |
| looped_T4 | 10201 | 0.830 [0.774, 0.886] | 0.835 [0.778, 0.891] | 0.043 [0.032, 0.055] | 0.049 [0.037, 0.062] | 0.049 [0.036, 0.061] | 0.068 [0.052, 0.084] | 0.030 [0.021, 0.039] | 0.030 [0.020, 0.039] |
| looped_T8 | 10201 | 0.831 [0.774, 0.886] | 0.837 [0.780, 0.892] | 0.045 [0.033, 0.057] | 0.048 [0.035, 0.060] | 0.048 [0.035, 0.061] | 0.072 [0.055, 0.088] | 0.030 [0.020, 0.039] | 0.030 [0.021, 0.039] |
| jacobi_T1 | 10290 | 0.854 [0.793, 0.911] | 0.860 [0.799, 0.918] | 0.044 [0.031, 0.057] | 0.049 [0.037, 0.062] | 0.066 [0.050, 0.081] | 0.075 [0.058, 0.090] | 0.037 [0.026, 0.047] | 0.031 [0.020, 0.041] |
| jacobi_T2 | 10291 | 0.849 [0.790, 0.908] | 0.855 [0.796, 0.913] | 0.044 [0.031, 0.056] | 0.050 [0.038, 0.063] | 0.070 [0.055, 0.085] | 0.083 [0.066, 0.098] | 0.033 [0.022, 0.042] | 0.032 [0.021, 0.042] |
| jacobi_T4 | 10293 | 0.853 [0.794, 0.911] | 0.861 [0.799, 0.921] | 0.044 [0.030, 0.056] | 0.050 [0.037, 0.063] | 0.062 [0.048, 0.075] | 0.088 [0.071, 0.104] | 0.035 [0.024, 0.045] | 0.031 [0.021, 0.042] |
| jacobi_T8 | 10297 | 0.833 [0.775, 0.889] | 0.834 [0.776, 0.891] | 0.046 [0.032, 0.059] | 0.048 [0.035, 0.061] | 0.049 [0.035, 0.061] | 0.068 [0.053, 0.082] | 0.031 [0.021, 0.039] | 0.030 [0.020, 0.039] |
| exact | 10290 | 0.839 [0.782, 0.897] | 0.842 [0.786, 0.900] | 0.037 [0.025, 0.047] | 0.044 [0.032, 0.056] | 0.034 [0.022, 0.044] | 0.058 [0.044, 0.072] | 0.033 [0.023, 0.043] | 0.034 [0.024, 0.044] |
| extremized_log_pool | - | 0.827 [0.771, 0.883] | 0.827 [0.771, 0.883] | 0.045 [0.032, 0.056] | 0.045 [0.032, 0.056] | 0.078 [0.062, 0.092] | 0.078 [0.062, 0.092] | 0.030 [0.021, 0.039] | 0.030 [0.021, 0.039] |
| sum_logodds | - | 0.827 [0.771, 0.883] | 0.827 [0.771, 0.883] | 0.132 [0.099, 0.166] | 0.132 [0.099, 0.166] | 0.417 [0.345, 0.494] | 0.417 [0.345, 0.494] | 0.259 [0.208, 0.314] | 0.259 [0.208, 0.314] |
| oracle_clearing | - | -0.000 [-0.000, 0.000] | -0.000 [-0.000, 0.000] | 0.017 [0.009, 0.024] | 0.017 [0.009, 0.024] | 0.017 [0.009, 0.024] | 0.017 [0.009, 0.024] | 0.018 [0.010, 0.024] | 0.018 [0.010, 0.024] |

**Regret to oracle_clearing (paired difference arm - oracle_clearing, 95% CI), default configs**

| arm | params | E1 full | E1 noisy | E2 full | E2 noisy | E3-sybil full | E3-sybil noisy | E3-collude full | E3-collude noisy |
|---|---|---|---|---|---|---|---|---|---|
| indep | 10341 | 0.830 [0.774, 0.885] | 0.832 [0.775, 0.887] | 0.028 [0.018, 0.037] | 0.029 [0.019, 0.038] | 0.059 [0.046, 0.072] | 0.060 [0.047, 0.073] | 0.012 [0.005, 0.019] | 0.012 [0.005, 0.019] |
| readout_softmax | 10341 | 0.890 [0.839, 0.942] | 0.887 [0.837, 0.938] | 0.024 [0.016, 0.033] | 0.029 [0.019, 0.038] | 0.044 [0.032, 0.055] | 0.055 [0.042, 0.068] | 0.014 [0.007, 0.020] | 0.013 [0.006, 0.020] |
| stacked_T1 | 10201 | 0.835 [0.778, 0.891] | 0.840 [0.782, 0.897] | 0.031 [0.020, 0.042] | 0.029 [0.019, 0.039] | 0.034 [0.023, 0.046] | 0.056 [0.042, 0.070] | 0.014 [0.007, 0.021] | 0.013 [0.006, 0.019] |
| stacked_T2 | 10153 | 0.838 [0.781, 0.895] | 0.839 [0.782, 0.895] | 0.034 [0.023, 0.045] | 0.029 [0.019, 0.039] | 0.034 [0.022, 0.045] | 0.058 [0.044, 0.072] | 0.016 [0.008, 0.023] | 0.012 [0.005, 0.019] |
| stacked_T4 | 11467 | 0.829 [0.773, 0.885] | 0.830 [0.775, 0.886] | 0.028 [0.018, 0.039] | 0.028 [0.019, 0.037] | 0.029 [0.019, 0.040] | 0.049 [0.036, 0.062] | 0.013 [0.006, 0.021] | 0.012 [0.005, 0.019] |
| stacked_T8 | 10477 | 0.830 [0.775, 0.885] | 0.830 [0.773, 0.885] | 0.030 [0.020, 0.040] | 0.030 [0.020, 0.040] | 0.032 [0.021, 0.043] | 0.044 [0.032, 0.058] | 0.014 [0.007, 0.020] | 0.013 [0.006, 0.020] |
| looped_T2 | 10201 | 0.833 [0.776, 0.888] | 0.837 [0.779, 0.893] | 0.033 [0.022, 0.043] | 0.030 [0.020, 0.039] | 0.030 [0.020, 0.040] | 0.053 [0.039, 0.068] | 0.013 [0.006, 0.020] | 0.012 [0.005, 0.019] |
| looped_T4 | 10201 | 0.830 [0.774, 0.886] | 0.835 [0.778, 0.891] | 0.027 [0.017, 0.037] | 0.033 [0.023, 0.044] | 0.032 [0.022, 0.043] | 0.052 [0.038, 0.066] | 0.013 [0.005, 0.019] | 0.012 [0.005, 0.019] |
| looped_T8 | 10201 | 0.831 [0.774, 0.886] | 0.837 [0.780, 0.892] | 0.028 [0.019, 0.039] | 0.031 [0.021, 0.042] | 0.032 [0.021, 0.043] | 0.055 [0.041, 0.070] | 0.012 [0.005, 0.019] | 0.013 [0.006, 0.020] |
| jacobi_T1 | 10290 | 0.854 [0.793, 0.911] | 0.860 [0.799, 0.918] | 0.027 [0.017, 0.038] | 0.033 [0.023, 0.043] | 0.049 [0.035, 0.063] | 0.058 [0.044, 0.072] | 0.019 [0.011, 0.027] | 0.013 [0.006, 0.021] |
| jacobi_T2 | 10291 | 0.849 [0.790, 0.908] | 0.855 [0.796, 0.913] | 0.027 [0.016, 0.037] | 0.034 [0.023, 0.045] | 0.053 [0.040, 0.067] | 0.067 [0.051, 0.080] | 0.015 [0.008, 0.022] | 0.014 [0.006, 0.022] |
| jacobi_T4 | 10293 | 0.853 [0.794, 0.911] | 0.861 [0.799, 0.921] | 0.027 [0.016, 0.039] | 0.033 [0.023, 0.045] | 0.045 [0.033, 0.057] | 0.072 [0.056, 0.086] | 0.017 [0.010, 0.025] | 0.014 [0.006, 0.022] |
| jacobi_T8 | 10297 | 0.833 [0.775, 0.889] | 0.834 [0.776, 0.891] | 0.029 [0.018, 0.041] | 0.032 [0.021, 0.043] | 0.032 [0.021, 0.042] | 0.051 [0.039, 0.064] | 0.013 [0.006, 0.020] | 0.012 [0.006, 0.019] |
| exact | 10290 | 0.839 [0.782, 0.897] | 0.842 [0.786, 0.900] | 0.020 [0.011, 0.028] | 0.028 [0.019, 0.038] | 0.017 [0.009, 0.025] | 0.042 [0.030, 0.053] | 0.016 [0.009, 0.023] | 0.016 [0.009, 0.023] |
| extremized_log_pool | - | - | - | - | - | - | - | - | - |
| sum_logodds | - | 0.827 [0.771, 0.883] | 0.827 [0.771, 0.883] | 0.116 [0.086, 0.148] | 0.116 [0.086, 0.148] | 0.400 [0.330, 0.473] | 0.400 [0.330, 0.473] | 0.241 [0.191, 0.294] | 0.241 [0.191, 0.294] |
| oracle_clearing | - | - | - | - | - | - | - | - | - |

**E2, full keys: regret to oracle_clearing across agents and overlap**

| arm | params | N=3 | N=5 (default) | N=12 | 1-3 pieces (low overlap) | 2-6 pieces (default) | 6-10 pieces (high overlap) |
|---|---|---|---|---|---|---|---|
| indep | 10531 | 0.018 [0.011, 0.026] | 0.028 [0.018, 0.037] | 0.039 [0.028, 0.050] | 0.028 [0.017, 0.039] | 0.028 [0.018, 0.037] | 0.015 [0.009, 0.022] |
| readout_softmax | 10531 | 0.014 [0.008, 0.021] | 0.024 [0.016, 0.033] | 0.041 [0.030, 0.051] | 0.029 [0.019, 0.040] | 0.024 [0.016, 0.033] | 0.013 [0.008, 0.019] |
| stacked_T1 | 10575 | 0.021 [0.013, 0.030] | 0.031 [0.020, 0.042] | 0.038 [0.027, 0.050] | 0.027 [0.015, 0.037] | 0.031 [0.020, 0.042] | 0.016 [0.010, 0.023] |
| stacked_T2 | 10417 | 0.019 [0.011, 0.028] | 0.034 [0.023, 0.045] | 0.039 [0.028, 0.051] | 0.027 [0.015, 0.038] | 0.034 [0.023, 0.045] | 0.016 [0.009, 0.022] |
| stacked_T4 | 11665 | 0.019 [0.011, 0.027] | 0.028 [0.018, 0.039] | 0.037 [0.026, 0.048] | 0.027 [0.015, 0.038] | 0.028 [0.018, 0.039] | 0.015 [0.009, 0.021] |
| stacked_T8 | 10609 | 0.016 [0.009, 0.023] | 0.030 [0.020, 0.040] | 0.039 [0.028, 0.051] | 0.028 [0.017, 0.040] | 0.030 [0.020, 0.040] | 0.016 [0.010, 0.022] |
| looped_T2 | 10575 | 0.020 [0.012, 0.028] | 0.033 [0.022, 0.043] | 0.038 [0.027, 0.049] | 0.029 [0.017, 0.041] | 0.033 [0.022, 0.043] | 0.016 [0.010, 0.023] |
| looped_T4 | 10575 | 0.015 [0.006, 0.024] | 0.027 [0.017, 0.037] | 0.041 [0.029, 0.053] | 0.026 [0.015, 0.038] | 0.027 [0.017, 0.037] | 0.016 [0.009, 0.022] |
| looped_T8 | 10575 | 0.015 [0.007, 0.023] | 0.028 [0.019, 0.039] | 0.042 [0.030, 0.055] | 0.028 [0.017, 0.039] | 0.028 [0.019, 0.039] | 0.016 [0.009, 0.022] |
| jacobi_T1 | 10736 | 0.016 [0.008, 0.023] | 0.027 [0.017, 0.038] | 0.045 [0.033, 0.057] | 0.024 [0.013, 0.034] | 0.027 [0.017, 0.038] | 0.017 [0.011, 0.024] |
| jacobi_T2 | 10737 | 0.014 [0.008, 0.021] | 0.027 [0.016, 0.037] | 0.041 [0.028, 0.053] | 0.028 [0.016, 0.039] | 0.027 [0.016, 0.037] | 0.016 [0.009, 0.023] |
| jacobi_T4 | 10739 | 0.015 [0.009, 0.022] | 0.027 [0.016, 0.039] | 0.039 [0.027, 0.052] | 0.029 [0.018, 0.041] | 0.027 [0.016, 0.039] | 0.018 [0.010, 0.025] |
| jacobi_T8 | 10743 | 0.013 [0.007, 0.020] | 0.029 [0.018, 0.041] | 0.042 [0.029, 0.055] | 0.027 [0.016, 0.040] | 0.029 [0.018, 0.041] | 0.013 [0.007, 0.018] |
| exact | 10736 | 0.013 [0.007, 0.020] | 0.020 [0.011, 0.028] | 0.031 [0.021, 0.041] | 0.017 [0.008, 0.027] | 0.020 [0.011, 0.028] | 0.011 [0.005, 0.017] |
| extremized_log_pool | - | - | - | - | - | - | - |
| sum_logodds | - | 0.050 [0.034, 0.067] | 0.116 [0.086, 0.148] | 0.405 [0.327, 0.483] | 0.055 [0.037, 0.074] | 0.116 [0.086, 0.148] | 0.253 [0.196, 0.311] |
| oracle_clearing | - | - | - | - | - | - | - |

**E3-sybil, full keys: regret to oracle_clearing across agents and overlap**

| arm | params | N=3 | N=5 (default) | N=12 | 1-3 pieces (low overlap) | 2-6 pieces (default) | 6-10 pieces (high overlap) |
|---|---|---|---|---|---|---|---|
| indep | 10531 | 0.063 [0.051, 0.076] | 0.059 [0.046, 0.072] | 0.053 [0.042, 0.066] | 0.077 [0.060, 0.094] | 0.059 [0.046, 0.072] | 0.022 [0.014, 0.030] |
| readout_softmax | 10531 | 0.043 [0.033, 0.054] | 0.044 [0.032, 0.055] | 0.054 [0.043, 0.067] | 0.066 [0.051, 0.082] | 0.044 [0.032, 0.055] | 0.017 [0.010, 0.024] |
| stacked_T1 | 10575 | 0.025 [0.015, 0.034] | 0.034 [0.023, 0.046] | 0.053 [0.040, 0.067] | 0.033 [0.022, 0.045] | 0.034 [0.023, 0.046] | 0.022 [0.014, 0.030] |
| stacked_T2 | 10417 | 0.025 [0.016, 0.034] | 0.034 [0.022, 0.045] | 0.046 [0.033, 0.058] | 0.030 [0.018, 0.041] | 0.034 [0.022, 0.045] | 0.016 [0.009, 0.024] |
| stacked_T4 | 11665 | 0.018 [0.010, 0.027] | 0.029 [0.019, 0.040] | 0.045 [0.032, 0.058] | 0.033 [0.022, 0.046] | 0.029 [0.019, 0.040] | 0.014 [0.008, 0.022] |
| stacked_T8 | 10609 | 0.016 [0.008, 0.024] | 0.032 [0.021, 0.043] | 0.043 [0.032, 0.056] | 0.027 [0.016, 0.038] | 0.032 [0.021, 0.043] | 0.016 [0.009, 0.023] |
| looped_T2 | 10575 | 0.023 [0.014, 0.031] | 0.030 [0.020, 0.040] | 0.050 [0.037, 0.063] | 0.032 [0.021, 0.044] | 0.030 [0.020, 0.040] | 0.021 [0.013, 0.030] |
| looped_T4 | 10575 | 0.025 [0.015, 0.036] | 0.032 [0.022, 0.043] | 0.044 [0.032, 0.057] | 0.027 [0.016, 0.038] | 0.032 [0.022, 0.043] | 0.017 [0.010, 0.025] |
| looped_T8 | 10575 | 0.016 [0.009, 0.023] | 0.032 [0.021, 0.043] | 0.045 [0.033, 0.058] | 0.033 [0.021, 0.045] | 0.032 [0.021, 0.043] | 0.021 [0.013, 0.030] |
| jacobi_T1 | 10736 | 0.045 [0.032, 0.058] | 0.049 [0.035, 0.063] | 0.055 [0.042, 0.068] | 0.054 [0.040, 0.069] | 0.049 [0.035, 0.063] | 0.020 [0.012, 0.028] |
| jacobi_T2 | 10737 | 0.046 [0.035, 0.058] | 0.053 [0.040, 0.067] | 0.056 [0.043, 0.070] | 0.064 [0.048, 0.081] | 0.053 [0.040, 0.067] | 0.018 [0.010, 0.027] |
| jacobi_T4 | 10739 | 0.036 [0.027, 0.047] | 0.045 [0.033, 0.057] | 0.045 [0.034, 0.056] | 0.069 [0.052, 0.087] | 0.045 [0.033, 0.057] | 0.016 [0.010, 0.023] |
| jacobi_T8 | 10743 | 0.037 [0.027, 0.047] | 0.032 [0.021, 0.042] | 0.037 [0.027, 0.048] | 0.070 [0.053, 0.087] | 0.032 [0.021, 0.042] | 0.015 [0.008, 0.022] |
| exact | 10736 | 0.015 [0.009, 0.021] | 0.017 [0.009, 0.025] | 0.032 [0.022, 0.042] | 0.014 [0.006, 0.021] | 0.017 [0.009, 0.025] | 0.009 [0.004, 0.015] |
| extremized_log_pool | - | - | - | - | - | - | - |
| sum_logodds | - | 0.401 [0.336, 0.473] | 0.400 [0.330, 0.473] | 0.636 [0.534, 0.755] | 0.302 [0.244, 0.362] | 0.400 [0.330, 0.473] | 0.545 [0.445, 0.647] |
| oracle_clearing | - | - | - | - | - | - | - |

**Paired regret difference arm - stacked_T1 (one self-attention layer), 95% CI; negative = better than one layer**

| arm | E3-sybil full | E3-sybil noisy | E3-sybil n_agents=3 full | E3-sybil n_agents=12 full | E3-sybil pieces_per_agent=1-3 full | E3-sybil pieces_per_agent=6-10 full | E2 full | E3-collude full |
|---|---|---|---|---|---|---|---|---|
| indep | 0.025 [0.014, 0.037] | 0.004 [-0.005, 0.012] | 0.038 [0.027, 0.051] | 0.000 [-0.006, 0.007] | 0.044 [0.027, 0.061] | 0.000 [-0.002, 0.002] | -0.003 [-0.008, 0.002] | -0.002 [-0.004, -0.000] |
| readout_softmax | 0.010 [-0.001, 0.021] | -0.001 [-0.009, 0.007] | 0.018 [0.007, 0.030] | 0.001 [-0.007, 0.008] | 0.033 [0.017, 0.049] | -0.005 [-0.011, 0.001] | -0.006 [-0.012, -0.001] | -0.000 [-0.003, 0.002] |
| stacked_T2 | -0.000 [-0.008, 0.009] | 0.002 [-0.007, 0.011] | -0.000 [-0.008, 0.008] | -0.008 [-0.017, 0.001] | -0.003 [-0.012, 0.005] | -0.005 [-0.009, -0.001] | 0.003 [-0.003, 0.010] | 0.002 [-0.001, 0.004] |
| stacked_T4 | -0.005 [-0.013, 0.004] | -0.007 [-0.018, 0.003] | -0.006 [-0.015, 0.002] | -0.008 [-0.017, 0.001] | 0.000 [-0.009, 0.009] | -0.007 [-0.012, -0.002] | -0.003 [-0.008, 0.003] | -0.001 [-0.002, 0.001] |
| stacked_T8 | -0.002 [-0.012, 0.008] | -0.012 [-0.023, -0.001] | -0.009 [-0.017, -0.001] | -0.010 [-0.019, -0.001] | -0.006 [-0.016, 0.003] | -0.006 [-0.013, 0.002] | -0.001 [-0.007, 0.004] | -0.000 [-0.002, 0.001] |
| looped_T2 | -0.004 [-0.012, 0.004] | -0.003 [-0.010, 0.004] | -0.002 [-0.009, 0.005] | -0.003 [-0.009, 0.002] | -0.000 [-0.008, 0.006] | -0.000 [-0.001, 0.001] | 0.002 [-0.003, 0.007] | -0.001 [-0.002, -0.000] |
| looped_T4 | -0.002 [-0.011, 0.008] | -0.004 [-0.013, 0.004] | 0.001 [-0.008, 0.010] | -0.009 [-0.016, -0.003] | -0.006 [-0.013, 0.003] | -0.005 [-0.008, -0.001] | -0.004 [-0.008, 0.001] | -0.002 [-0.003, -0.000] |
| looped_T8 | -0.002 [-0.012, 0.007] | -0.001 [-0.010, 0.009] | -0.009 [-0.016, -0.001] | -0.008 [-0.016, -0.000] | -0.000 [-0.009, 0.009] | -0.001 [-0.002, 0.001] | -0.002 [-0.008, 0.004] | -0.002 [-0.003, -0.000] |
| jacobi_T1 | 0.015 [0.003, 0.026] | 0.002 [-0.006, 0.010] | 0.020 [0.008, 0.034] | 0.002 [-0.006, 0.009] | 0.021 [0.008, 0.036] | -0.001 [-0.006, 0.003] | -0.004 [-0.009, 0.002] | 0.005 [0.000, 0.010] |
| jacobi_T2 | 0.019 [0.008, 0.030] | 0.010 [0.002, 0.018] | 0.021 [0.009, 0.034] | 0.003 [-0.003, 0.010] | 0.031 [0.015, 0.048] | -0.003 [-0.008, 0.001] | -0.004 [-0.008, 0.001] | 0.001 [-0.002, 0.004] |
| jacobi_T4 | 0.011 [0.000, 0.022] | 0.016 [0.007, 0.024] | 0.011 [0.001, 0.022] | -0.008 [-0.016, -0.001] | 0.037 [0.020, 0.054] | -0.005 [-0.010, -0.001] | -0.004 [-0.009, 0.002] | 0.003 [-0.000, 0.007] |
| jacobi_T8 | -0.002 [-0.011, 0.007] | -0.005 [-0.013, 0.003] | 0.012 [0.001, 0.022] | -0.016 [-0.024, -0.009] | 0.037 [0.019, 0.055] | -0.007 [-0.011, -0.002] | -0.001 [-0.007, 0.004] | -0.001 [-0.003, 0.001] |
| exact | -0.017 [-0.026, -0.007] | -0.014 [-0.024, -0.006] | -0.010 [-0.019, -0.001] | -0.021 [-0.033, -0.009] | -0.019 [-0.029, -0.008] | -0.013 [-0.020, -0.005] | -0.011 [-0.018, -0.003] | 0.002 [-0.005, 0.008] |

**Controls** (random-network and untrained-init, same predict pipeline): number of (config, arm) pairs where the control's mean regret to bayes is <= the trained arm's CI upper bound; and the best control regret per config.

- control <= trained CI upper bound: 25 of 448 pairs (E1__full:indep:random, E1__full:indep:untrained, E1__full:jacobi_T1:untrained, E1__full:jacobi_T2:random, E1__full:jacobi_T4:random, E1__noisy:indep:random, E1__noisy:indep:untrained, E1__noisy:jacobi_T1:untrained, E1__noisy:jacobi_T2:random, E1__noisy:jacobi_T4:random, E2__full:readout_softmax:random, E2__full:jacobi_T4:random, E2__n_agents=12__full:readout_softmax:random, E2__n_agents=3__full:readout_softmax:random, E2__noisy:readout_softmax:random, E2__noisy:jacobi_T4:random, E2__pieces_per_agent=1-3__full:readout_softmax:random, E2__pieces_per_agent=1-3__full:stacked_T4:untrained, E2__pieces_per_agent=1-3__full:stacked_T8:untrained, E2__pieces_per_agent=6-10__full:readout_softmax:random, E3-collude__full:readout_softmax:random, E3-collude__noisy:readout_softmax:random, E3-sybil__n_agents=12__full:readout_softmax:random, E3-sybil__noisy:readout_softmax:random, E3-sybil__pieces_per_agent=6-10__full:readout_softmax:random)
- best (lowest) control regret per config: E1__full 0.870 (jacobi_T4, random_network); E1__noisy 0.869 (indep, random_network); E2__full 0.051 (readout_softmax, random_network); E2__n_agents=12__full 0.047 (readout_softmax, random_network); E2__n_agents=3__full 0.039 (readout_softmax, random_network); E2__noisy 0.049 (readout_softmax, random_network); E2__pieces_per_agent=1-3__full 0.043 (stacked_T4, untrained_init); E2__pieces_per_agent=6-10__full 0.026 (readout_softmax, random_network); E3-collude__full 0.031 (readout_softmax, random_network); E3-collude__noisy 0.031 (readout_softmax, random_network); E3-sybil__full 0.086 (readout_softmax, random_network); E3-sybil__n_agents=12__full 0.061 (readout_softmax, random_network); E3-sybil__n_agents=3__full 0.084 (readout_softmax, random_network); E3-sybil__noisy 0.082 (readout_softmax, random_network); E3-sybil__pieces_per_agent=1-3__full 0.096 (readout_softmax, random_network); E3-sybil__pieces_per_agent=6-10__full 0.033 (readout_softmax, random_network)
