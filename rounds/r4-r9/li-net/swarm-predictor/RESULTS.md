# Swarm predictor on ShipHunt: results

eurisko study, 2026-10-05. Pre-registration: PROPOSAL.md, approved by Kelvin at about 01:40Z with a $40 GPU cap. Written by an Opus subagent for eurisko. Status: **stopped at Gate 0 by the pre-registered rule.** No main run took place. No swarm VM is live: `gcloud compute instances list` with the swarm filter returned 0 at 03:19Z and again at 03:23Z.

## 1. TL;DR

- **Verdict by the pre-registered rule: stopped at Gate 0.** Neither Qwen3-0.6B nor Qwen3-1.7B writes claims that beat the rules-only prior against exact Bayes. Pooled informativeness per claim is -0.384 for 0.6B and -0.137 for 1.7B, and both 95% CIs are below 0 (`stage0_summary.json`). PROPOSAL section 6 says: "If neither passes, stop and report: there is nothing to pool." So the primary test was not run (settlement-trained swarm against mean and vote at n = 16, L = 256, R = 2). The hypothesis is untested at this scale, neither supported nor refuted.
- The chains are close to noise. The rank correlation of claims with exact Bayes is -0.009 for 0.6B and +0.04 to +0.09 for 1.7B. Many claims copy the example numbers in the prompt.
- The pool handles such chains sensibly. Its learned stakes fall to kappa = 0.004-0.19, and it ends slightly below the prior (1.7B: -2.7 x 10^-3 Brier against Bayes, CI below 0). It also ends far below mean and vote (-15 to -48 x 10^-3), but that gap is shrinkage toward the prior, not pooled information, because mean and vote are worse than the prior. A future primary test needs the prior as a baseline.
- Settlement still ranks chains correctly. Within a board, a chain's payment correlates with its true informativeness at Spearman +0.43 to +0.52 (100 boards per cell). The payment signal works on real LM chains even when every chain is bad.
- Generation ran 4 to 5 times slower than assumed: 0.14-0.19 s per decode step against the planned 0.035 s. At that speed the full plan would cost about $68-71, so even a passing gate would have forced the cut plan (S0 and S1 only, about $24-25).
- GPU total: **$1.25** for two L4 rentals (GPU-LOG.md), against the $40 cap.

## 2. What ran

1. A CPU check of the pooling and settlement math on synthetic chains, before any GPU (RESULTS-SMOKE.md).
2. The code: `pool.py`, `settle.py`, `refocus.py`, `run_swarm.py` (the gate0, train, eval, train+eval and genbench modes), `analyze.py` and `stage0_report.py`, plus `gpu/` (launch, queue, driver and remote scripts). The harness was imported unchanged; `git -C harness status` is clean.
3. CPU tests: every mode with the harness's tiny random model, and real Qwen3-0.6B chains on CPU. `analyze.py` was tested on synthetic chain records with the real grid structure.
4. Stage 0, as two L4 rentals:
   - `swarm-stage0`: dev-board prompt choice and Gate 0 for 0.6B. It was stopped once the gate cells had synced; the 1.7B part could not have finished in time.
   - `swarm-stage0b`: a generation-speed benchmark on 1.7B, then dev-board prompt choice and Gate 0 for 1.7B. It was stopped after the gate cells, skipping the 1.7B timing steps.
5. Not run: Stage 1 (training) and Stage 2 (the full evaluation grid), because of the Gate 0 stop rule.

Every change to the plan is in PROPOSAL.md section 11 (deviations log), with timestamps. The ones that matter here, all fixed before Gate 0:
- L counts reasoning tokens, plus an allowance of 16 answer tokens per statement, because six answer lines take 89 tokens.
- Sampling uses temperature 0.7.
- The prompt was chosen between two variants on 30 separate dev boards, after a CPU check showed 0.6B copying the prompt's example numbers.
- The prompt states the base rate: 5 of the 20 cells hold a ship.

## 3. Gate 0

Gate 0 passes for a model if at least 80% of chains give all six claims and the mean informativeness per claim is above 0 (95% CI). Informativeness per claim is E_bayes[log q(y) / p0(y)], so 0 means "as good as the rules-only prior". Cells: 100 gate boards, 16 untrained chains per board, R = 1. Sources: `runs/gate0-q06/gate0.json`, `runs/gate0-q17/gate0.json`, `stage0_summary.json`; the pooled CIs are recomputed from `runs/*/gate0_chains.jsonl` by `stage0_report.py`.

| model | prompt (dev-board parse rate A / B) | L | chains | all claims parsed | informativeness per claim [95% CI] | rank correlation of claims with exact Bayes |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | B (55% / 88%) | 64 | 1600 | 76.9% | -0.376 [-0.390, -0.361] | -0.009 |
| Qwen3-0.6B | B (55% / 88%) | 256 | 1600 | 87.2% | -0.392 [-0.406, -0.378] | -0.009 |
| Qwen3-0.6B, both L pooled | | | 3200 | 82.1% | -0.384 [-0.394, -0.374] | gate: FAIL |
| Qwen3-1.7B | A (75% / 75%) | 64 | 1600 | 99.9% | -0.128 [-0.135, -0.121] | +0.038 |
| Qwen3-1.7B | A (75% / 75%) | 256 | 1600 | 78.0% | -0.149 [-0.159, -0.139] | +0.087 |
| Qwen3-1.7B, both L pooled | | | 3200 | 89.0% | -0.137 [-0.143, -0.131] | gate: FAIL |

Both models fail on informativeness. 1.7B passes the parse-rate half of the gate. Read against the true boards, the chains rarely use the evidence. One 1.7B chain claims 0.70 (the prompt's example) for a row that exact Bayes puts at 0.00, then 0.00 for a cell that Bayes puts at 0.56. Another writes the descending ladder 0.65, 0.50, 0.45, ... whatever the statements say (`/tmp/swarm-predictor/text/gate0-q17/gate0_texts.jsonl`).

## 4. What the pool and settlement do with these chains (descriptive, not pre-registered tests)

Brier against exact Bayes, x 1000, on the gate cells. D-fit uses stakes cross-fitted over boards (2 folds), and the CIs are paired bootstraps over the 100 boards (`gate_pool_ci.json`, from `gate_pool_ci.py`).

| model | L | prior | mean | vote | D-fit | D-fit minus prior [95% CI] | D-fit minus mean [95% CI] | D-fit minus vote [95% CI] | learned kappa (2 folds) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 64 | 54.7 | 82.1 | 101.3 | 53.4 | -1.3 [-1.7, -0.8] | -28.8 [-32.3, -25.4] | -47.9 [-53.0, -43.1] | 0.017, 0.068 |
| Qwen3-0.6B | 256 | 54.7 | 83.8 | 102.2 | 54.2 | -0.5 [-0.7, -0.3] | -29.6 [-33.1, -26.1] | -48.0 [-53.5, -42.6] | 0.004, 0.008 |
| Qwen3-1.7B | 64 | 54.7 | 67.3 | 70.0 | 51.9 | -2.7 [-3.5, -2.0] | -15.4 [-17.4, -13.5] | -18.1 [-20.9, -15.4] | 0.043, 0.191 |
| Qwen3-1.7B | 256 | 54.7 | 72.3 | 80.3 | 51.9 | -2.7 [-3.5, -1.9] | -20.4 [-22.5, -18.2] | -28.4 [-32.0, -24.7] | 0.119, 0.056 |

The pool learns to ignore chains that carry nothing: kappa falls toward 0 and D-fit stays within 3 x 10^-3 of the prior. For 1.7B it beats the prior by a small, clear margin, 2.7 x 10^-3 with a CI below 0, so some weak signal survives learned stakes plus the board projection.

The large gaps to mean and vote say little about pooling. Mean and vote are worse than the prior, 67-102 against 54.7, because the claims are miscalibrated: on the gate cells the average claim is 0.41-0.45 where exact Bayes averages 0.27 (`stage0_summary.json`, detail). Any pool that can shrink to the prior beats them. This is the trivial win that Gate 0 was meant to block. If the primary test is ever run, its baselines should include the rules-only prior and a recalibrated mean.

Stale prices and per-chain credit on the same cells (`stage0_summary.json`, detail):
- A "stale" statement is an open statement that no chain was asked about (12 of the 24 open statements were asked).
- Credit is each chain's settlement at the D-fit prices, set against its oracle informativeness.

| model | L | stale statements: prior | stale: D-fit | settlement vs informativeness, mean within-board Spearman (boards) | share of chains paid > 0 | payment 5% / 50% / 95% |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 64 | 42.2 | 40.2 | +0.43 (100) | 45% | -2.91 / -0.28 / +3.55 |
| Qwen3-0.6B | 256 | 42.6 | 41.9 | +0.45 (100) | 47% | -3.01 / -0.18 / +3.58 |
| Qwen3-1.7B | 64 | 42.2 | 38.2 | +0.52 (100) | 48% | -1.88 / -0.08 / +2.34 |
| Qwen3-1.7B | 256 | 42.6 | 38.5 | +0.50 (100) | 50% | -1.83 / +0.01 / +2.16 |

Through the board constraints, D-fit improves the statements nobody was asked about (1.7B: 38.2 against the prior's 42.2). Settlement pays the less bad chains more: the within-board Spearman between payment and oracle informativeness is +0.43 to +0.52, close to the 0.11-0.41 the CPU check found for synthetic chains (RESULTS-SMOKE.md). About half the chains are paid more than 0 and the median payment is near 0. That is what a market priced at the pool's own beliefs should give.

## 5. Martingale drift

Not measured. The drift between rounds needs rounds 2 and 3, which run only in the main experiment, and the stop rule skipped it. With learned stakes near 0, the pool would sit at the prior in every round, so a drift table from these chains would show about 0 and carry no information. No table is given rather than a meaningless one.

## 6. Speed and cost

| measurement | setting | seconds per decode step | file |
|---|---|---|---|
| Qwen3-0.6B Gate 0 cell L = 64 | dynamic cache, batches up to 145 chains, 160 new tokens | 0.176 | runs/gate0-q06/gate0.json |
| Qwen3-0.6B Gate 0 cell L = 256 | dynamic cache, batches up to 107 chains, 352 new tokens | 0.136 | runs/gate0-q06/gate0.json |
| Qwen3-1.7B Gate 0 cell L = 64 | dynamic cache, batches up to 145 chains, 160 new tokens | 0.192 | runs/gate0-q17/gate0.json |
| Qwen3-1.7B Gate 0 cell L = 256 | dynamic cache, batches up to 107 chains, 352 new tokens | 0.147 | runs/gate0-q17/gate0.json |
| Qwen3-1.7B benchmark | dynamic cache, batch 128, 352 new tokens | 0.144 | runs/genbench/genbench.json |
| Qwen3-1.7B benchmark | dynamic cache, batch 32, 352 new tokens | 0.081 | runs/genbench/genbench.json |
| Qwen3-1.7B benchmark | static cache, batch 128, 352 new tokens (includes compilation) | 0.283 | runs/genbench/genbench.json |
| Qwen3-1.7B benchmark | static-compile cache, batch 128, 352 new tokens | failed in both repetitions (CUDA graph error) | runs/genbench/genbench.json |

On 1.7B the default dynamic cache was the fastest option. Quadrupling the batch (32 to 128) only raised the step time from 0.081 to 0.144 s. A plausible cause, inferred from the timings and not profiled, is that the default cache re-copies itself every step and every step has fixed overhead. The static-cache run includes its compile time, and the forced CUDA-graph compile failed.

A preallocated cache that writes in place (`run_swarm.py --gen-impl prealloc`) gives greedy outputs identical to the default cache on CPU with Qwen3-0.6B. It was never timed on GPU.

GRPO steps measured (2 rounds x 8 boards x 8 chains):

| model | step | L | step seconds | generation | loss | file |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 1 | 64 | 53.4 | 38.2 | 15.2 | runs/gate0-q06/timing_train/steps.jsonl |
| Qwen3-0.6B | 2 | 64 | 53.3 | 38.2 | 15.1 | runs/gate0-q06/timing_train/steps.jsonl |
| Qwen3-0.6B | 3 | 256 | 116.6 | 95.5 | 21.0 | runs/gate0-q06/timing_train/steps.jsonl |
| Qwen3-1.7B | 1 | 64 | 72.3 | 41.2 | 31.1 | runs/gate0-q17/timing_train/steps.jsonl |
| Qwen3-1.7B | 2 | 64 | 71.0 | 41.0 | 30.0 | runs/gate0-q17/timing_train/steps.jsonl |

Re-costed plans from these measurements (`stage0_summary.json`, `stage0_report.py`). The step times come mostly or only from L = 64 steps (1.7B only at L = 64), so these rows are optimistic for the planned 50/50 mix of L = 64 and 256:

| model timings | plan | GRPO step (s) | training per arm-seed (h) | evaluation per checkpoint (h) | total L4-h | $ at $0.87/h |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | PROPOSAL full plan | 74 | 3.67 | 2.58 | 78.7 | 68.4 |
| Qwen3-0.6B | pre-registered cuts: no S2, long chains S1 only, 80 boards, 100 steps | 74 | 2.64 | 2.06 | 42.2 | 36.7 |
| Qwen3-0.6B | cut B: S0+S1 only, smaller grid, long chains S1 only, 150 steps | 74 | 3.67 | 1.64 | 28.0 | 24.3 |
| Qwen3-0.6B | cut B with 100 steps | 74 | 2.64 | 1.64 | 24.6 | 21.4 |
| Qwen3-1.7B | PROPOSAL full plan | 72 | 3.60 | 2.80 | 81.3 | 70.7 |
| Qwen3-1.7B | pre-registered cuts: no S2, long chains S1 only, 80 boards, 100 steps | 72 | 2.61 | 2.24 | 44.1 | 38.4 |
| Qwen3-1.7B | cut B: S0+S1 only, smaller grid, long chains S1 only, 150 steps | 72 | 3.60 | 1.77 | 29.2 | 25.4 |
| Qwen3-1.7B | cut B with 100 steps | 72 | 2.61 | 1.77 | 25.9 | 22.6 |

GPU spend (GPU-LOG.md DELETE lines; `analysis.json: gpu` gives the same total):

| rental | VM | hours | cost (rate ceiling x age) |
|---|---|---|---|
| swarm-stage0 | soup-gpu-swarm-stage0-1005015714-13490 | 0.63 | $0.55 |
| swarm-stage0b | soup-gpu-swarm-stage0b-1005023045-16412 | 0.80 | $0.70 |
| total | | | $1.25 |

## 7. What would make the test runnable (proposals, not done)

1. A model whose untrained chains beat the prior. Run Gate 0 on Qwen3-4B (the harness ran 4B on an L4 at 21 GB) or on an 8B model on an A100, before anything else. One gate rental costs on the order of the 1.7B rental above ($0.70).
2. Longer traces in Qwen3's thinking mode. This leaves the "short chain" design and multiplies the generation cost.
3. Faster generation before any training run: time the preallocated cache on GPU, or sample with vLLM.
4. A fairer primary comparison: add the rules-only prior and a recalibrated mean (fitted on held-out boards) as baselines.
5. A separate, cheaper question that this data makes plausible: can GRPO on settlement alone teach 1.7B to beat the prior? The payment already ranks chains correctly (section 4). It would need its own pre-registration.

## 8. Files

- `PROPOSAL.md`: pre-registration and deviations log (section 11). Rendered: `memo-serve/lir-agent/swarm-proposal.html`.
- `RESULTS-SMOKE.md`, `smoke_results.json`, `smoke_tables.md`: the CPU check.
- `stage0_summary.json` (from `stage0_report.py`): gate verdicts, per-cell detail, re-costed plans.
- `gate_pool_ci.json` (from `gate_pool_ci.py`): pool comparison CIs.
- `analysis.json` (from `analyze.py`): gate files and the GPU total.
- `runs/gate0-q06/`, `runs/gate0-q17/`, `runs/genbench/`: synced claim records, gate summaries and timings.
- Chain texts and logs, not on /srv: `/tmp/swarm-predictor/text/`, `/tmp/swarm-predictor/logs/`.
- Code: `pool.py`, `settle.py`, `refocus.py`, `run_swarm.py`, `analyze.py`, `stage0_report.py`, `gate_pool_ci.py`, `estimate_cost.py`, `gpu/`.
