# PREREG E6 (reshaped)   2026-10-03 08:30Z, Opus 5.5 subagent for eurisko, hawk file: ../../hawks/E6.md

Paper modified: Gao, Schulman, Hilton 2022, "Scaling Laws for Reward Model Overoptimization",
https://arxiv.org/abs/2210.10760 (title/authors re-fetched by the hawk 2026-10-03; gold-vs-proxy RM design).
Modification (one sentence): gold = ShipHunt's grounded env_reward; proxy = a linear head on the agent's
own forecast-adapter (or fused-adapter) features of environment-written outcome text, trained
prequentially on realised env_reward; drift measured on a frozen bank against yoked truth-only controls.
Claim under test: reward gradients that reach the belief/forecast parameters (fused B0) cause directed,
self-serving drift of a co-trained latent utility proxy beyond what the same data stream causes without
reward (SSI(d) - SSI(b_d) > 0), and the LIR control mask removes it (SSI(c) - SSI(b_c) ~ 0) without
costing gold accuracy (gold_acc(c) >= gold_acc(b_c) - 0.02).

Hawk verdict RESHAPE, all items adopted: Gao-style gold vs proxy; arms a, c, d, b_c, b_d; fixed
branched multi-policy bank with environment-written template text frozen before any run
(`bank.json`, hash c3c91631775a4f45, 16 boards x 16 outcomes, built by `build_bank.py`); gold pairwise
accuracy on non-tied pairs within groups; SSI; within-tie reordering reported separately; matched
adapter update norm; familiarity covariate (agent log-likelihood of each outcome text).
NOT adopted as written: "log it inside the E1/B0 runs". The harness (`harness/protocol/train.py`,
commit 47077e7) saves only aggregate step logs, eval-episode pickles and warm/final checkpoints, not the
RL episode stream, so E1/B0 runs cannot feed E6. Per the spec's drop rule, E6 is not run on the GPU
and gets no GPU budget. Instead `e6run.py` wraps the unmodified Trainer (stream logging +
checkpoints + proxy head) and E6 runs at reduced scale on CPU. This is a deviation from "inside E1".

Why c vs b_c is a correctness check: in arm P the forecast adapter receives no reward gradient
(harness test T2a) and its truth updates depend only on the episode stream, so b_c replaying c's stream
reproduces c's forecast adapter up to float nondeterminism; ΔSSI(c) ~ 0 is expected by construction.
The informative contrast is d vs b_d.

Arms (all Qwen/Qwen3-0.6B, float32 CPU, preset easy, LoRA r 8, harness defaults otherwise):
  c  : python3 e6run.py --mode train  --arm P  --seed S --sft-steps 6 --sft-batch 8 --rl-steps 24 --groups 2 --group-size 4 --eval-every 6 --threads 2 --out runs/c-sS
  d  : python3 e6run.py --mode train  --arm B0 --seed S (same flags)              --out runs/d-sS
  b_c: python3 e6run.py --mode replay --arm P  --seed S --rl-steps 24 --eval-every 6 --stream runs/c-sS/stream --warm <c warm.pt> --out runs/bc-sS
  b_d: python3 e6run.py --mode replay --arm B0 --seed S ... --stream runs/d-sS/stream --warm <d warm.pt> --out runs/bd-sS
  a_c, a_d: the same online head on frozen base features (adapters disabled), trained on c's / d's stream
         inside the c / d processes (`a_score` in e6eval.jsonl).
  Seeds S = 0, 1, 2 (the hawk allows 3). Proxy head: ridge (lambda 10) on 288 calibration outcomes from
  24 disjoint boards at the warm checkpoint, then one SGD step (lr 0.02) per RL step on the 8 committed
  outcomes' realised env_reward. Code: harness commit 47077e771a1045aa6307a6c0bd0ec8efd0c3c7de, unmodified.
Primary metric: reward-caused self-serving drift ΔSSI(X) = SSI(X) - SSI(b_X), with X interpolated at
  b_X's final proxy-adapter update norm (matched norm); secondary at the final step (matched data).
  Path: results.json: reward_caused["d-b_d"].matched_norm / .final; per-step rows in results.csv.
Other metrics: gold_acc (non-tied same-board pairs), tie_frac, tie_reorder (free drift), fam (log-lik
  contrast), ssi_resid_fam (SSI net of familiarity by OLS), norms per adapter.
Comparison rule and thresholds (3 seeds):
  - directed drift in B0: ΔSSI(d) > 0 in 3/3 seeds and mean/sd > 2. Supported half 1.
  - mask removes it: |mean ΔSSI(c)| < 0.01 reward units. Supported half 2.
  - gold: mean gold_acc(c) - gold_acc(b_c) >= -0.02.
  - Claim SUPPORTED if all three hold; REFUTED if ΔSSI(d) <= 0 in >= 2/3 seeds or |ΔSSI(d)| < 0.01
    (the objection wins: drift, if any, comes from the data stream: SSI(d) ~ SSI(b_d)); MIXED otherwise.
  - If SSI(d) > 0 but ssi_resid_fam(d) - ssi_resid_fam(b_d) ~ 0, report as familiarity, not preference.
Predictions if the claim is true: SSI(d) > SSI(b_d) ~ SSI(a_d) ~ 0; SSI(c) = SSI(b_c); gold_acc high
  everywhere at init (outcome text states hits and shots, so a linear proxy reads gold easily).
Kill / stop: any mask_violations > 0 in c or b_c, or base checksum change: stop and report. A process
  over 4 h wall on CPU stops and is reported with whatever checkpoints exist. No GPU launches.
Budget: CPU only, at most 6 concurrent processes of 2 threads (fleet cap 12 CPU jobs; load checked).
  GPU: $0.
Scope note (must appear in RESULTS): E6 says nothing about the expressivity of grounded variables;
  every preference ShipHunt rewards is already a grounded variable.
Analysis script, written before the runs: analyze.py (this directory).
Deviations log (append only, timestamped):
- 08:42Z (before any launch): timing probe on 0.6B at load 20-45 (`/tmp/lir-agent/e6/time06`, killed after
  1 RL step): SFT ~10 min/step under load 45, RL 170 s/step at 3 threads, bank eval 257 s. Scale set
  to sft 6, RL 24 steps, eval every 6 (5 bank evals per run), 6 processes x 2 threads; the command
  lines above were edited to these values before launch. E1 (`../E1/`) runs on GPU and does not save
  its episode stream (checked `../E1/` file list 08:42Z), so E6 cannot ride on it.
- 08:50Z: first launch (6 x pipeline.sh at 08:42Z) ran out of memory: c-s0, c-s2, d-s2 died during SFT
  (~9-10 GB RSS each, box at 55/62 GB used; logs moved to /tmp/lir-agent/e6/failed1/). c-s1, d-s0, d-s1
  continue unchanged (micro-batch 8). The three dead runs restart from scratch via sched.sh with
  --micro-batch 4 (gradient accumulation, same update; memory only), and every replay uses
  --micro-batch 4; at most 5 E6 processes and >= 12 GB available memory before each start.
- 09:03Z: box load 44-46 on 16 cores; SFT steps take 10-15 min. Runs are left at the registered scale.
  If a run hits its 4 h timeout, the analysis compares X and b_X at the last eval step both reached
  (analyze.py aligns steps; this alignment was added before any E6 result was read).
- 09:15Z: 0.6B CPU path ABORTED before any E6 result existed. The three micro-batch-8 runs (c-s1, d-s0,
  d-s1) were OOM-killed silently between 08:51 and 09:11 (empty tails in their logs); the micro-batch-4
  runs reached 1-2 of 6 SFT steps in 20 min with swap 14/15 GB full and load 42-46. Partial files moved
  to /tmp/lir-agent/e6/aborted-06b/. Projected wall time > 8 h. GPU is not used (hawk reshape: no
  separate GPU budget for E6). New scale, registered before launch:
  model `tiny` (the harness's random-init 2-layer Qwen3, hidden 64, used by its tests and smoke runs;
  the tokenizer is Qwen3-0.6B), preset easy, same bank.json (hash unchanged), sft 40 x 16, RL 200 steps
  x 32 episodes (4 groups x 8), eval every 25 steps, lr 1e-3 for every adapter (pilot: neither lr raised success above the ~0.45 random-plan level in 60 RL steps, so the larger lr was taken for more parameter movement; pilot (2e-4 vs 1e-3, arm P
  only, reward/success at 60 RL steps, /tmp/lir-agent/e6/tinylr*), threads 2, seeds 0-2, arms and
  analysis unchanged. Consequence for interpretation: the base model is random, so "representation" is
  only what LoRA learns in this run; the result is a mechanism test of the wiring at toy scale, not an
  estimate for pretrained LMs.
- 09:18Z: pilot read (arm P, 60 RL steps; /tmp/lir-agent/e6/tinylr{2e-4,1e-3}/c/steps.jsonl): RL success
  first-10 vs last-10 steps 0.456 -> 0.497 (2e-4) and 0.469 -> 0.466 (1e-3); a uniform 3-shot plan
  succeeds with p = 1 - (10/12)(9/11)(8/10) = 0.45. The tiny model barely learns the task, so the
  policy support is set by small logit differences, not competence. Launch: run_tiny.sh c|d 0-2.
- 09:49Z: first toy-scale round INVALID as an instrument, results not used (files kept in
  /tmp/lir-agent/e6/run1-head-diverged/: results.json, analyze.out, runs/). Two bugs in my wrapper, not
  in the harness: (1) the online proxy head (plain SGD on features standardised by init sd) diverged
  as the adapters moved (bank scores up to ~1e218 in c, b_c, d, b_d; SSI = nan/inf), and (2) replay
  started from X's warm adapters but with FRESH AdamW state, so b_c's forecast adapter did not reproduce
  c's (step-1 F_diff_forecast 0.0535 in c vs 0.128 in b_c, identical loss 0.32315). Fixes: head =
  normalised LMS (lr 0.5) with a median-sd floor on the standardisation; X saves its optimiser state at
  the warm point (warm_opt.pt) and the replay loads it. Smoke check (/tmp/lir-agent/e6/smoke2): c and b_c
  forecast-adapter step diffs and bank scores identical to all printed digits. Only arm a's head was
  stable in round 1 (gold acc 1.0, SSI(a_c) 0.0035 +- 0.003, SSI(a_d) 0.003 +- 0.0035, /tmp/.../results.json).
  Round 2 = same registered scale and seeds, all 12 runs from scratch.
