# Turtel et al. 2025 replication (stage 1): runbook

Staged 2026-09-22 for one-command launch on a rented 4xH100. Nothing has been rented; no Vast API call has been
made from this lane. Paper: arXiv 2505.17989 v4; recipe and stage table in
`docs/self-trade-rl-proposal-20260921.md` (sec. 2a, sec. 9); stage-D data in
`docs/market-game-results-20260922.md`.

## What exists

| piece | path | status |
|---|---|---|
| prompt template + parser | `training/turtel_prompt.py` | verified: all 1,265 released test prompts re-render byte-exact from their parsed components |
| prompt builder | `scripts/build_turtel_prompts.py` | run: train 9,500 / validation 500 (chronological tail by prediction_date); `--exclude-flagged leak-flags.jsonl` ready (contract in the docstring), tested on a synthetic flags file |
| built prompts | `data/turtel-prompts-20260922/{train,validation}.jsonl`, `manifest.json` (sha256 of inputs and outputs) | train.jsonl (65 MB) is not committed; rebuild with the command below (7 s) |
| reward | `training/turtel_reward.py`, tests `test_turtel_reward.py` (19 tests) | strict Brier, guardrails, no-std advantages, soft Brier, ECE |
| trainer | `training/turtel_grpo.py` | TRL 1.13 GRPOTrainer, LoRA, vLLM server rollouts, resumable, validation every N steps, final test eval |
| pins | `training/requirements-grpo.in` | unverified on GPU (see below) |
| ops | `ops/vast/turtel-14b/` | launch.sh, manifest.template.json, api/rent/setup/start/watchdog/sync_loop/finish/progress, bootstrap.sh, run.sh |
| CPU dry run | `runs/turtel-dry-092940/dry-run-checks.json` | all six checks OK (2026-09-22 09:56) |

## Commands

```bash
# prompts (stdlib python; --verify needs pyarrow -> venv-cft). Without leak flags:
/home/agent/workspaces/venv-cft/bin/python scripts/build_turtel_prompts.py \
    --out-dir data/turtel-prompts-20260922 --verify
# with the leak screen once /home/agent/data/polymarket-evidence-exa-20260922/leak-flags.jsonl exists:
/home/agent/workspaces/venv-cft/bin/python scripts/build_turtel_prompts.py \
    --out-dir data/turtel-prompts-20260922 --verify \
    --exclude-flagged /home/agent/data/polymarket-evidence-exa-20260922/leak-flags.jsonl

# unit tests
python3 -m unittest test_turtel_reward

# CPU dry run (Qwen3-0.6B, 8 prompts, 2 steps, ~27 min on 16 cores)
/home/agent/workspaces/venv-turtel-grpo/bin/python training/turtel_grpo.py --dry-run --run-dir runs/turtel-dry-$(date +%H%M%S)

# preflight + offer search only (no spend)
ops/vast/turtel-14b/launch.sh

# THE LAUNCH (rents, bootstraps, runs, syncs, verifies, destroys; unattended)
ops/vast/turtel-14b/launch.sh --go
# second lease if the pass did not finish inside max_hours: resumes from the synced checkpoint
ops/vast/turtel-14b/launch.sh --go --resume
```

Lane directory: `/home/agent/data/vast-turtel-14b/` (manifest.json, instance.json, lease.json, launch.log,
progress.log, results/). Follow with `tail -f /home/agent/data/vast-turtel-14b/progress.log`.

## Lifecycle (same pattern as data/vast-wave1, which rented and destroyed a 4xH100 on 20 Sep)

1. `rent.py --go`: POST /bundles/ with the manifest spec (4x H100 SXM, >=200 GB disk, CUDA >=12.8, inet >=500,
   reliability >=0.98, <=$12/h), refuses if cheapest x max_hours > budget or > credit, PUT /asks/{id}/ with
   image `vllm/vllm-openai:v0.28.0`, runtype ssh. Writes instance.json (mode 600).
2. `setup.py`: waits for ssh, uploads code.tar.gz, data.tar.gz, bootstrap.sh, run.sh, manifest.json; runs
   bootstrap.sh (pins, 14B download with the resolved commit sha recorded, 4x H100 assert plus a bf16 matmul on
   every GPU, unit tests, `BOOTSTRAP_OK`).
3. `start.py`: `nohup bash run.sh` on the box: vLLM server on GPU 3 (`trl vllm-serve`, max-model-len 12288),
   waits for /health, then `accelerate launch --num_processes 3` on GPUs 0-2.
4. `watchdog.py`: stops the instance at max_hours (16), at budget ($600 at the quoted $/h), or after 30 min
   without a lease refresh. `sync_loop.sh`: rsync runs/ every 5 min, refreshes the lease only on success.
5. `finish.py`: waits for `RUN_DONE` / `RUN_FAILED` / `TRAINER EXIT=`, final rsync, sha256 of every non-log file
   compared remote vs local, destroys only on zero mismatches, confirms the instance is gone, records credit.

Irreversible actions: rent (spend starts), destroy (box gone; happens only after verified sync). A watchdog stop
leaves a stopped instance that still bills storage until finish.py or a human destroys it.

## Recipe as implemented (training/turtel_grpo.py defaults)

G = 4, `scale_rewards="none"` (advantage = reward - group mean), `loss_type="grpo"`, beta 0.005 (reference =
initial policy, adapter disabled), epsilon 0.2, num_iterations 1 (on-policy), AdamW (0.9, 0.999, 1e-8, wd 0),
LR 1e-6 constant, grad clip 1.0, entropy 0.001, bf16, `shuffle_dataset=False` single pass in prediction_date
order (chronology asserted at load and checked in the dry run), 4,096 completion tokens, 16,000-char scorer
truncation, temperature 1.0 rollouts. 6 prompts x 4 = 24 completions per optimizer step, per-device 2 (the paper's
"two sequences per GPU"), grad accumulation 4 over 3 GPUs. LoRA r 64 / alpha 128 on all linear projections.
Validation: chronological head of the 500 held-out questions (200 per pass) every 100 steps, temperature 0.6,
1 sample. Test: 1,265 rows, 1 sample (`--test-samples 5` = median-of-5 for the paper's inference trick),
`test/metrics.json` carries soft Brier (0.25 for unparseable), strict Brier, ECE, extremization rate and the
Polymarket / o1 / ReMax comparators recomputed on the same rows from the released columns.

Reward: -(p - y)^2, -1.0 when unparseable (asterisk form first, then "Probability:", then a decimal or percent on
the final line), minus 0.5 x non-English share, 0.5 x gibberish share, 0.25 if fewer than 30 reasoning words.

## Throughput and cost (proposal sec. 9 arithmetic; nothing measured yet)

Assumptions: prompt ~1.6k tokens (training prompts mean 6.4k chars after capping summaries at 600 chars, max 9.7k),
1.5k generated, G = 4. Per step: 24 rollouts x 1.5k = 36k tokens on one H100 with vLLM at 1.5-2.5k tok/s ->
15-25 s; training forward/backward for 24 x ~3.1k tokens through 14B with checkpointing on 3 GPUs plus the
reference pass -> 15-30 s; LoRA merge + 28 GB NCCL push -> a few s. 35-70 s/step x 1,584 steps = 15-31 h.
The proposal's stage-1 figure (4xH100 ~40 h, $400-600) covers two arms; this manifest caps one lease at 16 h and
$600, at the 20 Sep quote of $8.86/h that is 16 h = $142; a second `--resume` lease finishes a slow pass.
Validation passes add ~2 min each (16 total); the test pass ~13 min at 1 sample. Paper: 8xH100 ~3 days for 110k
questions, i.e. ~6.5 h for 10k, consistent with the 15-31 h range on half the GPUs with generation off-loaded.

## Unverified (must be watched on the first lease)

- Library pins never run on a GPU: trl 1.13.0 + vllm 0.28.0 (torch 2.13.0) + transformers 5.16.1 + peft 0.20.0 +
  accelerate 1.15.0 + datasets 5.0.1. Field names checked against the trl v1.13.0 source; the CPU dry run used the
  same trl/transformers/peft/accelerate/datasets on torch 2.6.0+cpu.
- `vllm/vllm-openai:v0.28.0` under Vast's ssh runtype (entrypoint override) and its CUDA build vs the host driver;
  bootstrap falls back to `uv pip install vllm==0.28.0` and aborts before training if any GPU fails a matmul.
- Memory: 14B bf16 + LoRA + 2 sequences of up to ~6k tokens per GPU with gradient checkpointing (expected < 60 GB,
  not measured); TRL server-mode weight transfer of merged LoRA weights every step.
- HF Trainer resume with TRL's RepeatSampler (`--resume`) is exercised only as save/load in the dry run.
- Guardrail lambdas (0.5 / 0.5 / 0.25) and the explanation-quality bonus (not implemented: needs an LLM scorer) are
  ours; the paper gives no numbers. ReMax (learned value baseline) is not implemented.
- `vllm.LLM` + `LoRARequest` path of `--eval-only --vllm-inprocess` (post-hoc evaluation of a saved adapter).
- Dry run cannot check GPU-only branches: `use_vllm`, bf16, DDP across 3 processes.

## Dry run record (runs/turtel-dry-092940, Qwen3-0.6B on CPU, 16 threads, 27 min)

reward_parsing 32 completions scored (10 parsed at 48 tokens); advantage_no_std max |error| 4e-8 over 32
completions across 2 generation calls; finite_loss [-3.3e-4, -4.1e-4], kl [0, 4e-8]; optimizer_step 112/112 LoRA
tensors moved, global_step 2; checkpoint_save_load checkpoint-2 reloaded on a fresh base with 0 mismatches;
chronological_order 8/8 ids in file order; validation at step 2 (4 rows) and test eval on 20 rows wrote
predictions.jsonl + metrics.json with the comparators (Polymarket 0.172, o1 0.214, ReMax 0.180 on those rows).
