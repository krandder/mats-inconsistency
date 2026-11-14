# Synthetic CPU training integration

The repository now has an actual local generation → grouped reward → autograd →
optimizer path. The bounded run generated all five own-base anchor beliefs from
the saved initial synthetic model, then completed two SGD updates over four
joint samples / ten separate role trajectories per step. This verifies software
integration only. It does not establish candidate-model training readiness,
language understanding, or improved consistency.

The [run report](../data/research/synthetic-cpu-training-20260913.json) records
source hashes, environment, seeds, measured losses, gradients, parameter changes,
resource use and artifact hashes. The [independent loss oracle](../scripts/verify_cpu_loss.py)
checks all 210 position-logit derivatives against a separate standard-library
central-difference calculation; its [recorded result](../data/research/synthetic-cpu-loss-oracle-20260913.json)
has maximum absolute error `4.17e-11`.

## Run locally

The lock targets **CPython 3.12 / Linux x86_64**, with **PyTorch 2.6.0+cpu** and
16 hash-pinned wheels. Use an isolated environment; the repository's ordinary
non-tensor tests do not import torch. For a Python installation with venv
and ensurepip available:

```sh
python3.12 -m venv runs/cpu-dev-venv
runs/cpu-dev-venv/bin/python -m pip install --require-hashes --only-binary=:all: \
  -r training/requirements-cpu-py312-linux-x86_64.lock
runs/cpu-dev-venv/bin/python -m pip check
runs/cpu-dev-venv/bin/python -m unittest test_training_torch -v
runs/cpu-dev-venv/bin/python scripts/verify_cpu_loss.py
runs/cpu-dev-venv/bin/python scripts/run_synthetic_cpu_training.py \
  --output runs/synthetic-cpu-example
```

General's system Python lacks ensurepip. Its existing task-local environment was
bootstrapped using a separately SHA-256-verified pip 25.0.1 wheel; no global
packages were installed. The `.in` file records requested versions; the lock
enforces the inspected wheel bytes. No pretrained weights or candidate
tokenizers are downloaded by the runner or tests.

The CLI defaults to two updates, one CPU thread, model seed `20260913`, anchor
seed `101`, rollout seeds `202` and `203`, SGD learning rate `0.01`, and synthetic
anchor weight `2.0`. Those are fixture settings, not scientific hyperparameter
selections. The output directory must be new; prior and concurrently claimed
directories are never overwritten. The CLI permits one to five steps and one
or two CPU threads, and has no retry-until-valid loop. Every invocation's seeds
are declared before generation.

## What runs

`training/synthetic_cpu.py` constructs a **75,484-parameter** GRU plus trainable
bigram model, a reversible UTF-8 byte tokenizer with five merged response pieces,
and an explicit chat template. The synthetic model's finite initialization
biases favor `0.25`, `0.50`, and `0.80`; finite transitions make EOS likely after
a response piece. The other pieces represent a reasoning block and `ABSTAIN`
for negative tests. All 263 vocabulary logits remain finite and participate in
the unmodified softmax. These disclosed biases live in trainable model weights;
the sampler has no token mask, processor, forced EOS, or numeric-answer filter.

The runner saves initial weights, configuration, tokenizer, template, protocol,
and tensor manifest; reloads the local state with `weights_only=True`; and checks
actual tensor content against that manifest. It elicits the five training
propositions from these initial weights and freezes their **generated** readouts
through the existing anchor module. Anchors are neither hand-filled nor copied
from a provider benchmark. Training uses a coin And episode with three roles and
a coin Negation episode with two roles, each sampled twice. Distinct validation
and heldout coin fixtures remain excluded from training requests.

`training/torch_loop.py` samples every role from a fresh complete prefix using
full-vocabulary categorical softmax at temperature one. Eval mode is required
for both generation and loss evaluation; autograd remains enabled for the latter.
Sampled token IDs and log probabilities are retained. The tensor loss checks the
live tensor digest and sampler state against the behavior policy, then verifies
teacher-forced generated-token log probabilities against those recorded during
sampling. Model state is checked again after loss forward and backward.

The [existing reward adapter](training-integration.md) supplies the exact A2
joint reward and causal coefficients:

```text
R_joint = -Exploit - lambda * mean_role KL(Bern(candidate) || Bern(own_base))
L = -(1 / N_joint_samples) * sum_joint R_joint
                           * sum_roles sum_generated_tokens log pi(token | prefix)
```

The implementation multiplies differentiable causal log probabilities by
detached coefficients and sums once. It adds no role/token averaging, advantage
normalization, clipping, baseline, or token-policy KL. The last prompt-position
logits predict the first generated token. EOS receives credit even though the
tokenizer omits final EOS from the rendered text. Prompt and right-padding target
positions have zero credit; prompt embeddings can still receive gradients
through subsequent predictions.

## Failures and evidence

Strict probability parsing is unchanged. Reasoning text, refusal, `ABSTAIN`,
invalid numeric output, truncation, incomplete groups, or nonfinite rewards stop
the entire update. Unsupported reasoning channels are not extracted or silently
converted into probabilities. Streamed receipts retain completed and partial
attempts even when generation or decoding fails; RNG digests and failure records
remain on disk. Failed outputs are not resampled or dropped from the objective.

Optimizer updates require exactly the model's trainable parameters and finite
optimizer state. Nonfinite gradients, parameters, or optimizer moments reject
and restore both parameter and optimizer state. `OptimizerSession` rejects
replayed credit even with zero learning rate. Its replay ledger is in-process;
unique attempt directories provide the runner's durable duplicate protection.
Arbitrary crash-resume, distributed training, and session reconstruction are
unsupported. Artifact writes after a successful optimizer step are not an
atomic crash-recovery transaction.

The output contains initial artifacts and complete anchor provenance, plus each
step's plan, RNG digests, streamed rollout records, adapter credits, and summary.
`update-tensors.pt` stores full gradients, parameter deltas, resulting weights,
and optimizer state. `environment.json` pins package versions and relevant source
hashes, including the CLI that specifies run behavior. These remain in ignored
`runs/`; the committed report contains their hashes and synthetic measurements.

The 16 runtime tests cover hand-computed rewards and sequence loss, analytic
softmax gradients, padding invariance, genuine two-step updates, artifact reload
and drift, mode/distribution drift, group/split rejection, actually generated
reasoning and abstention, truncation, partial generation/decode/anchor failures,
optimizer rollback, poisoned Adam moments, replay, and output ownership races.
The complete repository suite passed all **221 tests** in the isolated CPU
environment. Without torch, the 16 optional runtime tests skip cleanly.
The independent retained-checkpoint audit re-evaluated all **50 sampled tokens**
from the five anchors and twenty training trajectories with exact log-probability
agreement, and checked both checkpoint transitions and every parameter delta.

The declared CLI run measured **1.098 seconds after imports**, **306,614,272
bytes peak process RSS**, and zero sampled-versus-evaluated log-probability error
on both updates. `wall_seconds` starts after imports and thread setup;
`cpu_seconds` and peak RSS cover the whole process, including imports. These tiny
fixture measurements are not estimates for a language model. No provider
inference or subscription requests occur in this runtime; agent development
subscription use is outside this CPU receipt.

## Remaining candidate work

Real Qwen or gpt-oss training still needs a selected immutable checkpoint and
tokenizer/template, verified candidate forward/backward support, complete own-base
anchors from that exact initial checkpoint over the real split, a defined output
channel contract, and an approved hardware/memory/batch plan. The synthetic model
has no scientific language-model capability. Tensor/config hashes do not attest
to arbitrary third-party Python forward implementations. No candidate weights,
GPU capacity, provider credentials, candidate training result, or scientific base
selection is established by this integration.
