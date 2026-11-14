# Stacking a frozen "belief" LoRA (A) with a trainable "action" LoRA (B) on Tinker

Date: 2026-10-09. Read-only research: no sign-up, no API key, no paid calls.

What I checked:
- The Tinker Python SDK source, tinker 0.33.1 (latest on PyPI, downloaded today). The files are `tinker/lib/public_interfaces/{service,training,sampling,rest}_client.py` and `tinker/types/*`.
- The `thinking-machines-lab/tinker` repo (HEAD 2026-10-08) and the `thinking-machines-lab/tinker-cookbook` repo (HEAD today).
- These docs pages:
  - https://tinker-docs.thinkingmachines.ai/ (nav map)
  - /tinker/lora-primer/
  - /tinker/losses/custom/
  - /tinker/losses/importance-sampling/
  - /tinker/howto/checkpoints/
  - /tinker/models/models_and_pricing/
  - /tutorials/advanced/multi-agent/

The API reference pages cited below (https://tinker-docs.thinkingmachines.ai/tinker/api-reference/serviceclient/, .../trainingclient/, .../samplingclient/, .../restclient/, .../types/) are generated from the same SDK docstrings I read in the source.

Legend:
- **[DOC]**: stated on a docs page.
- **[SRC]**: confirmed in SDK or cookbook source.
- **[INF]**: my inference, not tested against the live service.

## Bottom line

**Tinker cannot do parametric stacking.** You cannot train only some LoRA parameters, freeze a block of an adapter, use a custom or merged base model, or upload weights. Each training client holds one LoRA on a stock base model, and one global optimizer updates all of it.

So "B on top of a frozen base+A" cannot be done exactly on Tinker. The workable options are:
1. **Functional freezing:** one adapter initialised from A, trained with RL, with A's beliefs anchored by a KL penalty to a frozen A sampler and/or a log-score term.
2. **Architectural freezing:** A runs as a separate frozen sampler whose beliefs feed into B's prompt.
3. **Exact stacking off Tinker:** export A and train B on base+A on our own GPUs.

## Answers to the five questions

### 1. Custom base model (A merged in)?
**No.**
- `ServiceClient.create_lora_training_client(base_model, rank=32, seed, train_mlp, train_attn, train_unembed, user_metadata, optimizer)` takes `base_model` as a catalogue name, e.g. "Qwen/Qwen3-8B" or "openai/gpt-oss-120b" [SRC][DOC models page: "The exact string to pass to `create_lora_training_client(base_model=...)`"].
- The `base_model` override in `create_training_client_from_state(path, base_model=None, ...)` and in `LoadWeightsRequest.base_model` "must be compatible with the checkpoint's base model (e.g. a different context length)". In practice that means switching between `openai/gpt-oss-120b` and `openai/gpt-oss-120b:peft:131072`. It cannot point at new weights [SRC].
- The SDK has no weight-upload or import endpoint at all [SRC: grep of the whole SDK].
- The models page offers only "let us know" for models that are not listed [DOC].
- Training is LoRA-only, so you cannot merge A into a full-weight run either:
  - LoRA primer: "Tinker supports LoRA fine-tuning ... rather than full fine-tuning" [DOC].
  - SDK comment: "Right now all training runs are LoRa runs." [SRC]

### 2. Train from a checkpoint and train only some LoRA parameters?
**Loading from a checkpoint: yes. Training only some parameters: no.**
- These all exist and work on any `tinker://.../weights/<name>` checkpoint [SRC][DOC checkpoints page]:
  - `create_training_client_from_state(path)` (weights only)
  - `create_training_client_from_state_with_optimizer(path)`
  - `TrainingClient.load_state(path)` and `load_state_with_optimizer(path)`
- The rank and the `train_mlp/attn/unembed` flags are copied from the checkpoint. A loaded client therefore has exactly A's rank [SRC: it reads `weights_info.lora_rank`].
- `LoraConfig` has only `rank, seed, train_unembed, train_mlp, train_attn` [SRC]. These flags choose which module types get a LoRA when the client is created. They do not freeze anything inside an adapter that already exists.
- The optimiser is chosen once, at creation: `AdamOptimizerConfig` (type "adamw") or `DimuonOptimizerConfig` [SRC].
- The parameters for each step are global scalars:
  - `AdamParams(learning_rate, beta1, beta2, eps, weight_decay, grad_clip_norm)`
  - `DimuonParams(learning_rate, grad_clip_norm, beta1)`
- There are no param groups, no per-parameter learning rates, no masks, and no custom optimiser [SRC: `types/optimizer.py`; `optim_step(optim_params)` docstring].

### 3. "One adapter, two rank blocks": freeze A's block?
**Not possible.** Three things block it:
- **No gradient access.** `forward_backward_custom(data, loss_fn)` hands your function only per-token logprobs. The server turns dLoss/dlogprobs into weight gradients through a linear surrogate cross-entropy [DOC custom-loss page: "surrogate_loss = (logprobs * logprob_grads).sum()"; SRC: it sends `weights = -grad` to a second `forward_backward`]. Your code never sees parameters or their gradients, so it cannot mask a block.
- **No block-level learning rate**, as covered in question 2.
- **No way to build the block-structured adapter.** You cannot upload one. A rank-(rA+rB) client created inside Tinker has no way to place A's exact weights into a sub-block [INF].
  - Whether `load_state` of a rank-rA checkpoint into a rank-(rA+rB) client pads or errors is untested [INF; I expect a shape error].
  - Even if it pads, Adam would update every block.
- The LoRA algebra offers no workaround: a nonzero frozen block still gets nonzero gradients [INF, standard].

### 4. Download, re-upload, external weights for sampling?
**Download: yes. Re-upload: no. Sampling external weights: no.**
- Download works [SRC/DOC]:
  - `TrainingClient.save_weights_external(name)` then `RestClient.get_external_weights_urls(tinker_path)` gives signed per-file URLs in HF/PEFT-style format. This is the recommended path.
  - The deprecated path is `get_checkpoint_archive_url_from_tinker_path` on a `sampler_weights` checkpoint. It gives a tar containing `adapter_model.safetensors` and `adapter_config.json`.
  - The CLI is `tinker checkpoint download`. The cookbook has `tinker_cookbook.weights.download(tinker_path=..., output_dir=...)`.
- Merging is documented and works for both target models [SRC: cookbook `tinker_cookbook/weights/README.md`; DOC /cookbook/deployment/export-hf/ and /cookbook/deployment/lora-adapter/]:
  - `build_hf_model` merges into a full HF model.
  - `build_lora_adapter` converts to PEFT.
  - Supported: "Qwen3 dense (8B)" and "GPT-OSS (20B, 120B)". GPT-OSS gets an interleaved expert layout and a `.attn` to `.self_attn` remap.
  - Caveat: serving GPT-OSS with vLLM LoRA is blocked by an mxfp4+LoRA incompatibility. Merging works.
- Re-upload is not possible. `create_sampling_client(model_path=..., base_model=...)` only accepts `tinker://` paths to weights produced in Tinker [SRC/DOC].
- `copy_weights(path)` copies existing Tinker weights between projects. The copy "cannot be trained on". It is not an import [SRC].
- Publishing (`publish_checkpoint_from_tinker_path`) only shares Tinker checkpoints with other Tinker accounts [DOC].

### 5. Creative routes
All of these are covered by the ranking below. The building blocks are confirmed in the cookbook:
- **KL to a LoRA reference.** `KLReferenceConfig(base_model, load_checkpoint_path)` plus `kl_penalty_coef` in `tinker_cookbook/rl/train.py` (L83, L466-470, L2000-2003). The reference is built with `service_client.create_sampling_client(base_model=..., model_path=<A checkpoint>)`, and `rl/metrics.py:incorporate_kl_penalty` adds the KL term per token into the advantages.
- **On-policy distillation from a checkpoint teacher.** `distillation/datasets.py:TeacherConfig(base_model, load_checkpoint_path)` and `distillation/train_on_policy.py` (reverse KL per token to the teacher via `compute_logprobs_async`).
- **A frozen second model inside an RL env.** `recipes/multiplayer_rl/twenty_questions/env.py:244` builds an answerer `SamplingClient` that stays fixed while the policy trains.
- **Belief readout as a probability distribution.** `recipes/decisions` uses `SamplingClient.sample(..., target_prompt_logprobs=...)` and trains it with `forward_backward_custom` plus a custom loss on the probabilities. `recipes/forecasting` is an RL recipe with a Brier reward, which is close to this use case.
- **Per-token advantages** for the `importance_sampling`/`ppo` losses: the shape is (N,) [DOC /tinker/losses/importance-sampling/]. Setting a token's advantage to zero is therefore an effective mask [INF from the formula].

## Ranked approaches

### 1. One adapter initialised from A, RL plus a belief anchor (approximate, functional freeze)
**Recommended first.** Confidence that it runs: high (all cookbook-native). Confidence that it preserves beliefs: medium, so measure it.

**Idea:** start the action policy C from A's weights. Train C with the RL reward. Penalise any drift of its beliefs away from frozen A in three ways:
- (a) a KL penalty to an A sampler;
- (b) optionally, on-policy KL restricted to the belief tokens, if the output has a belief section such as a stated probability;
- (c) optionally, a log-score/CE term on held-out belief data mixed into the same update.

```python
A_state = "tinker://<runA>/weights/final"; A_samp = "tinker://<runA>/sampler_weights/final"
# cookbook RL config
cfg = rl.train.Config(model_name="Qwen/Qwen3-8B", load_checkpoint_path=A_state,
        kl_penalty_coef=0.05, kl_reference_config=KLReferenceConfig(
            base_model="Qwen/Qwen3-8B", load_checkpoint_path=A_samp), ...)
# or by hand: tc = svc.create_training_client_from_state(A_state); refA = svc.create_sampling_client(model_path=A_samp)
# per token: adv = reward_adv*action_mask - beta*belief_mask*(logp_C - refA.compute_logprobs(seq))
# plus belief data via one combined loss:
# tc.forward_backward_custom(rl_datums + belief_datums, lambda d, lp: rl_surrogate(d, lp) + lam*logscore(d, lp))
```

**Limitations:**
- Nothing is frozen at the parameter level, so A's knowledge can drift on inputs the anchor does not cover. KL only constrains the on-policy distribution.
- C has A's rank, because rank is inherited from the checkpoint. If B needs more capacity, train A at the larger rank from the start.
- `forward_backward_custom` costs 1.5x the FLOPs and up to 3x the wall time [DOC].
- Each KL query to A costs prefill at sampling prices.
- Track belief drift every N steps: compare log-score on a held-out belief set, scored with `compute_logprobs` on C and on A.

### 2. Two models: A as a frozen belief module feeding B (exact freeze of A, different architecture)
Confidence: high. A cannot change because it only ever sits behind a `SamplingClient`.

**Idea:** in each RL episode, query frozen A for its belief. That can be a probability via `target_prompt_logprobs` as in the decisions recipe, or a short belief text. Put the result in B's observation. B is a separate LoRA trained with RL. B can start from base, or from A's weights to inherit A's internal features; the second option combines with approach 1.

```python
A = svc.create_sampling_client(model_path=A_samp)          # frozen
tcB = svc.create_lora_training_client("Qwen/Qwen3-8B", rank=32)  # or create_training_client_from_state(A_state)
# in Env.initial_observation(): p = sample_decision(A, render(q)); obs = render(q + f"\nBelief model: P(yes)={p:.2f}")
```

**Limitations:**
- B sees A only through a text or number bottleneck. It does not get A's hidden states, so this is not "base+A+B" in weight space.
- Each step costs an extra sampling call to A.
- The decisions-style probability readout fits forecasting well; richer beliefs need a text interface.
- The cookbook already does this: the twenty_questions answerer is a frozen model inside the env.

### 3. Exact stacking off Tinker (exact)
Confidence: high for Qwen3-8B, medium for GPT-OSS-120B (heavy).

**Idea:**
1. Train A on Tinker.
2. Export it with `save_weights_external` and `get_external_weights_urls`, or with `weights.download`.
3. Merge A into the base with `build_hf_model`, or keep A as a frozen PEFT adapter.
4. Train B with RL on our own GPUs, for example on the MATS cluster with TRL/PEFT. Either merge A and attach a new LoRA, or use PEFT multi-adapter with A's `requires_grad=False`.

**Limitations:**
- B's training leaves Tinker, so you lose its RL infrastructure and pricing and run your own sampler (vLLM) and trainer.
- GPT-OSS-120B needs multi-GPU, and vLLM LoRA serving for GPT-OSS is blocked on mxfp4. Use a merged bf16 checkpoint.
- B cannot come back to Tinker, because there is no upload.

### 4. Post-hoc decomposition of approach 1 into an "A + B" stack (exact algebra, not a frozen-during-training guarantee)
Confidence: high for the mechanics.

**Idea:**
1. Run approach 1.
2. Download A and C (`save_weights_external`).
3. For each module, compute delta = B_C·A_C·s − B_A·A_A·s, where s is the LoRA scale from `adapter_config.json`.
4. Truncate delta with SVD to rank k. The result is "B" relative to base+A.
5. Deploy base+A merged plus B as an adapter, or as a full merge.

**Limitations:**
- This gives you the stacked form for deployment and analysis, for example to inspect or ablate the action delta.
- A's function was protected during training only by approach 1's anchors.
- The SVD truncation is lossy unless k ≥ rank(delta) ≤ rA + rC.

### 5. Context or prompt distillation variants (approximate)
Confidence: medium.

- **(a)** Sample A's beliefs offline into a dataset (cheap, frozen). Then train B, by SFT or RL, on prompts that include them. This is approach 2 with cached beliefs; cookbook /cookbook/recipes/prompt-distillation/ and /cookbook/recipes/sdft/.
- **(b)** After RL, run on-policy distillation from A as teacher on belief-only prompts (`TeacherConfig(load_checkpoint_path=A_samp)`) to repair belief drift in C, alternating with RL phases.

**Limitation:** this is iterative repair, not a constraint.

### Not feasible on Tinker (confirmed)
- A custom or merged base model.
- Full fine-tuning.
- Weight upload or import.
- Sampling from external weights.
- Per-parameter learning rates, param groups, or custom optimisers.
- Gradient masking or access to parameter gradients.
- Freezing a block of a single adapter.
- More than one LoRA per training client.

Running several training clients at once is allowed, but each one is an independent single-LoRA model, so there is no way to compose them in one forward pass. That is my inference from the API surface: no client takes two adapters.

## Untested points worth one cheap check (once there is an API key)
1. Does `load_state(rank_rA_ckpt)` into a rank-(rA+rB) client error, or zero-pad? This does not affect the ranking.
2. Do several `forward_backward` calls before one `optim_step` accumulate gradients? The cookbook pairs every forward_backward with an optim_step (`rl/train.py` around L286-350), and I found no doc stating accumulation. Use a single `forward_backward_custom` with a combined loss to avoid depending on it.
3. How much belief drift does approach 1 actually show at kl_penalty_coef 0.01 to 0.1? Measure log-score on a held-out set, C against A.
