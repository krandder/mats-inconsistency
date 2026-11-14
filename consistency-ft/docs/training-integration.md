# Offline episode reward adapter

`training/episode_adapter.py` implements the boundary between complete episode
rollouts and an on-policy custom training loop. It validates provenance, groups
separately prompted trajectories, computes episode rewards, and produces causal
token loss coefficients. It does not load weights, generate tokens, perform
autograd, or update a model. All integration tests use synthetic coin/die/geometry
fixtures and a character tokenizer; passing them is not evidence of GPU or
training readiness.

Run the CPU integration and anchor tests from the repository root:

```sh
python3 -m unittest test_training_integration test_anchors -v
```

This runnable walkthrough uses the same synthetic fixture and no model:

```python
from test_training_integration import EpisodeAdapterTests
from training import policy_gradient_loss

fixture = EpisodeAdapterTests()
fixture.setUp()
try:
    adapter = fixture.adapter()
    plan = fixture.plan(adapter)  # Two episodes, two samples each.
    trajectories = fixture.trajectories(plan)  # Synthetic completed outputs.
    credits = adapter.score(
        plan, trajectories, decode_completion=fixture.decode, pad_token_id=0,
    )
    logps = [[-1.0] * len(row) for row in credits.loss_coefficients]
    loss = policy_gradient_loss(
        logps, credits, trajectory_ids=credits.trajectory_ids,
        policy_sha256=credits.policy_sha256,
        runtime_sampling_distribution=fixture.sampling,
    )
    print(credits.joint_sample_count, len(credits.trajectory_ids), loss)
    # 4 joint samples; 10 separate role trajectories; a finite reference loss.
finally:
    fixture.doCleanups()
```

## Experiment and source bindings

Create `EpisodeRewardAdapter(binding, expected_binding_sha256=...,
split_paths=..., anchor_path=..., anchor_manifest_path=..., chat_template=...,
outcomes_path=...)` before planning any generation. The caller must pin the
expected binding hash in an external experiment record. Computing a new expected
hash from whatever files happen to be present provides no change detection.
The test fixture deliberately computes synthetic hashes locally.

The binding requires these exact fields; unknown or missing fields fail:

| Field | Contract |
| --- | --- |
| `schema_version` | Integer `1`. |
| `experiment_id` | Nonempty experiment identity. |
| `training_identity` | The immutable initial checkpoint, tokenizer, template, protocol, and effort identity below. |
| `sampling_distribution` | The explicit unfiltered categorical/eval-mode contract below; affects the experiment hash. |
| `protocol_id`, `protocol` | Named, hash-pinned elicitation protocol containing `system_instruction` and `verbal_prompt`; the latter inserts `{proposition}` exactly once. |
| `anchor_sha256` | The frozen anchor **payload** digest produced by the anchor module, not the file digest. |
| `split_sha256` | Exact byte hashes for `train`, `valid`, and `heldout`. |
| `arm`, `lambda_anchor` | Explicit arm and finite nonnegative weight; no scientific weight is selected by this adapter. |
| `output_parser` | `strict-decimal-v1`. |
| `split_policy` | `episode-ids-and-check-types-v1`. |
| `loss_reduction` | `mean-joint-sample-sum-role-tokens-v1`. |
| `nonfinite_reward_policy` | `reject-batch`. |
| `outcomes_sha256`, `training_cutoff` | Required for A1/Brier control; both must be `null` for A0/A2. |

`training_identity` has exactly these fields:

```text
model_id                    model_revision
weights_manifest_sha256     tokenizer_id
tokenizer_revision          tokenizer_files_sha256
chat_template_sha256        protocol_sha256
reasoning
```

Revisions must be immutable 40- or 64-character hexadecimal commit/content
identifiers. The four SHA-256 fields must be lowercase 64-character hexadecimal
digests. `reasoning` is explicit, including a value such as `none` when appropriate.
Template text and protocol content are checked against their hashes. A runtime
must derive the weight manifest and tokenizer-file hashes from its actual loaded
artifacts, and apply the specified template, protocol, effort, and generation
configuration. The adapter cannot inspect a Python encoder callback or attest to
loaded weights. Metadata claims alone do not prove checkpoint identity.

The only supported `sampling_distribution` is exactly this object:

```json
{
  "schema_version": 1,
  "sampler": "categorical",
  "temperature": 1.0,
  "top_p": 1.0,
  "top_k": 0,
  "logits_processors": [],
  "logits_warpers": [],
  "model_mode": "eval",
  "dropout_enabled": false
}
```

This means sampling from the full vocabulary's unmodified softmax distribution,
with temperature one, no top-p/top-k filtering, and eval mode with dropout
disabled during both generation and loss evaluation. Empty processor/warper lists
exclude **all** logit transformations, including library/model defaults such as
repetition penalties, min-p/typical filtering, forced/suppressed tokens, and
minimum-length EOS suppression. The runtime must inspect its effective generation
configuration rather than assume a `generate()` call has no defaults. Greedy
decoding, beams, constrained decoding, and custom processors require a different
adapter contract; they are rejected here. Eval mode does not disable gradients:
the loss forward pass must still preserve autograd.

The sampling digest travels through requests, plans, generation receipts, and
credit batches. `prepare` and the CPU loss require an independently reported
`runtime_sampling_distribution`; generation receipts must report the digest of
the effective settings actually used. Changes after planning therefore fail when
reported. These checks verify agreement of claims; neither the digest nor the
adapter inspects model state or proves a runtime obeyed those settings.

Every arm requires a complete anchor for the entire bound training split. In
addition to the checks in [the anchor module](kl-anchor.md), the adapter requires
the identical `training_identity` in the original hashed source run, and its
canonical SHA-256 as `training_identity_sha256` in every original anchored
receipt. The manifest, run metadata, receipt file, and individual receipts are
verified. Adding a sidecar assertion to a native-provider run cannot upgrade it
into verified trainable-checkpoint evidence. The existing partial GLM film
snapshot must fail: it is incomplete and lacks this checkpoint evidence.

For A1/control, the outcome file is also pinned by its byte hash. Schema version
1 contains an `outcomes` mapping keyed by full-proposition SHA-256, covering
exactly the training proposition union. Each entry needs binary `value`, matching
`text_sha256`, timezone-aware `resolved_at` no later than `training_cutoff`,
`evidence_sha256`, and a nonempty `source_reference`. Missing, unresolved, future,
or mismatched labels fail. Hashes and references bind declared evidence; they do
not independently establish that a real-world outcome was correctly resolved.

## Planning, grouping, and failure behavior

Call `prepare(episode_ids, samples_per_episode=..., batch_id=...,
policy_sha256=..., runtime_training_identity=..., runtime_sampling_distribution=...,
encode_messages=...)`. Episode
IDs must be unique, sorted training IDs. Validation/heldout/unknown IDs fail.
`batch_id` must be fresh. `policy_sha256` identifies the actual current rollout
weights; this changes after optimizer updates while the initial-checkpoint
`training_identity` and its own-base anchors remain fixed.

Requests are ordered by episode, sample index, then the episode's canonical role
order. Each request contains exactly one system message and one proposition user
message. Each episode/sample group is one joint sample, with a separate
trajectory for every required role. For example, two samples of a three-role And
episode and a two-role Negation episode produce four joint samples and ten
trajectory rows. Encode each request with the bound template; generate roles in
fresh conversations or independent attention rows, without other roles' answers
or shared conversational history.

Return `GeneratedTrajectory` rows in the exact planned order. Scoring validates
request/plan/experiment/policy/sampling/message hashes, integer prompt and completion token
IDs, fresh unique context IDs, one turn, no tools, and a completed `stop`/`eos`
finish. It checks the decoded generated tokens equal the recorded completion
text. Unique context IDs are auditable runtime claims, not proof that a provider
actually isolated contexts. The runtime must enforce and record that isolation.

The parser accepts only one ordinary decimal in `[0,1]`, allowing surrounding
whitespace. It does not accept percentages, scientific notation, commentary,
refusals, `ABSTAIN`, or truncated generations. `decode_completion` must decode the
entire assistant continuation, apart from tokenizer control tokens such as EOS;
it must not silently extract a favorable numeric answer. This interface does not
yet support a separate reasoning channel or a token span for a final probability
inside a longer completion. Such models need an explicitly specified and tested
extension before use.

Any malformed, incomplete, shuffled, duplicate, mismatched, invalid, or nonfinite
batch raises before returning any credit. A plan is consumed only after the
entire batch succeeds, and cannot then be replayed. A failed call may be retried
to repair transport/record assembly of the **same generated sample**; resampling
until valid or dropping failed samples would condition the rollout distribution
and change the objective. A real trainer must stop that step and retain all
attempts/failures for diagnosis, unless a different invalid-output treatment has
been explicitly designed and bound to a new experiment.

The split contract checks disjoint episode IDs and the builder's train/heldout
check-type sets. It also exposes `split_proposition_overlap`, the counts of exact
proposition-text overlap for each split pair. Reuse of a proposition across
different episodes is allowed by this existing split policy; these checks do
**not** establish proposition-level disjointness or absence of pretraining/data
contamination. See [the contamination audit](contamination-audit.md).

## Rewards and causal token credit

For a complete joint sample with role probabilities `p_i`, base beliefs `b_i`,
binary outcomes `y_i` where available, and `m` roles:

| Arm | Joint reward |
| --- | --- |
| `a0` | `-Exploit(p, episode)`; requires `lambda_anchor = 0`. |
| `a1` | `-Exploit(p, episode) - lambda_anchor * mean_i (p_i - y_i)^2`. |
| `a2` | `-Exploit(p, episode) - lambda_anchor * mean_i KL(Bern(p_i) || Bern(b_i))`. |
| `control-brier` | `-mean_i (p_i - y_i)^2`; requires `lambda_anchor = 0`; exploitability is diagnostic only. |

A2 calls `anchors.a2_reward` directly, preserving the snapshot's pinned exact or
explicit symmetric-mixture treatment and its mean-over-roles normalization.
There is no added epsilon, clipping, proxy target, reward normalization, or
token-policy KL. Exact infinite KL remains valid in anchor diagnostics, but a
nonfinite resulting reward cannot cross this training boundary. A2 with zero
weight preserves the anchor module's convention: the KL diagnostic may be
infinite while its penalty is zero and the reward is finite.

Every role trajectory receives its joint sample's complete reward. Let `N` be
the number of joint samples. The reference REINFORCE loss is:

```text
L = -(1/N) * sum_joint R_joint
                  * sum_roles sum_generated_tokens log pi(token | prefix)
```

The joint probability factorizes across the separately prompted roles. Its log
probability is therefore the sum of their sequence log probabilities, which is
why the joint reward credits every role. Dividing by role count, completion
length, or flattened trajectory count would change this objective. There is no
baseline, batch centering/scaling, PPO clipping, or GRPO advantage calculation.

For `T` trajectory rows and padded input width `W`, `TokenCreditBatch` contains:

| Field | Shape and meaning |
| --- | --- |
| `input_ids`, `attention_mask` | `[T, W]`; prompt plus completion, then right padding. |
| `completion_loss_mask` | `[T, W-1]`; generated **target** positions only. |
| `loss_coefficients` | `[T, W-1]`; `-R_joint/N` on generated targets, zero on prompts/padding. |
| `row_rewards`, `joint_sample_ids`, `trajectory_ids` | `[T]`; exact planned order and group membership. |
| `episode_metrics` | One reward/component record per joint sample. |

Coefficient `[i,j]` multiplies
`log pi(input_ids[i,j+1] | input_ids[i,:j+1])`. The first generated token uses the
last prompt-position logits. Retain the full completion, including generated
termination/control tokens where the tokenizer uses them; rendering away EOS
does not remove its policy-gradient credit.

`categorical_causal_logprobs` is a CPU equivalent of unmodified
`logits.log_softmax(-1).gather(targets)`, with causal logits shaped `[T, W-1, V]`
and full input IDs shaped `[T, W]`. It validates the sampling contract and computes
the log probabilities of the full-vocabulary categorical distribution. An
independent fixture samples a categorical CDF with integer weights `[1, 2, 5]`
and checks that logits `[log(1), log(2), log(5)]` produce gathered log probabilities
`log(1/8)`, `log(2/8)`, and `log(5/8)`, including under a large common logit shift.

`policy_gradient_loss` is a CPU numeric reference. It requires the same row order,
behavior-policy digest and runtime sampling distribution, correct shape, and
finite nonpositive causal log probabilities. A real tensor implementation must
preserve autograd, make the adapter coefficients constants, and **sum once**.
For this exact contract, generation and loss use the same softmax, conceptually:

```python
# Illustration only: requires a real tokenizer, model, tensor library, and loop.
model.eval()  # Keep dropout off for generation AND differentiable loss evaluation.
# Each generation step samples from unmodified next-token logits:
# next_token = torch.multinomial(next_logits.softmax(-1), num_samples=1)
# No temperature scaling, filtering, penalties, or other logits processing.
logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
causal_logps = logits.log_softmax(-1).gather(
    -1, input_ids[:, 1:, None],
).squeeze(-1)
loss = (causal_logps * detached_loss_coefficients).sum()
loss.backward()
optimizer.step()
```

Evaluate those log probabilities under the same weights and the bound unfiltered
categorical distribution that generated the trajectories before changing the
weights. The runtime must verify its actual sampler and model mode at generation
and loss evaluation; the adapter cannot enforce this through digests alone. The
CPU tests verify the numerical softmax/gather identity, not a live tensor backend
or generator. Distributed/accumulated batches must
preserve the global joint-sample denominator; averaging already-normalized local
losses without accounting for unequal sample counts changes the objective.

## Why this is a custom-loop interface

Official TRL `GRPOTrainer` source was checked at commit
[`cd2c52876d99b00baa5660328310e940b8360c6c`](https://github.com/huggingface/trl/blob/cd2c52876d99b00baa5660328310e940b8360c6c/trl/trainer/grpo_trainer.py)
(commit timestamp `2026-09-13T02:39:43Z`; raw file SHA-256
`c8d7ad8346c4c1c994c33a3c60c94e6cab8184ccdb3ee2cd433cada4770ebe97`).
Lines 177–202 describe per-completion custom reward functions. Lines 258–265 and
2231–2248 describe the experimental `rollout_func` contract, including
`prompt_ids`, `completion_ids`, and `logprobs`. Lines 2780–2813 reshape rewards by
`num_generations` and compute repeated-prompt group advantages.

That ordinary grouping unit is repeated generations of an individual prompt.
Our grouping unit includes several different role prompts. Repeating an episode
reward across flattened role rows does not make the standard GRPO grouping and
normalization equivalent to the objective above. This module therefore makes no
stock TRL callback compatibility claim. The illustrative tensor loss is a custom
on-policy integration point, not a tested TRL trainer.

## Remaining work before real training

A real smoke needs a selected immutable checkpoint, genuine tokenizer/template
and runtime identity derivation, complete beliefs elicited from that same initial
checkpoint over the bound training split, and a real generator plus tensor,
autograd, optimizer, and rollout-record integration. It also needs a concrete
resource quote and authorization for the inference/training resources. No GPU
capacity has been established here, and this offline work selects no scientific
base or anchor weight. The tests establish grouping, formulas, causal alignment,
and rejection behavior on synthetic data; they establish neither optimization
quality nor the scientific validity of a chosen model or dataset.
