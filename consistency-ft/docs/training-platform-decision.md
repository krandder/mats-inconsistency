# Training platform and base model: decision assessment

Task: `hub-t8k9c.13`. Evidence checked September 13, 2026. **The platform/model
choice is not final and no training infrastructure has been purchased.** The
Fireworks account and missing matched-reasoning baselines remain unresolved.

**Collection status:** the separate [corrected v3 campaign](qwen-cdf-collection-progress-20260913.md)
is collecting the full 2,926-proposition catalog using the validated CDF sampler;
complete matching anchors and candidate training remain unfinished.
A reproduced [CPU sampler numerical defect](qwen-cpu-sampler-defect-20260913.md)
stopped the earlier v2 campaign after 22 attempts. Its 21 parse-valid responses and
one invalid response are retained; they do not provide a verified complete
categorical anchor and are not imported into v3. Corrected collection is separate
from the scientific platform decision.

**Local engineering update:** the pinned Qwen 3.5 9B text checkpoint now loads
and runs on General's existing CPU resources. Three original readouts completed,
and a separate retained-answer numerical derivative produced finite gradients
on every LoRA target with unchanged weights. Peak RSS was 17.470 GiB for
generation and 17.808 GiB for the derivative. See the
[measured execution report](qwen-candidate-execution-20260913.md).
The [full-anchor resource scenarios](qwen-anchor-resource-scenarios-20260913.md)
show why completing 2,926 beliefs needs durable collection and a realistic
compute plan: even the four-token-per-answer CPU scenario is 57–65 hours of
generation before integrity and reload overhead. These conditional calculations
do not prove throughput for the population or training/GPU fit.

The [v2 collector's first two requests](qwen-own-base-collection-20260913.md)
produced **2 parse-valid readouts out of 2,926 required propositions** under
the actual initial seeded adapters. Both completed with four tokens, including
EOS, and unchanged base/adapter weights. The bounded invocation took 389.590
seconds internally and 391.292 seconds under its external supervisor, with
19.36 GiB peak RSS. A preceding initialization failure generated zero tokens;
its metadata-serialization defect and original failure evidence are retained.
At that historical cutoff, the full repository suite passed 263 tests. These observations establish durable
collection on the existing CPU host; they do not establish population throughput,
complete anchors, candidate optimizer fit or a scientific platform decision.
Subsequent v2 invocations and its terminal failure are recorded separately in
the [stopped collection report](qwen-own-base-collection-progress-20260913.md).
The [corrected sampler validation](../data/research/qwen-cdf-validation-20260913.json)
later passed all 274 unique tests before v3 began.

The original project preference was **Qwen 3.5 9B on Fireworks**; gpt-oss-20b
entered as an available serverless smoke fallback. That history remains relevant
to the scientific choice. The provisional alternative engineering path is a
**bounded self-hosted LoRA smoke test with gpt-oss-20b**, once a GPU allowance is
approved and a concrete runtime is prepared. It has an existing complete baseline in this repository and an
official single-H100 SFT example. This makes it a useful integration control;
it does not establish it as the best scientific base model or prove RL will
fit that hardware. Compare it with **Qwen 3.5 9B** before committing the full
research pilot. Qwen is smaller, open, and already named in the project, but
this repository has no complete measured Qwen 3.5 9B baseline.

## Why the earlier platform assumptions do not hold

| Path | Current evidence | Implication for this experiment |
| --- | --- | --- |
| Fireworks managed RFT | Custom evaluators; guide describes rewards in [0, 1]. gpt-oss-20b is listed for managed LoRA RFT. | It may be viable after account recovery and a quote, but the raw negative A2 reward is not directly within the documented range. |
| Fireworks Training API | `forward_backward_custom()` exposes per-token log-probabilities for a Python-defined scalar loss; custom GRPO loops are documented. | The old “no custom loss” objection is obsolete. Validate the required objective against the actual exposed tensors and supported training surface. |
| Qwen 3.5 9B on Fireworks | Catalog lists dedicated API LoRA/full training, with a 3 × B200 RL shape; no managed-RFT or serverless-training entry. | Do not price it using the old token-rate table or assume a cheap shared trainer. |
| Self-hosted TRL/verl | Full control over rollouts, episode aggregation, optimizer, and frozen reference. OpenAI documents gpt-oss LoRA SFT on one H100 80GB. | Best engineering fallback while Fireworks access is unresolved; RL memory and throughput still require an actual smoke test. |

All availability statements are about the public catalog. They do not prove that
this project's account can create the corresponding resources. The existing key
still reports account suspension on September 13. The recovered browser route
exposes a different account with credits; see the [account-mismatch investigation](fireworks-backfill.md).
[Source snapshots and exact catalog entries](../data/research/platform-evidence-20260913.json)
include the catalog generation time and fetched-page hashes.

## Preserve the research objective when adapting a trainer

The proposal's A2 reward is

`reward = -Exploit(c, S) - lambda * mean_i KL(Bernoulli(c_i) || Bernoulli(base_i))`.

The frozen base beliefs must come from the **same exact trainable base revision,
chat template, prompt, and reasoning treatment** used by the experiment. The partial
GLM snapshot is an integration fixture, not a replacement for a Qwen or gpt-oss base
snapshot. Missing or refused probabilities cannot be filled with 0.5, copied
from another model, or silently dropped during reward evaluation.

KL to a base probability of 0 or 1 can be infinite. Decide and record whether
the experiment excludes unsupported outputs or uses an explicit smoothing
transformation. Do not hide numerical clipping inside the loss. The
[KL anchor implementation](kl-anchor.md) keeps this choice in the snapshot
identity and checks coverage at lookup time.

If a managed evaluator requires [0, 1], a **fixed affine transform with a proven
finite bound** preserves the raw reward ordering and expected objective. A
nonlinear mapping such as `exp(reward)`, arbitrary clipping, or a batch-dependent
normalization changes the objective and must be treated as a different arm.
Unbounded exact KL does not admit a finite global affine map into [0, 1]. A
custom trainer avoids that particular interface constraint but still requires
finite numerical training rewards and an explicit boundary convention.

The trainer must preserve one rollout identity for each episode's related
questions, aggregate the episode reward against its constraint set, and attach
that reward to the correct generated-token trajectories. It must not train on
the frozen held-out check types. Model-token KL used by an RL library is a
separate regularizer; it is not this Bernoulli belief-space anchor.

The [episode-to-token adapter](training-integration.md) now supplies this grouping,
reward assignment, causal masking and sampling/checkpoint binding as a tested
CPU integration contract. It has no model generation, autograd or optimizer
loop, and establishes neither hardware fit nor trainer compatibility by itself.

## Model selection evidence available now

The published gpt-oss-20b baseline has all 300 original groups; its recorded
reasoning setting is the legacy `none-or-low` fallback, with effective effort
not captured per proposition. It cannot support a clean none-versus-high claim.
Qwen 3.5 9B has no complete baseline in the current archive. Larger high-ranking
models are not automatically suitable small-adapter pilot choices.

The new [Luna reasoning analysis](reasoning-sensitivity.md) shows why mode must
be controlled: medium/high do not uniformly improve the original cohort,
xhigh/max improve its shared mean substantially, and individual check families
can regress while the aggregate improves. The original and corrected cohorts
also show different patterns. That is evidence for a matched sweep, not evidence
that Luna's best effort transfers to an open model or that greater compute is
always better.

Before fixing a scientific base, obtain the same cohort, explicit reasoning
mode, valid-output coverage, contamination evidence, consistency per check,
and token/latency cost for the small candidates. The Fireworks sweep bead stays
open for the missing provider measurements; the completed Luna analysis is a
separate contribution.

## Concrete requirements before a paid pilot

1. Resolve the Fireworks key/browser account mismatch or select an available GPU
   quote; do not guess a cap increase from another account's billing figures.
2. Pin a model revision, tokenizer, chat template, reasoning mode, and training
   implementation. Verify LoRA target modules for that architecture.
3. Freeze the base probabilities for **every training proposition** under the
   chosen treatment and validate the snapshot hash and coverage. The 300-group
   benchmark does not cover the entire training episode corpus.
4. Complete the missing historical-provider probes and adjudicate responses.
   The separate [nine-setting native campaign](native-probes-20260913.md) is
   complete. Source-reviewed successful recall is evidence of an observed
   capability; it does not by itself date training or earliest knowledge. Lack
   of an answer does not certify a safe upper cutoff or approve Brier windows.
5. Run offline evaluator cases, then an approved short hardware smoke test with
   a hard time/cost stop, checkpoint recovery, and complete usage records.
6. Use measured rollout/training throughput to price the proposal's pilot. Keep
   the existing Gate-1 and Gate-2 research decisions; no approval is implied here.

Current prices, including why the old free-RFT claim and token rates are unsafe,
are in [COSTS.md](../COSTS.md). This assessment completes the source/compatibility
investigation; the final base/platform commitment and live pilot remain pending.

## Primary references

- [Fireworks training models](https://docs.fireworks.ai/fine-tuning/models)
- [Managed RFT](https://docs.fireworks.ai/fine-tuning/reinforcement-fine-tuning-models)
- [Training API](https://docs.fireworks.ai/fine-tuning/training-api/introduction)
- [Serverless custom RL loop](https://docs.fireworks.ai/fine-tuning/training-api/serverless)
- [OpenAI gpt-oss repository](https://github.com/openai/gpt-oss)
- [OpenAI gpt-oss LoRA SFT tutorial](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers)
- [Qwen 3.5 9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)
