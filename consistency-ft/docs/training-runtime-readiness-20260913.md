# General training-runtime readiness — 13 September 2026

**The offline adapter is ready for CPU development; General is not ready to run
either candidate's training experiment.** This inspection selects no scientific
base. Qwen3.5-9B remains the original project preference; gpt-oss-20b remains a
provisional fallback, as recorded in [the platform decision](training-platform-decision.md).

Evidence was captured at approximately 08:38–08:40 UTC against repository revision
`6989d64146198c7684c5f1f0ef7c71554f0371aa`. The
[sanitized inventory](../data/research/training-runtime-readiness-20260913.json)
contains exact counts, package results, candidate revisions, and source hashes.
Raw command/config evidence stays in ignored
`runs/training-runtime-readiness-20260913/`.

| State | Observed evidence and consequence |
| --- | --- |
| Ready | System Python 3.12.3 runs all **39 adapter/anchor tests**, including grouped rewards, causal token credit, and sampling-binding checks. This exercises synthetic data only. |
| Ready for CPU development | KVM guest reports **16 logical CPUs**, 8 cores × 2 threads, Intel Xeon Platinum 8581C; process affinity includes all 16. RAM is **62.8 GiB total / 44.0 GiB available** at capture, with no swap. No finite CPU quota or `memory.max` was found along the process cgroup ancestry. These are observations, not reserved capacity. |
| Missing in inspected GPU interfaces | No accelerator in `lspci` or relevant sysfs PCI classes/vendor checks; no NVIDIA, DRM render, AMD KFD, or accelerator device nodes; no GPU driver modules, NVIDIA version files, CUDA directories, or CUDA/ROCm runtime libraries in the linker cache. `nvidia-smi`, `nvcc`, and `rocm-smi` are absent. The generic `/dev/vfio/vfio` control node exists but does not establish a GPU. |
| Missing in active interpreter | `torch`, `transformers`, `peft`, `trl`, and `verl` are all absent by `importlib.metadata`; so are the 12 surveyed supporting ML packages, including NumPy, Accelerate, Triton, and kernels. No standard project/user virtual environment was found. Other private environments were not exhaustively searched. |
| Missing in accessible candidate-cache locations | No matching Qwen3.5 or gpt-oss-20b weights were found in nine scoped user/shared locations. One alternate-user cache was permission-denied and left untouched. This is not a machine-wide proof that no weights exist. |

## Candidate-specific gaps

These are inspected public revisions, **not selected experiment checkpoints**.
Weight-index totals describe stored tensors, excluding training activations,
gradients, optimizer state, rollout/KV buffers, and loading overhead.

| Candidate | Public metadata | Implication for this adapter |
| --- | --- | --- |
| Qwen/Qwen3.5-9B, `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | `Qwen3_5ForConditionalGeneration`, hybrid linear/full attention, BF16 text config; four shards with **17.98 GiB** of indexed tensors. The card requests current Transformers support; the config's `4.57.0.dev0` label is not a verified minimum version. | Its template opens a thinking block by default. `enable_thinking=false` supplies an empty thinking block in the prompt, offering a possible strict-decimal path that still needs real tokenizer/output verification. The pinned `generation_config.json` returns 404; library defaults must be inspected and overridden explicitly. |
| openai/gpt-oss-20b, `6cee5e81ee83917806bbde320786a8fb61efebee` | `GptOssForCausalLM`, MXFP4 expert weights; three shards with **12.82 GiB** of indexed tensors. The card's “within 16GB” statement concerns running the quantized model, not this training objective. | The template uses Harmony analysis/final channels and explicit reasoning effort. The current whole-completion decimal parser does not implement those channels. MXFP4/kernel/backward and LoRA compatibility remain untested; active expert count does not replace total-weight memory accounting. |

Sources: pinned Qwen [config](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json)
and [card](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md);
pinned gpt-oss [config](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/config.json),
[template](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/chat_template.jinja),
and [card](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/README.md).
The JSON inventory also pins weight-index and tokenizer-config evidence.

## Missing integration and next action

The [existing adapter](training-integration.md) supplies rewards and loss
coefficients, not a real tokenizer, generator, differentiable trainer, or optimizer.
It requires unfiltered categorical sampling at temperature one, eval mode with
dropout disabled, complete episode groups, and complete matching checkpoint-bound
own-base anchors. Neither candidate's complete training anchor is established
here: the existing 300-group API benchmark and partial GLM fixture do not provide
that evidence. Actual loading, backward compatibility, memory, throughput, and
runtime adherence to declared identities/settings remain unverified.

**Next engineering action:** connect actual tokenizer/model generation to the
adapter's rewards, differentiable causal-token loss, and optimizer step in an
isolated CPU development environment. A later implementation task must pin and
install its missing dependencies inside that environment, without global
installs. Reuse the adapter's existing snapshot/protocol/sampler checks and verify
the full generation → grouped reward → backward → parameter-update path with
explicitly synthetic tiny weights, a local test tokenizer, and coin fixtures.
Check that generated-token credit and the evaluated sampling distribution agree;
unsupported reasoning-channel output must still fail closed.

These are software integration tests, with no scientific base commitment or
candidate-weight download. An actual Qwen or gpt-oss training experiment remains
separate: it needs a selected immutable base and tokenizer, complete matching
own-base anchors, a concrete batch/sequence and memory plan, and suitable hardware
with an approved allowance. Neither the isolated environment nor this
implementation was started during the present audit.

No credentials, unrelated private files, model weights, or tokenizer vocabularies
were read/downloaded; fetched configs/cards/templates were inspected as data only.
No installs, model execution, paid inference, allocations, training, or machine
configuration changes occurred. The physical host and other machines were outside
this audit.
