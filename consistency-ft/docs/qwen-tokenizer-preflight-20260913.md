# Qwen tokenizer and original-corpus preflight — 13 September 2026

The actual tokenizer and chat template for `Qwen/Qwen3.5-9B` at inspected revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a` load locally and roundtrip every measured
prompt exactly. This is candidate preparation, **not a scientific checkpoint
selection or evidence that the model will return valid probabilities**. No model
weights were downloaded or loaded; no model answers or anchors were generated.

The [artifact manifest](../data/research/qwen-tokenizer-artifacts-20260913.json)
pins all downloaded public data by URL, byte count and SHA-256. The
[measurement report](../data/research/qwen-tokenizer-preflight-20260913.json)
records complete split hashes, tokenizer/runtime/source hashes, framing, token
distributions, group sizes, decoder checks and local resource measurements.

## Two distinct prompt treatments

The primary measurement uses exactly `PROTOCOLS['numeric-v1']` from
`harness/run_native_checks.py`: its original numeric system instruction and
`common.VERBAL_PROMPT` as the user message. This is a **proposed candidate
system/user treatment**, not a final scientific decision. Every role receives
one fresh system/user pair, with no other roles' answers or prior conversations.

The separate comparison uses only `common.VERBAL_PROMPT` in one user message,
matching the original Fireworks client's message framing. Both measurements
render the inspected **local Qwen template**, so the comparison does not verify
the hosted Fireworks service's template or reasoning behavior. Tokenization does
not establish sampling equivalence: the original API client uses temperature
zero, whereas the existing custom training loop requires unfiltered categorical
sampling at temperature one.

Both measurements explicitly set `enable_thinking=False` and
`add_generation_prompt=True`. The official template then ends the prompt with:

```text
<|im_start|>assistant
<think>

</think>

```

The exact suffix is `"<|im_start|>assistant\n<think>\n\n</think>\n\n"`.
The empty reasoning block belongs to the **prompt**; generated reasoning text
would remain in the continuation and fail the existing strict probability parser.
This template setting does not prove that pretrained weights will omit reasoning.
The official template's own whitespace handling is preserved. Each receipt keeps
the exact input messages and rendered prompt separately, with separate hashes.

## Complete original corpus

All original rows in `data/episodes/{train,valid,heldout}.jsonl` were measured;
none were substituted or sampled. The corpus contains **3,171 episodes, 9,342
role occurrences, and 5,549 distinct full propositions**. There are **11,098
distinct prompt records** across the two framings. Every record passed exact
rendered-text → token IDs → rendered-text roundtrip, without truncation or padding.

| Split | Episodes | Role occurrences | Proposed system/user p50 / p95 / max | Original user-only p50 / p95 / max |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,619 | 4,855 | 211 / 305 / 406 | 165 / 259 / 360 |
| Valid | 181 | 545 | 208 / 309 / 350 | 162 / 263 / 304 |
| Heldout | 1,371 | 3,942 | 204 / 350 / 494 | 158 / 304 / 448 |

Percentiles use the nearest-rank definition over all role occurrences. The system
message and its framing add exactly **46 tokens** to every measured prompt.
Training prompt tokens sum to **1,110,316** per pass over all role occurrences in
the proposed treatment, before generated tokens or repeated joint sampling.

Training has 270 two-role, 1,081 three-role and 268 four-role groups. Validation
has 30 / 119 / 32 respectively; heldout has 771 two-role and 600 four-role groups.
The largest training joint sample sums to **1,168 prompt tokens** across its
roles. These are observed input lengths, not a selected batch/sequence budget or
an estimate of training activations, gradients, optimizer or rollout memory.

The complete training proposition union contains **2,926 distinct propositions**.
A real own-base anchor would need valid, provenance-bound readouts covering all
of them from the same selected initial checkpoint. This preflight generated zero
readouts. Repeated propositions across episode splits remain possible under the
existing split policy; length measurement establishes no contamination guarantee.

## Two output-contract hazards

The public files disagree about termination:

| Source | Value |
| --- | --- |
| Tokenizer EOS | `<|im_end|>`, ID **248046** |
| Tokenizer PAD | `<|endoftext|>`, ID **248044** |
| Model text configuration EOS | ID **248044**, also the tokenizer's PAD |

The earlier pinned metadata inspection found no `generation_config.json` at this
revision. A candidate runtime must explicitly resolve termination. The narrow
adapter proposes stopping on **248046**; it does not change a model configuration
or claim that this proposal has been validated by real generation. Generated
248044 is retained as visible `<|endoftext|>` and cannot become a valid decimal.
Any future alternate terminal treatment requires an explicit revised contract.

The tokenizer has **248,044 base entries / 248,077 total entries**, while the
model output vocabulary has **248,320 positions**. Every ID in **248077–248319**
is unmapped: **243 IDs**. The actual stock decoder silently drops all of them,
even with `skip_special_tokens=False`. For example, manually supplied fixture
IDs for `0.5` followed by 248319 decode to `0.5`, concealing the malformed token.
This check exhaustively covered all 243 IDs; it used no sampled model outputs.

`training/qwen_tokenizer.py` rejects an unmapped or malformed ID before decoding
and retains the complete original token sequence on `CompletionDecodeError`.
The caller must persist these raw IDs and stop the failed step. It does not mask
those IDs in the model distribution, repair them, or regenerate a favorable
sample. It renders all mapped control/reasoning text with cleanup and special
token skipping disabled. Only **one final explicitly proposed EOS** is omitted
from rendered completion text; its raw token remains available for policy credit.
Repeated EOS, nonfinal EOS, PAD, reasoning blocks, abstention and commentary stay
visible and fail the existing whole-completion decimal parser.

## Reproduce without weights or inference

The separate environment pins **Transformers 5.17.0, tokenizers 0.23.2, Jinja2
3.1.6**, and all 32 wheel dependencies. It contains no torch. Use Python 3.12 on
Linux x86_64 with venv/ensurepip support, or the existing task-local environment
on General, where pip was bootstrapped from a separately verified 25.0.1 wheel:

```sh
python3.12 -m venv runs/tokenizer-venv
runs/tokenizer-venv/bin/python -m pip install --require-hashes --only-binary=:all: \
  -r training/requirements-qwen-tokenizer-py312-linux-x86_64.lock
runs/tokenizer-venv/bin/python scripts/preflight_qwen_tokenizer.py \
  --artifacts runs/qwen-public-artifacts --download --output runs/qwen-tokenizer-preflight
QWEN_TOKENIZER_ARTIFACTS=runs/qwen-public-artifacts \
  runs/tokenizer-venv/bin/python -m unittest test_qwen_tokenizer -v
runs/tokenizer-venv/bin/python -m pip check
```

`--download` fetches only the three manifest-pinned public JSON files into a new
directory, without authentication. It downloads no weights or remote Python.
Without that flag the collector is local-only. `AutoTokenizer` is called with
`local_files_only=True` and `trust_remote_code=False`; the template is the pinned
public Jinja data. Output directories must be new. The original synthetic CPU
environment and training-loop files remain unchanged.

The reusable `QwenTextTokenizer` exposes `encode_messages` and
`decode_completion` callbacks for the existing episode adapter. Call
`verify_runtime()` before and after each planned batch to inspect complete
artifact and backend content; each prompt also checks template, special IDs and
truncation/padding settings. These tokenizer bindings do not supply the missing
real-weight identity or complete anchor required to instantiate a real training
experiment. The separate user-only encoding method is a measurement comparison.

Private ignored `runs/` evidence contains every exact rendered prompt, its full
token IDs and hashes, per-episode role lengths, all unmapped-ID decoder checks,
and the complete report. The committed report hashes these receipts. Six tests
exercise actual roundtrips, fresh framing, termination, retained controls and
reasoning, all 243 unmapped IDs, malformed IDs and runtime/artifact drift. Ordinary
repository tests skip this optional module when the runtime or artifacts are
absent. Real candidate forward/backward, memory, output behavior, base-belief
collection and training quality remain separate work.
