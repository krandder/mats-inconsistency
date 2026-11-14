# Durable complete-split Qwen own-base collection

`scripts/collect_qwen_base.py` prepares and manages an engineering collection of
the complete original training split: **1,619 episodes and 2,926 distinct full
propositions**, including resolution criteria. Implementing this driver does not
select a scientific model, perform collection, establish a complete anchor, or
start training. No model downloads, provider calls, paid resources, optimizer,
lambda choice or experiment binding are implemented.

The commands are separate:

- `plan`: verify an externally SHA-pinned specification, copy its exact inputs,
  and publish the complete deterministic catalog and immutable training identity.
- `preflight` / `status`: verify the external plan pin, input/source identities,
  original full catalog and ledger coverage. These paths never import Torch or
  load candidate weights.
- `collect`: in a **fresh process**, load the pinned local checkpoint and actual
  existing `QwenLoraRuntime`, verify its initial base/adapter/execution identities,
  and attempt only untouched propositions under caller-supplied limits.
- `export`: with complete valid coverage only, explicitly derive an exact or
  symmetric-mixture anchor using the existing `anchors.freeze_from_manifest`.

## Externally pinned specification

The exact JSON object has these keys; no scientific settings have defaults:

| Key | Required value |
| --- | --- |
| `schema_version` | Integer `1`. |
| `collection_id` | Stable simple unique collection identifier. |
| `master_seed` | Explicit unsigned 64-bit integer. |
| `max_new_tokens` | Explicit fixed per-proposition cap, 1–4,096. |
| `protocol_id`, `protocol` | `numeric-v1` and its exact original system/user templates. |
| `sources` | Exact mapping from every relative path in `training.qwen_base_collection.SOURCE_FILES` to the current reviewed file SHA-256. |
| `execution` | Full expected runtime execution object, explicitly supplied before inference. |
| `initial_runtime_identity` | `{ "path": "/absolute/reference.json", "sha256": "..." }` for the verified initial base/seeded-A/zero-B manifest. |
| `checkpoint_manifest` | The same path/hash object for the pinned published checkpoint manifest. |
| `tokenizer_manifest` | The same path/hash object for the published tokenizer artifact manifest. |
| `train`, `valid`, `heldout` | Separate path/hash objects for all three complete original split files. |

The driver supports the verified CPU treatment: two threads, eager attention,
cache disabled, eval mode, deterministic algorithms, rank-16/alpha-32 adapters on
all 248 decoder Linear targets and 43,278,336 FP32 adapter parameters. The explicit
execution must equal the reference execution except for the reviewed current
runtime source hash and categorical sampler identity. These include the token
callback and shared FP32-softmax/FP64-CDF sampler. This produces a **new
execution/policy identity**. No previous three-role engineering readout is imported
or relabeled as a collector observation.

The source map includes `training/categorical.py`; the execution also binds its
algorithm name and SHA-256. Plans from before this sampler correction reject
against the revised source and execution identities. Existing campaigns retain
their original code pins and attempts; this change does not restart one.

The immutable nine-field `training_identity` follows the existing adapter schema.
Its `weights_manifest_sha256` binds both original base tensors and the full
initial seeded A/zero B adapter manifests, not merely the pretrained archive.
The runtime re-creates and verifies those actual bytes before the first attempt
of every invocation. Tokenizer files, chat template, original protocol, revisions,
split bytes, seed recipe, sampler, EOS and cap are also bound. The complete runtime
execution identity is bound separately from initial training weight identity.
Future training must retain that initial identity while current adapters change.

`source_identity()` is a pure helper for constructing the source map after review.
Constructing a spec is the caller's explicit step; the driver does not choose its
seed, cap, source reference or treatment. Capture the SHA-256 of the exact spec
file externally, then:

```sh
python3 -B scripts/collect_qwen_base.py plan \
  --spec "$QWEN_SPEC" --spec-sha256 "$QWEN_SPEC_SHA256" --directory "$QWEN_COLLECTION"
python3 -B scripts/collect_qwen_base.py preflight \
  --directory "$QWEN_COLLECTION" --plan-sha256 "$QWEN_PLAN_SHA256"
```

The plan command creates a new directory and prints its exact `plan.json` file
digest. Pin that value externally for every later command. Source or input drift
fails verification across resumes; it is never silently accepted as the same
collection. The plan copies original splits/manifests, so subsequent operations
use immutable local copies rather than mutable external input paths.

## Ordering, ownership and bounded invocations

Catalog order is ascending SHA-256 of each full UTF-8 proposition. Before any
inference, every row records its complete text, messages, index, stable seed and
seed derivation hash. Each seed is the unsigned big-endian first eight bytes of
SHA-256 of canonical JSON containing `master_seed`, `collection_id`, the external
spec digest and proposition ID. Each proposition starts a separate CPU generator.
Grouping untouched IDs into different bounded invocations cannot change their
starting RNG. Receipts retain seed and RNG-state **hashes**, not serialized states;
interrupted RNG trajectories are not resumed.

A stable `flock` file excludes concurrent collectors and exports; status reads
take a shared lock. Before sampling, the owner creates and fsyncs the proposition's
attempt directory, then publishes its exact prompt/seed/identity receipt. **Directory
existence permanently consumes the attempt**, even if a crash happens before its
first token. Empty markers, partial JSONL writes, failed callbacks, invalid answers,
refusals, abstentions, truncations and decode errors are preserved. There is no
cleanup, retry, replacement, reseed, filtering or denominator reduction.

Normal invalid outputs allow later untouched IDs to be attempted. Runtime failures
stop the invocation; a later process may attempt other untouched IDs. An interrupted
or invalid proposition leaves this campaign unable to export a complete valid
anchor. This is durable no-retry behavior, not a promise of successful full anchor
recovery after a crash. A new collection/treatment would require a separate explicit
caller decision and new identity.

The actual runtime's `generate_tokens(..., on_token=...)` owns sampling. The callback
fsyncs token ID, sampled logprob, EOS flag, post-sample RNG hash and timings immediately
after each sample and before the next forward. Each complete receipt also retains
the exact prompt/completion tokens, unmodified response, classification, probability
when valid, per-token logprobs, attempt/token-file hashes and initial policy identity.
The public runtime verifies full frozen-base contents and unchanged adapters before
returning; unsuccessful integrity checks cannot produce a valid anchor receipt.

Every collection invocation requires explicit positive limits:

```text
--max-attempts N --token-budget N --wall-seconds N
--cpu-soft-seconds N --cpu-hard-seconds N --address-space-bytes N
```

Address space cannot exceed 36 GiB; CPU hard limit must exceed the soft limit.
The two-thread treatment is fixed by the pinned execution. Use separate fresh CLI
processes for invocations: process resource limits and Torch interop configuration
are not a reusable in-process service interface.

The wall clock begins before weight loading. Admission checks elapsed time before
and after prompt preparation. An attempt starts only if the **entire fixed token
cap fits the remaining invocation token budget**. Remaining budget below that cap
means no new attempt; it never lowers the cap. Actual sampled tokens, including EOS,
consume the budget. A time limit can interrupt an admitted attempt; its evidence
remains consumed. No attempt is started after the deadline.

Use the existing offline environment and an external supervisor, retaining its
stdout, stderr, exit code and `/usr/bin/time -v` result outside the new invocation
directory. Example command shape (all values supplied explicitly by the caller):

```sh
PYTHONPATH="$QWEN_TOKENIZER_SITE_PACKAGES" /usr/bin/time -v -o "$QWEN_SUPERVISOR/time.txt" \
  timeout --kill-after=10s "$QWEN_EXTERNAL_WALL_SECONDS" "$QWEN_CPU_PYTHON" -B \
  scripts/collect_qwen_base.py collect --directory "$QWEN_COLLECTION" --plan-sha256 "$QWEN_PLAN_SHA256" \
  --checkpoint "$QWEN_CHECKPOINT_DIRECTORY" --tokenizer-artifacts "$QWEN_TOKENIZER_DIRECTORY" \
  --max-attempts "$QWEN_ATTEMPT_LIMIT" --token-budget "$QWEN_TOKEN_BUDGET" --wall-seconds "$QWEN_INTERNAL_WALL_SECONDS" \
  --cpu-soft-seconds "$QWEN_CPU_SOFT_SECONDS" --cpu-hard-seconds "$QWEN_CPU_HARD_SECONDS" \
  --address-space-bytes "$QWEN_ADDRESS_SPACE_BYTES" \
  >"$QWEN_SUPERVISOR/stdout.log" 2>"$QWEN_SUPERVISOR/stderr.log"
```

Python signals can be delayed inside native kernels; SIGKILL/OOM can prevent a final
report. The external supervisor remains necessary. Attempt markers/token receipts
and invocation phase records survive independently. No candidate processes are
launched by plan, preflight, status, export or the collector's tests.

## Separate complete-only anchor derivation

```sh
python3 -B scripts/collect_qwen_base.py export \
  --directory "$QWEN_COLLECTION" --plan-sha256 "$QWEN_PLAN_SHA256" --output "$QWEN_ANCHOR_OUTPUT" --mode exact
```

Alternatively require `--mode symmetric-mixture --smoothing-epsilon E`, with explicit
representable `0 < E < 0.5`. This applies the existing symmetric mixture to both
distributions during KL evaluation; it never edits observed probabilities. Exact
0/1 observations remain 0/1 in raw receipts and snapshots. Positive-lambda exact A2
can later have infinite rewards; collection/export does not choose lambda, silently
smooth endpoints, or change the training adapter's nonfinite-reward policy.

Export rejects anything below **2,926 valid probabilities**. It publishes a
compatible native-receipt manifest/catalog, frozen snapshot, immutable initial
training identity and initial weight manifest. The existing training adapter still
rejects partial original-split snapshots for every arm. No `allow_partial` or subset
option exists in this driver. Its exported envelope retains hashes/provenance, while
raw token-event, attempt and invocation evidence remains in the original collection
directory; preserve that directory for full runtime audit.

`python3 -B -m unittest test_qwen_base_collection -v` exercises full original planning
without Torch, exact source/split pins, lock exclusion, interrupted-marker retention,
per-ID RNG across tranche boundaries, cap admission, callback failure, malformed
claimed probability, receipt-binding tampering and complete exact/smoothing export
through the existing anchor loader. Generated values in these tests come from
explicit fake runtime fixtures; no candidate weights or candidate forecasts are used.
