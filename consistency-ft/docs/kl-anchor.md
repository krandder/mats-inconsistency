# Frozen base beliefs and the A2 reward

`anchors.py` implements the proposal's KL-to-base-beliefs arm without starting
inference, training, or a paid service. It can freeze suitable existing published
readouts, audit their coverage against an exact episode split, and compute the
reward for an episode whose base beliefs are all present.

**This is an offline implementation, not a training-ready checkpoint.** The base
model, platform, lambda, and any smoothing treatment still require an experiment
decision. Fireworks access remains separately blocked. The existing training
split has 1,619 episodes and 2,926 distinct propositions; a 274-question published
run cannot provide complete anchors for that split.

A native client's requested model name does **not** identify an immutable served
build or a trainable weight revision. These snapshots bind the observed request
identity and archived responses only. Before calling this the training base's
own-belief anchor, an experiment must identify the actual trainable checkpoint
and establish that its frozen beliefs came from that same checkpoint. The GLM
example provides no such weight-revision evidence.

For the pinned local Qwen checkpoint, the [durable own-base collector](qwen-base-collection.md)
binds the actual initial base plus all seeded A/zero B adapters before sampling
the full 2,926-proposition catalog. It records every attempt and token, preserves
invalid or interrupted questions as missing, and exports only a complete valid
snapshot. The [execution record](qwen-own-base-collection-20260913.md) distinguishes
initialization failures from sampled readouts. Raw collection leaves lambda and
the endpoint treatment undecided; it does not establish a training-ready anchor.
The v2 collection stopped following a reproduced
[CPU sampler numerical defect](qwen-cpu-sampler-defect-20260913.md): 21 responses
parse as probabilities, one is invalid and 2,904 propositions remain untouched.
Parser success does not validate that campaign's categorical sampling treatment.
The separate [corrected v3 collection](qwen-cdf-collection-20260913.md) uses the
validated full-vocabulary CDF sampler with the same original catalog and initial
weights, importing no earlier observations. Its [continuation report](qwen-cdf-collection-progress-20260913.md)
records partial coverage; complete matching anchors and candidate training remain
unfinished. Observed endpoints remain exact in the receipts; lambda and endpoint
treatment are still scientific choices.

## Definition and normalization

For an episode with roles `S`, candidate probabilities `c_i`, and frozen base
probabilities `b_i`, the anchor is the uniform arithmetic mean

```text
KL_i = c_i log(c_i / b_i) + (1-c_i) log((1-c_i) / (1-b_i))
A2_anchor = (1 / |S|) sum_i KL_i
reward = -Exploit(c, S) - lambda * A2_anchor
```

Logarithms are natural, so the unit is nats. The direction is **candidate to
base**, `KL(c || base)`, not its reverse. Each role contributes once; the anchor
is not an unnormalized sum, a mean of logits, or a joint-distribution KL. Reused
propositions share one frozen base value across episodes. Repeated text in
different roles still contributes once per role to that episode's mean.

Exploitability is the existing LP value from `metrics.exploit`, with its existing
stake bounds; this implementation does not rescale it. Lambda must be finite and
nonnegative. The module returns per-episode reward components and does not choose
batch weighting or a lambda sweep. A trainer that averages these rewards gives
episodes equal weight, which differs from pooling all roles across a batch.

### Exact endpoints and explicit smoothing

Exact probabilities are never clipped:

| Candidate `c` | Base `b` | Exact Bernoulli KL |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| Greater than 0 | 0 | Positive infinity |
| Less than 1 | 1 | Positive infinity |
| 0 | Interior `b` | `-log(1-b)` |
| 1 | Interior `b` | `-log(b)` |

An infinite anchor yields reward negative infinity when lambda is positive.
For lambda zero, reward is `-Exploit` and the diagnostic anchor can still be
infinite; the implementation avoids the undefined floating-point product
`0 * infinity`. It does not invent a finite penalty. Strict JSON writers and some
trainers cannot accept infinity, so transport/trainer handling must be specified
before using the exact treatment for training.
Overflow of an otherwise finite weighted KL raises an error, so it is not
misreported as the mathematical infinity caused by an exact base endpoint.

Optional `--smoothing-epsilon e` defines a separate treatment:
`T_e(p) = e + (1-2e)p`, applied to **both** candidate and base distributions before
KL evaluation. This is a mixture with a fair coin, not clipping. Epsilon must be
in `(0, 0.5)` and large enough that `1-e` is representably below one. The snapshot
retains the original observed probabilities and binds the smoothing mode and
epsilon into its hash. Changing epsilon requires a different frozen artifact
and experiment binding. There is no default smoothing.

Near equality, KL uses an algebraically equivalent series form to avoid
cancellation of leading terms. Extreme interior probabilities use differences
of logarithms rather than ratios, avoiding overflow in `c/b`. Analytical tests
also compare against an independent high-precision Decimal calculation.

## Freeze format and provenance

The JSON envelope contains a `payload` and `snapshot_sha256`, the SHA-256 of the
payload's canonical UTF-8 JSON: sorted keys, compact separators, no NaN/infinity.
The caller must pin this hash outside the artifact. Merely reading a hash from
the same untrusted file is not a tamper check.

The payload binds:

- Requested provider model identifier, reasoning effort, protocol ID, full protocol
  templates, and protocol-content hash.
- Source manifest hash, selected run ID and metadata hash, published receipt
  file hash, proposition-catalog hash, recorded timestamp evidence, and any
  explicit allowed-source-ID list.
- Dataset split name, exact episode-file bytes, each episode's content hash,
  and the complete distinct-proposition union. Full text includes resolution
  criteria and is identified by its UTF-8 SHA-256; no title-only matching occurs.
- Each accepted probability's full text, observation timestamp, prompt hash,
  and canonical published-receipt hash, plus snapshot creation time.
- KL direction through the format's defined A2 contract, explicit endpoint
  treatment, unit and normalization, coverage, and partial/complete status.

All selected receipts must agree with the requested model, effort, campaign,
protocol and exact prompt/system hashes. They must have a successful completed
turn, no tools, a numerical status, and a complete numeric response agreeing with
the saved probability. Out-of-range values are rejected without clipping.
Duplicate current IDs, ambiguous roles, mismatched source IDs/text, changed
source files, and missing observation timestamps are also rejected.

Refusals, `ABSTAIN`, and unsuccessful source turns cannot become anchors. Missing
training propositions are listed by ID and reason. By default a partial freeze
fails. `--allow-partial` permits an explicitly partial artifact, which also lists
complete and missing episode IDs; it does not shrink the dataset denominator.
An allowed-source-ID list restricts imports while retaining the complete bound
split and its missing-coverage audit.

Current legacy exports have labels such as `none-or-low (legacy fallback)` and
do not carry the full per-observation protocol/timestamp evidence needed here.
They are rejected rather than assigning an inferred effort or matching only a
title. The current supported published source is the full native-receipt format.
An older source would need equivalent verifiable provenance before adaptation.

## Offline CLI and snapshot loading

Select the base run explicitly; the tool has no default base model. For a
reviewed set of permitted IDs in a JSON array, a partial offline audit is:

```sh
python3 scripts/freeze_base_beliefs.py \
  --manifest data/benchmark-runs/manifest.json \
  --run-id native-20260913-scoped-final/glm \
  --episodes data/episodes/train.jsonl --split train \
  --allowed-ids runs/kl-anchor-example/allowed-film-ids.json \
  --allow-partial \
  --output runs/kl-anchor-example/glm-film-training.partial.json
```

This example names an existing source to demonstrate the format; it does not
select GLM as the training base. Removing `--allowed-ids` changes source eligibility
and requires the caller to have reviewed the requested scope. No command here
elicits a new answer. Source run and episode-split selection are explicit inputs.

Creation is atomic and refuses any existing destination, including symlinks.
The file is read-only after creation. This prevents accidental overwrite, not
changes by a filesystem administrator; the independently pinned hash detects
changed content even if someone recomputes the envelope's internal hash.

Load with a separately recorded experiment binding, then score only a complete
episode. The example below assumes `config` came from that external binding:

```python
from anchors import load_snapshot, a2_reward

frozen = load_snapshot(
    config["anchor_path"],
    expected_sha256=config["anchor_sha256"],
    expected_base=config["base"],  # model, reasoning, protocol_id, protocol_sha256
    episodes_path=config["episodes_path"],
    split="train",
    allow_partial=True,  # explicit only for a reviewed partial experiment
)
components = a2_reward(episode, candidate_probabilities, frozen, lambda_kl=config["lambda_kl"])
```

Loading rechecks the pinned hash, expected treatment, current split bytes,
episode/text identities, numerical values, timestamps, and coverage counts.
Returned mappings are read-only. `a2_reward` rejects changed or unknown episodes,
extra/missing candidate roles, invalid candidates, and missing base beliefs,
even when lambda is zero. Malformed model output is an error at this interface;
a trainer's malformed-output policy is a separate integration decision.

## Existing-observation example and tests

An ignored local example freezes only the three source-reviewed film/Academy
Award propositions of training episode `2028/AndChecker:11`, reusing published
final GLM receipts. It binds the original training file, not a reduced split:
**3/2,926 propositions and 1/1,619 episodes have complete anchors**. The remaining
2,923 proposition IDs and 1,618 episode IDs are audited as missing. No new
inference or training occurred.

Local artifact: `runs/kl-anchor-example/glm-film-training.partial.json`.
Its snapshot hash is
`a02a682ed527d1ba59da4558756244738adcbfd210c995ee9484e49c7def6530`;
the bound training-file hash is
`71ccc77578c201dc7f51bf06cb9f48f1fecfc1850e3f0472aa349a777924fd2e`.
Creation time is part of the hash, so a new freeze produces a new artifact even
when source probabilities are unchanged. The example remains private/ignored;
it is coverage evidence, not a complete anchor dataset or a model decision.

Run the offline checks with:

```sh
python3 -m unittest test_anchors -v
```

Tests cover known asymmetric values, role normalization, exact endpoints,
explicit smoothing, near-equality precision, extreme finite inputs, analytical
A2 rewards, partial coverage, refusals/nulls, source and split mismatches,
duplicate IDs/roles, invalid probabilities, immutable creation, and externally
pinned tamper detection.
