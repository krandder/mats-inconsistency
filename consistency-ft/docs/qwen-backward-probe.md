# One retained-readout numerical backward diagnostic

`scripts/probe_qwen_backward.py` prepares one actual pretrained numerical
forward/backward diagnostic. Implementation and fixture checks do not establish
that this candidate backward has run. It uses the original first predeclared
P-role readout from `2028/AndChecker:11`, only after the complete three-role base
probe has a successful report and unchanged-weight evidence. It never chooses a
later valid response, regenerates, manufactures an anchor/credit, or creates an
optimizer. The derivative is exactly `sum(logprob(completion_token)) * 1.0`,
including EOS, with no token/role normalization or training-quality claim.

The caller must capture an immutable JSON mapping from each of `plan.json`,
`loaded-checkpoint.json`, `readouts.jsonl`, `report.json`, `token-events.jsonl` to
its SHA-256 **before starting the candidate process**, then also capture the
digest of that exact JSON file. The currently completed original probe has:

```text
input map SHA256: e727456e9387ce4deb1ac269f5a28b80efa61a246c42d7090bb867856c76a66f
first P line SHA256: 05c9f5ae0c1fa72ed18010fd5cb989a534b0f7060c24e3619722c379c2a87cb9
```

Its four exact completion tokens, original prompt/messages and behavior logprobs
are retained unchanged. The script checks corpus, episode, prompt rendering,
tokenizer, protocol, source, checkpoint and token-event identities. The verified
loader source is pinned internally; the settled reviewed LoRA runtime source
digest must be supplied explicitly. Both modules must be installed in the same
checkout before executing this diagnostic; the implementation worktree does not
implicitly import another agent's mutable checkout.

Use the existing offline CPU/tokenizer environment, with a fresh outer attempt
directory for the supervisor records and a nonexistent inner result directory:

```sh
PYTHONPATH="$QWEN_TOKENIZER_SITE_PACKAGES" /usr/bin/time -v -o "$QWEN_ATTEMPT/time.txt" \
  timeout --kill-after=10s 900s "$QWEN_CPU_PYTHON" -B scripts/probe_qwen_backward.py \
  --checkpoint "$QWEN_CHECKPOINT_DIRECTORY" --manifest "$QWEN_CHECKPOINT_MANIFEST" \
  --tokenizer-artifacts "$QWEN_TOKENIZER_DIRECTORY" --base-probe "$QWEN_BASE_PROBE" \
  --input-digests "$QWEN_INPUT_DIGESTS" --input-digests-sha256 "$QWEN_INPUT_DIGESTS_SHA256" \
  --runtime-source-sha256 "$QWEN_REVIEWED_RUNTIME_SHA256" --output "$QWEN_ATTEMPT/result" \
  >"$QWEN_ATTEMPT/stdout.log" 2>"$QWEN_ATTEMPT/stderr.log"
```

The caller retains the supervisor exit code, logs and `/usr/bin/time` receipt,
including timeout/OOM/signal failures. Internally the script enforces 36 GiB
address space, CPU soft/hard limits 900/910 seconds, a 900-second alarm, two Torch
threads and one interop thread. Python cannot reliably interrupt native kernels
or write a final report after SIGKILL; the external supervisor is required.

The runtime installs all 248 decoder Linear targets: rank 16, alpha 32,
43,278,336 FP32 adapter parameters, seeded A and zero B. The initial effective
update is zero. One temporary top-level model forward hook retains the exact
selected-token FP32 log-softmax values from the single `role_loss` forward,
before the tested primitive compares them with behavior at its unchanged
`atol=2e-6, rtol=2e-6`. A mismatch is saved and fails; it never loosens tolerance,
changes numerical treatment, runs an extra forward, or proceeds to backward.
The hook is removed before backward.

On successful backward, bounded reductions check all adapter gradients for
presence and finiteness; zero A gradients are allowed. Full streaming base hashes
and adapter manifests check unchanged values afterward. No base clone, gradient
tensor save or optimizer snapshot is made. Per-tensor gradient summaries contain
only counts, shapes, norms and maxima. Ordinary failures retain the original
exception immediately, then attempt integrity audits within the declared budget;
insufficient audit budget remains a failure, never a successful diagnostic.
Inputs, sources and tokenizer are rechecked at completion.

Lightweight implementation checks: seven receipt tests covered the actual
successful base report, digest/content tampering, incomplete/failed reports,
first-role-only selection, behavior/token-event disagreement and invalid JSON.
Three tiny Torch fixtures exercised the existing `role_loss` primitive with
a one-forward mismatch hook, nonfinite exact-value retention, and finite/zero/
missing/nonfinite gradient summaries. These tests load no candidate weights.
