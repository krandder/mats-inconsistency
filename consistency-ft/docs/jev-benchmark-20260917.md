# Jev benchmark — September 17, 2026

Jev 1.13.0 answered **591/591** distinct original benchmark questions independently,
covering all **300/300** groups. The same observations also cover the reviewed
**274/274 questions / 157/157 groups**. No scope changed and no existing score changed.

| Comparison | Common groups | Jev mean algebraic violation | Rank |
|---|---:|---:|---:|
| Original overview | 264 | 0.0873459659 | 10 / 24 ranked settings |
| Corrected overview | 148 | 0.0863932432 | 8 / 24 ranked settings |
| All original Jev groups | 300 | 0.0922900167 | not a shared-cohort ranking |
| All reviewed Jev groups | 157 | 0.0895318471 | not a shared-cohort ranking |

Muse remains visible and unranked: 25 displayed settings, 24 ranked. The original
and corrected common group IDs, and all existing models' scores on those groups,
are unchanged. This is logical coherence, not accuracy or calibration; no
statistical-significance claim follows from the ordering.

## Fixed, untuned protocol

Provider identity is `typesafe/jev-1.13.0@noul`, campaign
`typesafe-20260917-original`, protocol `typesafe-noul-independent-v1`. Earlier
same-day exploratory calls resolved `jev-latest` to `jev-1.13.0`; this run requests
that exact version and rejects any different served model.

One fresh `POST https://api.typesafe.ai/v1/systemone` per distinct proposition,
with four concurrent requests. The state is the existing catalog's exact
proposition text, including complete resolution criteria. A single Noul question
named `proposition` has this instruction:

> Using your existing knowledge and the event definition and resolution criteria in the state, is the proposition true? Estimate its likelihood; do not use external tools or browsing.

No other propositions, supplied probabilities, relation hints, conversations,
retrieval or prompt optimization were added. Shared propositions reuse a single
observation. There is no explicit historical as-of constraint, matching the
existing catalog protocol's use of the model's knowledge at evaluation time.

Noul directly returns a yes probability. Its scalar is not a verbal answer or a
True/False token-logprob pair. No explicit abstention field or reasoning effort
control is exposed; the API does not provide evidence about hidden provider
implementation. The exact request/response bodies, model version, source IDs,
SHA-256 identities, usage and timing are archived. This channel differs from
native `descriptive-v2`, original native `numeric-v1`, and historical Fireworks
chat prompting. Equal group IDs and formulas do not isolate model quality from
readout method or collection date. Probabilities often have two-decimal precision.

## Attempts and cost

592 total attempts produced 591 probabilities. One HTTP 529 returned
`system_overloaded`; its original receipt remains intact. The first runner called
it `http_error`; the exporter preserves that original classification and marks
it infrastructure. Only this failed request was retried, once. No successful
answer was resampled. The run spanned 00:28:26–00:30:34 UTC.

Median successful end-to-end HTTPS latency: **0.662 seconds**. Reported totals:
**270,285 input tokens**, **13,593 output tokens**. Estimated reported-token cost:
**$0.011352**, at the published $0.042/million input tokens and free output.
This is not an invoice measurement; the failed overload attempt has no usage
telemetry, so its consumption is unknown, not asserted to be zero.

## Reproduction and receipts

- [All 592 request/response receipts](../data/benchmark-runs/typesafe-20260917-original.receipts.jsonl)
- [300 group scores](../data/benchmark-runs/typesafe-20260917-original.jsonl)
- [Hash-verified manifest](../data/benchmark-runs/manifest.json)
- [Collector](../harness/run_jev_checks.py), [exporter](../scripts/export_jev.py), [regression checks](../test_jev.py)

The credential stays in deploy-host's existing shared secret file and is never put in
receipts or the repository. The raw per-attempt files also remain on deploy-host at
`~/experiments/jev-consistency-20260917/receipts/`.

```sh
python3 scripts/export_benchmark.py
python3 scripts/build_dash.py
python3 -m unittest test_jev test_scoped_dashboard test_comparison_policy test_benchmark_data
```

The existing `common.violation` formulas and shared-cohort ranking generators are
used without a new metric or tuned readout. The database retains 65 runs.
