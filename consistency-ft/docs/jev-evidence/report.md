# Jev agent-update triage experiment — 2026-09-17

The latest alias `jev-latest` resolved to **jev-1.13.0**. It correctly classified all 24 pre-labelled synthetic agent updates (11 requiring attention, 13 routine), and selected `uncertain` for both deliberately underspecified updates. This is a promising smoke test, not evidence that it is safe to suppress real fleet alerts.

## Method

One independent API request per update from deploy-host, using a fixed Choice question with three options: attention, routine, uncertain. Labels were written before any calls. Inputs were synthetic and did not contain private fleet transcripts. The rubric requires escalation for serious unresolved incidents, material risks, genuine Kelvin-only decisions, and blockers only Kelvin can resolve; it excludes routine work, resolved incidents, redundant permission requests, and external waits with no useful Kelvin intervention. The update is explicitly treated as untrusted data.

Cases include negation, a quoted incident in a test fixture, a misleading 'no action needed' assertion, a redundant permission request, and one embedded instruction to misclassify an outage. These are tiny illustrative checks, not a security evaluation.

## Measurements

| Measure | Result |
|---|---:|
| Label agreement | 24/24 |
| Missed attention cases | 0/11 |
| False alarms | 0/13 |
| Ambiguous cases routed to uncertain | 2/2 |
| Median end-to-end latency | 690 ms |
| Maximum end-to-end latency | 1,569 ms |
| Reported input tokens | 12,236 |
| Estimated input cost at published $0.042/M tokens | $0.000514 |

Latency includes fresh HTTPS connections from deploy-host; this is not model-only inference time. Cost is a rate-based estimate, not an invoice measurement. Raw response usage is preserved in `results.json`.

## Confidence and practical use

The explicit approval and product-policy cases were correct but returned confidence 0.68 and winning probability 0.78. The two ambiguous cases returned confidence 0.81 and 0.45 respectively: a model can be confident that the appropriate category is uncertain. Confidence is a statistic of the returned distribution, not a separate learned guarantee of correctness.

Do not interpret confidence 0.9 as a verified 90% accuracy guarantee on fleet traffic. This sample has no errors, is small, was constructed by the evaluator, and is intentionally clear in most cases. It cannot establish calibration or sensitivity on realistic incidents. No incumbent model or simple-rule baseline was evaluated.

A sensible next experiment is an offline, independently labelled sample of real updates with realistic class balance, followed by shadow-mode routing. Measure missed incidents and decisions, false alarms, abstention rate, probability calibration, and latency. Until then, use Jev to suggest prioritization while retaining the full update stream. No fleet integration or alert suppression was enabled.

## Reproduce and inspect

- `eval.py`: fixed policy, pre-labelled examples, HTTP runner; loads the deploy-host credential without printing it.
- `cases.json`: cases and labels.
- `results.json`: full responses and observed latency.
- `summary.json`: aggregate results.
- API: https://docs.typesafe.ai/api
- Confidence: https://docs.typesafe.ai/confidence
- Published pricing: https://typesafe.ai/blog/introducing-system-one-models-and-jev
