# Jev API experiments — 2026-09-17

**73 successful calls to `jev-latest`, all resolved to `jev-1.13.0`.** Jev looks useful for parallel decisions, but its probabilities are demonstrably not guaranteed to form a coherent belief system. These probes challenge the simplest independent-fixed-score/softmax explanation without identifying its internal architecture or training objective.

## Design and scope

The initial 64 calls comprised 19 conditions repeated three times, with seven additional identical baseline calls (ten total). Order was shuffled with a fixed seed. A fixed synthetic software issue supplied the state for architecture probes. Four synthetic probability problems supplied logical probes. Nine follow-up calls tested direct yes/no wording, a joint Choice, and fully expanded event descriptions. Labels/conditions were specified in scripts before calls; follow-ups were selected after observing failures.

No actual fleet data was sent. Raw requests, full responses, model IDs, usage and end-to-end deploy-host timing are preserved. Every submitted question received an answer. Estimated cost for all 73 calls at the published input rate was **$0.001482**, for 35,283 reported input tokens; this is not an invoice measurement.

## Strongest result: cross-question incoherence

Let an integer be uniformly drawn from 1 through 10. Let A mean it is 1 or 2, and B mean it is at most 5. Necessarily A implies B, so A AND B equals A, and A OR B equals B.

| Elicitation | P(A) | P(A AND B) | P(B) | P(A OR B) |
|---|---:|---:|---:|---:|
| Original probability wording, one example | .27 | .33 | .59 | .59 |
| Direct yes/no, one example | .23 | .43 | .64 | .82 |
| Expanded descriptions without A/B shorthand, one example | .18 | .30 | .47 | .85 |
| Exact values | .20 | .20 | .50 | .50 |

The conjunction upper bound failed in **all nine calls** across these three formulations. Expanded-description repeats were A=.18/.18/.17 and conjunction=.30/.28/.28. These gaps exceed ordinary one-percentage-point output rounding. The original three repeats also gave P(A)+P(not A)=1.09–1.10. Expanded wording nearly repaired that particular complement relation while leaving the conjunction and disjunction failures.

A separate uncertain deployment-cause case violated inclusion–exclusion by .14–.20 in the first three runs. No known real-world frequencies are needed to detect the algebraic inconsistency. This is evidence of failed output coherence on these inputs, not a measurement of population calibration, benchmark ranking, or model-wide error rate.

**Control limitation:** direct yes/no and expanded arms also included a joint Choice question. Thus changes across arms cannot be causally attributed solely to wording. Each arm nevertheless contains internally inconsistent outputs on its own terms. This is a small targeted diagnostic, not the project's corrected Consistency-FT benchmark.

## Joint representation helps structure, not automatically truth

We also asked one Choice over four outcomes: both, A only, B only, neither. Deriving marginals from its normalized distribution guarantees Boolean identities over those four cells. It does not guarantee that the distribution respects constraints stated in the evidence.

With symbolic A/B descriptions Jev allocated .06–.11 to A-only, which is impossible here. With explicit integer categories it improved substantially: one response was {.19 both, .01 impossible, .23 B-only, .57 neither}, versus exact {.20, 0, .30, .50}. Its joint-derived P(A)=.20 disagreed with independently elicited .18, and joint-derived P(B)=.42 disagreed with .47. A representation can enforce some consistency mechanically while still having support errors and inconsistent independently elicited marginals.

This is precisely relevant to Kelvin's prior belief-consistency research: probability-native output is an interface; cross-proposition coherence is an additional property to train, enforce or audit. See `consistency-relevance.md` for the mapping to the existing project and its proper next benchmark.

## Architecture probes

| Probe | Observation | Supported interpretation and limit |
|---|---|---|
| Identical request, 10 calls | P(blocking) ranged .83–.89; same winning label every time | Returned probabilities are not deterministic here. Sampling, numerical randomness or service behavior remain possible. |
| 1 vs 101 questions | Median end-to-end latency .670s vs .688s; target means .866 vs .863 | Consistent with efficient parallel/shared computation. Three large-batch repeats and network overhead cannot identify the attention layout or establish general throughput. All 101 answers were present. |
| Rename question ID | Same input-token count; target probability .85 in all three repeats, within baseline range | Consistent with documentation saying IDs are excluded from inference. Does not prove exact invariance. |
| Reverse Choice order / add irrelevant option | Means .870/.867 versus .866 baseline | No clear effect beyond baseline variation on this example. |
| Hostile instruction in a neighbouring question | Target P(blocking)=.86–.89; correct winner | Consistent with question isolation. One payload is not a security guarantee. |
| Neutral Choice keys | Semantic winner mean .857 | Near baseline. |
| Conflicting Choice keys | Description for blocking, renamed `degraded`, won at .93 | Descriptions dominated the conflicting label here; changed probabilities suggest keys/format may still matter. |
| Duplicate degraded Choice description | Combined duplicate mass .123 versus original .134 | Duplicates mostly split existing mass. Simple independent fixed weights + softmax would predict combined mass about .236. Challenges that narrow model; correlated sampling, context-dependent scoring or duplicate handling remain possible. |
| Duplicate Score description | Combined duplicate mass .170 versus original .157 | Similarly does not show naive doubling (fixed-weight prediction about .271). Only three repeats. |
| Permute Score levels | Blocking-description mass means .843 original, .873 reversed, .893 rotated; dominant meaning follows its description | Broad semantic stability, but **not exact invariance**: order-associated shifts need more controls. Score values follow new indices as documented. Cannot claim hidden level isolation is proven. |

Published Score values matched weighted sums from returned probabilities within .01, consistent with output rounding. Choice/Score probabilities and Noul values in these calls were on a .01 grid within floating-point tolerance. That does **not** show there are 100 samples: rounding, quantization, or a different sample count can produce the same observation. Confidence sometimes differs for identical rounded probability vectors, so the returned rounded vector is insufficient to reconstruct its exact computation.

## Revised hypotheses

1. **Efficient parallel decision evaluation remains plausible and empirically useful.** The batch probe is consistent with the vendor's claim; it does not establish a particular encoder, KV-cache layout, or model size.
2. **The simplest independent fixed criterion score followed by softmax is inadequate for the duplicate examples.** More sophisticated scoring or correlated sampling could explain them. Choice and Score need not use identical internals.
3. **A globally coherent joint-belief engine is not guaranteed at the API boundary.** Several outputs violate elementary constraints even with explicit event wording. Possible coherent latent internals would still not make these returned judgments coherent.
4. **Nothing here distinguishes RLCD's training loss.** Proper scoring, supervised/distilled distributions and RL remain hypotheses. We did not infer an undisclosed recipe from black-box behavior.

Next bounded research task: evaluate Jev through a separately named Noul protocol on the existing corrected Consistency-FT benchmark, then compare raw probabilities with coherent repair on exact-outcome decision cases. No production changes, benchmark integration or finetuning was performed.

## Artifacts

- `probes.py`, `probe-plan.json`: initial design and shuffled execution schedule.
- `probe-results.json`, `probe-summary.json`: raw observations and grouped statistics.
- `controls.py`, `control-results.json`: direct-question and joint-outcome controls.
- `expanded_controls.py`, `expanded-control-results.json`: explicit event wording.
- `analyze_probes.py`: aggregation code; duplicate Score descriptions are combined per response.
- `consistency-relevance.md`: grounded mapping to Kelvin's earlier research.

Local folder: `/home/agent/experiments/jev-triage/`. Raw remote runs: `deploy-host:~/experiments/jev-triage/`. Public interface references: https://docs.typesafe.ai/api and https://docs.typesafe.ai/primitives/score.
