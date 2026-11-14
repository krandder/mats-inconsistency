# Reasoning-mode sensitivity: existing Luna runs

Task: `hub-t8k9c.10`. This analysis makes no new model requests. It uses the
six already published Luna efforts, with exact matching group IDs within each
protocol. The broader historical Fireworks sweep remains unrun.

## native-20260912 (numeric-v1)

264 shared scored groups from 300 target groups; 516 unique questions contribute to those scores.

| Effort | Shared mean violation | Available groups | Missing probabilities | Output tokens / shared question | Median seconds / shared question |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 0.109655 | 300/300 | 0/591 | 7.1 | 3.03 |
| low | 0.093294 | 280/300 | 22/591 | 39.0 | 3.75 |
| medium | 0.104767 | 272/300 | 26/591 | 72.8 | 4.63 |
| high | 0.102646 | 272/300 | 31/591 | 123.5 | 5.37 |
| xhigh | 0.066522 | 268/300 | 29/591 | 247.7 | 7.43 |
| max | 0.066611 | 278/300 | 27/591 | 732.1 | 12.00 |

Paired differences below are **right minus left** on that same shared set.
Negative differences mean lower violation with the right-hand effort.

| Left → right | Mean difference | Groups improved / worsened / tied |
| --- | ---: | ---: |
| none → medium | -0.004889 | 120 / 106 / 38 |
| none → high | -0.007009 | 124 / 100 / 40 |
| none → xhigh | -0.043133 | 142 / 76 / 46 |
| none → max | -0.043044 | 154 / 70 / 40 |
| low → medium | +0.011473 | 92 / 126 / 46 |
| low → high | +0.009353 | 112 / 109 / 43 |
| low → xhigh | -0.026772 | 138 / 82 / 44 |
| low → max | -0.026682 | 142 / 77 / 45 |

Per-check differences for **none → high**:

| Check | Groups | Mean difference |
| --- | ---: | ---: |
| And | 26 | +0.019919 |
| AndOr | 26 | -0.028387 |
| But | 25 | -0.013520 |
| Cond | 26 | -0.060286 |
| CondCond | 25 | -0.006179 |
| Consequence | 27 | +0.015152 |
| ExpEvidence | 27 | -0.045216 |
| Negation | 28 | +0.011639 |
| Or | 26 | -0.005631 |
| Paraphrase | 28 | +0.037929 |

Exact shared question texts link these groups into 78 components; the largest contains 12 groups. Additional dependence through differently worded versions of the same event is possible.

## native-20260913-scoped-final (descriptive-v2)

150 shared scored groups from 157 target groups; 260 unique questions contribute to those scores.

| Effort | Shared mean violation | Available groups | Missing probabilities | Output tokens / shared question | Median seconds / shared question |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 0.122420 | 150/157 | 5/274 | 7.0 | 2.28 |
| low | 0.104320 | 151/157 | 1/274 | 74.8 | 4.66 |
| medium | 0.090332 | 155/157 | 2/274 | 124.3 | 5.51 |
| high | 0.078166 | 151/157 | 1/274 | 198.7 | 7.40 |
| xhigh | 0.095156 | 151/157 | 1/274 | 353.3 | 10.18 |
| max | 0.063306 | 157/157 | 0/274 | 999.1 | 20.56 |

Paired differences below are **right minus left** on that same shared set.
Negative differences mean lower violation with the right-hand effort.

| Left → right | Mean difference | Groups improved / worsened / tied |
| --- | ---: | ---: |
| none → medium | -0.032088 | 79 / 49 / 22 |
| none → high | -0.044254 | 78 / 45 / 27 |
| none → xhigh | -0.027264 | 78 / 53 / 19 |
| none → max | -0.059113 | 88 / 39 / 23 |
| low → medium | -0.013989 | 72 / 52 / 26 |
| low → high | -0.026155 | 78 / 48 / 24 |
| low → xhigh | -0.009164 | 79 / 53 / 18 |
| low → max | -0.041014 | 77 / 50 / 23 |

Per-check differences for **none → high**:

| Check | Groups | Mean difference |
| --- | ---: | ---: |
| And | 14 | -0.010214 |
| AndOr | 15 | -0.143234 |
| But | 15 | -0.163007 |
| Cond | 14 | +0.012702 |
| CondCond | 2 | -0.151059 |
| Consequence | 20 | -0.006045 |
| ExpEvidence | 13 | -0.109107 |
| Negation | 22 | +0.025205 |
| Or | 15 | -0.085860 |
| Paraphrase | 20 | +0.024775 |

Exact shared question texts link these groups into 39 components; the largest contains 7 groups. Additional dependence through differently worded versions of the same event is possible.

## What this establishes and what remains

- All effort comparisons within a campaign use the identical six-effort intersection.
- Original and corrected protocols/cohorts are separate experiments; their raw means are not paired treatment effects.
- This is a single observed run per effort, with no randomized repeated trials; no causal or significance claim is made.
- Groups share questions and events. Counts of improved groups are not independent statistical trials.
- Coverage selection excludes refusals and abstentions; missing readouts are never zero.
- Latency and token counters describe original receipts on shared questions, including reused observations; they are not new consumption.
- Luna is one model. This does not complete the Fireworks model-wide medium/high sweep or choose an open training base.

The machine-readable [report](../data/research/reasoning-sensitivity.json) contains all
15 pairwise effort comparisons per campaign, every per-check delta, exact shared
IDs, frozen source hashes/timestamps, usage coverage, and numerical missingness.

Reproduce without inference:

```sh
python3 scripts/analyze_reasoning.py
```

Remaining Fireworks work: verify supported/effective effort settings, complete the
missing original models, and collect medium/high runs on the same original cohort
with explicit coverage limits. An accepted request field alone does not prove that
the provider applied the requested reasoning mode.
