# Joint forecasts with market evidence (approved by Kelvin, 23 Sep 2026)

Kelvin: "training on simple accuracy is kinda stupid, because the pressure to
just follow the market is intense. But on the joint P & Q kind of thing, you
have no price. You have to look at the prices, but still reason about the
conditional!" Then: "the model doesn't have to believe on the market
probabilities. The consistency should be internal, not w.r.t market price."
And on search: "We're not training for search of evidence yet."

## Evidence pack (input)

Per pair, one fixed evidence pack, built by deterministic retrieval (no model
writes it, so it carries no conclusions): both markets and the most related
other markets on Manifold, with prices and price history, recent trades and
comments, and descriptions as they stood at t (7 days before the earlier
close). Nothing at or after t (`asof.py`, leak-tested). Every call for the
pair, and every model (Qwen3-0.6B, Qwen3-32B, Qwen3-235B, Luna), gets the
identical pack. No tools and no model-driven search in this phase.

## Output

The model states its own probability for each of the 12 forms, one form per
call (12 separate calls per pair, same pack). Market prices are evidence, not
constraints: the model may disagree with them. Consistency is internal: the
12 answers are checked against each other (complements, De Morgan, Fréchet
bounds), and that is what consistency training targets.

## Scores

- Joint forms (A and B, A or B, signed variants): the main result, against
  the baseline of market prices with A and B assumed independent.
- Single forms: against the market price at t (does the model add anything
  to the market?).
- Internal consistency of the 12 separate answers.

Report by pair type (options of the same multiple-choice market are mutually
exclusive; same topic; other).

## Earlier results this affects

The GPT-6 Luna and Qwen3-235B zero-shot runs asked all 12 forms in one call,
which makes coherence easy and helps log loss; they are not comparable with
the one-form-per-call Qwen3-32B and 0.6B numbers until rerun one form per
call.
