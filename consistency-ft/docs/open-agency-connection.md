# Consistency research as a component of Open Agency Architecture

Research connection and proposed experiment, 2026-09-12. This note describes
an integration opportunity; it does not report an OAA implementation or a
completed training experiment.

[Explore the illustrated page](https://consistency.futarchy.ai/open-agency.html)
for a visual roadmap and interactive probability example
([HTML source](../site/open-agency.html)).

**The strongest connection is at the input to the world model:** our checks
can identify incompatible probability judgments before they feed a shared
model or a policy comparison. The testable hypothesis is that correcting
those judgments improves downstream decisions while preserving useful
information. Coherence alone does not establish accurate predictions,
faithful human preferences, or safe policies.

## What the roadmap proposes

[Roadmap for a collaborative prototype of an Open Agency Architecture][roadmap]
was published by **Deger Turan on May 10, 2023**. It credits Deger Turan,
Tantum Collins, davidad, and Charbel-Raphael Segerie with writing the post;
Andrew Critch, Ben Goldhaber, Ozzie Gooen, and Evan Miyazono reviewed it.
The complete post and metadata were read through LessWrong's
[official Markdown endpoint][roadmap-text] on 2026-09-12.

The post describes an AOI roadmap for an iterative, modular, human-in-the-loop
prototype, initially applied to institutional decisions such as funding
allocation and participatory budgeting. Its overview separates learning a
world model and eliciting human desirabilities, planning with those inputs,
and acting in real time. The nine projects distinguish feasible choices,
their predicted consequences, stakeholder preferences, and a choice mechanism.

In particular, **project 3** starts with a council of experts supplying
predictions. It proposes studying interfaces such as sliders or binary search
on betting odds, rather than assuming direct numerical elicitation is best.
**Project 4** considers Squiggle-style uncertainty representations and their
translation into finite-state PRISM/JANI models, including difficult
discretization questions. Neither section proposes our consistency objective
or claims that internally coherent probabilities solve model verification.

## Where this repository fits

The left column below summarizes the roadmap; the other columns are our
proposed connections and their limits.

| Roadmap component | Contribution from consistency-ft | Boundary |
| --- | --- | --- |
| **1–2: Formalize feasible actions**, through autoformalization or structured templates | Build proposition identities and logical relations from reviewed templates; check that paraphrases preserve the same event. See our [data design](../README.md#core-design-as-of-2026-08-19-brainstorm). | A probability check cannot prove that an action template captures the intended decision or all relevant constraints. |
| **3: Elicit and integrate expert probabilities** | Reuse the [elicitation comparison](../elicit/run_elicit.py) and [ten relation checks](../common.py) to audit judgments before integration. Test the roadmap's betting-odds interface against direct numbers. | Our existing comparison concerns LLM verbalized probabilities versus True/False token probabilities; it does not validate a human interface or reveal a unique underlying belief. |
| **4 and 6: Represent uncertainty; generate probabilistic programs** | Audit whether elicited marginals and conditionals could describe one joint distribution, then compare them with probabilities implied by a candidate program. | We have no Squiggle/PRISM/JANI adapter. Syntactic validity, semantic fidelity, discretization error, and runtime/model-checking properties require separate checks. |
| **5: Verified RL in evolving probabilistic environments** | Extend the static checks to sequential evidence and measure posterior error; our [environment proposal](../ENVIRONMENTS.md) suggests suitable controlled tasks. | Static `ExpEvidence` checks the law of total probability. It does not show correct updating over time or supply a verified RL controller. |
| **8–9: Elicit preferences and choose through Nash bargaining** | Supply audited forecasts to expected-utility calculations while keeping stakeholder utilities and the choice rule explicit. | Our ranking evaluates probability consistency. It neither elicits legitimate values nor validates a bargaining solution. The roadmap's direct-action utility prototype can also bypass a predictive world model entirely. |
| **7: Cross-scale world models and continuous-time coalgebras** | No implemented contribution claimed. | A finite collection of local probability checks does not establish compositionality of a physical world model. |

An initial integration would therefore sit between **elicitation and model
construction**: store the original judgments, show a concrete violated
relation, allow clarification of the event/evidence, and offer an explicitly
labeled repair. Retain both original and repaired values for review. Export
to a probabilistic program would be a later adapter, with its own validation.

## What we have actually established

Repository evidence reviewed at
[commit 46eed97](https://github.com/futarchy-fi/consistency-ft/tree/46eed979abfd6001d2d704a81b6287316025bc24)
includes the following. The commit link preserves the historical snapshot
as the dashboard and code evolve.

- **An implemented evaluation surface.** [common.py](../common.py) defines ten
  algebraic relation checks; [the harness](../harness/run_checks.py) elicits
  member probabilities separately. The [tuple corpus](../data/tuples)
  contains 3,171 tuples. These natural-language relations come from the
  Paleka corpus; the by-construction knowledge-graph corpus in the research
  design is a proposal, not a description of these data.
- **Observed inconsistencies in emitted forecasts.** The historical
  [dashboard snapshot](../dash/index.html) reports gpt-oss-20b's mean basic
  violation as 0.159 over 300 tuples, with 0.394 on 30 Negation tuples.
  This is evidence about those elicited responses under that run's settings,
  not an intrinsic model constant or evidence of improved task performance.
  Partial coverage and elicitation/reasoning settings matter when comparing
  models; new benchmark configurations do not change that interpretation.
- **A separate implemented exploitability metric.** [metrics.py](../metrics.py)
  maximizes guaranteed profit for bounded linear stakes over the allowed
  outcomes of one check. [test_metrics.py](../test_metrics.py) exercises all
  ten check types. The historical dashboard numbers above use the algebraic
  `violation` metric, not this LP metric.
- **Preparation for training, not a trained result.** The
  [episode builder](../scripts/build_episodes.py) produces 1,619 training,
  181 validation, and 1,371 held-out episodes, with four check families held
  out. It does not train a checkpoint. The
  [KL-anchor design](../README.md#core-design-as-of-2026-08-19-brainstorm) and
  [capability-transfer hypotheses](../HYPOTHESES.md) remain proposals.

For example, quoting 0.8 for success and 0.4 for its exact complement creates
a Negation violation of 0.2. Under the LP's frictionless bounded-stake
contract, selling one unit of each claim guarantees 0.2. That illustrates
incompatible prices; actual cash extraction additionally requires tradable
contracts, settlement and execution assumptions. Our LP normalization is
also distinct from the log-market-maker arbitrage metric described in the
[Paleka literature digest](../LITERATURE.md). Thresholds do not transfer
automatically between those metrics.

## Coherence, anchoring, and collective decisions

**Coherence is a compatibility condition.** Forecasts can be internally
consistent and confidently wrong. Calibration concerns agreement with observed
frequencies; proper scores such as Brier or log loss evaluate predictions
against outcomes. In a synthetic environment we can additionally compare
against the known posterior. A gain on a relation used for repair is expected
by construction; it is not evidence of training transfer or better forecasts.

**KL anchoring is a proposed way to limit distortion.** An explicit candidate
for the [existing research design](../README.md) is

```text
loss(q) = Exploit(q, S)
          + lambda * mean_i KL(Bernoulli(q_i) || Bernoulli(p_reference_i))
```

Here `S` is the checked relation set, and `p_reference` is a frozen copy of
the previous iteration's elicited probabilities. This is KL on reported
beliefs, distinct from a trainer's usual KL on token policies. Its direction,
weights, clipping near 0/1, and reference-update schedule must be specified.
It can be used on unresolved questions, but it preserves errors in the
reference too; repeatedly moving the reference can permit cumulative drift.
The roadmap does not propose this loss, and the repository has not shown it
improves forecasting or OAA policy selection.

**Multiple agents add integration problems.** Disagreement between experts
is not itself incoherence: they may have different information or mean
different events. Record event/resolution definitions, action assumptions,
information sets, timestamps, and source identities before auditing a shared
belief vector. Related model instances can also share errors; agreement is
not independent evidence.

Even individually coherent contributors do not make every aggregation rule
coherent. A fixed-weight mixture of full joint distributions is coherent;
separately averaging conditional and marginal probabilities need not preserve
their product identities. Our tuple checks can flag some resulting defects,
but passing a finite set of local checks does not establish consistency of
an entire shared model.

This complements the [futarchy bridge](../FUTARCHY-BRIDGE.md): both programs
need usable beliefs upstream of a collective decision. Futarchy's markets
and this roadmap's proposed Nash bargaining are different mechanisms.
The [Ngo connection note](ngo-connections.md) offers related coalitional and
market analogies; it is not evidence that our finite LP is equivalent to
logical induction, that OAA requires prediction markets, or that consistency
detects every strategic liar. A fabricated belief system can be coherent.

## A bounded first experiment: does an input audit improve a policy choice?

**Proposal only; no experiment was run for this note.** Start with an offline
controlled decision problem before building an OAA interface or training a
model. Funding allocation is one of the roadmap's suggested applications.

1. **Freeze 24 synthetic cases.** Each has two funding choices and a tiny
   finite Bayesian model with binary variables and known outcome probabilities
   under each action. Generate logical relations from the model, then review
   their natural-language renderings. Publish the seeds, action semantics,
   utilities, and an exact-posterior oracle. These simulated action effects
   are specified interventions, not causal conclusions from observational
   `P(outcome | action)` alone.
2. **Elicit a small panel.** Choose three fixed model/effort configurations
   from the benchmark. Give each the same evidence in independent contexts;
   obtain marginals for success, its complement, a second outcome, conjunction,
   and disjunction for each action. Keep raw responses, failures and usage.
   Reserve some paraphrases and relation questions for evaluation. This phase
   requires a future run; archived forecasting tuples are useful only for an
   offline scoring smoke check, not ground-truth institutional decisions.
3. **Compare three offline arms on identical recorded inputs.** Use raw
   forecasts; a frozen KL-minimal repair fitted to one finite joint
   distribution using only the designated fitting questions, clipping its
   reference probabilities to `[1e-6, 1-1e-6]`; and a coherent uniform-joint
   baseline. Tune no parameters on the test cases. For a
   policy depending on multiple events, predeclare the same evaluation
   expression for every arm and report incompatible alternate expressions
   in the raw arm. Use fixed equal weights across advisers; mix repaired
   joints before deriving any conditional probabilities.
4. **Hold the values and decision rule fixed.** Initially choose the action
   with greatest predicted utility under published synthetic stakeholder
   utilities and weights. This isolates the forecasting component; it is
   not a preference-elicitation study or a Nash-bargaining implementation.
   Measure true expected-utility regret against the oracle, posterior error,
   expected Brier loss, per-check algebraic/LP violations, repair size,
   coverage and cost. Include all cases and paired uncertainty estimates.
5. **Apply a nontrivial success criterion.** Repair must reduce decision
   regret without degrading predictive scores or hiding missing responses;
   lower fitted inconsistency alone does not count. Report the uniform
   baseline to expose uninformative coherence. The 24-case pilot establishes
   feasibility at most. If promising, scale up, reveal evidence in a second
   round to test updating, and only then test a trained model on untouched
   questions/check families and the roadmap's bargaining layer.

The deliverable would be a small reproducible adapter from elicited judgments
to an audited decision input, plus a report of when it helps or harms. It
would supply evidence for one modular part of the roadmap while leaving
world-model validity, stakeholder representation and policy verification as
separate research problems.

[roadmap]: https://www.lesswrong.com/posts/pHJtLHcWvfGbsW7LR/roadmap-for-a-collaborative-prototype-of-an-open-agency
[roadmap-text]: https://www.lesswrong.com/api/post/pHJtLHcWvfGbsW7LR
