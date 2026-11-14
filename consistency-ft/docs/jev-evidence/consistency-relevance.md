# Jev × Kelvin's consistency research

2026-09-17. Grounded in successful durable-memory recall, Hub's consistency-ft history, and the current `futarchy-fi/consistency-ft` README and research notes fetched directly from GitHub. This is a relevance assessment with an empirical addendum from the main agent's live probes; it does not claim Jev uses the project's method.

**Yes—Jev is directly relevant as a commercial implementation of part of the interface Kelvin discussed, and as a cheap experimental comparator. The central question is exactly the project's question: do its probability judgments fit together into one coherent belief system?** TypeSafe's disclosed RLCD objective does not establish that. Our small controlled probes below already find coherence violations; they do not establish performance on the existing benchmark or calibration. Jev is a promising system to interrogate, not evidence that the research is already solved.

## The prior discussion this connects to

This is the **Consistency-FT belief-consistency finetuning project**, including Jessica Taylor's collaboration, rather than generic consistency of chatbot answers. The [project README](https://github.com/futarchy-fi/consistency-ft/blob/main/README.md) records four particularly relevant ideas:

- Probabilities over propositions should obey negation, implication, conjunction, conditional-chain and other logical relations.
- Train against violations/exploitability, anchored by KL to **the same model's previous iteration's beliefs**, including unresolved propositions.
- A dedicated scalar proposition head was an explicit contemplated interface; compare it with verbalized/logit/probe readouts because an isolated head can become disconnected from behavior.
- Success requires held-out relation/capability transfer and useful predictions, not just satisfying constraints used during training.

Hub's [history](https://github.com/futarchy-fi/hub/blob/main/wiki/consistency-ft.md) preserves these originating discussions. The current repository is newer than parts of that history: its corrected `descriptive-v2` benchmark has **157 reviewed groups / 274 questions**, with **148 common scored groups** in the published 23-setting comparison. Historical headline numbers from the older 300-group sweep should not be treated as the current comparison. See [current evaluation methodology](https://github.com/futarchy-fi/consistency-ft/blob/main/docs/native-evaluation.md).

## Exact mapping

| Earlier idea | What Jev offers | What it does not establish |
|---|---|---|
| A scalar proposition head | **Noul** directly emits a yes-probability; Choice emits a categorical distribution | That the number is the model's unique “true belief,” or that it agrees with other propositions/actions |
| Avoid conversational probability artifacts | No need to ask a chatbot to write a decimal and parse it | Independence from wording, option descriptions, priors or framing |
| Many related beliefs per episode | Parallel questions over common state; unusually cheap output | That parallel answers interact or are constrained to one joint distribution |
| Better probability training | RLCD explicitly aims at calibrated decisions | Use of logical constraints, Dutch-book adversaries, KL-to-previous-beliefs, or our proposed loss |
| Coherence as useful upstream input | Fast probabilities could feed the existing repair/audit layer | Accurate causal forecasts, legitimate preferences, or improved downstream decisions |
| Capability gains from coherence training | A commercially useful decision model is an interesting external comparator | Any causal evidence that coherence training improves capabilities, since training details and ablations are unavailable |

The Score documentation says levels are evaluated separately and cannot see their numbers or neighbours. That strengthens the hypothesis of independent compatibility scoring with typed assembly. It makes cross-question coherence something to **measure**, not something to infer from shared batching. Choice/Score `confidence` is derived from their distributions; it is not a separate self-audit that can validate coherence.

## Three properties that must stay separate

**Calibration:** among events assigned 0.7, roughly 70% occur. This needs repeated labeled/outcome-resolved data. RLCD publicly targets this.

**Coherence:** probabilities for related events can arise from a common probability measure under the same evidence and event definitions. Examples:

- `P(A) + P(not A) = 1`.
- If `A implies B`, then `P(A) <= P(B)`.
- `max(0, P(A)+P(B)-1) <= P(A and B) <= min(P(A),P(B))`.
- `P(A or B) = P(A)+P(B)-P(A and B)`.
- `P(A and B) = P(A) P(B given A)` and the law of total probability.

These can be checked without waiting for events to resolve. For example, independent Noul answers `P(A)=0.8` and `P(not A)=0.4` violate negation by 0.2 even though each field is a valid typed probability. In a frictionless $1-settlement claim market, selling both for $1.20 guarantees $0.20 before execution costs. This is the structure the project studies.

**Projectivity / invariance:** equivalent representations preserve the same underlying beliefs. If a joint distribution over `A,B` is returned, summing its cells should match separately elicited `P(A)` and `P(B)`; refining a category then summing it back should preserve its probability. Adding unrelated batched questions should not change an unchanged proposition's evidence. These are stronger/different claims than a normalized Choice summing to one.

Not every option-set change is a valid projectivity test. A Choice asking “which of the available actions is best?” legitimately changes when a better action is added. Hold the event space and evidence fixed, and use explicit partition/coarsening tests. Question/criterion wording can also change what event is being elicited.

Passing each finite local tuple is still weaker than global coherence. Pairwise-compatible marginals can lack a compatible global joint. Likewise, passing static total-probability checks does not establish correct updating over time or a logical-induction theorem.

## The most useful experiments

1. **Add Jev as a distinctly labeled elicitation channel on the existing corrected benchmark.** Use Noul for each proposition independently, fixed model version, matching event wording/evidence and no supplied counterpart probabilities. Report coverage, algebraic violations and LP exploitability separately; the historical algebraic score is not the Paleka log-market-maker metric or the repository's separate bounded-stake LP. Preserve a new provider/protocol identity rather than silently folding it into an old campaign. This would answer whether decision-specific training buys coherence without explicit reasoning.
2. **Compare independent Noul queries with one parallel batch.** Same propositions and shared evidence; avoid giving only the batched arm extra logical hints. Near-identical answers show computational isolation/invariance, not necessarily coherence. Better batched coherence could indicate useful cross-question interaction or merely additional context.
3. **Compare independent marginals with one explicit joint Choice.** For two Boolean events ask for four exhaustive states: `A∧B`, `A∧¬B`, `¬A∧B`, `¬A∧¬B`. Derive all marginals in code. This makes those derived quantities coherent by construction; measure whether they retain accuracy and agree with independently queried judgments. Good consistency in the joint arm is a property of the representation, not proof the model learned coherence. Joint-state enumeration scales exponentially; the advertised 255-choice cap permits all assignments of only seven unconstrained Boolean atoms in one Choice (128 states; eight need 256).
4. **Use the existing controlled downstream-decision proposal.** The [Open Agency connection](https://github.com/futarchy-fi/consistency-ft/blob/main/docs/open-agency-connection.md) already proposes 24 synthetic finite Bayesian decision cases with exact posteriors. Substitute Jev as one raw forecaster, compare raw vs KL-minimal coherent repair vs coherent-uniform baseline, and measure true decision regret, posterior error/Brier, repair size and latency. Lower fitted violation alone does not count as improvement.
5. **If promising, test held-out logical families and evidence updates.** The key scientific question remains transfer beyond constraints fitted or directly supplied. Keep event definitions, time, information set and model version fixed; separate changes in evidence from prompt sensitivity.

The first experiment is the cleanest connection to existing work. The fourth asks whether the improvement matters for software and futarchy-style decisions. These are proposed follow-ups, not completed results in this report.

## Empirical addendum: first live coherence probes

The main agent's [64-call raw probe set](probe-results.json) resolves to **`jev-1.13.0`** throughout. I independently inspected the recorded request/answer pairs for the following cases. These are small exploratory counterexamples, not results on the existing Consistency-FT benchmark, a calibration study, or a comparison against other models.

For a uniform integer from 1 to 10, let `A = number is 1 or 2` and `B = number is at most 5`. Thus A implies B, and the exact probabilities are `P(A)=P(A and B)=0.2`, `P(not A)=0.8`, and `P(B)=P(A or B)=0.5`.

| Repeat | P(A) | P(not A) | P(A and B) | Complement excess | Conjunction above P(A) |
|---|---:|---:|---:|---:|---:|
| 0 | 0.27 | 0.82 | 0.33 | 0.09 | 0.06 |
| 1 | 0.27 | 0.83 | 0.33 | 0.10 | 0.06 |
| 2 | 0.27 | 0.82 | 0.31 | 0.09 | 0.04 |

The conjunction bound fails in all three repeats. This is more than inaccurate arithmetic: those emitted probabilities cannot all describe the same events under one probability measure. Rounding to two decimal places cannot account for a 0.04–0.06 conjunction gap or 0.09–0.10 complement gap.

A second, less formally specified deployment-cause case returned `P(A)=0.38`, `P(B)=0.44`, `P(A and B)=0.21`, and `P(A or B)=0.81` in one repeat. Inclusion-exclusion would give 0.61 for the union: a 0.20 residual. Other repeats had residuals 0.16 and 0.14. Natural-language causation has more semantic ambiguity than the finite-integer example, so the latter is the cleaner evidence.

The original Noul prompts ask “what is the probability that A occurs?” even though Noul is documented for yes/no questions. Symbolic event labels and that phrasing may be brittle. **Completed controls show the violation also occurs with direct yes/no questions and fully expanded plain-English events**, so it is not confined to that initial phrasing. I inspected both [direct yes/no receipts](control-results.json) and [expanded receipts](expanded-control-results.json).

For expanded questions such as “Is the sampled integer either 1 or 2?” and “Is the sampled integer both (either 1 or 2) and (at most 5)?”, three repeats returned:

| Repeat | P(1 or 2) | P((1 or 2) and ≤5) | Conjunction excess | P(≤5) | P((1 or 2) or ≤5) |
|---|---:|---:|---:|---:|---:|
| 0 | 0.18 | 0.30 | 0.12 | 0.47 | 0.85 |
| 1 | 0.18 | 0.28 | 0.10 | 0.46 | 0.85 |
| 2 | 0.17 | 0.28 | 0.11 | 0.47 | 0.86 |

The union is equivalent to “at most 5,” but was assigned 0.38–0.39 more probability. Direct yes/no controls retaining A/B notation similarly returned P(A)=0.23 and P(A and B)=0.43–0.44. Both control variants also add a joint-Choice question, so wording and additional batch context changed together; differences between variants cannot be causally attributed to wording alone. The reproducible finding is failure of the required identities under each of these recorded elicitation conditions.

The joint Choice is revealing in a different way: the symbolic version assigned 0.06–0.11 to the impossible A-and-not-B cell; fully expanded categories reduced this to 0.01 in each repeat, with the rest of the distribution closer to the exact partition. This demonstrates representation sensitivity and a residual support error, not a measured general advantage of joint over separate elicitation.

**Joint-Choice caution:** normalizing four declared outcomes produces a categorical distribution, and deriving all marginals from it guarantees algebraic identities over those outcome labels. It does **not** guarantee respect for supplied logical support. When A implies B, `A and not B` is impossible and must have probability zero. Positive mass on that cell is a semantic/support violation even though normalization and algebraic identities hold. A robust constrained wrapper would exclude impossible assignments by construction, while recognizing that this makes coherence a wrapper property rather than evidence of learned logical consistency.

## Relevance to the training plan—and a trap to avoid

Jev is closed: the reviewed public docs provide no weight access or fine-tuning mechanism. It cannot presently be substituted as the trainable base for the planned Qwen/LoRA experiment. Its chief immediate roles are **baseline, probability elicitation service, and input to external coherent repair**.

It also cannot replace the Qwen own-base anchor collection while preserving the original experiment. In the planned objective

`Exploit(q,S) + lambda * mean KL(Bernoulli(q_i) || Bernoulli(p_previous_i))`,

`p_previous` belongs to the **same candidate** under the specified previous checkpoint/readout. Using Jev's probabilities instead turns the anchor into **cross-model distillation**—a reasonable additional arm, but a different research question. Jev's observed output rounding would also require explicit endpoint/precision treatment before any KL-based objective.

There is no public basis to equate RLCD with RLE or Kelvin's KL-anchored proposal. RLCD names a goal and broad training family; its actual rewards and data are undisclosed. The closest evidence-based statement is that TypeSafe is commercially validating the value of a probability-native decision interface while leaving our relational-coherence question open.

## Papers most relevant to this specific connection

These links/titles/abstracts were checked live. They contextualize the comparison; no evidence says TypeSafe uses them.

- **Paleka et al., [Consistency Checks for Language Model Forecasters](https://arxiv.org/abs/2412.18544)** — the project's anchor: relation tuples, instantaneous arbitrage-style evaluation, comparison with outcome-based scoring.
- **Matta et al., [Rethinking Uncertainty Evaluation in Large Language Models](https://arxiv.org/abs/2607.19367)** — direct conceptual match: calibration alone admits incoherent estimators; assesses structural coherence, faithfulness and usefulness. The project's literature notes distinguish this evaluation paper from the later RLE workshop/thesis formulation; do not cite this arXiv abstract as though it disclosed the RLE training algorithm.
- **[Logically Consistent Language Models via Neuro-Symbolic Integration](https://arxiv.org/abs/2409.13724)** — LoCo-LMs, the project's closer precedent for training with logical relations and testing transfer.
- **Xu et al., [A Semantic Loss Function for Deep Learning with Symbolic Knowledge](https://arxiv.org/abs/1711.11157)** — machinery for constraints on neural output vectors, directly relevant if implementing a constrained probability head.
- **Garrabrant et al., [Logical Induction](https://arxiv.org/abs/1609.03543)** — theoretical lineage behind the Dutch-book/efficient-trader perspective. Jev has disclosed no mechanism or guarantee meeting its criterion; a static finite LP is not logical induction.

Public Jev sources: [launch](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [AI primer / RLCD](https://docs.typesafe.ai/introduction/machine-learning-primer), [Score level isolation](https://docs.typesafe.ai/primitives/score), [confidence derivation](https://docs.typesafe.ai/confidence). General architecture hypotheses and additional papers remain in [research.md](research.md).
