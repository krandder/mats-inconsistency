# Section 1: Adversarial audit of `ast_leak_auditor.py v2` and proposed S11–S14 gates

**Verdict: the supplied findings invalidate Wave-1’s affected comparative performance claims. Wave-2 is a repair plan, not yet evidence of algorithmic superiority.** Renaming forbidden variables would defeat much of v2; runtime counterfactual tests would substantially improve assurance, but none of the proposed gates constitutes a general proof of leakage absence.

This review uses only the supplied material. The reported code findings, violation counts, and numerical verifications are premises of this review, not independently reproduced observations.

Three distinctions must survive throughout the paper:

- **Training a real neural network does not establish that its predictions caused the reported performance.**
- **Passing a leakage checker does not establish that an experiment measures the claimed capability.**
- **Removing known shortcuts does not establish fair resource allocation, strong baselines, or generalization.**

## 1.1 What S1–S10 actually establish

The statement “0 false negatives on Wave-1” is too strong. The defensible statement is:

> v2 detects all previously identified Wave-1 shortcut categories represented in the supplied audit cases.

A false-negative rate requires a denominator of actual violations, including violations not previously discovered. Updating a detector against known failures and then measuring detection on those same failures is a regression test, not an independent sensitivity estimate. Likewise, seven caught canaries establish coverage of seven constructions.

“Core modules were 100% clean” should become “no violations were identified by the stated checks.” A library can expose a perfectly legitimate API that a runner uses dishonestly.

Several rules also risk false positives. Budget-dependent control flow is necessary for implementing budgeted search; training and scoring legitimately access labels; benchmark-specific normalization can be valid. The important property is the **information flow and experimental role**, not whether a particular spelling appears.

### Identifier aliasing and indirect writes

S7–S10 are principally lexical restrictions. Equivalent prohibited behavior can be expressed without their vocabulary:

```python
answer_channel = compute_solution(instance)
features[..., feature_index] = answer_channel
```

Or without a direct assignment into any named feature tensor:

```python
features = torch.cat([public_features, encoded_answer], dim=-1)
```

Further bypasses include:

- Passing a tensor view into a utility that mutates it.
- Using `copy_`, `scatter_`, indexed setters, or array constructors.
- Loading a reference policy as an unnamed checkpoint buffer.
- Deriving a gain from a configuration field rather than a literal constant.
- Moving the operation into an imported package, compiled extension, or serialized artifact.

A static scanner must follow definitions, aliases, mutation, imports, and configuration to detect these reliably. Even sophisticated whole-program analysis cannot generally establish arbitrary Python program noninterference.

### Dictionary dispatch and polymorphism

An asymmetric helper requires no `if paradigm == ...`:

```python
finishers = [identity, identity, identity, privileged_solver]
candidate = finishers[method_index](candidate)
```

Equivalent asymmetry can live in:

- Subclass overrides.
- Dependency injection.
- Callback registries.
- Configuration tables.
- Different default constructor arguments.
- Method-specific candidate submission rates.
- A shared helper with different recursion limits supplied upstream.

The last two matter especially: **identical helper code does not imply identical helper access or effective solver strength.**

### Exempt-function laundering

Exemptions are the largest explicitly advertised escape route.

An exempt dataset builder can attach answer-derived features to an observation. An exempt trainer can place a reference table in a model buffer. An exempt calibrator can fit on evaluation labels. The evaluation function can then access only apparently permitted tensors.

Supervised learning necessarily lets training labels affect learned weights. The distinction must therefore be:

\[
\theta=\operatorname{Train}(D_{\mathrm{train}},D_{\mathrm{validation}})
\]

with a prespecified validation protocol, while

\[
\hat y_i=\operatorname{Infer}(\theta,x_i)
\]

has no access to evaluation labels, evaluation-derived calibration, or answer-dependent preprocessing.

A broad exemption by function name does not encode that distinction. Exemptions should be narrow, role-specific capabilities with restricted outputs, not blanket immunity.

### Structural and aggregate leakage

S10 misses answer information encoded in structure:

- Sorting vertices by shortest-path distance.
- Placing the correct candidate first.
- Encoding a solution in padding length, graph size, sample order, or an identifier.
- Selecting only examples that satisfy an answer-dependent difficulty filter without documenting the resulting task.
- Normalizing a batch using target-derived statistics.
- Sharing train/test examples through isomorphic graphs, renamed SAT formulas, or rotated puzzles.

Ordinary feature-only normalization across train and test is also a protocol issue: it may be permitted transductive learning, or an undeclared use of test-distribution information. It should not automatically be called label leakage. The protocol must state which setting is intended.

## 1.2 S11: Target-channel noninterference and feature provenance

Let:

- \(G\) denote all legitimate public instance information.
- \(T\) denote protected labels, hints, solutions, and reference policies.
- \(r\) denote generator randomness.
- \(s\) denote generator state.
- \(X=f(G,T;r,s)\) denote the complete model-visible input.

The desired property is:

\[
\forall G,T,T',r,s:\quad
f(G,T;r,s)=f(G,T';r,s).
\]

This is an information-flow requirement. It is not established merely by observing equality for several random permutations.

### Concrete gate

For each audited instance, execute feature generation in isolated, freshly initialized processes with identical \(G,r,s\), changing the protected channels through:

1. Independent label permutations.
2. Constant replacements.
3. Adversarially chosen replacements.
4. Access-denying sentinel objects.
5. Removal of protected fields entirely, where the API permits it.

Compare the complete serialized model-visible payload:

- Numeric tensors.
- Masks and legal-action lists.
- Ordering, identifiers, lengths, and padding.
- Graph connectivity and indexing.
- Auxiliary metadata used by inference.

Require identical shape, dtype, metadata, and bytes.

For finite numeric tensors, bitwise equality implies

\[
\|X-X'\|_\infty=0.
\]

The converse is not literally sufficient for byte equality: signed zeros and representation details can differ. Use byte comparison as the primary check, and reject unexpected NaNs.

The generator should run on a deterministic path for this gate. Comparing independently randomized augmentations would test the wrong property.

### Why the proposed test does not prove zero leakage

Suppose the feature generator ignores the supplied target argument and recomputes the solution directly from \(G\). Then:

\[
f(G,T)=f(G,T')
\]

even though an exact solver is producing an answer feature.

Thus S11 proves neither “no symbolic solving” nor “no answer-derived computation” without additional controls. Moreover, the desired target is itself a function of \(G\); demanding unconditional independence between \(X\) and the answer would forbid useful prediction.

The correct distinction is:

> Model inputs may contain declared public information and approved preprocessing of it; they may not receive protected answer channels or undeclared solution computation.

For the proposed CLRS features, an independently implemented allowlist encoder is practical. Verify that the complete input equals that encoder’s output. This is stronger than merely checking whether a poisoned label tensor was read.

### Static component

Use an explicit data boundary:

```text
PublicInstance -> ApprovedEncoder -> ModelInput
PrivateLabels  -> TrainingLoss or SealedScorer
```

Track protected-data flows into returned observations, globals, buffers, closures, and serialized outputs. Scan the reachable dependency set and configuration, not just runner files.

**Acceptance:** all observed counterfactual checks pass; encoder provenance and data boundaries match the declared protocol.

**Permitted claim:** “Passed the specified target-channel noninterference tests.”

**Impermissible claim:** “Proved zero leakage.”

## 1.3 S12: Neural causal participation and executed computation

The requested gate contains two invalid universal requirements.

### Forward-call count is not a universal compute measure

The condition

\[
\#\operatorname{forward\ calls}\geq B
\]

is appropriate only if \(B\) is explicitly defined as sequential neural evaluations per instance.

It is not appropriate across all listed paradigms:

- A batch of \(B\) candidates may require one Python forward call.
- A recurrent model may execute \(B\) internal iterations in one call.
- Compilation can obscure module-level hooks.
- Legitimate caching can reuse an unchanged infoset prediction.
- An autoregressive candidate requires multiple token predictions.
- A solved instance may terminate early.

Removing the Wave-1 cutoff is necessary because it changed the policy into a different mechanism. Requiring redundant forward passes on every repeat encounter is not necessary.

Define and record a resource vector:

\[
C =
(F_{\mathrm{neural}},
N_{\mathrm{token}},
N_{\mathrm{controller}},
N_{\mathrm{simulator}},
N_{\mathrm{helper}},
t_{\mathrm{GPU}},
t_{\mathrm{CPU}},
M_{\mathrm{peak}}).
\]

Use a primary compute constraint plus secondary resource accounting. An identical nominal \(B\) is not compute parity.

### Arbitrarily small perturbations need not change actions

For a deterministic policy,

\[
a_\theta(x)=\arg\max_a \ell_\theta(x,a),
\]

a positive logit margin implies that sufficiently small perturbations leave the action unchanged. Stable outputs can indicate confidence or successful search.

Likewise, a correct complete solution can remain unchanged under large perturbations if search reliably finds it. Requiring every instance’s trajectory and output to change would punish robustness.

### Concrete gate

For every method and requested budget:

1. Trace actual neural evaluations, internal iterations, expansions, token generation, simulator calls, and helper operations.
2. Account for batching, caching, and termination explicitly.
3. Record a dependency chain from neural outputs to selected nodes, proposals, evaluations, or final predictions.
4. Repeat with fixed random streams and:
   - Several scaled weight perturbations.
   - A reinitialized checkpoint.
   - Uniform or shuffled neural outputs.
   - Replayed, deliberately altered neural outputs at the controller boundary.
5. Compare continuous predictions, trajectories, and aggregate performance.

For example:

\[
D_{\varepsilon,B}
=
\frac{1}{n}\sum_i
\operatorname{TV}
\left(
p_{\theta,B}(\cdot\mid x_i),
p_{\theta+\varepsilon\xi,B}(\cdot\mid x_i)
\right).
\]

Measure this across several perturbation scales and independently chosen directions. Establish a numerical reproducibility baseline before interpreting small differences.

The continuous prediction can legitimately be a point mass; in that case, measure trajectory changes and aggregate behavioral sensitivity rather than demanding differentiability.

**A dummy neural call followed by a formula can pass call-count checks and weight-sensitive dummy logging.** Therefore neural-output interventions must affect the computation that produces the reported prediction or metric.

Finally, recompute metrics independently from saved predictions and sealed labels. A runner should never be trusted to supply its own headline score.

**Acceptance:** requested computation is actually executed or explicitly terminated; neural outputs causally participate; saved predictions reproduce reported metrics.

**Not required:** monotonic performance improvement with budget. Genuine search can get worse.

## 1.4 S13: Shared evaluator and resource-contract equivalence

Literal call-graph isomorphism is the wrong target. Diffusion, autoregression, recurrent reasoning, and tree search have legitimately different call graphs.

Require isomorphism only for the **external evaluation protocol**, not the internal algorithm.

A suitable interface is:

```text
evaluate(policy_adapter, frozen_instances, resource_contract, helper_capability)
```

The evaluator owns:

- Instance sampling and order.
- Player assignment and random seeds.
- Observable information.
- Legal-action validation.
- Resource limits.
- Helper access.
- Termination.
- Scoring.

Adapters return predictions, actions, or proposed partial states. They cannot access scoring labels, override the evaluator, or select different benchmark instances.

### Runtime checks

- Record identical scenario identifiers and budgets across methods.
- Use the same helper implementation and configuration digest.
- Enforce a shared total helper allowance, not merely the same helper function.
- Reject hidden network access and undeclared files during inference.
- Compare evaluator event schemas and permitted transitions.
- Run adapter-label swaps: relabeling a method must not alter the harness behavior.
- Record timeouts, illegal outputs, and early termination as outcomes.

Executed work need not be identical when one method solves earlier. The same **limits and charging rules** must apply.

Exact prediction equality after changing only a display label is a useful counterfactual test. Exact equality across genuinely different algorithms is not.

## 1.5 S14: Autograd isolation and conditional information noninterference

The condition `z_m.grad_fn is None` establishes little by itself:

- A detached tensor can share storage.
- A leaf tensor can receive gradients.
- Modules can share parameter storage or optimizer state.
- A global variable can carry labels.
- A scorer can mutate a buffer.
- File writes, caches, shared random generators, and callbacks can transmit information.

Define the local information history:

\[
\mathcal F_{m,t}
=
\sigma(
D_{m,\leq t}^{\mathrm{allowed}},
\theta_{m,\leq t},
O_{m,\leq t},
R_{m,\leq t},
B_{<t}
),
\]

where \(O_m\) is local optimizer state, \(R_m\) is local randomness, and \(B_{<t}\) contains earlier authorized broadcasts.

For current broadcast \(B_t\), require:

\[
\Delta\theta_{m,t}
=
U_m(\mathcal F_{m,t},B_t).
\]

Equivalently, for hidden nonlocal information \(H_{-m,t}\),

\[
I(
\Delta\theta_{m,t};
H_{-m,t}
\mid
\mathcal F_{m,t},B_t
)=0.
\]

This is a formal target, not something finite traces can prove.

### Concrete counterfactual gate

Snapshot a module’s local parameters, optimizer state, data, randomness, and authorized broadcast. Then:

1. Change other modules’ observations, labels, state, and parameters.
2. Hold the authorized broadcast fixed.
3. Replay the local update in a deterministic isolated process.
4. Require identical parameter updates and local state transitions.

Holding the broadcast fixed is essential: other modules are legitimately allowed to affect the market-cleared message.

Also test timing:

- Before outcome disclosure, changing the outcome must not change actions or pre-clearing messages.
- After disclosure, outcome dependence is allowed only through the channels authorized by the protocol.

Use process isolation, serialized message schemas, parameter-storage audits, optimizer-ownership checks, and timestamped messages. Autograd tracing should confirm no unauthorized graph crosses the module boundary.

A “scalar” channel is not automatically a low-information channel. An unrestricted real-valued number can encode many bits. Specify precision, range, message frequency, and computation semantics.

### Required verifier behavior

`verify_paper_numbers.py` should fail closed on missing evidence, producing separate statuses:

- `PASS`
- `FAIL`
- `NOT_RUN`
- `INAPPLICABLE`, with justification

The artifact must bind the checks to exact code, checkpoint, dataset, configuration, dependency versions, seeds, and prediction files. Running gates against one checkpoint and reporting another is otherwise an easy bypass.

# Section 2: Remaining loopholes in R22A, R22B, and R23

## 2.1 Imperfect-information games: correct networks can still evaluate the wrong game

### Infoset consistency

For player \(i\), a valid policy must satisfy:

\[
h\sim_i h'
\quad\Longrightarrow\quad
\pi_i(\cdot\mid h)=\pi_i(\cdot\mid h'),
\]

where \(h\sim_i h'\) means the histories belong to the same information set.

Passing an OpenSpiel `state` object to a policy adapter is common. It is also dangerous: the adapter can inspect hidden cards or pieces even when its neural observation is clean.

Audit the entire action-selection path:

- Neural inputs.
- Search-state construction.
- Rollout initialization.
- Legal-action masks.
- Terminal tests.
- Cache keys.
- Heuristic evaluations.

Use paired hidden-state interventions within the same infoset. With coupled randomness, the policy distribution must be unchanged.

`information_state_tensor` is often appropriate, but its semantics are game-specific. An `observation_tensor` may omit remembered information and create an unfairly weak baseline. Neither name guarantees adequacy.

For Stratego, searching the true hidden deployment is cheating. A simulator that allows hypothetical actions is legitimate only when hidden-state hypotheses come from the player’s information and the protocol permits the computation. Naive determinization also suffers from strategy fusion even without leakage.

### Legal-action masking

The mask must encode legal actions as known to the acting player, not actions filtered by inaccessible hidden facts. Failed probes or illegal-attempt semantics in hidden-information games require particular care.

Use one centrally audited mask implementation for all methods.

### What `nash_conv` measures

For strategy profile \(\pi\),

\[
\operatorname{NashConv}(\pi)
=
\sum_i
\left[
\max_{\pi_i'}
u_i(\pi_i',\pi_{-i})
-
u_i(\pi)
\right].
\]

Its computation assumes a well-defined strategy profile. The evaluator must not encounter a policy that changes because of call order, previous unrelated evaluations, or mutable global search state.

A stochastic search procedure needs a clearly defined induced behavioral policy. Fresh noisy probability estimates on repeated calls can make the evaluation internally inconsistent.

Further pitfalls:

- Payoff scaling changes the numerical value.
- In two-player zero-sum settings, commonly reported exploitability is half NashConv.
- Exact tree traversal may be feasible for Kuhn and some Leduc configurations but prohibitive for Phantom TTT or Dark Hex.
- A restricted or approximate best response gives a lower bound on exploitability, not an upper bound certifying near-equilibrium play.
- Cross-play win rate is not a substitute for exploitability.

The plan must specify exact game parameters. “Dark Hex” without board size is not a reproducible benchmark.

### Average policy versus last iterate

CFR convergence results principally concern properly weighted average strategies. Neural approximation introduces additional errors; “Deep CFR” is not a guarantee of convergence at a particular finite budget.

For behavioral strategies, the relevant average generally involves own reach probabilities:

\[
\bar\sigma_i(I,a)
=
\frac{
\sum_t w_t\,
\pi_i^{\sigma^t}(I)\,
\sigma_i^t(I,a)
}{
\sum_t w_t\,
\pi_i^{\sigma^t}(I)
}.
\]

A plain average of network weights is not this strategy.

PPO, GRPO, and AlphaZero-style self-play can cycle or overfit opponents. Their last iterates can therefore be exploitable, but the claim that they are *necessarily highly exploitable* is unjustified. They can learn near-equilibrium behavior under suitable training.

Report both last-iterate and appropriate average-policy results where supported. Choose checkpoints using validation opponents or validation exploitability, never the final test suite.

### Where should GLC-AOM win?

There is no established basis here for predicting universal GLC-AOM superiority.

| Setting | Defensible expectation |
|---|---|
| Kuhn | A strong tabular CFR control should approach equilibrium cheaply. Deep CFR and R-NaD are important neural comparators. Any claimed decisive GLC advantage requires scrutiny of convergence and tuning. |
| Leduc | Regret-based and regularized imperfect-information methods have a strong structural fit. GLC may compete; superiority is unestablished. |
| Phantom TTT | Belief maintenance and information-sensitive planning could favor GLC, but strategy fusion and imperfect best-response evaluation are serious risks. |
| Dark Hex | Similar opportunity, with stronger scaling and evaluation challenges. Search alone does not establish an advantage. |
| Stratego 4×4 | Small tactical search may dominate; mature baselines could tie or win. |
| Stratego 6×6 | Longer planning and hidden-piece inference create a plausible opportunity, alongside greater model error. |
| Stratego 8×8 OOD | Representation and training-distribution shifts may overwhelm all methods. |
| LOGO | Transfer depends on genuine shared structure and target-family exclusion. Any-order consistency alone does not imply transfer. |

Beating **Deep CFR** on Kuhn is not inherently suspicious: an approximate neural implementation can be inefficient or poorly tuned. Beating a converged, appropriately evaluated CFR reference by a large margin is much harder to interpret because NashConv already has a zero floor.

Wins on Phantom TTT, Dark Hex, and Stratego would be more informative about demanding capabilities. They are plausible opportunities, not predictions licensed by architecture alone.

For Stratego, pair games across colors and deployments; report win, draw, and loss separately; cluster uncertainty by deployment and training seed. Hundreds of matches from one checkpoint do not replace independent training runs.

## 2.2 Public reasoning: symmetry does not remove task-definition problems

### Difficulty must be calibrated without selecting for the winner

For models with 1M–15M parameters, “extreme” datasets can put every method at zero exact solves. That tests the current training regime’s inadequacy but gives little resolution among methods.

Construct difficulty strata before test evaluation. Use independent baseline pilots or source-provided difficulty measures, then freeze the selection rule. Do not tune difficulty until GLC separates.

Report full distributions and failures, not only an aggregate pass gate.

### Task-specific verification

**Sudoku**

- Preserve all clues.
- Require every row, column, and box to satisfy the constraints.
- Accept any valid completion unless uniqueness is part of the benchmark definition.
- Split by source/template or sufficiently strong equivalence classes.
- Exclude given cells from prediction-accuracy metrics.

**SATLIB 3-SAT**

- State which SATLIB families and SAT/UNSAT mixtures are used.
- Verify assignments against the original clauses.
- Treat “no satisfying assignment found” as unknown, not an UNSAT certificate.
- Verify UNSAT certificates if UNSAT solving is claimed.
- Check overlap under variable renaming and clause permutation.

**Sokoban**

- Replay the complete action sequence in the authoritative simulator.
- Validate pushes, walls, goal occupancy, and terminal state.
- Do not accept a proposed solved board without a legal path from the initial state.
- Split levels by generator/template where applicable.

**Countdown**

- Parse with a restricted expression grammar.
- Enforce the supplied-number multiset.
- Specify allowed arithmetic and intermediate values.
- Evaluate exactly using integers or rationals.
- Reject expressions that reach the target by violating the number-use rules.

### Compute accounting across paradigms

An AR sequence of length \(L\), one MDLM denoising pass, one recurrent iteration, and one tree expansion can differ greatly in FLOPs and information produced.

Report performance as a function of:

- Training compute and data exposure.
- Inference compute.
- Latency.
- Number of candidates and environment interactions.
- Verifier or propagator calls.

Parameter count and nominal \(B\) are secondary descriptors.

BoN needs multiple complete candidates. A tree method may submit many partial candidates. If helper access scales with candidate count, Track B can silently restore the original asymmetry.

### “One-step propagator” needs a formal definition

One SAT unit assignment, closure under all unit clauses, and one scan over a formula are different operations.

The same ambiguity appears in Sudoku. Sokoban and Countdown have no canonical counterpart to SAT unit propagation.

Define a task-specific primitive, its allowed inputs, its output, and whether it may:

- Fill values.
- Reject states.
- Rank candidates.
- Repeat until convergence.

An unlimited sequence of “one-step” helper calls can implement a powerful solver. Count the total primitive operations per instance.

## 2.3 CLRS: removing answer features is necessary but insufficient

### The proposed encoder does not establish CLRS-30 coverage

Source indicators and weighted graph features describe a shortest-path-style task. CLRS-30 contains diverse algorithmic problems with different inputs and outputs.

If the runner evaluates only shortest paths, call it a shortest-path benchmark. Claiming CLRS-30 requires:

- The declared algorithm set.
- Appropriate input schemas.
- Standard or explicitly documented splits.
- Algorithm-specific output semantics and metrics.
- Per-algorithm results.

Training hints may be legitimate supervision. Test-time ground-truth hints are not.

### Degree normalization requires a corrected statement

For dense Erdős–Rényi graphs \(G(N,p)\) with fixed \(p\),

\[
\mathbb E[\deg(v)/N]
=
p(1-1/N),
\]

so the mean is approximately stable as \(N\) grows. Its variance contracts.

For sparse graphs with \(p=c/N\),

\[
\mathbb E[\deg(v)/N]\approx c/N,
\]

which changes substantially between \(N=16\) and \(N=64\).

Thus `deg/N` is not universally defective. Its suitability depends on the graph family.

Other shifts remain:

- Path lengths.
- Disconnectedness.
- Edge-weight extrema.
- Neighborhood aggregation magnitudes.
- Positional encodings.
- Required recurrent depth.
- Number of possible predecessor choices.

A size increase is not a single controlled intervention unless graph generation is specified.

### Predecessor ties

If several predecessors yield the same shortest distance, comparing against one arbitrary pointer can mark a correct solution wrong.

For reachable \(v\neq s\), verify:

\[
(u,v)\in E,
\qquad
d(v)=d(u)+w(u,v),
\]

and that predecessor chains terminate correctly at the source. Zero-weight edges require explicit cycle handling.

Alternatively, impose and document a canonical tie-breaking rule shared by target generation and scoring. Do not let incidental vertex order become the hidden task.

### Mask imbalance and misleading F1

If positives become rare at \(N=64\), a trivial negative predictor can have high accuracy. Micro-F1 can also conceal imbalance; for ordinary single-label classification across all classes, micro-F1 equals accuracy.

Report:

- Positive-class precision, recall, and F1.
- Macro metrics.
- Exact graph-level correctness.
- Predecessor accuracy on eligible vertices.
- Performance by graph size, density, and reachability.

### Genuine execution does not establish a scaling law

Three sizes and four budgets support a small empirical response surface. They do not automatically identify universal exponents.

A proposed model such as

\[
E(N,B)=E_\infty+cN^{-\alpha_N}B^{-\alpha_B}
\]

has strong assumptions. Error floors, optimization shortfalls, and \(N\)-by-\(B\) interactions can make the exponents unstable.

Fit competing models, report residuals and uncertainty, and mark an exponent unidentifiable when appropriate. Under fixed training compute, larger models may perform worse because they are undertrained.

If the target distribution is unknown, empirical label NLL is not automatically KL divergence. Naming a quantity `base_kl` does not establish that its reference distribution is available or correct.

## 2.4 R23: real tensors do not guarantee meaningful decentralization

### First define the actual prediction task

For each environment, identify:

- Which positions are queried.
- Which labels are withheld.
- Whether the task is classification, action imitation, belief prediction, or solution generation.
- Whether multiple answers are valid.
- Which local information each module receives.

On Sudoku, predicting observed givens is trivial. On Stratego, predicting hidden piece identities is different from selecting winning actions. A unique `target_tokens` array can incorrectly penalize valid alternative Sudoku or SAT solutions.

“Real EnvBatch” describes provenance, not scientific adequacy.

### Categorical LMSR

For query position \(j\) with \(K\) outcomes, a categorical LMSR uses

\[
C_j(q_j)
=
b_j\log\sum_{k=1}^K e^{q_{jk}/b_j},
\]

with prices

\[
p_{jk}
=
\frac{e^{q_{jk}/b_j}}
{\sum_\ell e^{q_{j\ell}/b_j}}.
\]

Separate markets across \(L\) positions cost \(O(LK)\), avoiding a joint market over \(K^L\) sequences.

But this trades away joint expressiveness. Consistent marginals do not guarantee a valid Sudoku grid, SAT assignment, or legal action sequence. Specify whether the market handles:

- Unconditional marginals.
- Conditional predictions given a partial assignment.
- Structured candidate outcomes.

For a uniformly initialized categorical LMSR, the standard worst-case loss bound is \(b_j\log K\); independent positions add their bounds. Repeated rounds, fees, budgets, and settlement rules must be included in the actual mechanism analysis.

### Kyle clearing is not automatically inherited from binary parity

A categorical batch mechanism needs a vector-valued price-impact rule, treatment of the simplex’s redundant direction, and a stable clearing procedure.

For example, a linearized rule

\[
p=p_0+\Lambda q
\]

must respect probability constraints and appropriate properties of \(\Lambda\) on the relevant subspace. Combining Kyle-style price impact with an LMSR cost function requires a specified mechanism; naming both does not establish equilibrium or truthfulness.

### Sybil invariance is narrower than collusion resistance

A batch mechanism can be invariant when one aggregate order is split across identities. That does not mean it is immune to:

- Colluders changing aggregate order flow.
- Separate identity-based budgets.
- Subsidies or rebates.
- Coordinated information withholding.
- Timing manipulation.
- Coalition transfers.

LMSR final prices depend on final inventory. Individual profits and allocation under sequential trading can still depend on timing and counterparties.

The numerical invariance claims must state exactly what remains fixed under the tested transformation.

### Scalar REINFORCE versus market targets

For a factorized policy,

\[
\nabla_\theta\mathbb E[R]
=
\mathbb E\left[
(R-b)
\sum_{j=1}^L
\nabla_\theta\log p_\theta(y_j\mid h_j)
\right].
\]

There is no universal variance law proportional to \(KL\). Variance depends on the reward, baseline, policy, parameterization, and correlations.

Nevertheless, sparse exact-solution rewards can become devastatingly uninformative. Under an illustrative uniform independent policy,

\[
P(\text{all }L\text{ positions correct})=K^{-L}.
\]

As this event becomes rare, useful positive-reward samples disappear. An absolute variance may even shrink while relative estimation error and sample complexity explode.

Dense token rewards, strong baselines, or causal credit assignment can improve M3a substantially. They must be disclosed because they change the information supplied by the scalar channel.

A market-cleared target can support a local loss such as

\[
\mathcal L_m
=
-\sum_{j,k}
\tilde p_{jk}\log p_{m,jk},
\]

which usually offers more informative credit than a single sequence payoff. Its usefulness depends on the target’s accuracy and calibration.

### A detached cotangent can reproduce backpropagation

If the market broadcasts the exact global derivative with respect to a module output,

\[
c_m=\frac{\partial\mathcal L}{\partial z_m},
\]

then the local update

\[
\nabla_{\theta_m}\mathcal L
=
\left(\frac{\partial z_m}{\partial\theta_m}\right)^\top c_m
\]

can reproduce monolithic backpropagation despite detachment at the process boundary.

That would demonstrate a distributed implementation of gradient computation. It would not establish learning from a weak scalar market signal.

Report message dimension, precision, semantics, and bandwidth. “No cross-module autograd edge” is not equivalent to “no global gradient information.”

A reasonable default expectation is that M4 is strongest when objectives and resources are matched; M3 may approach it; M3a is more vulnerable to sparse credit. This is a prior, not a theorem.

### VoI manipulation and wash trading

An uncalibrated module can manufacture apparent value by first worsening the aggregate prediction and then “repairing” it, or by coordinating offsetting orders with another module.

A useful external predictive score is

\[
\Delta\ell
=
\ell(p_{\mathrm{before}},y)
-
\ell(p_{\mathrm{after}},y),
\]

evaluated on held-out outcomes. Even this is not automatically decision value.

If a trade changes the chosen action, the resulting outcome is endogenous. Decision VoI needs a prespecified utility and an appropriate counterfactual design: randomized actions, supported off-policy estimation, or another justified causal method.

Use frozen evaluation streams, leave-one-module-out comparisons, coalition tests, and net coalition profit after all fees and subsidies. Test false-information injections as well as identity splitting.

# Section 3: What the forensic correction does—and does not—contribute

## 3.1 Wave-1 to Wave-2: accountability is valuable; invalid results remain invalid

The known Wave-1 failures are serious:

- R18 substituted privileged reference policies and arm-dependent heuristics for the claimed neural comparison.
- R19 compared unequal algorithmic assistance.
- R20-E2 measured a synthetic task while presenting real-environment provenance.
- R21-E1 supplied answer information through inputs.
- R21-E2 reported an imposed formula as executed search scaling.

These are not minor implementation imperfections. They alter the scientific estimand.

The supplied evidence does not establish the developers’ intent. Reward pressure and pass/fail incentives make benchmark-targeting behavior plausible, but a forensic report should distinguish observed code behavior from psychological explanation.

The correction can strengthen the **credibility of the revised process**, relative to hiding the failures. It does not retroactively strengthen the contaminated results.

A benchmark-integrity contribution would require more than a narrative:

1. Preserve affected versions and clearly withdraw affected claims.
2. Publish minimal reproductions of each failure.
3. Release historical and independently designed detector cases.
4. Measure false positives and unseen bypasses.
5. Freeze Wave-2 evaluation before final outcomes are inspected.
6. Publish all completed runs, failures, exclusions, and compute costs.
7. Recompute conclusions from corrected predictions.

The Wave-1/Wave-2 performance difference cannot itself estimate “the effect of removing cheating” if training, datasets, architecture, and budgets also change. A controlled forensic ablation would hold those factors fixed.

R20-E1/E3 may remain valid within their tested assumptions. Mathematical or numerical mechanism checks do not establish real-task learning efficacy.

## 3.2 Track A and Track B: a useful factorial design, not a bulletproof ablation

Track A measures performance of the declared neural system without external solution-completion assistance.

It does **not separately isolate**:

- Representational consistency.
- Any-order marginal calibration.
- Search-controller quality.
- Training-objective quality.

All can affect the same solve rate. Direct calibration tests, controlled component ablations, and matched training are still needed.

Track B measures how the full system uses a specified shallow observable helper. This includes representation compatibility, candidate quality, and helper scheduling—not merely “belief quality.”

Let \(Y_{m,h}\) denote performance for method \(m\) and helper condition \(h\in\{0,1\}\). The within-method helper effect is

\[
\Delta_m=Y_{m,1}-Y_{m,0}.
\]

A useful interaction contrast is

\[
I_{m,\mathrm{GLC}}
=
(Y_{\mathrm{GLC},1}-Y_{\mathrm{GLC},0})
-
(Y_{m,1}-Y_{m,0}).
\]

This asks whether GLC benefits more from the same helper.

Wave-1 R19 effectively compared

\[
Y_{\mathrm{GLC},1}-Y_{\mathrm{baseline},0},
\]

which confounds architecture and helper access.

The \(2\times7\) design removes that particular confound only if:

- Instance sets and training protocols are comparable.
- Helper strength, limits, and costs are identical.
- Helper access cannot be amplified by method-specific scheduling.
- Verification is common.
- Hyperparameter tuning is controlled.
- Failed and timed-out runs remain in the analysis.

Use the same checkpoint across tracks to estimate an inference-only helper effect. Separately retrained Track-B models answer a different question and should be reported separately.

A larger Track-B gain does not prove superior intrinsic reasoning. It may indicate that a method produces better partial states—or simply that its representation aligns more naturally with the helper.

# Section 4: `[PRIOR]` forecasts and methodological rubric

## 4.1 Status and assumptions of these numbers

**Every interval below is a subjective 80% predictive interval, not an observed result, confidence interval, or empirically calibrated forecast.** The supplied material lacks training schedules, dataset distributions, game sizes, helper semantics, and baseline tuning details. Numerical precision beyond broad ranges would be fictitious.

These are a **draft preregistration**. They become a preregistration only if frozen before the relevant results are examined.

The intervals concern aggregate test performance of a prespecified training-and-evaluation protocol, including training-seed variability. They are marginal intervals, not a joint 80% guarantee across all table cells. Derived differences and rankings require separate forecasts.

Working assumptions:

- Models receive meaningful training with comparable tuning opportunities.
- Training and inference resources are disclosed and reasonably matched.
- No protected evaluation information reaches inference.
- Reasoning forecasts use the largest proposed budget, \(B=64\), with a meaningful compute normalization.
- Public-reasoning models are near the upper end of the proposed small-model range.
- Track B uses a fixed shallow primitive and a common cap on total helper operations.
- Evaluation sets contain nontrivial but not uniformly impossible instances.

If these assumptions fail, the forecast should be marked inapplicable rather than quietly reinterpreted.

## 4.2 R22A: OpenSpiel NashConv

For comparability, rescale each player’s terminal utility affinely into \([0,1]\) using the game’s declared utility range. The table forecasts **NashConv in those normalized units**, with theoretical range \([0,2]\) for two players.

For an original common utility range \(R\),

\[
\operatorname{NashConv}_{\mathrm{original}}
=
R\operatorname{NashConv}_{[0,1]}.
\]

Do not compare these numbers directly with unnormalized OpenSpiel output.

| Method | Kuhn `[PRIOR]` | Leduc `[PRIOR]` | Phantom TTT `[PRIOR]` | Dark Hex `[PRIOR]` |
|---|---:|---:|---:|---:|
| Deep_CFR | 0.001–0.12 | 0.02–0.40 | 0.12–1.20 | 0.15–1.40 |
| R_NaD | 0.002–0.18 | 0.02–0.45 | 0.10–1.10 | 0.12–1.30 |
| AlphaZero_PUCT | 0.02–0.40 | 0.10–0.80 | 0.15–1.40 | 0.15–1.50 |
| PPO_GAE | 0.02–0.45 | 0.10–0.90 | 0.20–1.50 | 0.20–1.60 |
| GLC_AOM_v2 | 0.005–0.30 | 0.05–0.65 | 0.10–1.20 | 0.10–1.40 |

The Phantom TTT and Dark Hex intervals are conditional on frozen, small configurations. They concern the latent exact quantity. If exact evaluation is infeasible, report an approximate best-response lower bound under its own name; do not substitute it into this table as exact NashConv.

These priors weakly favor specialized imperfect-information methods on small poker. They do not predict a decisive GLC win on the larger games.

## 4.3 R22A: Stratego and LOGO

Define literal win rate as \(P(\mathrm{win})\), reporting draws separately. For transfer, use score

\[
s=P(\mathrm{win})+\tfrac12P(\mathrm{draw})
\]

because a draw-heavy environment otherwise makes ratios unstable and hard to interpret.

| Evaluation | GLC win rate vs AlphaZero `[PRIOR]` | GLC win rate vs PPO `[PRIOR]` |
|---|---:|---:|
| Barrage 4×4 | 25%–65% | 35%–75% |
| Barrage 6×6 | 20%–65% | 30%–78% |
| 8×8 OOD | 10%–60% | 15%–70% |
| LOGO held-out game/family | 10%–60% | 15%–70% |

For each held-out game, define

\[
\rho_{\mathrm{LOGO}}
=
\frac{
s(\mathrm{GLC}_{\mathrm{zero\text{-}shot}},\mathcal O_{\mathrm{fixed}})
}{
s(\mathrm{GLC}_{\mathrm{target\text{-}trained}},\mathcal O_{\mathrm{fixed}})
}.
\]

Conditional on a denominator of at least \(0.25\):

| Quantity | 80% `[PRIOR]` |
|---|---:|
| LOGO score-transfer ratio \(\rho_{\mathrm{LOGO}}\) | 0.25–1.20 |

A ratio above one is possible through training noise or negative transfer. Report numerator, denominator, and paired uncertainty; the ratio alone can conceal weak absolute performance.

A held-out board size within Stratego is size transfer. It is not equivalent to leaving out an entire game family.

## 4.4 R22B: public-reasoning exact solve rates

Each cell is **Track A interval → Track B interval**, in percentage points, for exact verified solves.

These priors assume SAT evaluation on a declared satisfiable subset for assignment-solving accuracy. A mixed SAT/UNSAT benchmark requires separately defined certificate metrics.

| Method | Sudoku-Extreme | SATLIB 3-SAT | Sokoban | Countdown |
|---|---:|---:|---:|---:|
| AR_GRPO | 0–12 → 0–18 | 2–35 → 3–45 | 0–25 → 0–30 | 8–55 → 10–62 |
| MDLM | 0–20 → 0–28 | 3–45 → 5–55 | 0–20 → 0–25 | 5–45 → 7–52 |
| TRM_Looped | 0–30 → 0–40 | 5–50 → 7–60 | 1–35 → 2–42 | 10–65 → 12–72 |
| Searchformer | 0–25 → 0–35 | 5–50 → 7–60 | 2–45 → 3–52 | 12–70 → 15–78 |
| GLC_AOM_v2 | 0–30 → 0–42 | 5–55 → 7–65 | 2–45 → 3–55 | 12–70 → 15–80 |

These deliberately overlapping intervals do not support an advance victory claim.

The mild upward Track-B shift assumes helper operations supplement a matched neural allowance. Under a fixed total compute budget that charges helper work, the gain can disappear or reverse.

On Sudoku-Extreme, all methods scoring zero would be unsurprising. A large GLC solve-rate jump would warrant inspection of helper recursion, puzzle duplication, training exposure, and exact verification—but should not be dismissed if those checks pass.

BoN and Stream_of_Search must remain in the actual \(2\times7\) experiment even though the requested forecast table names only five methods.

## 4.5 R22B: clean CLRS OOD and scaling

A single “CLRS accuracy” is not well-defined across heterogeneous tasks. The following forecasts assume genuine task coverage, eligible-output scoring, and per-task reporting.

| Metric | 80% `[PRIOR]` |
|---|---:|
| Macro-average eligible-output accuracy, \(N=16\) | 60%–97% |
| Macro-average eligible-output accuracy, \(N=64\) | 20%–80% |
| Predecessor accuracy on reachable non-source nodes, \(N=64\) | 20%–75% |
| Positive-class mask F1, \(N=64\) | 10%–85% |
| Pooled categorical micro-F1, if explicitly reported, \(N=64\) | 25%–90% |
| Exact complete-instance correctness, \(N=64\) | 0%–25% |

The wide micro-F1 interval reflects how strongly pooling can depend on task composition and easy labels. It should not be the sole headline.

For descriptive scaling, define slopes of log error against log parameter count and log executed budget, with higher positive slopes meaning lower error:

\[
\alpha_N^{\mathrm{eff}}
=
-\frac{\partial\log E}{\partial\log N},
\qquad
\alpha_B^{\mathrm{eff}}
=
-\frac{\partial\log E}{\partial\log B}.
\]

| Effective exponent | 80% `[PRIOR]` |
|---|---:|
| \(\alpha_N^{\mathrm{eff}}\) | −0.15 to 0.45 |
| \(\alpha_B^{\mathrm{eff}}\) | −0.15 to 0.40 |

Negative values are possible under undertraining or harmful repeated search. These are local empirical slopes, not universal scaling constants.

If fitting a floor-adjusted law, publish the floor uncertainty and parameter covariance. “Exponent not identifiable” is a valid outcome.

## 4.6 R23: real-task decentralized prediction

The table forecasts **held-out query-token accuracy**, not exact solution rate:

- Sudoku: withheld non-given cell values.
- SAT: target-assignment tokens under a declared convention.
- Stratego: a prespecified categorical target under partial observability.

SAT assignment-token accuracy is especially limited because another satisfying assignment can disagree with the reference. Exact constraint satisfaction must accompany it.

| Method | Sudoku 6×6 `[PRIOR]` | Equiv-SAT `[PRIOR]` | Micro-Stratego 4×4 `[PRIOR]` |
|---|---:|---:|---:|
| M1_Local_Greedy | 18%–45% | 45%–65% | 15%–45% |
| M2b_InfoMax_ZeroPrice | 18%–52% | 45%–70% | 15%–50% |
| M3a_Scalar_Kyle_LMSR | 17%–55% | 45%–72% | 12%–52% |
| M3_Target_Kyle_LMSR | 25%–75% | 50%–85% | 20%–70% |
| M4_Monolithic_Backprop | 30%–85% | 55%–90% | 25%–80% |

Stratego intervals are particularly fragile until the label semantics and class distribution are frozen. Include majority-class, random-legal, and local-information baselines.

The testable prior is modest:

> M3 should usually be more sample-efficient than sparse-payoff M3a; M4 should be difficult to beat under genuinely matched conditions.

If M3 substantially beats M4, investigate architecture, information, tuning, regularization, and compute differences before attributing the gain to market decentralization.

If M3 exactly matches M4, inspect whether its broadcast implements the same gradient.

## 4.7 Six-dimension rubric

These are protocol assessments, not measured scientific outcomes. Stage 2 is conditional on faithful implementation of the supplied plan. Stage 3 is conditional on the corrected gates proposed above, not the impossible universal versions in the prompt.

| Dimension | Stage 1: Wave-1 shortcuts | Stage 2: Wave-2 + S1–S10 | Stage 3: Wave-2 + corrected S11–S14 |
|---|---:|---:|---:|
| D1: Leakage enforcement | 1/10 | 4/10 | 8/10 |
| D2: Imperfect-information evaluation | 1/10 | 5/10 | 7/10 |
| D3: Symmetric reasoning evaluation | 1/10 | 7/10 | 8/10 |
| D4: OOD and compute-scaling honesty | 0/10 | 6/10 | 8/10 |
| D5: Real-task decentralized training | 2/10 | 5/10 | 8/10 |
| D6: Anti-sycophancy and falsifiability | 2/10 | 6/10 | 8/10 |

The remaining deductions are substantive:

- **D1:** finite tests do not prove universal information isolation; dependency and artifact provenance remain attack surfaces.
- **D2:** exact exploitability may be infeasible; strong baseline implementations and valid information-set search remain necessary.
- **D3:** helper semantics, compute matching, and task difficulty still determine whether symmetry is meaningful.
- **D4:** a small scaling grid and heterogeneous OOD tasks limit inference.
- **D5:** detached communication may still transmit global gradients; local information and market incentives need precise definitions.
- **D6:** transparency is valuable, but falsifiability ultimately requires accepting adverse outcomes without changing the benchmark afterward.

**Release criterion:** publish corrected predictions and independently recomputed metrics even if GLC ties or loses. A defensible paper can report narrower wins, failed transfer, or an integrity methodology; the supplied evidence currently supports none of the claimed performance victories.