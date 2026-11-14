# Round 4 — Expressivity, Supervised Imitation R13, and Unsupervised Discovery R14

**Recommendation:** proceed with R13, but replace the claims of “exact three-layer MCTS” and “complete resolution of five expressivity barriers” with narrower, testable statements. Descendant access addresses a real aggregation gap; the proposed v2 additions still need precise semantics for simulation counts, sequential execution, transpositions, information sets, ranking, and sound conflict reuse.

R13 should separate **representation**, **learnability**, and **execution fidelity**. R14 should use the same architecture, initialize independently of R13, and test whether task rewards alone produce useful search behavior and eliminate the multi-witness KL regression.

Everything below is an **inline mathematical audit and proposed preregistration**, not a fresh code inspection. Repository details and R11/R12 measurements are accepted as supplied context. All forecasts are explicitly marked **[PRIOR]**.

---

## 1. Audit of R4.1–R4.3

### 1.1 Definitions needed before making expressivity claims

A search representation should distinguish three objects:

1. A **state or occurrence node** \(j\).
2. A **directed action edge** \(e=(j,a,j')\).
3. A **simulation or evaluation event** \(t\), with traversed edge sequence \(p_t\) and backed-up return \(G_{t,e}\).

For ordinary tree MCTS, define

\[
N_e=\sum_t \mathbf 1[e\in p_t],\qquad
W_e=\sum_t \mathbf 1[e\in p_t]G_{t,e},\qquad
Q_e=
\begin{cases}
W_e/N_e,&N_e>0,\\
Q_{\mathrm{FPU}}(e),&N_e=0.
\end{cases}
\]

Here \(Q_{\mathrm{FPU}}\) is the teacher’s declared first-play-urgency convention. The return \(G_{t,e}\) includes the appropriate player perspective, discounting, and intervening rewards.

This is different from averaging one immutable value per descendant node. The latter equals MCTS backup only under restrictive assumptions:

- Every simulation creates exactly one new evaluation record.
- Each record belongs to exactly the ancestors traversed by that simulation.
- Repeated terminal visits and repeated leaf evaluations are represented.
- The return can be converted into the receiving ancestor’s value convention.
- No event is counted twice.

A useful general representation therefore includes a simulation–edge incidence relation

\[
A_{e,t}=\mathbf 1[e\in p_t],
\]

or an equivalent incremental update mechanism. On a tree, descendant masks can implement this relation if evaluation events are attached to the correct occurrences. On a graph, state reachability alone cannot replace it.

**Consequent correction:** “subtree size,” “number of children,” and “MCTS visit count” are three different quantities.

---

### 1.2 R4.1: the locality obstruction is valid; the stated numerical bound needs assumptions

#### A. Correct locality theorem

Let an \(L\)-layer network operate on a fixed graph using only one-hop messages:

\[
h_j^{(\ell+1)}
=
F_\ell\!\left(
h_j^{(\ell)},
\operatorname{Agg}_{k\in\mathcal N(j)}
G_\ell(h_j^{(\ell)},h_k^{(\ell)},e_{jk})
\right).
\]

Assume:

- Initial \(h_j^{(0)}\) contains only local information.
- There are no global tokens, precomputed subtree summaries, or nonlocal normalization.
- The topology is fixed independently of the perturbed leaf values.

Then \(h_j^{(L)}\) is independent of inputs outside \(j\)’s radius-\(L\) neighborhood. In particular,

\[
\frac{\partial h_j^{(L)}}{\partial v_k}=0
\quad\text{if}\quad
\operatorname{dist}(j,k)>L,
\]

where the derivative exists. The stronger statement is functional independence, which also covers nondifferentiable networks.

This does not automatically apply to a recurrent system whose persistent states already contain earlier nonlocal information. Its effective communication depth includes previous updates.

#### B. Correct worst-case error bound

Let the target be

\[
Q_j=\frac1n\sum_{k\in\mathcal D_j}v_k,\qquad v_k\in[-1,1],
\]

and let \(u\) of the \(n\) values lie outside the receptive field. Construct two inputs agreeing on all visible values and setting every invisible value to \(+1\) or \(-1\). Their targets differ by \(2u/n\), while the model output is identical. Therefore,

\[
\inf_{\widehat Q}\sup_v|\widehat Q-Q_j|
\ge \frac un.
\]

The supplied expression

\[
\frac{D-L}{D+1}
\]

is a valid special case for a chain with \(D+1\) included values and \(D-L\) invisible values. It is not a topology-independent bound determined by depth alone. For values in \([0,1]\), the corresponding bound is \(u/(2n)\).

#### C. Correct flat-attention obstruction

An unmasked attention operator is permutation invariant over a bag of key–value pairs, conditional on its query. If two trees give a target node the same query and the same accessible bag but different target aggregates, that operator cannot distinguish them.

If the targets differ by \(\Delta\), worst-case absolute error is at least \(\Delta/2\).

The parent-swapping construction supports this claim only if the accessible representations do **not** already encode the relevant parent membership. Parent IDs, path encodings, or earlier topology-aware computation may distinguish the bags. Thus:

> The obstruction concerns missing relational information, not unmasked attention in isolation.

The supplied \(1/2\) error is achievable under a suitable normalization—for example, parent value zero and one child value \(\pm1\), averaged together—but is not universal.

#### Implication for R12

Conditional on the supplied implementation summary:

- Immediate-parent `n_exp` updates do not implement subtree visits.
- Write-once node values do not implement incremental ancestor backup.
- A fixed PUCT formula is an architectural policy, not a learned search controller.
- One-hop mixing and content attention provide no demonstrated exact simulation of MCTS backup.

R12’s reported performance remains evidence of useful task learning. It does not establish faithful MCTS representation.

---

### 1.3 R4.2: descendant pooling is sufficient for a restricted aggregate, not yet exact three-layer MCTS

#### A. Clarify whether ancestry includes self

The recurrence

\[
M_{\mathrm{tree}}[b,:]
=
M_{\mathrm{tree}}[\operatorname{pa}(b),:]\lor e_b
\]

stores **reflexive ancestry**: every node is its own ancestor.

Define explicitly

\[
\operatorname{Anc}^{+}(j)=\operatorname{Anc}(j)\cup\{j\},
\qquad
\operatorname{Desc}^{+}(j)=\operatorname{Desc}(j)\cup\{j\}.
\]

Otherwise, leaf counts, root exclusions, and empty reductions are ambiguous.

Only actual evaluation-event tokens should contribute to MCTS statistics. Structural nodes without evaluations require zero weight.

#### B. Counts from LSE need an exact convention

For a nonempty set of \(n\) valid events, zero logits give

\[
\operatorname{LSE}(0,\ldots,0)=\log n.
\]

But the proposed stabilized expression gives

\[
\log(\epsilon+n),
\]

which is not exactly \(\log n\). Recovering \(n\) requires subtracting \(\epsilon\) after exponentiation, with numerical consequences.

Prefer explicit masked sum/count primitives:

\[
N_j=\sum_t A_{j,t},\qquad
W_j=\sum_t A_{j,t}G_{t,j}.
\]

Use integer counts and floating-point sums; handle \(N_j=0\) explicitly. Uniform attention may compute \(W_j/N_j\) when the set is nonempty, but does not itself preserve count information.

#### C. Return perspective must be correct

For alternating two-player zero-sum games, a leaf value in the leaf player’s perspective generally needs

\[
G_{t,j}=(-1)^{d(\ell_t)-d(j)}v_{\ell_t}
\]

when there are no intermediate rewards and no extra turns. General environments require player-aware reward accumulation, not a hardcoded depth-parity rule.

A cleaner convention stores values in a declared global player basis and applies a receiving-player readout. Multi-player and general-sum settings require vector values or another explicit utility representation.

#### D. Standard MLPs do not automatically implement exact PUCT arithmetic

The PUCT expression uses division and square roots:

\[
U_e=Q_e+cP_e\frac{\sqrt{N_j}}{1+N_e}.
\]

A finite ReLU MLP is piecewise linear and cannot exactly implement these operations on an unrestricted continuous domain. Exactness requires one of:

- Explicit arithmetic primitives.
- A finite-domain construction with declared bounds and precision.
- An approximation theorem with numerical error bounds.

For this project, use the third interpretation for learned arithmetic, and the first for architectural capability checks. Report them separately.

#### E. Finite-temperature extrema are approximate

For \(m\) values \(x_1,\ldots,x_m\), define

\[
\operatorname{smax}_\beta(x)
=
\frac1\beta\log\sum_{r=1}^{m}e^{\beta x_r}.
\]

Then

\[
\max_r x_r
\le \operatorname{smax}_\beta(x)
\le \max_r x_r+\frac{\log m}{\beta}.
\]

For the softmax-weighted mean,

\[
\operatorname{Attn}_\beta(x)=\sum_r p_r x_r,
\qquad p_r\propto e^{\beta x_r},
\]

the corresponding bound is

\[
0\le \max_r x_r-\operatorname{Attn}_\beta(x)
\le \frac{\log m}{\beta}.
\]

Analogous bounds hold for softmin.

Therefore finite \(\beta_{\max}\) does not establish exact path minima or sibling maxima. Use hard reductions at evaluation, or state a margin condition that exceeds the total approximation error.

#### F. The path-margin claim mishandles ties and singletons

If the sibling set includes \(j\), then

\[
U_j-\max_{s\in\operatorname{Sib}(j)}U_s\le0
\]

always. If it excludes \(j\), a unique winning child has positive margin, but a tied winning child has margin zero. A sole child requires an empty-set convention.

Use a deterministic total order. For example, select by

\[
K_e=(U_e,-\operatorname{id}(e))
\]

under lexicographic comparison. Define

\[
w_e
=
\mathbf 1\!\left[
e=\operatorname*{arg\,max}_{f:\operatorname{pa}(f)=\operatorname{pa}(e)}
K_f
\right].
\]

Then the exact prefix-consistency predicate is

\[
g_j
=
\prod_{e\in\operatorname{path}(0,j)}w_e
=
\mathbf 1\!\left[
\sum_{e\in\operatorname{path}(0,j)}(1-w_e)=0
\right].
\]

This handles ties and singletons without infinitesimal numerical perturbations.

Every prefix of the greedy trajectory satisfies \(g_j=1\). To identify the expansion target, also require the teacher’s stopping predicate:

\[
\operatorname{target}(j)
=
g_j\land\operatorname{stop}(j).
\]

Unexpanded legal actions must appear as candidate edges or equivalent virtual entries. Selecting among existing nodes alone cannot reproduce all MCTS expansion decisions.

#### G. The claimed three-layer schedule hides dependent operations

In a standard transformer block, attention weights are computed before that block’s output MLP. Consequently, a block cannot generally:

1. Fetch parent counts.
2. Compute new PUCT scores.
3. Attend over siblings using those newly computed scores.

The dependency graph is

\[
\text{aggregate}
\to
\text{fetch parent statistics}
\to
\text{compute edge scores}
\to
\text{compare siblings}
\to
\text{reduce along paths}.
\]

This can have constant depth independent of tree depth, but “three layers” depends on a nonstandard block definition or extra preattached features. Count actual sequential attention/reduction stages.

#### Corrected R4.2

> Given valid evaluation-event incidence, the required value convention, explicit count access, deterministic tie-breaking, and suitable arithmetic/comparison primitives, MCTS statistics and a greedy selection path can be computed with a constant number of relational stages independent of tree depth. Finite-temperature and learned-arithmetic versions require explicit error and score-margin bounds.

This is a useful positive result. It is narrower than exact AlphaZero MCTS in three ordinary transformer layers.

---

### 1.4 R4.3: useful missing capabilities, but neither a complete impossibility theorem nor complete resolution

The five proposed additions identify important engineering needs. Several supporting impossibility claims are too strong.

A fixed-width, fixed-depth architecture needs a precise computational model before a lower bound follows: token access, positional information, precision, nonlinearities, recurrence, and width scaling all matter. Boolean formula evaluation is associated with \(\mathsf{NC}^1\), but invoking that class does not prove a lower bound for an unrestricted real-valued attention network.

Likewise, a bounded-horizon model might memorize rank, topology, or algorithm traces. The relevant question is whether it has a **compact, robust, generalizing implementation**.

#### Barrier 1: minimax, alpha–beta, and PNS

Minimax follows

\[
V(j)=
\begin{cases}
u(j),&j\text{ terminal},\\
\max_{k\in\operatorname{Ch}(j)}V(k),&j\text{ maximizing},\\
\min_{k\in\operatorname{Ch}(j)}V(k),&j\text{ minimizing}.
\end{cases}
\]

A flat commutative pool of leaves cannot generally recover this nested computation without information about their grouping. However, calling it simply a “non-commutative alternating semiring” obscures the actual requirement: topology-dependent composition of different reductions.

A bottom-up spine update can maintain exact values **if**:

- Child summaries are valid.
- Internal nodes are updated from deepest to shallowest.
- Terminal and unexpanded-node semantics are declared.
- Updates propagate until all affected summaries are current.

This takes \(O(D)\) sequential update rounds along a depth-\(D\) path, even if only \(O(D)\) nodes change.

Alpha–beta additionally requires:

\[
(\alpha_j,\beta_j,\text{child cursor},\text{return state},\text{bound type}).
\]

A directed older-sibling mask does not supply this execution state by itself. Creation order is not necessarily the teacher’s move-search order.

For proof-number search,

\[
\begin{aligned}
\text{OR:}\quad
\operatorname{pn}(j)&=\min_k\operatorname{pn}(k),&
\operatorname{dn}(j)&=\sum_k\operatorname{dn}(k),\\
\text{AND:}\quad
\operatorname{pn}(j)&=\sum_k\operatorname{pn}(k),&
\operatorname{dn}(j)&=\min_k\operatorname{dn}(k).
\end{aligned}
\]

Solved and disproved states need exact sentinel semantics. `df-pn` additionally needs threshold state and its declared update rules.

**Fix V2.1:** a recurrent, typed update controller with a stack/cursor, generic min/max/sum operations, explicit solved-state flags, and configurable work limits. A fixed three-pass network is not an equivalent replacement.

#### Barrier 2: transpositions and MCGS

The proposed

\[
E_{\mathrm{trans}}M_{\mathrm{desc}}
\]

is not, in general, transposition-aware reachability closure. Ordinary multiplication counts paths; Boolean multiplication performs only one composition. Neither guarantees arbitrary transitive closure.

Also,

\[
\mathbf 1[\text{equal state}]
\lor
\exp(-\|h_j-h_k\|^2/\tau)
\]

mixes Boolean and real-valued semantics. Interpreted by nonzero support, it links every pair because the exponential is positive.

**Exact fix:**

1. Construct verified state equivalence \(\sim\), respecting every transition-relevant component.
2. Quotient occurrences into equivalence classes.
3. Build the quotient transition relation \(A\).
4. If reachability is required, compute Boolean closure

   \[
   R=I\lor A\lor A^2\lor\cdots
   \]

   with a declared algorithm.

State identity must include side to move, relevant repetition/history state, and any other variable affecting transitions or rewards. A Zobrist hash is a lookup accelerator, not a proof of equality; verify collisions against a canonical representation.

The quotient graph may contain cycles. Options include:

- Horizon-indexed unrolling.
- Explicit repetition termination rules.
- Strongly connected component processing with a well-defined value equation.
- A named graph-search teacher with explicit cycle handling.

Do not equate reachable states with simulation visits. A return is backed up to edges actually traversed under the teacher’s rules. Two paths reaching one state do not imply two independent pieces of evidence.

**Fix V2.2:** separate exact identity, approximate retrieval, graph adjacency, simulation provenance, and edge/state statistics. Learned similarity may propose matches; it must not silently authorize exact merging.

#### Barrier 3: CFR+, information sets, and the LIR firewall

Equality of current observations is insufficient for general extensive-form information sets. Use the acting player’s permitted observation–action history, or an equivalent sufficient state under a declared perfect-recall game model:

\[
I_i(h)=I_i(h')
\]

only when the player cannot distinguish histories \(h,h'\).

Counterfactual reach includes chance and all opponents:

\[
\pi_{-i}^\sigma(h)
=
\pi_c(h)\prod_{k\ne i}\pi_k^\sigma(h).
\]

For action \(a\),

\[
r_t(I,a)
=
v_i^{\sigma_t}(I,a)-v_i^{\sigma_t}(I),
\]

where counterfactual values sum over histories in \(I\), weighted by \(\pi_{-i}^{\sigma_t}\), and include continuation values after the action.

CFR+ updates

\[
R_t^+(I,a)
=
\max\{R_{t-1}^+(I,a)+r_t(I,a),0\},
\]

then uses

\[
\sigma_{t+1}(I,a)=
\begin{cases}
R_t^+(I,a)/\sum_{a'}R_t^+(I,a'),&\sum_{a'}R_t^+(I,a')>0,\\
1/|\mathcal A(I)|,&\text{otherwise}.
\end{cases}
\]

A complete teacher also declares simultaneous versus alternating updates and the averaging schedule. CFR’s equilibrium guarantee concerns the appropriate average strategy, not arbitrary final iterates.

**Firewall requirements:**

- One policy distribution per information set, not one per hidden-state determinization.
- Hidden-world tokens may inform an authorized belief integration, but cannot select different observable actions for indistinguishable real histories.
- Actual private opponent state and oracle labels remain outside the actor’s permitted inputs.
- Counterfactual evaluation must integrate worlds using declared reach/sampling weights.
- A simulated hidden-world action must not be treated as a legal contingent plan for the real player.

These prevent strategy fusion and hidden-state leakage. Perfect-recall assumptions and sampling corrections must be tested explicitly.

**Fix V2.3:** information-set storage, counterfactual evaluation, persistent regret and average-strategy accumulators, legal-action masking, and a firewall-preserving policy readout.

ReBeL and R-NaD are not implemented merely by adding regret matching. Their objectives and state constructions require separate specifications.

#### Barrier 4: Gumbel sequential halving

Define a deterministic descending rank among active siblings:

\[
r_j
=
1+
\sum_{s\ne j}
\mathbf 1[
S_s>S_j
\;\lor\;
(S_s=S_j\land\operatorname{id}(s)<\operatorname{id}(j))
].
\]

Keep the first \(k_p\) actions under the teacher’s phase schedule. This fixes ties exactly.

For the sigmoid relaxation and all nonzero score gaps at least \(\delta\),

\[
\left|
\sigma((S_j-S_s)/\tau)
-
\mathbf 1[S_j>S_s]
\right|
\le e^{-\delta/\tau}.
\]

Thus the normalized percentile error is at most \(e^{-\delta/\tau}\) away from ties; the unnormalized rank error is at most \((m-1)e^{-\delta/\tau}\). At ties, the sigmoid gives \(1/2\), not a deterministic rank.

The relaxation therefore does not guarantee exact elimination unless the threshold margin exceeds its error.

Sequential halving also requires:

- Persistent active-set membership.
- Phase and allocation state.
- Fixed, replayable Gumbel draws.
- The teacher’s completed-value and score convention.
- Explicit treatment of budgets not divisible by the phase schedule.

**Fix V2.4:** exact stable ranking and phase bookkeeping for capability tests; a separately evaluated smooth or stochastic relaxation for learning.

“Softmax cannot compute rank” is not a general bounded-domain impossibility theorem.

#### Barrier 5: CDCL and global nogoods

A globally accessible conflict token is not yet a learned clause.

For nogood

\[
C=\neg\ell_1\lor\cdots\lor\neg\ell_m,
\]

a partial assignment is:

- Conflicting if every \(\ell_r\) is true.
- Unit if all but one are true and the remaining literal is unassigned.
- Otherwise neither.

A scalar clash score does not establish clause entailment, assignment compatibility, or a valid backjump.

**Fix V2.5:** persistent clause tokens containing literals, a trail with decision levels and reasons, a declared conflict-analysis procedure, Boolean constraint propagation, backjump control, and independently checkable clause soundness.

Neural conflict proposals may guide search. They may prune as exact constraints only after verification. Otherwise, label them heuristic and measure any loss of completeness separately.

---

### 1.5 Computational complexity: v2 is not uniformly \(O(b)\) per step

Let \(b\) be active tokens, \(D\) depth, \(H\) attention heads, \(d\) hidden width, and \(w\) machine-word width.

| Operation | Representative cost | Qualification |
|---|---:|---|
| Append one ancestry bitset | \(O(b/w)\) word operations | All stored ancestry masks use \(O(b^2/w)\) words |
| Dense relational attention | \(O(Hb^2d)\) | Per full refresh, excluding projection constants |
| Maintain MCTS along one path | \(O(D)\) | With event counts and sums |
| Refresh alternating summaries | \(O(D)\) update rounds | More work if each child set is rescanned |
| Directed sibling mask construction | Up to \(O(b^2)\) explicitly | Can be implicit |
| Pairwise sibling ranking | \(O(\sum_p m_p^2)\) | Worst case \(O(b^2)\) |
| Stable sorting of sibling scores | \(O(\sum_p m_p\log m_p)\) | Often preferable to pairwise ranking |
| Full graph reachability | \(O(b^3)\) naïvely | Bitsets improve constants; updates remain graph-dependent |
| One CFR iteration | Proportional to traversed game tree | Infoset pooling alone does not perform traversal |
| Clause matching/propagation | Depends on literal occurrences and propagation cascade | Not bounded by tree depth alone |
| Leave-subtree-out reruns | Up to number of removals × search cost | Needs subsampling or a cheaper declared estimator |

At \(b=256\), one Boolean relation contains \(65{,}536\) entries: about \(8\) KiB bit-packed or \(256\) KiB in float32. Seven float32 masks alone occupy about \(1.75\) MiB, before heads, batch dimensions, activations, and gradients.

Quadratic operations may be entirely practical here. They should be measured rather than relabeled linear. Refreshing \(O(b^2)\) attention after every append gives cumulative \(O(B^3)\) attention work over a search unless caching or sparse updates change the computation.

### 1.6 What corrected v2 can and cannot claim

Corrected v2 is a **bounded recurrent search machine with generic relational access and persistent state**. It plausibly supports the six selected families. It does not exhaust useful search algorithms.

Additional capabilities may be needed for:

- Memory-bounded search with deletion, reopening, and regenerated states.
- Iterative deepening, restarts, and cross-search caches.
- Chance-node expectation and specialized stochastic pruning.
- Continuous-action optimization and progressive widening.
- Particle filtering and continuous-state belief maintenance.
- Simultaneous-move matrix-game solving.
- Distributed asynchronous search and virtual-loss bookkeeping.
- Arbitrarily long proofs, histories, or searches beyond finite storage.

Some are implementable with the same generic recurrent machinery; others require new storage or environment interfaces. Passing R13 establishes support for the declared bounded teachers, not universal search-algorithm expressivity.

---

## 2. Specification input: `PREREG-R13-ALGO-IMITATION.md`

### 2.1 Objective and separation of claims

R13 asks three distinct questions:

1. **Capability:** can the architecture execute the required computations?
2. **Learnability:** can it learn them from a modest number of teacher trajectories?
3. **Generalization:** does that learned implementation work at greater depth and budget?

Use three evaluation modes:

- **Constructive mode:** manually configured operators/controller or verified compilation into the architecture. This tests capability.
- **Teacher-forced mode:** predict teacher state transitions from correct preceding states. This tests local learning.
- **Free-running mode:** execute the learned controller without state correction. This tests accumulated errors and actual algorithm imitation.

A constructive demonstration alone does not pass a learned-imitation gate. A failed training run does not establish mathematical non-representability.

### 2.2 Freeze a concrete architecture

Before training, freeze:

- Token schemas for states, edges, evaluations, clauses, information sets, and controller state.
- Which relations are exact versus learned.
- Arithmetic, hard reductions, comparison, ranking, and verification primitives.
- Recurrent update scheduling and maximum microsteps.
- Precision and deterministic tie-breaking.
- The actor’s permitted inputs.
- Memory and work limits.
- Which operations incur evaluation budget versus internal compute cost.

Any reference controller used in constructive tests must be disclosed. The trained controller must choose operations rather than receive a hidden teacher dispatch.

No environment-specific depth, wave, or stall constants belong in the core controller.

### 2.3 Teacher suite and recorded targets

| Family | Teacher specification | Required trace targets |
|---|---|---|
| **Algo 1: MCTS-PUCT** | Frozen PUCT variant; declared FPU, reward perspective, stopping and expansion conventions | Edge \(N,W,Q,U\), selected path, expansion target, root policy |
| **Algo 2: Minimax/alpha–beta and PNS** | Separate alpha–beta and PNS teachers; `df-pn` is a separately scored extension | Values/bounds, stack/cursor, cutoffs; proof/disproof numbers, most-proving node |
| **Algo 3: CDCL-global-nogood** | Frozen clause-learning, branching, propagation, restart and tie conventions | Trail, reasons, learned clause, backjump, propagation, final certificate |
| **Algo 4: MCGS and disjunctive belief propagation** | Separate graph-search and finite multi-witness inference fixtures | Verified state classes, edge/event statistics, root policy; posterior or marginal targets |
| **Algo 5: CFR+** | Perfect-recall finite games; frozen update and averaging schedule | Infoset reach, action values, regrets, next strategy, average strategy |
| **Algo 6: Gumbel sequential halving** | Frozen root scores, Gumbel stream, completed values and phase allocation | Active sets, ranks, allocations, eliminations, final action |

Two distinctions are mandatory:

- Minimax and PNS have different semantics and must each pass; averaging their scores hides failures.
- MCGS is not belief propagation. On loopy factor graphs, BP is generally not exact inference. Use exact enumeration or junction-tree targets where exact posteriors are required; label loopy-BP imitation as imitation of that particular approximation.

Teachers are evaluated on fully specified finite fixtures before generating training data. Teacher agreement with another implementation is useful but does not replace semantic invariants or certificate checking.

### 2.4 Data, splits, and sample-efficiency curves

Use nested training sets

\[
N_{\mathrm{train}}\in\{128,512,2048\}
\]

**per family**, with at least five training seeds per cell.

A training example is a complete problem/rollout bundle, not an arbitrarily sliced teacher transition. Report both trajectory counts and total supervised transitions. Split family-internal subtracks evenly and disclose their counts.

Proposed splits:

- Training: \(B\le32\), maximum dependency depth \(8\).
- IID test: unseen instances at the same ranges.
- Depth OOD: depth \(16\) and \(32\), with comparable token budgets.
- Budget OOD: blind \(B\in\{64,128\}\).
- Joint OOD: deeper structures and larger budgets together.
- Numerical stress: tied, nearly tied, zero-count, extreme-value, and empty-mask cases.
- Structural stress: transpositions, repeated terminals, cycles, multiple witnesses, long clauses, and unequal information-set reaches.

Canonicalize instances before splitting. Related puzzles, game trees, and symmetry variants remain in one split. Freeze teachers, seeds, and hyperparameters before opening blind tests.

Use at least 1,000 held-out problem bundles per main test stratum where simulation cost permits. Confidence intervals must cluster by problem and training seed, not treat correlated tokens as independent samples.

### 2.5 Losses

Let \(a_t\) be the teacher operation, \(x_t\) its numeric state, and \(p_t\) a policy distribution. A general loss is

\[
\mathcal L_{\mathrm{R13}}
=
\lambda_a\mathcal L_{\mathrm{op}}
+
\lambda_x\mathcal L_{\mathrm{state}}
+
\lambda_p\mathcal L_{\mathrm{policy}}
+
\lambda_c\mathcal L_{\mathrm{certificate}}.
\]

Recommended components:

\[
\mathcal L_{\mathrm{op}}
=
-\log p_\theta(a_t\mid s_t),
\]

with structured pointer losses for selected nodes, edges, clauses, or information sets;

\[
\mathcal L_{\mathrm{state}}
=
\sum_r
\operatorname{Huber}\!\left(
\frac{\widehat x_{t,r}-x_{t,r}}{s_r}
\right),
\]

where \(s_r\) is fixed using training data only;

\[
\mathcal L_{\mathrm{policy}}
=
D_{\mathrm{KL}}(p_t\Vert\widehat p_t).
\]

For counts and proof numbers, also report absolute error on raw values; log-space accuracy can hide off-by-one failures. Treat solved-state sentinels as discrete flags.

For ties, distinguish:

- **Trace fidelity:** reproduce the frozen teacher’s exact tie-break.
- **Semantic fidelity:** choose any action with the same admissible value or certificate.

Report both. The main deterministic imitation gate uses trace fidelity.

### 2.6 Pass bars: R13-D0 through R13-D6

All numbers below are **proposed preregistered bars**. A gate passes only when all listed conditions pass at \(N_{\mathrm{train}}=2048\). Curves at 128 and 512 are mandatory, not additional pass requirements.

For accuracy, use a one-sided 95% lower confidence bound; for errors, an upper bound. Correct within each gate for its declared primary endpoints. Report every seed, including failures.

| Gate | Required pass |
|---|---|
| **R13-D0: architecture and validity** | Supplied-style static/purity checks show zero violations; all firewall canaries pass; hidden-field poisoning gives exact invariance in deterministic evaluation; empty masks, ties and zero counts have declared finite behavior; verified primitives match references on adversarial fixtures; no algorithm-specific controller is hidden in the learned condition |
| **R13-D1: MCTS-PUCT** | Free-running next-expansion fidelity \(\ge99\%\) IID and \(\ge97\%\) joint OOD; aggregate-state normalized RMSE \(\le0.01\) IID and \(\le0.03\) OOD; \(R^2\ge0.99\) IID and \(\ge0.97\) OOD on nonconstant \(Q,W\) targets; root-policy mean TV \(\le0.02\) IID and \(\le0.05\) OOD |
| **R13-D2: alpha–beta and PNS** | Each subtrack separately: next-operation fidelity \(\ge98\%\) IID and \(\ge95\%\) OOD; root result agreement \(\ge99\%\) IID and \(\ge97\%\) OOD; zero accepted invalid proof/cutoff certificates in the verifier suite; at least 80% exact complete traces IID and 60% OOD |
| **R13-D3: CDCL-global-nogood** | Zero unsound accepted clauses or incorrect certified SAT/UNSAT outputs; verified clause production succeeds on \(\ge95\%\) IID and \(\ge85\%\) OOD attempted learning steps; propagation/backjump fidelity \(\ge98\%\) IID and \(\ge95\%\) OOD; solve rate at least 95% of teacher IID and 90% OOD at equal logical work budget |
| **R13-D4: MCGS and multi-witness inference** | Graph-search next-operation fidelity \(\ge98\%\) IID and \(\ge95\%\) OOD; edge-count conservation exact; no unverified merges; posterior KL \(\le0.02\) IID and \(\le0.05\) OOD on exact-inference fixtures; invariance to duplicate syntactic encodings and witness enumeration order within \(10^{-6}\) |
| **R13-D5: CFR+** | Infoset policy mean TV \(\le0.02\) IID and \(\le0.05\) OOD; regret-increment \(R^2\ge0.98\) IID and \(\ge0.95\) OOD; indistinguishable histories produce the same policy; exploitability of the learned average strategy no more than teacher exploitability \(+0.02\) IID or \(+0.05\) OOD, with payoffs normalized to \([-1,1]\) |
| **R13-D6: Gumbel halving** | Correct phase active sets on \(\ge99\%\) IID and \(\ge97\%\) OOD traces; final-action agreement \(\ge99\%\) IID and \(\ge97\%\) OOD; exact budget accounting and monotone active-set elimination; deterministic agreement on the tie stress suite |

For \(R^2\), exclude constant-target strata from that statistic and report their absolute errors separately:

\[
R^2
=
1-\frac{\sum_n(\widehat y_n-y_n)^2}
{\sum_n(y_n-\bar y)^2}.
\]

Do not silently remove difficult near-constant strata.

Step fidelity does not imply whole-trace fidelity. Under an illustrative independent-error approximation, \(99\%\) step accuracy over 128 steps gives only \(0.99^{128}\approx0.276\) exact traces. Always report complete traces and final results alongside local accuracy.

### 2.7 Ablation matrix

| Capability | Full-GLCSearch-v2 | Abl-GLCSearch-v1 | Abl-R12-1Hop-Flat |
|---|---|---|---|
| Exact ancestry/descendant relations | Yes | Yes | No descendant relation |
| Count-preserving pooling | Yes | Yes | Supplied R12-style representation |
| Recurrent typed updates and controller state | Yes | No v2 controller state | No v2 controller state |
| Verified transposition identity/provenance | Yes | No | No |
| Information-set state and regret accumulators | Yes | No | No |
| Stable rank and phase state | Yes | No | No |
| Persistent verified clause store | Yes | No | No |
| Trainable operation selection | Yes | Yes | Yes in the learnability comparison |

The supplied R12 selector is fixed. Therefore include an additional **frozen R12 historical baseline**. A trainable R12-shaped ablation is required to isolate representational limits from the trivial fact that a frozen selector cannot imitate arbitrary teachers.

Run:

1. Approximately parameter-matched comparisons.
2. Matched internal-compute comparisons.
3. End-to-end wall-time comparisons.

Add one-at-a-time v2 removals for each proposed primitive. For each, test a fixture where the primitive is relevant and a control where it should not matter.

A mechanistic attribution is supported if removing the primitive causes a preregistered degradation—such as at least 10 percentage points in operation fidelity or a twofold increase in primary state error—with a paired confidence interval excluding no effect. If v1 succeeds without the primitive, revise the necessity claim.

---

## 3. Specification input: `PREREG-R14-UNSUP-DISCOVERY.md`

### 3.1 What “unsupervised discovery” means here

R14 uses **task-driven training without search-algorithm targets**. It is not label-free in every sense: terminal outcomes, verified witnesses, and oracle belief scores are training signals.

To preserve the intended distinction:

- Initialize R14 independently of R13.
- Do not use R13 checkpoints, teacher traces, teacher action labels, or imitation losses.
- Do not tune against mechanistic alignment scores.
- Do not reward agreement with MCTS, CFR+, CDCL, or any other named algorithm.
- Freeze the architecture and generic primitives before R14.
- Keep algorithm names and teacher identities out of the actor’s task interface.

The same architecture may contain generic sums, minima, comparisons, persistent memory, and verifiers. That is an inductive bias. It does not establish that their use was learned unless the controller’s behavior and causal dependence are measured.

An R13-initialized transfer condition may be useful, but must be reported separately and cannot support the primary discovery claim.

### 3.2 Task-only objective

For a finite target distribution \(P^*\) and end-of-search belief \(\mu_B\),

\[
L_{\mathrm{belief}}
=
D_{\mathrm{KL}}(P^*\Vert\mu_B).
\]

The oracle distribution is available to the evaluator or reward computation, never as an actor input. If only a realized outcome \(Y\sim P^*\) is available, use log loss:

\[
L_{\mathrm{log}}=-\log\mu_B(Y),
\]

whose expectation is \(H(P^*)+D_{\mathrm{KL}}(P^*\Vert\mu_B)\).

Prevent infinite scores using a declared full-support output distribution, for example

\[
\mu_B=(1-\epsilon)\widetilde\mu_B+\epsilon\nu,
\]

where \(\nu\) has full support and \(\epsilon\) is frozen across methods. Report the scoring convention because it affects comparisons with R12.

Let

- \(W_B\in\{0,1\}\) denote discovery of a valid witness under an independent verifier.
- \(u_B\) denote normalized interactive payoff.
- \(C_B\) denote measured search cost.

A task-level reward can be

\[
R_B
=
-\lambda_{\mathrm{belief}}L_{\mathrm{belief}}
+\lambda_{\mathrm{witness}}W_B
+\lambda_{\mathrm{game}}u_B
-\lambda_{\mathrm{cost}}C_B.
\]

Use only applicable terms for each environment. Normalize terms using training/calibration data, freeze weights before blind testing, and report component outcomes separately.

For small imperfect-information games with exact best responses, exploitability may replace or supplement sampled payoff. Approximate exploiters generally provide a **lower bound on exploitability**, not a certificate of low exploitability.

### 3.3 Leave-subtree-out credit

Use a loss-oriented proper score \(S\), so the supplied credit

\[
I_{i,s}
=
S(q_i^{-s},y_i)-S(q_i^{\mathrm{full}},y_i)
\]

is positive when subtree \(s\) improves the prediction.

There are two different experiments.

#### A. Evidence-removal credit

Hold the realized search fixed; remove evidence and derived summaries attributable to \(s\); recompute the final belief.

This measures contribution to the readout. It is not the causal effect of choosing a different search action.

#### B. Intervention-and-rerun credit

Restore the checkpoint before subtree \(s\) was chosen, prohibit that choice, and rerun with the same total budget and a declared alternative-selection rule. Couple random numbers when possible.

This measures a policy-dependent intervention effect. It includes alternative use of the released budget.

Both require provenance. Removing \(s\) must also invalidate its dependent:

- Backed-up statistics.
- Learned clauses.
- Transposition contributions.
- Regret/value updates.
- Belief messages.

Shared evidence needs an explicit ownership rule. A multiply reached state cannot simply disappear because one incoming path was removed.

One-at-a-time removal credits need not add to total improvement. With redundant witnesses, every individual credit can be small although the set is valuable. Include sampled group removals or Shapley-style random coalition probes on the multi-witness suite.

For normalized approximation,

\[
\phi_s
=
\mathbb E_\pi
\left[
F(P_\pi(s)\cup\{s\})-F(P_\pi(s))
\right],
\]

where \(F=-S\) and \(P_\pi(s)\) is the set preceding \(s\) in a random ordering. These estimates can be expensive; report the number of coalitions and their compute cost.

### 3.4 GRPO training

For \(K\) search trajectories on the same problem,

\[
A_k
=
\frac{R_k-\overline R}
{\operatorname{sd}(R_1,\ldots,R_K)+\epsilon_A}.
\]

Use a clipped policy objective such as

\[
J(\theta)
=
\mathbb E_{k,t}
\left[
\min\left(
r_{k,t}A_k,\,
\operatorname{clip}(r_{k,t},1-\eta,1+\eta)A_k
\right)
\right]
-
\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}}),
\]

with

\[
r_{k,t}
=
\frac{\pi_\theta(a_{k,t}\mid s_{k,t})}
{\pi_{\mathrm{old}}(a_{k,t}\mid s_{k,t})}.
\]

Subtree credits may be assigned to their responsible operations as an explicitly declared auxiliary estimator. Detach credit targets and policy baselines appropriately.

Three caveats belong in the preregistration:

1. Clipping, group normalization, and substituted local credits do not preserve an exactly unbiased gradient of terminal reward.
2. Equal-reward groups supply no normalized advantage; log their frequency and use a declared fallback or additional proper-score training.
3. An entropy term is an exploration regularizer, not evidence of algorithmic discovery.

Require an ablation of terminal-reward GRPO alone versus terminal reward plus subtree credit.

### 3.5 Training and evaluation protocol

Use independently initialized v2, v1, and R12-shaped trainable models with the same task distribution and outcome signals.

Proposed primary protocol:

- Training budgets \(B\in\{16,32\}\).
- Blind test budgets \(B\in\{64,128,256\}\).
- At least five independent training seeds.
- Frozen total training-compute allocation per model class.
- Outcome-learning curves at 1%, 4%, 16%, and 100% of that allocation.
- Separate matched-evaluation-budget and matched-wall-time results.

A secondary budget-exposed run may train through \(B=128\), leaving \(B=256\) blind. Do not pool its results with the stronger budget-generalization condition.

Freeze the Mini-ECI definition, task weights, scoring convention, opponent pool, Elo fitting procedure, and baseline selection before testing. Select fixed-algorithm hyperparameters using calibration data only.

Comparisons against historical R12 numbers require reproducing the same metric protocol. If scoring or suites change, report new paired R12 evaluations; do not interpret raw historical differences as gains.

### 3.6 Mechanistic algorithmic alignment probes

Behavioral agreement is insufficient: different algorithms often choose the same winning move. Use both held-out predictions and controlled interventions.

Probes are fitted only after the policy is frozen and cannot affect model selection.

| Task group | Quantitative probe | Causal intervention |
|---|---|---|
| **Hex / Connect-4: MCTS** | Held-out \(R^2\) for event counts, backed-up means and PUCT-like ordering; rank correlation on nontrivial decisions | Modify one leaf return while fixing topology; measure affected-ancestor statistics and selection response |
| **Hex / Connect-4: minimax** | Agreement with alternating max/min summaries on trees where minimax and leaf averaging disagree | Change a dominated child versus the decisive child; test the predicted zero/nonzero root sensitivity |
| **Sudoku / Equiv-SAT: CDCL** | Recovery and verified reuse of conflict cores in disjoint branches | Remove a clause token; measure loss of valid cross-branch pruning |
| **Sudoku / Equiv-SAT: PNS** | \(R^2\) for proof/disproof summaries and most-proving-node choice | Alter one AND/OR child status and test the correct min/sum response |
| **Proposal-Poker: CFR+** | Regret-update \(R^2\), strategy TV to regret matching, average-strategy exploitability | Perturb counterfactual action values; test predicted regret and policy changes while preserving infoset invariance |
| **ShipHunt / Stratego: graph and multi-witness inference** | Identity/visit accounting, posterior calibration, sensitivity to alternative hypotheses | Duplicate an encoding without new evidence; add a genuinely distinct witness; compare the two responses |

Predeclare strong alignment on a task family as:

- Behavioral agreement or rank correlation \(\ge0.90\) on diagnostic cases.
- Held-out state-probe \(R^2\ge0.80\) where that statistic applies.
- Causal-response cosine similarity \(\ge0.80\), or the correct discrete response on \(\ge90\%\) of relevant cases.
- A primitive ablation producing the predicted task-specific degradation.

For a perturbation vector \(v\), compare

\[
J_{\mathrm{model}}v
\quad\text{with}\quad
J_{\mathrm{reference}}v,
\]

using finite differences when decisions are discrete. Exclude zero-response pairs from cosine similarity and score them by absolute error.

A powerful nonlinear decoder can manufacture apparent alignment from incidental information. Use preregistered low-capacity probes, random-feature controls, shuffled-target controls, and diagnostic cases where the reference algorithms disagree.

**Failure to align with a named algorithm is not itself a failure of discovery.** Report performance and mechanistic alignment as separate axes.

### 3.7 Pass bars: R14-D0 through R14-D6

| Gate | Proposed pass |
|---|---|
| **R14-D0: task-only integrity** | No algorithm labels, traces, imitation checkpoints, or alignment rewards in training; same v2 schema as R13; firewall and purity checks pass; reward, compute and provenance audits complete |
| **R14-D1: learnable search value** | At \(B=128\), outperform the independently initialized task-trained R12-shaped baseline on paired Mini-ECI with a 95% lower confidence bound above zero; show improvement over the no-search policy; at least four of five training seeds improve |
| **R14-D2: multi-witness correction** | On the frozen ShipHunt/Stratego aggregate, \(B=256\) gives at least 5% relative KL improvement over \(B=128\), with the CI excluding no improvement; neither environment worsens by more than \(0.002\) nats; 12-env \(\Delta\mathrm{KL}_{128\to256}\) has upper 95% bound \(\le0\) |
| **R14-D3: discovery in perfect-information tasks** | Hex/Connect-4 and Sudoku/Equiv-SAT each improve over the R12-shaped baseline on their primary task metric; at least one preregistered MCTS/minimax signature and one CDCL/PNS signature meet the strong-alignment criteria |
| **R14-D4: imperfect-information quality** | In exact-best-response poker fixtures, exploitability is at least 10% below the task-trained R12-shaped baseline when that baseline is meaningfully above zero; otherwise learned excess exploitability is \(\le0.01\); infoset invariance and no-fusion canaries pass; actual payoffs are noninferior |
| **R14-D5: beats fixed algorithms** | At \(B=128\), Mini-ECI exceeds the calibration-selected fixed-algorithm portfolio by at least 2 points, with a paired 95% lower confidence bound above zero; beats that portfolio at matched wall time as well; game performance gains at least 50 Elo with lower confidence bound above zero against the frozen opponent pool |
| **R14-D6: robustness and cost** | Primary gains survive blind \(B=256\), depth and representation-order tests; no verified-witness regression exceeding 1 percentage point and no environment KL regression exceeding \(0.01\) nats against the declared comparator; median latency \(\le2\times\) and peak memory \(\le3\times\) the R12-shaped model at matched \(B\) |

R14-D3 is a **mechanistic-alignment gate**, not a necessary condition for useful task discovery. Report a model that passes performance gates but fails D3 as “successful task-driven search; named-algorithm alignment not established.”

For R14-D2, give every environment’s paired change. A favorable grand mean must not hide continued witness collapse in Stratego or ShipHunt.

### 3.8 Direct tests of the multi-witness failure hypothesis

The R12 regression suggests, but does not prove, single-witness reinforcement. Test that explanation rather than assuming it.

Construct finite instances with known posterior mass over multiple valid hypotheses. Measure:

\[
D_{\mathrm{KL}}(P^*\Vert\mu_B),
\qquad
H(\mu_B),
\qquad
\operatorname{TV}(P^*,\mu_B),
\]

and mass assigned to each true hypothesis class.

Perform three paired interventions:

1. **Duplicate one witness:** repeat its syntactic representation without adding evidence. The posterior should remain unchanged.
2. **Add an alternative valid witness:** posterior mass should change according to its prior and likelihood contribution.
3. **Add evidence disfavoring the first witness:** the model should revise its belief rather than retain a search-history-induced commitment.

For finite hypotheses \(h\),

\[
P(h\mid E)
=
\frac{P(h)P(E\mid h)}
{\sum_{h'}P(h')P(E\mid h')}.
\]

When hypotheses overlap, probabilities of unions cannot be obtained by naive addition; use disjoint assignments, an exact counting representation, or a declared approximation.

Discovering one verified witness proves existence. It does not justify concentrating the posterior on that witness.

---

## 4. Updated society rubric and preregistered forecasts

### 4.1 Rubric definitions

The earlier society dimension definitions were not supplied. The following is an explicit proposed six-dimension rubric; these numbers should not be merged with an existing rubric without mapping its definitions.

Scores run from 0 to 10 and are judgmental evidence summaries, not measured percentages.

- **D1 — Outcome quality:** belief accuracy, witness discovery, game performance.
- **D2 — Algorithmic coverage:** demonstrated representation and execution of useful search families.
- **D3 — Learnability and efficiency:** data, parameter, training-compute, and inference efficiency.
- **D4 — Generalization:** depth, budget, topology, and multi-witness robustness.
- **D5 — Information and proof integrity:** firewall preservation, strategy consistency, sound inference.
- **D6 — Evidence quality:** falsifiability, causal tests, reproducibility, and uncertainty reporting.

“R4 risk-adjusted R12” changes confidence in the architecture’s scope; it does not revise the supplied empirical results.

| Dimension | R3 realized-R12 audit | R4 risk-adjusted R12 without descendant/v2 support | [PRIOR] Proj-R13 | [PRIOR] Proj-R14 |
|---|---:|---:|---:|---:|
| D1 — Outcome quality | 8.0 | 8.0 | 8.0 | 8.7 |
| D2 — Algorithmic coverage | 4.0 | 2.5 | 7.5 | 8.0 |
| D3 — Learnability and efficiency | 8.0 | 7.0 | 7.0 | 7.0 |
| D4 — Generalization | 6.5 | 4.5 | 7.0 | 7.5 |
| D5 — Information and proof integrity | 8.0 | 6.5 | 8.0 | 8.5 |
| D6 — Evidence quality | 7.5 | 6.5 | 8.5 | 8.5 |

Interpretation:

- R12 earns strong outcome and efficiency scores from the supplied results.
- R12-D0 supports important invariants, but does not establish CFR-style strategy consistency or CDCL soundness.
- R13 is not forecast to improve task quality automatically; it primarily improves evidence about representation and learning.
- Projected R13/R14 scores assume the stated protocols are executed. They are not current achievements.
- Approximate judgment uncertainty is ±1 point for supplied-result assessments and ±1.5–2 points for projections.

### 4.2 [PRIOR] R13 imitation accuracy and \(R^2\)

These are subjective **80% predictive intervals**, not confidence intervals. They concern Full-GLCSearch-v2 at \(N_{\mathrm{train}}=2048\), conditional on the corrected primitives being implemented faithfully and a fixed, adequate optimization budget.

Operation accuracy is measured on free-running encountered states against the teacher’s prescribed operation for those states. State \(R^2\) is measured on the separate held-out teacher-state set.

| Family | [PRIOR] IID operation accuracy | [PRIOR] joint-OOD operation accuracy | [PRIOR] IID state \(R^2\) | [PRIOR] OOD state \(R^2\) |
|---|---:|---:|---:|---:|
| Algo 1: MCTS-PUCT | 98–99.9% | 93–99% | 0.985–0.999 | 0.94–0.995 |
| Algo 2: alpha–beta / PNS, weaker subtrack | 93–99% | 75–95% | 0.92–0.995 | 0.65–0.96 |
| Algo 3: CDCL | 88–98% | 65–91% | 0.75–0.96 | 0.40–0.85 |
| Algo 4: MCGS / exact multi-witness inference | 95–99.5% | 83–97% | 0.95–0.998 | 0.82–0.98 |
| Algo 5: CFR+ | 94–99.5% | 82–97% | 0.94–0.995 | 0.78–0.98 |
| Algo 6: Gumbel sequential halving | 98–99.9% | 94–99.5% | 0.98–0.999 | 0.93–0.995 |

For CDCL, numeric \(R^2\) applies only to declared numeric bookkeeping such as backjump levels or propagation counts. It is secondary to soundness, useful clause production, and solve rate; there is no natural \(R^2\) for a clause certificate.

For Algo 4, report graph statistics and posterior targets separately even though the forecast table groups them.

#### [PRIOR] sample-efficiency curve

Forecast ranges for **family-macro IID next-operation accuracy**, weighting each of the six families equally:

| Training bundles per family | Full v2 | v1 | R12-shaped trainable ablation |
|---:|---:|---:|---:|
| 128 | 65–85% | 48–74% | 35–65% |
| 512 | 82–94% | 58–82% | 45–73% |
| 2048 | 94–98.5% | 65–87% | 50–78% |

These aggregate ranges should not conceal family differences. The strongest v1 expectation is MCTS-like aggregation on correctly represented event trees. Its expected weakness is persistent sequential and cross-branch state.

**[PRIOR] Probability that Full v2 passes all R13 gates on the first frozen training protocol: 15–35%.** CDCL trace fidelity, deep sequential execution, and OOD information-set bookkeeping are the main risks. This is compatible with high average operation accuracy.

### 4.3 [PRIOR] R14 Mini-ECI, KL, Elo, and witness gains

Forecasts below are subjective 80% predictive intervals across a completed, correctly implemented R14 campaign. They assume the Mini-ECI and KL protocols remain comparable to the supplied R12 measurements.

| Quantity | [PRIOR] median | [PRIOR] 80% interval |
|---|---:|---:|
| Mini-ECI at \(B=128\), relative to the supplied 106.53 anchor if that anchor uses the same evaluation condition | 109.5 | 104–114 |
| Mini-ECI gain over re-evaluated R12 at \(B=128\) | +3 | −2 to +7 |
| Mini-ECI gain over calibration-selected fixed-algorithm portfolio, matched wall time | +1 | −4 to +5 |
| 12-env mean KL at \(B=128\) | 0.123 | 0.110–0.145 |
| 12-env mean KL at \(B=256\) | 0.118 | 0.102–0.145 |
| Paired mean-KL change, \(256-128\) | −0.004 | −0.012 to +0.008 |
| ShipHunt/Stratego relative KL reduction, \(128\to256\) | 7% | −5% to +20% |
| Elo gain over R12, same opponents and matched evaluation budget | +90 | −60 to +240 |
| Elo gain over fixed-algorithm portfolio, matched wall time | +30 | −100 to +150 |
| Verified-witness discovery gain over R12 at \(B=256\) | +1.5 percentage points | −2 to +5 points |

The supplied context does not unambiguously attach the 106.53 Mini-ECI score to a specific budget. Its use above is explicitly conditional; a fresh paired baseline resolves this before registration is finalized.

Marginal intervals for KL at two budgets do not determine the interval of their paired difference; the difference forecast assumes strongly correlated evaluations on the same instances.

Additional forecasts:

- **[PRIOR] Probability of fixing the multi-witness regression under R14-D2:** 55–75%.
- **[PRIOR] Probability of outperforming the fixed-algorithm portfolio on both Mini-ECI and matched wall time:** 25–45%.
- **[PRIOR] Probability of strong mechanistic alignment in at least two task groups:** 50–75%.
- **[PRIOR] Probability of strong alignment in every named task group:** 10–30%.
- **[PRIOR] Probability of passing every R14 gate in the first frozen campaign:** 5–20%.

These are deliberately more conservative than the architecture’s representational promise. Task-reward credit assignment, optimization, and inference overhead can prevent an expressive system from discovering an efficient algorithm.

### 4.4 Decision rule after R13 and R14

| Result | Supported conclusion | Next action |
|---|---|---|
| Constructive capability fails | Architecture or primitive semantics remain insufficient | Repair the smallest failing computation |
| Constructive capability passes; imitation fails | Learning or optimization remains unresolved | Diagnose local transitions and credit before scaling |
| Teacher-forced imitation passes; free-running imitation fails | Small errors accumulate through execution | Improve state robustness and recurrent training |
| R13 passes; R14 fails | Known algorithms are learnable with supervision; task-only discovery remains unproven | Work on task distributions and outcome credit |
| R14 improves tasks without named alignment | Useful task-driven search discovered; reference-algorithm identity unproven | Analyze the learned mechanism without forcing a label |
| R14 improves tasks and passes causal probes | Evidence supports both useful discovery and specific algorithmic mechanisms | Expand budgets, environments, and independent replication |

The immediate authorized deliverable is the specification above; no implementation or experiment has been performed in this response. The next bounded step is to freeze corrected v2 semantics and the adversarial teacher fixtures before launching R13.