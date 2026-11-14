# arXiv 2606.21173, "Inverting the Bellman Equation: From Q-Values to World Models": deep read and relation to our agenda

Bead pool-dce2y0. Written 2026-11-07. Labels used throughout:
- **PROVEN**: proved in the paper (section or theorem given).
- **MEASURED**: the paper's own experiments.
- **MEASURED-by-us**: a check I ran on CPU today (scripts under `/home/agent/workspaces/mats/work/arxiv-2606.21173/scripts/`, quoted below).
- **OUR READING**: my interpretation. Not established.

## TL;DR

1. **Paper (PROVEN):** with enough reward functions (one generic goal for deterministic finite MDPs; |G| ≥ |S| for stochastic ones), an agent's Q-values plus rewards and policy uniquely determine the transition kernel. P-learning is least squares on the Bellman residual with Q fixed. A single reward never suffices, even with all policies and discounts (Prop. 11).
2. **Critic:** "encodes a world model" means "determines one through the Bellman equation". Uniqueness holds for almost any Q-table (Prop. 9). Single-goal recovery needs Q precision that shrinks like |S|^−1.8 (MEASURED-by-us).
3. **Experiments overstated (MEASURED-by-us):** under the paper's own metric, the MountainCar WM (6.7e-3) is only 1.7x better than "nothing moves" and 24x worse than a model that ignores the action. "Reacher" is gymnax's linear double integrator, not MuJoCo. Reacher's WM is genuinely good (14x better than ignoring the action).
4. **For us:** the strongest link is moderate, to the reasoner/searcher design. Thm 1 + Prop. 11 make "reward training moves beliefs" exact: along the value direction, toward high-value states, which supports "beliefs trained only on data". ARC and aggregation links are weak or analogy; Löb has none. Nothing of ours is contradicted.
5. [removed]

## 1. The paper in brief

[removed]

**Question.** Verbatim: "*When do model-free agents implicitly encode an accurate model of their environment?*" (Sec. 1).

**Core move.** For a fixed policy, the Bellman equation reads Q(s,a,g) = E_{s'~P(s,a)}[M(s',g)] with M(s',g) = r(s',g) + γV(s',g) (Eq. 1). If Q, π and r are known, this is a *linear system in P*: for each (s,a), M·P(s,a) = Q(s,a), where M is a |G|×|S| matrix built from the agent's own values (Eq. 2). "P-learning" is gradient descent on the Bellman residual over the model parameters with Q held fixed, the mirror image of Q-learning (Sec. 3, Alg. 1).

**What is proven (PROVEN).**
- Thm 1 (Sec. 3, proof App. A.2): tabular P-learning converges to P∞ = M⁺Q + (I − M⁺M)P₀. Verbatim: "when M is rank-deficient (M⁺M ≠ I), multiple WMs yield identical Q-values, and P-learning selects among them based on the initial guess P₀." This is the standard least-squares/pseudo-inverse result.
- Thm 2 (deterministic, finite S): for Lebesgue-almost every reward, even **one goal** makes M column-injective, so P is unique. Recovery from ε-approximate Q is exact iff ε is below about half the minimum column gap ("any estimator fails ... for ε ≥ gap/2").
- Thm 3 (stochastic, finite S): with |G| ≥ |S|, M is full rank for generic rewards (or for generic policies or discounts if the goals span S). Error bound ‖P̂ − P‖₁ ≤ ‖M⁺‖₁(1+γm)ε, with a matching Ω(ε) lower bound. ‖M⁺‖₁ is not bounded in general. The clean O(ε) bound (4ε/(1−2ε)) holds only for an unconditional policy with indicator goals G = S (Prop. 6, App. C.3.2).
- Prop. 11 (App. C.7, multi-horizon impossibility): one goal plus *all* policies and *all* discount factors still cannot identify P in general (n ≥ 4). Breaking value equivalence needs multiple **reward functions**.
- Local MDPs (Thm 5, App. C.8): N-local kernels need N−1 generic goals if the support is known, and 2N−1 if it is not.
- Continuous S (App. D): identifiability becomes "measure determining" (stochastic) or "point separating" (deterministic). Stochastic results (Thms 6, 7, App. D.3) hold **only for unconditional policies**. Deterministic results: indicator goals covering S ⊕ B_σ with termination (Thm 4), a finite goal set with Gaussian rewards and termination (Thm 8), or 2d+1 generic goals when V^π is real-analytic (Thm 9). Thm 9 also assumes an unconditional policy, a condition the main text's summary sentence leaves out (Sec. 4.2 vs. App. D). Verbatim limitation: "The challenging intersection of stochastic kernels and general (goal-conditioned) policies is beyond the scope of this paper."
- Props. 9, 10 (App. C.6, "arbitrary Q-values"): for Lebesgue-almost every Q, *including Q-tables that are not the values of any policy in the true MDP*, there is a unique Bellman-residual minimiser P* = M⁺Q (|G| ≥ |S|). Verbatim: "there is no guarantee the extracted world model will approximate the true kernel." The minimiser may be a signed "pseudo-world model" off the simplex (App. B.2).

**What is measured (MEASURED).** PQN + HER agents, 10 seeds, then P-learning on the frozen Q.
- FourRooms (68 states; App. F): deterministic, 1 noise-perturbed goal gives exact recovery (column matching). Windy (local), 4 goals plus a *given* 4-neighbour support prior come within ~1% of optimal on unseen goals. Teleporting, 20 goals come within ~0.1%, below the |G| = |S| = 68 that the theory needs.
- "Reacher", 4 fingertip goals: Q NMSE 5.7e-1 but WM NMSE 1.2e-4. Value iteration inside the extracted WM is near-optimal on 3 unseen goals (far fingertip, target angle, target velocity), e.g. 0.647 vs 0.652 on target velocity.
- MountainCar, 4 position goals: WM NMSE 6.7e-3. A second agent trained on 4 velocity goals gives a WM 1.7e-4 from the first and 7.2e-3 from the truth, which led to the informal "local-global hypothesis" (Sec. 5.2, labelled by the authors as a hypothesis "left for future exploration").
- A 42-architecture Reacher sweep: Spearman ρ = −0.98 between agent return and WM error.

**Limitations the authors state (Sec. 7).** No POMDPs. No agents that do not learn Q. No large finite environments. Stochastic continuous results only for unconditional policies. "P-learning is not a substitute for model-based RL."

### 1.1 What the paper shows, judged against its own definitions (critic pass)

**(a) "Encode" means "determines through the Bellman equation", not "represents".** OUR READING, grounded in PROVEN statements. The headline "value-based agents ... implicitly encode a unique and accurate world model" (abstract) splits into two separate claims:
- *Uniqueness* is close to free. Prop. 9 says almost every Q-table, even a wrong one, determines a unique P* once |G| ≥ |S|. That is a dimension count (|G| numbers per (s,a) against |S|−1 unknowns) plus genericity. It carries no information about the agent.
- *Accuracy* is inherited from the accuracy of Q (Thm 3c; bound scales with ‖M⁺‖₁).

Nothing in the paper shows that the network computes, stores or uses P internally. The model is a property of the input-output function Q(s,a,g) together with an external solver. The deterministic single-goal case makes this plain. Q(s,a) = r(s') + γV(s'), so if the map s' ↦ r(s')+γV(s') is injective you can read s' off by lookup ("column matching", Sec. 4.1). The "world model" is the value function inverted.

**(b) Precision cost: the single-goal result needs Q precision that shrinks faster than 1/|S|.** PROVEN plus MEASURED-by-us. Thm 2b needs ε < gap/2. With one goal, the |S| column values lie in a bounded range, so by pigeonhole the minimum gap is at most range/(|S|−1) (elementary). I measured the actual gap in deterministic k×k gridworlds with the paper's own noise-perturbed single-goal reward (App. F.2: r = δ_g − ξ, ξ ~ U[0,1]), γ = 0.99, optimal policy, 20 draws per size (`/home/agent/workspaces/mats/work/arxiv-2606.21173/scripts/single_goal_gap.py`):

| \|S\| | 16 | 64 | 256 | 1024 | 4096 |
|---|---|---|---|---|---|
| median gap(M) | 1.9e-2 | 1.4e-3 | 1.3e-4 | 2.1e-5 | 8.5e-7 |
| pigeonhole ceiling range/(\|S\|−1) | 4.8e-1 | 1.1e-1 | 4.9e-2 | 2.8e-2 | 1.2e-2 |

The median gap falls about as |S|^−1.8, which is what you expect for |S| near-random values in a fixed range (minimum spacing ~ range/|S|²). At |S| = 4096, exact single-goal recovery needs Q correct to about 4e-7, on values of order 10-100. In plain words, the identity of the next state sits in the low-order digits of one real number per (s,a). For FourRooms (|S| = 68) the median gap is about 1e-3, which explains why the paper's single-goal FourRooms result works. "One goal suffices" is a measure-zero-exception statement; at finite precision, the number of goals needed grows with |S|. This is the bit-accounting point in §2a.

**(c) The continuous-control fidelity numbers are inflated by the metric. MEASURED-by-us.** The code (`eval/world_model.py` L157-168) defines WM NMSE as MSE / Var(s'_d) per (action, dimension), averaged. It normalises by the variance of the next *state*, not of the state *change*. The WM is a residual net P(s,a) = s + h(s,a) initialised near the identity (App. E, Tables 3-4: "init scale 0.01"), so most of that variance comes free. I computed the same metric for simple wrong models, with the dynamics re-implemented in numpy from gymnax 0.0.9 (scripts `/home/agent/workspaces/mats/work/arxiv-2606.21173/scripts/identity_baseline.py`, `mc_wrong_models.py`, `reacher_wrong_models.py`; seed 42; 2.5e5 uniform states for MountainCar, 2e5 reset states for Reacher):

| Model scored with the paper's metric | MountainCar Avg NMSE | Reacher Avg NMSE |
|---|---|---|
| Paper's extracted WM | **6.7e-3** (pos. goals), 7.2e-3 (vel. goals) | **1.2e-4** |
| Identity, s' = s ("nothing moves") | 1.13e-2 | 2.03e-3 |
| True dynamics but **ignores the action** | **2.8e-4** | 1.67e-3 |
| True dynamics but ignores gravity | 1.3e-3 | n/a |
| True dynamics but no position walls (inelastic left wall) | **6.58e-3** | n/a |

- **MountainCar:** the reported WM is only 1.7× better than "nothing moves". It is 24× *worse* than a model that ignores the action, and 5× worse than one that ignores gravity. A model that is exact everywhere except at the walls (about 2% of uniformly sampled states hit a wall in one step) scores 6.58e-3, almost exactly the reported 6.7e-3. So the reported number cannot tell "accurate model with a wrong wall" from "poor model". It does not support "match the true transition kernel with high fidelity" (Sec. 5.2) on its own. OUR READING: the two MountainCar WMs agreeing with each other 42× more closely than with the truth fits a *shared* error at the walls or in under-determined directions. Thm 1's (I − M⁺M)P₀ term says unidentified directions keep the initial guess, and here both runs start near the identity. That explains the agreement at least as well as the "platonic" reading. I did not test it, because that needs their trained checkpoints (GPU).
- **Reacher:** the WM is 14× better than ignoring the action and 17× better than the identity, so here it captures real action effects. But "Reacher" in the code is **gymnax's `misc/reacher`**, a decoupled, frictionless double integrator (ω' = ω + 0.05·τ, θ' = θ + 0.05·ω', τ ∈ {−1,0,1}²; verified in the gymnax 0.0.9 wheel), not MuJoCo Reacher as Sec. 5 says ("MuJoCo Reacher [Todorov2012]"). The code docstring only says the observation *layout* matches MuJoCo Playground. Transfer to velocity goals is much less surprising when ω' is linear in the action. The comparison "Q NMSE 0.57 vs WM NMSE 1.2e-4" sets two differently normalised quantities side by side. It is not evidence that the WM is "more accurate than" Q.

**(d) The "hidden generalisation capabilities" come from the planner we add.** OUR READING. The near-optimal returns on unseen goals come from *value iteration on a 50- or 200-point grid inside the extracted model with a reward we supply* (App. E). The agent's own Q is never queried on those goals. What is shown is that Q contains enough information for an external planner. That is not a capability of the agent.

**(e) ρ = −0.98 is expected from the construction.** OUR READING. The extracted WM is a deterministic function of Q, and both return and WM error improve as Q gets more accurate with capacity. The correlation does not show that the agent uses a model.

**(f) What stands.** The finite-MDP identifiability theorems, the multi-horizon impossibility (Prop. 11), the clean statement that P-learning is least squares with an initialisation-dependent null-space component (Thm 1), and the FourRooms results, where the evaluation is exact and tabular. Those are solid but modest. The step from "uniquely determined by Q" to "the agent has a world model" is definitional, not empirical.

**Relation to the closest prior work (read).**
- Richens, Abel, Bellot, Everitt, "General agents contain world models" (ICML 2025, arXiv 2506.01622): "any agent capable of generalizing to multi-step goal-directed tasks must have learned a predictive model", extracted "from the agent's policy". It uses policies and multi-step goals. This paper uses Q-values and single-step goals and gets much smaller goal counts, at the price of needing *values* rather than behaviour (Sec. 6, "they do not address the more realistic setting of single-step goals").
- Grimm et al. 2020, value equivalence: the obstacle this paper breaks with multiple rewards.
- Richardson et al., "Local Inconsistency Resolution" (AISTATS 2026, arXiv 2604.17140). This paper cites it in Sec. 6: "RL, Q-learning, and P-learning can all be thought of as different resolutions to the same inconsistency (the Bellman residual) between a world model P, action-values Q, and rewards r -- an instance of a general pattern that also has analogues in decision theory and supervised learning [richardson_lir_26]." The LIR abstract lists EM, belief propagation, adversarial training, GANs and GFlowNets as instances and does not mention RL or Bellman equations. Oliver Richardson is a co-author of both papers.
- Follow-ups or critiques (web search 2026-11-07): none substantive. One citation in passing, Gonzalez Laiz & Schneider, "Understanding the Limits of Latent World Models" (arXiv 2610.37588, 27 Oct 2026), which mentions that predictive information "might be encoded implicitly in policy and value networks". There is a student thesis repo (MickWPM/ai-thesis) applying it to FourRooms, and a Pith "desk verdict" with no content. No OpenReview page found.

## 2. Relation to each agenda thread

How I mapped the agenda: `recall` (general reasoner, LIR, searcher, estimation ARC, MATS agenda, research agenda tentative), `cass search`, the hub wiki pages (recall copies), the MATS outbox, the ToC deck, and mats.ekelvin.com (fetched 2026-11-07). Path keys: OUT = `/home/agent/workspaces/mats/outbox/`, CJ = `/home/agent/workspaces/mats/work/bubbles/CONJECTURES.md`. Wiki line numbers are file lines of the hub page.

**Two corrections to the brief.**
- No file in our workspace defines a "general reasoner agenda". The closest written statements are Kelvin's ToC abstract (10-17) and the 10-19 orientation note (OUT/research-orientation-2026-10-19.md).
- Apart from this report, nothing in our files mentions goal-conditioned agents, Q-learning, Richens, or the claim that agents encode world models. So no claim of ours is directly about this paper's subject.

### 2a. ARC: estimating the mean of a random deep MLP (sampling vs. deduction, bit accounting)

- **What the paper gives:** no technique we can use, and no theorem about estimating means. Its tools are standard: finite-dimensional linear algebra plus genericity (determinants are non-zero polynomials), and measure-determining test families via analyticity and Fourier transforms (App. D.5). Our cumulant and moment-matching work (OUT/cumulant-ambiguity-2026-11-03.md, OUT/moment-matching-2026-11-03.md) faces a different problem: estimating moments of a pushforward under compute limits, not identifying a distribution from exact expectations.
- **The one transferable lesson (methodological, weak):** "uniquely determined from exact values" is a different statement from "recoverable at the precision available". §1.1(b) measures this. With one goal the minimum gap shrinks like |S|^−1.8, so the information sits in low-order digits. That matches the discipline we already use in bit accounting (OUT/bit-accounting-2026-11-02.md L85-95: code costs measured in bits rather than identifiability claims). It gives no new result for ARC.
- **Contradicts or supports our claims?** Neither. The paper's "test functions determine the distribution" (App. B.1, Vandermonde) is a loose analogy to moment matching, and I would not lean on it.
- **Next step:** none.

### 2b. The "general reasoner" / learned Searcher (LIR, PDG energy, world model kept separate from the searcher)

Our design, verbatim:
- The world model is "**fixed, ignorant of the searcher**" (wiki `inconsistency-searcher-and-world-model-design` L28).
- "For one player, minimising the PDG energy (KL to the prior plus the utility edge, dynamics fixed) gives the soft Bellman condition" (same page L32).
- ToC deck threat model: "Training for reward moves what the model believes" (work/toc-slides/index.html L161).
- The design answer on the deck: "Beliefs are trained only on data, and the goal is a single utility term we can inspect" (index.html L240).
- Kelvin's hope: "The Searcher does not succeed by fooling the Predictor" (wiki `lir-alignment-hope-and-cruxes` L22).
- From the orientation note: "Decomposition is the point: learning world models vs searching/planning" (OUT/research-orientation-2026-10-19.md L9).

What the paper gives, from strongest to weakest:

1. **A theorem-backed account of how a value-driven update moves beliefs, and why a single utility leaves beliefs underdetermined. Strength: MODERATE, the strongest link.**
   - The paper itself frames Q-learning and P-learning as two ways to resolve one inconsistency, the Bellman residual between P, Q and r. They differ in which variable is under control, and the paper cites LIR for this (Sec. 6). That is exactly our PDG/LIR vocabulary: same inconsistency, different focus and control.
   - Our design puts P out of the searcher's control. The paper's results say what happens if it is not:
     - **Thm 1 (PROVEN):** gradient descent on the Bellman residual over P moves P only inside the row space of M (the span of the value probes r_g + γV_g). The component in the null space stays at P₀.
     - **Prop. 11 (PROVEN):** with a **single** reward function, P is not identified even from the Q-values of *all* policies at *all* discounts.
     - **Our derivation, one line from Eq. 3:** with one utility, M is a single row mᵀ, and the update is ΔP(s,a) = α·m·(Q(s,a) − mᵀP(s,a)). If the searcher's Q is optimistic (Q > mᵀP), probability mass moves toward successor states with higher r + γV (after projection onto the simplex). That is wishful thinking: belief shifts toward what is valued. Every value-equivalent direction is left to the initial guess or the regulariser.
   - So in the linear, tabular case, "training for reward moves what the model believes" has an exact form. The movement happens only along the value direction, and its sign follows the value error.
   - This **supports** the deck's design choice (index.html L240, beliefs trained only on data). It does not prove the choice suffices: our searcher can still be biased in *which* low-energy μ it finds. That is the open crux C4 (CJ L19: "search target should depend only on world model and utility, not on the search policy", untested).
   - Scope: OUR READING that this transfers to the PDG energy. The paper's result is for the plain Bellman residual with fixed π. Our energy uses a soft best response, a softmax over Q_j (wiki L29), which changes M through π but not the linear structure in P.
2. **A possible diagnostic for "the Searcher fools the Predictor". Strength: WEAK-to-MODERATE. Conditional, not ready.**
   - Props. 9-10 say any Q-table on enough goals implies a unique P* = M⁺Q. So a learned searcher's values carry an *implied* world model. The distance between that P* and the explicit predictor P is a well-defined inconsistency: nonzero exactly when the searcher's values are not the values of the model it claims to plan with.
   - The catch: by Prop. 11 this needs values for **many** reward functions (|G| ≥ |S| for stochastic, or N−1 for an N-local MDP with known support, Thm 5). Our searchers optimise a single utility, so the implied model is not identified. The diagnostic only applies if we query the searcher on many utilities, for example in a small gridworld or ShipHunt toy.
   - I have not tested this.
3. **The "general agent contains a world model" result (Richens et al. 2025; this paper). Strength: WEAK for our bet.**
   - Our decomposition bet is about *training and architecture*: learn the world model from data, and search separately.
   - The paper shows only that a competent multi-goal model-free agent's Q *determines* a model. It does not show that the agent factors internally (§1.1a).
   - So the paper neither supports nor undermines "decomposition is the point". It does suggest that "no world model" is not a meaningful property of a competent multi-goal agent, which is consistent with our "simulator definitely not; the searcher must search over a learned world model" (orientation L15).
4. **"Compute = inversion"** (Kelvin, 10-13, session transcript): P-learning inverts the *value map*, not the generative model. The shared word "inversion" is an analogy only, and I would not draw on it.

**Contradicts any of our claims?** No. The paper is consistent with wiki L32 (single player: energy minimum = soft Bellman).

### 2c. In-context dedup and aggregation (rich messages vs. scalar reports)

- Our claims:
  - "forecasts alone cannot reach Bayes across arbitrary structures ... 'keys need provenance' is the expected result" (OUT/agg-literature-map-2026-10-21.md L76).
  - C8 is "supported in part: near-Bayes with source keys; not met blind" (CJ L23).
  - Standing preference: aggregation agents send high-dimensional messages, and scalar reports are only a baseline.
- **What the paper gives: an analogy, WEAK.** Value equivalence is a clean instance of "one scalar summary per (s,a) underdetermines the hidden distribution, while |S| linearly independent summaries determine it" (Thm 3, Prop. 11). That is the same shape as "scalar forecasts lose information; rich messages can keep it". But the objects differ. The paper recovers a *known-form kernel* from *exact expectations*. Our aggregation recovers a *posterior* from agents with *overlapping evidence*, and the hard part there is double-counting (C9, dedup), which the paper has nothing to say about.
- **Contradicts or supports?** Neither, beyond the analogy.
- **Next step:** none.

### 2d. Löbian / modal cooperation (PrudentBot, energy-prudentbot)

No relation. The paper is single-agent MDPs with no proof-based or modal content. I found nothing to connect, and I am not stretching for one.

### 2e. G5, in-context Bayes in small transformers

No relation beyond the generic idea of reading an implied model off a trained network's input-output function. That is not specific enough to act on.

### 2f. People: a context fact, not research

- [removed]
- [removed]
- So this paper is the most concrete existing bridge between LIR and RL that also has a reachable MATS author.

## 3. Cruxes this moves

| Crux / claim (ours) | Effect | Basis |
|---|---|---|
| Threat model "training for reward moves what the model believes" (index.html L161), and the response "beliefs trained only on data" (L240) | **Sharpened and supported** in the linear, tabular case. If P is a controlled variable in a single-utility Bellman inconsistency, it moves only along the value direction, toward high-value successors when Q is optimistic, and stays underdetermined elsewhere | Thm 1, Prop. 11 (PROVEN); the wishful-update sign is our one-line derivation from Eq. 3; transfer to the PDG energy is OUR READING |
| C4: search target depends only on world model and utility (CJ L19, untested) | **Not moved.** Prop. 11 only says a single utility cannot pin the model; it says nothing about search-policy dependence | n/a |
| "Decomposition is the point" (orientation L9) | **Not moved.** "Q determines P" is informational, not architectural | §1.1(a) |
| ARC, aggregation, Löb cruxes | **Not moved** | §2a, 2c, 2d |

## 4. Recommended next steps (at most 3, all bounded)

1. **Record the Thm 1 / Prop. 11 point on the design wiki page** (`inconsistency-searcher-and-world-model-design`), as a labelled paragraph (PROVEN in 2606.21173, transfer = inferred) under the world-model/searcher separation. One hub PR, about 20 minutes. This is the one finding with lasting value for the reasoner thread.
2. [removed]
   - (i) the MountainCar NMSE calibration (identity 1.13e-2, ignores-action 2.8e-4, no-walls 6.58e-3 vs. reported 6.7e-3), and a request for numbers normalised by the variance of the state change;
   - (ii) the single-utility null-space point as the LIR reading of their Sec. 6 sentence.

   Sending as Kelvin needs his OK. I can draft it in 10 minutes if he wants it.
3. **Only if (2) goes ahead:** reproduce MountainCar on CPU with their repo (`uv sync`, JAX CPU, 1 seed). The repo says "~minutes" on an H100; I estimate under 1 h at nice 19. That gives the delta-normalised error and tests whether the two WMs share their wall error (the alternative to the "platonic" reading). Not worth doing for our agenda alone.

Not recommended: building P-learning into our experiments. Our searchers have one utility, so by Prop. 11 the implied model is not identified (§2b, item 2).

## 5. Sources read

- The paper: full TeX source from arxiv.org/e-print/2606.21173 (main.tex and all 12 section files, including the appendices), and the PDF (48 pp.) for theorem numbering. Abstract-page metadata.
- Code: github.com/aletcher/inverting-bellman (shallow clone 2026-11-07): `eval/world_model.py` (metric), `configs/reacher.py`, `configs/mountaincar-position.py`, `envs/reacher.py`, `envs/samplers.py`, `envs/env_dynamics.py`, `uv.lock` (gymnax 0.0.9), README.
- gymnax 0.0.9 wheel from PyPI: `misc/reacher.py`, `classic_control/mountain_car.py`.
- Prior work and follow-ups: Richens, Abel, Bellot, Everitt, arXiv 2506.01622 (ICML 2025), abstract page; Richardson et al., LIR, arXiv 2604.17140 (AISTATS 2026), abstract page; Gonzalez Laiz & Schneider, arXiv 2610.37588 (its single citation of the paper); web searches for critiques, OpenReview and follow-ups (none substantive). Grimm et al. 2020 and Adamczyk 2025 only through the paper's own account (not re-read).
- Our side: as listed in §2 (recall outputs and wiki copies in `/tmp/agenda-map/`, mapped by a sub-agent; I re-checked the cited lines).
- MEASURED-by-us scripts (numpy, CPU, nice 19, seconds each): `/home/agent/workspaces/mats/work/arxiv-2606.21173/scripts/identity_baseline.py`, `mc_wrong_models.py`, `reacher_wrong_models.py`, `single_goal_gap.py`.

## Correction (2026-11-08, after Kelvin's question)
The summary and §3 overstated the goal count. |G| ≥ |S| is the worst case (dense stochastic kernels). For N-local kernels Thm 5 (PROVEN) needs only N−1 generic goals with known support, 2N−1 without; Windy reached ~1% of optimal on unseen goals with 4 goals plus a given 4-neighbour support (MEASURED); Teleporting reached ~0.1% with 20 goals, below |S| = 68, without a theorem covering it (MEASURED). Consequence: the "searcher fools the predictor" diagnostic needs roughly one utility per possible successor, not |S|, so it is cheaper than stated (moderate rather than weak), but still needs several utilities and a known locality structure.
