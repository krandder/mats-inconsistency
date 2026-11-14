# Dark room and white noise as PDG inconsistency of the agent (key: pdg-inconsistency-of-the-agent)

Script: `math/darkroom/pdg-inconsistency-of-the-agent/toy.py`. Output: `math/darkroom/pdg-inconsistency-of-the-agent/output.txt` (numpy/scipy, about 25 min).

## Verdict in one paragraph

Taken literally, the PDG framing (the agent is a PDG made of its constitution, its physics model and its perception edges, and free energy is its inconsistency) is a surprise minimiser. It avoids white noise without any parameter, but it prefers the dark room whenever darkness is not physically more dangerous. In the toy it picks the dark room under every hazard setting at β_phys ≤ 10. It switches only when the dark room's extra hazard, multiplied by a physics-to-perception exchange rate β_phys > 48, outweighs the 0.43 nats/step of the learnable stream. So the dark room is **not solved**.

The constitution part works as hoped. If breaking a constraint stops the computation, the constraint is a deterministic edge, and for any β > 0 its confidence makes no difference (proved; checked in A1). It is not a fitted parameter, but it does not remove the weight either. The weight moves to the agent's physics edge.

The LIR reading ("attend to the inconsistency that control can reduce") divides the expected inconsistency of each perception edge into a reducible part, KL(law ‖ model), and an irreducible part, H(law). Darkness and noise differ only in the irreducible part, and that part has zero gradient with respect to every belief. Drop it and neither room is attractive or aversive on account of its entropy. They become the same kind of room, one whose law is simple. Whether the agent then stops attending to both depends on the prior: it has to contain "simple law" hypotheses (uniform i.i.d., constant), as an Occam or Solomonoff prior does. A flat parametric prior gives each of the two rooms a permanent share of attention (13% and 17%). Verdict for the candidate with LIR (author): **both rooms are solved with a parameter**. The parameter is the Occam prior, which is structural: any positive atom mass gives a vanishing share as T → ∞, but the mass sets how long that takes. When room costs differ, a second hidden parameter appears, the exchange rate between model nats and survival nats (the model's use horizon W). **Skeptic downgrade (see Skeptic section): white noise under LIR is not solved.** The Occam atom dismisses the noise room only when the noise law is exactly the atom (uniform). Generic i.i.d. noise keeps a permanent share of attention, and the Dirichlet concentration a0 decides whether noise or structure gets most of it.

## 1. Framing in plain language

The agent is a set of beliefs, each held with a confidence. Some beliefs are its constitution: "I am alive", "my energy is above zero". These are not preferences. They are conditions that hold whenever the agent runs, because when they fail the computation stops. Others are its model of physics: "in this room I survive a step with probability 1 − h_r". Others are perception: "the next symbol has distribution q_θ(·|past)", together with the observed symbol itself. The agent's "free energy" is how inconsistent this set is (Richardson 2022 sec. 7: ⟨⟨M_Ψ⟩⟩₁ = −log Z_Ψ, local `2202.txt` l.993-1020; https://arxiv.org/abs/2202.11862). It acts and learns to reduce that inconsistency. LIR (Richardson et al. 2026, local `lir2026.txt` l.262-300; https://arxiv.org/abs/2604.17140) adds attention φ (which edges are looked at) and control χ (which parameters may move): θ ← exp_θ(−χ ⊙ ∇_θ ⟨⟨φ ⊙ M(θ)⟩⟩).

## 2. Formal objective

The variables at each tick are the room r_t (under control), the observation O_t and Alive_t. The edges are:

- the constitution δ(Alive_t = 1), with confidence β_c;
- the agent's physics model p(Alive_t | r_t) = Bern(1 − h_r), with confidence β_phys;
- the perception model q_θ(O_t | O_{<t}, r_t), with β = 1, plus the observation δ(O_t = o_t).

The inconsistency (Richardson 2022 eq. (1)) is Inc(μ) = Σ_L β_L E_μ KL(μ(Y|X) ‖ p_L(Y|X)).

**O1 (the candidate as stated).** J₁(policy) = Σ_t [ −log q_θ(o_t | o_{<t}) + β_phys · (−log(1 − h_{r_t})) ].
Derivation: the observation edge pins μ to δ_o, so the perception edge contributes the surprisal (Prop. 2 of Richardson 2022). The constitution edge pins Alive = 1, so the physics edge contributes β_phys·(−log(1−h)). With γ = 1 nothing changes, because −H(μ) = 0 on pinned variables (A5).

**O2 (LIR reading).** Take expectations over the law: E[−log q(o)] = KL(law ‖ q) + H(law). The H(law) term is the same for every θ, so the control χ on θ receives zero gradient from it. "Attend to what control can reduce" therefore keeps KL(law ‖ q_θ) only. The agent does not know the law; its Bayesian estimate of the reducible part of the next step is
  g_r = H(q) − E_posterior[H(p_θ)] = I(θ, model; O_next | past)
(the mutual information, i.e. expected information gain / BALD). The room choice is made by maximising W·g_r − (−log(1 − h_r)), where W is how many future ticks the improved model is used.

## 3. Results

**A. Constitution edges (theorem plus check).** Let c = δ(Alive=1) with any confidence β_c > 0. Then KL(μ‖δ) = ∞ unless μ = δ, so ⟨⟨{c, p_phys}⟩⟩ = β_phys · (−log(1 − h)), whatever β_c is. In A1, β_c from 0.01 to 1000 always gives 0.010050 = −log 0.99. So "the computation stops when it is violated" gives infinite confidence for free, and that confidence turns out to be irrelevant.

Two things show this depends on the edge being deterministic. First, a soft constitution ("alive w.p. 0.999") makes β_c decide the value (A2: 0.00014 to 0.0067). Second, giving the physics edge infinite confidence as well makes the inconsistency +∞ for every h > 0 (A4), so no room can be ranked. The confidence that actually matters is β_phys relative to the perception edges, and it is an exchange rate, not a unit. The "one report per edge per tick" reading (β = 1 for both) fixes it by convention. That convention ties it to the sensor sample rate: doubling the sensor rate doubles the weight of sensory surprise.

**B. Three rooms, each on its own** (K = 4 symbols, T = 2000, 40 seeds). The rooms are dark (constant 0), noise (i.i.d. uniform) and struct (order-1 Markov, next = prev+1 w.p. 0.9, entropy rate 0.435 nats). The learner is a Bayes mixture of Markov orders 0-2 with KT Dirichlet(½) priors (flat), optionally with uniform/constant atoms added (Occam).

| learner | room | sample inc S | reducible R = ΣKL(law‖q) | irreducible ΣH(law) | agent-estimated ΣG | growth of R per doubling of T (×log 2) |
|---|---|---|---|---|---|---|
| flat | dark | 12.8 | 12.8 | 0 | 8.2 | 1.5 |
| flat | noise | 2783.0 | 10.7 | 2772.6 | 21.5 | 1.6 |
| flat | struct | 891.9 | 31.1 | 869.9 | 30.3 | 6.4 |
| Occam (atoms 0.5) | dark | 2.8 | 2.8 | 0 | 2.4 | 0.0 |
| Occam (atoms 0.5) | noise | 2774.0 | 1.3 | 2772.6 | 6.8 | 0.0 |
| Occam (atoms 0.5) | struct | 892.6 | 31.8 | 869.9 | 29.4 | 6.4 |

S = R + I holds within Monte Carlo error (|residual| ≤ 9 against an s.d. of about 6). Under O1 nearly all of the gap between rooms comes from the irreducible term. In reducible terms, dark and noise are the same kind of room. With a flat prior both grow like (K−1)/2·log T = 1.5·log T, as any parametric context does. With Occam atoms both are bounded. Struct grows at 4 contexts × 1.5 = 6 (measured 6.4). This is the finite form of Solomonoff's bound: the total expected KL of the universal mixture against a computable law μ is ≤ K(μ) ln 2 (Hutter 2007, eq. (13), https://arxiv.org/abs/0709.1516, read in the fetched PDF). The amount left to learn in a room is the complexity of its law, not of its samples. A constant and a fair coin are both short laws.

**C1. O1, static choice of room over the horizon:** the dark room wins in every one of the 18 settings (two learners × three hazard scenarios × β_phys ∈ {0.1, 1, 10}). In the "dark starves" scenario (h_dark = 0.01 against 0.001), struct beats dark only if β_phys > 48.6 (flat) or 49.2 (Occam). That threshold is decided by the exchange rate.

**C2. Sequential choice, rooms unlabelled, 10 forced visits each, T = 1500, 20 seeds** (shares are [dark, noise, struct]):
- O1, β_phys = 1, any hazard scenario: [0.987, 0.007, 0.007], i.e. the dark room. With β_phys = 100 and dark starving, the flat learner moves to struct. The Occam learner does not, because a myopic O1 never returns to struct to improve its model of it.
- LIR, equal hazards: flat [0.13, 0.18, 0.69]; Occam (atoms 0.5) [0.017, 0.033, 0.95]; Occam with atoms 1e-6 [0.13, 0.18, 0.69], so a tiny atom mass behaves like the flat prior at this horizon.
- LIR with unequal hazards: with W = 1 (myopic), the dark-room share falls from 0.13 to 0.05 under "dark starves". With W = 1000 (≈ 1/h, the expected remaining life) the hazards hardly matter. Here W decides the outcome.

**D. Long run (LIR, equal hazards, T = 20000, shares per quarter of the run):** flat stays at about [0.13, 0.17, 0.70] in every quarter, so the noise and dark rooms are never dismissed. Their shares track parameter counts (3 : 3 : 12, adjusted by the dark room's faster KT convergence). With atoms 0.5 the shares fall to [0.002, 0.0001, 0.998]. With atoms 1e-6 the noise share only starts to fall (0.176 → 0.165), as expected: the posterior odds of the uniform atom grow like n^{(K−1)/2} = n^{1.5}, so they overcome 10⁻⁶ only after about 10⁴ noise visits.

## 4. Verdict per room

| | O1: inconsistency of the agent PDG | O2: LIR reducible inconsistency |
|---|---|---|
| dark room | **not solved**: chosen whenever not physically lethal; the switch is set by β_phys | **solved with a parameter**: darkness is never rewarded (parameter-free), but leaving it for good needs an Occam prior; residual 13% share under a flat prior |
| white noise | **solved** (skeptic: holds for maximum-entropy noise at equal costs; it avoids entropy, not unlearnability, see S4) | ~~solved with a parameter~~ **not solved** (skeptic): dismissed only if the noise law is itself a prior atom; generic i.i.d. noise keeps 20% of attention forever (S1), and with a 16-symbol noise screen a0 = 0.05 vs 0.5 flips the majority of attention between noise and structure (S5) |

Hidden parameters:
1. β_phys / β_perc, the survival-vs-surprise exchange rate. It is set to 1 per edge per tick by convention and scales with the sensor sample rate.
2. The Occam prior: whether simple-law atoms exist (structural), and their mass (sets finite-T shares only).
3. W, the use horizon of a model improvement. It matters only when room costs differ. 1/h is a natural choice, but it is a choice.

Not a parameter: β_c of a deterministic constitution edge.

## 5. What this means for a reasoning network

- Do not train or act on raw inconsistency (surprisal) of what the network observes. That is O1, and it rewards closing the eyes.
- Use the reducible part instead: the inconsistency a change of the network's own controllable parameters could remove, estimated as the mutual information between parameters and the next observation. This is LIR's control mask turned into a criterion. It coincides with the information-gain ("novelty") term of expected free energy and with BALD, so it is not new. What the PDG view adds is the reason to discard the entropy: that term has zero gradient under every control.
- The network needs cheap "this is noise" and "this is constant" hypotheses: hypotheses with short descriptions, such as low-complexity traders in a logical inductor. Otherwise the reducible-inconsistency estimate keeps paying ~(d/2)·log T of attention to every room in proportion to its parameter count, noise included. With such hypotheses, attention to a stream is bounded by the complexity of its law (Hutter eq. 13).
- Constitution constraints can be added as hard (deterministic) edges with no tuning. The exchange rate between learning and survival does not disappear, though: it becomes a horizon (how long the improved model will be used). The framing gives a natural value for it (expected lifetime), but that value is a choice, not a theorem.

Sources: Richardson 2022 (local 2202.txt; arXiv 2202.11862), Richardson et al. 2026 LIR (local lir2026.txt; arXiv 2604.17140), Hutter 2007 "On Universal Prediction and Bayesian Confirmation" (arXiv 0709.1516, eq. 13 verified in the fetched PDF), Friston, Thornton & Clark 2012 "Free-energy minimization and the dark-room problem", Front. Psychol. 3:130 (fetched; their answer is the phenotype prior "Agents that predict rich stimulating environments will find the 'dark room' surprising", i.e. the definitional move this note avoids). The parametric regret of d/2·ln n is Hutter 2007 eq. (10) (verified by the skeptic in the fetched PDF); the toy measures it as 1.5 per doubling at K = 4.


## Skeptic (2026-10-03)

Scripts: `pdg-inconsistency-of-the-agent/skeptic.py` (K = 4, output `skeptic_output.txt`, T = 20000, 6 seeds, about 3 min) and `skeptic_s5.py` (K = 16, output `skeptic_s5_output.txt`, T = 20000, 4 seeds). Both reuse `toy.py` unchanged. All runs are LIR with equal hazards, so W plays no role. Shares are [dark, noise, struct], given per quarter of the run.

**Hidden parameter found: the noise law has to be an atom of the prior, and when it is not, the Dirichlet concentration a0 decides the outcome.** The note's Occam prior adds exactly two kinds of atom, "uniform i.i.d." and "constant", which are the laws of the two test rooms. White noise in the spectral sense means i.i.d. with any marginal law, not necessarily uniform.

- **S1, generic noise (K = 4, atoms 0.5).** With uniform noise the last-quarter share is [0.002, 0.001, 0.997], as the note reports. With biased noise [.4, .3, .2, .1] it is [0.002, 0.199, 0.799]. With a noise law drawn per seed from Dirichlet(1) it is also [0.002, 0.199, 0.799]. The noise room is never dismissed, and it gets slightly more attention than under the flat prior (0.174), because the atoms have already emptied the dark room. The dark room is dismissed in every case, since a constant is always an atom.
- **Why this is not a finite-T effect.** Information gain on an i.i.d. room with d free frequencies behaves like d/(2n), the same as on a structured room with d parameters (Hutter 2007 eq. (10), p. 6: D_n ≤ ln w(θ₀)⁻¹ + d/2·ln(n/2π) + ...). Solomonoff's finite bound D_∞ ≤ K(μ) ln 2 (eq. (13)) needs K(μ) < ∞. If the bias of a noise source is a generic real number, K(μ) is infinite, and the bound falls back to "d/2 ln n plus a constant" (Hutter 2007, p. 17 of the arXiv PDF: "for essentially all parametric classes, Dn(µ||ξ) ∼ d/2 ln n grows logarithmically in n for all (incl. computable) µ"). The LIR criterion spends attention in proportion to parameter count, and it cannot tell noise frequencies from structure parameters. A full Solomonoff prior would eventually dismiss [.4, .3, .2, .1], which is short, but not a physically generated bias.
- **S5, the noisy-TV flip (K = 16, orders 0-1).** The structured room alternates 0↔1 with probability 0.9 (0.33 nats/step). The noise room is i.i.d. over 16 symbols, with its law drawn from Dirichlet(1).

  | prior | a0 = 0.5 (KT) | a0 = 0.05 |
  |---|---|---|
  | flat | [0.227, 0.309, 0.464], struct majority | [0.136, 0.537, 0.327], **noise majority** |
  | atoms 0.5 | [0.000, 0.400, 0.600] | [0.005, 0.560, 0.434], **noise majority** |
  | atoms 0.5, uniform noise | [0, 0, 1] | [0.010, 0.000, 0.990] |

  The prior concentration a0 decides whether the agent spends most of its time on the noise screen or on the learnable stream. The mechanism is that the pseudo-counts on the structured room's 14 unused symbols inflate its information gain (n·g ≈ 11.3 against 7.5 for the noise room at a0 = 0.5, checked directly). This is the hidden fitted parameter the brief asked for: its value flips which room wins.
- **S3, a0 with K = 4 (flat).** The dark-room share varies with a0: 0.048 at a0 = 0.05, 0.127 at 0.5, 0.144 at 1, about 0.16 at 4 and 16. It never becomes the majority, and struct always leads. So a0 controls how much attention the dark room gets under a flat prior, but it does not flip the result there.
- **S2, model class (max Markov order).** With atoms 0.5, orders {0-1, 0-2, 0-3} all give about [0.002, 0.001, 0.997], so the result is robust once the model class contains the struct room's true order. With orders = (0,) the struct room's marginal is uniform, so it looks like noise. The flat prior then splits [0.267, 0.366, 0.366], and with atoms 0.5 the shares drift (last quarter [0.58, 0.22, 0.20]). This is a structural requirement (the model has to be able to represent the structure), not a tunable knob, so it does not count against the verdict.
- **S4, O1 noise verdict.** O1 avoids noise because noise has high entropy, not because it cannot be learned. An i.i.d. noise room with law [.91, .03, .03, .03] has 0.401 nats/step, against 0.435 for struct, so without a dark room O1 picks that noise room over the learnable stream, with no parameter involved. With unequal hazards (struct 0.01, noise 0.001), O1 picks white noise over struct once β_phys > 105, the same exchange-rate knob as for the dark room. For uniform noise at equal costs the "solved" verdict stands, but it is the dark-room tendency viewed from the other side.

**Not overturned.** A1/A4 (the confidence β_c of a deterministic constitution edge is irrelevant) is checked and correct under PDG semantics. The C1 threshold β_phys ≈ 48.6 is independent of T (both the numerator and the denominator scale with T). LIR's dark-room verdict, "solved with a parameter", stands: a constant law is always a short hypothesis, so a genuine Occam prior covers it, and a0 changes only the size of the residual share under a flat prior. Not tested: the 10 forced burn-in visits (they matter only for the first 30 steps).

**Citations checked.** arXiv 2202.11862 = Richardson, "Loss as the Inconsistency of a PDG", AISTATS 2022 (fetched). Local 2202.txt l.993-1020 is sec. 7, Free energy (yes), and Prop. 2 (surprisal) is at l.299-303. arXiv 2604.17140 = Richardson, Samiei, Shakerinava, Viviano, El Kabid, Parviz, Bengio, LIR, AISTATS 2026 (fetched). The local lir2026.txt l.262-300 gives the attention/control masks, and the update θ ← exp_θ(−χ⊙∇⟨⟨φ⊙M⟩⟩) is at l.305-312. Hutter arXiv 0709.1516: eq. (13), Σ E[h_t] ≤ D_∞ ≤ K(µ) ln 2, with D_n the total expected KL, is verified, and eq. (10), d/2 ln n, is verified too. Friston, Thornton & Clark 2012, Front. Psychol. 3:130, doi 10.3389/fpsyg.2012.00130: the quote is verified, and it continues "...and will leave at the earliest opportunity."

**Skeptic verdict.**
- O1 dark: not solved.
- O1 noise: solved only for maximum-entropy noise at equal costs (avoidance of entropy).
- LIR dark: solved with a parameter (Occam constant atom; a0 sets the residual).
- LIR noise: **not solved**. It is dismissed only when the noise law is a prior atom. Otherwise its share is permanent and set by parameter count, and a0 flips which room gets the majority.
