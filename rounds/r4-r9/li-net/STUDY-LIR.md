# Study: Local Inconsistency Resolution (Richardson et al. 2026) for separate trader losses

Written 2026-10-03 for eurisko and Kelvin. Sources: `lir2026.txt` (arXiv 2604.17140, AISTATS 2026), `2202.txt` (Richardson 2022), `pdg2021.txt`. Script: `math/lir/toy2.py`, with its output in `math/lir/output.txt` (numpy 2.5.3, runs in about 12 s). Each claim is tagged **[paper §/eq]**, **[own]** (my derivation, not in the paper) or **[checked Cn]** (verified by check Cn of the script). The brief mentioned a transcript summary, but none was attached, so I worked from BRIEF-MATH, IDEAS-LI-NET §1-2 and SPEC-LEGIBLE-WM.

**Verdict: applies with a modification.** LIR does give separate per-trader losses that agree exactly with one global loss, but the global loss is not the PDG's inconsistency with the data. It is the *increment* in inconsistency that the data causes, which equals the house confidence β_0 times the market's log loss. A trader's local focus also has to include one extra edge: a copy of the market price, held at infinite confidence and attended with **negative** weight. The paper uses the same device for the GAN generator. With that edge, a trader's local cost can go negative (this is the missing upside), stake-setting is no longer degenerate, and the local gradients add up exactly to the global one (checked to 1e-10). Without it (attention on only the trader's own edges plus the data), every stake goes to 0, both in the paper's framework and in our toy.

## 1. What LIR is

**Objects.** A parametric PDG M(Θ) is a hypergraph of variables. Each arc a carries a cpd P_a(T|S; θ_a) with parameters θ_a **[§2, Def. 1]**. Its observational incompatibility is OInc_M(μ) = Σ_a β_a KL(μ(T,S) ‖ P_a(T|S)μ(S)) **[eq. 1]**, and its structural deficiency is SDef(μ) = −H(X) + Σ_a α_a H(T_a|S_a) **[eq. 2]**. The γ-inconsistency ⟨⟨M⟩⟩_γ = inf_μ [OInc + γ SDef] **[eq. 3]**. Most examples use γ = 0.

**Attention and control.** These are the two "focus" knobs **[§3]**:
- **Attention** φ = (α, β, γ) says how much each arc counts in the inconsistency being resolved. The paper's key move is that the classical PDG confidences β are attention: "we can reinterpret all three classical weight parameters as components of the attention mask". φ_a = 0 takes an arc out of mind. Negative values are used for GANs (§4.3).
- **Control** χ ∈ [0, ∞] per parameter coordinate says how far each parameter may move. χ = 0 freezes it, χ = ∞ means "move to a fixed point".

**The local step** (Algorithm 1): Refocus picks (φ, χ), usually from a fixed finite set F of foci. Then θ ← exp_θ(−χ ⊙ ∇_θ ⟨⟨φ ⊙ M(θ)⟩⟩), which is gradient flow on the attended inconsistency, with coordinate k run for time χ_k. With small χ this is one gradient step. With χ = ∞ the paper expects θ ← argmin_θ ⟨⟨φ ⊙ M(θ)⟩⟩ **[eq. 4]**. Gradients pass through the inner inf over μ by the envelope theorem **[App. A.1]**.

**Guarantees.** There are few, and the paper says so.
- Convexity: if every parametrised cpd is unconditional and log-concave and β ≥ γα, then θ ↦ ⟨⟨M(θ)⟩⟩ is convex **[Prop. 1]**. For conditional cpds, convexity can fail. That eq. 4 still holds is a conjecture, and quasi-convexity in p(Y|X) is called "a key open question" **[§3]**.
- Local versus global: by monotonicity (adding arcs or raising β never lowers inconsistency, **[2202 Lemma 1]**), the attended inconsistency underestimates the global one. The paper does **not** prove that a local step lowers the global inconsistency. It says the opposite can happen ("fixing one inconsistency can easily create others out of view", §1). The conclusion lists "Under what assumptions does LIR converge?" as open **[§6]**.
- Evidence is empirical only: on 4-7-variable random PDGs, resolution averages 99.2% (uniform attention), 99.4% (hub) and 72.2% (partial) **[App. A.2, Table 2]**. The paper measures "distortion" because "set all beliefs to uniform (or delete them)" resolves everything, but does nothing to prevent it.
- Two sufficient conditions for a local step to lower the global inconsistency **[own]**: (i) attention covers every arc and control is partial, which makes the step block-coordinate descent on one function (this is why EM is monotone, Prop. 2 and Neal-Hinton); (ii) infinite-confidence data pin μ completely, so the global inconsistency splits into a sum of per-arc terms.

## 2. Algorithms recovered, as attention/control choices

| Algorithm | Attention (what is looked at) | Control (what may move) | Where |
|---|---|---|---|
| Training a classifier | p_θ(Y\|X) and the labelled pair (x, y) | θ | §4.1, eq. 5 |
| Inference | the same | the label y | §4.1 |
| Adversarial example | the same | the input x | §4.1 |
| SGD | a minibatch's empirical distribution at high confidence | θ, with χ·φ = learning rate | §4.1 |
| Adversarial training | alternates two foci: attack (move x′ toward y′) and patch (move θ) | x′, then θ | §4.1, Fig. 1 |
| EM | the whole model p(Z, X) + q(Z) + x | alternately all of q (E-step) and all of p (M-step) | Prop. 2 |
| VAE | encoder, decoder, prior, minibatch | encoder and decoder, not the prior | §4.2 |
| GAN | D: everything, with the "50/50" belief e attended at −1. G: e at +1 and D at −1 | D, then G | Prop. 3 |
| Belief propagation | one factor or variable neighbourhood of messages | the outgoing message | Prop. 4, App. B |
| Transformer layer | arcs N(x′_j \| v_i) with attention exp⟨k_i, q_j⟩ | the outputs x′ | Prop. 5 |
| GFlowNet (TB) | P_F and P_B, each attended by its excess surprisal (treated as fixed) | P_F (and optionally P_B) | Prop. 6, eq. 8 |
| Decision making | belief p(S), outcome model τ, utility u, a "faith" soft constraint b | the action A | Prop. 7 |

In plain words: one model plus "what I look at" and "what I may change" gives training, inference, attacks, EM, message passing or a GAN. Opposed players are the same edges with opposite-sign attention (§4). Only in the GFlowNet case does attention depend on θ, and there it is treated as fixed (stop-gradient) **[Prop. 6]**; the paper's one new practical result is the length-normalised TB loss derived this way (Fig. 3).

## 3. The central question: separate per-trader losses consistent with a global one

**Setup** **[own]**. X is the outcome. Trader i sees I_i = ι_i (pinned at ∞) and holds a cpd p_i(I_i | X; θ_i) with stake β_i(ι_i); ℓ_i(x) = log p_i(ι_i | x). A house prior p_0(X) has fixed weight β_0. (PDGs have one β per arc; with I_i pinned, β_i(ι_i) is a constant.)

**3a. The paper never controls β.** β is part of φ, set from outside by Refocus; attention is never a controlled parameter. Learning an adaptive Refocus is an open question **[§6]**, and a trader choosing its own stake is exactly that. Nothing in the paper stops "β → 0".

**3b. Plain inconsistency is degenerate** **[own; checked C4]**. With the outcome x observed at ∞, μ = δ_x and ⟨⟨M + x⟩⟩ = β_0(−log p_0(x)) + Σ_i β_i(−ℓ_i(x)). This splits into one term per trader, so a local step really is a global step. But every coefficient multiplies a non-negative quantity, so descending on β drives every stake to 0, and the market loss becomes log 2. The schedule in the brief (attention = own edge + data edge, control = own cpd and stake) is exactly this objective. Its local LIR steps also send every stake to 0 (C5, last line).

**3c. The right global loss is an increment in inconsistency** **[own; checked C1]**. Let M be the traders' edges plus the house edge, with no outcome. Then

  −β_0 log μ*(x) = ⟨⟨M + x⟩⟩ − ⟨⟨M⟩⟩,  with μ* = argmin of ⟨⟨M⟩⟩ ∝ p_0 Π_i p_i(ι_i|·)^{β_i/β_0}.

The market's log loss is the extra inconsistency caused by learning the outcome, per unit of house confidence. This is the chain rule (−log p(x|ι) = −log p(x, ι) + log p(ι)) written in inconsistency form. It is not one inconsistency, so the paper's "minimise ⟨⟨·⟩⟩" does not cover it directly. Its gradient, however, splits trader by trader **[checked C2]**:

  ∂L/∂β_i = −(1/β_0) s_i,  ∂L/∂θ_i = −(β_i/β_0)(∇ℓ_i(x) − E_μ ∇ℓ_i),  where s_i = ℓ_i(x) − E_μ[ℓ_i].

Here s_i is the payoff of trader i's log-likelihood bet, centred so that it costs nothing at market prices: E_μ[s_i] = 0 (checked to 1e-16). A trader needs only its own cpd, the outcome and the market price. It does not need the other traders.

**3d. The same split as an LIR focus** **[own; checked C3]**. Add a copy variable X′ to the PDG. Trader i's focus T_i attends to:
- the data δ_x on X, at ∞;
- the market price m on X′, at ∞ (m is the output of the previous price focus);
- its own edge p_i(ι_i|X), with weight +β_i;
- its own edge p_i(ι_i|X′), with weight −β_i.

Control is (θ_i, β_i). The infinite-confidence edges pin μ = δ_x ⊗ m, just as the GAN proof pins μ, so the negative weight cannot drive the inf to −∞. The focus's inconsistency is

  ⟨⟨φ_{T_i} ⊙ M⟩⟩ = β_i (E_m ℓ_i − ℓ_i(x)) = −β_i s_i,

which **can be negative**. That supplies the upside. Its gradients in (θ_i, β_i) equal β_0 × the global gradients; the script's maximum difference is 1.6e-10. The schedule alternates a **price focus** P (all trader edges, full control of m, so m = the pool) with trader foci T_i (small control), BP-style: the price is the one message each trader reads.

**3e. A stake under the trader's own control** **[own; checked C5, C6]**:
- With m frozen, the local objective is *linear* in β_i. Full control (χ = ∞) therefore sends β_i to 0 or to ∞. Stakes need small χ and a refreshed m, i.e. two timescales.
- For fixed θ, the global L is convex in β (a log-partition function). β is identified only relative to β_0 (the house), which is how the house sets the scale.
- 2-trader toy: X ~ Bern(½); both see regime R; trader i's signal equals X w.p. 0.8 if R = i, else noise. Direct descent on L hits the Bayes loss 0.50040 with effective weight β_i·θ_i = logit 0.8 = 1.386 when R = i and exactly 0 otherwise. The LIR schedule (4000 rounds × 64 samples) reaches 0.50057, stakes ≈ 1.3 when relevant, 0 / 0.035 when not.
- At the optimum, the KKT conditions give E_q[s_i] = 0 for every trader with β_i > 0 and E_q[s_i] ≤ 0 for every trader with β_i = 0 (checked to 1e-11). On average over the data, no trader in the market profits from its own bet at market prices. This is a logical-induction-flavoured inexploitability, relative to the market's own traders and in expectation, not over a worst-case sequence.

**3f. The pool matters** **[own; checked C9]**. The γ = 0 minimiser of posterior edges p_i(X|ι_i), μ ∝ Π p_i^{β_i/B}, has weights that sum to 1, so it *averages*: two independent 0.9 signals give 0.9. Bayes gives 0.988. Adding evidence needs the prior edge at weight β_0 − Σβ_i, which is **negative** once the stakes exceed the house's; with that edge the solver returns 0.9878. Softmax attention computes the averaging (normalised) pool (Prop. 5). An evidence-adding block needs unnormalised weights minus a prior. On "low α": α only matters through γ. With α_i = 0 on trader edges, γ·SDef = −γH(μ) = γ KL(μ ‖ uniform) + const, which is a uniform house edge at confidence γ. That edge tempers the pool, so "low α, γ > 0" is just a house prior **[own]**.

## 4. Opposition: the GAN case, and a market of traders

**How LIR expresses opposition** **[paper Prop. 3; checked C10]**. There is one PDG: coin C ~ ½ at ∞, X | C at ∞ (p_data or G), discriminator D(C|X), and the belief e(C|X) = ½. The two foci differ only in the signs of their attention: discriminator (D: +1, e: −1) and generator (D: −1, e: +1). Because the ∞ arcs pin μ completely, negative attention is harmless. The two foci's inconsistencies sum to exactly 0 for every (G, D): this is a zero-sum game, with no global potential and no monotone quantity. The paper does not claim convergence. An erratum, checked numerically: the proof drops the coin's ½, so the correct form is ⟨⟨M⟩⟩ = −(β_D/2) L_GAN + (β_D + β_e) JS − β_D log 2. The foci give −L_GAN/2 − log 2 and L_GAN/2 + log 2, not ±L_GAN ± log 2. The argmins are unchanged.

**Applied to the market** **[own]**. In the potential version the traders are not opposed to each other; they share the interest L. The opposition is between each trader and the *market price*: trader i's focus attends to its own edge positively on the real outcome and negatively on the market's copy. This is the generator disbelieving D, with the market in the role of D. The focus the brief proposed (one trader's edges + the data edge, control = that trader's cpd and stake) is expressible but degenerate (3b). Adding the pinned market copy at negative attention makes it correct (3d).

**Leave-one-out** **[own; checked C7, C8]**. r_i = log μ(x) − log μ_{−i}(x) = −L + (a term that does not depend on trader i's choices). It is therefore an **exact potential game** with potential L (Monderer-Shapley). Its gradient in the trader's own parameters equals −∇L exactly (C7). LOO is the integral of the marginal settlement along β_i from 0, and for small stakes r_i ≈ (β_i/β_0)s_i. So the earlier finding "LOO makes it Kelly-like but not a potential game" holds for **stake-weighted P&L**, not for LOO as a training signal. Take the Kelly payoff π_i = β_i E log(p_i/μ) on the normalised pool: its cross-partials are asymmetric (−0.011 vs −0.065), so it has no exact potential. Also Σ_i β_i log(p_i(x)/μ(x)) = −⟨⟨M⟩⟩ for every x: when every edge bets at market odds, the house takes the inconsistency, the same amount in every outcome (consistent with IDEAS §1, "inconsistency = riskless arbitrage profit"). I have not seen the earlier derivation; if wealth compounded into the next stake, that dynamic is the non-potential part.

## 5. Verdict

| Option | Fits? | Why |
|---|---|---|
| Applies directly | No | Taken literally (minimise the inconsistency, with β as attention), it gives β → 0 (3b). The paper does not control β, and has no convergence result for opposed or partial foci. |
| **Applies with this modification** | **Yes** | Use the increment ⟨⟨M + x⟩⟩ − ⟨⟨M⟩⟩ as the global loss. Each trader's focus = own edge (+), data (∞), pinned market copy (∞) with own edge (−); control = own θ and stake, at small χ, alternating with a price focus. Local and global gradients are then identical (C3), the local cost has an upside, stakes learn relevance (C5) and the optimum is inexploitable by the market's own traders on average (C6). Every ingredient is an LIR device the paper already uses: pinned copies, negative attention, BP-style messages. |
| Not much / not at all | No | The 3c formulas alone are standard log-linear-pooling / PoE gradients, but LIR adds the realisation vocabulary (foci schedule, Prop. 5 transformer link, negative attention) and the reason "own edges + data" fails. |

Limits: no LIR convergence theorem (ours comes from convexity of L in β); LI properties beyond average inexploitability are untouched; the θ-landscape is non-convex (β and θ are confounded).

## 6. Three things to try

1. **Evidence-adding market head.** N heads emit ℓ_i and stakes β_i(I_i); price = softmax(log p_0 + Σ β_i ℓ_i / β_0) (unnormalised weights, not averaging). Train on the market loss, log s_i and LOO r_i as credit. Measure on a generator with known relevance: stake-relevance correlation, held-out KKT residual E[s_i], gap to Bayes, and the averaging pool as control. Success: stakes ≈ 0 exactly where information is irrelevant; lower loss than the averaging pool on independent signals.
2. **One block = one LIR sweep.** Price focus = the PDG projection (IDEAS §1); each rule trader = focus "own rule (+), market copy (−), control = own position", which I conjecture is IDEAS idea 2 (Hildreth) in LIR form. Let a transformer choose foci (learned Refocus, Prop. 5); compare inconsistency per block at 20/100/500 against round-robin.
3. **Earned β by KKT (SPEC-LEGIBLE-WM arm E).** Replace the prequential-gain heuristic with β = argmin of held-out market loss over stakes, θ frozen: convex, with certificate "β_c > 0 ⇒ held-out E[s_c] = 0; β_c = 0 ⇒ E[s_c] ≤ 0". Test whether it stops arm F's β inflation, and whether the uniform house edge (γ, "low α") calibrates stakes when claims are few.
