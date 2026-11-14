# No-history traders: what a fixed point of history-free traders gives, and what it loses

Key: `no-history-traders` (eurisko li-net math workstream, 2026-10-02).
Scripts and raw output: `math/no-history-traders/part{1,2,3,4}_*.py` and `part{1,2,3,4}_output.txt`
(numpy 2.5.3, scipy 1.18.1; each runs in seconds except part 2, about 2 min).
Paper numbering below follows `full.txt` (arXiv 1609.03543, https://arxiv.org/abs/1609.03543).
In that numbering Persistence is 4.2.3, Preemptive Learning 4.2.4, No Expected Net Update (NENU) 4.12.2,
Self-Trust 4.12.4 and Paradox Resistance 4.11.2. The brief's numbers 4.2.3, 4.3.9 and 4.3.12 are from another version.

**TL;DR.**
1. At a fixed point, history-free traders do get coherence and the values the axioms force. One continuous
   trader is enough: it trades toward the projection onto the coherent polytope.
2. "Frequency on independent sentences" does not come from inexploitability. Every point of relint C is
   unexploitable. The frequency has to come from a prior, applied over joint models; projecting prices does not work (counterexample below).
3. The time-indexed theorems really do need history (for Convergence this is proved only for history-free traders whose
   rates have constant sign; the general case is a conjecture, see Skeptic): Convergence, Preemptive Learning, Persistence, NENU, Self-Trust
   and full Provability Induction. But the traders in their proofs only need **their own holdings**, which are a state
   variable rather than attention to past days. In the block setting these theorems also become **ensemble
   (martingale) properties** that the evaluator measures.
4. Paradox resistance is a same-day property and holds with history-free traders. A trader with resolution δ has
   fixed point p/(1+δ). With δ = 0 there is no fixed point.

---

## 0. Setting and definitions

* Finite propositional language with atoms `A`, a theory `Γ` (axioms) and a finite sentence set `S` given at block 0.
  Models `Mod(Γ)` = assignments to `A` that satisfy `Γ`; each model `w` induces a vector `w ∈ {0,1}^S`.
* **Coherent polytope** `C = conv{ w : w ∈ Mod(Γ) } ⊂ [0,1]^S` (the marginal polytope of Γ's models on S).
* **Price vector** `P ∈ [0,1]^S`. A **trade** is a share vector `x ∈ R^S` bought at prices `P` (cash `−x·P`).
  Its value in world `w` is `x·(w − P)`.
* **History-free (HF) trader**: a continuous map `T : [0,1]^S → R^S`. It may depend on the structure
  (Γ, the syntax of S) and on the block index t, but not on past price vectors.
  (The paper's traders read `P_≤n`; Def. 3.4.4–3.4.5, full.txt 931–990.)
* **Market-maker fixed point** (paper Lemma 5.1.1, full.txt 2722–2760): `fix_T(V) = clip(V + T(V), 0, 1)`. Brouwer
  gives a fixed point because T is continuous. At any fixed point `T(V)·(w − V) ≤ 0` for **every** world w.
* **Sure profit / Dutch book** at P: `min_{w∈Mod Γ} x·(w−P) > 0`. **Weak book**: `≥ 0` for all w and `> 0` for some w.
* **Stationary market**: `P_n ≡ P` for all n and `D_n ≡ Γ`, so `PC(D_n) = Mod(Γ)`. "Exploit" is as in Def. 3.5.1.

Toy instance used in the checks: atoms a, b, c; `Γ = {a→b, b→c}` (4 models: 000, 001, 011, 111);
S = {a, b, c, a→c, a∧c, ¬b, b∨c, a∧¬c, c→a}. Here `a→c` is entailed and `a∧¬c` is refuted.

---

## 1. Fixed-point properties: history-free traders suffice

**R1 (theorem). Coherence trader.** `T_coh(P) = π_C(P) − P`, where π_C is the Euclidean projection onto C.
For every model w: `T_coh(P)·(w − P) ≥ dist(P, C)²`.
*Proof.* Write π = π_C(P). Then `(π−P)·(w−P) = (π−P)·(w−π) + |π−P|²`. The projection inequality for convex C gives
`(P−π)·(w−π) ≤ 0` for w ∈ C, so the first term is ≥ 0. ∎
T_coh is 1-Lipschitz (projections are non-expansive), history-free, and reads only P and the structure (C).
The fixed points of `fix_{T_coh}` are exactly C: `fix(V) = π_C(V)` since C ⊂ [0,1]^S. With step size η = 1 the block
reaches C in one step.
*Check:* the minimum of [profit − dist²] over 300 random P and all models is −4e−8 (QP tolerance). The block
`P ← P + 0.5·T_coh(P)` reaches dist(P, C) = 1.3e−9 after 30 blocks.

**R1′ (theorem). Bregman or logit version** (for C inside the open cube, i.e. on sentences that Γ leaves undecided).
For convex F with Bregman divergence D, let `π = argmin_{V∈C} D(V,P)` and `x = ∇F(π) − ∇F(P)`. Then
`x·(w−P) = D(w,P) − D(w,π) + D(P,π) ≥ D(π,P) + D(P,π)` for all w ∈ C. The equality is algebra. The inequality is the
generalised Pythagorean inequality `D(w,P) ≥ D(w,π) + D(π,P)` for Bregman projections onto convex sets
(Bregman 1967; Csiszár 1975 for KL, https://doi.org/10.1214/aop/1176996454, recalled (unverified)).
With F = Σ binary negentropy, the trader buys in **log-odds**: x = logit(π) − logit(P). This trader blows up at decided
sentences (logit(1) = ∞), so decided sentences should use the affine traders of R2.
*Check:* min [profit − symKL] = −3e−6 over 100 P (SLSQP tolerance).

**R2 (theorem). Values forced by the axioms, and affine constraints.** If `a·w ≥ b` for all models w, the trader
`T(P) = max(0, b − a·P)·a` has sure profit `≥ max(0, b − a·P)²`. An equality constraint is two inequalities.
A theorem (Γ ⊨ φ) is the case `w_φ ≥ 1`, and a refutation is `w_φ ≤ 0`. All of these are history-free.
*Check:* for a→b (`w_b − w_a ≥ 0`), min [profit − gap²] = 0.0 over 1000 random P.
At the coherence fixed point, P(a→c) = 1.000000 and P(a∧¬c) = 0.000000.

**R3 (known: de Finetti coherence plus Stiemke's lemma; the proof here is complete).** A stationary market P is
unexploitable by every trader, history or not (on a constant market, history carries no information), **iff
P ∈ relint C**, i.e. P = Σ_w λ_w w with every λ_w > 0.
*Proof.* (⇐) Let X_n be the holdings and v_w = X_n·(w−P). Then Σ λ_w v_w = 0, so if every v_w ≥ −B then every
v_w ≤ B/λ_min. Worth bounded below therefore implies worth bounded above. (⇒) If P ∉ C, a strong Dutch book x exists
(separating hyperplane). If P lies on a proper face of C, a supporting hyperplane gives a weak book x. In both cases the
constant trader x, repeated daily, exploits P. ∎
References (recalled, unverified): https://en.wikipedia.org/wiki/Dutch_book_theorems and
https://en.wikipedia.org/wiki/Farkas%27_lemma (Stiemke variant); strict coherence is due to Shimony and Kemeny (1955).
*Check:* the LP for the weak-book value gives 0 at the uniform-over-models point and 7.0 at a boundary point (mixture
of two models).
**Consequence (theorem):** inexploitability cannot select frequencies on independent sentences. Every point of
relint C, however extreme its values on independent sentences, is unexploitable. The frequency must come from a prior.

**R4 (theorem plus a refutation). Prior traders.** Let μ0 be a prior over assignments. The target is the conditional
`m* = E_{μ0(·|Γ)}[w]`.
(a) `T_prior(V) = m* − V` is history-free. If m* ∈ C, the unique fixed point of `T_coh + T_prior` is m*. *Proof:* at
an interior fixed point π(V) − V = V − m*. Since m* ∈ C, |V − π(V)| ≤ |V − m*|, with equality only if π(V) = m*, which
then forces V = m*. ∎ *Skeptic patch (the proof above covers only interior fixed points; m* has coordinates 0 and 1
here, so clipped fixed points must be excluded too):* at any fixed point of the clipped map, T(V)·(u − V) ≤ 0 for every
u ∈ [0,1]^S (coordinate-wise, as in Lemma 5.1.1). Take u = m*: (π(V)−V)·(m*−V) + |m*−V|² ≤ 0, and the first term is
≥ |π(V)−V|² ≥ 0 by the projection inequality, so V = m*. ∎ *Check:* |V − m*| = 0.0; skeptic check S1: 14 starts including
cube corners all end at m* (max error 2e−16).
(b) The joint-level I-projection of μ0 onto {μ : μ(Γ) = 1} is μ0(·|Γ), and its KL equals −log μ0(Γ). This is known
(Csiszár; the PDG form is Richardson 2202.11862 Prop. 2, "inconsistency = surprisal", 2202.txt 290–305; conditioning as
inconsistency resolution, pdg2021.txt 804–811). *Check:* KL = 0.544727 = −log μ0(Γ), and the marginals equal m*.
(c) **Refuted:** "projecting the local prior's sentence marginals m0 onto C gives the conditional." Euclidean
projection is 0.329 away from m*; the binary-KL projection is 0.307 away. (Skeptic: "the geometry of the joint is lost" is too strong. With a product prior and the atoms in S, m* is a
nonlinear function of m0, so the information is still there. What fails is the Euclidean and per-sentence KL projection.)
(d) (theorem) With soft weights `w_c·T_coh + T_prior(m0)`, the fixed point is
`V = (w_c·π(m0) + m0)/(1+w_c)` and `dist(V, C) = dist(m0, C)/(1+w_c)`. Coherence holds only as w_c → ∞, i.e. axioms at
confidence ∞, as in a PDG.
*Check:* 0.1613 / 0.0293 / 0.0032 for w_c = 1 / 10 / 100, against the predicted 0.1613 / 0.0293 / 0.0032.

**R5 (theorem). Model-holding traders (particles).** Trader k holds a model w_k ∈ Mod(Γ) and trades
`T_k(V) = w_k − V`. The fixed point of Σ_k T_k is the empirical mean of the w_k. That point is automatically coherent
(a convex combination of models) and gets the forced values right. No trader has a sure (strong) profit there, since V ∈ C. *Skeptic correction:* this is not R3. V lies in relint C only if
every model is sampled. Otherwise V sits on a proper face of C and a weak book exists, so a stationary market at V is
exploitable (skeptic check S2: K = 10 misses a model in 28.5% of draws; weak-book value 2 to 3, strong-book margin 0). If w_k ~ μ0(·|Γ), the error is O(K^{-1/2}).
*Check:* |V − m*| = 0.46 / 0.11 / 0.024 for K = 10 / 100 / 10⁴, with dist(V, C) ≤ 3e−9.

---

## 2. What is lost without history

| Theorem (full.txt) | Proof trader needs | History-free at a fixed point | Block / ensemble replacement |
|---|---|---|---|
| 4.1.1 Convergence | its own holdings H_{n−1} (6.1.1, full.txt 3510–3530) | **lost** (R6: proved for sign-constant traders, conjecture in general) | the dynamics must converge (contraction or Lyapunov function), not the traders |
| 4.2.1 Provability Induction (over a sequence) | one open position until resolution (proof sketch, 1255–1265; ROI lemma B.1.3) | only along sparse e.c. subsequences (R8); trivial at an exact propositional fixed point | block-timeliness (sec. 3), as an ensemble property |
| 4.2.3 Persistence, 4.2.4 Preemptive | buy φ_n on day n, sell on a later day m: needs the position | **lost** | martingale ⇒ preemptive via optional stopping (R10) |
| 4.12.2 NENU, 4.12.4 Self-Trust | trades on day n and on day f(n), linked by position; sentences about its own future prices | **lost** (and a finite propositional language has no self-reference) | E[P_{t'} \| P_t] = P_t, E[V* \| P_{t'} > p] ≥ p, measured externally (R10) |
| 4.3.x calibration, 4.4.x frequencies | resolution feedback over days | **lost** with no new facts | prior over models (R4, R5) plus ensemble calibration |
| 4.5.x affine coherence | same-day | **kept** (R1, R2) | — |
| 4.11.2 Paradox Resistance | same-day (proof F.2, full.txt 7558–7590) | **kept** (R11) | — |

**R6 (theorem only for history-free traders whose rate sequences r_n and s_n each have eventually constant sign;
conjecture for general history-free traders; downgraded by the skeptic, see Skeptic S3). History-free inexploitability does not imply
convergence.** Let prices alternate between 0.3 and 0.7. A history-free trader trading only φ is a pair of e.c. rate
sequences: `r_n = T_n(0.3)` and `s_n = −T_n(0.7)`. The market, using its own history and an oracle (so it is not
computable), runs low phase j until every trader k ≤ j with Σ r = ∞ has worth(φ=0) < −j, then high phase j until every
trader k ≤ j with Σ s = ∞ has worth(φ=1) < −j. Each phase is finite.
Case analysis: if Σr = Σs = ∞, the worth is unbounded below. If only Σr = ∞, net shares go to +∞ and worth(φ=0) goes
to −∞. Symmetrically if only Σs = ∞. If both are finite, the worth is bounded. So no history-free trader exploits the
market, yet the price never converges. The trader with holdings, from the paper's proof (buy up to holding 1 at 0.3,
sell to 0 at 0.7), has worth ≥ −0.3 and gains 0.4 per cycle, so it exploits. ∎
The paper itself names this failure ("buy 10, then 100, then 1000 … unbounded below", full.txt 3497–3506).
*Check (part2):* 4 history-free traders (rates 1, n^-1/2, n^-1/4, n^-1/3) for 3 joint phases reach min worth
[−5.9e6, −365, −1.6e4, −2.5e4]. A single constant-rate trader over 10 phases reaches min worth −10.0 at day 5.1e8.
The holdings trader stays in [−0.30, 4.30], rising 0.4 per cycle.
Status in the full language: **conjecture** that the same holds (traders could also trade ¬φ, φ∧ψ, …; the market
keeps those coherent).

**R7 (observation plus conjecture). "Markov" traders are enough.** The proof trader for 4.1.1 is a function of
(n, P_n, H_{n−1}), the current price and its own holdings; the paper unrolls H as a feature of the price history only
because traders are formally stateless. **Conjecture:** every trader used in the proofs of Section 4 is Markov in
(n, P_n, own holdings, the budgeter's wealth). Checked by me only for 4.1.1 and the 4.2.1 sketch; the rest is
unverified. If true, "no attention to past days" is compatible with all of Section 4, provided each trader carries a
position/wealth vector in the block state.

**R8 (theorem). History-free provability induction along sparse subsequences.** Let φ be e.c. theorems with an e.c.
bound f such that φ_n ∈ D_{f(n)}. Let a_{k+1} = f(a_k) + 1. The history-free trader that on day a_k buys
`Ind_ε(P(φ_{a_k}) < 1−ε)` shares, and does nothing on other days, has at most one unresolved position (worth ≥ −1)
and gains ≥ 2ε each time it buys a full share (P ≤ 1−2ε, where Ind_ε = 1) and > 0 on partial purchases, once the
purchase resolves (skeptic wording fix; the conclusion is unchanged). So `P_{a_k}(φ_{a_k}) → 1`.
**Conjecture:** the full diagonal `P_n(φ_n) → 1` fails for history-free inexploitability. Sketch: with risk weights
c_a, bounded risk requires Σ_{a ≤ n < f(a)} c_a ≤ 1, so c_a → 0 when f(a) − a → ∞. An adversarial bad set B of density 0
then keeps Σ_{a∈B} c_a finite.

**R9 (observation, informal; downgraded from theorem by the skeptic: exploitation is an asymptotic notion, so "the
block-t prices that no trader can exploit" means a stationary market with D ≡ D_t, which is just R3). Timeliness is not an inexploitability property in the block setting.** Let D_t be what a reference
derivation produces in t blocks, and C(D_t) = conv of 0/1 assignments to S consistent with D_t (a relaxation,
C(D_t) ⊇ C(Γ)). By R3 applied to D_t, the block-t prices that no trader can exploit relative to D_t are exactly
relint C(D_t). That includes prices far from the truth on sentences whose derivation needs d > t blocks.
With an exact propositional PC(Γ) at block 0, provability induction is trivial at the fixed point (R2), and all the
difficulty is computational. In LI, pattern bets become sure bets only because the position is **held until φ_n is
proved** (history). Within a run of H blocks with d(n) > H, the bet never resolves, so trusting a pattern can only be
justified by the training distribution.

**"Time" → "block index", the reformulation.** Present all of φ_1..φ_N at block 0. Replace day n by block t. Replace
D_n by D_t (internal derivation depth). Replace the diagonal P_n(φ_n) by P_{t(n)}(φ_n) with a budget t(n) = o(d(n)).
The evidence for a pattern is **cross-sectional**: siblings already derived sit in the current state (their prices are
about 1), so a history-free trader can read the pattern from P_t. It cannot be paid for it within the run (R9). What
replaces the time-indexed theorems is R10.

**R10 (known: Doob; tower property).** Let F_0 ⊆ F_1 ⊆ … be the information that blocks 0..t can compute, with the
instance drawn from a distribution 𝒟. If `P_t = E[V* | F_t]`, then P_t is a bounded martingale, and:
(i) NENU (ensemble form): E[P_{t'} | F_t] = P_t;
(ii) Self-Trust (ensemble form): E[V* | P_{t'} > p] > p;
(iii) Preemptive Learning: for τ = first t with P_t ≥ p, optional stopping gives P_0 = E[P_{τ∧H}] ≥ p·Pr(τ ≤ H);
(iv) Persistence (skeptic correction: martingale convergence gives Convergence, not Persistence; the within-horizon
analogue of 4.2.3 is Doob's maximal inequality, Pr(sup_{t≤s≤H} |P_s − P_t| ≥ ε) ≤ E|P_H − P_t|²/ε²; check S5).
https://en.wikipedia.org/wiki/Optional_stopping_theorem (recalled, unverified).
*Check (part3):* P_0 = 0.7043 ≥ 0.9·Pr(hit) = 0.6352; E[φ | P_30 > 0.5] = 1.0.

---

## 3. A definition of "timely" when time is block depth

**Data.** A distribution or family of instances (Γ, φ). A reference derivation with depth d(Γ, φ), the number of
blocks after which D_t decides φ (∞ if φ is independent). Target V* = 1 or 0 if Γ decides φ, otherwise μ0(φ | Γ).
Network prices P_t(φ) after t blocks.

**Definition (block-timely; analogue of Def. 4.2.2, full.txt 1308–1313).** A network is *timely on the family*
(Γ_n, φ_n) with d(n) → ∞ if there is a budget t(n) with t(n)/d(n) → 0 (strong form: t(n) = O(1) or O(log d(n))) such
that `P_{t(n)}(φ_n) − V*(φ_n) → 0`. It is *persistently* timely if, in addition,
`sup_{s ≥ t(n)} |P_s(φ_n) − V*(φ_n)| → 0`.

**What a good reasoner does when d > t:** output `E_𝒟[V* | F_t]`, the calibrated best guess from what t blocks can
compute. It should not hold 0.5, and it should not hold its block-0 prior (both fail below).

**How to measure it** (per block t ∈ {20, 100, 500}, on instances still undecided at t, i.e. d > t):
* Brier_t, overall and on undecided instances, broken out by d (the timeliness profile e(t, d));
* ECE_t: calibration on undecided instances;
* NENU drift: |E[P_{t+k} − P_t | bin(P_t)]| per bin. It should be 0;
* persistence violations: the fraction of instances that leave an ε-ball around V* after entering an ε/2-ball;
* outpacing ratio t_ε/d, where t_ε is the first block after which the price stays within ε of V*.

*Check (part3, φ = OR of 40 bits with q = 0.03; derivation reads one bit per block; mean d = 23.4):*

| reasoner | t | Brier(undecided) | ECE(undecided) | NENU drift (k=10) |
|---|---|---|---|---|
| Bayes E[φ\|F_t] | 0 / 10 / 20 / 30 | .208 / .240 / .248 / .194 | ≤ .002 | ≤ .002 |
| deduction-only (0.5 until derived) | same | .250 at every t | .206 / .101 / .042 / .237 | .13–.24 |
| frozen prior (ignores absence of evidence) | same | .208 / .251 / .309 / .389 | .002 → .442 | .078 → .442 |

**Implication for training.** A loss only at block 20 trains nothing at t < 20 and gives no guarantee at 100 or 500.
**Theorem (tower):** if every P_t minimises the expected *strictly* proper score given F_t, and F_t is nested (the block never
forgets), then (P_t) is a martingale, so the ensemble forms of NENU, Self-Trust, Preemptive Learning and Persistence
all hold. This suggests an anytime loss: a proper score at a random depth t ~ U{1..20}.

---

## 4. Paradox resistance with history-free traders at a fixed point

Take χ ↔ (P(χ) < p). At price x the truth is W_x(χ) = 1[x < p], so the value of s shares is s·(1[x<p] − x).
Call a continuous trader **δ-truth-seeking** if T(x) > 0 for x ≤ p−δ and T(x) < 0 for x ≥ p.

**R11 (theorem).**
(a) Every δ-truth-seeking trader has a fixed point of fix_T (IVT on T, by continuity), and every fixed point lies in
[p−δ, p). For `T_δ(x) = Ind_δ(x<p) − x` with `Ind_δ = clip((p−x)/δ, 0, 1)`, the fixed point is exactly
**x* = p/(1+δ) → p** as δ → 0.
(b) **Continuity is what makes it exist.** For δ = 0, T(x) = 1[x<p] − x has |T| ≥ min(1−p, p) everywhere, so fix has
no fixed point and the block iteration cycles 1, 0, 1, 0.
(c) **Lemma:** no continuous trader has positive value both at x = p and as x → p⁻. The value just below p tends to
T(p)(1−p), the value at p is −T(p)·p, and these have opposite signs.
(d) At x*, χ is **true** (x* < p), so a single deterministic fixed point gets the truth of χ wrong by about 1−p. A
trader with finer resolution δ' < p − x* profits there. The paper's "frequency p" (4.11.2 with 4.3.6) appears only
along a sequence, as a limit point. **Refuted:** "a single static fixed point prices χ at p and gets its frequency
right". It prices χ within δ of p, and χ is true there.
*Check (part4):* x* = 0.7272727273 / 0.7920792079 / 0.7992007992 / 0.7999992 for δ = 0.1 / 0.01 / 0.001 / 1e−6,
matching p/(1+δ) to 1e−10, with χ true each time. For δ = 0, min|fix(x) − x| = 0.2. Over 10⁴ random continuous
traders there are 0 violations of the lemma.

**R12 (theorem). Resolution costs depth.** For the block `x ← x + η·T_δ(x)`, the slope at x* is 1 − η(1 + 1/δ).
The iteration is stable iff η < 2δ/(1+δ). (Skeptic correction of the block count.) Total blocks ≈ (1/η)·log((1−x_0)/(1−p+δ)) for the far phase, where T = 1−x, plus
log(1/tol)/(−log|1 − η(1+1/δ)|) for the local phase. The far phase dominates: for δ = 0.01, η = 0.005 it is 241 + 20,
against 252 observed (check S4). The earlier formula log(1/tol)/(η(1+1/δ)) gave 27. With η < 2δ/(1+δ) the total is
still Ω(1/δ), so the conclusion stands. Sharper (more
paradox-resistant) traders need smaller steps or a fixed-point solver inside the block.
*Check:* for δ = 0.01, η = 0.005 converges in 251 blocks and η = 0.015 in 90; η = 0.025 and η = 0.05 oscillate with
spread 8e−3 and 4e−2.

**R13 (theorem for the static part; the wealth dynamics are a numerical observation, not proved). A population: static weights get stuck; wealth in the state fixes it.** For any static
positive weighting of δ_k-truth-seeking traders, the fixed points lie in [p − max_k δ_k, p). With
δ = (0.5, 0.1, 0.01, 0.001) and equal weights, x* = 0.725, so the coarse traders set the price. If the block state
carries each trader's **wealth**, updated by the realised value (a Markov state, not attention to past prices), the
coarse traders lose. They sell at prices where χ is true. Wealth moves to the finest trader, and x* goes to
p/(1.001) = 0.799201 (day 1: 0.725; day 10: 0.762; day 100: 0.79920). This is the Budgeter's role
(full.txt 2690–2700) in Markov form.

---

## Implications for the network

* A block can be: prices ← fixed point (or a damped step) of Σ history-free traders. Coherence comes from projection
  or log-odds projection onto the polytope that Γ allows, or the gradient of the PDG inconsistency with axioms at
  confidence ∞. The frequencies come from a prior **at the level of models** (particles or model-holding traders, or the
  PDG joint), not from projecting local sentence priors (R4c).
* Kelvin's no-history rule costs convergence and every time-indexed theorem (R6, R8). Two compatible repairs:
  (1) carry per-trader **holdings and wealth in the block state** (a Markov state, R7, R13); (2) turn the time-indexed
  theorems into **ensemble martingale properties**, trained with an anytime proper loss and measured per block (R10,
  sec. 3).
* Inexploitability inside one run never forces trust in a pattern before the derivation finishes (R9). Timeliness is
  learned from the training distribution and should be reported as a timeliness profile e(t, d) at t = 20, 100, 500.
* Paradox resistance needs finite resolution (continuous traders, e.g. sigmoid ramps with a temperature δ). The cost
  is O(1/δ) blocks for plain iteration (R12).

---

## Skeptic (2026-10-02)

Script: `math/no-history-traders/skeptic_checks.py`, output in `skeptic_output.txt`. Parts 1–4 were rerun
(`no-history-traders/rerun/`) and match the stored outputs byte for byte. Paper numbering (4.1.1, 4.2.1, 4.2.3, 4.2.4,
4.3.6, 4.11.2, 4.12.2, 4.12.4, Lemma 5.1.1, eq. 6.1.1) and the line ranges checked against full.txt; Richardson Prop. 2
(2202.txt 290–305) and conditioning as inconsistency resolution (pdg2021.txt 804–811) checked and correct.

**Holds as stated:** R1, R1′ (π lies in the open cube because every kept sentence is undecided, so the uniform mixture
of models is interior), R2, R3 (both directions; the weak-book normal can be taken rational because C is a rational
polytope), R4(b), R4(d) (the fixed point is unique even with clipping: it is the minimiser over the cube of the strongly
convex w_c·dist²(V,C)/2 + |V−m0|²/2, and the unconstrained minimiser lies in the cube), R8 (theorem part), R10(i)–(iii),
R11, the static part of R13, and the section-3 table numbers.

**Corrected in place:**
* R4(a): the proof handled only interior fixed points; patched with the variational inequality. The result holds (S1).
* R4(c): the refutation stands; "the joint's geometry is lost" softened.
* R5: "(R3 with weights 1/K)" was wrong. There is no strong book, but a weak book exists whenever a model is unsampled (S2).
* R8: gain per resolved purchase is ≥ 2ε for a full share, not "≥ ε each time it buys". The conclusion is unchanged.
* R10(iv): Persistence corresponds to Doob's maximal inequality, not to martingale convergence (S5:
  Pr(sup|P_s−P_30| ≥ 0.5) = 0.104 ≤ bound 0.308).
* R12: the block-count formula ignored the far-field phase. Corrected; it was off by about 10× (S4). The O(1/δ) conclusion stands.
* Section 3 tower theorem: needs a *strictly* proper score.

**Downgraded:**
* **R6 → theorem only for sign-constant rate traders, conjecture in general.** The case analysis ("Σr = ∞", "if both
  are finite, the worth is bounded") assumes r_n, s_n ≥ 0. For sign-changing rates, "Σ r" is ambiguous. If it means the
  all-day sum, the "both finite ⇒ bounded worth" step is false. In S3, trader B has r = +1 on the last low day and −1 on
  the first high day, where r is unused. Likewise s = +1 on the last high day and −1 on the first low day. Its all-day
  partial sums stay in [0,1] and [−1,0], so the rule ignores B, yet B buys at 0.3 and sells at 0.7 once per cycle. On the
  schedule built for trader A, B's worth stays in [−0.30, 5.10] over 12 cycles: it exploits that market. If "Σ r" means
  the sum over the low days actually played, the rule is circular. This does **not** refute R6 for the full
  enumeration market: B was built from the A-only schedule, and the full schedule is uncomputable. It shows that the
  proof does not cover general history-free traders. A repair would have to pick each phase's stopping day so that it
  also controls sign-changing traders, for example stopping near the liminf of each trader's partial sums and targeting
  traders round-robin. That is unproven. TL;DR item 3 and the table were updated.
* **R9 → informal observation.** At a single block no exploitation statement exists. The rigorous content is R3 for a
  stationary market with D ≡ D_t.
* **R13 dynamic part → numerical observation.** That wealth moves to the finest trader is shown only by simulation.

**Refuted:** none of the results tagged theorem or known.

**Notes for the network:** the practical conclusions survive. These are: coherence by projection, frequencies from a
model-level prior, finite-resolution paradox traders costing O(1/δ) blocks, and ensemble martingale versions of the
time-indexed theorems. The claim that history-free traders cannot force convergence now rests on R6 restricted to
sign-constant traders. A full proof, or a counterexample market, is open.
