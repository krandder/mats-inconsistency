# block-dynamics: what a repeated block can run (eurisko li-net, 2026-10-02)

Script: `math/block-dynamics/dynamics.py` (numpy/scipy; matplotlib from a `pip --target /tmp/bd_pylib` copy, plot only).
Output: `math/block-dynamics/output.txt`, plot `math/block-dynamics/inconsistency.png`, trajectories `traj.npz`.
Run: `cd math/block-dynamics && python3 dynamics.py` (about 1 min).
Skeptic pass (2026-10-02): `math/block-dynamics/skeptic/` (`skeptic.py`, `diag_*.py`, `output.txt`, `diag_hb.out`); see section 6.
Tags changed by the skeptic are marked **[skeptic: ...]** in place.

## 1. Setting and definitions

- Atoms x_1..x_n, worlds Ω = {0,1}^n. Axioms Γ (fixed); admissible worlds W = {ω : ω ⊨ Γ}.
- Query sentences φ_1..φ_Q; truth matrix A ∈ {0,1}^{W×Q}, A_{ωi} = [ω ⊨ φ_i]. Price vector p ∈ [0,1]^Q.
- **Coherent set** C = {A^T µ : µ ∈ Δ(W)} = conv{rows of A}. This is Garrabrant et al.'s coherence restricted to the
  finite sentence set and the propositional theory Γ (full.txt sec. 4.1 / C.10 "limit coherence"; affine coherence,
  Thm 4.5.5, full.txt line 1781).
- **Relations** R: affine (in)equalities a·p ≤ b (or =) valid on every row of A (Fréchet bounds for ∧, ∨, →;
  P(¬A)=1−P(A); axiom-forced entailments P(φ_i) ≤ P(φ_j) when Γ ⊨ φ_i→φ_j). Relaxation L(R) = {p ∈ [0,1]^Q : R holds} ⊇ C.
- Divergences: Euclidean E(p,q)=½Σβ_i(p_i−q_i)²; Bernoulli KL D_F(p,q) = Σ β_i KL(Bern p_i ‖ Bern q_i), the Bregman
  divergence of F(p) = Σ β_i [p_i log p_i + (1−p_i) log(1−p_i)] (∇F = β·logit).
- Π^F_K(q) = argmin_{p∈K} D_F(p,q) (Bregman projection), for K ∈ {C, L}.
- Claims q = the block-0 prices (incoherent in general), confidences β_i.

## 2. Results

### R1 (theorem). PDG inconsistency with axioms at ∞ = Bernoulli-KL projection onto C.
PDG M(q,β,Γ): one variable "world"; for each φ_i an edge carrying the cpd Bern(q_i) on the indicator 1[φ_i] with
confidence β_i; each axiom an edge carrying δ_1 on 1[γ] with confidence ∞. By eq. (1) of 2202.txt (line ~200),
Inc_M(µ) = Σ_i β_i KL(Bern µ(φ_i) ‖ Bern q_i) + ∞·Σ_γ KL(Bern µ(γ) ‖ δ_1). The last term is 0 if µ(W)=1 and +∞ otherwise
(KL(·‖δ_1)=∞ unless µ(γ)=1; convention ∞·0=0). Hence ⟨⟨M⟩⟩ = min_{µ∈Δ(W)} D_F(A^Tµ, q) = min_{p∈C} D_F(p,q), attained at
p* = Π^F_C(q). D_F(·,q) is strictly convex in p (h'' = 1/(p(1−p)) > 0), so p* is unique; µ* is not.
**[skeptic: holds.]** Two refinements: (1) confidence ∞ on the axiom edges is not needed; any β_γ > 0 gives the same
⟨⟨M⟩⟩ because KL(Bern µ(γ) ‖ δ_1) = ∞ whenever µ(γ) < 1. (2) uniqueness of p* needs β_i > 0 on every claimed sentence;
an unclaimed query sentence (β = 0) is set only by the tie-break below.
The γ→0 tie-break of 2202.txt (line 246, "unique best distribution") selects among µ with A^Tµ = p* the one of maximum
entropy (IDef with α=0 is −H(µ)). **[skeptic: holds for every α, not only α=0:** on the set A^Tµ = p* each edge term
α_i H_µ(1[φ_i]) = α_i h(p*_i) is constant and H(1[γ]) = 0, so IDef = −H(µ) + const there; the selection argument is the
proof of Prop. 3.3 in pdg2021.txt (lines ~1189-1228).]
Check: variational inequality min_ω ⟨∇D_F(p*), A_ω − p*⟩ = −7e−15 (≈0, optimal).

### R2 (theorem). What coherence at the fixed point buys (LI properties, finite propositional case).
Every p ∈ C satisfies every affine relation valid on W (linear in µ, holds on each vertex). So at p*: Γ ⊨ φ ⇒ p*_φ = 1,
Γ ⊨ ¬φ ⇒ p*_φ = 0, all Boolean identities hold exactly, Γ ⊨ φ→ψ ⇒ p*_φ ≤ p*_ψ. "Provability induction" here is free: no
derivation is needed because C is defined by world enumeration. The hardness has moved into computing C (R7).

### R3 (theorem, standard, **conditional** [skeptic: downgraded from unconditional]). (a) anchored: projected gradient on Inc over worlds converges to Π^F_C(q) *if the iterates stay in a region where prices are in [ε,1−ε]*.
Block state µ_t ∈ Δ(W) (prices p_t = A^Tµ_t), claims q carried unchanged (input injection).
f(µ) = D_F(A^Tµ, q) is convex (convex ∘ linear). On any region where all non-decided prices stay in [ε,1−ε], ∇f is
Lipschitz with L_E ≤ ‖A‖²·max β/(ε(1−ε)), so projected gradient with step 1/L_E gives f(µ_T)−f* ≤ L_E‖µ_0−µ*‖²/(2T)
(Nesterov 2004, Thm 2.2.14 (unverified numbering)); price error then follows as in R5. The region assumption is real:
L_E blows up near the cube boundary, which is why (a) is slow in practice.
**[skeptic: the region assumption is not proved and fails in practice.** Nothing makes {p ∈ [ε,1−ε]} invariant: Bernoulli KL
is finite at p ∈ {0,1}, so sublevel sets of f reach the boundary, where ∇f is infinite. Skeptic S5: 20 claim vectors
with entries near 0.002/0.998, same step rule (1/λ_max of the Hessian at p*): price error < 1e−3 within 20000 iterations
in only 10/20 (reference: EG at the proved step, VI ≈ 1e−12); one run touches the boundary (p_i < 1e−9) 1181 times, and
in `diag_s5.py` 2 of 5 inspected runs have non-monotone Inc (step exceeds the local smoothness away from p*), while for
others the step 1/λ_max shrinks like min p*(1−p*) and progress stalls. So (a) has no usable rate as a block.]
Numerics: step 1/λ_max(Hessian at p*) = 4.6e−4; Inc gap < 1e−3 after **714** iterations, ‖p_t−p*‖_∞ < 1e−3 after
**2183**; linear in practice (0.9985/iter).

### R4 (theorem + refutation). (a) self-referential: claims = current prices.
If the block has no copy of q_0 and the claims are its own prices q_t, the inconsistency is d(q) = min_{p∈C} D_F(p,q).
(i) Fixed set: d(q)=0 ⇔ q ∈ C, so every coherent point is a fixed point; the attractor is the set C, not a point.
(ii) **Mirror (dual) flow preserves the projection.** Update ∇F(q_{t+1}) = ∇F(q_t) + η(∇F(Π(q_t)) − ∇F(q_t)), η∈(0,1].
Proof: p* = Π^F_C(q) iff ⟨∇F(q) − ∇F(p*), p − p*⟩ ≤ 0 ∀p∈C (Bregman projection characterisation). The update multiplies
the vector ∇F(q) − ∇F(p*) by (1−η) ≥ 0, so the inequality still holds and Π(q_{t+1}) = p*. Then
∇F(q_t) − ∇F(p*) = (1−η)^t(∇F(q_0) − ∇F(p*)) → 0: geometric convergence to Π^F_C(q_0). For Euclidean F this is the
gradient flow of ½dist²(q,C), a straight line to the projection.
(iii) **Refuted:** "Euclidean gradient flow of the PDG inconsistency on q converges to Π(q_0)". By the envelope theorem
∇_q d(q) = −∇²F(q)(p*(q) − q), so the flow is dq/dt = H(q)(Π(q)−q) with H = diag(β/(q(1−q))); this is not the dual
segment, so Π(q_t) drifts.
Numerics: mirror update max drift of Π(q_t) = 6e−16; primal flow ends in C (dist 1e−8) at a point **0.037** (ℓ∞) away
from Π(q_0). **[skeptic: (ii) and (iii) hold.** Closed-form 2-sentence check (C = {p_1 ≤ p_2}, q_0 = (0.9, 0.2),
Π(q_0) = (0.6, 0.6)): primal ODE integrated with rtol 1e−11 ends at (0.5355, 0.5355), drift 0.064; mirror ODE drift 1e−16.
So the 0.037 is not a discretisation artefact.]
(iv) Both need the projection Π as an inner oracle; the self-referential form only moves the work.

### R5 (theorem). (b) exponentiated gradient over worlds = damped simultaneous IPF; rate Σβ·log|W|/T; max-entropy limit.
Update µ_{t+1}(ω) ∝ µ_t(ω) exp(−η ∂f/∂µ(ω)).
(i) Identity: ∂f/∂µ(ω) = Σ_i β_i[1φ_i(ω) log(m_i/q_i) + (1−1φ_i(ω)) log((1−m_i)/(1−q_i))] with m = A^Tµ, so
µ_{t+1}(ω) ∝ µ_t(ω) Π_i (q_i/m_i)^{ηβ_i [ω⊨φ_i]} ((1−q_i)/(1−m_i))^{ηβ_i [ω⊭φ_i]}: a product of single-claim IPF
rescalings, each damped to power ηβ_i. With ηβ_i = 1 and one claim it is exactly IPF (Csiszár 1975). Check: 2e−17.
(ii) Relative smoothness (proof): the linear terms of f drop out of its Bregman divergence, so
D_f(µ,ν) = Σ_i β_i KL(Bern µ(φ_i) ‖ Bern ν(φ_i)) ≤ (Σ_i β_i) KL(µ‖ν) by the data-processing inequality (Bern µ(φ_i) is the
image of µ under ω ↦ 1φ_i(ω)). So f is L-smooth relative to negative entropy with L = Σβ_i. Mirror descent with step 1/L
then satisfies f(µ_T) − f* ≤ L·KL(µ*‖µ_0)/T ≤ (Σβ_i) log|W| / T from uniform µ_0 (Lu, Freund, Nesterov 2018,
https://arxiv.org/abs/1610.05708; Bauschke, Bolte, Teboulle 2017).
(iii) Price error: generalized Pythagoras for Bregman projections onto a convex set, D_F(p,q) ≥ D_F(p,p*) + D_F(p*,q) for
p ∈ C, gives Inc(p) − ⟨⟨M⟩⟩ ≥ D_F(p,p*) ≥ 2β_min‖p−p*‖² (Bernoulli Pinsker). So
‖p_T − p*‖_2² ≤ (Σβ_i) log|W| / (2 β_min T).
(iv) Implicit bias (theorem, sketch; **[skeptic: holds; the 'consequence' sentence is narrowed]**): log µ_t − log µ_0 lies in span{1, columns of A} for all t, so every iterate is in the
exponential family E = {µ ∝ µ_0 exp(Aθ)}. Any limit lies in cl(E) ∩ {A^Tµ = p*}, which contains exactly one point, the
I-projection of µ_0 onto the moment set (Csiszár 1975). From uniform µ_0 this is the max-entropy distribution with prices
p*, the same µ as the PDG γ→0 tie-break in R1. Consequence: a sentence ψ that no claim constrains gets the
max-entropy (model-frequency) price, |{ω∈W: ω⊨ψ}|/|W| when the claims are uninformative.
**[skeptic: narrowed.** "Uninformative" has to mean q = A^T·uniform exactly (then p* = q and µ = uniform); for any other
claims ψ gets its max-entropy conditional price, not the model frequency. Example: atoms a, b, Γ empty, claim P(a) = 0.9,
ψ = a∧b: max-entropy price 0.45, model frequency 0.25. Also "model frequency over W" is not the LI property of the same
name (frequencies along a sequence of independent sentences), so implication 1 below is reworded. Side fact: here the
max-entropy µ has worlds of mass 2e−60 (p* lies on a face of C), so the limit is in cl(E) \ E, as the proof allows.]
Numerics: bound (ii) holds at every T (L=22, |W|=48). **[skeptic: (i)-(iii) hold; S4: D_f(µ,ν)/((Σβ)KL(µ‖ν)) ≤ 0.16 on
2000 random pairs with random β. But η = 4/L is outside the theorem and is not safe:** on 9 of 20 extreme-claim vectors
(S5b) EG at η=4/L does not converge (normalised VI −0.8 to −8, price error 0.12-0.93 after 30000 steps); the case
inspected in `diag_eg.py` is a period-2 cycle (|p_T − p_{T−1}| = 0.96, |p_T − p_{T−2}| = 1e−16, Inc 32.70 vs optimum
29.71), while η ∈ {1, 1.5, 2, 2.5, 3}/L all converge on that case. The 44/64 figures below hold for this q_0 only.] Inc gap < 1e−3: **178** iterations (η=1/L), **44** (η=4/L);
price error < 1e−3: **260** / **64**. Linear in practice (0.983 / 0.932 per iter). EG limit vs max-entropy µ (dual
solve): 1.3e−9; entropies 1.7555142 vs 1.7555142.

### R6 (known + refutation). (c) row-action / IPF-like projections onto one relation at a time, onto L.
- Equalities only: cyclic Bregman projections onto hyperplanes started at q converge to Π^F_L(q) (Bregman 1967;
  Csiszár 1975 for IPF) (unverified).
- Inequalities: plain cyclic projections (POCS) converge to *some* point of L, not the projection. **Refuted** for the
  naive Sinkhorn-on-inequalities idea: numerics, POCS ends feasible (viol 0) but **0.056** from Π^E_L(q).
- Fix (known): carry one dual variable λ_j ≥ 0 per inequality and do exact dual coordinate ascent: KKT is
  ∇F(p) = ∇F(q) − R^Tλ; step on row j: find δ with a_j·p(δ) = b_j where ∇F(p(δ)) = ∇F(p) − δa_j, clip δ ≥ −λ_j.
  Euclidean F = Hildreth's method (Hildreth 1957); general Legendre F = Bregman's row-action method
  (Censor & Zenios 1997, ch. 6) / Dykstra with Bregman projections (Bauschke & Lewis 2000) (all unverified).
  Converges to Π^F_L(q). Rate: linear for polyhedral L (Euclidean: Deutsch & Hundal 1994; dual coordinate ascent:
  Luo & Tseng 1992/93) (unverified).
- Numerics (30 relations, + box rows for Euclid): relation violation < 1e−3 after **7 sweeps** (Hildreth, 74 rows/sweep)
  and **9 sweeps** (KL row action, 30 rows/sweep); parallel (Jacobi) versions: **64** layers (Euclid, step 1/‖M‖²) and
  **94** layers (KL, averaged 1/k); error to Π_L < 1e−3 at 7 / 9 / 66 / 100.

### R7 (theorem + known). The relations are a relaxation: projecting onto L is not projecting onto C.
C ⊆ L always (each relation holds on each world). Numerics: ‖Π^E_L(q)−Π^E_C(q)‖_∞ = **0.24** with the 30 relations and
**0.21** with all 55 generated candidates; Π^{KL}_L(q) is at distance **0.285** from C (not coherent).
**[skeptic: corrected.** The 0.285 is the ℓ∞ residual of an approximate Euclidean projection onto C, an upper bound; the
exact ℓ∞ distance to C by LP is **0.244** for Π^{KL}_L(q) (0.230 for Π^E_L(q), 0 for Π^{KL}_C(q)). Conclusion unchanged.]
Known: deciding p ∈ C (probabilistic satisfiability, PSAT) is NP-complete (Georgakopoulos, Kavvadias, Papadimitriou
1988, J. Complexity 4(1):1-11) (**[skeptic: citation and result confirmed via secondary sources, e.g. Biazzo, Gilio,
Lukasiewicz, Sanfilippo, "Probabilistic logic under coherence: complexity and algorithms"; primary not read]**). Corollary (theorem, given that): no polynomial-time computable family of
polynomially many linear relations, with or without polynomially many auxiliary sentences (extended formulation), defines
C for all theories, unless P = NP; otherwise LP would decide PSAT. So a fixed-width block with local relations cannot be
exactly coherent on general theories.
**[skeptic: holds, and can be made unconditional for the relation/extended-formulation part:** with Γ empty and the query
set {atoms} ∪ {pairwise conjunctions}, C is the correlation polytope COR(n), whose extension complexity is 2^{Ω(n)}
(Fiorini, Massar, Pokutta, Tiwary, de Wolf, https://arxiv.org/abs/1111.0837; the abstract states it for the cut polytope,
a linear image of COR(n); that the proof is for COR(n) is my recollection (unverified)). So no polynomial-size family of
linear relations with auxiliary sentences describes C, with no P≠NP assumption and no uniformity needed. The P≠NP form is
still needed for blocks that compute something other than an LP over a fixed polytope.]
Known exact case (unverified): pairwise/clique local consistency equals the marginal polytope when the constraint graph
is a tree / has a junction tree (Wainwright & Jordan 2008, ch. 4); Sherali-Adams level k is exact at k = treewidth+1.

### R8 (theorem, constructive). What a block can implement exactly.
- **Parallel Hildreth = one residual ReLU layer.** State (p, λ) with p = q − Mᵀλ (q injected):
  λ ← relu(λ + τ(M(q − Mᵀλ) − b)) on inequality rows (no relu on equality rows); hidden width = number of relations.
  Instance-specific M (sparse, 2-3 entries per row) = one attention read (relation token reads its 2-3 sentence prices),
  one elementwise ReLU, one attention write. Exact, linear convergence (64 layers to 1e−3 here).
- **Cyclic Hildreth by colour classes.** Rows with disjoint supports commute, so one sweep = χ layers, χ = chromatic
  number of the row-conflict graph ≤ max row-degree + 1. Gets the 7-sweep cyclic rate at χ layers per sweep.
- **EG / damped IPF over worlds = one attention-like layer** (**[skeptic: proved rate only for η ≤ 1/L = 1/Σβ; η = 4/L
  can 2-cycle, see R5]**) (keys = world tokens, values = A rows): m = Aᵀsoftmax(z),
  z ← z − η A g(m,q), g elementwise log-ratio. Exact if the block has log/softmax; width |W| ≤ 2^n.
- KL row action needs a 1-D root per row: not exact in finitely many layers; a few Newton steps (quadratic) approximate it.
- Mirror self-referential update (R4 ii) needs an inner projection: not one layer.

### R9 (theorems + counterexample + numerics). (d) learned update p_{t+1} = f_θ(p_t, structure).
(i) **Necessary for un-anchored iteration** (no copy of q_0): if f^t(p_0) → Π(p_0) for every p_0 and f is continuous,
then Π∘f = Π (f preserves projection fibres) and f(p) = p for every p ∈ C. Proof: lim f^t(f(p_0)) = lim f^{t+1}(p_0), so
Π(f(p_0)) = Π(p_0); for p ∈ C, Π(p) = p and continuity give f(p) = lim f^{t+1}(p) = p. The mirror update of R4 satisfies
this; generic learned f does not. Recommendation: inject q_0 (anchored form) so the target is one fixed point.
(ii) **Sufficient: contraction.** If z ↦ f(z; q_0) is a ρ-contraction (any norm) with fixed point z_f and the train loss
gives ‖f^{20}(p_0) − p*‖ ≤ ε, then for every T ≥ 20: ‖f^T(p_0) − p*‖ ≤ ε + 2ρ^{20}‖p_0 − z_f‖. Proof:
‖z_f − p*‖ ≤ ‖z_f − f^{20}p_0‖ + ε ≤ ρ^{20}D + ε and ‖f^T p_0 − z_f‖ ≤ ρ^T D. So 500-block error is bounded by 20-block
error. Weaker sufficient class: averaged nonexpansive f with Fix f = {p*} (Krasnosel'skii-Mann; Bauschke & Combettes
2017) (unverified); local: spectral radius of ∂f at p* < 1 suffices, ≤ 1 is necessary (Ostrowski).
(iii) **Linear f: training at 20 does extrapolate.** ‖M^{20}‖ ≤ ε < 1 ⇒ ‖M^{500}‖ ≤ ε^{25} (submultiplicativity).
Numerics: 2.0e−3 → 6.6e−78.
(iv) **Counterexample: end-only loss at 20 does not determine 500.** A block with a counter coordinate (ReLU-realisable)
that jumps to p* at block 20 and drifts afterwards has zero train loss. Numerics: error 0 at T=20, 0.86 at T=100 and 500.
This is the "overthinking" failure (Bansal et al. 2022, https://arxiv.org/abs/2202.05826: fixed by recall/input
injection and progressive loss over random iteration counts). Path independence predicts extrapolation in equilibrium
models (Anil et al. 2022, https://arxiv.org/abs/2211.09961). DEQ (Bai, Kolter, Koltun 2019) and Jacobian regularisation
(Bai et al. 2021); short-horizon bias of learned optimisers (Wu, Ren, Liao, Grosse 2018); unrolled ISTA needs weight
coupling to converge (Chen et al. 2018) (these four unverified).
(v) Learned parameters of a convergent template: heavy-ball parallel Hildreth with learned (τ, momentum) selected only by
max error at T=20 on 8 random claim vectors picks τ=2.0/‖M‖², momentum 0.8: error 4.9e−3 @20, 3.4e−7 @100, 3e−16 @500.
Parameters that explode at 500 (τ=2.6, mom 0.8: 1.5e3 @20) are already rejected by the T=20 loss; the template has no
hidden clock to exploit. **[skeptic: refuted as a general claim.** (1) τ=2.8/‖M‖², mom 0.3 has max error 1.3e−2 @20 over
8 claim vectors (smaller than the convergent τ=2.0, mom 0: 6.4e−2), yet one of the 8 runs goes 1.3e−2 @20, 1.7e−2 @100,
4.2e−2 @200, 9.5e34 @500 (skeptic S6): a slowly growing unstable mode, excited only once some ReLU rows switch, acts as
the hidden clock. (2) The pick τ=2.0, mom 0.8 is selected on 8 inputs; on 200 fresh inputs it explodes on 4 (2%, error
9.8 to 370 already at 20, so an all-inputs loss would catch these, an 8-input loss did not; `diag_hb.out`). Heavy ball is
linearly stable only for τ‖M‖² < 2(1+mom), so both picks sit at or beyond the edge where the projection rows decide.] Conjecture: learning only the parameters of a provably convergent template (step sizes, row
weights, colour schedule) extrapolates from 20 to 500 whenever the train loss at 20 is small for all inputs; a free f does
not, as (iv) shows. **[skeptic: stays a conjecture, only with the parameters constrained by construction to the provable
range (e.g. τ = 2·sigmoid(a)/‖M‖², no momentum, or momentum with τ‖M‖² < 2(1+mom) and the projected-heavy-ball
conditions); a free scalar step already admits the clock of point (1).]**

## 3. Numerical check (summary of output.txt)

Random theory: 6 atoms, 2 random 3-clause axioms (|W| = 48 of 64 worlds), Q = 24 sentences (6 atoms, 2 axioms fixed at 1,
16 random compounds not decided by Γ), 30 relations drawn from 55 valid candidates, claims q_0 ~ U(0.05, 0.95).
Iterations to 1e−3 (plot: inconsistency.png, left = PDG inconsistency gap, right = relation violation):

| dynamics | target | Inc gap / violation < 1e−3 | price error < 1e−3 |
|---|---|---|---|
| (a) projected gradient on Inc, worlds | Π^KL_C | 714 | 2183 |
| (b) EG η=1/L (proved rate) | Π^KL_C | 178 | 260 |
| (b) EG η=4/L (no proof; fails to converge on 9/20 extreme q, skeptic S5b) | Π^KL_C | 44 | 64 |
| (c) Hildreth cyclic, Euclid | Π^E_L | 7 sweeps | 7 sweeps |
| (c) Bregman row action cyclic, KL | Π^KL_L | 9 sweeps | 9 sweeps |
| (c) parallel Hildreth (1 ReLU layer each) | Π^E_L | 64 | 66 |
| (c) parallel KL row action | Π^KL_L | 94 | 100 |

Caveat: (c) reaches Π_L, which is 0.21-0.29 away from coherent (R7). Only (a)/(b) reach C, at width |W|.

## 4. Implications for the network

1. The PDG reading of Kelvin's idea is exactly a Bregman (Bernoulli-KL) projection of the block-0 prices onto the coherent
   polytope; its fixed point has the LI coherence properties and, with the γ→0 / EG tie-break, max-entropy prices on
   unclaimed sentences (equal to the model frequency over W only when the claims are the uniform-world prices).
2. Best exact block for small theories: EG / damped IPF over world tokens (one softmax-attention layer per iteration,
   proved O(Σβ log|W| / T) at η ≤ 1/Σβ and linear in practice; 178-260 blocks to 1e−3 here at the proved step, 44-64 at
   η = 4/L, which fails to converge (2-cycles) on extreme claims, so do not let training push η above the proved range). Cost 2^n world tokens: fine for n ≤ ~12.
3. For larger theories, price-only blocks running relations reach a relaxation L, not C. Coherence then needs either the
   right relations (clique/junction-tree marginals, Sherali-Adams lifts with auxiliary conjunction sentences) or
   exponential width; no fixed poly-size block is exactly coherent in general (PSAT).
4. Keep q_0 in the residual stream (anchored form). Without it, the iteration must preserve projection fibres (R9 i).
5. Train with random block counts and a fixed-point residual loss, or learn only parameters of a convergent template
   (Hildreth/EG step sizes, schedules) reparametrised so they cannot leave the provable range; end-only loss at 20 blocks
   alone admits clock solutions that fail at 500, even inside a template with one free step size (R9 v, skeptic).

## 5. Open
- Provability induction "before derivations finish" in the relaxation: does the number of sweeps needed to push an
  entailed sentence to 1 in L scale with the resolution-derivation length? (conjecture, untested)
- Paradox resistance and self-reference: not covered; C is finite and propositional here.
- Linear-rate constants for EG on this problem (relative strong convexity holds only in p, not µ).

## 6. Skeptic (2026-10-02)

Rerun: `dynamics.py` reproduces `output.txt` exactly (only RuntimeWarnings differ: np.where evaluates log(1−q) on the
axiom coordinates with q = 1, masked out; harmless). Independent checks: `skeptic/skeptic.py` → `skeptic/output.txt`,
`skeptic/diag_s5.py`, `skeptic/diag_eg.py`, `skeptic/diag_hb.py` → `skeptic/diag_hb.out`.

| result | verdict | evidence |
|---|---|---|
| R1 PDG inconsistency = Bernoulli-KL projection onto C | holds (+2 refinements: ∞ not needed; tie-break holds for all α) | eq. (1) 2202.txt; Prop. 3.3 proof pdg2021.txt |
| R2 coherence at the fixed point | holds (trivial) | — |
| R3 projected gradient over worlds | **downgraded to conditional**: region assumption unproved; misses 1e−3 in 20000 its on 10/20 extreme claims | S5 |
| R4 (ii) mirror flow keeps projection; (iii) primal flow refuted | holds | S1 closed form: drift 0.064 vs 1e−16 |
| R5 (i)-(iv) EG = damped IPF, rate L·log #W / T, max-entropy limit | holds; "model frequency" consequence narrowed; η=4/L not covered and fails to converge on 9/20 | S3, S4, S5b, diag_eg |
| R6 row-action / Hildreth / POCS | holds as known (citations unverified, unchanged) | original numerics |
| R7 L is a relaxation; PSAT | holds; **0.285 corrected to 0.244** (exact LP ℓ∞); strengthened unconditionally via Fiorini et al. | S2; arXiv 1111.0837 |
| R8 constructions | holds; EG layer only proved at η ≤ 1/L | — |
| R9 (i)-(iv) | holds (proofs checked; (i) needs no continuity for Π∘f = Π) | — |
| R9 (v) "exploding parameters are rejected at T=20; template has no clock" | **refuted** | S6: 1.3e−2 @20 → 9.5e34 @500; doc's pick explodes on 4/200 fresh inputs |

Citations checked: Lu-Freund-Nesterov arXiv 1610.05708 (title/authors confirmed; theorem statement not read), Bansal et al.
arXiv 2202.05826 (overthinking, recall, progressive training confirmed from abstract), Anil et al. arXiv 2211.09961 (path
independence ⇔ upward generalisation, confirmed from abstract), Georgakopoulos-Kavvadias-Papadimitriou 1988 (bibliographic
data and NP-completeness confirmed via secondary sources). Others remain marked (unverified).

Net effect for the network: the exact story (R1, R2, R4, R5 at η ≤ 1/Σβ, R7) survives. Two practical claims weaken:
the fast EG step is not safe, and "learn the step sizes of a convergent template" extrapolates only if the parametrisation
keeps them inside the provable range; otherwise a slow unstable mode is a clock that passes the 20-block loss.
