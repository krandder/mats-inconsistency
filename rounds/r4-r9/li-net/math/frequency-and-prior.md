# Frequency and prior: what PDG inconsistency needs a learned base measure for

Key: `frequency-and-prior`. Author: mathematician subagent, 2026-10-02.
Script: `math/frequency-and-prior/check.py` (numpy/scipy), output: `math/frequency-and-prior/output.txt`
(re-run: `cd math/frequency-and-prior && OMP_NUM_THREADS=4 python3 check.py`, about 1 min).

Short version. With no base measure, PDG inconsistency of a price table given the axioms is zero for
*every* coherent table, so on its own it cannot produce model frequencies. Add a base prior pi with finite
confidence and three things follow. The fixed point is pi(.|T). The inconsistency sits between
"coherence inconsistency" and "soft-label cross-entropy to model frequencies". Iterating the projection
block by block converges, with a Lyapunov function. Provability induction and pseudorandom frequencies
are then the two extreme cases of one fact: the best budget-N price is E[truth | what N blocks can see],
under the generator's prior over theories.

## 0. Setting and definitions

- Atoms x_1..x_K, worlds W = {0,1}^K. A *theory* T is a finite set of clauses; Mod(T) is the set of
  worlds that satisfy every clause. A *query* phi is a Boolean function on W.
- A *prior over models* pi: a distribution on W with full support. The *model frequency* of phi given T is
  f_pi(phi|T) = pi(phi and Mod T) / pi(Mod T). We write pi_T = pi(.|Mod T).
- **Generator G** (the "dataset generator"). Draw T as m random 2- or 3-clauses, conditioned on
  |Mod T| >= 3. Draw queries phi_1..phi_J: random literals, 2-clauses, 2-conjunctions and 2-XORs, plus one
  entailed query (a weakened axiom) and one refuted query (its negation). Draw one world w ~ pi_T. The
  labels are y_j = phi_j(w), or the soft labels f_j = f_pi(phi_j|T).
- **Price table** P in (0,1)^J: the network's output at block N.
- **Marginal (coherent) polytope** C(T) = conv{ (phi_1(w),...,phi_J(w)) : w in Mod T }. P is coherent iff
  P is in C(T). This is the propositional form of the paper's coherence: limit coherence and affine
  coherence (abridged Thm 4.2.10) both reduce to it in this finite setting.
- PDG notation follows Richardson 2022 (arXiv 2202.11862, local `2202.txt`):
  Inc_M(mu) = sum over edges L of beta_L E_mu KL(mu(Y|x) || p_L(Y|x)), and <<M>> = inf_mu Inc_M(mu)
  (eq. 1). Edges used here:
  - base pi(W) at confidence beta_0;
  - each axiom as the event "clause true" at confidence infinity (so mu must sit on Mod T);
  - each price as an unconditional edge Bern(P_j) on the binary variable Phi_j = phi_j(W) at confidence
    beta_P (with the deterministic edge W -> Phi_j at infinity);
  - data as the empirical distribution at confidence infinity (Props 3 and 5).
- h(x) = x log x + (1-x) log(1-x). D(x,P) = sum_j KL(Bern x_j || Bern P_j) is the Bregman divergence of
  H(x) = sum_j h(x_j).
- **Coherence inconsistency.** Coh_T(P) := <<T(inf), P(1)>> = min over x in C(T) of D(x,P).

## 1. Results

### R1. The Bayes-optimal price is the model frequency (known)
Under G, the minimiser over price functions of E[ sum_j CE(y_j, P_j(T,phi_j)) ] is
P_j = E[y_j | T, phi_j] = f_pi(phi_j | T). This is strict propriety of the log score (Gneiting & Raftery
2007, JASA 102(477):359-378, https://doi.org/10.1198/016214506000001437; DOI checked via Crossref by the skeptic). Also
E_y CE(y,P) = sum_j H(f_j) + D(f,P).

**R1b (theorem, one line).** Take a world-first generator instead: w ~ pi, then m clauses drawn iid
uniformly from the k-clauses that w satisfies. It gives the same posterior pi_T. The reason is that every
world satisfies exactly C(K,k)(2^k - 1) k-clauses, so p(T|w) is proportional to 1[w in Mod T].
Caveat: a generator that draws clauses non-uniformly (for example, Horn-biased) gives a posterior
pi(w) p(T|w), which is not pi_T. The network learns whatever the generator implies, so "model frequency"
is the correct target only for generators like these.

### R2. A coherent projection with a base prior recovers the model frequency (theorem; Richardson Props 2/4, scaled)
<<pi(beta_0), T(inf)>> = -beta_0 log pi(Mod T), and the unique minimiser is pi_T.
Proof: for mu supported on Mod T, KL(mu||pi) = KL(mu||pi_T) - log pi(Mod T), which is at least
-log pi(Mod T), with equality iff mu = pi_T. So the projection's prices are mu*(phi_j) = f_pi(phi_j|T).
The prior surprisal of the theory is the inconsistency. This is Prop 4 of 2202 with the axioms read as a
partial observation, scaled by beta_0.

### R3. Base prior + axioms + prices: closed form, tilt, sandwich (theorem)
Let M = {pi(beta_0), T(inf), P(beta_P)}. Then
<<M>> = -beta_0 log pi(T) + min over mu in Delta(Mod T) of [ beta_0 KL(mu||pi_T) + beta_P D(A mu, P) ],
where A is the J x |Mod T| query-indicator matrix.
(a) **Tilt form.** The minimiser is unique (strictly convex in mu). From the KKT conditions,
mu*(w) is proportional to pi_T(w) exp( -(beta_P/beta_0) sum_j phi_j(w) [logit x*_j - logit P_j] ),
where x* = A mu*. [Skeptic: the sum runs over non-forced rows only. A forced row (entailed or refuted query) has
logit x*_j = +-inf, but its KL term is constant on Delta(Mod T), so it drops out of the KKT conditions.]
(b) **Sandwich.**
-beta_0 log pi(T) + beta_P Coh_T(P)  <=  <<M>>  <=  -beta_0 log pi(T) + beta_P D(f, P).
Lower bound: drop the KL term. Upper bound: plug in mu = pi_T.
(c) **Limits.** As beta_0/beta_P goes to 0, <<M>> minus the surprisal tends to beta_P Coh_T(P): pure
coherence. As beta_0/beta_P goes to infinity, it tends to beta_P D(f,P), which is soft-label
cross-entropy to the model frequencies minus H(f). So "inconsistency relative to a learned base measure"
interpolates between coherence and cross-entropy, and the ratio of the two confidences is the knob.
(d) **Coherence is blind to frequency (refutes "PDG inconsistency alone gives frequency").** If P is in
C(T), then Coh_T(P) = 0 whatever its values. Numerically, the uniform-prior frequencies f_unif are
coherent (Coh = 0) but are off by sum KL(f||f_unif) = 0.87 under the true prior. Only the base edge
separates "coherent" from "correct frequency".

### R4. Coherence + cross-entropy decomposition = Bregman Pythagoras (theorem; standard Bregman projection result)
Coh_T(P) is the Bregman (log-score) projection of P onto C(T): x* = argmin over x in C of D(x,P). For
every coherent y in C(T), in particular y = f:
  D(f, P)  >=  Coh_T(P) + D(f, x*).
In words: excess log-loss to the model frequencies >= coherence inconsistency + excess log-loss of the
projected table. Proof: the three-point identity
D(y,P) = D(y,x*) + D(x*,P) + <grad H(x*) - grad H(P), y - x*>, and the last term is >= 0 by first-order
optimality of x* over the convex set C. Coordinates forced to 0 or 1 (entailed or refuted queries) have
y_j = x*_j and contribute 0.
[Skeptic, gap closed: the first-order condition needs grad H finite at x*, so x* must be interior on the free rows.
It is: the uniform mixture over Mod T puts every free row in (0,1), and h has infinite slope at 0 and 1, so the
Bregman projection of an interior P stays interior on the free rows (h is of Legendre type).]
Corollaries. (i) Coh_T(P) <= D(f,P) (R6 below). (ii) Projecting an incoherent table never increases the
expected log loss under any coherent truth, and strictly decreases it when the table is incoherent. This
is the "coherence dominates incoherence under proper scoring rules" theorem (Predd, Seiringer, Lieb,
Osherson, Poor, Kulkarni 2009, IEEE Trans. Inf. Theory 55(10), https://doi.org/10.1109/TIT.2009.2027573,
pp. 4786-4792; DOI checked via Crossref by the skeptic). R4 identifies Richardson's <<T(inf),P(1)>> with that projection for the log score.

### R5. The training objective: the same PDG gives "prior learning + label cross-entropy" (theorem; Props 3, 5 of 2202 combined)
Add data D = (w, T) at confidence infinity. mu is pinned to delta_w, which lies on Mod T, so
<<M + D(inf)>> = beta_0 (-log pi(w)) + beta_P sum_j CE(y_j, P_j).
Subtract the label-free part <<pi(beta_0), T(inf)>> = -beta_0 log pi(T) (R2), which does not depend on P:
  L(theta) := <<M_theta + D(inf)>> - <<pi_theta, T(inf)>> = beta_0 [-log pi_theta(w | Mod T)] + beta_P sum_j CE(y_j, P_theta,j).
So the conditional NLL of the base prior and the label cross-entropy are one objective. Numerically,
conditional ML recovers the prior's parameters (0.811/0.308/0.614/0.488/0.197/0.701 against true
0.8/0.3/0.6/0.5/0.2/0.7). Unconditional ML (forgetting that the world was conditioned on T) is biased
(0.768/0.347/...).
**Recommended end-of-block-20 loss:**
  L = E_G[ sum_j CE(y_j, P^(20)_j) ] + lambda E_G[ Coh_T(P^(20)) ],
~~or, equivalently in the PDG view, E_G[ <<pi_theta(beta_0), T(inf), P^(20)(beta_P), D(inf)>> - <<pi_theta, T>> ].~~
**[Skeptic: "equivalently" is REFUTED.]** By R5's own formula, the PDG-with-data objective equals beta_0 times the prior's conditional NLL
plus beta_P CE. It contains no Coh term. Skeptic S3 has two price tables with the same CE (4.2567) but Coh = 0 and
Coh = 0.1018: the PDG objective is the same for both, and CE + Coh is not. The correct link is only an inequality.
By Lemma 1 and R3(b), the PDG objective >= beta_P Coh_T(P) (and so does CE alone). The lambda*Coh term is an extra
design choice, not a consequence of the PDG view.
- Term (i), axioms respected: Coh_T >= 0, and it is 0 iff P is coherent.
- Term (ii), frequency: CE.
- They are one object only in the label-free M of R3: Coh and D(f,P) are its two ends in beta_0/beta_P. (With data at
  infinity the Coh end disappears; see the Skeptic note above.) With data at infinity
  (Prop 5), CE is the inconsistency of the network's prices with the data. By Lemma 1 (monotonicity),
  the label-free inconsistency <<M>> is at most the labelled one.
- By R6 the Coh term is redundant at the CE optimum. It is still worth keeping, because it needs no
  labels: it can be scored at blocks 100 and 500, and on theories with no labels.

### R6. Cross-entropy dominates coherence (theorem)
Coh_T(P) <= D(f,P) for every P, because f is in C(T) (it is the marginal vector of pi_T). Numerically,
0.273 <= 0.630.

### R7. Iterated projection with a base prior converges to model frequency: a block dynamics with a Lyapunov function (theorem; skeptic re-checked it, holds)
Block map: P_{t+1} = A mu_{t+1}, where mu_{t+1} = argmin over mu of beta_0 KL(mu||pi_T) + beta_P D(A mu, P_t).
(a) The unique fixed point is P = f. At a fixed point x* = P, so the tilt in R3(a) vanishes and
mu* = pi_T. This holds whatever the starting table.
(b) For t >= 1, P_t = A mu_t, so the step is a Bregman proximal-point step on F(mu) = beta_0 KL(mu||pi_T),
with Bregman function H(A mu) and step 1/beta_P. Descent: F(mu_{t+1}) + beta_P D(A mu_{t+1}, A mu_t) <= F(mu_t),
so KL(mu_t||pi_T) is non-increasing (a Lyapunov function in block depth).
(c) Rate: beta_0 KL(mu_T||pi_T) <= beta_P D(f, P_1) / (T-1). Proof: optimality of mu_{t+1}, plus the
three-point identity, plus convexity of F give
F(mu_{t+1}) - F(pi_T) <= beta_P [D(f, P_t) - D(f, P_{t+1})]. Sum and use monotonicity. The iterates are
interior because of the tilt form, so all gradients are finite (forced rows dropped). This is the
Chen & Teboulle 1993 argument (SIAM J. Optim. 3(3):538-543, https://doi.org/10.1137/0803026; DOI checked via Crossref by the skeptic).
[Skeptic: re-derived step by step. The per-step inequality also gives F(mu_{t+1}) - F(pi_T) <= beta_P[D(f,P_t) - D(f,P_{t+1}) - D(P_{t+1},P_t)].
It holds for t = 0 with an arbitrary interior P_0, which gives the variant bound beta_0 KL(mu_T||pi_T) <= beta_P D(f,P_0)/T.
On 5 more seeds with beta_0/beta_P in {0.01, 0.03, 0.1, 1}, the KL was monotone and the bound held every time.
The decay looked geometric, but the per-step ratio ranged from 0.94 to 0.99. At beta_0/beta_P = 0.01, KL is still
0.28 at t = 120, so "safe" does not mean "fast".]
Numerically the decay is geometric, much faster than the bound: KL = 7.8e-3 at t=20, 6.0e-5 at t=100,
2.2e-7 at t=200, against the bound's 0.13, 0.025 and 0.013. max|P_t - f| is 3e-4 at t=200.
Implication: a block that approximately runs this map has a depth-monotone objective, so running 100 or
500 blocks after training at 20 moves prices toward f and not away. beta_0/beta_P is a step size: large
means one block nearly finishes, small means slow but stable.

### R8. Provability induction and pseudorandom frequency are two cases of one Bayes fact (DOWNGRADED by the skeptic: theorem only in a declared view model; as stated, F_N is degenerate)
Let F_N be the sigma-algebra of what any N-block network can compute from (T, phi): its "view". The
minimiser of expected CE among N-block predictors (if the class is rich enough to represent E[y|F_N]) is
P_N = E_G[y | F_N].
- **Budget-N provability induction** (definition, no resolutions or deductive process): for every
  (T,phi) in supp G with T |= phi, including those whose shortest derivation needs more than N block
  steps, P_N(phi) >= 1 - eps. By the Bayes fact this holds with eps = 0 iff Pr_G(Ent | F_N) = 1 on
  entailed instances, that is, iff entailment is F_N-measurable on supp G. The "e.c. pattern" has to be
  visible within the budget, and the generator must give (near) zero mass to non-entailed look-alikes.
  If look-alikes have mass delta and frequency f', the price is 1 - delta(1 - f'): approximate
  provability induction.
- **Pseudorandom frequency** (analogue of abridged Thm 4.2.7, full Thm 4.4.2): if y is independent of
  F_N given a class with frequency p, then P_N = p.
- **[Skeptic: gap in the definition of F_N.]** "The sigma-algebra of what any N-block network can compute" is degenerate. Any
  injective function of the input (an embedding, or a real-valued code of (T, phi)) is computable in O(1) blocks,
  so F_N = sigma(T, phi) for every N. Then E[y | F_N] = f for every N, which would give full provability induction
  at every budget and contradict R10. The hypothesis "rich enough to represent E[y|F_N]" then fails, so R8 as
  stated is vacuous. A computational limit is not an information limit. The statement is a theorem only in a
  *view model*: fix a view map V_N (as R10 does), set F_N := sigma(V_N), and assume the N-block class contains
  every measurable function of V_N. Whether a real N-block net behaves like some view model is a modelling
  assumption (conjecture).
- [Skeptic: the pseudorandom-frequency bullet also needs the class to be F_N-measurable. If y is independent of
  F_N given an F_N-measurable class with frequency p, then E[y | F_N] = p. If the class is not visible within
  the budget, the conclusion can fail.]
- In LI the pattern is learned from the e.c. *sequence* by a trader that exploits its repetition. Here it
  is learned across training instances: the generator plays the role of the e.c. sequence, and the learned
  prior plays the role of the trader that found it.

### R9. Martingale across budgets: convergence and no expected net update (theorem for Bayes predictors of nested views V_N, per the R8 skeptic note; conjecture for a trained net)
If F_N is a subset of F_{N+1} (a block can implement the identity, as with residual connections), then
P_N = E[y|F_N] is a martingale in N. So it converges almost surely (Doob), and P_N = E[P_M | F_N] for
M > N. This is the analogue of No Expected Net Update (abridged Thm 4.3.9). It holds for the Bayes-optimal
predictor at each budget. A network trained only at N = 20 is not guaranteed to produce E[y|F_100] at
block 100; that is a conjecture, and it is what the depth-extrapolation tests measure.
[Skeptic: in check.py, view30 is not literally a refinement of view10. For d in 11..30, view30 = ("derived", y)
drops the template feature. The check still tests the right thing: a "derived" view reveals y, so
E[y | view30] = E[y | view10, view30], and the tower property applies to the joined view.]

### R10. Concrete families (theorem in the toy unit-propagation block model; conjecture for neural blocks)
Toy block model: one block = one round of unit propagation, plus O(1)-depth syntactic features.
Chains: anchor x_0 = b, constraints x_i XOR x_{i+1} = c_i, length n = 60, query x_d with d uniform in
1..n. Derivation length is d rounds.
- **Family A, learnable.** b = 1 and all c_i = 0 (an equivalence chain with a positive anchor). The feature
  "all constants 0 and anchor positive" is an AND, so it is constant-depth (AC0). It implies entailment.
  The Bayes budget-20 price on entailed queries with d > 20 is **1.000**.
- **Family B, not learnable at this budget.** b and c_i uniform. The value of x_d is the parity of d+1
  random bits, and no O(1)-depth feature carries information about it. The Bayes budget-20 price on
  entailed queries with d > 20 is **0.498**: they get the pseudorandom frequency, which is R8's other case.
- **Mix 50/50, template feature visible:** A gets 1.000 and B gets 0.504. **Template feature hidden:**
  every entailed long query gets 0.754, which is 1/2 + 1/4, the mixture.
- Caveat (conjecture/known). A neural block is not limited to one unit-propagation step per block. Parity
  along a chain can be done by pointer doubling in O(log n) blocks, so with N = 20, family B is
  non-learnable only for n > ~2^20. That bounded-depth attention cannot do parity at all is a known
  limitation for hard attention (Hahn 2020, TACL, arXiv 1906.06755; title checked by the skeptic). For soft attention, Hahn shows
  only that confidence decays with length. ~~soft attention does it only with precision blow-up~~ **[Skeptic: corrected.]**
  Chiang & Cholak 2022 (arXiv 2202.12172, abstract checked) build a fixed-depth soft-attention transformer that
  recognises PARITY with perfect accuracy, and with layer normalisation its cross-entropy goes arbitrarily close to 0.
  So if the chain is presented in order (prefix parity), family B may be learnable at *constant* depth. Then the
  "non-learnable only for n > ~2^20" bound is not the real limit, and family B is not a robust non-learnable
  probe for a transformer (conjecture either way; it depends on the encoding).
- ~~A family that is robustly non-learnable~~ **[Skeptic: DOWNGRADED to conjecture.]** Candidate: random 3-CNF near the
  satisfiability threshold, with the query a random literal. Chvatal & Szemeredi 1988 (JACM 35(4):759-768,
  https://doi.org/10.1145/48014.48016; DOI checked) prove exponential resolution size for *unsatisfiable random*
  k-CNF. Here T is satisfiable and the formula to refute is T and not-l, which is not a uniform random formula, so
  their theorem does not transfer directly to backbone literals. Feige's hypothesis (STOC 2002, "Relations between
  average case complexity and approximation complexity", https://doi.org/10.1145/509907.509985; checked) is about
  *polynomial-time refutation* of random 3-CNF with a large constant clause density, not about bounded-depth
  circuits near the threshold. "Entailed instances get a price strictly below 1" is therefore a conjecture.
- Numerical check of R9 on the mix (r1 = 10, r2 = 30): E[P_30 | view_10] equals P_10 to 4 decimals in
  every view class (0.5035 vs 0.5035; 1 vs 1; 0 vs 0).

## 2. Numerical check (K = 6), excerpt of output.txt
```
#models = 28  pi(Mod T) = 0.398  -log pi(T) = 0.9213
[R2] min_{mu on Mod} b0*KL(mu||pi) = 1.842607  vs -b0 log pi(T) = 1.842607  max|mu-pi_T| = 1.3e-09
noisy prices P: [0.381 0.853 0.703 0.059 0.256 0.652 0.79  0.171 0.215 0.239 0.997 0.006]
[R4] Coh(P) = 0.273233 (L-BFGS on softmax) ; 0.273233 (SLSQP)
     projected prices x* = [0.584 0.777 0.703 0.059 0.256 0.5 0.777 0.278 0.223 0.416 1. 0.]
[R6] Coh(P) <= sum KL(f||P): 0.27323 <= 0.62967 True
[R4] Pythagorean: sum KL(f||P) = 0.62967 >= Coh + sum KL(f||x*) = 0.6055 True
     Pythagorean margin on 200 random coherent y: min = 4.39e-03, none negative
     KKT: min over models v of <grad h(x*) - grad h(P), v - x*> = -2.1e-07 (solver tol)
     log-loss excess vs f before / after projection: 0.62967 -> 0.33226
     coherent but wrong prices (f_unif): Coh = 0.0  sum KL(f||f_unif) = 0.87475
[R3] sandwich, bP=1 (lo <= value <= hi for every b0; tilt form verified to <=3e-7)
  b0=0.01: lo=0.28245 value=0.28861 hi=0.63889   max|x-f|=0.1476
  b0=1.00: lo=1.19454 value=1.39375 hi=1.55098   max|x-f|=0.0849
  b0=100 : lo=92.40356 value=92.75606 hi=92.76000 max|x-f|=0.0029
[R7] t=20 KL=7.8e-03 (bound 1.3e-01); t=100 KL=6.0e-05 (bound 2.5e-02); t=200 KL=2.2e-07; monotone True
[R1] empirical CE = 6.617 vs H(f)+sum KL(f||P) = 6.6158 (200k sampled worlds)
[R5] conditional ML theta = [0.811 0.308 0.614 0.488 0.197 0.701] vs true [0.8 0.3 0.6 0.5 0.2 0.7];
     unconditional ML = [0.768 0.347 0.597 0.489 0.235 0.667] (biased)
[R1b] each world satisfies 140 of 160 3-clauses; world-first posterior = pi(.|Mod T) to 1e-17
[R10] budget-20 Bayes price on entailed, derivation > 20: A 1.000 | B 0.498 | mix visible A 1.000, B 0.504 | mix hidden 0.754
```

## 3. What this implies for the network
1. The block's fixed point needs a base measure. Use "minimise the inconsistency of my own price table
   given the axioms" with no base edge, and every coherent table is a fixed point (R3d). Frequencies are
   then arbitrary, and so is anything the initial guess happened to be. A learned base edge with finite
   confidence makes the fixed point unique and equal to the Bayes price (R2, R7a).
2. The base measure need not be explicit over 2^K worlds. Training with CE at block 20 makes the network
   carry it implicitly (R1). R5 shows that "learn the prior" and "fit the labels" are the same PDG
   inconsistency, so one loss suffices.
3. Loss at block 20: CE to labels (or soft model-frequency labels, which have lower variance), plus
   lambda times Coh_T. Coh_T needs no labels, so it can also be applied at
   random larger depths and on unlabelled theories. Computing Coh_T exactly needs the marginal polytope,
   which is NP-hard in general. For K <= ~16, enumerate. Beyond that, use sampled models or a local
   (pairwise) polytope relaxation, which gives a lower bound.
4. Give the block the R7 form (approximately): prices move toward the base-tilted projection with step
   beta_P/beta_0. Its Lyapunov function KL(mu_t||pi_T) makes depth extrapolation (100 and 500 blocks)
   safe in this model.
   [Skeptic: this holds for the exact map only. The map needs pi_T, and given pi_T one block could output f
   directly. With per-block approximation error there is no guarantee, and at a small beta_0/beta_P convergence is
   slow: KL is 0.28 after 120 blocks at a ratio of 0.01.]
5. Provability induction is a property of the training distribution, not of the inconsistency objective.
   It appears iff the generator makes entailment predictable from features the budget can compute (R8,
   R10). If the dataset contains long-derivation entailments with no visible pattern, the correct
   behaviour is the base rate, which is Thm 4.2.7's behaviour.

## 4. Ideas
- Self-distillation across depth. By R9, P_N = E[P_M | F_N], so training block N's prices with CE toward
  stop-grad(P_M), M > N, is unbiased for the Bayes target ~~if the deeper prices are calibrated~~
  **[Skeptic: REFUTED as stated; corrected]** only if P_M = E[y | G] for some G containing F_N (nested Bayes views).
  Calibration alone is not enough. Skeptic S4: a constant P_M = base rate 0.499 is perfectly calibrated, but
  E[P_M | z] = 0.499 while E[y | z] = 0.098 / 0.899. Constant prices are also a fixed point of pure
  self-distillation, so this loss needs a label anchor at block 20 against collapse.
- Train with soft labels f (exact for K <= 16) rather than sampled y. Same optimum (R1), with lower
  variance.
- Curriculum on beta_0/beta_P: start with the prior dominant (frequency), then anneal toward the
  coherence-dominant end.
- Probe sets that separate the properties: family A (provability induction should give 1 beyond the
  budget), family B (frequency 0.5 should hold until the budget reaches d), coherent-but-wrong-prior
  tables (Coh = 0 but CE > 0).

## 5. Open
- Does a trained 20-block network reach E[y|F_100] at block 100 (R9 for learned nets)?
- Is the decay of KL(mu_t||pi_T) in R7 always geometric (a linear-convergence condition for this Bregman
  proximal map)?
- A tractable, tight surrogate for Coh_T beyond K ~ 16 (sampled models vs local polytope), and its bias.
- Length generalisation of the family-A template feature from the training n to the test n.

## Skeptic

Skeptic pass, 2026-10-02. Files: `math/frequency-and-prior/skeptic.py`, its output `skeptic-output.txt`, and
`solver_md.py`, an accurate mirror-descent solver with Armijo backtracking that drops forced rows. Run:
`cd math/frequency-and-prior && OMP_NUM_THREADS=4 python3 skeptic.py`, about 1 min. I re-ran `check.py`, and its
output matches `output.txt` to the printed digits.

**Holds after the attack.**
- R1, R1b, R2, R3(a-d), R4 with its corollaries, R5's formula, R6, and R7(a-c). Every step was re-derived by hand.
- R1b also holds for the generator's mixed widths 2 and 3 and for without-replacement draws: each world satisfies
  45 2-clauses and 140 3-clauses, and the max posterior error over 50 theories is 8e-17.
- S1: 40 random (theory, prior, price) draws, with price noise up to 3 logits. With the accurate solver, the
  worst Pythagorean margin is -3e-5 and the worst KKT value is -6e-5, both solver tolerance. The sandwich slack
  is >= 1e-3, R6 holds every time, and the tilt-form error is <= 2e-4.
- S2: R7 holds on 5 more seeds and step ratios (monotone KL, bound satisfied).

**Numerical caveat about check.py.** The author's solver runs L-BFGS on softmax logits and can stall with a free
coordinate stuck at 0. That happened in 5 of 40 draws with extreme prices, overshooting Coh by up to 1.6e-3. On
those draws R4's Pythagorean check "fails" by -21 and R7's bound "fails" (first skeptic run). Both failures came
from the solver and disappear with `solver_md.py`. The reported seed-0 run (sigma = 1) is unaffected (its KKT
value is -2e-7). Use an accurate solver for any extension.

**Refuted or corrected.**
1. R5: "or, equivalently in the PDG view" is refuted. The PDG-with-data objective is beta_0 times the
   conditional NLL plus beta_P CE, with no Coh term (S3: equal CE, different Coh). Only the inequality
   PDG objective >= beta_P Coh survives.
2. Ideas, self-distillation: "unbiased if the deeper prices are calibrated" is refuted (S4). It needs nested
   Bayes views, and pure self-distillation can collapse to constant prices.
3. R10 caveat on Chiang & Cholak was misstated. They give fixed-depth soft-attention PARITY with perfect
   accuracy and, with layer norm, near-zero cross-entropy. Family B may be learnable at constant depth.

**Downgraded.**
- R8: from theorem to "theorem in a declared view model". F_N defined as "what any N-block net can compute" is
  degenerate (it equals sigma(T, phi)), which makes the statement vacuous. The pseudorandom bullet also needs an
  F_N-measurable class.
- R9: theorem only for nested *views*. The check's views are not literally nested, but the joined view makes the
  check valid.
- R10: the claim that random 3-CNF near the threshold is "robustly non-learnable" is now a conjecture.
  Chvatal-Szemeredi covers unsatisfiable random formulas, not backbone literals of satisfiable ones, and Feige's
  hypothesis concerns polynomial-time refutation at large density, not bounded-depth circuits.

**Citations.** I checked the DOI or arXiv title of Gneiting-Raftery, Predd et al., Chen-Teboulle,
Chvatal-Szemeredi, Feige (STOC 2002), Hahn, and Chiang-Cholak, using Crossref and arXiv. All match the cited
works, and page and issue numbers were added. Richardson 2022 eq. (1), Lemma 1, and Props 2-5 match local `2202.txt`
(Inc uses KL(mu || p), as used here). LI theorem numbers match local sources: abridged 4.2.7, 4.2.10, and 4.3.9;
full 4.4.2.
