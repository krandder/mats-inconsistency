# Round 5 Red-Team Report: Econometrica vs. ICLR Smoking-Gun Audit of R17

**Scope note.** I have not re-read `THEORY-4.md`, `STUDY-LIR.md`, or `01-econometrica-vs-iclr-critical-audit.md` in this turn. Everything below works from the summary in the prompt. Where my analysis depends on implementation details I cannot see, I say so. All numbers in §4 are pre-registered priors, not results.

---

## 0. Executive Verdict

1. **Theorem R5.1 is not DSIC as stated.**
   - It is a Groves/Clarke-pivot mechanism over a forecast pool. It achieves at most ex-post IC, and only under two conditions: (C1) a Clarke-term independence condition, and (C2) an aggregation-sufficiency condition that fails generically under misspecification.
   - Under false-name splitting into complements, it is strictly manipulable. Per-identity rent does not fix this. A Yokoo–Sakurai–Matsubara-style impossibility applies.
   - There is also a structural flaw. **Strict Gershgorin dominance is incompatible with exact clone neutrality of the pool weights** (§1.4). The "Gershgorin" in the name is the problem, not the solution.

2. **R17-E2 contains a conceptual trap: what crosses a detached boundary must be one of three message types.**
   - (i) A vector cotangent, which is backprop.
   - (ii) A target, which is target propagation or synthetic gradients.
   - (iii) A scalar payment, which is RL/ES/zero-order.
   - There is no fourth option. A market with outcome-verified payments is (roughly) a statistical-query algorithm. It cannot beat the $n^{\Omega(k)}$ SQ lower bound for $k$-sparse parity, and neither can SGD.
   - The honest claim is therefore "matches backprop at comparable compute with type-(iii) messages and bounded communication", not "learns parity without backprop by magic".
   - Separately, an additive logit pool cannot represent XOR at all. A composer module is mandatory.

3. **R17-E3 has two internal contradictions.**
   - Profit mitosis creates clones, and clone annihilation then kills them. Simultaneous bankruptcy of substitutes prunes both copies.
   - On SQ-hard tasks, no outcome statistic can distinguish a plateaued parity learner from junk before the breakthrough. Infant-mortality control is therefore necessarily a bounded subsidy, never a free lunch. The design question is how to bound it.

4. **Prior-art threat is high for both venues.** Relevant work includes:
   - Lambert et al. 2008 (self-financed, sybilproof wagering mechanisms).
   - Storkey 2011 (machine-learning markets).
   - Chang et al., ICML 2020 (decentralized RL via local Vickrey auctions, with truthfulness proofs). This is very close to R17.
   - Jaderberg et al. 2017 (synthetic gradients, which are literally "prices" on activations).
   - Belilovsky et al. 2019 / Nøkland & Eidnes 2019 (local losses).
   - Meulemans et al. 2020 (target propagation ≈ Gauss-Newton).
   - Bates–Granger 1969 (optimal forecast combination $w\propto\Sigma^{-1}\mathbf 1$, of which R5.1's $w^*$ is a heuristic regularization).
   - Ghorbani & Zou 2019 (Data Shapley); Bergemann–Välimäki 2010 (dynamic VCG).
   - Any submission that does not position against these will be desk-level rejected.

---

## 1. Critique of Theorem R5.1 (Econometrica Bar)

### 1.1 Formal setting

Agent $m$ has private signal $s_m$ and reports $r_m=(q_m,\hat\beta_m,\hat k_m)$, where $q_m$ is a logit vector. The clearinghouse computes

$$
A(r)=\hat\Lambda+I+\rho\hat G_{\text{off}},\quad w^*=A^{-1}\mathbf 1,\quad \mu^*=\sigma\Big(\textstyle\sum_j w_j^* q_j\Big),
$$

$$
W_m=\beta_0\big[\ell(\mu^*_{-m},y)-\ell(\mu^*,y)\big]-c_{\text{rent}}.
$$

### 1.2 When is it IC? An exact statement

**Proposition 1 (Groves alignment).** Fix $r_{-m}$. Then

$$
\mathbb E_m[W_m\mid s_m]=\underbrace{\beta_0\,\mathbb E_m[\ell(\mu^*_{-m},y)]}_{h_m}-\beta_0\,\mathbb E_m[\ell(\mu^*(r_m,r_{-m}),y)]-c .
$$

Report $r_m$ affects only the second term if and only if:

- **(C1) Clarke independence.** $\hat\Lambda_{-m}$, $\hat G_{-m}$, $w^*_{-m}$, and the other agents' parameters $\theta_{-m}$ are measurable with respect to $r_{-m}$ and data uninfluenced by $m$.

When C1 holds, $m$'s best response is any report under which $\mu^*$ equals $m$'s posterior $\pi_m=P(y\mid s_m,r_{-m})$. This follows from strict propriety of log loss.

**Corollary (exact IC conditions).**

| Notion | Conditions |
|---|---|
| **DSIC** | C1, plus **(C2-D)**: for *every* $r_{-m}$, including others' misreports, the truthful $r_m$ makes $\mu^*=\pi_m$. This requires the pool to be the exact Bayes combination even off-equilibrium. It holds essentially only in the trivial case of conditionally independent signals with a correctly specified product-of-experts pool. **Generically false.** |
| **Ex-post IC** | C1, plus **(C2-E)**: at truthful $r_{-m}$, the pool with weights $w^*$ is the Bayes combination. In a log-linear/Gaussian model this requires $A$ to equal (an affine image of) the true forecast-error precision, i.e. Bates–Granger. The Gershgorin proxy $\rho \hat G_{\text{off}}$ is a heuristic approximation, so C2-E holds only approximately. |
| **ε-ex-post IC (asymptotic)** | C1, plus consistent estimation of the error covariance. Then $\varepsilon=O\big(\beta_0\,\|\hat A-A^\star\|\cdot\|q\|^2\big)$, which vanishes as audit sample size $n_{\text{audit}}\to\infty$. |
| **Bayes–Nash** | The fallback when reports are simultaneous and $m$ does not observe $r_{-m}$. $\pi_m$ then depends on beliefs about $r_{-m}$. |

**Three unpleasant facts follow.**

1. **IC is weak, and $\hat\beta$ is not identified.**
   - $\mu^*$ depends on $(q_m,\hat\beta_m,\hat k_m)$ only through the pooled logit.
   - Any rescaling $\hat\beta_m\uparrow$, $q_m\downarrow$ that preserves $\mu^*$ is payoff-equivalent.
   - So the planned R17-E1 metric "optimal inflation ratio $\hat\beta/\beta^{\text{true}}$ under VCG" is **not well-defined** if $\beta$ is a reported quantity. It is a flat ridge of indifference.
   - **Fix:** $\hat\beta$ and $\hat k$ must not be reports. The clearinghouse must estimate them from behavior (§1.3). Then the only report is $q_m$ and IC is about $q_m$.

2. **Misspecification makes "non-truthful" optimal, and that is welfare-improving.**
   - If the pool double-counts $m$ and $j$, $m$'s best response is to shade $q_m$ so the pool equals $\pi_m$.
   - This is *outcome implementation*, not truth-telling.
   - The correct Econometrica framing is: "the efficient forecast is implemented in ex-post equilibrium". It is not "agents report truthfully".

3. **C1 fails in any trained system (dynamic hold-up).**
   - If $\theta_{-m}$ were trained in $m$'s presence, then $\ell(\mu^*_{-m})$ depends on $m$'s past behavior.
   - $m$ can then *make itself indispensable*: crowd others out of its niche during training so that removing $m$ later is catastrophic, which inflates $W_m$.
   - This is a dynamic manipulation that static VCG does not cover. You need the Bergemann–Välimäki (2010) dynamic pivot mechanism, where the counterfactual is the welfare of the *re-optimized* system without $m$. That requires counterfactual retraining, which is expensive.
   - A practical proxy: compute the Clarke term with $\theta_{-m}$ from a held-out shadow population trained without $m$, or settle against a frozen snapshot from before $m$ entered.

**Bottom line for the theorem.**
- Retitle it as "ε-ex-post IC under consistent covariance estimation and Clarke independence".
- Prove the ε rate.
- State explicitly that DSIC fails, with a two-agent counterexample: correlated Gaussian signals where the pool weight uses a misspecified $\rho$. $m$'s best response is then $q_m=\kappa(\rho)\,q_m^{\text{true}}$ with $\kappa\neq1$.

### 1.3 Key-direction falsification ($\hat G_{mj}\to 0$)

**When is falsification profitable?**
- Under pure Groves alignment with C1, faking orthogonality to gain weight is *unprofitable in expectation under correct beliefs*. Inflating $w_m$ when $m$ is redundant with $j$ double-counts, raises $\mathbb E\,\ell(\mu^*)$, and lowers $W_m$.
- It becomes profitable as soon as **any payoff component is share-based or convex**:
  - stake returns proportional to $w_m$;
  - mitosis eligibility based on wealth or weight;
  - survival thresholds under limited liability (§3.3). These make payoff convex, which is the standard risk-shifting incentive.
- R17-E3 introduces exactly these. So falsification must be closed off mechanically, not by appeal to alignment.

**How the clearinghouse must compute $G$:**

1. **Never accept reported keys.** Define, on audit inputs $x\in\mathcal D_{\text{audit}}$:
   $$
   G_{mj}=\frac{\mathbb E_{\text{audit}}\big[\tilde q_m(x)^\top F(x)\,\tilde q_j(x)\big]}{\sqrt{\mathbb E[\tilde q_m^\top F\tilde q_m]\,\mathbb E[\tilde q_j^\top F\tilde q_j]}},\qquad \tilde q_m=q_m-\mathbb E[q_m],
   $$
   where $F(x)=\mathrm{diag}(\mu^*)-\mu^*\mu^{*\top}$ is the Fisher information of the pooled softmax.
   - This is the second-order (Bates–Granger) object that governs double-counting.
   - Better still, use the signed residual covariance $\mathbb E[(q_m-\ell'^{-1}(y))(q_j-\ell'^{-1}(y))^\top]$.
   - **Use signed quantities.** The current $\hat k_m=|\hat\delta_m|/\|\hat\delta_m\|$ discards sign. Anti-correlated modules then look like overlapping substitutes, which is the wrong crowding direction.

2. **Commit–reveal with indistinguishable audits.**
   - Agents commit $q_m(x)$ for a batch before learning which points are audit points.
   - Audit points are drawn i.i.d. from the live distribution.
   - Otherwise agents behave orthogonally on audits and correlated live: the "Volkswagen" attack.

3. **Cross-fitting.**
   - Estimate $\hat\Lambda$ and $\hat G$ on fold $A$; settle $W$ on fold $B$; then swap.
   - This removes the second-order incentive to manipulate the weight channel through the same outcomes that determine payment.

4. **Interventional salience over $V\cup Z_{\text{aux}}$ is a weaker substitute.**
   - Define $s_{m,v}=\mathbb E\,|q_m(x)-q_m(x^{\,do(v\leftarrow v')})|$, computed by the clearinghouse via black-box interventions.
   - This is verifiable and resists gradient masking, unlike white-box saliency.
   - But it measures *what a module reads*, not *whether its errors co-move*. Two modules reading disjoint variables that are themselves correlated would get $G\approx 0$ yet double-count.
   - Use salience only as a regularizer on top of residual covariance, never as the primary $G$.

5. **Nonnegative weights.**
   - $A=D+\rho G_{\text{off}}$ with $G_{\text{off}}\ge0$ is **not** an M-matrix, so $A^{-1}\mathbf 1$ can have negative entries.
   - Negative weights invert a module's forecast in the pool, and its LOO incentive then rewards being *anti-informative in a controlled way*.
   - **Fix:** solve $w^*=\arg\min_{w\ge0}\tfrac12 w^\top A w-\mathbf 1^\top w$, a convex QP.
   - LOO re-clearing is then $M$ warm-started QPs.
   - In the unconstrained case, use the Schur downdate $(A_{-m})^{-1}=(A^{-1})_{-m,-m}-\frac{(A^{-1})_{-m,m}(A^{-1})_{m,-m}}{(A^{-1})_{mm}}$, which costs $O(M^2)$ per module.

### 1.4 Structural flaw: Gershgorin vs. clone neutrality

Take $K$ exact clones with $G_{mj}=1$ and diagonal $a=1+\Lambda$. The clone block is $(a-\rho)I+\rho\mathbf 1\mathbf 1^\top$, which gives

$$
w_{\text{clone}}=\frac{1}{a+(K-1)\rho},\qquad \text{total clone weight}=\frac{K}{a+(K-1)\rho}\;\xrightarrow{K\to\infty}\;\frac{a}{\rho}\cdot\frac1a .
$$

**Consequences:**
- Total clone influence equals the single-module influence $1/a$ **iff $\rho=a$**.
- At $\rho=a$ the block is $a\mathbf 1\mathbf 1^\top$, which is singular.
- Strict Gershgorin dominance ($\rho\sum_j G_{mj}<a$) therefore **guarantees** that clones gain pool influence, up to a factor $a/\rho$.
- "Clone annihilation" in R17-E1 can only come from the *payment* side, not the weight side. Each clone's LOO is $\approx 0$ or negative, because removing one reduces double-counting.
- So the claim must be: "clones gain influence but earn non-positive payouts". That is a weaker and more honest claim. Influence without payment still distorts the forecast.

**Fix:** drop Gershgorin. Use ridge Bates–Granger, $w^*=(\hat\Sigma_{\text{err}}+\lambda I)^{-1}\mathbf 1$, projected to $w\ge0$, with $\lambda\to0$ as $n_{\text{audit}}\to\infty$. Clone neutrality then holds asymptotically (pseudo-inverse limit). This is also the version an econometrician will recognize.

### 1.5 Coalitions, complements, false names

**(a) Complements overpay; budget is violated.**
- Two XOR agents hold $x_1$ and $x_2$, with $y=x_1\oplus x_2$, and a composer exists.
- With either removed, the remainder is uninformative, so $\text{LOO}_A=\text{LOO}_B=V$, where $V=\ell(\varnothing)-\ell(\text{full})$.
- Total payout is $2V$ against a surplus of $V$. This is the standard VCG non-budget-balance with complements (Green–Laffont; Ausubel–Milgrom 2006).

**(b) False-name splitting is strictly profitable, and rent does not fix it.**
- One agent holding $(x_1,x_2)$ registers $k$ sybils, each holding one complementary piece.
- Each sybil is pivotal, so the total payout is $kV$.
- With per-identity rent $c$, the gain from splitting is $(k-1)(V-c)>0$ whenever the agent is viable at all ($V>c$).
- Yokoo–Sakurai–Matsubara (2004): no efficient mechanism is false-name-proof in general under complementarity.
- **R17-E1 "Sybil/Clone Resistance" is only true for substitute-sybils.** Say so.

**(c) Substitute collusion is efficiency-benign but revenue-relevant.**
- Two substitutes each earn $\approx0$. They can collude: one withdraws, the other earns $V$, and they split it.
- The forecast is unchanged, so there is no welfare loss. Only the designer's cost changes.
- This is loser collusion in VCG. It matters only for D5 (budget).

**Fix: core-capped / budget-normalized settlement.**

$$
W_m=\beta_0\,\text{LOO}_m\cdot\min\Big(1,\frac{V(N)}{\sum_j \max(\text{LOO}_j,0)}\Big)-c .
$$

- In the substitute regime ($\sum\text{LOO}\le V$) this is unchanged VCG.
- In the complement regime, total payout is capped at $V$. False-name splitting then yields at most $V-kc<V-c$, which is unprofitable.
- **Cost:** individual IC degrades to ε-IC, because $m$'s report moves others' LOOs through the normalizer. Bound $\varepsilon$ by $\beta_0\cdot\partial(\sum_j\text{LOO}_j)/\partial q_m$.
- A real Econometrica result would characterize this tradeoff: no mechanism is simultaneously efficient, false-name-proof, and budget-feasible; the normalized rule is approximately optimal within a stated class.
- That impossibility-plus-approximation pair is a plausible genuine contribution. "VCG-Gershgorin is IC" is not.

---

## 2. Critique of R17-E2 (Zero-Global-Backprop Decentralized Training)

### 2.1 The representability problem precedes the learning problem

With modules seeing $x_A$ and $x_B$ separately and an additive pool $\text{logit}=f(x_A)+g(x_B)$, parity is **unrepresentable**. No training rule fixes this.

R17-E2 must therefore include a **composer** $C$ that reads detached messages $z_A,z_B$ (the $Z_{\text{aux}}$) and has a multiplicative or MLP interaction. The question becomes how $A$ and $B$ learn to send $z$'s that carry $x_A$ and $x_B$ when $I(x_A;y)=0$.

### 2.2 The three-message trichotomy

Under `detach()` at boundaries, the only learning signal $A$ can receive about $C$'s needs is one of three types:

| Message $C\to A$ | What it is | Prior art | Status vs. THEORY-4 bar |
|---|---|---|---|
| $\partial\ell_C/\partial z_A$ (vector) | One-hop backprop | Standard | **Fails D6.** This is just backprop with the chain cut and restitched. |
| Target $z_A^\dagger$ (vector) | Target prop / synthetic gradient | Lee et al. 2015; Jaderberg 2017; Meulemans 2020 | Legitimate if targets come from local inversion. Near-Gauss-Newton; not obviously a "market". |
| Scalar price/payment $R_A$ | RL / ES / zero-order | Chang et al. 2020; Baum's Hayek machine; Holland's bucket brigade | The **only** genuinely market-native option. Pays $O(d)$ variance per scalar. |

"Pinned $\mu^*$ with $-\beta_i$" (STUDY-LIR §3d) produces the *exact local outcome gradient with respect to a module's own output at the cleared market*. That is a type-(i) message at the output boundary.

So M3 must specify which type crosses each internal boundary. If it is type (i) anywhere, the "zero global backprop" claim reduces to "local losses with a learned aggregator", which is Belilovsky/DGL territory.

### 2.3 The exact mechanism by which M3 *can* learn parity

Take the honest type-(iii) version. $A$ is paid by $C$ via an interventional LOO on the auxiliary channel:

$$
R_A=\ell_C\big(z_A\leftarrow \text{resample}\big)-\ell_C(z_A).
$$

**Step 1. The chicken-and-egg.** For parity, $R_A\approx0$ until $C$ has partially learned, because no single bit predicts $y$. If $A$ starts at a constant output, $C$ sees nothing, so $R_A\equiv 0$. This is an **absorbing no-trade equilibrium**.

**Step 2. The only escape: information-preserving upstream features independent of prices.**
- Give $A$ a local objective that preserves its inputs, for example an InfoMax/reconstruction loss $L_A^{\text{loc}}=\|x_A-\text{dec}(z_A)\|^2$, or rely on random features with $d_z\ge|x_A|$.
- Then $z_A$ carries $x_A$ by construction, regardless of prices.

**Step 3. $C$ solves the hard part with local SGD.**
- $C$ learns the parity from $(z_A,z_B)$ by ordinary SGD inside its own boundary.
- It pays the usual plateau of $\sim n^{k-1}$ steps (Barak et al. 2022).

**Step 4. Only then does the market do work.**
- Once $C$ is off the plateau, $R_A$ becomes informative.
- $A$ uses it, for example via REINFORCE on a stochastic bottleneck, to **select and compress** which bits to forward under a bandwidth constraint $d_z<|x_A|$.

**Implications:**
- Under a type-(iii) protocol, the market does not "discover XOR features".
- It does **bandwidth allocation after a downstream learner has found the interaction.**
- This is a real, publishable mechanism only if the bandwidth constraint binds, i.e. if a non-market baseline with the same bottleneck and local losses fails.
- Kill condition for the stronger claim: if M3 with $d_z\ge|x_A|$ and InfoMax upstream equals M3 with prices zeroed, the market contributes nothing to D2.

**SQ ceiling.**
- Outcome-verified payouts are estimated expectations over samples, i.e. statistical queries.
- For $k$-sparse parity over $n$ bits, any SQ learner needs $n^{\Omega(k)}$ queries at tolerance $n^{-\Omega(k)}$ (Kearns 1998; Abbe et al. for SGD).
- No market protocol beats this. Pre-register the claim as "matches M4 within $X$ points at $\le Y\times$ compute". Do not claim superiority on parity.

### 2.4 Failure modes and exact fixes

**F1. Cold-start no-trade equilibrium** (all aux prices 0, $R\equiv0$).

Fix: decaying liquidity subsidy plus information-preserving baseline:

$$
p_{z_A}(t)=\max\big(\widehat{R}_A(t),\;\varepsilon_0(1+t/\tau)^{-1}\big),\qquad L_A=L_A^{\text{loc}}(\text{InfoMax})-\lambda\,p_{z_A}\,\log\pi_A(z_A\mid x_A).
$$

- The subsidy behaves like an LMSR market-maker with liquidity $b$. Its worst-case cost is $b\log K$ per aux variable. Report that cost in D5.

**F2. Symmetry.** Identical initializations of $A$ and $B$ give identical rewards. Use independent initializations plus dithered stochastic bottlenecks: $z=\text{enc}(x)+\sigma\xi$.

**F3. Stake collapse and rich-get-richer lock-in** (STUDY-LIR §3b).
- If the learning signal is $\propto w_m$, a module that falls behind gets $w_m\to0$. Its gradient then vanishes and it can never recover.
- Fix: **stake-invariant learning signal.** Train $\theta_m$ on the residual market as if it had unit weight:
  $$
  \nabla_{\theta_m}\,\ell\big(\sigma(\text{sg}[z^*]-\text{sg}[w_m q_m]+\bar w\,q_m(\theta_m)),\,y\big),\qquad \bar w=\max(w_m,w_{\min}).
  $$
  This is boosting-style residual fitting. Weights decide *influence*; they do not decide *who gets to learn*.

**F4. Two-timescale instability** (θ chases β, β chases θ).
- Use Borkar-style separation with
  $$
  \sum_t\eta_\theta=\sum_t\eta_\beta=\infty,\qquad\sum_t(\eta_\theta^2+\eta_\beta^2)<\infty,\qquad \frac{\eta_\beta(t)}{\eta_\theta(t)}\to0 .
  $$
- So $\beta$ and $A$ move slowly. That is correct anyway, because they are covariance estimates and need many samples.
- At fixed $\theta$, $w^*$ is the unique minimizer of a strongly convex QP, so the fast system sees a quasi-static Lipschitz environment. The ODE method then gives convergence to the set of local equilibria of the θ-game.
- **This is not convergence to a global optimum.** Be precise in the paper.

**F5. Settlement-training leakage.** If $W_m$ is computed on the training batch, modules overfit the settlement. Use the cross-fitting from §1.3.

---

## 3. Critique of R17-E3 (Bankruptcy Pruning and Profit Mitosis)

### 3.1 Why re-cleared LOO avoids the 25.6% sign trap, and where it does not

$|w\cdot\nabla|$ is a first-order Taylor estimate, $\ell(\theta_{\setminus i})-\ell(\theta)\approx -w_i g_i$. It fails for two reasons:

- **Saturation:** $g_i\approx0$, yet removal is catastrophic. This is curvature.
- **Substitute co-adaptation:** removal is compensated by others, but the frozen-others linearization cannot see it.

Re-cleared LOO is an **exact finite difference with the remaining pool weights re-optimized**. It therefore handles saturation (no linearization) and pool-level substitution (re-clearing lets a substitute absorb the weight).

It does **not** handle three things:

1. **Within-module substitution that requires re-training $\theta_{-m}$.** Re-clearing re-weights; it does not re-learn.
2. **Estimation noise.** When true LOO is near 0, the sign error rate is $\approx\Phi(-|\text{LOO}|\sqrt{n}/\hat\sigma)$. Report sign errors stratified by $|\text{LOO}|$.
3. **Clone pairs.** Each has LOO ≈ 0. The re-cleared LOO is *correct* that each is individually dispensable, but **simultaneous** bankruptcy prunes both.
   - Fix: **sequential bankruptcy.** Prune the single lowest-LOCB (lower confidence bound) module, re-clear, recompute all LOOs, and repeat.
   - The surviving clone's LOO jumps to $V$.
   - Alternatively, use group-LOO over clusters with $G_{mj}>\tau$.

The pre-registered comparison must be against strong baselines, not just $|w\cdot\nabla|$: second-order OBD/OBS (curvature-aware), exact LOO-with-finetune (an oracle), and Data/Neuron-Shapley. Beating only first-order Taylor will not survive ICLR review.

### 3.2 Mitosis contradicts clone annihilation

**The problem.** A daughter starts as a clone with $G_{md}=1$, so her LOO is 0 or negative. She is born bankrupt and is annihilated. As specified, the mitosis rate should be ~0.

**Fix: joint account until divergence.**
1. Parent and daughter share one wealth account, and their joint LOO is computed as a group.
2. They are forced apart by:
   - disjoint data shards;
   - an additive decorrelation penalty $\gamma\,\widehat{G}_{md}$ in local training;
   - different input subsets $V\cup Z_{\text{aux}}$.
3. Accounts separate only when $\widehat G_{md}<\tau_{\text{split}}$ with confidence.
4. If divergence does not occur within $T_{\text{div}}$ steps, the daughter is merged back. This is a "failed mitosis" and is not counted as a prune.

### 3.3 Infant mortality: what is possible and what is not

**Impossibility (informal, but defensible).**
- On SQ-hard parity, before the breakthrough, the module's LOO is $n^{-\Omega(k)}$ and statistically indistinguishable from junk with any polynomial audit budget.
- Hence **no outcome-based rule can avoid both infant mortality and junk subsidy.**
- The only choice is how much junk subsidy to buy and how to bound it.
- Barak et al.'s hidden progress (Fourier-gap growth) is detectable only white-box. It is not a verifiable market quantity unless you allow audited internal measurements.

**Recommended design: tranche venture funding with a portfolio cap.**

1. **Endowment sized to the plateau.** Allow a newborn of class $\kappa$ (e.g., bilinear with fan-in $f$) an endowment
   $$
   E_0(\kappa)=c_{\text{rent}}\cdot\hat T_{\text{plateau}}(\kappa),\qquad \hat T_{\text{plateau}}\sim C\cdot f^{\,k-1}/\eta,
   $$
   calibrated once on synthetic parities. This is a pre-registered constant, not a tuned one.

2. **Age-ramped rent.** Charge $c_m(t)=c_{\text{rent}}\cdot\min(1,\,t_m/\tau_{\text{mature}})$, with $\tau_{\text{mature}}\approx\hat T_{\text{plateau}}$.

3. **Progress rebate** (information-gain-adjusted rent).
   - Fit a one-sided trend test on the module's held-out LOO over a sliding window $H$:
     $$
     c^{\text{eff}}_m=c_m\cdot\big(1-\kappa\,\mathbb 1[\hat s_m/\widehat{\text{se}}(\hat s_m)>z_{0.95}]\big).
     $$
   - This rewards statistically significant learning progress, as distinct from level.
   - It helps only after the plateau ends, which is when it is least needed. Be honest about that.

4. **Portfolio cap (the real guarantee).** Total incubator spend satisfies $\sum_{m\in\text{infant}}c_m(t)\le\phi\cdot B_{\text{compute}}$.
   - Junk subsidy is then provably at most a $\phi$ fraction of compute.
   - New infants queue when the cap binds. Use a Gittins/UCB priority on $\widehat{\text{LOO}}+\text{bonus}(\kappa)$.

5. **Bankrupt on LCB, not on realized wealth.** Prune when $\text{LCB}_{0.95}(\overline{\text{LOO}}_m)<c_m$ *and* age exceeds $\tau_{\text{mature}}$.
   - Cumulative realized wealth selects for lucky variance, i.e. survivorship of noisy modules.
   - Under limited liability it also gives strategic agents a convex payoff near zero wealth, which is the Jensen–Meckling risk-shifting incentive. That gambling motive reopens the key-falsification attack from §1.3.

Pre-register $\phi$ (e.g., 10%) and report the realized junk-subsidy fraction. That number, alongside the survival rate of true XOR modules, is the honest E3 result.

---

## 4. Rubric Scores and Pre-Registered Priors

### 4.1 Rubric (0–10)

`R16` is the metaphor-level framing. `Proj-R17` is what R17 would earn **if executed as designed with the fixes above**; it is a projection, not an achievement. `Proj-R17 as-written` is the proposal before the fixes.

| Dim | R16 | Proj-R17 as written | Proj-R17 with fixes | Why |
|---|---|---|---|---|
| D1 IC & Sybil/Clone | 1 | 2 | 5 | As written: $\hat\beta$ unidentified, Gershgorin clone leak, complement false-name attack. Fixed: ε-ex-post IC plus impossibility/approximation. Prior art (Lambert 2008, Chang 2020) caps novelty. |
| D2 Zero-backprop non-separable | 1 | 2 | 4 | SQ ceiling. Market role is bandwidth allocation after downstream discovery. A 7+ needs a bottleneck regime where local-loss baselines fail and M3 succeeds. |
| D3 Architecture discovery | 2 | 3 | 5 | Mitosis/annihilation contradiction as written. A fixed sequential-LOCB scheme plausibly beats $\|w\cdot\nabla\|$ but may tie OBS. |
| D4 Strategic heterogeneous aggregation | 2 | 3 | 5 | Bates–Granger + VCG for heterogeneous black-box models is the most natural and least contrived use case. Likely the strongest real paper. |
| D5 Budget, rent, infant mortality | 1 | 2 | 5 | Complement overpayment is unaddressed as written. A portfolio cap gives a provable bound. |
| D6 Rigor & separation from backprop | 2 | 3 | 5 | Separation holds only if internal messages are type (ii)/(iii), with stated communication cost. Type (i) anywhere drops this to 2. |

### 4.2 Pre-registered `[PRIOR]` forecasts (80% CIs)

**Assumed configuration** (adjust and re-register if it differs): $M=8$ modules, 4 honest and 4 clones of honest module #1; Hidden-Parity with $n=20$, $k=3$; Equiv-SAT chain of 4 hops; compute budget calibrated so that M4 reaches ≥95% accuracy.

**R17-E1: strategic behavior**

| Quantity | Naive-LMSR | VCG-Gershgorin as written | VCG (ridge Bates–Granger, behavioral $G$, normalized) |
|---|---|---|---|
| Optimal $\hat\beta/\beta^{\text{true}}$ | [2.0, 8.0], often hitting the cap | **Not identified** (indifference ridge). If forced by tie-breaking: [0.5, 3.0] | Not a report. Effective-weight distortion [0.85, 1.20] |
| Clone cartel payout share (4/8 clones) | [55%, 85%] | [0%, 10%] payout, but pool influence share [35%, 60%] | [0%, 8%] payout, influence [15%, 30%] |
| Complement-split sybil payout (1 agent → 2 XOR halves) | [40%, 70%] | [60%, 95%], i.e. ≈2V | [≤ fair share]: gain [−c, +5%] |

**R17-E2: accuracy (%) and KL (nats; chance on binary ≈ 0.69)**

| Method | Hidden-Parity acc | Hidden-Parity KL | Equiv-SAT acc | Equiv-SAT KL |
|---|---|---|---|---|
| M1 Local Greedy (detached) | [49, 53] | [0.66, 0.70] | [50, 62] | [0.60, 0.70] |
| M2 Naive LMSR (detached ≡ TBPTT(1)) | [49, 54] | [0.65, 0.70] | [50, 65] | [0.58, 0.70] |
| M3 VCG-Gershgorin-LIR, type-(iii) messages only | [50, 95], **bimodal**; P(≥90%) ≈ 0.35 | [0.05, 0.69] | [55, 92]; P(≥90%) ≈ 0.30 | [0.08, 0.65] |
| M3 with InfoMax upstream, no bottleneck | [85, 99] | [0.02, 0.30] | [80, 98] | [0.03, 0.35] |
| M4 Global Backprop | [95, 100] | [0.00, 0.10] | [95, 100] | [0.00, 0.10] |

Additional prior for M3: P(M3 needs ≥3× M4 compute to reach 90%) ≈ 0.6.

**Interpretation rule (pre-registered):**
- If M3 with InfoMax matches M3 with prices zeroed (within 2 points), the market's D2 contribution is null.
- The D2 claim then reduces to "a local composer learns parity", which is not novel.

**R17-E3: pruning and architecture**

| Quantity | VCG-LOO sequential + LCB + incubator | VCG-LOO simultaneous, flat rent (as written) | GradMag $\|w\cdot\nabla\|$ | OBS (2nd-order) |
|---|---|---|---|---|
| Junk pruning recall | [80, 97] | [80, 97] | [55, 85] | [70, 92] |
| Clone pruning (exactly one of each pair kept) | [85, 99] | [5, 40]: both pruned | [20, 60] | [40, 75] |
| XOR bottleneck survival | [60, 95] | [5, 40] | [30, 75] | [40, 80] |
| Mitosis → stable divergent daughter | [10, 50] | [0, 5] | n/a | n/a |
| Realized junk-subsidy fraction (cap $\phi$=10%) | [3, 10] | n/a | n/a | n/a |
| Final acc. gap vs GradMag on bottleneck tasks (pts) | [−2, +8] | [−15, +2] | 0 | [−3, +4] vs GradMag |
| Sign errors on top-causal units (cf. 25.6%) | [2, 10]%, concentrated at small \|LOO\| | [2, 10]% | ≈25% (reproduction) | [8, 18]% |

---

## 5. What Would Actually Be a Smoking Gun

**Econometrica (theory-first, experiments secondary):**
- A characterization theorem for forecast-pool mechanisms:
  - efficient aggregation (Bates–Granger) + ex-post IC + false-name-proofness + budget feasibility is impossible under complementarity;
  - the budget-normalized Clarke rule is the unique (or approximately optimal) rule within a natural class, with an explicit ε bound;
  - plus a dynamic-hold-up result showing static LOO is manipulable when $\theta_{-m}$ co-trains, together with the dynamic-pivot fix.
- Must beat Lambert et al. 2008 on aggregation and Chang et al. 2020 on scope.
- My estimated odds that the current program yields this: low (~10–15%) without a dedicated theorist pass.

**ICLR (empirics-first):**
- A bandwidth-constrained decentralized parity/Equiv-SAT benchmark where all three hold:
  1. local-loss, synthetic-gradient, and target-prop baselines with the same bottleneck fail;
  2. scalar-price M3 succeeds;
  3. communication bits per step are reported and M3 is competitive.
- Or the D4 heterogeneous black-box model-aggregation story with real LLM or vision ensembles, demonstrating clone resistance and strategic robustness. This is the more likely win (~25–35%), because it is where incentives are genuine rather than imposed on cooperative neurons.

**Kill criteria worth writing down now:**
1. If every internal message in M3 is effectively a cotangent, abandon the D6 claim.
2. If the price-zeroed ablation matches M3, abandon the D2 claim.
3. If sequential LOO pruning does not beat OBS by ≥3 points on bottleneck survival, demote E3 to an appendix.

The market perspective survives this audit as serious, but in a narrower form than R17 states. It works as **mechanism design for aggregating strategic, heterogeneous, possibly cloned predictors under bounded subsidy**. It does not work as a replacement for gradients in learning non-separable features. On the evidence so far, that second claim is the one most likely to be refuted.
