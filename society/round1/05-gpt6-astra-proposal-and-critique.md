# Verified Innovation Search: A Proposal and Red-Team Critique of Fractal Prospectus Plans

**Society of Thinkers · Round 1 · Independent theoretical and algorithmic proposal**  
**Contributor:** GPT-6-Astra  
**Repository:** `futarchy-fi/inconsistency-search`  
**Date:** 2026-10-18  
**Status:** Design proposal. No commands executed, files inspected, or experiments run. Historical results below are attributed to the supplied charter and research-direction note; they were not independently verified.

## Abstract

The charter’s strongest idea is a learned search machine that chooses both where to compute and what unresolved question to pursue, while sharing parameters across environments and tree depths. Its central unresolved issue is the distinction between **additional computation, additional information, and additional confidence**. These are not interchangeable.

I propose **Verified Innovation Search**: a universal KV-cache tree whose shared prospectus head targets residual questions; whose evidence ledger separates exact ancestry from uncertain semantic overlap; whose clearing operator solves a well-defined conditional estimation problem; and whose belief updates pass a separately justified acceptance rule. Hindsight plans train representations and controllability, while prospectus selection receives signed, independently scored credit. Present-time memory preserves information-set beliefs and their provenance rather than recycling previous consensus as fresh evidence.

Three conclusions constrain the design:

1. Weight sharing across depths does not establish fractal behavior. A unit direction requires a contextual metric, an achievement magnitude, and learned termination to distinguish macro-plans from micro-plans.
2. Exact prefix deduplication and convergent Gram clearing do not guarantee decreasing true-posterior KL. An unconditional guarantee is impossible for unrestricted learned updates. A valid guarantee requires exact Bayesian refinement or a sound risk certificate, potentially with abstention.
3. Utility-independent beliefs require separation at inference as well as during training. A frozen belief model can still produce utility-dependent consensus through search selection, memory, aggregation, or a utility factor applied to a joint distribution.

The proposed experiments prioritize these distinctions before a larger league can obscure them.

---

## 1. Objects, targets, and claims that must remain separate

Let \(I_t\) denote an agent’s legally available information at game turn \(t\), and let

\[
P^\star_t(X)=P_{\mathrm{env}}(X\mid I_t)
\]

be the environment’s reference posterior over a declared query \(X\). For counterfactual outcomes, the target must instead specify an intervention, such as

\[
P^\star_t(Y\mid \operatorname{do}(A=a)).
\]

At fixed turn \(t\), search step \(b\) produces a tree \(\mathcal C_b\) and an output belief \(\mu_b\). The following targets are different:

- **Truth:** \(P^\star_t\), under the environment’s information structure.
- **Model coherence:** the optimizer of a fixed world-model inconsistency objective.
- **Numerical convergence:** the solution of a particular clearing system.
- **Decision quality:** expected utility, exploitability, or tournament payoff.

Improving one does not generally improve the others. A model can become more internally coherent and less accurate. A linear solver can converge to the wrong estimator. A strategically successful policy can communicate distorted beliefs.

The charter should therefore state its primary belief claim as

\[
R_b=\mathbb E\!\left[
\operatorname{KL}(P^\star_t\Vert\mu_b)
\right],
\]

with the expectation’s task distribution, search randomness, and information conditions declared. Policy independence and payoff improvement should be separate claims.

### 1.1 Utility must not redefine the epistemic target

A utility factor multiplying a joint belief generally changes uncontrolled marginals:

\[
\widetilde\mu_U(x,a)
\propto
\mu(x,a)e^{\lambda U(x,a)}.
\]

Even if no belief parameter receives a utility gradient, marginalizing \(\widetilde\mu_U\) can change beliefs about \(x\). Gradient masking does not prevent this inference-time effect.

Use two explicit objects:

\[
\mu_{\mathrm{belief}}(X\mid I_t)
=
\mathcal A_{\mathrm{epistemic}}(I_t,\mathcal C_b),
\]

\[
\pi_U(a\mid I_t)
\propto
\pi_{\mathrm{ref}}(a\mid I_t)
\exp\!\left[
\lambda\,
\mathbb E_{\mu_{\mathrm{belief}}}
\left[U(Y,a)\mid I_t,\operatorname{do}(A=a)\right]
\right].
\]

Utility can allocate computation, but finite-budget utility-driven selection must be audited for its effect on the resulting belief. Full-support search and a fixed asymptotic objective do not establish finite-budget independence.

---

## 2. Active prospectus plans: retain sharing, repair the objective

### 2.1 What the shared head does—and does not—provide

A shared head

\[
q_{\mathrm{plan}}(p\mid C_j,K_{\mathrm{done}}),
\qquad
p\in\mathbb S^{D_{\mathrm{amp}}-1},
\]

is a useful inductive bias. It allows the same operation—select an unresolved direction—to recur at arbitrary checkpoints.

It does not by itself establish:

- consistent meaning of a direction at different checkpoints;
- causal control of subsequent computation;
- useful coverage rather than repeated fashionable plans;
- scale equivariance;
- correct credit for an apparently successful branch.

A strong test of active planning intervenes on \(p\) while holding the checkpoint and random-number stream fixed. If the plan is meaningful, that intervention changes the questions explored and the achieved information, not merely an auxiliary embedding.

### 2.2 Failure A: hindsight survivorship and hallucination reinforcement

Suppose the current correct binary belief is

\[
P^\star=\mu_b=(1/2,1/2).
\]

A fabricated branch yields \(\nu=(0.99,0.01)\). Its entropy is much lower, but

\[
\operatorname{KL}(P^\star\Vert\nu)
=
\frac12\log\frac{0.5}{0.99}
+
\frac12\log\frac{0.5}{0.01}
\approx 1.614.
\]

Entropy reduction is therefore an invalid reward for epistemic progress.

The same problem affects consensus loss when the system generates both the claim and its confirming target. Selecting only successful leaves compounds it: the training distribution conditions on survival under the system’s own validator.

#### Repair: signed proper-score innovation

For a branch or segment \(e\), freeze its candidate prediction \(\nu_e\) before revealing an independent validation outcome \(X\). Define

\[
g_e(X)
=
\log \nu_e(X)-\log\mu_b(X).
\]

Then

\[
\mathbb E_{X\sim P^\star}[g_e(X)]
=
\operatorname{KL}(P^\star\Vert\mu_b)
-
\operatorname{KL}(P^\star\Vert\nu_e).
\]

This is the required marginal consensus improvement when consensus loss is a proper held-out log score.

For conditional usefulness in a completed tree, use removal of a **unique innovation**, not removal of one copied rollout:

\[
g_e^{\mathrm{LOO}}(X)
=
\log\mu_{\mathcal E}(X)
-
\log\mu_{\mathcal E\setminus\{e\}}(X).
\]

All descendants or cached reports that reuse \(e\) must lose that contribution in the counterfactual. Otherwise LOO attribution mistakes duplication for independent corroboration.

Train a gain predictor on out-of-fold outcomes:

\[
\mathcal L_{\mathrm{gain}}
=
\mathbb E\!\left[
\left(
\widehat g_\psi(F_e)
-
\operatorname{sg}[g_e(X)]
\right)^2
\right],
\]

where \(F_e\) contains only features available before the validation label. Use a conservative estimate

\[
\underline g_e
=
\widehat g_\psi(F_e)-c_\delta\widehat\sigma_\psi(F_e)
\]

for scheduling or acceptance. Its validity depends on the calibration protocol; a neural uncertainty estimate is not automatically a certificate.

Keep negative gains. Training only on \([g_e]_+\) rewards lucky winners while discarding the losses needed to estimate their expectation.

#### Repair: separate hindsight representation learning from prospectus optimization

A hindsight direction

\[
p^{\mathrm{HER}}_{j\to j'}
=
\frac{
\operatorname{sg}[\Delta h_{j\to j'}]
}{
\|\operatorname{sg}[\Delta h_{j\to j'}]\|_2
}
\]

describes what happened. It does not establish that sampling this direction prospectively would cause the same result.

Use hindsight for:

- learning a stable achievement representation;
- learning whether the step policy can reach a specified achievement;
- training conditional step behavior on complete trajectories, including failed ones.

Train prospectus selection primarily on forward-sampled plans:

\[
\nabla_\phi J
=
\mathbb E_{\substack{p\sim q_\phi\\\tau\sim P_\theta(\cdot\mid p)}}
\left[
\left(
g(\tau)-\kappa\,\mathrm{cost}(\tau)-b(C_j)
\right)
\nabla_\phi\log q_\phi(p\mid C_j,K_{\mathrm{done}})
\right].
\]

Here \(g\) is signed proper-score improvement, not self-reported certainty.

If hindsight trajectories also train the forward objective, the correction is a **joint-distribution** correction. For behavior trajectory density \(m(\tau)\) and hindsight relabeling density \(h(p\mid\tau)\),

\[
\omega(\tau,p)
=
\frac{
q_\phi(p\mid C_j,K_{\mathrm{done}})
P_\theta(\tau\mid C_j,p)
}{
m(\tau)h(p\mid\tau)
}.
\]

This requires support and inclusion of all plan-dependent selection and termination decisions. A deterministic hindsight label is singular relative to a continuous spherical density; smoothing it gives support but may still produce intolerable variance. Clipping introduces bias. Forward sampling is the cleaner default.

A single lucky realized outcome remains noisy evidence. Proper scoring fixes the expectation; replication and cross-fitting address estimation error.

### 2.3 Failure B: collapse onto already-covered keys

The naïve hindsight objective

\[
-\sum_e[g_e]_+\log q_\phi(k_e^{\mathrm{post}}\mid C_j,K_{\mathrm{done}})
\]

can concentrate the head on familiar directions. Easy subproblems produce more accepted leaves, which produce more training targets, which produce more visits.

Coverage should represent **remaining uncertainty**, not merely visitation.

Let \(K_b\) contain unique, accepted innovation keys and let \(\Omega_b\) be their calibrated information precision. Define

\[
\Sigma_b
=
\left(\lambda I+K_b^\top\Omega_bK_b\right)^{-1}.
\]

For a candidate direction \(p\), use the linear-Gaussian information-gain proxy

\[
n_b(p)
=
\frac12
\log\left(
1+\widehat\beta(p)\,p^\top\Sigma_b p
\right).
\]

Repeated reliable observations diminish \(p^\top\Sigma_b p\). Repeated noisy observations can remain valuable. This avoids the mistake of forbidding every previously visited direction.

Define a target plan density

\[
q^\star_b(p)
=
(1-\varepsilon_b)
\frac{
q_{\mathrm{ref}}(p)
\exp\!\left[
\big(
\underline g_b(p)
+\alpha n_b(p)
-\kappa\widehat c_b(p)
\big)/T
\right]
}{
Z_b
}
+
\varepsilon_b q_{\mathrm{ref}}(p).
\]

Train \(q_\phi\) toward this density, with the target detached during that update. The information-gain term is a scheduling proxy; it is not evidence of truth.

For a finite discrete set of legal search choices, persistent exploration can use

\[
\varepsilon_b>0,
\qquad
\sum_b\varepsilon_b=\infty,
\]

together with a reference policy assigning positive probability to each relevant choice. On the sphere, full density support covers neighborhoods, not exact directions. Neither condition guarantees useful coverage within \(128\) steps.

Record both attempted and accepted coverage. Otherwise the model can repeatedly revisit a failed direction because rejected evidence never enters \(K_{\mathrm{done}}\).

### 2.4 Failure C: macro/micro scale ambiguity on the sphere

Normalization discards magnitude:

\[
\frac{\Delta h}{\|\Delta h\|}
=
\frac{100\Delta h}{\|100\Delta h\|}.
\]

A one-bit tactical clarification and a broad strategic resolution can therefore receive the same plan. Raw latent coordinates also have a gauge problem: representation changes can alter angles without altering predictions.

Retain the unit direction, but attach contextual scale and termination:

\[
(p,\rho)\sim q_\phi(\,\cdot\mid C_j,K_{\mathrm{done}}),
\qquad
p\in\mathbb S^{d-1},
\quad \rho\ge 0.
\]

Let \(G_j\succ0\) be a regularized metric derived from predictive sensitivity or calibrated latent uncertainty. Define

\[
r_{j\to j'}=G_j^{1/2}\Delta h_{j\to j'},
\]

\[
p^{\mathrm{HER}}=\frac{r_{j\to j'}}{\|r_{j\to j'}\|_2},
\qquad
\rho^{\mathrm{HER}}=\|r_{j\to j'}\|_2.
\]

Zero achievements receive a separate null-plan label rather than an arbitrary normalized direction.

The step policy receives the remaining target in the starting checkpoint’s coordinates:

\[
u_t
=
\rho p-G_j^{1/2}(h_t-h_j),
\]

\[
\pi_{\mathrm{step}}(v,z\mid C_t,p,\rho,u_t).
\]

A shared learned termination head

\[
P(\mathrm{terminate}\mid C_t,p,\rho,u_t)
=
\sigma(f_{\mathrm{term}}(C_t,p,\rho,u_t))
\]

decides when to replan. No fixed hierarchy levels or game-specific depth thresholds are required.

The metric and latent basis must be frozen or versioned within a search episode. Across training updates, align representations or invalidate incompatible cached plans. The added \(\rho\) is a target achievement magnitude, not a hardcoded rollout horizon.

**Defensible claim:** a shared, context-normalized planning rule operates at arbitrary checkpoints.  
**Claim requiring evidence:** its behavior is self-similar across task scales.

---

## 3. A unified Tree-Prefix + Salience Gram Clearing Operator

### 3.1 Exact ancestry is bookkeeping, not a learned similarity

Index tree edges by \(e=(\operatorname{parent}(j)\to j)\). Define the path-incidence matrix

\[
A_{re}
=
\mathbf 1\{e\text{ lies on the path }C_0\to C_r\}.
\]

For cumulative node reports \(V_r\) and edge innovations \(D_e\),

\[
V=AD.
\]

With every non-root node present and ordered after its parent, \(A\) is invertible. Its inverse is the tree-difference operator:

\[
D_j=V_j-V_{\operatorname{parent}(j)},
\qquad V_0=0.
\]

Consequently,

\[
V_r
=
V_{\mathrm{LCA}(r,b)}
+
\sum_{e\in\mathrm{path}(\mathrm{LCA}(r,b),r)}D_e.
\]

The shared prefix enters once. It must not be relearned from cosine similarity between terminal keys.

There is an important limitation: differences of arbitrary neural summary tokens need not be meaningful information increments. The preferred implementation emits edge-level claims explicitly. Differences of canonical, centered log probabilities can telescope algebraically, but that alone does not make them independent likelihood evidence.

### 3.2 An edge can be a hypothesis rather than an observation

Appending a sampled assignment \(Z=z\) to a cache creates a conditional world. It does not establish that \(Z=z\) is true.

For mutually exclusive branches,

\[
\mu(Y\mid C_j)
=
\sum_z
\mu(Z=z\mid C_j)\,
\mu(Y\mid C_j,Z=z).
\]

Do not add sibling logits as though their hypotheses were jointly observed.

Unexpanded probability mass must remain represented. For expanded children \(E_j\),

\[
\mu(Y\mid C_j)
=
\sum_{z\in E_j}m_z\,\mu(Y\mid C_j,z)
+
\left(1-\sum_{z\in E_j}m_z\right)
\mu_{\mathrm{res}}(Y\mid C_j).
\]

Renormalizing only surviving leaves creates survivorship bias unless a sound refutation establishes that the omitted mass is impossible.

The clearing operator below acts on compatible claims about a declared conditional target. Branch marginalization is a separate operation.

### 3.3 Deduplicate exact sources before solving a statistical problem

Maintain a ledger of unique source identities. A source identity should include, where applicable:

\[
(\text{model version},\text{query},\text{conditioning assignment},
\text{information set},\text{observation provenance}).
\]

Two cache edges can evaluate the same deterministic conditional even when they are not structural ancestors. Those are exact duplicates too.

Conversely, two genuinely independent noisy measurements of the same latent quantity are not duplicates. They share a target but carry distinct observations.

Thus:

- copied cache evidence receives one ledger entry;
- repeated deterministic queries receive one contribution;
- independent noisy observations retain separate entries;
- uncertain overlap remains a covariance-estimation problem.

No single cosine threshold reliably separates all four cases.

### 3.4 The linear conditional model

After exact deduplication, let:

- \(n\): number of unique compatible innovations;
- \(d=D_{\mathrm{amp}}\): latent feature dimension;
- \(m\): output dimension;
- \(K\in\mathbb R^{n\times d}\): normalized salience-feature rows;
- \(D\in\mathbb R^{n\times m}\): conditional innovation claims;
- \(R\succ0\): residual error covariance;
- \(W\in\mathbb R^{d\times m}\): latent correction coefficients.

Use

\[
D=KW+E,
\qquad
E_{\cdot a}\sim\mathcal N(0,R),
\qquad
W_{\cdot a}\sim\mathcal N(0,\lambda^{-1}I).
\]

The posterior mean/MAP solves

\[
\boxed{
\left(K^\top R^{-1}K+\lambda I\right)\widehat W
=
K^\top R^{-1}D.
}
\]

When residual errors are independent,

\[
R^{-1}=B=\operatorname{diag}(\beta_1,\ldots,\beta_n),
\]

recovering the requested normal equations. Here the unknown is a latent coefficient matrix, not a vector of rollout weights. Conflating those objects leads to dimensionally ambiguous formulas.

A target decoder \(\Phi\in\mathbb R^{m\times d}\) produces a candidate correction; for output \(a\),

\[
\delta\ell(a)=\Phi_{a\cdot}\widehat W_{\cdot a},
\qquad
\nu=\operatorname{softmax}(\ell_{\mathrm{anchor}}+\delta\ell).
\]

The precise decoder can differ, but must be specified and validated. Solving the regression does not identify a unique probabilistic decoder by itself.

### 3.5 Where exact prefixes and semantic overlap meet

For a scalar output, consider edge claims

\[
D=K w+\eta,
\quad
w\sim\mathcal N(0,\Sigma_w),
\quad
\eta\sim\mathcal N(0,\operatorname{diag}(\tau_e^2)).
\]

Cumulative reports \(V=AD\) then have covariance

\[
\boxed{
\operatorname{Cov}(V)
=
A K\Sigma_wK^\top A^\top
+
A\operatorname{diag}(\tau_e^2)A^\top.
}
\]

The second term satisfies

\[
\left[
A\operatorname{diag}(\tau_e^2)A^\top
\right]_{rb}
=
\sum_{e\preceq\mathrm{LCA}(r,b)}\tau_e^2,
\]

which gives the exact structural-prefix contribution. The first term carries semantic overlap, including overlap after the fork.

Equivalently, solve in cumulative coordinates:

\[
\min_W
\frac12
\left\|
V-AKW
\right\|_{\Sigma_V^{-1}}^2
+
\frac{\lambda}{2}\|W\|_F^2,
\]

where \(\Sigma_V=ARA^\top\) is the **conditional residual covariance**, excluding the latent prior contribution. With complete node reports and invertible \(A\), this reduces to the edge-coordinate normal equations.

If independently noisy node readouts are differenced, the resulting edge errors are correlated. One must propagate that covariance; replacing it with diagonal \(B\) silently reintroduces incorrect independence assumptions.

### 3.6 Salience Gram matrices are hypotheses about sources

The footprint

\[
k_e=\frac{|\Delta\ell_e|}{\|\Delta\ell_e\|_2}
\]

can identify overlapping effects, but overlapping effects do not prove overlapping sources:

- independent observations may move the same belief coordinate;
- contradictory claims can have identical absolute footprints;
- one observation can affect disjoint downstream outputs;
- a near-zero shift has an unstable direction.

Use salience to propose covariance structure, while retaining signed claims, source provenance, and uncertainty. Validate the induced covariance against controlled shared-source experiments.

The Gram kernel must be positive semidefinite. \(KK^\top\) has that property; an arbitrary entrywise clipping or power transform need not preserve it.

### 3.7 Iterative implementation inside the named aggregators

Write

\[
H=K^\top R^{-1}K+\lambda I,
\qquad
F=K^\top R^{-1}D.
\]

A shared preconditioned Richardson block performs

\[
W^{(\ell+1)}
=
W^{(\ell)}
+
\eta P^{-1}\left(F-HW^{(\ell)}\right),
\]

with \(P\succ0\). It converges if

\[
0<\eta<
\frac{2}{
\lambda_{\max}(P^{-1/2}HP^{-1/2})
}.
\]

A Jacobi-style choice uses \(P=\operatorname{diag}(H)\), with damping satisfying this bound. Positive definiteness alone does not guarantee convergence of undamped Jacobi iteration.

For `looped_attn_T3`, constrain the shared block to approximate this residual operation and expose its solver residual. For `jacobi_lin_T4_prec`, use an explicit damped iteration. Three or four steps give an approximation, not an exact solve.

A numerical certificate is

\[
\|W^{(\ell)}-\widehat W\|_F
\le
\frac{\|F-HW^{(\ell)}\|_F}{\lambda_{\min}(H)}
\le
\frac{\|F-HW^{(\ell)}\|_F}{\lambda}.
\]

This certifies coefficient error. Converting it to belief-risk error requires decoder sensitivity and probability-support bounds.

For each new unique row with independent residual noise, update sufficient statistics once:

\[
H_{b+1}=H_b+\beta_e k_ek_e^\top,
\qquad
F_{b+1}=F_b+\beta_e k_eD_e.
\]

An exact duplicate leaves both unchanged. Full correlated-noise updates require the corresponding block-covariance calculation.

---

## 4. Anytime monotonicity: the valid theorem and the impossibility boundary

### 4.1 Why the charter’s ingredients are insufficient

Suppose the current belief happens to equal truth:

\[
\mu_b=P^\star.
\]

Any accepted update \(\mu_{b+1}\ne\mu_b\) gives

\[
\operatorname{KL}(P^\star\Vert\mu_{b+1})>0
=
\operatorname{KL}(P^\star\Vert\mu_b).
\]

An unrestricted learned system cannot know whether it is in this case from internal confidence alone.

Therefore root anchoring, residual gates, deduplication, and self-consistency cannot provide a distribution-free nontrivial guarantee of true-KL improvement. They can reduce failure rates, but they do not resolve the identifiability problem.

Nested candidate classes only guarantee nonincreasing **optimized objective value** when that objective is solved correctly. They do not guarantee nonincreasing true risk.

### 4.2 Exact guarantee under Bayesian refinement

Let \(\mathcal F_b\) be a nested filtration of computationally revealed information, including search selections and their outcomes. Treat \(P^\star\) as a random target distribution across tasks. Suppose

\[
\mu_b=\mathbb E[P^\star\mid\mathcal F_b].
\]

Then

\[
\boxed{
\mathbb E[
\operatorname{KL}(P^\star\Vert\mu_b)
]
-
\mathbb E[
\operatorname{KL}(P^\star\Vert\mu_{b+1})
]
=
\mathbb E[
\operatorname{KL}(\mu_{b+1}\Vert\mu_b)
]
\ge0.
}
\]

**Proof.** The difference is

\[
\mathbb E\!\left[
\sum_x P^\star(x)
\log\frac{\mu_{b+1}(x)}{\mu_b(x)}
\right].
\]

Conditioning on \(\mathcal F_{b+1}\) replaces \(P^\star\) by \(\mu_{b+1}\), yielding the stated KL.

This is an exact anytime theorem, including adaptive computation, provided the conditional expectations are correct. The Gaussian clearing model supplies such a posterior mean for its latent regression model, but a softmax of that mean is not automatically the posterior predictive distribution needed here.

Also, if \(\mathcal F_0\) already contains all information determining \(P^\star\), exact Bayesian conditioning leaves nothing to improve. Computational refinement is meaningful because the bounded agent initially lacks the computed answer, not because its simulated branches constitute new external observations.

### 4.3 A second exact guarantee: robust acceptance with abstention

Let the clearing system propose \(\nu_b\). Form

\[
\mu_{b+1,\alpha}
=
(1-\alpha)\mu_b+\alpha\nu_b,
\qquad \alpha\in[0,1].
\]

Let \(\mathcal P_b\) be an uncertainty set guaranteed to contain the true posterior. Define

\[
U_b(\alpha)
=
\sup_{P\in\mathcal P_b}
\mathbb E_{X\sim P}
\left[
\log\frac{\mu_b(X)}{\mu_{b+1,\alpha}(X)}
\right].
\]

Accept only an \(\alpha\) satisfying

\[
\boxed{U_b(\alpha)\le0.}
\]

Then, pointwise,

\[
\operatorname{KL}(P^\star\Vert\mu_{b+1,\alpha})
-
\operatorname{KL}(P^\star\Vert\mu_b)
\le0.
\]

The safe fallback is \(\alpha=0\).

This gives a guaranteed Tree-Prefix + Salience clearing **system**, with clearing as its proposal mechanism and robust acceptance as its safety mechanism. The guarantee belongs to the acceptance condition, not to Gram algebra.

The cost is unavoidable: if \(\mathcal P_b\) is the full simplex, only the unchanged distribution can generally pass. Nontrivial safe progress needs independently justified information about the target.

### 4.4 What practical validation can establish

Held-out roots can estimate

\[
\Delta R_b
=
\mathbb E\left[
\log\frac{\mu_b(X)}{\mu_{b+1}(X)}
\right].
\]

A one-sided upper confidence bound below zero supports population non-degradation for a fixed, independently evaluated update rule. It does not establish a per-root guarantee.

Across \(b=0,\ldots,128+\), use simultaneous bounds or a valid confidence sequence. Adaptive reuse of validation labels for tuning invalidates naïve intervals.

Report four separate quantities:

1. Mean proper-score change and simultaneous uncertainty.
2. Fraction and magnitude of harmful root-level updates.
3. Acceptance rate and amount of useful progress.
4. Whether the result is exact, high-probability, empirical, or by construction.

An always-reject gate is monotone and useless. The supplied unresolved-board result illustrates why safety and progress must be reported separately.

---

## 5. Present-time memory and counterfactual information-set search

### 5.1 What R8-E3 actually says

According to the supplied research note:

- the current-turn anchor had \(\mathrm{KL}(0)=0.0488\);
- four-ply window rollouts had KL \(0.080\);
- large aggregators trained on 2,000 hard-label positions reached test KL \(0.6348/0.5849\);
- registered scalar mixing reached \(0.0463\);
- depth-one claims had lower MSE than depth-four claims, \(0.0397\) versus \(0.0451\).

These observations support a narrow diagnosis: the four-ply reports were weak incremental inputs relative to a strong present-time anchor, and the aggregation fit overgeneralized poorly. They do not show that longer planning is intrinsically inferior.

A report useful relative to a \(t-4\) anchor may be redundant relative to a current-turn anchor. The relevant target is

\[
\mathbb E\left[
\log\nu(X)-\log\mu_{\mathrm{current}}(X)
\mid I_t,\text{query}
\right].
\]

Historical standalone accuracy is insufficient.

### 5.2 Memory must preserve a belief state, not repeat a conclusion

The ideal memory is a sufficient statistic for the legal history:

\[
b_t(z)
=
P(Z_t=z\mid I_t),
\qquad
M_t=\operatorname{Encode}(b_t,\text{provenance},\text{uncertainty}).
\]

A generic filter is

\[
b_t(z)
\propto
P(o_t\mid z,a_{t-1})
\sum_{z'}
P(z\mid z',a_{t-1})b_{t-1}(z').
\]

In strategic hidden-information games, opponent actions can reveal private information. Their likelihood must be included under an explicit opponent model. The agent’s own chosen action is an intervention, not fresh evidence about its own hidden-state hypothesis.

Train compressed memory against a reference belief or future predictive queries:

\[
\mathcal L_M
=
\mathbb E_t\left[
\operatorname{KL}
\left(
b_t^\star\Vert q_\theta(\cdot\mid M_t,o_t)
\right)
\right].
\]

A single hidden vector may not retain multimodal correlations. Token sets, weighted particles, or structured latent factors are reasonable alternatives under the same interface.

Memory and the root are not independent experts:

\[
\mu_0=q_\theta(\cdot\mid o_t,M_t).
\]

Once memory has conditioned the root, injecting the same prior consensus again as independent precision double-counts history.

Memory also needs source/version metadata. Otherwise an erroneous search conclusion can be promoted to “remembered fact” and reinforce itself indefinitely.

### 5.3 Counterfactual queries target residual value

Present-time search should ask questions whose answers can improve current decisions or unresolved beliefs:

\[
Q_t(a,a')
=
\mathbb E[
U(Y)\mid I_t,\operatorname{do}(A=a)
]
-
\mathbb E[
U(Y)\mid I_t,\operatorname{do}(A=a')
],
\]

or

\[
Q_t(A,B)=P(A=B\mid I_t).
\]

For Micro-Stratego, useful queries include hidden-piece correlations, opponent-response distributions, and action outcomes marginalized over the current information set.

Hidden worlds may be sampled internally, but the acting policy must remain information-set measurable:

\[
\pi(a\mid z,I_t)=\pi(a\mid z',I_t)
\]

whenever \(z,z'\) are indistinguishable to that player at the decision point. Otherwise determinization produces strategy fusion: the search chooses actions using information the player does not possess.

Memory plus these queries addresses the mechanism behind redundant historical rollouts. It is a proposed remedy, not an established explanation of all R8-E3 failures. If the present anchor already contains the relevant information, correct behavior may be to preserve it and spend computation on action evaluation.

---

## 6. Latent lemmas and relational query tokens

A large latent dimension permits expressive intermediate computation; it does not demonstrate a learned lemma system.

### 6.1 Do not force all embeddings to mean the same thing

Separate, with learned maps where useful:

- **prospectus direction:** desired achievement;
- **salience key:** predicted source/effect overlap;
- **lemma representation:** reusable semantic content.

Equating these spaces by normalization alone invites reward hacking: the model can rotate achievements toward popular plan directions without discovering better arguments.

### 6.2 Query tokens need coherent semantics

For equality queries, basic consistency conditions include

\[
P(A=B)=P(B=A),
\]

\[
P(A=B)
\ge
P(A=C)+P(C=B)-1.
\]

These are useful diagnostics or auxiliary constraints, but they do not establish a globally consistent joint distribution.

Evaluate whether relational-query training improves held-out conditional predictions and whether those conditionals improve held-out relational queries. Use splits by constraint graph or compositional structure, not merely by example.

A lemma should transfer to new queries sharing its underlying relation. If its usefulness disappears when the original root or answer label changes, it may be an answer code rather than a reusable deduction.

---

## 7. Multi-game evaluation, nontransitive leagues, and economic calibration

### 7.1 Elo is a summary, not a universal competence coordinate

Standard Elo assumes approximately

\[
P(i\text{ beats }j)
=
\sigma(r_i-r_j).
\]

It cannot represent a rock–paper–scissors cycle. Hidden-information exploitation and multiplayer externalities make such violations likely.

Retain Elo for accessibility, but add an antisymmetric interaction model:

\[
\operatorname{logit}P(i\succ j)
=
r_i-r_j+u_i^\top J u_j,
\qquad J=-J^\top.
\]

This mElo-style term captures cyclic pairwise structure. Validate its predictive advantage on held-out matches.

For two-player zero-sum games, estimate a payoff matrix \(A\) and equilibrium mixtures:

\[
x^\star\in
\arg\max_{x\in\Delta}
\min_{y\in\Delta}x^\top Ay.
\]

Report performance against a frozen reference mixture, exploitability where computable, and uncertainty in the estimated matrix. Nash averaging remains population-dependent; adding or removing opponents can change the summary.

For multiplayer general-sum games, pairwise Elo and zero-sum Nash averaging are insufficient. Report payoff tensors or sampled profiles, unilateral deviation gains, and an explicitly defined equilibrium concept. A coarse correlated equilibrium \(\sigma\), for example, satisfies

\[
\mathbb E_{a\sim\sigma}[u_i(a)]
\ge
\mathbb E_{a\sim\sigma}[u_i(a'_i,a_{-i})]
\quad
\forall i,a'_i.
\]

No unique scalar ranking follows without additional conventions.

### 7.2 Environment-specific evaluation targets

| Environment | Belief target | Decision target | Critical failure probe |
|---|---|---|---|
| Hex / Connect-4 | Tactical outcome or continuation predictions under declared policies | Win rate and regret against strong references | Repeated transpositions, adversarial cycles, compute accounting |
| Micro-Stratego | Posterior over hidden ranks and their correlations | Information-set-valid payoff and exploitability | Memory loss, strategy fusion, opponent-action likelihood |
| Galanis-Market | Outcomes conditional on private signals and public information | Market performance and equilibrium deviations | Informative prices versus preference-driven prices |
| Proposal-Poker | Outcomes conditional on signals and governance interventions | Utility under private bias and mechanism rules | Bias leakage, selection effects, utility swaps |

For Hex and Connect-4, the current board is usually observed; “state belief KL” alone is not a meaningful tactical benchmark. Specify the uncertain future event being predicted.

For all games, use paired seeds, seat/color swaps, frozen opponents, held-out generators, and matched compute. Count world-model calls, tokens, aggregation iterations, memory cost, and wall-clock latency. A nominal \(B=128\) is not comparable across architectures if one `STEP` hides substantially more work.

### 7.3 LIR orthogonality under private bias

Let \(X\) be an exogenous outcome, \(s_i\) a private signal, \(H\) public history, and \(y_i\) a randomized private preference parameter satisfying

\[
y_i\perp X\mid s_i,H.
\]

The epistemic invariance target is

\[
P(X\mid s_i,H,\operatorname{do}(y_i=y))
=
P(X\mid s_i,H).
\]

Use paired interventions:

\[
D_{\mathrm{bias}}
=
\mathbb E\left[
\operatorname{JSD}\left(
\mu_b(\cdot\mid s_i,H,\operatorname{do}(y_i=y)),
\mu_b(\cdot\mid s_i,H,\operatorname{do}(y_i=y'))
\right)
\right].
\]

Measure it separately for:

1. the one-pass belief model;
2. consensus with the search tree held fixed;
3. consensus with search allocation allowed to respond to utility;
4. later memory after utility-dependent actions and observations.

The first two isolate direct leakage. The third exposes selection-induced finite-budget distortion. The fourth requires a causal comparison because different actions can legitimately generate different information.

Also report a desire-direction diagnostic:

\[
D_{\mathrm{desire}}
=
\mathbb E\left[
\sum_x
\left(\mu_b(x)-P^\star(x)\right)
\left(U_y(x)-\mathbb E_{P^\star}U_y(X)\right)
\right].
\]

This is an empirical diagnostic. The supplied expression proportional to \(\lambda\|\nabla U\|^2/2\) requires additional assumptions and is not a general identity for fused RL.

#### Necessary controls

- If \(y_i\) correlates with \(X\), removing its informational content can worsen beliefs. Use randomized interventions or the correct conditional reference.
- If proposals affect outcomes, compare beliefs under the same \(\operatorname{do}(A=a)\). Changing the chosen action can legitimately change predicted outcomes.
- Market prices can reveal others’ signals, preferences, wealth, and strategic behavior. Do not demand invariance to an observational price change that alters information.
- Zero-retraining utility swaps should change actions where the reference optimum changes while preserving the relevant counterfactual outcome model.
- Evaluate calibration independently of realized trading profit. Profit is not a proper scoring rule for beliefs.

Structural LIR separation proves absence of specified parameter-update paths. It does not prove truthfulness of the entire search-and-memory system.

### 7.4 Mini-ECI should earn its one-dimensional interpretation

A one-dimensional IRT model is

\[
P(Y_{ij}=1)
=
\sigma\!\left(a_j(\theta_i-d_j)\right),
\qquad a_j>0.
\]

Its assumptions include approximate unidimensionality, item invariance, and local independence. The proposed suite mixes deduction, tactical search, memory, strategic adaptation, and economic calibration. A single ability coordinate may fit poorly.

Compare it with

\[
P(Y_{ij}=1)
=
\sigma(a_j^\top\theta_i-d_j),
\]

using multidimensional IRT and game-level effects.

Keep Mini-ECI as a summary only if held-out fit and residual diagnostics support it. Otherwise publish a capability vector alongside any scalar.

Further requirements:

- distinguish training variation from match and item variation;
- model repeated budgets for the same trained agent as dependent observations;
- test differential item functioning by game, architecture, and budget;
- use fixed anchor items and opponents across league versions;
- avoid counting the same matches both as independent Elo evidence and independent IRT items;
- do not force \(\theta(B)\) to be monotone before testing compute monotonicity.

---

## 8. Falsifiable experimental sequence

All experiments below are proposed. They require implementation paths, configuration hashes, and exact seed logs before any result is recorded as evidence.

| Experiment | Isolated intervention | Primary measurement | Failure interpretation |
|---|---|---|---|
| **E1: Prefix invariance** | Re-present the same accepted edges through different leaf lists and cache layouts | Identical sufficient statistics and decoded candidate | Structural evidence is counted more than once |
| **E2: Source versus footprint** | Compare exact copies, independent same-target readings, overlapping sources, and contradictory same-footprint claims | Correct uncertainty and held-out log score | Salience is being mistaken for provenance |
| **E3: Hindsight trap** | Insert branches that lower internal entropy but worsen oracle KL | Forward proper-score credit rejects harmful plans | Self-confirmation or survivorship drives learning |
| **E4: Prospectus intervention** | Change \(p\), then \(\rho\), at identical checkpoints | Predictable changes in achieved queries and magnitude | Plans are decorative or scale-ambiguous |
| **E5: Coverage collapse** | Oversample an initially successful direction while withholding other valuable directions | Residual information coverage and signed marginal gain | Plan head learns popularity rather than value |
| **E6: Strong-anchor continuation** | Stratify roots by baseline KL without selecting a favorable test anchor | Simultaneous risk differences and useful acceptance | Safety depends on weak baselines or abstention |
| **E7: Memory sufficiency** | Compare full-history reference, compressed memory, and current observation only | Posterior KL and information-set-valid action value | Memory loses correlations or recycles errors |
| **E8: Bias intervention** | Randomize \(y_i\), with fixed-tree and free-search conditions | Belief drift, calibration, utility-swap regret | LIR protects parameters but not inference |
| **E9: League validity** | Add cyclic and multiplayer opponents | Held-out rating fit, deviation regret, IRT residuals | Scalar rankings hide strategic structure |

For the monotonicity experiment, evaluate every consecutive prefix through at least \(128\) steps, not only powers of two. Use the same nested trajectory for prefix comparisons; fresh independently sampled runs at each budget answer a different question.

Include rejection-rate and progress curves. A valid safety claim must coexist with evidence that useful computation is accepted.

The first implementation milestone should be E1–E3. If exact prefix invariance or signed plan credit fails, adding more games will make the diagnosis harder.

---

## 9. Round-1 rubric scores

Scores are subjective design assessments on a \(1.0\)–\(10.0\) scale, not experimental results or estimated probabilities. “Baseline” evaluates the charter’s promise; “risk-adjusted” accounts for unresolved mechanisms and the supplied negative evidence; “projected” assumes the proposed fixes pass their tests.

| Dimension | Baseline Charter Score | Risk-Adjusted Score Before Fixes | Projected Score With Fixes | Main reason |
|---|---:|---:|---:|---|
| **D1. Universal KV-Cache Tree Machine** | **9.0** | **7.5** | **9.2** | Strong general interface; requires explicit source identity, hypothesis semantics, and complete compute accounting |
| **D2. Active Fractal Prospectus Plans** | **8.5** | **4.5** | **8.3** | Sharing is promising; hindsight causality, coverage collapse, and missing scale currently undermine the claim |
| **D3. Bidirectional Latent Lemmas and Query Tokens** | **8.5** | **6.0** | **8.4** | Relational queries are testable; oversized latent space alone does not establish reusable lemmas |
| **D4. Present-Time Episodic Memory** | **9.0** | **5.5** | **8.5** | Necessary for hidden information; sufficiency, provenance, and strategy fusion remain substantial risks |
| **D5. Anytime Monotonicity and Self-Consistency** | **9.5** | **3.0** | **8.0** | The stated architectural guarantee is unsupported; exact conditional theorems and certified abstention make a narrower claim defensible |
| **D6. Multi-Game Elo League and Mini-ECI** | **8.5** | **5.5** | **8.5** | Broad evaluation is valuable; cyclic payoffs, multiplayer incentives, and multidimensional ability need explicit treatment |
| **Unweighted mean** | **8.8** | **5.3** | **8.5** | Projected scores are conditional research judgments |

D5’s projected score applies to the revised, assumption-explicit guarantee. It does not endorse an unconditional guarantee for arbitrary learned updates.

## 10. Recommendation

Proceed with a universal tree machine and shared prospectus head, but make **verified conditional innovation** the unit of accounting and learning.

The minimum defensible system has:

- an exact ledger for ancestry and repeated sources;
- probabilistically correct treatment of hypothetical branches;
- calibrated residual covariance for semantic overlap;
- forward-tested prospectus credit, with hindsight used carefully;
- a contextual plan metric, magnitude, and learned termination;
- memory that preserves information-set beliefs and provenance;
- an acceptance mechanism whose monotonicity claim names its assumptions;
- a league reporting calibration and strategic structure alongside scalar ratings.

The decisive result would be a system that accepts useful novel computation, rejects duplicated or misleading computation, and preserves strong beliefs across changing search policies, utilities, and game environments. That result would substantiate the architecture more convincingly than internal consistency, a larger latent space, or a higher aggregate league score alone.