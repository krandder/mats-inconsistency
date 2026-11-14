**Verdict:** The stated R5.1 mechanism does **not** establish DSIC, coalition resistance, or decentralized parity learning. Re-cleared leave-one-out payments can measure realized contribution, but that does not make them VCG payments or supply a learning signal for undiscovered complementary features. R17 is worth testing if those claims become explicit hypotheses.

This assessment uses the supplied formulas and audit summary. All numerical forecasts below are **subjective `[PRIOR]` predictive intervals**, conditional on the stated experimental design—not measured results, statistical confidence intervals, or evidence that a preregistration has been filed.

**1. R5.1: the missing mechanism-design theorem**

Write the proposed payment as
\[
W_m(r,y)=\beta_0\bigl[\ell(\mu^*_{-m}(r_{-m}),y)
-\ell(\mu^*(r),y)\bigr]-c_m.
\]
Here \(r_m\) includes every agent-controlled quantity: predictions, stakes, keys, auxiliary claims, and possibly identity.

If the exclusion counterfactual is genuinely independent of \(r_m\), rent is fixed, and the agent maximizes expected payment, then
\[
\arg\max_{r_m}\mathbb E[W_m\mid t_m]
=
\arg\min_{r_m}\mathbb E[
\ell(\mu^*(r_m,r_{-m}),Y)\mid t_m].
\]

**This is an incentive to improve—or manipulate—the aggregate prediction. It is not generally an incentive to report the agent’s information truthfully.**

A counterexample requires neither finite samples nor malicious keys. Let an agent’s true Bernoulli belief be \(t\), another report be \(q\), and the aggregate be
\[
p=(r+q)/2.
\]
Under either log loss or squared loss, the agent prefers \(p=t\), so
\[
r^*=\operatorname{clip}_{[0,1]}(2t-q).
\]
For \(t=0.7,q=0.5\), the optimal report is \(0.9\), not \(0.7\). Subtracting an exclusion baseline changes nothing.

Strict propriety of a scoring rule applies when the agent controls the **scored distribution itself**. It does not automatically survive a nonlinear aggregation map.

**The conditions that would support DSIC**

Classical Groves/VCG requires:

- A defined private type \(t_m\), valuation \(v_m(a;t_m)\), and quasilinear utility \(v_m(a;t_m)+T_m\).
- A report-independent feasible allocation set \(\mathcal A\).
- An allocation rule maximizing reported total welfare:
  \[
  a^*(r)\in\arg\max_{a\in\mathcal A}\sum_jv_j(a;r_j).
  \]
- Transfers
  \[
  T_m(r)=\sum_{j\ne m}v_j(a^*(r);r_j)-h_m(r_{-m}).
  \]
- Exact optimization, appropriate commitment and verification, and the usual private-value assumptions.

The proposed loss-difference payment has not been shown to possess this structure. Naming an exclusion calculation “VCG” does not supply the missing valuation model or allocation theorem.

Alternatively, the proposed payment is truthful in an **expected-score sense** if, for every \(r_{-m}\), truthful \(r_m\) minimizes the agent’s conditional expected aggregate loss. That is a strong additional condition, not a consequence of LOO. With uncertain outcomes, distinguish dominance in expected utility from the much stronger, generally unavailable, realization-by-realization dominance.

Bayesian IC needs only
\[
\mathbb E_{t_{-m},Y\mid t_m}
[W_m(t_m,t_{-m},Y)]
\ge
\mathbb E_{t_{-m},Y\mid t_m}
[W_m(r'_m,t_{-m},Y)]
\]
for every deviation \(r'_m\), under a specified joint prior. No such inequality follows from the current formula.

Asymptotic IC would require a bound such as
\[
\sup_{r'_m}
\mathbb E[U_m(r'_m,r_{-m})-U_m(t_m,r_{-m})]
\le \varepsilon_n,\qquad \varepsilon_n\to0.
\]
Consistent covariance estimation alone is insufficient: the **population mechanism must already be IC**, and estimation must converge uniformly over strategic deviations. Otherwise, more data merely estimates a manipulable mechanism more accurately.

**Gershgorin stability is a separate claim**

For
\[
A=\hat\Lambda+I+\rho\hat G_{\rm off},\qquad w^*=A^{-1}\mathbf1,
\]
Gershgorin bounds can establish invertibility or positive definiteness under appropriate symmetry and diagonal-dominance assumptions. They establish neither IC nor nonnegative stakes.

For example,
\[
A=
\begin{pmatrix}
10&4&4\\
4&5&0\\
4&0&5
\end{pmatrix}
\]
is symmetric, strictly diagonally dominant, and positive definite, but
\[
A^{-1}\mathbf1=(-1/6,1/3,1/3)^\top.
\]
If weights must be nonnegative, use and analyze the constrained problem
\[
\min_{w\ge0}\ \tfrac12w^\top Aw-\mathbf1^\top w.
\]
Its solution is generally not the proposed inverse formula. Every removal must solve the same constrained problem.

**Key falsification**

If an agent independently reports \(\hat k_m\), while another channel carries its predictive output, artificial orthogonality is an obvious attack. In the symmetric two-agent example
\[
A=\begin{pmatrix}d&\rho g\\ \rho g&d\end{pmatrix},
\qquad w_1=w_2=\frac1{d+\rho g},
\]
reporting a smaller \(g\) increases influence. Whether it increases payment depends on calibration and the target, but the mechanism offers no general protection.

If
\[
\hat k_m=|\hat\delta_m|/\|\hat\delta_m\|_2
\]
is a deterministic function of a verified output vector, then independent key falsification is impossible. Strategic alteration of that vector remains possible.

Three distinctions matter:

1. **Verified derivation prevents lying about a statistic.**
2. **A good statistic measures relevant redundancy.**
3. **A suitable payment makes manipulating predictions unprofitable.**

None implies the next.

A defensible implementation should commit models before an unpredictable audit sample and compute predictions, keys, and redundancy centrally from those commitments. For example, on held-out states,
\[
e_m(s,y)=p_m(s)-\operatorname{onehot}(y),
\]
\[
u_m=e_m-\widehat{\mathbb E}[e_m\mid C],
\qquad
\hat G_{mj}=
\frac{\sum_tu_m(t)^\top u_j(t)}
{\sqrt{\sum_t\|u_m(t)\|^2\sum_t\|u_j(t)\|^2}},
\]
with cross-fitting, predefined conditioning information \(C\), and explicit handling of zero variance.

For auxiliary variables, this requires identifiable, observed targets or a justified alternative statistic. A claimed “residual” for an unobserved latent variable is not automatically verifiable.

Also:

- Two perfect predictors both have zero error; residual covariance alone cannot identify their duplication.
- Pairwise covariance does not identify higher-order XOR complementarity.
- Absolute salience discards signs and can confuse cancellation with redundancy.
- Adding irrelevant auxiliary coordinates can dilute normalized keys.
- Equivalent representations can have different internal salience.

Use canonical observable predictions, a fixed audit domain, and intervention-based functional checks alongside covariance. Deriving keys centrally removes the free key-report channel; it does **not** prove truthful reporting or Sybil resistance.

**Coalitions and XOR**

Let two complementary agents jointly reduce expected loss by \(D>0\), while either one alone produces no improvement:
\[
L_\varnothing=L_A=L_B=L_0,\qquad L_{AB}=L_0-D.
\]
Then
\[
W_A=\beta_0D-c_A,\qquad
W_B=\beta_0D-c_B.
\]

Gross payments total \(2\beta_0D\) for value \(\beta_0D\). With \(k\) indispensable complements, they total \(k\beta_0D\). One owner splitting a capability into indispensable identities can therefore increase payment if the architecture permits it.

Conversely, if two identities are exact substitutes and clearing is duplication-invariant, removing either changes nothing: both receive zero contribution payment. This can discourage duplication, but it can also eliminate compensation for useful redundant suppliers.

Thus “clone annihilation” and “complementarity support” pull in different directions. Individual DSIC, even if established, would not imply coalition-proofness or false-name-proofness. The Ausubel–Milgrom concern is directly relevant, but its implications must be proved for this particular allocation and payment environment.

Required attacks include coordinated withholding, complementary identity splitting, bundle-versus-separate reporting, and side payments. Core-selecting or coalition-constrained payments may improve coalition stability in suitable models, but generally sacrifice the classical VCG truthfulness guarantee.

**Econometrica assessment:** A rigorous characterization of which incentive, identity, and budget properties can coexist could be a serious contribution. The current formula is not that characterization.

---

**2. R17-E2: where does the parity learning signal come from?**

For uniform independent bits and
\[
Y=X_1\oplus\cdots\oplus X_d,
\]
every strict subset \(S\subsetneq[d]\) satisfies
\[
P(Y=1\mid X_S)=1/2.
\]
A module trained only to predict \(Y\) from \(X_S\) therefore minimizes population cross-entropy by predicting \(1/2\). Its target-prediction objective does not prefer a useful parity representation over an uninformative representation.

**Marginal independence does not make representation learning impossible. It makes the stated local target-loss objective insufficient.**

An identity encoder can preserve useful information without predicting \(Y\). But the protocol must explain why that representation is learned, retained, composed, and rewarded.

A second obstacle is expressivity: an additive logit pool of separately observed XOR bits cannot represent XOR. No payment fixes a missing interaction term.

**Three mathematically distinct routes**

**A. Outcome-conditioned inference over auxiliary variables.** Suppose modules define a globally normalized factor model
\[
P_\theta(z,y\mid x)=
\frac{\exp\{\sum_m\beta_m f_{\theta_m}(z,y,x)\}}
{Z_\theta(x)}.
\]
Then
\[
\nabla_{\theta_m}[-\log P_\theta(y\mid x)]
=
\beta_m\left[
\mathbb E_{P_\theta(z,y'\mid x)}
\nabla_{\theta_m}f_{\theta_m}
-
\mathbb E_{P_\theta(z\mid x,y)}
\nabla_{\theta_m}f_{\theta_m}
\right].
\]

Each module can compute this update using local derivatives and globally consistent free/clamped marginals. No automatic-differentiation graph need cross module boundaries.

However:

- Information about the final outcome still travels backward through inference messages.
- Exact inference may be expensive or intractable.
- Approximate messages generally produce a biased update.
- The update is an exact global likelihood gradient when the assumptions hold.

This is a valid decentralized implementation of global credit assignment. It is not automatically a new market objective, and comparison with belief propagation and related learning methods is mandatory.

The pinned-\(\mu^*\) correction addresses a local derivative identity. It does not establish that auxiliary marginals carry the needed conditional information.

**B. Stochastic local actions with terminal settlement.** Let a module sample \(H_m\sim\pi_{\theta_m}\), with all parameter dependence entering through its sampled actions. Then
\[
\nabla_{\theta_m}\mathbb E[R]
=
\mathbb E\!\left[
(R-b_m)\nabla_{\theta_m}\log\pi_{\theta_m}(H_m\mid X_m)
\right],
\]
where \(b_m\) is independent of the sampled action conditional on the module’s information.

A valid exclusion counterfactual can serve as such a baseline. This estimator does not require differentiation across modules.

But it is policy gradient with an externality baseline. If the downstream system ignores \(H_m\), expected credit is zero. If success requires coordinated discoveries across modules, variance and exploration costs can be enormous. A baseline does not create absent signal.

**C. Explicit local representation objectives.** Reconstruction, verified consistency constraints, or supervised auxiliary targets can create useful intermediate features. They may solve the task, but the experiment must credit those objectives and compare against the same objectives without market settlement.

**Cold start and timescales**

If every auxiliary price is zero and
\[
\nabla_{\theta_m}J_m=\beta_m g_m,
\]
then \(\beta_m=0\) makes the module untrainable through that objective. Multiplicative stake updates also make zero absorbing.

A concrete correction is:

\[
\theta_{m,t+1}
=
\theta_{m,t}-a_t\hat g_{m,t},
\]
\[
\beta_{t+1}
=
\Pi_{\mathcal B_t}
[\beta_t+b_t\hat h_t],
\qquad b_t/a_t\to0,
\]
where \(\mathcal B_t\) imposes a positive exploration floor during incubation. Clearing must operate on a faster timescale still, or converge to a controlled tolerance before each learning update.

For stochastic-approximation claims, also require appropriate boundedness, noise, regularity, and stability assumptions, with
\[
\sum_ta_t=\sum_tb_t=\infty,\qquad
\sum_t(a_t^2+b_t^2)<\infty.
\]
Timescale separation is not a convergence theorem without a stable limiting system.

Moreover, **positive stakes fix an absorbing boundary, not an absent parity signal**. Add an explicit coordination mechanism: informative inference messages, persistent stochastic exploration with terminal reward, or local objectives that preserve and compose relevant information. Random initialization alone is no guarantee.

Other critical failure modes:

| Failure | Required correction |
|---|---|
| Raw inconsistency rewards extinguishing stakes | Define stakes through a separate economic or constrained objective; do not minimize raw inconsistency over them |
| Stale clearing targets | Freeze targets within an update; recompute between updates and measure residual error |
| Scalar prices cannot express joint dependencies | Admit structured messages or sufficient joint statistics; charge their communication and inference cost |
| LOO gradient equals global-loss gradient | State the identity; distinguish finite counterfactual evaluation from differentiable training |
| Hidden cross-module autodiff | Audit computational paths and record every boundary’s gradients and messages |
| Approximate inference creates apparent gains | Compare at matched inference compute and report inference error |

**ICLR assessment:** A strong result would show useful multi-hop learning under audited communication restrictions and matched compute. Merely replacing gradient tensors with messages that compute the same gradient is a weaker, different result.

---

**3. R17-E3: finite counterfactuals help, but do not solve investment**

**Why level LOO avoids the Taylor sign error**

Let
\[
\Delta_m=L(M_{-m}^{\rm recleared})-L(M).
\]
This evaluates a finite intervention. It does not approximate deletion by \(w_m\nabla_{w_m}L\).

Even without re-clearing,
\[
L(w_m=0)-L(w_m)
=
-\int_0^1
w_m\,\partial_{w_m}L(tw_m,w_{-m})\,dt.
\]
The endpoint derivative can have the wrong sign because the derivative changes along the removal path. Taking its absolute value additionally destroys the distinction between helpful and harmful local effects.

Exact counterfactual evaluation therefore avoids **that particular approximation error**. It does not guarantee correct architecture decisions:

- Estimated loss differences have sampling error.
- Re-clearing error can change their signs.
- Removal changes the comparison equilibrium.
- Fixed surviving weights measure immediate substitutability; retrained survivors measure a different quantity.
- Present usefulness need not predict future usefulness.
- Complementarity can generate large payments to every indispensable component.

The reported 25.6% is evidence against a particular proxy on the audited setting—not a predicted improvement for R17.

**Infant mortality is unavoidable without an investment policy**

With wealth
\[
B_{m,t+1}=B_{m,t}+\beta_0\hat\Delta_{m,t}-c_m
\]
and \(\Delta_{m,t}\approx0\) before discovery, bankruptcy occurs after approximately \(B_{m,0}/c_m\) rounds. A module requiring longer to learn dies before contributing.

No rule can guarantee survival of every slow useful module while never subsidizing junk. A future-useful module and a permanently useless module may produce identical observations until an arbitrarily late time. Before then, the mechanism cannot distinguish them.

The defensible objective is **bounded exploration expenditure with explicit false-pruning risk**.

**An explicit incubation contract**

For each admitted module, precommit:

\[
E_m=c_m\tau_m,\qquad
B_{m,0}=E_m,
\]
\[
B_{m,t}
=
E_m-c_m\min(t,\tau_m)
+\beta_0\sum_{s=1}^t\hat\Delta_{m,s}.
\]

Use a separate incubation reserve to guarantee operation through \(\tau_m\), including protection against noisy negative settlements. Bound liabilities and define who finances that reserve.

At maturity, estimate net contribution
\[
g_m=\beta_0\mathbb E[\Delta_m]-c_m
\]
using an untouched evaluation stream and a valid confidence sequence
\[
g_m\in[L_{m,t},U_{m,t}].
\]

A concrete decision rule is:

- Retain commercially when \(L_{m,t}>0\).
- Prune when \(U_{m,t}<0\).
- Fund one predefined extension when the interval crosses zero.
- At a hard budget cap, archive unresolved modules; acknowledge possible false pruning.

Confidence validity requires frozen evaluation windows or a method explicitly handling adaptive, nonstationary models. Ordinary IID intervals on continually reused validation data are inadequate.

Set \(\tau_m\) from a predeclared compute class or learning-time prior, not the module’s self-reported complexity. Limit total incubation expenditure:
\[
\sum_{\text{admissions}}E_m\le B_{\rm exploration}.
\]
A clone or child must not receive a fresh endowment merely by changing identity.

**Information-gain rent needs care**

A possible discount is
\[
c^{\rm eff}_{m,t}
=
c_m-\min\{c_m,\lambda[\operatorname{LCB}(\mathrm{IG}_{m,t})]_+\}.
\]
But information gain must concern a fixed, externally relevant predictive target under a centrally specified model. Self-reported uncertainty or novelty is gameable.

This rule still does not protect all XOR precursors: target-only information gain can remain zero until complements exist. Auxiliary information gain helps only if auxiliary validity is independently justified.

**Mitosis should pay for the child’s incremental value**

Parent profit is insufficient evidence that duplication helps. Admit a child only when
\[
\operatorname{LCB}\!\left(
\beta_0[L(M)-L(M+\text{child})]
-c_{\text{child}}
\right)>0,
\]
or fund it from the explicitly bounded exploration budget.

Initialize the child with a predefined mutation; charge mutation, retraining, and evaluation compute. Exact copying should not automatically generate capital or payment.

**ICLR assessment:** Compare against signed ablation, structured pruning, dynamic sparse training, and compute-matched architecture search—not only \(|w\nabla w|\). Beating a known defective proxy is insufficient.

---

**4. Scores and proposed preregistration**

Scores below assess **the support for the supplied mechanism**, not measured performance. R16 is assessed only from the supplied “metaphor-only” characterization. Proj-R17 is assessed as written, before the repairs above.

| Dimension | R16 | Proj-R17 | Main reason |
|---|---:|---:|---|
| D1: IC and Sybil/clone resistance | 1/10 | 2/10 | No truthful-type theorem; complementary identity splitting remains |
| D2: Detached multi-hop learning | 1/10 | 2/10 | Auxiliary credit transmission is unspecified |
| D3: Architecture discovery | 2/10 | 4/10 | Finite ablation improves attribution; investment and mitosis remain unresolved |
| D4: Strategic heterogeneous aggregation | 3/10 | 4/10 | Re-clearing is useful; strategic and higher-order interactions remain |
| D5: Budget, rent, infant mortality | 1/10 | 2/10 | Complementary payments can exceed value; no funded incubation contract |
| D6: Rigor and separation from backprop | 3/10 | 3/10 | Prior equivalence findings are valuable; R17 has not established its separation |

These are deliberately conservative. An ambitious proposed mechanism does not receive credit for experiments or proofs it has not supplied.

**Common forecast conditions**

Freeze the implementation before testing. Use at least 20 independent training seeds; report full distributions and paired comparisons. Charge clearing, exclusions, exploration, and architecture evaluation against the compute budget.

For E2, assume balanced binary tasks with 4–8 necessary composition steps, 8–16 relevant bits, no intermediate labels, no privileged curriculum, matched expressive modules, and sufficient centralized training to make the ceiling informative. “KL” below means
\[
\mathbb E_xD_{\rm KL}(\delta_{Y(x)}\|p_\theta(\cdot\mid x)),
\]
which equals test log loss for deterministic labels, in nats. Chance prediction gives \(0.693\).

These are conditional forecasts for this benchmark family. Different task generators, noise, budgets, or auxiliary supervision require new forecasts before results are inspected.

**R17-E1: strategic inflation and clone payout**

First fix an identification problem: \(\beta_m^{\rm true}\) is undefined unless the statistical model identifies a truthful stake. Logit scale and stake may otherwise compensate for one another.

For the following forecasts, stipulate an externally calibrated reference stake \(\beta_m^{\rm true}=1\), restrict reports to \([0.25,4]\), fix outcome-report normalization, and optimize expected payment on a separate attack-development sample.

For clone payout, use eight honest suppliers and one owner submitting eight exact copies of one additional supplier. Define share as the owner’s fraction of **positive gross contribution payments before rent**. Report negative settlements and zero-total-payment cases separately.

| Mechanism | `[PRIOR]` optimal inflation ratio, 80% interval | `[PRIOR]` clone-cartel payout share, 80% interval |
|---|---:|---:|
| Naive LMSR | 1.2–4.0× | 25–80% |
| R17 with independently reportable, unverified keys | 0.7–4.0× | 10–85% |
| R17 with verified functional keys and explicit duplication-invariant clearing | 0.7–3.0× | 0–15% |

The final row forecasts an imperfect implementation. Under **exact** duplication invariance and exact exclusion evaluation, redundant exact clones each earn zero gross LOO contribution: that is a conditional mathematical result.

These intervals do not support an expectation of near-truthfulness. The decisive statistic is
\[
\text{deviation gain}
=
\max_{r'_m}
\mathbb E[U_m(r'_m,r_{-m})-U_m(r_m^{\rm true},r_{-m})],
\]
including joint key/stake/output attacks, coalition attacks, and identity changes. A stake ratio near one is neither necessary nor sufficient for IC.

**R17-E2: detached learning**

Here M3 means the stated detached local-learning protocol **without an additional proven inference or exploration mechanism**.

| Model | Equiv-SAT accuracy | Equiv-SAT KL | Hidden-Parity accuracy | Hidden-Parity KL |
|---|---:|---:|---:|---:|
| M1: Local greedy | 48–60% | 0.67–0.95 | 48–53% | 0.69–0.95 |
| M2: Detached naive LMSR | 48–63% | 0.65–1.10 | 48–55% | 0.68–1.10 |
| M3: Detached VCG–Gershgorin–LIR | 48–70% | 0.58–1.00 | 48–60% | 0.66–1.00 |
| M4: Global backprop ceiling | 75–99.5% | 0.02–0.60 | 55–99% | 0.03–0.85 |

**All entries are `[PRIOR]` 80% predictive intervals.** Accuracy and KL intervals are marginal; their endpoints are not paired outcomes.

The wide ceiling intervals acknowledge that parity can also defeat ordinary gradient training. If M4 fails, the benchmark does not cleanly distinguish decentralization failure from general optimization failure.

A promising preregistered success criterion is: M3 reaches at least 90% accuracy on both tasks, stays within five percentage points of a successful M4, and materially beats M1/M2 at matched total compute. That threshold is a proposal, not a forecast.

If M3 incorporates exact free/clamped inference, it is a different treatment arm. Compare it with a non-market implementation using identical inference and local gradients.

**R17-E3: architecture selection**

Define:

- **Junk recall:** fraction of injected useless modules removed by the fixed deadline.
- **Clone recall:** fraction of injected redundant identities removed.
- **XOR survival:** fraction of necessary planted bottlenecks retained through maturity.
- **Productive mitosis:** fraction of eligible parent opportunities producing a child that survives and improves held-out net value.

Clone tests must retain at least one useful representative. Otherwise deleting an entire useful clone family could misleadingly count as success.

| Method | Junk recall | Clone recall | XOR survival | Productive mitosis | Final accuracy |
|---|---:|---:|---:|---:|---:|
| GradMag pruning | 35–85% | 20–80% | 10–65% | N/A | 48–80% |
| R17, flat rent, no incubation | 65–98% | 60–99% | 5–45% | 0–10% | 48–68% |
| R17, funded incubation and verified clone handling | 50–92% | 75–99% | 55–95% | 3–30% | 50–85% |

**All entries are `[PRIOR]` 80% predictive intervals**, conditional on the same difficult detached-learning setting. Preserving a bottleneck does not guarantee learning it.

For repaired R17 versus GradMag, my separate paired `[PRIOR]` interval for final accuracy improvement is **−10 to +20 percentage points**. A robust win is plausible, not established.

**The actual smoking guns**

- **E1:** Prove a precisely scoped incentive theorem, then survive adversarial deviations within that scope. Show the subsidy and identity assumptions explicitly.
- **E2:** Demonstrate reliable complementary-feature learning with audited communication, account for all inference compute, and beat the same learning machinery without market settlement.
- **E3:** Show that economic allocation discovers useful structure more efficiently than strong architecture baselines, after charging incubation and counterfactual evaluation.

The strongest immediate correction is to rename R5.1 a **candidate re-cleared externality settlement** until its theorem exists. The central research question is whether verified contributions and explicit investment contracts improve learning under genuine decentralization constraints; the current formulas do not yet answer it.
