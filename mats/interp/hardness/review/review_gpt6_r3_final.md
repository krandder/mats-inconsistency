The revision is **not yet cleared**. Most core proof repairs work, but the explicit-design claim remains false, the counting reduction has a degree mismatch, and Sections 7–9 still overstate what the restricted results establish.

Checked [draft.md](/home/agent/workspaces/mats/work/hardness/draft.md), Sections 1–9, against both prior reports and versions. Ignored the appended measurement placeholders. Snapshot checked **2026-10-29 19:05 UTC**, SHA-256 `8b50fce140d87790533bcfde970d6b296197cc19a5f3ebaa11d0fe62e333c530`. No files written; numerical checks were sequential, under the requested limits.

1. **Corollary 4.3: padding and smoothed variance — FIXED.** Lines 277–279.

   The three disjoint branches give
   \[
   \Pr(C''=1)=\frac14+\frac18+\frac14\Pr(C=1)
   =\frac38+\frac14\Pr(C=1).
   \]
   Thus the Boolean variance is at least \(15/64\).

   The smoothing claim is valid, but include its missing calculation. If \(B=C''(\operatorname{sign}x)\), \(0\le f\le1\), and \(f=B\) outside an event of probability \(\delta\), then
   \[
   |\operatorname{Var}f-\operatorname{Var}B|\le3\delta.
   \]
   With circuit size counting its inputs,
   \[
   \delta\le\frac1{48\sqrt{2\pi}},\qquad
   \operatorname{Var}f\ge\frac{15}{64}-\frac3{48\sqrt{2\pi}}
   =0.209441\ldots>\frac15.
   \]
   The acceptance gap also survives smoothing and the estimator’s \(1/24\) error.

2. **Corollary 4.3: counting reduction — PARTLY FIXED.** Lines 268–283.

   The clause-indexed lists preserve assignment multiplicities and the satisfying-assignment count. The nondeterministic output semantics are now correct.

   However, this construction gives \(d=O(v^k)=O(\log^k n)\), while the proof still guesses \(H\) of degree \(O(n\log^2 n)\). The referenced construction only guarantees
   \[
   \deg Q=O(nd).
   \]
   For general fixed clause width \(k\), the stated degree cap need not contain \(Q\), so completeness is not established.

   Replace the degree bound throughout by \(O(nd)=n\,\mathrm{polylog}\,n\). This preserves the running-time contradiction and soundness for sufficiently large \(n\). Also explicitly verify the guessed prime and its range.

   **Fast multipoint evaluation — FIXED.** Evaluating all \(H(i)\) in \(\widetilde O(n)\) time, then summing modulo \(p>n^2\), repairs the previous near-quadratic bottleneck.

3. **Theorem 4.1(c): restoration and global output — FIXED.** Lines 207–234.

   The accounting is consistent: restored wires have error at most \(2\tau\); OR, AND, and NOT have errors at most \(6\tau,5\tau,3\tau\). Thus \(\tau=1/24\) works, including restoration after binarization.

   The global decomposition
   \[
   f=R_{\rm out}+(\xi_1-\xi_2),\qquad 0\le R_{\rm out}\le1,
   \]
   correctly uses the final noise’s **unconditional** zero mean. The false bounded-individual-unit and relative-layer-RMS claims have been removed. The exponent \(1152\) is correct.

4. **Corollary 2.2: constants, small-network variant, embedding — PARTLY FIXED.** Lines 92–108.

   The trapezoid moments, finite-ramp ratio, triangular example, and thin-ramp limit are correct. The Gaussian constants \(5.9\) and \(3.3\) remain valid under the small-ramp condition used in their derivation; explicitly restore \(3.3w'<1\).

   **NEW ERROR:** “One unit suffices per network” contradicts the construction. It is **one trapezoid, four ReLU units**. The constant-parameter hardness conclusion is correct.

   The displayed single-bump risk formulas require common bump moments \(a,b\), as in equal-length cells under uniform input. Equal cell probabilities alone do not ensure common moments for Gaussian trapezoids. State the uniform version here.

   **Embedding — FIXED.** The positive/negative input and accumulator pairs support the stated sufficient depth
   \[
   1+\left\lceil\frac{4m}{n-4}\right\rceil.
   \]
   A brief scheduling explanation would make this explicit: each layer accumulates the previous hinge batch while producing the next batch. The disjoint-support, finite-second-moment extension also repairs the homogenization argument.

5. **Theorem 2.3: independent \(\psi\), MC bound, factor — PARTLY FIXED.** Lines 110–118.

   Independence fixes the variance calculation:
   \[
   \operatorname{Var}(\psi+g)=\frac12+\operatorname{Var}g\le\frac34.
   \]
   A centered, scaled clamp provides the stated \(\psi\). For uniform \(x_2\), an explicit choice is
   \[
   \psi(x_2)=\sqrt6\bigl(\operatorname{clamp}(x_2,0,1)-1/2\bigr).
   \]
   Thus \(3/(4N)\) is correct.

   The proof establishes a limiting factor, however. From squared error \(1/16-o(1)\), it yields
   \[
   \frac N{12}(1-o(1)),
   \]
   not the unqualified exact factor \(N/12\) for finite ramps. Retain the limiting qualification or supply a finite-ramp argument.

6. **Proposition 3.2: Lévy bound — FIXED.** Lines 138–150.

   The two horizontal margins and combined vertical slack give
   \[
   d_L\le\max(2\delta,\eta+\mathrm{adv}).
   \]
   Its local consequence correctly concerns the marginal distribution of means.

   That qualification is subsequently contradicted by Section 5’s opening assertion that undetectable planting “do[es] not make means hard.” Remove that broader assertion.

7. **Proposition 3.5: joint invariance — FIXED; explicit designs — NEW ERROR.** Lines 177–194.

   Joint network-law invariance is now the correct hypothesis. It proves
   \[
   \operatorname{Risk}(X)=\mathbb E_Q\operatorname{Risk}(QX).
   \]
   It does **not** prove that either risk is bounded by Monte Carlo’s risk.

   A counterexample satisfying joint invariance is
   \[
   f_w(x)=(wx)_+,\qquad w\sim N(0,2),\quad x\sim N(0,1).
   \]
   For the explicit two-point design \(X=(1,1)\),
   \[
   \mathbb E_w(\widehat\mu_X-\mu)^2
   =1-\frac2{\sqrt{2\pi}}+\frac1\pi
   =0.520425\ldots,
   \]
   whereas two-sample MC has risk
   \[
   \frac{1-1/\pi}{2}=0.340845\ldots.
   \]

   Delete “any explicit design … inherits the randomized guarantee” and the unconditional **uniform** conclusion. What survives is nonuniform existence by averaging, plus rotation equivalence. An explicit design requires its own error guarantee. Proposition 4.4(3) repeats the invalid inference.

8. **Section 5: JSH(F) — FIXED as a conditional formulation.** Lines 356–364.

   The assumption now directly bounds estimation of the joint sum and separately assumes the required second moments. This repairs the invalid single-scale-to-multiscale inference.

   Its strength should remain explicit: it assumes the principal computational lower bound, including resistance to shared computation. When translating to time, specify evaluation and aggregation costs; the oracle interpretation uses oracle-call cost. Conclusions transferred with \(1\pm o(1)\) hypotheses should retain corresponding asymptotic qualifications.

9. **Proposition 5.2, items 1–3 — FIXED within the stated oracle model.** Lines 366–377.

   - **Item 1:** Oracle-independent initial information or charged preprocessing replaces the false context-independence claim.
   - **Item 2:** \(O(\log J+\log(1/\epsilon))\) bits is the correct range-dependent encoding bound.
   - **Item 3:** Charging information carried by both \(\theta\) and \(\pi\), through \(Q+t_T+t_E\), repairs the accounting.

   These conclusions use the model’s oracle-independent seed and an externally drawn context. They do not automatically cover oracle-dependent context selection.

   **Interpretation — PARTLY FIXED.** The concluding “hard instances … must place their difficulty in computation that nobody performed” is broader than the oracle result, and the following sentence does not establish hardness of random He-ReLU networks.

10. **Theorem 6.5: independence and asymptotic theorem — FIXED; interpretation — PARTLY FIXED.** Lines 432–471.

    The condition \(A\perp(B,\xi)\) closes the information leak. The conditional expectation, orthogonal decomposition, integrated covariance bound, and MC second-moment bound are valid. The quadratic observable is now explicitly distinguished from ARC’s per-neuron means.

    The population ratio
    \[
    N_*=\frac{\mathbb E_W\operatorname{Var}_x f_L}
    {\operatorname{Var}(\mathbb E[S\mid A])}
    \]
    is an **upper bound on sample equivalence**, not the optimal estimator’s actual equivalence. Its asymptotic ceiling is correctly constant, about \(91.2\) for the stated example.

    The numerical separation is substantially improved: the table identifies point estimates without confidence intervals, and the mean-field column reproduces correctly. But “is worth at most … (measured)” still sounds like a certified ceiling. Say instead that the **estimated upper-bound ratio** is approximately \(1.26\)–\(3.04\). These estimates do not certify integer sample counts; treating them as exact would give sufficient counts \(2,3,3,4\), in the corresponding rows.

    The extension to statistics of \(A\) at lines 464–465 remains unproved. Small explained variance of a **single-input gain** does not establish small explained variance of \(\mathbb E[S\mid A]\). For supplied statistics \(T(A)\), a relevant bound is
    \[
    \mathbb E\operatorname{Var}\!\left(\mathbb E[S\mid A]\mid T(A)\right).
    \]
    Establish that quantity or remove the example as a claimed consequence.

11. **Lemma 6.4: fourth moment and remainder — FIXED.** Lines 419–430.

    The exact moment
    \[
    M_4(n)=1+\frac{30}{n}+\frac{251}{n^2}+\frac{558}{n^3}
    \]
    is correct. Starting the enhanced recurrence at layer \(3\), keeping the sharp concentration exponent, and using the final multiplier without requiring a good final cosine repair the proof.

    The remainder vanishes for \(L=O(n)\). For the four table configurations its displayed upper bounds are approximately
    \[
    52.30,\quad592.18,\quad75.43,\quad923.44.
    \]
    Consequently those finite-width certified lower bounds are vacuous.

    **NEW ERROR, minor:** “at \(n\le128\) the remainder bound exceeds 50” is not true for every depth. At \(n=128,L=2\), the displayed remainder is approximately \(1.93\). Restrict that sentence to the table configurations.

12. **Proposition 6.7 — FIXED.** Lines 487–494.

    Positive scale factors, the quadratic-observable distinction, and
    \[
    \operatorname{Var}\!\left[\prod_l(1+t_l)-1\right]
    =\prod_l(1+v_l)-1
    \]
    correctly replace the previous unrestricted linear-accumulation assertion. The unsupported uniform \(O(\sigma^2)\) noise claim is gone. Relative-error terminology presumes a nonzero reference output.

13. **Proposition 6.9 versus Heuristic 6.9′ — PARTLY FIXED.** Lines 509–524.

    The essential repair succeeds: deterministic mean-field angle asymptotics are proved separately, while multilayer transport and propagated-error orthogonality are explicitly unproved.

    The finite-depth arithmetic still mixes two approximations. Under the substitution \(\theta_l=3\pi/l\), for \(4\le\ell\le L\),
    \[
    \prod_{l=\ell}^{L}(1-3/l)
    =\frac{(\ell-3)(\ell-2)(\ell-1)}{L(L-1)(L-2)}.
    \]
    Summing these exact substituted products gives \((L-3)/4\). The expression
    \[
    L/4+1/2+1/(4L)
    \]
    instead sums \((\ell/L)^3\). Both motivate leading growth \(L/4\); they are not the same finite-depth identity.

    Retain actual gate factors in early layers, and label the accumulation expression as an approximation under the additional scalar transport model.

14. **Lemma 6.10 — FIXED.** Lines 526–540.

    The new adapted conditional-moment hypotheses directly prove the identity by successive conditioning; layerwise independence and conditional centering are unnecessary.

    The Hermite coefficients and limit \(1-e^{-5r_D\alpha}\) are correct. The true-trajectory requirement, single-input scope, and unresolved heavy-tail simulations are now disclosed. Add \(a+b>0\) for the normalized ratio; the unnormalized identity also covers the degenerate case.

15. **Theorem 6.12 — FIXED.** Lines 553–558.

    The corrected bound
    \[
    |\mu-\widehat\mu|
    \le C\eta^L d_F(P_0,Q_0)
      +Ce\frac{1-\eta^L}{1-\eta}
    \]
    includes the initial discrepancy, the correct geometric sum, \(0\le\eta<1\), and depth-uniform \(C\). Finite dual distances address the seminorm compatibility issue.

16. **Proposition 6.13 — FIXED.** Lines 560–568.

    The exact Gaussian TV coefficient, finite/infinite image-diameter distinction, and Gaussian-tail asymptotic are correct. Diameter scaling is now conditional, and the conclusion no longer rules out all worst-case contraction or other metrics. The mixing observations are explicitly empirical.

17. **Other previously reported local issues.**

    | Item | Verdict | Check |
    |---|---|---|
    | Section 1 definitions | **FIXED** | Noise included in variance; positive-variance convention; continuous uniform input; actual passes distinguished from normalized work; arithmetic/bit costs separated. |
    | Theorem 2.1 | **FIXED** | Revealed signs require \(g_j(x)\ne0\); risk includes algorithmic coins; generalized supports have finite second moments. |
    | Proposition 3.1 | **FIXED** | Upper-bound wording, efficiently sampled inputs, and absolute-precision scope repaired. |
    | Corollary 3.3 | **FIXED** | Exact realization is on Boolean inputs; Gaussian use has an explicit strip error. A negligible Gaussian conclusion requires choosing \(K\) accordingly. |
    | Proposition 3.4 | **FIXED** | Correlation defined and He-ReLU consequences restricted by observable and depth. |
    | Theorem 4.1(a),(b) | **FIXED** | Repeated-input preprocessing, integer \(K\), and “depth at most” qualifications supplied. |
    | Corollary 4.2 | **FIXED** | Accepting-branch semantics, advice quantifiers, and common-size padding supplied. |
    | Proposition 4.4 | **PARTLY FIXED** | Context/advice qualification repaired; item 3 still relies on the false explicit-design inference. The asserted trainer-seed averaging quantifier needs a source quotation; the supplied source note does not establish it. |
    | Theorem 5.1 | **FIXED** | Initial information, domain separation, hard oracle budget, optimal-estimator convention, inclusion assumptions, and normalized-work convention supplied. |
    | Lemmas 6.1–6.3 | **FIXED** | Auxiliary extinction recursion is consistent; \(\lambda=0.215\) gives rate \(0.064628\ldots\), validating \(0.0646\). |
    | Lemma 6.6 | **PARTLY FIXED** | Domains and cancellation language repaired. Explicitly require all displayed pairings to be finite; finite signed measures alone need not integrate unbounded backward functions. |
    | Proposition 6.8 | **PARTLY FIXED** | Exact theorem’s hypotheses repaired. “Applies only approximately” is unsupported without an approximation bound, and full support span is not guaranteed at every deeper layer. |
    | Proposition 6.11 | **FIXED** | Independence, positive trace, RMS normalization, preactivation scope, and empirical covariance qualification supplied. |

18. **Sections 7–9 consistency — PARTLY FIXED.**

    Several previous overclaims survive the revised statements:

    - **Section 7.2, lines 588–591:** The expansion concerns interactions among the independent **gain variables**. These gains depend on preceding weights, so it does not automatically characterize arbitrary descriptions using a few weight layers. “Misses most” is also false generally: at \(\alpha=1/4\), orders through five capture approximately **99.74%** of gain variance. State the positive limiting truncation tail for this particular expansion.
    - **Section 7, lines 594–596:** Lemma 6.4 establishes correlation of the quadratic output scale, not every linear-readout output.
    - **Section 8, line 612:** “Ruled out” still transfers single-input surrogate results to general estimators of input-averaged means. That transfer remains absent.
    - **Section 9, lines 621–623:** Hard-coded-design existence must remain explicitly nonuniform; it does not settle uniform deterministic estimation.
    - **Section 9, lines 624–629:** The stated “necessary conditions” exceed the theorems. The scale precision \(\epsilon/\sqrt L\) requires assumptions such as independent centered errors. Multiplicative errors can cancel exactly, for example \((1+t_1)(1+t_2)=1\). Also, \(\sum e_\ell^2(\ell/L)^3\) models a **squared-error expectation**, not the signed telescoping error itself.
    - **Sections 8–9, lines 614 and 635–637:** “Unchanged” noisy hardness and “every standard” architecture retain unsupported generality. Preserve the small absolute-noise condition and the near-linear DAG size requirement; carried-wire overhead for MLPs remains unresolved.
    - **Section 9, lines 640–643:** Empirical forgetting in chaotic/noisy systems does not establish Theorem 6.12’s backward-function contraction hypothesis. State the implication conditionally.

Remaining required changes, ordered by severity:

1. Remove the false explicit-design/uniform-derandomization inference and its downstream uses.
2. Restrict Sections 7–9 to the proved observables and estimator classes; remove unsupported necessary conditions and architecture/noise generalizations.
3. Repair Corollary 4.3’s degree bound to \(O(nd)\) and explicitly validate the guessed prime.
4. Qualify Theorem 6.5’s measured ratios and remove or prove its extension to statistics of early layers.
5. Finish the local repairs: four-unit wording, finite-ramp factor, common bump moments, heuristic product versus sum, finite pairings, and unsupported approximate-orthogonality language.
