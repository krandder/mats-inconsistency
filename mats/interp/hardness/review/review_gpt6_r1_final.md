The draft is **not ready to circulate as fully proven**. The principal black-box bounds, covariance identities, noiseless circuit reduction, and Chen–Tell reduction survive scrutiny. The main failures are the inference from single-scale SH(F) to multiscale hardness, the claim that context-independent advice contains no useful information, and several overstatements about uniform derandomization and worst-case network recovery. Concurrent edits repaired the original Lévy-distance constant and noisy-gate accounting, but introduced an unjustified variance bound in Corollary 4.3.

Reviewed snapshot: **2026-10-29 18:50:18 UTC**, Sections 1–5 SHA-256 `94366a216f5dbc8e91544d08a69f445a58a0dbee754f1ee6603d3d52a5c79c52`. Earlier versions encountered during review are distinguished below. The managed filesystem policy prohibits writes, including the requested scratch directory, so **`/tmp/hardness_review_r1/review.md` could not be created**. The full report follows.

## Section 1: definitions and scope

**Verdict: CORRECT WITH A FIXABLE GAP.**

The basic access models and risk definitions are useful, but several conventions need correction.

- For noisy networks, define \(V(W)=\operatorname{Var}_{x,\xi} f_W(x,\xi)\) when a forward pass samples both input and noise. Otherwise the asserted identity \(\mathrm{MSE}_{MC}=V/N\) is not generally true.
- State a convention for \(V(W)=0\), or restrict normalized comparisons to positive variance. The revised Theorem 2.3 addresses its particular zero-variance example, but the definitions still leave \(0/0\) undefined.
- Include uniform continuous input, used in Section 2, among the permitted input laws.
- At an actual budget \(T\), ordinary Monte Carlo uses \(\lfloor T/c_{\mathrm{fwd}}\rfloor\) complete passes. The expression \(Vc_{\mathrm{fwd}}/T\) is an idealized work-normalized curve unless divisibility is assumed.
- Arithmetic-operation cost and bit-operation cost must remain distinct. \(E\) edges do not imply \(O(E)\) bit-time evaluation of arbitrary rational-weight networks. Weight encoding, intermediate precision, and approximate Gaussian generation matter.
- Replace “below one forward pass nothing is possible” with “the ordinary sampling baseline requires at least one forward pass.” Constant estimates and special white-box instances provide immediate counterexamples to the former wording.

The quoted MSP formulation should be identified as a supplied paraphrase unless checked against the blog itself. Wu et al., Section 7.3, does explicitly identify \(L\to\infty\) as open. It does **not** single out \(L\propto n\); that is a reasonable specialization by this draft.

## Theorem 2.1: adaptive randomized query lower bound

**Verdict: CORRECT WITH A FIXABLE GAP.**

The lower bound handles adaptive randomized algorithms correctly. Condition on the algorithm’s random tape. At each step, its next query depends only on previously revealed signs. Unrevealed signs remain independent fair bits; adaptivity does not disclose information about them.

There is one error in the stated posterior identity: hitting \(J_j\) need not reveal \(s_j\), because \(g_j\) may vanish at the queried point. Consequently, with the current definition of \(R\),
\[
E[\mu\mid\mathcal T]=\sum_{j\in R}s_ja_j
\]
need not hold.

**Exact fix:** define \(R\) to contain indices queried at a point where \(g_j\ne0\). Then the displayed conditional mean and variance are correct. Alternatively, retain the existing \(R\) and use only the conditional-variance lower bound.

The Monte Carlo calculation is correct:
\[
E_s\operatorname{Var}_x f_s=\sum_j(b_j-a_j^2).
\]

Specify that the risk averages over both the signs and algorithmic randomness.

## Corollary 2.2: realizations, constants, and embedding

**Verdict: CORRECT WITH A FIXABLE GAP.**

The four-ReLU trapezoid formula is correct when each ramp width satisfies
\[
0<w\le(\beta-\alpha)/2.
\]

For uniform input, write \(\ell=1/(2N)\). Then
\[
a=\ell-w,\qquad b=\ell-\frac43w.
\]
The optimal **Bayes** risk under independent uniform signs is \(Na^2\), attained by revealing \(N\) signs and assigning posterior mean zero to the rest. Therefore
\[
\frac{\mathrm{MSE}_{MC}}{\mathrm{BayesRisk}}
=\frac{2(b-a^2)}{Na^2}
\longrightarrow 4\left(1-\frac1{2N}\right).
\]
The limiting constant is correct.

The literal factor-four statement needs its limiting qualification. It is false for arbitrary finite ramps: with triangular bumps, \(w=\ell/2\), the ratio is
\[
\frac{16}{3}-\frac2N,
\]
which exceeds four for \(N\ge2\).

For the Gaussian bound, let
\[
p=\Pr(|Z|\le1)=0.6826894921.
\]
Define \(w'\) explicitly as an upper bound on each ramp’s width divided by its interval’s length. A conservative estimate gives
\[
a_j\ge \frac p m(1-2e^{1/2}w'),\qquad b_j\le\frac p m.
\]
Since \(2e^{1/2}<3.3\) and \(4/p=5.859179094<5.9\), the stated bound follows for \(3.3w'<1\). Thus **5.9 and 3.3 are valid conservative constants** under this definition.

The width/depth claim is plausible but not proved as stated. A signed running sum cannot simply occupy one ordinary ReLU unit. A concrete repair is to carry the input and accumulator as positive/negative pairs, leaving \(n-4\) units for new hinge terms. This gives a straightforward bound of order
\[
L\le 1+\left\lceil\frac{4m}{n-4}\right\rceil,\qquad n\ge5,
\]
with explicit scheduling. Replace “roughly” by a construction and a stated sufficient bound.

The homogenization is valid, with a continuous extension at \(x_2=0\). The latest edit correctly warns that its Gaussian constants require recomputation. However, Theorem 2.1 does not apply “verbatim”: the homogenized bumps are unbounded and have multidimensional supports. Its proof extends immediately to disjoint supports with finite second moments; state that extension.

## Theorem 2.3: deterministic black-box failure

**Verdict: CORRECT, with the revised independent-coordinate construction.**

The indistinguishability argument is sound: the algorithm follows the same adaptive path on the baseline function and on the function augmented by bumps outside all queried intervals.

In the latest version, explicitly require the auxiliary coordinate to be independent of the bump coordinate. Then
\[
\operatorname{Var}(\psi+g)=\tfrac12+\operatorname{Var}(g)\le\tfrac34,
\]
so the weaker claimed bound \(\operatorname{Var}\le1\) is valid. Mean separation \(1/2-o(1)\) gives worst-case squared error at least \(1/16-o(1)\), and the claimed \(N/16\) loss follows.

The original version’s unnormalized lower bound and \(N/8\) comparison were also valid, but its zero function made normalized error undefined. Also, “errors differ” needed to mean **signed errors**, not absolute errors. The revision avoids both problems.

## Section 2 remarks

**Verdict: WRONG as written.**

The inference from recovery of a **generic** network to the absence of worst-case lower bounds beyond parameter count is invalid.

A counterexample uses only **four hidden units per network**. Partition \([0,1]\) into \(m=2N\) candidate intervals and let the family contain
\[
f_{j,s}=s\,g_j,\qquad j\in[m],\ s\in\{-1,1\}.
\]
Choose \(j\) and \(s\) uniformly. Each network contains one trapezoid. A query can discover or eliminate at most one candidate interval. After \(N\) queries, the probability of discovering the active interval is at most \(1/2\).

For nearly rectangular bumps of common mass \(a\approx1/m\), optimal Bayes risk is
\[
(1-N/m)a^2,
\]
whereas Monte Carlo risk is \((b-a^2)/N\). Their ratio again tends to
\[
4-\frac2N.
\]
Thus a constant-factor sampling lower bound persists for arbitrarily large \(N\) with a constant number of network parameters. Weight precision and magnitudes grow with \(N\), which parameter count alone does not capture.

There is also an arithmetic inconsistency: the displayed condition \((n-3)L\ge8m\), together with \(m=2N\), yields \(N\le(n-3)L/16\), not `/8`.

The sample-model implication is correct. The quantum remark needs the usual coherent-oracle and input-state-preparation qualifications. None of these worst-case constructions establishes optimality on the iid He-Gaussian network distribution.

## Proposition 3.1: no coarse randomized hardness

**Verdict: CORRECT WITH A FIXABLE GAP.**

The variance and Hoeffding bounds are correct; write “MSE **at most** \((b-a)^2/(4N)\).”

The polynomial-time conclusion requires efficiently samplable inputs and polynomial-time approximate evaluation with controlled error. Those conditions can be supplied for the stated rational ReLU setting, but they should be explicit.

The conclusion concerns randomized estimation at inverse-polynomial **absolute** precision. It does not automatically cover relative precision, exponentially small tolerances, or sampling from an inaccessible input distribution.

## Proposition 3.2: Lévy distance and planting

**Verdict: CORRECT WITH A FIXABLE GAP.**

The latest statement’s bound
\[
d_L\le2\delta+\eta+\mathrm{adv}
\]
is correct. The original \(\delta+\eta+\mathrm{adv}\) was not established by its proof.

The two estimation margins produce a horizontal shift of \(2\delta\). The two one-sided failure probabilities contribute total vertical slack \(\eta\), giving
\[
F_1(x-2\delta)\le F_0(x)+\eta+\mathrm{adv}
\]
and its counterpart with indices exchanged. In fact, the proof gives the slightly sharper bound
\[
d_L\le\max(2\delta,\eta+\mathrm{adv}).
\]

To retain a horizontal shift \(\delta\), run the test with estimation margin \(\delta/2\), increasing the sample count by approximately four.

The consequence is overclaimed. Closeness of the **marginal distributions of means** does not show that computing the instance-specific mean is easy, or that planting cannot affect fine-grained estimation difficulty. Replace the last sentence with a statement limited to detectable changes in the distribution of the mean.

The Goldwasser et al. attribution was not independently verified from the supplied local papers.

## Corollary 3.3: predictable PRF means

**Verdict: CORRECT WITH A FIXABLE GAP.**

The cryptographic argument is correct under standard PRF security, with \(m\) serving as a security parameter and uniformly distributed input bits.

If a non-negligible fraction of keys have bias exceeding \(1/q(m)\), polynomially many oracle samples distinguish those functions from a random function. Choose the sampling failure probability sufficiently smaller than that non-negligible fraction. The random-function concentration bound is correct.

For each fixed polynomial sampling budget, the constant estimate \(1/2\) consequently outperforms Monte Carlo except on a negligible fraction of keys.

The ReLU realization needs correction: a nonconstant
\[
F_k(\operatorname{sign}(x))
\]
is discontinuous and cannot be represented exactly by a finite ordinary ReLU network. Exact threshold-gate simulation on Boolean inputs does not fix this.

**Exact fix:** first state the result for Boolean input. For Gaussian input, introduce clipped sign approximations and bound the transition-strip error, as in Theorem 4.1. Choose the strip width small enough for the intended negligible-error claim.

## Proposition 3.4: annealed-estimator algebra

**Verdict: CORRECT WITH A FIXABLE GAP.**

Under finite second moments, all displayed identities and the inequality comparing the annealed estimate with Monte Carlo are correct.

Make “average correlation” precise by defining, when the denominator is positive,
\[
\rho=
\frac{E_{x,x'}K(x,x')}
{E_xK(x,x)+\operatorname{Var}_x m(x)}.
\]
Then the annealed estimate wins exactly when \((N+1)\rho<1\).

The He-ReLU consequence is too broad. For a single output
\[
f_w(x)=\operatorname{ReLU}(w^\top x),\qquad
w\sim N(0,2I_n/n),
\]
the mean is \(\|w\|/\sqrt{2\pi}\), and
\[
\operatorname{Var}_w\mu\sim\frac1{2\pi n},
\qquad
E_w\operatorname{Var}_x f_w=1-\frac1\pi.
\]
The annealed estimate therefore matches order-\(n\) samples, not merely a few.

Restrict the consequence to the specific observable and depth regime actually established. A covariance statement about squared output scale does not automatically establish the corresponding claim for an arbitrary scalar readout.

## Proposition 3.5: fixing randomness and rotations

**Verdict: WRONG as stated; the averaging assertion is correct.**

Fixing the random tape proves the finite-instance existence claim. The rotation hypothesis, however, specifies only marginal invariance of \(W_1\), while the proof requires invariance of the **entire network law**.

Counterexample: let \(g\sim N(0,2)\), \(W_1=g\), \(W_2=\operatorname{sign}(g)\), and
\[
f_g(x)=\operatorname{ReLU}\!\left(
\operatorname{sign}(g)\operatorname{ReLU}(gx)\right).
\]
The first-layer law is rotation-invariant in dimension one. But \(f_g(x)=g\operatorname{ReLU}(x)\) for \(g>0\), and zero otherwise. The one-point designs \(x=1\) and \(x=-1\) have different average errors.

**Exact fix:**
\[
(W_1Q,W_2,\ldots,v,b)\stackrel d=(W_1,W_2,\ldots,v,b).
\]
Independent isotropic first-layer weights satisfy this condition.

The consequence also confuses nonuniform existence with uniform computation. A good fixed tape or design for every size need not be efficiently constructible, uniformly describable, or exactly representable with finitely many bits. State the nonuniform existence result separately. Chen–Tell explicitly distinguishes that elementary existence fact from its conditional uniform derandomization theorem.

## Theorem 4.1(a): biased circuit embedding

**Verdict: CORRECT WITH A FIXABLE GAP.**

The binarization and gate formulas are correct. Both invariants hold, and
\[
\Pr(B)\le\frac{m}{K\sqrt{2\pi}}
\]
gives the stated expectation error.

Representing an OR wire by a difference of two unit outputs is legitimate: consumers can incorporate that difference into their affine preactivation.

Depth is at most \(d+1\) when the final affine readout does not count as a ReLU layer. There are at most \(2m+2s\) ReLU units and \(O(m+s)\) edges. The layered width bound is valid when \(w\) includes the necessary carried wires.

One small weight-alphabet issue remains: repeated inputs, such as AND\((a,a)\), can produce a coefficient \(2\) when parallel contributions are combined. Simplify idempotent gates first, permit parallel edges, or enlarge the alphabet.

For complexity applications, state rational \(K\) and its encoding length.

## Theorem 4.1(b): bias-free embedding

**Verdict: CORRECT.**

The homogeneity and interval invariants hold. The extra Gaussian coordinate is independent of all circuit-input sign bits.

The required estimate is the weighted bad-event bound:
\[
E[r1_{B^\circ}]
\le \frac{m\phi(0)}K E[r^2]
=\frac{m}{K\sqrt{2\pi}}.
\]
Dividing by \(Er=\sqrt{2/\pi}\) gives \(m/(2K)\).

Depth \(d+3\) is a safe upper bound with explicit preparation and carrying of \(r\) and signed input coordinates. A DAG implementation can save a preparation layer. Size remains \(O(m+s)\); carrying \(r\) adds only \(O(d)\) units.

Write “depth at most” to avoid implying minimality.

## Theorem 4.1(c): noisy embedding

**Verdict: CORRECT WITH A FIXABLE GAP in the revised absolute-noise version; the relative-RMS extension is WRONG.**

The original \(4\tau\) accounting failed. Each restored wire is a difference of two noisy units, so its error can be \(2\tau\). An OR gate can therefore incur
\[
2\tau+2\tau+2\tau=6\tau.
\]
For \(\tau=1/16\), two logical-zero input wires can each equal \(2\tau\); adding gate noise \(2\tau\) gives \(v=3/8\), and restoration returns \(1/4\), not zero.

The current repair is correct:
\[
\tau=\frac1{24},\qquad
\Pr(\text{some noise exceeds }\tau)
\le2Ue^{-1/(1152\sigma^2)}.
\]
AND and NOT errors are bounded by \(5\tau\) and \(3\tau\). Restore the binarization outputs as well as gate outputs. The resulting depth is at most \(2(d+1)\), with \(O(m+s)\) units and edges.

Two statements still need repair:

1. Individual restore-unit outputs are not bounded by one. Their **difference before its own additive noises** is bounded in \([0,1]\), globally.
2. Relative noise based on physical layer RMS is not bounded by the stated absolute noise. Binarization units such as \(\operatorname{ReLU}(Kx)\) have RMS of order \(K\), despite their difference being bounded.

Delete the relative-RMS parenthesis unless a separate noise model and proof are supplied.

For the expectation argument, write the final output globally as a bounded restored value plus its final zero-mean noise difference. Take that noise’s unconditional expectation; do not rely on its mean after conditioning on a success event.

The numerical code implements the differences and restoration described. Its six examples support implementation sanity, not the uniform theorem. At \(\sigma=0.02\), the displayed union-bound guarantee is already vacuous for these circuit sizes.

## Corollary 4.2: IKW barrier

**Verdict: CORRECT WITH A FIXABLE GAP.**

The deterministic implication is correct. The revised choice \(K=12(m+s)\) resolves the input-count issue and gives reduction error below \(1/12\). Combining it with estimator error \(1/12\) yields CAPP error below \(1/6\).

IKW Section 5.1 requires:

- **For every** \(\epsilon>0\), a nondeterministic algorithm of time \(2^{n^\epsilon}\).
- Advice length at most \(n^\epsilon\).
- Infinitely many common good sizes, on each of which the guarantee holds for **every** circuit of that size.
- At least one accepting computation per input, with **every accepting computation** outputting a valid approximation.

The initial draft’s unspecified \(n^\epsilon\) advice was insufficiently quantified. The current \(E^{o(1)}\) formulation is sufficient with appropriate padding and size bookkeeping.

Add the accepting-branch semantics and explain padding reduced networks to common good sizes. Distinguish circuit encoding size from gate count and network edge count; logarithmic encoding factors do not obstruct the \(o(1)\) exponents.

The attribution to Theorem 32 is correct. Remark 33 explicitly gives the deterministic polynomial-time CAPP equivalence with promise-BPP versus promise-P; Corollary 34 supplies the subexponential implication. These sources do not establish a deterministic equivalence merely from \( \mathrm{NEXP}\not\subseteq\mathrm{P/poly}\).

## Corollary 4.3: Chen–Tell adaptation

**Verdict: CORRECT WITH A FIXABLE GAP.**

The original constant-precision hardness claim for sparse ReLU DAGs follows from the proposed adaptation. **#NSETH suffices.** The latest positive-variance strengthening needs a padding correction.

For the underlying reduction:

1. \(P_2(x)\) and the guessed polynomial \(H(x)\) can be evaluated with \(n\,\mathrm{polylog}\,n\) field operations. The sum defining \(Q(x)\) has the same bound.
2. With \(p\in(n^4,2n^4]\), field elements have \(O(\log n)\) bits, giving a Boolean circuit of size \(n\,\mathrm{polylog}\,n\).
3. A fixed number \(b=O(\log n)\) of uniform bits suffices: map a uniform \(b\)-bit integer modulo \(p\), choosing \(b\) so that \(p/2^b\le n^{-10}\). This avoids an unbounded rejection sampler.
4. Completeness is one. Soundness is at most \(\deg(H-Q)/p+n^{-10}<1/3\).
5. An \(O(E^{2-\epsilon})\) deterministic estimator therefore verifies the guessed identity in \(n^{2-\epsilon'}\) time for some \(\epsilon'>0\).

**An essential missing implementation detail:** evaluating \(\sum_{i=1}^nH(i)\) by \(n\) separate Horner evaluations would take near-quadratic time and defeat the argument. Use fast multipoint evaluation, in \(\widetilde O(n)\) time for the stated degree, then sum modulo \(p\). The true count is at most \(n^2<p\), so its residue determines the integer count.

Chen–Tell’s Lemma 6.3 is headed and phrased using NSETH, while the surrounding section and Theorem 6.4 claim #NSETH. Do not silently attribute the stronger counting statement to that lemma. Supply the counting reduction: split SAT variables into two groups, retain partial-assignment multiplicities, and count orthogonal pairs. For fixed clause width, the number of clauses is polynomial in the variable count, so dimension \(\mathrm{polylog}\,n\) suffices. The construction preserves the count, establishing the needed #NSETH consequence. “Unambiguous” means correct answer or \(\bot\) on every branch, with some correct branch, not a unique accepting branch.

**New variance gap.** The revised padding gives a Boolean acceptance probability in \([1/4,1/2]\), hence Boolean variance at least \(3/16\). The Gaussian ReLU approximation need not retain that exact lower bound. Smoothing an AND of two fair bits already decreases its variance below \(3/16\) for sufficiently large finite \(K\).

Two repairs are available:

- Weaken the variance promise to \(1/8\), with an explicit moment-error estimate.
- Preserve \(3/16\) by padding to acceptance probability \(3/8+\Pr[C=1]/4\), creating enough variance margin before smoothing.

DAG structure matters to the proof of near-linear reduction size. Extension to MLPs needs separate accounting for padding and carried wires.

Finally, “Monte Carlo takes \(O(E\log(1/\delta))\)” is an arithmetic-cost comparison. In the declared bit model, use evaluation bit-cost, or a suitably qualified \(\widetilde O(E\log(1/\delta))\) bound for this restricted construction. State the conditional conclusion as excluding every fixed polynomial improvement over quadratic worst-case time, rather than as a pointwise lower bound for every size.

## Proposition 4.4: implications for MSP

**Verdict: CORRECT WITH A FIXABLE GAP.**

The main non-refutation conclusion is correct. At fixed precision, instance-specific advice can contain the answer when the hard circuit is placed entirely in \(\theta\). For a Boolean circuit, its exact acceptance count also has a short description.

Three qualifications are necessary:

- Storing one answer does not handle an arbitrary context-dependent function of \(c\).
- Proposition 3.5 supplies nonuniform existence, not an unconditional uniform deterministic estimator.
- The train-and-explain formulation quoted in Section 1.5 does not specify whether guarantees average over trainer seeds or hold for each seed. “Likewise an average-case statement” requires that quantifier to be checked.

Chen–Tell’s average-case result is conditional, uniform, and concerns samplable finite input distributions. It is relevant evidence about derandomization, not a direct theorem establishing the full Gaussian MSP.

## Section 5 oracle model

**Verdict: CORRECT WITH A FIXABLE GAP.**

Specify that algorithms’ initial information is independent of the random oracle unless explicitly charged as preprocessing. Otherwise the posterior arguments can fail before the first query.

Use injective domain separation for \((j,v)\). State whether budget is a hard bound or an expected cost, especially for part (c). Oracle/sign operations also extend the ordinary ReLU model and should remain explicitly identified as such.

## Theorem 5.1(a): single-scale posterior variance

**Verdict: CORRECT.**

With \(M=2^m\) relevant oracle values and \(q\) distinct revealed values,
\[
\operatorname{Var}(\mu\mid\mathcal T)=\frac{M-q}{M^2}.
\]
Adaptive randomized queries cannot improve this beyond the number of distinct relevant values revealed.

Also,
\[
E_H\operatorname{Var}_x f=1-\frac1M.
\]
At \(Q=N=M/2\), the exact ratio is
\[
4\left(1-\frac1M\right),
\]
consistent with the stated asymptotic bound.

## Theorem 5.1(b): multiscale allocation and advantage

**Verdict: CORRECT WITH A FIXABLE GAP.**

The posterior-risk optimization, greedy allocation, and variance calculation are correct:
\[
E_H\operatorname{Var}_x f
=J-\sum_{j=1}^J2^{-j}
=J-1+2^{-J}.
\]

For \(S_k=2^{k+1}-2\le Q<S_{k+1}\), the exact optimal risk is
\[
R(Q)=2^{-k}-2^{-J}-(Q-S_k)4^{-(k+1)}.
\]
This verifies
\[
R(Q)\ge\frac1{Q+2}-2^{-J}.
\]
The \(4J^2\) upper bound on advantage and the \(J(J-1)/4\) coarse-enumeration advantage follow on the stated interior range.

The latest edit correctly requires at least \(J\) calls for the Monte Carlo comparison. But the subsequent claim of a uniform window \([1,B]\) remains inconsistent with that repair. Below \(J\), ordinary Monte Carlo cannot make one full pass.

**Exact fix:** state a feasible window beginning at \(J\), and either require \(J\mid Q\) for the exact Monte Carlo formula or use \(\lfloor Q/J\rfloor\), adjusting constants. Alternatively, explicitly call \(J(J-1+2^{-J})/Q\) an idealized work-normalized baseline.

## Theorem 5.1(c): general multiscale construction

**Verdict: CORRECT WITH A FIXABLE GAP.**

The current additions of uniform cells conditional on inclusion and nonnegative weights repair two missing hypotheses.

Also require \(0<\pi_j\le1\) for nonzero weights and oracle-independent sampling/inclusion rules. Most importantly, define \(\operatorname{Adv}\) here as the advantage of the **optimal estimator**, or of the specified enumeration strategy. The inequality is false for an arbitrary poorly chosen estimator family.

Under those conventions, the proof works. Let \(c=\sum_j\pi_j\). Then
\[
V=\sum_jw_j^2(1/\pi_j-2^{-j})
\ge\frac12\sum_j\frac{w_j^2}{\pi_j},
\]
and
\[
cV\ge\frac12\left(\sum_j|w_j|\right)^2.
\]
Enumeration yields
\[
\sum_{k=1}^J2^kR(2^k)\le2\sum_jw_j^2.
\]
Consequently,
\[
\sum_{k=1}^J\frac1{\operatorname{Adv}(2^k)}
\le
\frac{4\sum_jw_j^2}{(\sum_j|w_j|)^2}.
\]
The \(J/4\) and equal-weight \(J^2/4\) conclusions follow.

As in part (b), distinguish the continuous expected-work baseline from a feasible hard-budget Monte Carlo algorithm. If small dyadic budgets are removed, the pigeonhole count must use the remaining budgets.

## Section 5 interpretation

**Verdict: WRONG as a general claim about oracle models.**

The proved bounds concern specified families and budget windows. They do not show that exhaustive deduction has only polylogarithmic advantage in arbitrary oracle models.

Even part (a)’s family gives infinite advantage over ordinary Monte Carlo after all \(2^m\) values are enumerated, because enumeration then has zero error. Other budget regimes also have different behavior.

Likewise, “mechanistic and sampling cannot be separated at all” is not a theorem without a formal definition of mechanistic computation. Restrict the interpretation to the proved estimation-risk statements.

## SH(F) instantiation

**Verdict: WRONG.**

The stated single-scale assumption does not justify the claimed multiscale conclusion.

SH(F) bounds the difficulty of estimating each
\[
\mu_j(k)=2^{-j}\sum_vF_k(j,v)
\]
separately. The target in part (b) is \(\sum_j\mu_j(k)\). Individual lower bounds do not provide a direct-sum lower bound: cross-scale correlations, cancellation, shared work, or a special algorithm for the sum are not excluded.

The oracle proof’s key identity is a **joint posterior-variance decomposition into independent unrevealed values**. SH(F) contains no replacement for that identity. This is the failing step.

**Required replacement:** formulate a joint summation-hardness assumption directly lower-bounding estimation risk for the multiscale sum by the allocation expression, with an explicit time accounting. Label the resulting theorem conditional on that stronger assumption.

Additional fixes:

- Specify \(F_k\in\{-1,1\}\), or center/rescale Boolean outputs. The constants currently use Rademacher outputs. For \(\{0,1\}\)-valued functions, the stated \(j=1,t=0\) lower bound already conflicts with the universal midpoint estimate.
- Transfer the Monte Carlo variance calculation separately; single-scale SH does not establish the required joint second moments.
- Explain why \(t/t_F\) is the appropriate work measure, including shared evaluation work.
- Use Boolean input or a quantified Gaussian smoothing construction.

The warning that ordinary PRF security does not imply SH(F) is correct.

## Proposition 5.2: advice and contexts

**Verdict: WRONG; the revised preprocessing accounting repairs part of item 2.**

Item 1 is false. Independence from the **realized context** does not imply independence from its answer.

For a finite context space, let advice contain the table
\[
\pi=(\mu_H(c):c\in\mathcal C).
\]
This advice is computed before \(c\) is drawn and is independent of the random choice of \(c\). Nevertheless, after receiving \(c\), the estimator reads its exact answer without making oracle calls. The assertion permits advice “of any length,” so this is a direct counterexample.

**Exact fix:** require the advice and other initial information to be independent of the entire relevant oracle, or bound and charge all oracle preprocessing. ARC’s prohibition on dependence on the realized \(c\), by itself, does not supply that stronger condition.

The latest item 2 correctly changes the budget to
\[
Q+t_T+t_E.
\]
State that the seed initially contains no oracle information and include information carried by \(\theta\), not merely by \(\pi\). Adaptive context selection must also be covered by the charged transcript.

For multiscale means in \([-J,J]\), encoding to absolute accuracy \(\epsilon\) needs \(O(\log J+\log(1/\epsilon))\) bits; the displayed \(\log(1/\epsilon)+O(1)\) applies only to a uniformly bounded range.

The final claims about all planted or backdoored networks do not follow. Knowing a planting mechanism does not necessarily reveal its mean, and Proposition 3.2 constrains a marginal distribution rather than instance-specific computational difficulty.

## Required fixes, ordered by severity

1. **Replace SH(F)’s invalid direct-sum inference** with an explicit joint assumption. Keep the resulting instantiation conditional.
2. **Replace Proposition 5.2(1)** with an oracle-independence or charged-preprocessing statement. Context-independent advice can encode every context’s answer.
3. **Separate uniform algorithms from nonuniform fixed designs**, and strengthen Proposition 3.5’s rotation hypothesis to joint network-law invariance.
4. **Repair Corollary 4.3’s new variance promise.** Gaussian smoothing does not preserve the exact \(3/16\) endpoint. Add fast multipoint evaluation and the explicit counting reduction.
5. **Delete the relative-layer-RMS extension in Theorem 4.1(c).** Retain the repaired \(6\tau\), \(1/24\), and \(1152\) accounting; correct the claim about bounded individual units.
6. **Remove the parameter-count ceiling on black-box hardness.** Generic recovery does not establish it; the four-unit counterexample directly refutes it.
7. **Narrow the planting, He-ReLU, and oracle-model interpretations** to the actual proved distributional, observable-specific, and budget-specific statements.
8. **Complete the IKW quantifiers and nondeterministic semantics**, including common good sizes and padding.
9. **Resolve the sampling-budget conventions** in Sections 1 and 5, and define which estimator achieves part (c)’s advantage.
10. **Finish the local proof repairs:** revealed-sign definition in 2.1, finite-ramp qualification and explicit embedding in 2.2, Gaussian PRF smoothing, noisy variance definition, and arithmetic-versus-bit cost accounting.

Validation used the supplied paper texts, numerical source, and existing log. A sequential Python check under `nice -n 19` with `OMP_NUM_THREADS=1` verified 16,365 finite allocations and the reported constants in approximately **0.12 CPU-seconds**. The large Monte Carlo experiment was not rerun.
