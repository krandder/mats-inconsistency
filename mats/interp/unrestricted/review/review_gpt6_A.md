**Verdict: major revision.** Several core identities are sound, but the report overstates derandomisation, numerical evidence, and the spin-glass analogy. Required changes below are prioritised. Checked read-only on 2026-10-31; no files modified.

1. **Define the theory’s risk separately from the experimental metric (§1, §§3–5).**  
   The proofs concern \(R(A)=E_W\|A-\mu\|^2\), whereas §1 defines per-instance nMSE. In general,
   \[
   E_W[\|A-\mu\|^2/V(W)]\ne R(A)/E_WV(W).
   \]
   The Hermite bounds and sample-equivalent numbers establish the latter comparison. Import FORMAL’s definitions of \(R,V,V_{\rm sph},S,m_n\); specify the convention for \(V(W)=0\). Plain MC scores \(1/N\) **in expectation**, not on every realization.

2. **The arbitrary-estimator simulation is valid only under the stronger computational convention in FORMAL (§3.1, fourth bullet).**  
   The distributional argument works: conditional on the first-layer Gram matrix, multiplying by independent Haar randomness restores the Gaussian law, preserves \(\mu\), and leaves independently extracted \(R\) independent of \(W'\). But generating independent real-valued coordinates by splitting infinitely many digits is not ordinary finite-cost arithmetic or inverse-CDF evaluation. Label this **PROVEN UNDER digit-splitting and exact-transform primitives**; justify randomness-access costs separately from the \(O(n^3)\) rotation. For finite-entropy quantized weights, the proposed exact reconstruction is impossible: \(H(W',R)=H(W)+H(R)>H(W)\). “Determinism excludes nothing” must retain the model qualification.

3. **The threshold-circuit seed construction is incomplete (§3.1).**  
   A uniform permutation supplies a uniform element of \(S_n\), not automatically \(\log_2(n!)\) independent fair bits. Signs plus a free permutation have \(2^n n!\) equiprobable states; exact extraction of \(2n\) fair bits requires \(2^{2n}\mid2^n n!\), which never holds because \(v_2(n!)<n\). At \(n=2\), there are eight states, versus sixteen required seed pairs. Discrete weights also introduce sorting ties; symmetric laws can have mass at zero. Specify \(\{\pm1\}\) inputs, nonzero weights, and supply a different extractor/construction. The current **PROVEN** label is undeserved.

4. **“Any weight law with a density” is false as a risk-convergence claim (§3.1).**  
   Riemann–Lebesgue gives joint weak convergence; it does not supply uniform integrability of squared errors. Astra’s argument establishes bounded wrapped densities and moments for **Gaussian** weights. Even bounded weights can have a joint density concentrated near \(w_1+w_3=2-t\), proportional to \(1/[t\log^2(1/t)]\), making the constructed Gaussian sample’s second moment infinite. Restrict to the Gaussian ensemble or state sufficient integrability assumptions. The noisy extension needs the corresponding argument too.

5. **Correct the claimed mean-field agreement and “mean Hermite degrees” (§3.5).**  
   Re-running [nstab_mf_compare.py](/home/agent/workspaces/mats/work/unrestricted/scratch/nstab_mf_compare.py) reproduces the log. Independent quadrature gives these relative discrepancies after normalizing by measured \(D(1)\):

   | \(n=32\) | \(\delta\) | Discrepancy from mean field |
   |---|---:|---:|
   | \(L=16\) | \(10^{-2}\) | −7.4% |
   | \(L=32\) | \(10^{-5}\) | +28.7% |
   | \(L=32\) | \(10^{-4}\) | +24.8% |
   | \(L=32\) | \(10^{-2}\) | −20.7% |

   The 5% claim holds for \(L=4\), not the stated deeper range. At \(L=32\), \(D(10^{-5})/10^{-5}=27.25\), or **17.37** after normalization. The quoted 14 comes approximately from the normalized secant at \(10^{-3}\), not the derivative at zero. Call these finite-difference proxies and identify the normalization.

6. **The polynomial numbers reproduce, but are estimates rather than established class ceilings (§3.5).**

   | \(n=32\) | Degree | Registered calculation | Post-hoc \(D(1)\)-normalized calculation |
   |---|---:|---:|---:|
   | \(L=16\) | 4 | 2.340 | 6.501 |
   | \(L=32\) | 4 | 1.200 | 3.870 |
   | \(L=16\) | 8 | 14.538 | No positive tail bound |
   | \(L=32\) | 8 | 5.679 | 16.910 |

   The normalized calculation replaces \(\widehat S_1\) by \(\widehat S_1\widehat D(1)\), retaining the measured squared differences. State that explicitly and label it post hoc. These compare against **plain MC**, since the code uses \(EV\).

   The deeper \(n=32\) observations have reported relative standard errors around 20–60%; denominator uncertainty is additional. Common-scramble squared differences include integration error, and minimizing noisy estimates requires simultaneous uncertainty control. Replace “worth at most” with “plug-in estimate of the ceiling,” or provide valid confidence bounds. Astra already flagged these issues.

7. **The field-adjusted spin-glass check also fails at depth 9 (§4.3).**  
   I reproduced the recorded scans and Astra’s interval refutation. Extending the same outward-rounded calculation to \(f=\kappa^{\circ9}\), \(q=1/2\), gives
   \[
   f'(q)-qf''(q)\approx-0.00221067927253818,\qquad
   ((f'')^{-1/2})''(q)\approx0.0423629872843181>0.
   \]
   Since the matching function decreases from a positive value, \(\hat q<1/2\); the violation lies inside the field-adjusted interval. Numerically \(\hat q\approx0.492633655\). Thus depth 10 is a sampling omission, not the first failure. Also replace global “concave for \(L\le8\)” with “no violation on the tested grids” unless a global proof is supplied. P12 is correctly refuted.

8. **No spin-glass obstruction to actual network optimization has been established (§4.3).**  
   Failure of a sufficient concavity condition does not, by itself, prove an overlap gap. The final-ReLU field is non-Gaussian, smoothness assumptions fail, and the applicable variational theorem remains unchecked. These caveats undermine “its failure would mean an overlap gap” and the verdict “precise, testable obstruction.” Use **a testable surrogate-kernel observation**. The claim that \(10^{-12}\) tails lie “far below” the relevant threshold also needs compatible normalization and an evaluated OPT/algorithmic threshold; \(\sqrt n\,\mathrm{OPT}\) alone does not establish it.

9. **Narrow the barrier statements (§3.3).**  
   “No current technique proves super-linear time lower bounds for explicit problems in general models” is false literally; time-hierarchy constructions already defeat that formulation. Say that no applicable technique is supplied for this problem. Derive the claimed consequence using an explicit accuracy schedule and cost model.

   Random self-reducibility does **not** generally require polynomial estimands; discrete logarithm is a counterexample. Only the cited polynomial-interpolation route has that requirement, possibly after reparameterization. The nonpolynomiality of \(\mu\) is correct but excludes less than claimed. Finally, the cited IKW corollary assumes \(2^{E^{o(1)}}\) time; generic “subexponential” \(2^{o(E)}\) is too broad.

10. **Repair the hardness/precision comparison (§3.4).**  
    “In each, sampling fails” contradicts the cited randomized volume algorithms. Volume oracle bounds are worst-case information bounds, not an identified average-case distributional hardness result. Separate exact random-circuit/permanent hardness from approximate average-case conjectures; the bare \(2^{-n}\) scale is not a sufficient theorem statement, and Gaussian permanents need normalization.

    The free-energy conversion loses a factor proportional to the coefficient of variation; it is exponential only when that coefficient is exponentially large. The scalar identity \((\log Z)'(0)=\mu\) and fixed-instance zero-tail comparison are correct. Across nets, the relevant comparison is
    \[
    E[p^2]\quad\text{versus}\quad E[p(1-p)]/N.
    \]
    “Usually tiny” probabilities do not suffice. Rare events are not automatically low-temperature glassy events.

11. **The threshold analogy is useful, but its verdict exceeds the evidence (§4.1).**  
    The gate identity requires integer-valued \(S-\theta\); account for biases or cite the homogeneous lifting, input units, wires, and weight encoding. Constant depth is preserved up to overhead, not necessarily “the same depth.”

    The \(2^k\eta\) union bound is valid; its failure to give an efficient algorithm does not prove that all positive theory stops at depth two. A PRG’s upper runtime bound likewise is not a lower bound against sampling. State the correlation inequality for \(0<c<1\) in the Gaussian mean-field setting.

    The cited planting proposition only bounds changes in the **distribution** of the mean at specified accuracy and computational scales. It explicitly does not exclude instance-specific difficulty or fine-grained reductions. Replace “not plausible” with “no suitable transfer established.”

12. **Keep the volume correspondence; weaken the claimed consequences (§4.2).**  
    The cone-first-moment decomposition is correct, and the oracle comparison is informative with its worst-case scope. However, known deterministic polytope algorithms do not universally “need correlation decay”; that describes particular approaches. Failure of full scale contraction does not exclude methods that handle scale analytically.

    Promise-BPP = promise-P derandomizes suitable **explicit finite-input** approximation problems, not arbitrary membership-oracle access. The cited #NSETH result for ReLU means is not already a lower bound for volume of one explicit cone. Exact average-case derandomisation also does not rule out hardness shared by randomized and deterministic estimators. Astra’s actual assessment is “no reduction supplied,” weaker than the report’s claimed agreement.

13. **Restore qualifications for the remaining restricted-class statements (§3.5).**  
    Standard SQ queries are scalar and bounded: the vector mean requires coordinate queries, and ReLU outputs require truncation. The substantive objection to unrestricted expectation oracles survives.

    The GP cubature rate is conditional on applicable spectral/cubature hypotheses, not merely unverified constants; handle parity degeneracies, including the missing higher odd degrees at one layer. State the infinite-width-before-depth limit explicitly.

    Partition rates are **upper bounds**, not matching rates or limitations on all partition methods. The decorrelation assumption is unproved for this algorithm. Cone enumeration’s quoted \(\varepsilon\) crossover suppresses enumeration, integration, and normalization costs and should be **HEURISTIC**. Hanin–Rolnick’s cited abstract does not verify the displayed bias-free \(n-1\) exponent.

14. **Fix the independence provenance (§3.5 and the tag legend).**  
    [PROVENANCE.md](/home/agent/workspaces/mats/work/unrestricted/coprover/PROVENANCE.md) explicitly says Claude wrote the degree-\(D\) projection bound after seeing Astra’s progress statement. Therefore **[C+A] = “both independently”** is not justified for that bound. Credit Claude’s prior stability idea, Astra’s projection bound, and joint checking separately.

    The principal remaining attributions agree with the artifacts: rotational extraction, batching, even-tail bound, cell selection, and necessary conditions [A]; oscillating seeds [C]; mean-field proof [A] following [C]’s heuristic; global concavity refutation [A]; field computation [C]. The arbitrary-estimator repair is [C], written after Astra’s objection, not independently validated in Astra’s answer.

15. **C0 identifies the bottleneck correctly, but the ledger is not yet precise (§5).**  
    Specify the actual polynomial budget window, operations and precision, expected versus worst-case cost, preprocessing charges, risk, and dependencies of \(c\). “A stated polynomial window” currently names an omitted parameter. Replace “not implied by any standard hypothesis” with “no implication established here.”

    C1 needs a quantified joint \((n,L,\delta)\) window; linear-regime measurements do not test its \(2/3\) claim. C3 needs the tail cutoff quantified against each budget exponent \(p\). C2 must retain the distinction between covering mass, discarding cells, and downstream certificates. T-relative needs the ensemble probability and success quantifiers already supplied in Astra’s round-one formulation.

16. **Make literature verification levels and the surviving results explicit.**  
    The notes mark bibliography/abstract/snippet checks for Goldreich–Wigderson/Hoza, Hanin–Rolnick, Gamarnik–Kizildag, discrepancy references, Ritter, and selected threshold/structured-volume papers. They do not establish all quoted theorem hypotheses. IKW/Chen–Tell rely on prior-lane verification; spin-glass applicability, most quantitative PRG/cubature claims, and several volume/random-circuit/permanent claims remain memory-based. External rechecking was blocked by DNS failure.

    The following core arguments survive: QR/torus sampling under its stated primitives; Strassen batching; kernel discrepancy and Mehler positivity; positive-design tail bound; Hermite projection and two-scale inequality; mean-field \(2/3\) asymptotic and even-tail limit; stability inequality; selected-cell mass/error bounds; all four §3.6 identities; and the cone representation. Registration hashes for M1, M3, amended PREREG, and frozen v1 match. These results support a revised report, but not its present unrestricted or spin-glass conclusions.
