# Literature check: noisy critical He-ReLU stacks (2026-10-31)

Method: equations below were read in the arXiv LaTeX sources or PDF text, unless a line says otherwise. "Our check" marks things we derived or computed ourselves; they are not in the cited source.
Not verified: the Goldie 1991 PDF (Project Euclid blocks scripted access; the statement was checked in BDM 2016 and in Damek's restatement), the full text of Dufresne 1990, the full text of de Calan et al. 1985, and the Rider–Valkó paper (cited only through Gautié et al.).

## 1. PDLT: finite-width kernel and vertex recursions
- **Citation.** D. A. Roberts and S. Yaida, "based on research in collaboration with Boris Hanin", *The Principles of Deep Learning Theory*, Cambridge University Press 2022, doi:10.1017/9781009023405; arXiv:2106.10165. Hanin is not a co-author. Equation numbers are from arXiv v2 (Aug 2021).
- **Normalisation (§4.3).**
  - Stochastic metric (4.70): `Ĝ^(ℓ+1)_{α1α2} = C_b + C_W (1/n_ℓ) Σ_j σ_{j;α1} σ_{j;α2}`, the conditional covariance of z^(ℓ+1) given layer ℓ. The book calls p(z^(ℓ+1)|z^(ℓ)) a "transition matrix", so the network is an exact Markov chain in the Gram matrix (4.69–4.71).
  - Vertex (4.76): `(1/n_ℓ) V^(ℓ+1)_{(α1α2)(α3α4)} ≡ E[ΔĜ_{α1α2} ΔĜ_{α3α4}]`. Connected four-point function (4.77, 4.117): `(1/n_{ℓ-1}) [δ_{i1i2}δ_{i3i4} V_{(α1α2)(α3α4)} + 2 perms]`.
  - So V is O(1), V/n is the variance of the stochastic metric, and a single neuron's fourth cumulant is 3V/n.
- **General recursions.** (4.118): `K^(ℓ+1)_{α1α2} = C_b + C_W <σ_{α1}σ_{α2}>_{K^(ℓ)}`. (4.119) is the multi-input V recursion; (4.114–4.115) give the K and NLO-metric recursions.
- **Single input, equal widths (5.108–5.110).** `V^(ℓ+1) = χ_∥² V^(ℓ) + C_W² [<σ⁴>_K − <σ²>_K²]` and `G^{1}(ℓ+1) = χ_∥ G^{1}(ℓ) + (1/8) j(K) V/K²`.
- **Scale-invariant class, σ = a₊z (z ≥ 0), a₋z (z < 0) (§5.4.1).**
  - (5.113): `C_W²[<σ⁴> − <σ²>²] = C_W²(3A₄ − A₂²)K²` and `j(K) = 0`, with A₂ = (a₊²+a₋²)/2 and A₄ = (a₊⁴+a₋⁴)/2.
  - (5.115–5.117): `K' = C_b + χK`, `V' = χ²(3A₄/A₂² − 1)K² + χ²V`, `G{1}' = χ G{1}`. Since G{1}(1) = 0, the NLO metric vanishes: there is no O(1/n) shift of the mean kernel.
  - At criticality, C_b = 0 and C_W = 1/A₂ (He init is (0,2)) give (5.119) `K^(ℓ) = K*` and (5.120) **`V^(ℓ) = (ℓ−1)(3A₄/A₂² − 1)(K*)²`**. The book says the coefficient "evaluates to 2 for linear … in contrast to =5 for ReLU". The normalised fluctuation is therefore `V^(ℓ)/(n K*²) = 5(ℓ−1)/n`.
- **Our check: the exact chain contains this.** With A = (2/n)χ²_M and M ~ Bin(n,1/2), E A² = 1 + 5/n exactly, so Var A = 5/n, and Ĝ^(ℓ) = K* ∏_{k<ℓ} A_k gives `Var Ĝ^(ℓ)/K*² = (1+5/n)^{ℓ−1} − 1`.
  - Eq. (5.120) is the O(ℓ/n) term of this, and the same 5 is the 5 in 2/(5Z).
  - Deep-linear analogue, exact in the book (3.25, 3.28): `G₄ − G₂² = [(1+2/n)^{ℓ−1} − 1] G₂²`.
- **Bias-like source term.** The closest the book comes to noise is "semi-criticality" (§5.2 and §5.4.1, third bullet): C_W = 1/A₂ with C_b > 0.
  - K grows linearly "towards a nontrivial fixed point at infinity". V ~ ℓ³, and V/K² still grows ∝ ℓ.
  - For a single input, our noise adds 2σ² to the mean metric exactly as C_b does. It also adds its own fluctuation terms to (4.70), which the book does not treat.
- **Two inputs, exact for ReLU (§5.5).**
  - (5.160) is the nonperturbative off-diagonal recursion. At criticality it becomes (5.162): `cos ψ' = cos ψ + ρ[sin ψ − ψ cos ψ]`, with ρ = (a₊−a₋)²/(π(a₊²+a₋²)) = 1/π for ReLU (5.163).
  - (5.166): ψ^(ℓ) = (3/ρ)/ℓ + O(log ℓ/ℓ²) = 3π/ℓ. (5.168): K_[2] = K_d (9/(4ρ²)) ℓ⁻², so 1 − cos ψ ≈ 9π²/(2ℓ²). Our check: iterating the arc-cosine map numerically gives (1−c)ℓ² → 44.40 = 9π²/2.
  - The book does not solve the multi-input V for ReLU in Ch. 5. Its reason (§5.4(i)) is that these solutions "do not add anything to the criticality analysis".
- **Noise in the book.** No per-layer noise and no stationary distribution anywhere; a grep of the source finds noise only as Bayesian output noise in ch. 6.
- **Follow-ups (none has per-layer noise).**
  - S. Yaida, "Non-Gaussian processes and neural networks at finite widths", MSML 2020, arXiv:1910.00019. This is the original vertex recursion; I read only the abstract.
  - B. Hanin, "Random fully connected neural networks as perturbatively solvable hierarchies", JMLR 25(267):1–58, 2024; arXiv:2204.01058.
    - Defines κ₄ := (1/3)·(4th cumulant) = E[Δ²], which equals PDLT V/n. For the ReLU class it solves κ₄ = K²(6(a₊⁴+a₋⁴)/(a₊²+a₋²)² − 1)Σ1/n_ℓ; the factor is 5 for ReLU.
    - Log-normal joint limit: z → √K Z₁ exp(−μ + σZ₂), with μ = σ² = 5ξ/4 for ReLU and ξ = L/n. At criticality the variance is preserved exactly at finite width.
    - Proof structure: "the sequence of fields z^(ℓ) is a Markov chain", with conditionally Gaussian layers driven by "collective observables".
    - **Error to avoid.** JMLR eq. (77) and Prop. 26 state 1 − Corr ≈ (2/(3π))ℓ⁻². That contradicts the paper's own recursion ε' = ε − (4/(3π))ε^{3/2}, which gives ε = (1−Corr)/2 ≈ 9π²/(4ℓ²). PDLT and Li–Nica (q* = 9π²/2, §3) agree with 9π²/2. Use PDLT's constant.
- **Relevance.** The 1/n expansion at fixed ℓ/n reproduces only the low moments of our chain. The stationary law is heavy-tailed with infinite mean and needs ℓ ≫ n, which is outside its regime. Semi-criticality is the book's mean-field picture of our noise: no fixed point, linear growth.

## 2. Noise injection and signal propagation in rectifier nets
- **Pretorius, van Biljon, Kroon, Kamper, "Critical initialisation for deep signal propagation in noisy rectifier neural networks", NeurIPS 2018 (31:5717–5726), arXiv:1811.00293.**
  - Model: `h̃^l = W^l(x^{l−1} ⊙ ε^{l−1}) + b^l`, where ⊙ is + or ×. With + this is our model.
  - ReLU map: `q̃^l = σ_w²[q̃^{l−1}/2 ⊙ μ₂] + σ_b²`. With additive noise a fixed point requires (σ_w, σ_b, μ₂) = (√2, 0, 0): "critical initialisation is not possible using any amount of zero-mean additive noise". With multiplicative noise the critical choice is σ_w² = 2/μ₂ (dropout: 2p).
  - Mean field only (infinite width, averaged over weights and noise). No distributional or finite-width analysis.
- **Follow-up: Pretorius, Kamper, Kroon, Pattern Recognition Letters 138:75–81 (2020), arXiv:1910.05563.** Noisy NNGP: `k^L(x,x) = (σ_w²/2)^L k⁰ + Σ_l (σ_w²/2)^l (μ₂ + σ_b²)`. "For any amount of additive noise, all possible settings … result in a degenerate kernel in the limit of infinite depth"; their table gives σ_w² ≥ 2 → ∞ and σ_w² < 2 → a constant independent of x.
- **Schoenholz, Gilmer, Ganguli, Sohl-Dickstein, "Deep information propagation", ICLR 2017, arXiv:1611.01232, §Dropout.**
  - Independent masks per input. The diagonal is the noiseless map with σ_w² → σ_w²/ρ. At c̄ = 1: `c̄^{l+1} = 1 − ((1−ρ)/(ρ q̄*)) σ_w² ∫Dz φ²(√q̄* z) < 1`, so "even infinitesimal amounts of dropout destroys the mean field critical point".
  - This is the analogue for our independent per-input noise: it decorrelates inputs, so c* = 1 is not a fixed point.
- **Hanin and Rolnick, "How to start training: the effect of initialization and architecture", NeurIPS 2018, arXiv:1803.01719, Thm 5.**
  - With He weights and random biases of variance ν₂^(j), limsup_d M_d < ∞ a.s. **iff** Σ_j ν₂^(j) < ∞, where M_d = ‖act‖²/n. E M_d = M₀ + ½Σν₂, M_d − ½Σν₂ is a martingale, and with Σν₂ < ∞, `exp(½Σ1/n_j) ≤ E M_d² ≤ C₁exp(C₂Σ1/n_j)`.
  - For one input, independent per-layer bias draws are additive preactivation noise (our remark). Constant noise variance therefore gives limsup = ∞ a.s., which is consistent with a recurrent chain with a heavy-tailed stationary law.
- **Gürbüzbalaban and Hu, "Fractional moment-preserving initialization schemes for training deep neural networks", AISTATS 2021 (PMLR 130:2233–2241), arXiv:2005.11878. This is the closest prior work on noise plus Kesten.**
  - Gives exact s-th moments for ReLU via χ² with binomial degrees of freedom. Noiseless ReLU: x^(k) → 0 a.s. for every σ.
  - Thm "Heavy-tailed a.s. limit", proved for **linear** activation only, with s-moment-preserving σ and s ∈ (0,2): with i.i.d. Gaussian noise on post-activations, the layer outputs have a non-zero limit with infinite moments of order p > s. The limit is the Diaconis–Freedman backward series Σ_j(∏W)ξ_j, so the "a.s." limit is really the pullback limit.
  - No tail constant, no theorem for ReLU with noise, and no large-n limit law.

## 3. Joint depth–width limits, Gram-matrix chains and SDEs
- **Hanin and Nica, "Products of many large random matrices and gradients in deep neural networks", Commun. Math. Phys. 376:287–322 (2020), arXiv:1812.05994.**
  - Product M = ∏(pn)^{−1/2} D W with D a Bernoulli(p) diagonal. Thm 1: `ln((n₀/n_d)‖Mu‖²) ≈ N(−β/2, β)` with `β = (3/p − 1)Σ1/n_i + (μ₄−3)/(p n₁)‖u‖₄⁴`. Moments: `exp(C(k,2)β + O(Σn_i⁻²))`. For the ReLU Jacobian (p = ½): β = 5Σ1/n_i, plus a 4th-moment term.
  - Also gives the joint law of the norms ‖Ju_i‖² for several vectors u_i. There is no Gram-matrix chain and no noise.
- **Li, Nica, Roy, "The future is log-Gaussian", NeurIPS 2021, arXiv:2106.04013.** Fully connected case (α = 0, λ = 1): `z_out =d (‖x‖/√n_in) e^{G/2} Z`, with G ≈ N(−β/2, β) and β = 2/n + 5d/n. In the fully connected case the direction vectors are uniform on the sphere and independent from layer to layer; in the ResNet proof the direction sequence ẑ^ℓ is a Markov chain. Single input, no noise.
- **Li, Nica, Roy, "The neural covariance SDE", NeurIPS 2022, arXiv:2206.02768.**
  - States explicitly that "the norm … forms a Markov chain", and that (c/n)‖φ_{⌊tn⌋}‖² converges to the geometric Brownian motion exp(N(−σ²t/2, σ²t)), with σ² = 5 for ReLU.
  - **Exact Gram-matrix chain:** `z_{ℓ+1} | F_ℓ ~ N(0, (c/n)[⟨φ^α_ℓ, φ^β_ℓ⟩] ⊗ I_n)`. For unshaped ReLU this chain has **no SDE limit**: ρ jumps to 1.
  - For shaped slopes s± = 1 + c±/√n, Thm 3.2 gives `dV = b(V)dt + Σ(V)^{1/2}dB`, with b^{αβ} = ν(ρ)√(V^{αα}V^{ββ}), ν(ρ) = ((c₊−c₋)²/(2π))(√(1−ρ²) − ρ arccos ρ), and Σ = V^{αγ}V^{βδ} + V^{αδ}V^{βγ}. "Recurrent networks will not lead to a Markov chain or SDE limit." No noise.
- **Li and Nica, "Differential equation scaling limits of shaped and unshaped neural networks", arXiv:2310.12079.** For the unshaped MLP, q = ℓ²(1−ρ) with t = ℓ/n satisfies `dq = 2q((1 − (√2/(3π))q^{1/2})/t − 1)dt + 2√2 q dB` for t ≥ t₀ > 0. Its deterministic fixed point, q = 9π²/2, matches PDLT.
- **Zavatone-Veth and Pehlevan, "Exact marginal prior distributions of finite Bayesian neural networks", NeurIPS 2021, arXiv:2104.11734.** Exact single-input prior: for ReLU it is a binomial mixture over active widths k_ℓ ~ Bin(n_ℓ, ½) of deep-linear (Meijer-G) priors, plus an atom at 0. In the Edgeworth correction the linear factor ¼ becomes 5/4 for ReLU. Single input only, no noise.
- **Noci et al., "Precise characterization of the prior predictive distribution of deep ReLU networks", NeurIPS 2021, arXiv:2106.06615.** Exact kurtosis `κ = 3((m+5)/m)^{L−1}`, which tends to 3e^{5γ} when L = γm.
- **Answer to item 3.**
  - An exact Gram-matrix Markov chain from conditional Gaussianity and rotation invariance is in PDLT (4.69–4.71), Hanin (fields and collective observables) and LNR 2022 (stated explicitly).
  - SDEs exist only for shaped activations (LNR 2022) or for rescaled correlations (Li–Nica). **None of these works includes noise.**
- **Multi-input template (linear case).** Gautié, Bouchaud, Le Doussal, "Matrix Kesten recursion, inverse-Wishart ensemble and fermions in a Morse potential", J. Phys. A 54:255201 (2021), arXiv:2101.08082.
  - Recursion: `Z_{n+1} = √(εI+Z_n) ξ_n √(εI+Z_n)`. Its continuum stationary law is **inverse-Wishart**; for N = 1 this is the inverse gamma.
  - Related: Rider and Valkó, "Matrix Dufresne identities", IMRN 2016(1):174–218, and Grabsch and Texier, J. Phys. A 53:425003 (2020). Our remark: ReLU breaks rotation invariance in input space, so these results are not directly our law.

## 4. Perpetuities X = AX + B
- **Kesten–Goldie theorem.** C. M. Goldie, "Implicit renewal theory and tails of solutions of random equations", Ann. Appl. Probab. 1(1):126–166 (1991), doi:10.1214/aoap/1177005985. This is Thm 4.1, a special case of Kesten (Acta Math. 131, 1973) Thm 5.
  - Statement as given in BDM Thm 2.4.4 (p. 47). Conditions: A ≥ 0; log A given A > 0 non-arithmetic; E A^α = 1, E|B|^α < ∞, E A^α log⁺A < ∞; P(Ax+B = x) < 1.
  - Conclusion: `P(X > x) ~ c₊ x^{−α}` with **`c± = E[(AX+B)±^α − (AX)±^α]/(α m_α)`**, where m_α = E[A^α log A] (2.4.55–56). For α = 1 and A, B ≥ 0 this gives c = E B / E[A log A].
- **B correlated with X.** Goldie's implicit renewal theorem (Thm 2.3) needs only M ⊥ R and ∫|P(R>t) − P(MR>t)| t^{α−1} dt < ∞.
  - E. Damek, "Stochastic recurrence equation with diagonal matrices", arXiv:2106.11243, §"One dimensional case": for R =d MR + Q with R ⊥ M, Q possibly dependent on R, and E|Q|^{α+δ} < ∞, `c₊ + c₋ = E(|MR+Q|^α − |MR|^α)/(α m)`.
  - Relevance: the cross term 2g√(AQ/n) makes our B depend on Q. The formula C = E[Q' − AQ]/E[A log A] still holds.
- **Other representations of the constant.** c₊ + c₋ = lim E|Y_n|^α/(α m_α n), with Y_n = Σ_{i≤n} Π_{i−1}B_i. This is in BDM's comments after Thm 2.4.4, citing Bartkowiak et al. and Buraczewski et al. Also: Enriquez, Sabot, Zindy, PTRF 144:581–613 (2009).
- **Rates of convergence to the Kesten tail (BDM comments after Thm 2.4.4).**
  - Goldie's Thm 3.2 assumes A has a Lebesgue density and E A^{α+δ} < ∞, and gives `|(c₊+c₋) − x^α P(|X|>x)| ≤ const·x^{−σ₁}`. Our A has an atom at 0 of mass 2^{−n}.
  - Buraczewski, Damek, Przebinda, Electron. J. Probab. 20:1–35 (2015), give const·(log x)^{−σ₂} under weaker assumptions. I found no rate that holds uniformly as A → 1.
- **Dufresne.** D. Dufresne, Scand. Actuarial J. 1990:39–79; read only through Matsumoto and Yor, Probab. Surveys 2:312–347 (2005), Thm 6.2: for μ > 0, `∫₀^∞ exp(2(B_s − μs)) ds =d 1/(2γ_μ)` with γ_μ ~ Gamma(μ).
  - Our check: log A has mean ≈ −5/(2n) and variance ≈ 5/n. Rescaling time by 4/5 gives μ = 1 and Q/n → 2/(5γ₁), which matches the lane's limit.
- **Vervaat.** W. Vervaat, Adv. Appl. Probab. 11(4):750–783 (1979), doi:10.2307/1426858: existence, moments, and explicit examples (BDM §2.1–2.2).
- **Finite-n corrections to the inverse-gamma (diffusion) limit.** **No theorem found.** The physics weak-disorder treatment of z = 1 + x₁ + x₁x₂ + … is in de Calan, Luck, Nieuwenhuizen, Petritis, J. Phys. A 18:501–523 (1985). I could not read it; whether it gives correction terms is unverified.
- **Left tail near the lower end of the support.** **No general result found.**
  - BDM §2.5 covers the structure of the support. For exponential functionals of subordinators, Pardo, Rivero, van Schaik, Bernoulli 19(5A):1938–1964 (2013) give the density's behaviour at 0.
  - Our note: in the limit law, P(Q/n < x) = exp(−2/(5x)). At finite n, Q' ≥ χ²_{n−1}/n exactly, because σ²|P_⊥ξ|² ≤ |a+σξ|². This gives the upper bound P(Q < x) ≤ P(χ²_{n−1} ≤ nx).
- **Large deviations.** BDM §3.4.2, "The Kesten–Goldie case", in the chapter on univariate limit theory. I saw only the table of contents, which suggests it covers large deviations of partial sums.

## 5. Noisy deep networks at infinite depth, stationary and quenched laws
- The closest results are Gürbüzbalaban–Hu (linear activation with noise, a heavy-tailed pullback limit) and Hanin–Rolnick (per-layer bias noise, limsup finite iff Σν₂ < ∞), both in §2.
- Dunlop, Girolami, Stuart, Teckentrup, "How deep are deep Gaussian processes?", JMLR 19(54):1–46 (2018), arXiv:1711.11280. Deep GP samples are "Markovian with respect to the depth parameter", and effective depth is read from whether that chain is ergodic.
- Chehreghani, arXiv:2607.28185 (2026). A recurrent GNN with fresh Gaussian noise at every step is a geometrically ergodic chain with a unique invariant measure, given a globally contractive map. Weights are fixed and tails are light, so it is unlike our critical random-weight setting.
- General framework for the quenched setting, where the weights are the environment: Markov chains in random environments (Cogburn, PTRF 66:109–128, 1984) and iterated random functions and pullback limits (Diaconis and Freedman, SIAM Rev. 41, 1999). I found no paper that specialises this to random networks.
- **Not found anywhere (2015–2026):** the stationary law of a critical random-weight ReLU net with per-layer additive noise, its Kesten tail constant, the 2/(5Z) large-width limit, finite-width (1/n) recursions with a noise source, or a quenched-versus-annealed comparison for such nets.
