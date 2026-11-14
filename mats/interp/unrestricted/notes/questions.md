# Unrestricted estimators lane: assumptions and open questions (31 October 2026, 10:50Z)

Bead pool-uwmcvo (leaf .1). Each item says how it gets checked.

## Setting (assumptions)
A1. Nets: plain He-ReLU, width n, depth L, W_l iid N(0, 2/n), no bias, x ~ N(0, I_n);
    h_l = relu(W_l h_{l-1}) + s_l xi_l for l < L, output a_L = relu(W_L h_{L-1}) (vector in R^n).
    Noise s_l = sigma r_l, r_l = RMS of relu(W_l h_{l-1}) in a seeded pilot (narrow-deep convention, lib/nd.py).
    Check: my lib reproduces nd.py's forward on a shared seed (test).
A2. Estimand mu(W) = E a_L in R^n. Metric nMSE = |est - truth|^2 / tr Var(a_L); plain MC with N paths scores 1/N.
    Check: simulate MC replicates on one net (test).
A3. Cost: FLOPs, flopscope conventions as narrow-deep / arc-noisy (matmul M N (2K-1), elementwise 1, cdf 96, pdf 54,
    exp/log 16, normal draw 16). One noiseless MC path costs L n (2n - 1) + L n FLOPs. Deterministic arms charged
    for every bound pass, every cell integral and every refinement decision that is O(n) or larger.
    Check: counters unit-tested against hand counts on a 2-layer net.
A4. Deterministic arms work on the noiseless net (sigma = 0): certificates need piecewise linearity. The noise effect
    |mu_sigma - mu_0|^2 / tr Var at sigma = 1e-3 is measured separately (paired MC) and reported as a bias floor.
    Check: paired-MC noise effect on every registered net.
A5. Radius: for sigma = 0, f is positively homogeneous, so E f = E|x| E_u g(u) exactly; every arm (MC, RQMC, B&B)
    gets the radius for free (RB samplers), so the comparison is on the angular integral. Plain MC is reported too.

## Definitions to fix before data
D1. Deterministic: no random bits. Sample-free (ARC's working test): the estimate is not an average of f_W over a
    point set in input space. Certified: outputs an interval proven to contain mu (up to float64 rounding).
D2. The B&B point estimate that uses the network's local linear map at cell centres is a deterministic adaptive
    quadrature (point evaluations), not sample-free; its bound midpoint uses only bounds (deductive). Report both.
D3. "Beats sampling": ratio of nMSE (or certificate half-width squared / tr Var) to MC's and RQMC's at equal FLOPs,
    over a budget ladder, per net; median over nets.

## Open questions (theory)
Q1. Is "deterministic" vacuous on average over continuous random weights even uniformly (seed from the weights'
    low-order bits)? Check: write the proof with explicit TV and Lipschitz terms; cite typically-correct
    derandomization (Goldreich-Wigderson 2002; Shaltiel 2010 survey) after verifying the sources.
Q2. For fixed point sets (deterministic QMC, designs), is the average-over-W error exactly a kernel energy with a
    zonal kernel, so that design gains are bounded by the harmonic tail? Check: proof from rotation invariance
    (hardness note Prop 3.5) plus the radial factorisation; numerical check of the energy formula against simulated
    designs on one width.
Q3. Is the target mu(W) noise-stable in W (so overlap-gap/stability arguments cannot apply)? Check: measure
    Corr(mu(W), mu(W_rho)) on correlated net pairs; look for a proof via the Poincare/Hermite identity.
Q4. Which known average-case hard problems are relative-precision (low temperature) problems, and does the mean
    problem sit at the beta = 0 end? Check: literature statements (Gamarnik-Kizildag SK partition function;
    random-circuit output probabilities; Barany-Furedi/Elekes deterministic volume) read before citing.
Q5. Does any fine-grained average-case reduction framework (random self-reducibility of low-degree polynomials)
    apply to ReLU means? Check: state the obstruction precisely (mu is not a polynomial in W; piecewise structure).
Q6. Exact deterministic cost at fixed width: does the number of linear regions met by the sphere grow like a
    polynomial in L of degree about n - 1? Check: Hanin-Rolnick's hypotheses (biases?) and a direct count at n = 2, 3.
Q7. Is "no unrestricted lower bound is provable" a barrier statement we can make precise (superlinear lower bound
    for an explicit average-case problem; since determinism is free on average, the question equals optimality of
    MC among all algorithms)? Check: write it as a logical relation, have gpt-6-astra attack it.

## Open questions (experiments)
E1. Does an adaptive certified B&B over input cones (exact on stable cells, linear-relaxation bounds elsewhere)
    beat MC/RQMC at equal FLOPs at widths 2-8, and where is the crossover width? Check: PREREG predictions, grid
    n in {2, 3, 4, 6, 8} x L in {4, 8, 16, 32, 64}, plus widths 16 and 32 for the failure scaling.
E2. How does the B&B's cells-to-precision scale with n and L? Check: fit log(cells at fixed nMSE) against n and log L.
E3. Is the B&B's error dominated by the certified-exact mass or by the unstable mass? Check: report the fraction of
    Gaussian mass certified linear versus cells.
E4. Do deterministic (unscrambled) Sobol points match scrambled Sobol on average over nets, as rotation invariance
    predicts? Check: both on the same nets.
E5. Truth precision: MC with exact radius, 2^24+ paths; at n = 2-3 the B&B's exact limit cross-checks the truth.
