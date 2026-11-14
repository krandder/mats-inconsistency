# Two analogies Kelvin asked about (coordinator, 11:33Z): my assessment before reading astra's round 2

Superseded by section 4 of the report (outbox/unrestricted-estimators-2026-10-31.md), which merges astra's round 2
and GPT-6 review A: verdicts there read "no transfer established" instead of "not plausible", the threshold-circuit
seed claim is withdrawn, and the spin-glass statement is about a Gaussian surrogate, failing from depth 9.

Labels as in the report. [V] = checked this session by search; [P] = an earlier lane's verified source; [M] = memory.

## 1. Derandomising threshold circuits (CAPP for TC0)

**How close.**
- Worst case, PROVEN (hardness lane, Thm 4.1): a threshold gate on Boolean values is relu(S - theta + 1) - relu(S - theta);
  Gaussian inputs are binarised by steep ramps relu(Kx) - relu(Kx - 1). So depth-d threshold circuits of size s
  become ReLU nets of depth O(d) and size O(s), with integer weights of size poly(s), whose Gaussian mean is the
  acceptance probability up to m / (K sqrt(2 pi)). Deterministic estimation for depth-O(1) ReLU nets therefore
  implies deterministic CAPP for the same depth of TC0.
- Mean estimation is an integral of approximate-counting problems: E relu(z) = int_0^inf P(z > t) dt, and each
  event {a_{L,j}(x) > t} is accepted by a depth-L threshold-like circuit (gates sign(w . h) on rectified values).
- What is known for TC0 [M unless marked]: unconditional PRGs and deterministic approximate counting for halfspaces
  (Gopalan-Kane-Meka 2015, seed nearly logarithmic [V: title, venue]), low-degree PTFs (Meka-Zuckerman; Kane),
  intersections of few halfspaces (Harsha-Klivans-Meka; O'Donnell-Servedio-Tan), functions of few halfspaces
  (Gopalan-O'Donnell-Wu-Zuckerman). For depth-2 threshold circuits of polynomial size nothing comparable is known;
  even lower bounds are about n^{1.5} gates / n^{2.5} wires (Kane-Williams STOC 2016 [V]; THR o THR n^{2.5-eps} for a
  function in E^NP via Williams' method [V: abstract]). Williams' program ties non-trivial CAPP/SAT algorithms for a
  class to lower bounds against it.
- Where the depth-1 positive theory stops for ReLU nets with Gaussian input: one ReLU unit has a closed-form mean
  (|w| / sqrt(2 pi)), so depth 1 is trivial. At depth 2, E relu(w_2 . relu(W_1 x)) is a function of n linear forms of
  an n-dimensional input with 2^n orthant pieces; it is not a function of few halfspaces, not a regular polytope and
  not a bounded-degree PTF, so none of the depth-1 constructions applies. The theory stops at depth 2.

**Calibration (PROVEN as stated relations; the cited bounds [M]).**
- Derandomisation asks for deterministic polynomial time at constant eps. ARC's bar (match sampling at every eps) is
  stronger. Even for one halfspace, PRG-based deterministic counting costs 2^seed = (n/eps)^{O((log log(n/eps))^2)}
  evaluations, more than sampling's O(1/eps^2) as eps -> 0. And by Beck's discrepancy bound no N-point set has error
  below about N^{-1/2 - 1/(2(n-1))} on every spherical cap, so for a rich class a deterministic point set can at best
  match sampling up to a factor eps^{-2/n}.

**Random weights (average case).**
- PROVEN (FORMAL Prop. 1; astra's exact construction): on average over continuous random weights determinism is
  free, uniformly. The same holds for random discrete threshold circuits whose weight law has a symmetry that
  preserves the acceptance probability (sign flips of an input coordinate and its weight column: canonicalising the
  signs of one row frees n uniform bits independent of the acceptance probability). So the average-case version of
  "derandomising TC0" has no barrier at all; what remains is efficiency (beating sampling), which the TC0 literature
  does not address. I found no study of average-case CAPP for random threshold circuits under an SD-normalised error.

**What a proof of similarity would have to show, and plausibility.** A cost-preserving reduction from worst-case CAPP
for depth-2 TC0 to mean estimation on random critical ReLU nets. Obstacles: (i) determinism is free on average, so
no determinism barrier can transfer; (ii) planting a circuit into a random net either changes the weight law
detectably or cannot change the law of the mean (hardness lane Prop. 3.2); (iii) the embedding needs integer weights
and steep ramps (K ~ size), far from He-ReLU weights. Verdict: not plausible as a hardness transfer. It is accurate
worst-case context (deterministic estimation of depth >= 2 ReLU nets sits at the TC0 frontier) and a useful
calibration (ARC's bar is stronger than derandomisation even at depth 1).

## 2. Convex-body volume

**What is known.** Randomised FPRAS (Dyer-Frieze-Kannan 1991) for membership oracles; deterministic oracle algorithms
cannot approximate within exponential factors (Elekes 1986; Barany-Furedi 1987: (c n / log n)^{n/2}) [M]. Exact volume
is #P-hard for H- and V-polytopes [M]. For explicit H-polytopes a deterministic polynomial-time (1 + eps)
approximation is open in general; deterministic quasi-polynomial or FPTAS algorithms exist only for structured
polytopes, via correlation decay or zero-freeness (Gamarnik-Smedira, CPC, arXiv 2312.03906; Bencs-Regts 2024; the
truncated fractional matching polytope, arXiv 2409.07283) [V: abstracts]. If promise-BPP = promise-P, DFK
derandomises, so a white-box deterministic lower bound at polynomial scale is believed false.

**ReLU analogue.**
- White box: mu_j(W) = sum_s b_{s,j} . E[X 1{X in C_s}], a weighted sum of Gaussian first moments of explicit polyhedral
  cones (each cut out by at most n L inequalities; up to 2^{nL} of them) (astra r1, 4.1). So the mean is a white-box
  Gaussian-volume problem with exponentially many explicit polytopes.
- Black box: the hardness lane's Thms 2.1 and 2.3 are exactly the ReLU Elekes/Barany-Furedi: no query algorithm
  (randomised, adaptive) beats MC by more than about 4x on explicit ReLU families, and a deterministic query algorithm
  loses a factor of order its number of queries, because mass can be hidden where it does not look (PROVEN, worst case).
- White box, worst case: the analogue of "deterministic H-polytope volume" is open; under #NSETH deterministic
  estimators lose a factor E (hardness lane Cor. 4.3). Average case over random weights: false (determinism free).
- A link worth stating (HEURISTIC): the known deterministic volume algorithms for explicit polytopes need correlation
  decay. The ReLU counterpart is contraction of the backward value functions across depth (hardness lane Thm 6.12:
  contraction makes layer-local errors depth-free). At criticality the scale mode provably never contracts (hardness
  lane Prop. 6.7) and angles collapse only polynomially, so this is the route that fails for He-ReLU nets; it may
  work in ordered or noisy regimes.

**What a proof of similarity would have to show, and plausibility.** A white-box lower bound for deterministic
Gaussian-volume approximation of explicit cones. At polynomial scale that is a promise-P != promise-BPP-type
statement (believed false); fine-grained it is the #NSETH statement we already have in the worst case; on random
weights it is false. Verdict: the similarity is already proven in the oracle model (black-box <= 4x; deterministic
factor-N loss); a white-box or average-case version is not plausible.

## 3. Spin glasses (for comparison; FORMAL section 5, updated by M3)
- Mean = infinite temperature: d/dbeta log E exp(beta f) at beta = 0; additive SD-scale precision, which MC reaches.
- Tails at tiny p = low temperature / relative precision; ARC's SD-normalised tail error is trivial there (zero beats
  MC when p << 1/N; astra r1), so a tail-hardness statement must be relative-precision.
- COMPUTED (scratch/spin_glass.py, registered; independent finite-difference check): for the infinite-width ReLU
  kernel kappa_L, (kappa_L'')^{-1/2} is concave on (0, 1) for L <= 8 and not for L >= 16 (violations at t ~ 0.5-0.9).
  This contradicts my registered P12.
- COMPUTED, post hoc (scratch/spin_glass_field.py), with the external field: the linear term h^2 = kappa_L'(0) is an
  external field (0.05 at L = 8, 0.013 at L = 16, against non-constant mass 0.17 and 0.07). In the spherical case the
  relevant interval is [qhat, 1) with h^2 + xi'(qhat) = qhat xi''(qhat) (Huang-Sellke 2021; formula from memory,
  unverified). qhat falls from 0.94 (L = 1) to 0.39 (L = 64); xi''^{-1/2} is concave on [qhat, 1) for L <= 8 and not
  for L >= 10. PROVEN UNDER the spherical theorems (as recalled) and the GP limit (input dimension -> infinity at fixed
  depth, infinite hidden width): Lipschitz algorithms reach the maximal output over input directions at depth <= 8,
  and an overlap gap (ALG < OPT) separates them from it at depth >= 10.
- Scale (HEURISTIC): the maximum of the output field over the sphere is about sqrt(n) OPT standard deviations (16 OPT
  at n = 256), while ARC's 1e-12 tails sit at about 7-13 SD, i.e. normalised energy 0.4-0.8, below the algorithmic
  threshold, so the gap concerns near-maximal outputs, not the tails ARC measures, and nothing about the mean.
- So of the three analogies, spin glasses is the only one that gives a precise, testable obstruction, and it is about
  extreme tails at depth, not about the mean.
