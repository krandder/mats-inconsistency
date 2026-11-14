You are gpt-6-astra, co-prover on the cut-rank problem (round 3: you check my new theorems; then the ledger).
Read-only sandbox. Files: coprover/astra_r1_answer.md and astra_r2_answer.md (yours), coprover/PROVENANCE.md (who did
what; I re-derived your B2 and B3 and accept your corrections to my Lemma C), scratch/ray_eps_scaling.py and
logs/ray_eps_scaling.log, runs/analysis_cut_*.json (measured cut matrices; scratch/tabulate.py prints a table).

The coordinator reshaped the question: what matters for matching sampling is how r(eps) grows as eps -> 0 against
sampling's 1/eps^2. If r(eps) grows faster than any fixed power, or like (1/eps)^{a n}, one-way linear summaries lose
to sampling at fine precision ("description size exponential in precision" instead of in k*).

My two new statements; please check each line, constants and hypotheses, and say PROVEN / WRONG / NEEDS X:

Theorem E (one back layer, exact). Fronts h = R u, u uniform on S^{n-1}, R >= 0 independent of u, E R = 1,
Var R = vR > 0. Back: one Gaussian ReLU row (its kernel is the arc-cosine kernel exactly, at any width). By isotropy the
cut operator is diagonal: squared singular values n eta_l^2 (your closed form), multiplicity N(n, l); Vbar = vR / 2.
Then for fixed n, r(eps) = Theta_n((eps^2 vR)^{-(n-1)/3}) as eps -> 0. Proof idea: energy per even degree ~ C l^{-4},
tail beyond D ~ D^{-3}; D(eps) ~ (eps^2 vR)^{-1/3}; r = number of harmonics of degree <= D (up to one partial degree)
~ 2 D^{n-1}/(n-1)!. Exact computation gives local exponents 1.33, 1.99, 2.66, 3.33, 4.66, 5.99, 7.31, 9.96 for
n = 3, 4, 5, 6, 8, 10, 12, 16 (prediction 2(n-1)/3). So r(eps) grows faster than 1/eps^2 iff n >= 5.

Theorem E' (multilayer finite-width back; uses your B2 and your gradient identity). Same fronts, back of any finite
width n and depth d. (i) Tail: sum_{l >= D} a_l^{(d)} >= 2^{-(d-1)} sum_{l >= D} a_l^{(1)} >= c_n 2^{-(d-1)} D^{-3}.
(ii) Eigenvalue per harmonic: lambda_l = a_l / N(n, l) <= (n-1) / (l (l+n-2) N(n,l)) <= c'_n l^{-n}. (iii) Block lemma
with B = {l >= D}: r(eps) >= (beta T_B - eps^2 Vbar) / (beta lambda*_B); choosing D with c_n 2^{-(d-1)} D^{-3} =
2 eps^2 Vbar / beta gives r(eps) >= c''_{n} [2^{-(d-1)} beta / (eps^2 Vbar)]^{(n-3)/3}. Upper bound (your A3):
r(eps) <= S(n, m), m ~ beta / (eps^2 Vbar) (check the normalisation), i.e. exponent <= 2(n-1). So at any finite depth,
for n >= 7 the exponent of 1/eps is > 2: one-way linear summaries lose to sampling as eps -> 0 for these fronts.
(iv) For actual synchronised ReLU fronts on coordinate faces of dimension |S| ~ n/2 (your face-by-face argument), the
same with n replaced by |S| and the orthant factor.

Then, in numbered sections:
1. Verdict on Theorem E and E' (with corrections).
2. Can E' be made uniform in depth (no 2^{-(d-1)} loss) inside the window d << n, using your gate-jump identity B3 or
   the kink coefficient growing linearly in depth (Bietti-Bach 2021 Cor. 2 at infinite width)? Prove or state the
   weakest conjecture.
3. The measured data (runs/analysis_*.json; tabulate.py): r_in(0.1) at F' = 256 fronts is 14/48/70/108 for
   n = 8/16/32/64 at c = L' = 8; 25/48/23 at c/n = L'/n = 1/2 (n = 8/16/32); 117/89/60 for L' = 2/4/8 at n = 16, c = 8.
   Measured local exponents d log r / d log(1/eps) between eps = 0.1 and 0.03 are about 0.6-1.5, i.e. below 2, but
   ranks are capped by the number of fronts. Is any of this in tension with the theory? What does the theory predict
   for spread (not synchronised) fronts as eps -> 0?
4. The conjecture ledger, final form. For each step needed to get from the proven statements to the original Gaussian
   fronts in the window (and to the "almost every fixed net" quantifier), give the weakest precise conjecture, which
   formulation it belongs to (ensemble one-way linear; fixed back with rich readouts; local slice), known open problem
   or new (references only if you are sure), evidence for and against, and the cheapest refuting experiment. Mark the
   single bottleneck.
Be rigorous and concise; plain text to stdout.
