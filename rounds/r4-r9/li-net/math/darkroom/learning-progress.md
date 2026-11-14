# Learning progress / compression progress / predictive information vs the dark room and the white-noise room

Key: `learning-progress`. Author: researcher subagent, 2026-10-03.
Scripts: `math/darkroom/learning-progress/toy.py` (output `output.txt`, ~45 s), `sens.py` (`sens-output.txt`, ~70 s),
`prng.py` (`prng-output.txt`, ~5 s). Re-run: `cd math/darkroom/learning-progress && python3 toy.py`. Skeptic scripts: `skeptic.py`, `skeptic2.py`.

## Short version

The slogan "constant streams and i.i.d. noise both have zero predictive information and zero learning progress"
is true only for a learner that already knows the room. For a learner that does not, both rooms carry
(1/2)·log2 t bits of learnable information (the one unknown bias parameter), which is Bialek-Nemenman-Tishby's own
logarithmic case with k = 1. What learning progress (LP) actually does:

- **White noise: solved with a parameter.** For a Bayesian learner whose model class contains the truth, LP over a
  horizon t is about (k_eff/2)·log2 t, where k_eff counts the parameters the stream pins down. Noise and dark both have
  k_eff = 1; a structured room has more, so it wins as t → ∞. At any finite horizon, though, the model class and its
  prior decide. With a uniform prior over Markov orders 0..8, a curious agent prefers noise to structure until about
  2^27 steps; with orders 0..12 noise leads by 165 bits at t = 8192. With an Occam prior 2^-(2^k), structure wins
  from the start. Below the right order, structure looks exactly like noise. A pseudo-random room (LFSR) is noise to a
  context-tree learner forever, and Berlekamp-Massey learns it in 62 steps. "Noise" means "unlearnable *for this
  class*", so the class is the parameter.
- **Dark room: not solved.** LP only postpones it. Every room's LP decays to 0 (as k/(2t) per step), and a room that
  is structured but already learned is worth as little as the dark room. (Skeptic: the toy's "learned room slightly
  *below* the dark room" compared a pre-trained S* with an untrained D. With equal pre-training, S* gives about 5x the
  dark room's information. Both still fall to 0 as 1/N0, so the conclusion holds; see Skeptic.) With any positive cost difference, the dark room wins after horizon t with
  λ* ≈ (k_S − k_D)·log t / (2t) → 0. So *any* exchange rate λ > 0 eventually sends the agent to the cheapest room,
  and the switch time is set by λ. Without costs, the long-run agent is indifferent. Avoiding the dark room is a
  property of the world (it must keep supplying new learnable structure), not of the objective.
- The level variant, **excess entropy E = I(past; future)** of the source itself (not the learner's progress), is
  0 for both rooms. It is positive for structure, keeps its value after learning, and needs no weight in the
  infinite-data limit. But it is learner-free, so a pseudo-random room (E = 31 bits) beats the learnable one
  (E = 1.43 bits) even though no bounded learner can tell it from noise. That is the white-noise room again. Its
  plug-in estimator's window also decides at finite data: for a noise stream of length 256 at block length 10, it
  estimates 5.7 bits, more than the structured room's 1.75.

The only parameter-free statement that survives: among options whose only payoff is information, measured in bits on
both sides, a Bayesian LP agent ranks rooms with no weight, *given its model class and prior*. The class is the one
input you cannot drop, and LP never says why not to stop.

## 1. Framing in plain language

Curiosity rewards the *improvement* of your own predictions, not their quality (Schmidhuber 1991; 2010) and not their
error (prediction-error curiosity, which loves noise). Dark rooms are boring because they are already predictable.
Noise rooms are boring because they never become predictable. What is interesting is "neither arbitrary (like
incompressible random white noise) nor regular in an already known way" (Schmidhuber's creativity page). Oudeyer &
Kaplan (2007) call the same idea learning-progress motivation: reward the decrease of prediction error, which steers
away from situations "too easy or too difficult". Bialek, Nemenman & Tishby (2001) supply the measure: predictive
information I_pred(T), the mutual information between past and future, stays finite, grows logarithmically with a
coefficient counting the model's parameters, or grows as a power law.

## 2. Formal objectives tested

Stream x_1..x_t, learner M (a Bayesian mixture or plug-in model), true source μ. All quantities are in bits.

- **A. Prediction error** (baseline): −log M(x_t | x_<t).
- **B. Error-drop LP** (Oudeyer-Kaplan): LP(t) = mean loss in [t−θ−τ, t−θ) − mean loss in [t−τ, t), within a region.
- **C1. Compression progress** (Schmidhuber): r_t = C(p_{t−1}, x_≤t) − C(p_t, x_≤t), recompressing the whole history
  with the old and the new model (plug-in KT parameters).
- **C2. Expected information gain** (Bayesian surprise, the KL-KSA reward of Orseau, Lattimore & Hutter 2013):
  I(model; x_t | x_<t) = H(M(·|x_<t)) − E_post H(ν(·|x_<t)). The agent can compute this from its own beliefs.
- **C3. Realized information about the truth**: Σ_t [−log M(x_t|x_<t) + log μ(x_t|x_<t)]
  = log post_t(μ)/w(μ). This is the pointwise form of the learner's predictive information. It is bounded by
  −log w(μ) in a countable class, which is an Occam bound like logical induction's, and it grows as
  (k/2) log t in a k-parameter continuous class. Only the experimenter can compute it, because it needs μ. It means
  "information gain" only when μ is in the class; otherwise it grows linearly with misspecification.
- **E. Excess entropy** of the ergodic source: E = lim_L [H(X_1..L) − L·h].

## 3. Toy

Binary streams, horizon t ≤ 8192, 16 seeds. Rooms: **D** constant 0; **N** i.i.d. fair coin; **S** noisy XOR chain
x_t = x_{t−1} ⊕ x_{t−2}, flipped with probability 0.05 (entropy rate 0.286; its order-0 and order-1 statistics are
exactly uniform, so below order 2 it *is* noise); **S\*** the same chain, with the learner pre-trained on 20,000
samples; **P** (prng.py) an LFSR x_t = x_{t−28} ⊕ x_{t−31} with an unknown 31-bit seed. Learners: KT
(Beta(1/2,1/2)) Markov models of fixed order k, and a Bayesian mixture over orders 0..D. Costs, where used: D 0,
N 1, S 1.

Key output (excerpts; full tables in `output.txt`, `sens-output.txt` and `prng-output.txt`):

```
C3 realized info about truth, cumulative bits    t=   16    256   1024   8192
  mix0..8   D  2.84  4.83  5.83   7.33     ~ (1/2) log2 t
            N  1.17  6.67  7.42   9.36     ~ (1/2) log2 t  (tie with D up to a constant)
            S  5.83 15.75 19.41  25.93     ~ (k_eff/2) log2 t, k_eff ~ 4
            S*-0.01  0.15  0.18   0.94     already learned: ~0, below D
  order8    N  0.54 17.64 137.25 541.91   fixed high order: noise is the most "informative" room
            S  7.68 40.83  76.17 170.71
C2 expected info gain (agent-side), mixture 0..8: D 4.95  N 42.73  S 24.98  S* 0.99   (noise preferred at t=8192)
C1 Schmidhuber progress, order 8:                 D 6.60  N 723.5  S 189.0  S* 32.5
   order 0/1 (structure invisible):               N = S exactly (6.83 / 12.65)
sens.py, C2 at t=8192, S vs N:
   uniform order prior  D=2: S>N   D=4: S>N   D=8: N leads by 20 bits, S overtakes at t ~ 2^27   D=12: N leads by 165 bits
   prior 2^-(k+1)       S>N for D<=12 (gap shrinking with D)
   prior 2^-(2^k)       S>N for every D
B error-drop LP (x1000 bits/step, cells D N S S*), single-run windows: signs flip with (tau,theta) and t, e.g.
   tau=32,theta=32: t=64: 88.6 -1.3 184.8 -16.7 | t=8192: 0.0 -0.6 -59.2 -39.9   (estimator noise ~ signal)
E exact: D 0  N 0  S 1.427  S* 1.427 (persists after learning)
  plug-in, n=256, L=10: D 0.000  N 5.726  S 1.754   (window decides at finite data)
P (LFSR): true E = 31 bits; order mixture loss 1.0001 bits/step forever (noise); Berlekamp-Massey exact after 62 steps
Costs (C2, mixture): lam* above which D beats S over horizon t: 0.197 (t=16), 0.0147 (t=1024), 0.00245 (t=8192) -> 0
   learned S* vs D at lam=0: -0.0005 bits/step (D slightly better)
```

## 4. Verdicts

- **Dark room: not solved.** LP gives the dark room the *same* order of reward as noise: (1/2)·log t, the one bias
  parameter. With a naive initial predictor, the dark room even shows a real early error drop (B: +88.6 at t = 64).
  LP prefers structure only while structure remains unlearned. After that, a learned rich world, the dark room and
  noise are all ≈ 0, and the choice falls to costs, tie-breaks or the stopping option. Any exchange rate λ > 0 between
  bits and external cost hands the long run to the cheapest room, and λ sets when. Orseau et al. say the same of
  KL-KSA: an agent in a trap has value 0. It "could jump knowingly into an inescapable trap" once it has learned what
  it can, and it avoids a trap only if information remains elsewhere.
- **White-noise room: solved with a parameter.** (Skeptic: the model class decides the ordering even asymptotically,
  not only at finite horizons; see Skeptic, check 3.) Once the noise room's bias is learned, a Bayesian LP agent gets no
  reward from noise (Orseau et al., Prop. 9: KL-KSA values a policy > 0 iff it is informative). Asymptotically,
  structure beats noise for any prior that contains the truth *in the Markov-order class*. (Skeptic: not true in
  general. In K2 = {Bernoulli} U {XOR chain with a flip rate}, which also contains the truth, noise stays ahead.) But the model class and prior decide at every finite
  horizon (crossover at 2^27 steps for a uniform prior over orders 0..8). Fixed high-order learners give the
  noisy-TV ordering (N ≫ S under C1, C2 and C3). Pseudo-random structure is noise or not depending on the class (LFSR).
  B (error-drop) adds windows τ, θ and a region partition, and its sign flips with them.
- **Structured but already learned:** ≈ 0 under every LP variant. (The toy's "below the dark room" comes from
  unequal pre-training; see Skeptic, check 2.) LP is boredom,
  not a value on knowing. Only E keeps a learned structure valuable, and E inherits the PRNG problem.
- **Maximisable without a preference weight?** Yes among purely informational options. C2 is computable from the
  agent's own posterior, and comparisons are bits against bits. But the total is finite: ≤ prior entropy in a
  countable class. So KL-KSA needs a summable discount or a finite-entropy prior for its value to exist, which the
  authors call an "important concern" ("choice of parameters/discount vector"). That is one more hidden choice.

## 5. A small negative result (argument, not a proof)

Any objective that is a functional of the stream's law alone, such as E or I_pred of the true process, separates a
pseudo-random room from a fair coin (31 bits vs 0), so it rewards rooms that are noise for every bounded learner. If
cryptographic PRGs exist (standard assumption, unverified here), no polynomial-time learner gains information from
such a room. Any objective that depends on the learner, such as LP or information gain, takes the model class as
input, and the toy shows the class flips the noise-vs-structure ordering at finite horizons. So "white noise" is
defined only relative to a learner's class, and no objective in this family avoids it without fixing that class. This
matches the logical-induction view: a pseudorandom sequence is one whose best bounded price is its frequency.

## 6. What it would mean for a reasoning network (li-net)

LP is the LIR attention rule "attend where control can reduce inconsistency", written as a derivative. It is a good,
unit-free *scheduler*: rank candidate queries or sub-graphs by the expected reduction of the network's own loss
(C2 computed on its own beliefs). It needs no weight, because both sides are in bits. It automatically drops settled
queries (dark) and irreducible ones (noise relative to its trader class: quote the frequency and move on, as logical
induction does). Total attention a query can attract is bounded by about its description length (the Occam bound
on C3). But LP gives no reason to keep reasoning rather than idle, and it makes "already understood" worth zero.
So a network driven only by LP needs an external supply of new questions (open-endedness) or a non-LP term. Its
notion of "unlearnable" is exactly its trader/model class: change the class and the noise room moves.

## Skeptic (2026-10-03)

Scripts: `learning-progress/skeptic.py` (output `skeptic-output.txt`, ~20 s) and `skeptic2.py` (`skeptic2-output.txt`).
The author already names several choices that decide the outcome: the model class, the prior over orders, the horizon,
B's windows and the exchange rate λ. The toy shows each one. I did not re-run toy.py; the numbers above come from its
stored outputs. My scripts re-derive the D, N and S cells I use from scratch, and the values match. I found three more
problems, two of which change claims above.

**Check 1. Under the realized measure C3, the edge exponent α of the prior decides the dark room.** toy.py fixes the
prior for each context at Beta(1/2, 1/2) (KT / Jeffreys). The dark room's parameter sits on the *boundary* (θ = 0).
Under a Beta(α, α) prior the Bayesian regret there is α·log2 t. Each interior parameter (noise, the XOR chain) costs
(1/2)·log2 t whatever α is. So the headline "both rooms carry (1/2)·log2 t" holds only for the Jeffreys prior; it is not
a property of the rooms. Order mixture 0..4, T = 8192:

```
alpha   C3 at T:  D      N      S     ordering   | C2 at T: D     N      S    ordering | D: C3 slope per log2 t
0.05            1.68  10.78  32.47   S > N > D   |        2.59  14.32  25.18  S > N > D |  0.050
0.50            7.33   7.98  25.72   S > N > D   |        4.95  14.68  23.80  S > N > D |  0.500
1.00 (Laplace) 13.00   7.30  27.89   S > D > N   |        5.20  13.71  21.73  S > N > D |  1.000
3.00           33.10   6.34  43.25   S > D > N   |        4.97  13.03  17.87  S > N > D |  2.997
```

Under C3, the uniform Beta(1,1) prior already ranks the dark room above noise. With any α > 2, the dark room's slope
(α per log2 t) beats the structured room's (4 parameters × 1/2 = 2), so over a long enough horizon C3 sends the agent
to the dark room. The agent-side measure C2 holds up. For an all-zero stream, C2's rate × t tends to
α(ψ(α+1) − ln α)/ln 2 bits (my derivation): 0.18, 0.53, 0.61, 0.68 and 0.71 for α = 0.05, 0.5, 1, 3 and 10. The
simulated rates are 0.19, 0.54, 0.63 and 0.70. Every value is below the interior value 1/(2 ln 2) = 0.72, so C2 always
ranks the dark room last and α only scales it. In a continuous class, C3 is not information about the truth. It is
regret, and at a boundary it includes the price of the prior's edge. Base the verdicts on C2, not C3.

**Check 2. "Learned structure is worth less than the dark room" is an artefact.** toy.py compares S* (pre-trained on
N0 = 20,000 samples) with a D that has not been trained at all. Here is the comparison with equal pre-training (C2 over
the next 8192 steps, α = 1/2): N0 = 0: D 4.95, S 23.2. N0 = 2000: D 0.86, S 4.76. N0 = 20,000: D 0.18, S 0.99. The ratio
stays near 5, roughly k_S/k_D. The last line of toy.py's own output shows the same thing: per-step rate × t at T is
0.85 for S* and 0.54 for D. So a learned structured room still beats a dark room that has been learned for as long.
The dark-room verdict still holds for the author's reason: every room decays as 1/N0, so any λ > 0 eventually picks the
cheapest room. It does not hold because learning makes structure worse than darkness.

**Check 3. The asymptotic win comes from parameter counting, not from structure, and the class sets the count.** Take
the claim "structure beats noise asymptotically for any prior that contains the truth". It holds in the Markov-order
mixture only because the XOR chain needs 4 KT parameters there and noise needs 1. Now take the class
K2 = {Bernoulli(θ)} ∪ {x_t = x_{t-1} ⊕ x_{t-2} ⊕ b, b ~ Bernoulli(ε)}: a uniform prior over the two models and KT on
θ and ε. K2 contains both true sources. C2 (cumulative bits):

```
        t=256   1024   8192   rate x t at T
  D      3.13   3.85   4.95   0.54
  N      5.19   6.59   8.49   0.98
  S      4.44   5.43   6.93   0.74   (-> 0.72, one parameter)
```

Noise leads at every checkpoint and keeps the higher rate. The reason is that noise fits both models (θ = 1/2 and
ε = 1/2), so the agent keeps refining two parameters and never settles which model is right. Structure settles in one
model. In both classes the truth is in the class, yet the white-noise ordering is reversed between the Markov mixture
and K2. So the class decides the ordering even asymptotically, not only at finite horizons. Modelling the structured
room more accurately makes it *less* interesting than noise.

**Smaller points.** sens.py's 2^27 crossover extrapolates from N's rate × t at T, which is 0.91 and still falling
toward 0.72. Treat it as an order-of-magnitude figure, not a computed crossing. B (error-drop) also depends on a third
hidden choice besides τ and θ, the region partition. Oudeyer & Kaplan introduce regions because without them,
comparing error windows over time is "in fact nonsense" (fetched). Their reason is that an agent would be rewarded just
for switching from an unpredictable situation (a leaf in the wind) to a trivially predictable one (a white wall). That
is the dark-room problem in their own example, and their fix is to choose a partition.

**Citations checked (fetched 2026-10-03).**
- Schmidhuber, arXiv:0812.4360: the abstract quote is correct.
- Schmidhuber's creativity page: both quotes are correct. IEEE TAMD 2(3):230-247 (2010), SAB 1991 pp. 222-227 and
  IJCNN 1991 vol. 2 pp. 1458-1463 are confirmed.
- Bialek, Nemenman & Tishby, Neural Computation 13, 2409-2463 (2001): correct, including the finite / logarithmic /
  power-law cases and "coefficient that counts the dimensionality of the model space".
- Burda et al., arXiv:1808.04355: "limitations of the prediction-based rewards in stochastic setups" is correct.
- Oudeyer & Kaplan, Front. Neurorobotics 1 (2007), doi 10.3389/neuro.12.006.2007: "too easy or too difficult", the τ
  window and "in fact nonsense" are correct. The doc's paraphrase of "nonsense" understates it: it is the dark-room
  example (see above).
- Orseau, Lattimore & Hutter (hutter1.net/publ/ksaprob.pdf, text extracted): all correct. That covers Prop. 9
  ("Suppose γk > 0 for all k. Then V^π_γ(h) > 0 if and only if π is informative"), Prop. 10 (V*_γ(h) = 0 in a trap),
  "could jump knowingly into an inescapable trap" and "One important concern lies in the choice of parameters/discount
  vector". The paper also says that "the expected information gain when observing statistics depending on a
  continuous parameter is typically infinite". That supports the (k/2) log t divergence of C2 here. The venue (ALT
  2013, LNAI 8139, pp. 158-172) is not printed on the fetched PDF: (unverified).
- Ay et al. 2008: still bibliographic only, as the author marked.

**Verdicts after review.**
- Dark room: **not solved** (unchanged). C2 ranks the dark room last for every Beta(α, α) edge. But its margin over the
  other rooms shrinks as 1/t, so the exchange rate λ on any cost decides when the agent goes there. Drop the "learned
  room below the dark room" evidence (check 2). C3 is worse: α decides whether the dark room ranks below noise at all
  (check 1).
- White-noise room: **solved with a parameter**, and the parameter matters more than the doc says. The model class
  decides the ordering asymptotically, not just at finite horizons, even when the truth is in the class (check 3). LP
  rewards the number of parameters a room makes the learner fit, and a class can make noise need more of them than
  structure does.

## Sources

- Schmidhuber, "Driven by Compression Progress", arXiv:0812.4360 (fetched): interestingness is "the first derivative
  of subjective beauty or compressibility, that is, the steepness of the learning curve".
- Schmidhuber, creativity page https://people.idsia.ch/~juergen/creativity.html (fetched): citations for IEEE TAMD
  2(3):230-247, 2010 and the 1991 IJCNN / SAB papers; "zero whenever the unexpected is just random white noise".
- Oudeyer & Kaplan, "What is intrinsic motivation? A typology of computational approaches", Front. Neurorobotics 2007,
  doi 10.3389/neuro.12.006.2007 (fetched): LPM with window offset θ and smoothing τ; regions R_n; comparing windows
  across different situations "is in fact nonsense" without regions.
- Bialek, Nemenman & Tishby, "Predictability, complexity and learning", Neural Computation 13:2409-2463 (2001),
  arXiv:physics/0007070 (fetched): I_pred finite, logarithmic (coefficient counts parameters) or power-law.
- Orseau, Lattimore & Hutter, "Universal Knowledge-Seeking Agents for Stochastic Environments", ALT 2013, LNAI 8139
  pp. 158-172, http://www.hutter1.net/publ/ksaprob.pdf (fetched): Prop. 9 (noise insensitivity), Prop. 10 (trap value 0),
  discount/prior-entropy caveat.
- Burda et al., "Large-Scale Study of Curiosity-Driven Learning", arXiv:1808.04355 (fetched): "limitations of the
  prediction-based rewards in stochastic setups".
- Ay, Bertschinger, Der, Güttler & Olbrich, "Predictive information and explorative behavior of autonomous robots",
  Eur. Phys. J. B 63:329-339 (2008), doi 10.1140/epjb/e2008-00175-0 (bibliographic data via search; text not fetched):
  predictive information as a robot objective.
- Local: lir2026.txt (LIR, attention and control), abridged.txt / full.txt (logical induction).
