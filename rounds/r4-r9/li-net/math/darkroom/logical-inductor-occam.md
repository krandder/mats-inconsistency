# Dark room and white-noise room: the logical-induction / Solomonoff framing

Key: `logical-inductor-occam`. Researcher subagent, 2026-10-03.
Toy: `math/darkroom/logical-inductor-occam/toy.py` (numpy only). Output: `.../output.txt`.
Re-run: `cd math/darkroom/logical-inductor-occam && OMP_NUM_THREADS=4 python3 toy.py > output.txt` (about 70 s).

**Short version.** The logical induction criterion is about prices, not actions, so on its own it ranks no
environment. The obvious way to turn it into a chooser is a "self-market": pick the room where your own
traders make the most profit. Read with the market's own beliefs, that expected profit is exactly zero in
every room, because prices are fair. Only the log-wealth (Kelly) reading gives a signal, and that signal is
Bayesian expected information gain (EIG). Read after the fact, the profit equals the Occam bound: the
complexity of the room's simplest explanation. That number rejects both the dark room and the white-noise
room with no fitted parameter: dark 1.8 bits, noise 4.3, structure 17.3 in the toy. But the criterion
itself caps every trader's profit in **every** room, so the drive to learn runs out everywhere. After that,
cost decides, and the dark room wins. Verdict (after the skeptic pass, sec. 6): white noise is **solved
only with a parameter**. The bounded payout holds for exactly fair noise in a coarse trader class; the
precision of the class decides whether the agent ends up stuck on noise. The dark room is **not solved**: it needs an exchange rate between
bits and cost, and that rate must grow with the horizon. A logical inductor makes it worse, because a
dark room where the agent sets itself problems pays as well as the world does.

## 1. The framing in plain language

A logical inductor (Garrabrant et al., local `abridged.txt`, `full.txt`) is a market whose prices cannot be
exploited by any efficiently computable trader. Two of its theorems are the relevant ones here:

- **Occam Bounds** (abridged Thm 4.1.3, full Thm 4.6.4): P_inf(phi) >= C 2^-kappa(phi) for every phi
  that Gamma does not refute. In a sequence-prediction setting this makes P_inf dominate the universal
  semimeasure (abridged Thm 4.1.4, full 4.6.5). The trader that knows the true program starts with a
  weight of about 2^-K and can therefore win at most about K bits of log-wealth.
- **Learning Pseudorandom Frequencies** (abridged Thm 4.2.7, full 4.4.2): if a sequence is pseudorandom
  relative to the market's own generable weightings, its price converges to the frequency p. Nothing
  more is extracted from it. "Unlearnable" is judged relative to the inductor's own traders, so it is
  not a separate modelling choice.

These are statements about beliefs. The criterion holds for any deductive process the world supplies. It
contains no actions, no utilities and no preference about which sentences show up. So the first answer
is direct: **the criterion says nothing about which environment to seek.**

To get an action rule, let the agent choose the room where "its traders profit". This can be read in
three ways:

1. **Money profit expected at the market's own prices.** Under the market's own beliefs, a share bought
   at price P_n(phi) has expected value P_n(phi). So the expected profit of every trader is 0, in every
   room, at every step. In a logical inductor this is the content of No Expected Net Update (abridged
   Thm 4.3.9): the market does not expect to update. A market that expected its traders to profit would be
   exploitable.
2. **Expected log-profit of the right trader.** Take the Kelly log-wealth of the trader that is right,
   averaged over the market's own belief about which trader that is. This equals
   sum_i w_i KL(P_i || M) = I(trader; next observation), which is Bayesian EIG. It is the KL-KSA
   objective of Orseau, Lattimore & Hutter 2013 (see Sources) and the epistemic term of expected free
   energy.
3. **Realised log-profit (after the fact).** This is log2 P_true(x_1:t)/M(x_1:t), the wealth the right
   trader actually took from the market. It is capped by -log2 w_true, which is the Occam bound.

## 2. Formal objective(s)

Rooms r, with stream x^r and per-step cost c_r. Market M = sum_i w_i P_i, with w_i = 2^-len(i), where
len(i) is a prefix code length, then normalised.

- O1 (criterion): none; prices only.
- O2 (money): V_r = E_M[sum_t profit_t] - c_r T = 0 - c_r T.
- O3 (Kelly/EIG, plannable): V_r = lambda * sum_t I_t(H; x^r_{t+1}) - c_r T.
- O4 (realised): V_r = lambda * log2(P_true(x^r_1:T) / M(x^r_1:T)) - c_r T <= lambda * K_r - c_r T,
  where K_r = -log2 w_true(r).

lambda converts bits into cost units. When every cost is equal it only sets units and is allowed. When
costs differ it is a weight that has to be large enough, which the standard does not allow.

## 3. The toy

**Traders.** There are 2061 of them, each with a prior of 2^-(prefix code length):
- iid Bernoulli(theta) for theta in {0, 1} and dyadic j/2^d with d <= 4;
- every periodic pattern with period 2 to 10.

Each trader also has change-point variants: "behave like me until time k, then output the constant c".
These get prior mass 1/2 * q(k) * 1/2, where q(k) = 2^-gamma(k) is the Elias-gamma prior, so that
"the light may come on later" is a live hypothesis. This follows Orseau et al.'s Prop. 12 construction.

**Rooms.**

| room | stream | cost/step |
|---|---|---|
| dark | 000... | 0 |
| noise | iid fair coin | 1 |
| pseudorandom | high bit of an LCG (period far beyond the class) | 1 |
| structured | period-9 pattern 101100111 | 1 |
| think | dark room plus a self-posed computable problem (period-10 pattern) | 0 |

The **think** room stands in for a logical inductor's deductive process: theorems it produces itself,
with no sensor involved.

**Output** (excerpt from `output.txt`, T = 4000):

| room | -log2 prior of best trader (Occam cap) | realised log2 wealth of best trader (O4) | cum. ex-ante EIG (O3) | E_M money (O2) |
|---|---|---|---|---|
| dark | 3.28 | 1.77 | 2.86 | 0 (max abs 0.0e+00) |
| noise | 4.28 | 4.28 | 10.06 | 0 |
| pseudorandom | 4.28 | 4.28 | 6.65 | 0 |
| structured | 17.28 | 17.28 | 8.62 | 0 |
| think | 18.28 | 18.28 | 8.74 | 0 |

- **Prices.** They converge to 0 (dark), 1/2 (noise and pseudorandom: Thm 4.2.7 behaviour) and the
  pattern (structured). The excess log-loss on noise over its 1 bit/step entropy is 4.28 bits in total,
  which is exactly the Occam cap.
- **Contrast: a market without Occam weights.** A KT (Beta(1/2,1/2)) market on the same fair coin lets
  the simple trader "theta = 1/2" grow its log2 wealth as 0.5 log2 n: 2.0, 3.7, 5.1, 7.0, 8.6, 10.3 at
  n = 10 to 10^6. That trader's wealth is bounded below and unbounded above, which is exploitation
  (Def. 3.5.1). So the criterion **forbids** this lasting "noisy TV" payout. A non-Occam learner keeps
  paying out on noise forever; an inductor pays K(frequency) + O(1) bits once.
- **lambda needed for the structured room to beat the dark room under O3.** 3.8 (T = 20), 17.8 (100),
  174 (1000), 694 (4000), 17360 (10^5). The information saturates, so lambda* grows linearly in T.
- **Greedy agent** (each step picks argmax_r lambda * EIG_r - c_r; T = 3000):

| scenario | lambda | visits | final 500 steps |
|---|---|---|---|
| A: dark, noise, pseudorandom, structured | 0 or 1 | dark 3000 | dark |
| A | 10 | struct 20, noise 21, prng 13 | dark |
| A | 100 | struct 32, noise 46 | dark |
| A | 10^4 | noise 1152, prng 268, struct 255, dark 1325 | dark 499 of 500 |
| B: A plus think | 1 | think 1977, dark 1023, world 0 | think |
| B | 100 | think 1867, world 110 | think |
| C: A with all costs 0 (lambda is pure units) | any | noise 1337, struct 640, prng 512, dark 511 | dark 256, struct 129, prng 80, noise 35 |

## 4. Verdicts

**White-noise room: downgraded by the skeptic (sec. 6) to solved with a parameter.** The author's original verdict was solved, with no fitted parameter, as a bounded transient. Original reasoning:
- Under every self-market reading the noise room pays a finite amount and then nothing.
- The realised payout is capped at K(frequency model) + O(1) (Thm 4.1.3), the same order as the dark
  room and far below the structured room (4.28 against 17.28 bits).
- A pseudorandom stream that the market's traders cannot crack gets the same treatment (Thm 4.2.7).
  No "is this learnable?" judgement is added from outside; it is relative to the agent's own traders.
- The criterion rules out the lasting log-n payout a non-Occam learner gives (the KT contrast).

The caveats are honest ones:
1. The plannable version (O3) still pays a slow trickle for refining the frequency estimate. With
   lambda large, or with all costs equal, the myopic agent spent more steps in the noise room than in
   the structured one (1152 against 255; 1337 against 640), although the total stays bounded.
2. Whether noise ranks above dark (4.3 against 1.8 bits) depends on the reference machine, an O(1)
   constant.

**Dark room: not solved.**
- O1 has no preferences.
- O2 is identically 0, so cost alone decides and the dark room wins.
- O3 and O4 are bounded in every room. That boundedness is the criterion itself: no budgeted trader
  profits without limit anywhere. So after learning every room is a dark room, and any cost gap sends
  the agent to the cheapest one for good (scenario A: dark at the end for every lambda).
- Visiting the structured room first needs lambda > (c_s - c_d) T / (K_s - K_d), a weight that must be
  large enough and grows with the horizon.
- With equal costs, what is left after learning is decided by the prior's tail of change-point
  hypotheses ("the light might come on"). The dark room ends up with half of the last 500 steps.
- Scenario B is worse for the framing. Profit does not care whether the bits come from sensors or from
  the agent's own deductive process. A dark room in which the agent poses itself problems beats
  engaging with the world at every lambda >= 1. This is Sun & Firestone's "daydreaming in the dark room"
  (unverified), and a logical inductor makes it real.

## 5. What it would mean for a reasoning network

- Read as a learning signal, the Occam/LI view gives the white-noise half for free. Pay traders (or
  heads, or experts) in log-wealth against an Occam-weighted mixture, and noise or pseudorandom data
  stops being worth attention after K(frequency) bits. No noise detector or tuned curiosity weight is
  needed. The network should not reward raw log-loss reduction against a non-Occam baseline, because
  that baseline pays log n forever on noise.
- **A beliefs-only framework cannot address the dark room.** Any objective derived only from the market's
  own profit is either 0 ex ante (fair prices), or bounded in every room. That bound is the defining
  property of a good inductor. Once nothing is learnable, the framework is indifferent, and costs or
  priors decide.
- To avoid sensory deprivation the network needs something outside the criterion:
  - an open-ended world that keeps supplying new structure; or
  - a preference, such as persistence or empowerment; or
  - an explicit rule that sensory sentences count and self-generated ones do not. That rule is a
    modelling choice; the think room shows profit alone will not draw the line.
- For LIR's attention/control split (`lir2026.txt`): "attend to inconsistency that control can reduce"
  inherits the same problem. Choosing a room is control. The inconsistency it can reduce is bounded and
  runs out, and the network's own reasoning supplies inconsistency more cheaply than the world does.

## 6. Skeptic (2026-10-03)

Script: `logical-inductor-occam/skeptic.py`, output `skeptic_output.txt` (about 3 min). It uses the
same Occam code lengths, change-point traders and greedy rule as `toy.py`. It changes two things the
toy held fixed: **D**, the precision of the i.i.d. traders' frequency grid (the toy uses dyadic
j/2^d with d <= 4, so D = 4), and **theta**, the noise coin's bias (the toy uses exactly 1/2).

**Hidden parameter found: the precision of the trader class (D), together with the prior over
frequencies. It decides the white-noise verdict.**

1. **The flip (equal costs, where lambda only sets units, so this is the parameter-free scenario).**
   Rooms are dark, noise and structured. A greedy EIG agent runs for T = 6000. The table shows the
   rooms it chose in the final 1000 steps.

   | D | theta | final 1000 steps |
   |---|---|---|
   | 4 (toy) | 1/2 | structured 560, dark 425, noise 15 |
   | 4 | 1/pi | dark 739, structured 217, noise 44 |
   | 12 | 1/2 | **noise 725**, dark 275, structured 0 |
   | 12 | 1/pi | **noise 958**, dark 42, structured 0 |

   With a finer grid there is always a nearby frequency left to resolve. Per-step EIG on noise then
   decays polynomially: about 1/(2n ln 2), scaled by the posterior mass still on nearby frequencies.
   That is slower than the structured room's leftover trickle, which comes only from the change-point
   tail (about log n / n^2 under the toy's Elias-gamma prior). In this late phase the choice comes from
   comparing tiny trickles whose sizes D and the prior set. The agent
   ends up in front of the noise: the noisy TV comes back. This happens even for exactly fair noise.
   When a cost gap is present (scenario A, lambda = 10^4), the dark room still takes the tail. That is
   the existing dark-room verdict, not a white-noise rescue.
2. **The plannable objective already prefers noise in the toy.** Cumulative ex-ante EIG (O3) over
   T = 4000:

   | D | dark | noise (1/2) | noise (1/pi) | structured |
   |---|---|---|---|---|
   | 4 | 2.86 | 8.86 | 10.03 | 8.62 |
   | 12 | 2.88 | 9.16 | 10.06 | 8.32 |

   The section 4 "caveat 1" is therefore the main result, not a caveat. The only reading an agent can
   plan with ranks noise at or above structure.
3. **The Occam cap (O4) is bounded only for noise whose statistics have a short exact code in the
   class.** S1 below is an i.i.d.-only market. Each entry is O3/O4 in bits; for comparison, the
   structured room scores 17.28 on O4.

   | D | theta | n = 10^4 | n = 10^5 | n = 10^6 |
   |---|---|---|---|---|
   | any | 1/2 | about 3.2/3.4 | about 3.2/3.4 | about 3.3/3.4 |
   | 4 | 1/pi | 6.3/8.3 | 6.3/21.9 | 6.3/122.5 |
   | 8 | 1/pi | 7.2/8.3 | 10.4/13.6 | 14.0/25.5 |
   | 12 | 1/pi | 7.2/8.2 | 9.9/12.5 | 11.7/14.6 |
   | 16 | 1/pi | 7.2/8.1 | 9.8/12.1 | 11.5/14.1 |

   - **theta = 1/2:** O3 and O4 are flat, as the toy claimed.
   - **D = 4 or 8, theta = 1/pi:** O4 grows linearly. The coin is not in the class, so the market
     stays mispriced and a computable trader exploits it.
   - **D = 12 or 16, theta = 1/pi:** both O3 and O4 grow at 0.5 log2 n (+1.7 bits per decade), and
     O4 crosses the structured room's 17.28 bits within a few more decades.

   In the parameter-free limit (all computable frequencies, prior 2^-K), Rissanen's redundancy lower
   bound applies. Rissanen 1986 is verified; the "(1/2) log n for all theta outside a measure-zero
   set" form is from memory (unverified). It makes the noise payout grow as 0.5 log n for almost every
   bias. White noise is bounded only because the reference class is coarse and the coin is exactly
   fair.
4. **Universal prior: every room's total O3 is infinite.** Orseau, Lattimore & Hutter 2013 prove
   V*_inf = inf under w = 2^-K (Prop. 12, via the 0^k 1^inf "light comes on" environments). Their own
   fixes are parameters: a prior of 2^-(1+eps)K (Prop. 13) or a summable discount. Their conclusion
   says: "One important concern lies in the choice of parameters/discount vector". The toy's
   Elias-gamma change-point prior has finite entropy (2.5 bits), so it plays the role of their eps
   fix. A 1/(k log^2 k)-type prior has unbounded entropy: 7.0, 9.4 and 16.3 bits at 2^100, 2^1000
   and 2^(10^6) (S3). So the finite "bounded transient" in O3 is itself a prior choice.
5. **Citations.**
   - **Overstated.**
     - Occam Bounds (abridged Thm 4.1.3) bound P_inf of *undecidable* sentences away from 0 and 1. It
       is a statement about the limit, not a K-bit cap on a trader's log-wealth.
     - Domination (Thm 4.1.4) is about P_inf for bit predicates *not mentioned in Gamma*, so it
       describes a limit prior, not online prices P_n on observed bits.
     - The criterion gives no rate. The "-log2 w_true" cap is the Bayes-mixture dominance bound of
       the toy's own prior, not an LI theorem.
     - The KT contrast shows that the criterion forbids an e.c. trader's unbounded profit when the
       frequency is itself efficiently computable. It does not bound the market's total learning on
       noise of generic bias.
   - **Correct.**
     - Thm 4.2.7, Thm 4.3.9 and Def. 3.5.1 match `abridged.txt` (the full paper numbers them 4.4.2,
       4.12.2 and 3.5.1).
     - Orseau, Lattimore & Hutter: authors, abstract quote ("does not consider noise to be
       information"), Prop. 9, Prop. 10 ("would have converged to optimal behaviour") and Prop. 12
       all match the fetched PDF.
     - Their noise-insensitivity demo (Fig. 1) uses two environments that share a *known* 0.5 coin.
       It does not cover noise of unknown bias, which is exactly the case point 3 shows.
     - Schmidhuber arXiv:0812.4360: verified.
     - Sun & Firestone: bibliographic data verified (TiCS 24(5):346-348, 2020,
       doi:10.1016/j.tics.2020.02.006). The "daydreaming in the dark room" wording is still
       unverified.

**Verdicts after the skeptic pass.**
- **White noise: solved with a parameter.** The class precision and frequency prior (D, or the eps or
  discount in the universal limit) decide whether the agent is captured. In the parameter-free
  universal limit the plannable objective is not solved: noise keeps paying about 0.5 log n bits. What
  remains parameter-free is narrower: noise whose statistics are exactly simple (a fair coin) costs at
  most K(frequency) + O(1) bits of realised regret.
- **Dark room: not solved (unchanged).** Point 4 adds a further reason. Under the universal prior the
  dark room's undiscounted EIG value is already infinite ("the light may come on"), so the ranking
  among rooms comes from the discount or eps, not from the principle.

## Sources

- Local: `abridged.txt` (Thms 4.1.3, 4.1.4, 4.2.7, 4.3.9; Def. 3.5.1), `full.txt` (Thms 4.4.2, 4.6.4, 4.6.5).
- Orseau, Lattimore, Hutter, "Universal Knowledge-Seeking Agents for Stochastic Environments", ALT 2013,
  LNCS 8139:158-172, http://www.hutter1.net/publ/ksaprob.pdf. Fetched and text-extracted. Used: the
  abstract ("does not consider noise to be information"), Prop. 9 (value > 0 iff the policy is
  informative), Prop. 10 (value 0 in a trap; the agent "would have converged to optimal behaviour" there)
  and Prop. 12 (undiscounted value infinite under 2^-K(nu), via the 0^k 1^inf environments).
- Schmidhuber, "Driven by Compression Progress", arXiv:0812.4360 (abstract fetched): interestingness is
  the first derivative of compressibility, and is therefore transient by construction.
- Sun & Firestone, "The Dark Room Problem", Trends in Cognitive Sciences 24(5):346-348, 2020, doi:10.1016/j.tics.2020.02.006 (bibliographic data verified by search 2026-10-03; quoted wording unverified).
- Rissanen, "Stochastic Complexity and Modeling", Annals of Statistics 14(3):1080-1100, 1986, https://projecteuclid.org/journals/annals-of-statistics/volume-14/issue-3/Stochastic-Complexity-and-Modeling/10.1214/aos/1176350051.full (abstract fetched; the (k/2) log n almost-everywhere form is unverified).
- Orseau et al. 2013, also used by the skeptic: Prop. 13 (2^-(1+eps)K prior) and the conclusion on parameters and discount (fetched PDF text).
- Gell-Mann & Lloyd's "effective complexity" (unverified) is the closest named concept to the O4 payout,
  the model part of a two-part code.
