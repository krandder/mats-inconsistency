# Expected free energy: does it avoid the dark room and the white-noise room without fitted parameters?

Key: `expected-free-energy`. Author: researcher subagent, 2026-10-03.
Script: `math/darkroom/expected-free-energy/toy.py` (numpy/scipy), output: `.../output.txt`
(re-run: `cd math/darkroom/expected-free-energy && python3 toy.py`, about 1 min).

**Short version.** The epistemic part of expected free energy (EFE) is *not* parameter-free in the
sense that matters. Its value depends on which randomness the modeller calls "hidden state" and which
"likelihood noise". The data can never settle that choice: two generative models that give every
observation sequence the same probability get opposite verdicts on the white-noise room (one stays in
the noise room forever, the other is indifferent). The epistemic term also decays to zero once
everything learnable has been learned. After that, the prior preferences C decide everything, and
when the dark room is cheapest the agent ends up there for every positive preference weight. The dark
room is avoided only because C is chosen to rank its outcomes low (Friston's own answer). The
white-noise room is never avoided at equal cost. Even the parameter-learning term ("novelty") prefers
noise to a learnable stream for every Dirichlet prior, and no preference over sensory symbols can
reverse that. Verdict: dark room **solved with a parameter (C)**, white noise **not solved**.

## 1. The framing in plain language

Active inference says that an agent picks the policy with the lowest expected free energy G. G splits
into two parts:

- **pragmatic (extrinsic) value**: how well the predicted outcomes match the agent's *prior
  preferences*, a fixed distribution p~(o) over outcomes;
- **epistemic (intrinsic) value**: how much the agent expects to learn. It counts information gained
  about hidden states ("salience") and about model parameters ("novelty").

The hope behind the question is that the epistemic part alone does the work. A dark room teaches
nothing, so information gain would push the agent out of it. White noise teaches nothing either:
noise is "ambiguity", and ambiguity is penalised. If both held with no tuned weight, EFE would be the
parameter-free answer.

## 2. The formal objective, and what "prior preferences" are

From Da Costa, Parr, Sajid, Veselic, Neacsu & Friston, "Active inference on discrete state-spaces: a
synthesis", J. Math. Psych. 99:102447, Dec. 2020, arXiv:2001.07203 (eqs. 10, 13, 16; text fetched):

    Q(pi) = softmax(-gamma G(pi))
    G(pi) = KL[Q(s|pi) || P(s)] + E_Q(s|pi) H[P(o|s)]                        (risk + ambiguity)
          ~ -E_Q(o|pi)[ln P~(o)]  -  E_Q(o|pi) KL[Q(s|o,pi) || Q(s|pi)]  -  E KL[Q(A|o,s,pi) || Q(A)]
              extrinsic value         salience (info gain about states)      novelty (info gain about parameters)

**Prior preferences** are the distribution P~(o) (the "C vector", or P(s) when preferences are over
states). They are part of the generative model and are not inferred from data. Da Costa et al. say
that innate preferences "have hyperpriors that are infinitely precise (e.g., a Dirac delta
distribution) and thus cannot be updated in an experience dependent fashion" (sec. 9). pymdp
(Heins et al. 2022, arXiv:2201.03904) stores C "in terms of relative log probabilities", so it need
not be a normalised distribution. A preference is a log-probability in nats. Its *magnitude*
therefore sets the exchange rate between a nat of information and a unit of preference. This
magnitude, written lam below, is not a choice of units: the argmin flips at a definite value of it.
The policy precision gamma, by contrast, is harmless. It multiplies all of G, so it changes how
peaked the softmax is but never which policy minimises G. Friston et al. 2015 make gamma inferred
("ad hoc softmax parameters become the expected (Bayes-optimal) precision"). That removes the
softmax temperature, but not C.

Friston et al. 2015, "Active inference and epistemic value" (Cogn. Neurosci. 6(4):187-214,
doi:10.1080/17588928.2015.1020053; abstract verified via Europe PMC) state the intended dynamics:
"Epistemic value is maximized until there is no further information gain, after which exploitation is
assured through maximization of extrinsic value." The toy confirms this, and it is the problem: once
the information gain runs out, the dark room is exploitation whenever C does not penalise it.

Where G comes from: Millidge, Tschantz & Buckley, "Whence the expected free energy?" (Neural Comput.
2021, arXiv:2004.08128, fetched) show that G is not simply "the free energy in the future". Their
natural extension of variational free energy *discourages* exploration. They write that the epistemic
term "arises not from some connection to variational inference but is present by construction" (EFE
= their FEF minus information gain). Biehl, Pollock & Kanai, "A technical critique of some parts of
the free energy principle" (Entropy 2021, arXiv:2001.06408, fetched) target the Markov-blanket
derivation. They state that their issues "do not affect the validity of approaches where an
(expected) free energy minimizing agent is assumed a priori". So EFE is a postulated objective. It is
not a theorem of the FEP, and "the principle forces information-seeking" is not available as an
argument.

The dark-room debate itself: Friston, Thornton & Clark 2012 (Front. Psychol., PMC3347222, full text
fetched) answer that "the whole shape of the space of prior expectations is what determines
surprise... there might... be species for whom the best way to avoid decay and disorder is to find the
dark (but nutrient-rich) place and stay in it." Sun & Firestone 2020 (TiCS 24:346-348,
doi:10.1016/j.tics.2020.02.006; abstract verified) argue that such answers risk making predictive
processing "explanatorily empty" (that phrase is not in their abstract; it is how the Seth et al.
reply summarises them, seen only in a search snippet, so (unverified) as a quote). The replies by Seth, Millidge, Buckley & Tschantz ("Curious
inferences", the epistemic-value answer) and by Van de Cruys, Friston & Clark ("Controlled optimism",
the optimistic-priors answer) appeared in TiCS 24(9) 2020. Their titles and authors are verified via
Europe PMC; their texts were not retrievable (403), so their arguments here are (unverified)
recollection.

## 3. The toy

Three rooms. Each step the agent picks one room and sees one symbol from K = 4.

- **dark**: o = 0 forever, cost c_D;
- **noise**: o i.i.d. uniform, cost c_N;
- **structured**: o_{t+1} = sigma(o_t) for a fixed 4-cycle sigma that the agent does not know, cost c_S.

The agent's model of each room is a first-order Markov chain whose rows have Dirichlet(a0) priors and
are learned from counts. This one predictive model can be written as an active-inference generative
model in two ways. Both give *every* sequence the same probability:

- **F_state**: hidden state s_{t+1} = o_{t+1} (A = identity). The randomness of the next symbol is
  state uncertainty, so salience = H(predictive).
- **F_amb**: hidden state s_{t+1} = o_t, which is already known. The randomness is in the likelihood
  A (ambiguity), so salience = 0.

Novelty, the information gained about the Dirichlet parameters, is the same in both. Cost enters the
only way active inference allows: as a second outcome modality with ln p~ = -lam * c. The agent
picks rooms greedily by G (equivalent to gamma -> infinity) and breaks exact ties at random. Horizon
T = 20000.

### Output (from `output.txt`)

1. **Known parameters.** Per-step salience in F_state is noise ln 4 = 1.386 and dark/structured 0.
   In F_amb it is 0 for all three rooms.
2. **Observational equivalence.** On 2000 noise symbols the log evidence is identical in both
   families (-2791.48). Cumulative salience is 2743.5 nats under F_state and 0 under F_amb. No
   Bayesian model comparison can choose between them.
3. **Novelty per row visit (times n).** The noise row gives (K-1)/2 = 1.50 for every a0. A learned
   deterministic row gives (K-1)·a0(psi(a0+1) - ln a0) = 0.56, 1.09, 1.27, 1.38, 1.47 for a0 = 0.1,
   0.5, 1, 2, 10. This is always below 1.5, because psi(x) < ln x - 1/(2x) implies
   a0(psi(a0+1) - ln a0) < 1/2. Noise gives more parameter information than learnable structure, for
   every symmetric Dirichlet prior (the Clarke-Barron (d/2) log n rate holds only at interior points,
   and noise is an interior point). All novelty decays like 1/n.
4. **Greedy agent.** Costs (dark, noise, struct) = (0, .05, .05) ("dark cheapest") or (.2, .05, .05)
   ("C dislikes dark", Friston's "no food in the dark room"). Shares are fractions of the last 10% of
   steps:

| family, prior | costs | lam = 0 | lam = 1 | lam = 20 | lam = 30 | lam = 100 |
|---|---|---|---|---|---|---|
| F_state, known | dark cheapest | noise 1.0 | noise 1.0 | noise 1.0 | **dark 1.0** | dark 1.0 |
| F_state, a0=1 | dark cheapest | noise 1.0 | noise 1.0 | noise 1.0 | dark 1.0 (settles at step 265) | dark 1.0 |
| F_state, a0=0.1 | dark cheapest | noise 1.0 | noise 1.0 | **dark 1.0** (settles at 26) | dark 1.0 | dark 1.0 |
| F_state, any | C dislikes dark | noise 1.0 | noise 1.0 | noise 1.0 | noise 1.0 | noise 1.0 |
| F_amb, known | dark cheapest | 1/3 each (all tie) | dark 1.0 | dark 1.0 | dark 1.0 | dark 1.0 |
| F_amb, known | C dislikes dark | 1/3 each | noise .49 / struct .51 (tie) | tie | tie | tie |
| F_amb, a0=1 | dark cheapest | d .10 n .49 s .41 | dark 1.0 (settles at 1940) | dark 1.0 | dark 1.0 | dark 1.0 |
| F_amb, a0=1 | C dislikes dark | d .10 n .49 s .41 | n .54 s .46 | n .56 s .44 | n .52 s .48 | n .50 s .50 |
| F_amb, a0=0.1 | C dislikes dark | d .06 n .70 s .24 | n .73 s .27 | n .72 s .28 | n .71 s .29 | n .65 s .35 |

   F_state flips from noise to dark at lam = ln 4 / 0.05 = 27.7. The Dirichlet prior a0 moves that
   flip: with a0 = 0.1 the early predictive for the noise room is overconfident, so its salience is
   underestimated, the agent leaves, and it never comes back. That makes a0 a second parameter that
   decides the outcome.
5. **Preferences over symbols cannot help.** Over 2000 random C (normalised over the 4 symbols),
   G_struct - G_noise is +ln 4 in F_state and exactly 0 in F_amb, for every C. In general, for a
   structured stream and a noise stream with the same symbol frequencies m, the long-run gap is
   H(m) - h_struct >= 0 in F_state, where h is the stream's conditional entropy rate. So *the more
   learnable the stream, the more EFE penalises it relative to matched noise*. In F_amb the gap is 0.

## 4. Verdicts

**Dark room: solved with a parameter.** The epistemic terms keep the agent out of the dark room only
in two cases. In the first, something is still being learned elsewhere, which is transient: novelty is
about 1/n, so once it falls below lam·(cost gap) the agent settles in the dark room, at a time that
scales like 1/lam. In the second, F_state sends the agent to the noise room, which swaps one pathology
for the other. Permanent avoidance holds only when C ranks the dark room's outcomes below the
alternatives ("C dislikes dark" rows). That is Friston-Thornton-Clark's answer, and C is set from
outside: Da Costa et al. call it an infinitely precise hyperprior. With lam = 0 and no cost
difference, F_amb is exactly indifferent among the three rooms.

**White-noise room: not solved.** At equal cost, no variant prefers the learnable stream to noise in
the long run:

- F_state seeks noise forever, because salience equals predictive entropy (the noisy-TV problem).
- F_amb ties when the parameters are known and prefers noise while learning, because noise novelty
  exceeds structure novelty for every a0.
- No normalised preference over the symbols changes this (section 5).

Only two moves remove the noise room. One is a preference that singles out the noise room itself, a
stipulation rather than a solution. The other is the modeller putting the noise in A, which buys a
tie, not avoidance. The usual claim that "ambiguity is penalised" holds only *relative to the
modeller's declared state space*, and data cannot fix that declaration, because F_state and F_amb have
identical evidence.

**Does the pragmatic term reintroduce the fitted preference?** Yes. Once learning is done, G reduces
to its extrinsic term, so behaviour is set entirely by C and its scale lam. Kelvin's "weight that must
be large enough" appears here as lam·(cost gap) versus nats of information gain: lam = 27.7 in
F_state, and any lam > 0 sends the agent to the dark room in F_amb.

**Hidden parameters that decide the outcome:**

- C, the prior preferences: which outcomes count as bad;
- lam, the precision or scale of C relative to nats;
- the state/likelihood partition of the generative model (F_state vs F_amb), which no amount of data
  can identify;
- the Dirichlet concentration a0, which decides whether learning gets trapped.

gamma does not decide the outcome; it only sets units.

## 5. What it would mean for a reasoning network

For the LI-net, the transfer has two parts, one encouraging and one not.

1. **The partition problem mostly disappears in deduction.** In a theory-only reasoning net, an
   "observation" is the result of a computation, such as a derivation step or a propagated
   constraint. A sentence independent of the theory is never revealed by any computation. Its truth
   in a sampled world is irreducible noise, and it automatically sits on the likelihood (F_amb) side.
   Expected information gain about a query's price from one more block is then zero both for settled
   sentences (the dark room) and for independent sentences (the noise room). This matches LIR's
   attention/control split (Richardson et al. 2026, local `lir2026.txt`): attend to the
   inconsistencies that the parameters under control can reduce. The semantics fixes the partition,
   so the modeller does not have to.
2. **Self-extinction is a feature for a reasoner.** EFE goes quiet once nothing is left to learn. For
   a fixed query set, that is when the prices have converged. "Stay in the dark room" then means
   "stop thinking", which is the right behaviour, so the dark room is not a problem for a reasoner. C
   is not gone, though. It reappears as *which sentences are queries* and how much each one matters,
   which is the reasoner's agenda. The salience term must also be computed on query variables only.
   Computed on all latent variables, it rewards resolving any internal randomness at all, which is
   the F_state noisy-TV.

So EFE gives the LI-net a usable attention rule: spend blocks where the expected reduction of query
inconsistency is largest. That rule is parameter-free only because deduction removes the
state/likelihood ambiguity and because the query set is given. It does not answer the general
agent question. For an agent with real sensors, EFE does not solve both rooms without fitted
parameters.

## Sources

- Da Costa et al. 2020 (J. Math. Psych. 99:102447, doi:10.1016/j.jmp.2020.102447), arXiv:2001.07203: https://arxiv.org/abs/2001.07203 (PDF fetched; eqs. 10, 13-16, sec. 9 quoted)
- Millidge, Tschantz, Buckley 2021, arXiv:2004.08128: https://arxiv.org/abs/2004.08128 (fetched)
- Biehl, Pollock, Kanai 2021, arXiv:2001.06408: https://arxiv.org/abs/2001.06408 (fetched; scope statement quoted)
- Heins et al. 2022 (pymdp), arXiv:2201.03904: https://arxiv.org/abs/2201.03904 (fetched; C as relative log-probabilities)
- Friston et al. 2015, doi:10.1080/17588928.2015.1020053 (abstract via Europe PMC REST; full text not fetched)
- Friston, Thornton, Clark 2012, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3347222/ (full text via Europe PMC)
- Sun & Firestone 2020, doi:10.1016/j.tics.2020.02.006 (abstract via Europe PMC)
- Seth et al. 2020, doi:10.1016/j.tics.2020.05.011; Van de Cruys, Friston, Clark 2020, doi:10.1016/j.tics.2020.05.012 (metadata only; content unverified)
- Parr & Friston on generalised free energy (Biol. Cybern. 2019): (unverified), not fetched
- Clarke & Barron 1990 on the (d/2) log n rate of parameter information: (unverified) recollection; the toy computes the rates directly

## Skeptic (2026-10-03)

Script: `expected-free-energy/skeptic.py`, output `expected-free-energy/skeptic_output.txt` (about 1 min).
Shares are (dark, noise, struct).

**Result: the verdicts stand. Two more hidden choices decide outcomes; the horizon does not.**

1. **The prior's model class decides which room novelty favours (S1).** With a Dirichlet prior,
   noise gives more parameter information than structure (the author's result). Swap in a discrete
   class per row, {uniform} plus {one-hot}, with weight w on uniform. Now the deterministic row
   carries more total novelty than the noise row for every w: 1.61 vs 1.07 nats (w = 0.5), 1.43 vs
   1.41 (w = 0.01), 1.26 vs 0.03 (w = 0.99). The transient ranking flips from noise to structure.
   It does not last: both totals are finite, so after a few steps everything ties. At equal costs
   the long-run shares are (0.33, 0.34, 0.33): the agent sits in the dark room a third of the time
   and in the noise room a third. So "noise beats structure for every prior" in section 3 holds only
   within the Dirichlet family. With a discrete class it becomes a tie. Neither gives a lasting
   preference for structure.
2. **The Markov order of the model decides the F_state verdict (S2).** At equal costs:
   - F_state, order 1: (0, 1.00, 0). The agent watches noise.
   - F_state, order 0: (0, 0, 1.00). The agent leaves the noise room, but for a bad reason. An
     order-0 model sees the 4-cycle as uniform noise. The cycle's counts are *exactly* uniform, so
     its predictive entropy beats the noise room's by a hair. The agent picks the room that looks
     the most random.

   One modelling choice flips the white-noise behaviour completely, and the "good" setting is good
   by accident. F_amb is indifferent in both cases: (0.10, 0.48, 0.41) at order 1 and
   (0.30, 0.35, 0.35) at order 0.
3. **The planning horizon is not decisive (S3).** I tested committed H-step policies valued by Monte
   Carlo rollouts with imagined count updates, under F_amb with a Dirichlet prior (a0 = 1), at
   H = 1, 4 and 16:
   - "dark cheapest": (1, 0, 0) at every H.
   - "C dislikes dark": about (0, 0.54, 0.46) at every H.
   - equal costs: the dark share *rises* with the horizon, from 0.11 to 0.13 to 0.22.

   This matches the scaling argument. Novelty and cost both add up linearly over H, so H cancels in
   the comparison of the two. A longer horizon does not rescue either room.
4. **Why F_amb is blind to structure, in closed form (S4).** Once the parameters are learned, F_amb
   gives G = KL(q(o)||C) + E H[P(o|s)]. For any stream with outcome marginal m, this equals the
   cross-entropy -E_m ln C. I checked a non-uniform example, a deterministic order-2 cycle
   0,1,0,2,0,3 against i.i.d. noise with the same marginal. Over 1000 random C,
   |G_struct - G_noise| < 2e-15. So after learning, EFE ranks rooms only by how their symbol
   *frequencies* match C. Whether the stream is learnable never enters. That is the general form of
   section 3, item 5.
5. **A modelling detail that does not change the result.** Da Costa et al. define novelty over the
   likelihood parameters A (eq. 16, checked in the PDF). In F_state, A is the identity and is known.
   The toy's novelty there is information gain about the *transition* rows. Those already contribute
   to salience through the parameter-averaged predictive, so F_state counts them twice. Dropping the
   novelty term cannot help the structured room: the noise room wins on salience alone (ln 4 per
   step against 0).

**Citations checked.**
- Exact quotes found in fetched text: Da Costa ("infinitely precise (e.g., a Dirac delta
  distribution)", in the arXiv PDF, plus eq. 16), Millidge et al. ("present by construction"),
  Biehl et al. ("do not affect the validity of approaches where an (expected) free energy minimizing
  agent is assumed a priori"), pymdp ("relative log probabilities"), Friston 2015 abstract ("Epistemic
  value is maximized until..."), and Friston-Thornton-Clark 2012 ("whole shape of the space of prior
  expectations", "dark (but nutrient-rich) place").
- Crossref metadata matches for all nine DOIs.
- Two corrections are made in place above. Da Costa et al. is J. Math. Psych. 99 (Dec. 2020), not
  2021. "Explanatorily empty" is not in Sun & Firestone's abstract, so the quote is marked
  (unverified).

**Final verdicts.**

| Room | Verdict | Why |
|---|---|---|
| Dark room | solved with a parameter | Avoidance needs C to stipulate that dark outcomes are bad. With equal real costs, no variant avoids it without trading it for noise: F_amb gives dark a 0.10-0.33 share forever (more with a longer horizon), and F_state avoids it only by watching noise. |
| White-noise room | not solved | After learning, F_amb is exactly indifferent for every C (S4). F_state seeks noise unless an order-0 model makes the structured room look noisier (S2). |

Decisive hidden choices: C, lam, the state/likelihood partition, the prior family and a0, and the
model order. Not decisive: gamma and the horizon.
