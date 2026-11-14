# Persistence as the objective: the nonequilibrium-steady-state framing of the dark room and the white-noise room

Key: `persistence-steady-state`. Researcher subagent, 2026-10-03. Brief: `../../BRIEF-DARKROOM.md`.
Script: `persistence-steady-state/check.py` (numpy/scipy), output: `persistence-steady-state/output.txt`
(re-run: `cd persistence-steady-state && OMP_NUM_THREADS=4 python3 check.py`, about 20 s).

**Verdict after skeptic review (2026-10-03, see the Skeptic section at the end).** White-noise room:
**solved with a parameter**, downgraded from "solved, parameter-free". Dark room: **not solved** at the
default settings. Avoiding it needs a parameter. One hidden physical parameter, the sensor's energy gap
b = (E1 − E0)/kT, which the toy fixed at 0, decides both rooms. Only a comparative statement is
parameter-free: between streams with equal mean sensory energy, the one with the lower entropy rate always
yields more fuel.

**Original verdict (researcher).** White-noise room: **solved, with no parameter**. A stream of i.i.d. noise holds zero free
energy for every model and every memory, which follows from Gibbs' inequality and the information-ratchet
second law. Dark room: **not solved.** The persistence framing does not avoid sensory deprivation. It is
indifferent to it (reading R1), prefers it (R2), or prefers it whenever the dark room can sustain the system
(R3). The dark room is avoided only when its physics starves the system. That depends on a fact about the
room (energy cost, and whether "dark" means "no input" or "constant input"). It does not follow from
persistence. The central claim, that the dark room has high free energy for any system that exists, is a
theorem only in the form "a stationary system descends its own stationary surprisal". That form holds for a
system that lives in the dark room too, so as a principle for choosing rooms it is a tautology.

## 1. The framing in plain language

Friston's physics framing does not start from a goal. It starts from existence. Anything that persists has
a boundary (a Markov blanket) between inside and outside, and a long-run distribution p\* over its states
(a nonequilibrium steady state, NESS). Write the surprisal as ℑ(x) = −log p\*(x). It can be shown that the
system's average flow runs down ℑ, plus a circulating part that leaves ℑ unchanged. Read ℑ as "free
energy" and the system "will appear to" infer and act so as to keep itself in its typical states. The
preferences are then not fitted. They are p\*, the physics of the system's own persistence. The dark room
answer follows: a creature like us is rarely found in a dark room, so the dark room has high surprisal for
us, so we leave.

Sources (fetched 2026-10-03):

- Friston 2013, "Life as we know it", J R Soc Interface 10:20130475 ([PMC3730701](https://pmc.ncbi.nlm.nih.gov/articles/PMC3730701/)).
  Claim: life is "an inevitable and emergent property of any (ergodic) random dynamical system that
  possesses a Markov blanket". Internal states "will appear to model—and act on—their world". The paper
  "resolves the somewhat tautological aspect of this argument by turning it around": "any system that exists
  will appear to minimize free energy". It ends: "the ensuing homoeostasis is as illusory as the free energy
  minimization upon which it rests".
- Parr, Da Costa, Friston 2020, "Markov blankets, information geometry and stochastic thermodynamics",
  Phil Trans R Soc A 378:20190159 (abstract read through [Crossref](https://api.crossref.org/works/10.1098/rsta.2019.0159); the publisher page returns 403).
  Claim: "at non-equilibrium steady-state, the flow of internal states can be construed as a gradient flow on
  ... Bayesian model evidence".
- Kirchhoff, Parr, Palacios, Friston, Kiverstein 2018, "The Markov blankets of life", J R Soc Interface
  15:20170792 ([PMC5805980](https://pmc.ncbi.nlm.nih.gov/articles/PMC5805980/)). Claim: "for organisms to
  maintain their integrity they must minimize variational free energy"; "an organism does not merely have a
  model of its world; rather, it is a model".
- Friston, Thornton, Clark 2012, "Free-energy minimization and the dark-room problem", Front Psychol 3:130
  ([full text](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2012.00130/full)).
  The physicist's answer: "surprise only exists in relation to model-based expectations"; "Dark-Room agents
  can only exist if they can exist. The tautology here is deliberate"; "Dark-Room agents do exist:
  Troglophiles". He also says the priors "are updated and optimized to minimize free energy".
- Schwartenbeck, FitzGerald, Dolan, Friston 2013, Front Psychol 4:710
  ([full text](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2013.00710/full)).
  The paper says the dark room is surprising because of "starvation and sensory deprivation". The goal
  priors are "fixed and defined a priori".
- Critiques: Biehl, Pollock, Kanai 2021, Entropy 23:293 ([arXiv 2001.06408](https://arxiv.org/abs/2001.06408)).
  They show that steps rewriting the dynamics of blanketed systems "are not generally correct without
  additional (previously unstated) assumptions". Aguilera, Millidge, Tschantz, Buckley 2022, Phys Life Rev
  40:24 ([arXiv 2105.11203](https://arxiv.org/abs/2105.11203)). They find the blanket condition and the
  solenoidal-flow constraints "are only valid for a very narrow space of parameters". Neither critique is
  needed for the result below. Here I grant the framing in full.
- Thermodynamics of harvesting a stream: Boyd, Mandal, Crutchfield 2016, New J Phys 18:023049
  ([arXiv 1507.01537](https://arxiv.org/abs/1507.01537)), Eq. (5), read in the PDF:
  ⟨W⟩ ≤ k_B T ln2 (h′_μ − h_μ), where h_μ is the Kolmogorov–Sinai entropy rate of the input tape. The same
  authors, J Stat Phys 2017 ([arXiv 1609.05353](https://arxiv.org/abs/1609.05353)): "a ratchet must have
  memory to most effectively leverage structure and correlation in its environment".
- Richardson 2022 (local `2202.txt`), Prop. 14: the inconsistency of a factor graph's PDG is ⟨⟨M_Ψ⟩⟩ = −log Z_Ψ,
  the Helmholtz free energy.

## 2. Formal objectives: three readings of "persist"

The system's state is x = (η, s, a, μ): external, sensory, active and internal states. Under a policy π
(which room to stay in), the stationary density is p\*_π.

- **R1, self-evidencing (the identity).** The flow is f = (Q − Γ)∇ℑ_π, with ℑ_π = −log p\*_π, Q
  antisymmetric and Γ ⪰ 0. In discrete time the matching statement is that KL(p_t ‖ p\*_π) never increases.
  The "objective" is the system's own ℑ_π.
- **R2, entropy bound.** The system keeps the long-run average surprisal of its sensory states low:
  minimise E_{p\*_π}[−log p(s)], which is the sensory entropy, or the entropy rate h_μ under the best model.
  This is the "upper bound on the entropy of the Markov blanket" in Friston 2013.
- **R3, persistence proper.** The system has an energy reserve E. Existing costs c per step. Keeping a
  model of memory order k costs εk per step (Landauer-type upkeep, ε > 0). Free energy comes from the sensory
  stream through an information ratchet. A protocol tuned to the agent's prediction q(x | past) extracts
  W(x) = 1 + log2 q(x | past) units of k_B T ln2 per binary symbol. Its mean is 1 − h_k, which reaches the
  Boyd–Mandal–Crutchfield bound 1 − h_μ when k matches the source's order. Objective: maximise the
  probability of being alive after T steps, where death means E < 0. The persistence margin (mean net gain)
  breaks ties.

Only one quantity in R3 is fitted to the agent: model order k, and the agent chooses it to maximise R3. No
weight is placed on information. The scale k_B T ln2 sets units only. Everything else (c, ε, E_0, E_max, T,
and the room statistics) is a fact about the world.

## 3. The toy

Four rooms, each a binary sensory stream:

| room | stream | h_0 = H(X_t) | h_μ (entropy rate) |
|---|---|---|---|
| D dark | constant 000… (or, variant C3/C4, no stream at all) | 0 | 0 |
| N noise | i.i.d. fair bits | 1 | 1 |
| S structured | Markov chain, flips w.p. 0.95 (learnable with one bit of memory) | 1 | 0.286 |
| A alternating | 0101… (deterministic structure) | 1 | 0 |

**Part A, R1.** Over 1000 random stable linear diffusions, solving the Lyapunov equation always gives
Q = Γ − BΣ antisymmetric, with max |Q + Qᵀ| = 6e-14. Every stable linear system is "a gradient flow on its
own surprisal". The decomposition constrains nothing. In a chain over (room, bit) with four policies, KL to
the policy's own p\* falls monotonically for all 200 random starts, every time:

```
policy               D      N      S   argmin  KL-monotone
dark-dweller       0.029  6.644  6.644    D       True
noise-dweller      6.644  0.029  6.644    N       True
structure-dweller  6.644  6.644  0.029    S       True
roamer             1.585  1.585  1.585   tie      True      (room surprisal, bits)
```

For the dark-dweller, the dark room is the lowest-free-energy place. Each policy is a self-evidencing fixed
point, so "the dark room has high free energy" is true exactly for the systems that do not live there.

**Part B, R2.** Ranking by sensory entropy rate: D (0) = A (0) > S (0.286) > N (1). A also needs a
one-bit model, while D needs none. R2 picks the dark room. Noise always comes last, and no model can beat
h_μ = 1 on it.

**White-noise theorem (checked).** Over 10^5 random predictors q, the mean work 1 + E[log2 q(X)] on uniform
X has a maximum of −6e-10: it is ≤ 0, with equality only at q = 1/2. No model and no amount of memory gets
free energy out of noise.

**Part C, R3.** E_0 = 10, E_max = 20, T = 1000, 4000 runs.

- C1, c = 0.5, ε = 0.05: D survival 1.000, margin +0.50 (k = 0); A 1.000, +0.45 (k = 1);
  S 0.842, +0.16 (k = 1); N 0.000. **Winner D.**
- C2, sweep over c ∈ {−0.2…1.1} and ε ∈ {0, 0.01, 0.2}: D wins in every cell where anything survives. It
  ties with A only at ε = 0 and never loses to S or N. When c < 0 (food from elsewhere), all four rooms
  survive, pure persistence cannot rank them, and the margin picks D.
- C3/C4, dark = no input (W = 0): D is avoided iff c > 0. With the brief's three rooms only, S wins for
  0 < c ≲ 0.66 = 1 − h_S − ε, and nothing survives above that. At c = 0 every room survives. The survival
  maximiser then picks the order-0 (non-learning) policy even in S, because learning S adds downside
  variance (a −3.3 hit on each mispredicted bit) and buys nothing persistence needs.

## 4. Verdict per room, and what is and is not parameter-free

**White noise: solved, parameter-free.** Under R2 and R3, noise is never preferred to any room with
non-negative free energy, for every model class. The step is a theorem (Gibbs' inequality, the second law
for ratchets). Caveat: noise is avoided because the objective has no drive toward information at all. The
same objective gives no credit to S for being learnable, only for the fuel it yields.

**Dark room: not solved.** Note that no fitted agent parameter decides it. It fails differently in each reading:
- R1 is a theorem about every stationary process: the Helmholtz/Lyapunov identity. As a criterion for choosing a room it is a tautology. The preferences are read off the behaviour, so every persisting behaviour, the dark-dweller included, satisfies it. Friston accepts this in plain words ("the tautology here is deliberate"; troglophiles).
- R2 prefers the dark room. This is the original problem, unchanged.
- R3 prefers the constant dark room whenever it can sustain the system, and does so strictly for any ε > 0,
  with no threshold. The physics points the wrong way: the free energy in a stream is 1 − h_μ, which is
  largest for a constant stream. The dark room is avoided only when "dark" means "no input" and c > 0, that
  is, when the room starves the system. That outcome is decided by c, a fact about the environment, and it
  rewards food, not information. When food comes from elsewhere (c ≤ 0), persistence is indifferent.

**Parameter-free:** the R1 identity; the zero free energy of noise; the bound 1 − h_μ on fuel; the
ordering D ≥ A > S > N on fuel; D strictly beating A for every ε > 0; and no agent weight on information
anywhere. **Not parameter-free:** whether the dark room is viable (c against the fuel available), what
"dark" means physically (constant input or no input), and the risk settings (E_0, E_max, T), which decide
between learning and not learning S. None of these is fitted, but each is set from outside the principle,
and each decides the dark-room outcome. The claim that "the organism's factors are the physics of its
persistence" is correct and moves the problem without solving it: the factors are now facts about the
environment, and those facts decide the outcome.

Answer to the brief's question: the framing is a theorem in form R1, and that theorem has no normative
content. It becomes a tautology when used to explain why organisms leave dark rooms. Once it is given real
physical content (R3), it gives a real verdict, and the verdict favours the dark room.

## 5. What it would mean for a reasoning network

In Richardson's terms, −log Z is the network's PDG inconsistency (Prop. 14). A network whose objective is to
persist, meaning it keeps its own inconsistency low and stays near its own fixed point, will, if it can choose
which queries or inputs to attend to, settle on inputs it already prices coherently. That is the dark room:
a stale question set. It will ignore noise for the right reason, but it will not seek learnable structure
either. Self-consistency and survival-style objectives are fine as stabilisers and are not usable as
attention signals.

The thermodynamic toy points to the single change that would fix this, which belongs to the sibling keys
`learning-progress` and `logical-inductor-occam`. Measure fuel against the agent's current predictor instead
of against equilibrium: W(x) = log q_new(x) − log q_current(x). In a logical-induction market this is trader
profit, the market's exploitability. For noise it is zero once frequencies are learned, for the dark room it
is zero once the constant is learned, and it is positive only on structure the market has not yet absorbed.
This matches the LIR 2026 split (`lir2026.txt`): attention should go where control can still reduce
inconsistency, not where inconsistency is merely low. This last paragraph is a pointer and was not computed
here.

## Skeptic (2026-10-03)

Script: `persistence-steady-state/skeptic.py`, output: `persistence-steady-state/skeptic-output.txt`
(re-run: `cd persistence-steady-state && OMP_NUM_THREADS=4 python3 skeptic.py`, about 1 min).

**The hidden parameter is the energy gap of the sensory symbols.** Part C takes its fuel formula,
1 − h_μ, from Boyd–Mandal–Crutchfield. Their ratchet uses an information reservoir whose bit states have
equal energy. Sec. III of 1507.01537 says the energies "E0 and E1, of both bit states ... are taken to be
zero for symmetry and simplicity" (checked in the PDF). With a gap b = (E1 − E0)/kT, the equilibrium law of a
symbol is p_eq(1) = e^−b/(1+e^−b). A ratchet tuned to the agent's predictor q can then extract at most
W(x) = log2 q(x|past) − log2 p_eq(x) per symbol. Its mean is the relative-entropy rate of the stream from
equilibrium (the work bound in Esposito & Van den Broeck 2011, EPL 95:40004,
[arXiv 1104.5165](https://arxiv.org/abs/1104.5165), abstract read: work bounded by the free-energy difference
plus or minus kT times the information "relative to the corresponding equilibrium distributions"). At b = 0
this reproduces check.py exactly (section 1 of the output). b is dimensionless, so it does not just set
units. It decides the ordering:

| b = gap/kT | fuel ranking (D, N, S) | dark room | noise room |
|---|---|---|---|
| < 2h_S ln2 = 0.397 (toy: 0) | D > S > N | preferred | avoided |
| 0.397 to 2 ln2 = 1.386 | S > D > N | avoided | avoided |
| > 1.386 (visible photons at body temperature: b ≈ 85) | S > N > D | avoided | preferred to darkness |

In closed form: fuel(N) − fuel(D) = −1 + b·log2(e)/2 and fuel(S) − fuel(D) = −h_S + b·log2(e)/2. As
b → ∞ the constant 000… stream becomes the equilibrium tape and carries no fuel. The researcher's two
readings of "dark" (constant input, no input) then become one. Only fuel(S) − fuel(N) = 1 − h_S = 0.714 and
fuel(bright constant) − fuel(N) = 1 are the same for every b.

**Flips shown in the toy** (same choice rule as check.py: maximise survival, then the margin; ε = 0.05,
T = 1000):

- The dark room. In the researcher's C1 setting (c = 0.5), changing only b moves the winner from D (b = 0,
  0.5) to S (b = 1, 2).
- The noise room. Take b = 5 and a per-step cost of 0.8 on room S (the brief's rooms each carry a cost;
  0.8 is larger than S's information surplus of 0.714). The winner is S with buffer (E0, Emax) = (10, 20)
  or (30, 60), and N, the white-noise room, with (100, 200) or (300, 600). The agent then picks noise over
  both darkness and structure, because noise is bright, not because it is novel.
- A second decisive knob. Survival adds a second parameter, the energy buffer against the fluctuation
  scale. At b = 1 with no room cost, (E0, Emax) = (2, 4) or (10, 20) picks D and (30, 60) picks S.
  The researcher had already seen this for learning versus not learning S. It also decides between rooms.

**What survives.** The theorem behind the white-noise claim is correct: it is Gibbs plus the ratchet
second law. Read properly, though, it is comparative. Between streams with the same energy content, every
bit of entropy rate costs one bit of fuel. A persistence agent therefore never prefers noise to an equally
bright stream with less entropy. It does prefer noise to darkness whenever b > 2 ln 2. So it is a
light-seeker, not a structure-seeker, and the size of S's advantage over N (0.714 out of about 57 bits per
step at b = 80) does not grow with b. Neither room's avoidance follows from "persist" alone. Both follow from
b, c, the room costs and the buffer. These are facts about the world, not agent weights. Each of them
decides the outcome, so under the brief's standard the verdicts are white noise: solved with a parameter;
dark room: not solved, and avoidable only for some parameter values. The researcher's R1/R2 conclusions and
Part A are unaffected. The R1 identity is parameter-free and has no normative content, as stated.

**Citation check.** I fetched and confirmed every quoted phrase: Friston 2013 (all six phrases, PMC3730701);
Friston, Thornton, Clark 2012 (all five); Schwartenbeck et al. 2013 (both); Kirchhoff et al. 2018 (both,
through the Europe PMC full text); the Parr, Da Costa, Friston 2020 abstract (Crossref); the abstracts of
Biehl et al. and Aguilera et al.; Boyd–Mandal–Crutchfield 2017, the "requisite variety" phrase; and
Richardson 2022 Prop. 14 (local `2202.txt` line 1023). BMC 2016 Eq. (5) is in the body of the PDF, not in
the abstract, as the file says. One correction: the same PDF fixes E0 = E1 = 0, which is the assumption
discussed above. The md presents 1 − h_μ as "the free energy in a stream". That holds only under this
assumption.
