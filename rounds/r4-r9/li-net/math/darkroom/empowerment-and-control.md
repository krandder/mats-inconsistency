# Empowerment and control: does "maximise control" solve the dark room and the white-noise room?

Key: `empowerment-and-control`. Researcher subagent, 2026-10-03. Brief: `../../BRIEF-DARKROOM.md`.
Toy: `empowerment-and-control/toy.py` (numpy only, runs in seconds), output: `empowerment-and-control/output.txt`.
Re-run: `cd math/darkroom/empowerment-and-control && python3 toy.py`.
Primary source text used: `empowerment-and-control/src/salge2013.txt` (extracted from arXiv 1310.1863).

## Verdict in three lines

**Skeptic downgrade (2026-10-03, see the Skeptic section at the end):** white-noise room: **not solved**
(Burda et al.'s literal noisy TV, where an action redraws the image, has positive empowerment that grows
with the noise entropy and beats darkness and learnable structure with no parameter; only
action-independent noise is rejected, and that only relative to a controllable room, with lam, sensor
boundary and estimator prior deciding). Dark room: **not solved** in the brief's own toy (dark, noise,
passive learnable stream, each with a cost: the dark room wins for every lam > 0); solved with a parameter
only against a controllable room. The researcher's original lines follow.


- **White-noise room: solved, no parameter** (skeptic: downgraded to not solved). Action-independent noise has empowerment exactly 0 for
  every horizon, action set and sensor; noise mixed into a controllable channel always lowers it. Caveat:
  an agent that estimates the channel from finite data sees spurious empowerment in noise, and the
  estimator's prior (a Dirichlet concentration) decides whether it does (toy part D).
- **Dark room: solved with a parameter** (skeptic: not solved in the brief's toy). Darkness (sensors that do not respond to action) is the minimum,
  0, so it is never strictly preferred to a controllable room. But the horizon, the sensor boundary and the
  exchange rate to any external cost each decide the outcome, and the dark room *ties* a passive learnable
  stream and the white-noise room (all 0), so empowerment gives no reason to prefer learnable structure.
- **For the LIR split:** empowerment is exactly "the inconsistency that control over actions can force
  against the best fixed belief" (redundancy = capacity, checked numerically). Control over actions kills
  noise but ties dark with passive structure; control over beliefs also kills noise but ties dark with
  passive structure too (they differ only in how long learning takes). Neither control alone separates the
  learnable stream from darkness without a time or horizon parameter.

## 1. The framing in plain language

Empowerment (Klyubin, Polani, Nehaniv 2005; Salge, Glackin, Polani 2013) scores a state by how much the
agent's next n actions can change what it will sense: the number of distinguishable futures it can steer
to, in bits. A dark room scores zero because nothing the agent does shows up in its senses. A white-noise
room scores zero because what it senses does not depend on what it does. The hope is that one objective
rejects both rooms with no tuning.

The answer is: it rejects white noise cleanly, and it rejects darkness only relative to rooms where the
agent has *control*, not rooms where it can *learn*. Empowerment is not about learning at all; it is a
property of the true action-to-sensor channel.

## 2. The formal objective

n-step, open-loop, state-dependent empowerment (Salge et al. 2013, sec. 4.4.3-4.4.4, eq. 4.8):

    E_n(r) = max_{p(a_1..a_n)} I(A_1..A_n ; S_{t+n} | r)

the Shannon capacity of the channel p(s_{t+n} | a_1..a_n, r), computed by Blahut-Arimoto (Blahut 1972,
Arimoto 1972, as cited in Salge et al. sec. 4.4.7). In a deterministic world E_n = log |reachable sensor
states| (Salge et al. eq. 4.15). The behavioural hypothesis: move toward states of higher E_n.

Equivalent min-max form (the channel-capacity / minimax-redundancy theorem, Gallager 1979 and Ryabko 1979
(unverified, cited via van Erven and Harremoës, arXiv 1206.2459, who extend it)):

    E_n(r) = min_q max_a KL( p(S | a, r) || q(S) )

In PDG terms (Richardson 2022; LIR 2026 eq. 1, 3): take the PDG with a high-confidence arc tau(S|A)
(beta = inf) and one belief arc q(S) (beta = 1). Fixing A = a gives inconsistency <<M_q + A=a>> =
KL(tau(.|a) || q). So **empowerment is the inconsistency that control over A can force against the best
single uncontrolled prediction q of the future sensation**: the radius of the set of futures the agent can
choose between. Toy part E checks the identity on six rooms (C = max_a KL(.||q*) to 4 decimals).

## 3. The toy

Sensor alphabet K = 8, actions {left, stay, right} on a ring of 8 positions, exact channels, horizons 1-6.
Rooms: dark (sensor constant); dark + proprioception (sensor = own position); white noise (i.i.d.,
independent of action); passive periodic stream (a fixed period-8 sequence, learnable, not controllable);
lit ring (sensor = position); lit ring with noise eps; "hash TV" (sensor = fixed random hash of the whole
action history, 64 symbols); dark corridor whose first new sensation is 4 steps away, then the lit ring.

### A. Empowerment E_n in bits (exact channel)

| room | n=1 | n=2 | n=3 | n=4 | n=5 | n=6 |
|---|---|---|---|---|---|---|
| dark (sensor constant) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| dark + proprioception | 1.58 | 2.32 | 2.81 | 3.00 | 3.00 | 3.00 |
| white noise (action-independent) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| passive periodic stream | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| lit ring | 1.58 | 2.32 | 2.81 | 3.00 | 3.00 | 3.00 |
| lit ring + noise eps=0.25 | 0.95 | 1.33 | 1.55 | 1.63 | 1.63 | 1.63 |
| lit ring + noise eps=0.75 | 0.16 | 0.20 | 0.22 | 0.23 | 0.23 | 0.23 |
| hash TV | 1.58 | 3.17 | 4.32 | 5.49 | 5.95 | 6.00 |
| dark corridor -> lit ring | 0.00 | 0.00 | 0.00 | 1.58 | 2.32 | 2.81 |

Readings:
- Dark, white noise and the passive learnable stream are all exactly 0 at every horizon: empowerment
  cannot tell them apart.
- "Dark + proprioception" equals the lit ring. Whether a room is "dark" depends on where the sensor
  boundary is drawn. Salge et al. (sec. 4.5.3) state this directly: "which state has the highest
  empowerment might depend on an agent-sensor configuration (and not only on the world dynamics)".
- The corridor is worth 0 until the horizon reaches the first new sensation (n = 4). Salge et al. (sec.
  4.5.4): "something just beyond the horizon ... could change this completely". Too long a horizon flattens
  everything to log |S| ("Tragedy of the Greek Gods", same section).
- Noise on a controllable channel lowers empowerment monotonically (1.58 -> 0.95 -> 0.16 at n=1). This
  is the parameter-free part: empowerment is noise-averse, not noise-seeking. (Skeptic: only for noise
  added on top of a fixed channel. Noise the agent can switch on is sought, see Skeptic S2.)
- The hash TV beats the structured ring: empowerment rewards control bandwidth, not structure. A
  button-to-random-image TV with a large screen is the most empowering room in the toy. It is not a
  white-noise room (the image is a fixed function of the button history), but it shows that empowerment
  is indifferent to whether the controllable future is learnable in any compact way: with the true channel
  given, a 3^n-entry random lookup table counts as fully as a simple law.

### B. External costs (dark 0, noise 0.5, passive 0.5, lit 1.0), objective J = E_3 - lam * cost

lit wins for lam <= 2.5; dark wins for lam >= 3.0; switch at lam* = E_3(lit) / (c_lit - c_dark) = 2.807.
The passive stream never wins. The exchange rate between bits of control and cost decides the dark
room. It is not a unit: the outcome flips across it.

### C. Endogenous cost (no lam): the lit ring drains one energy unit per step from e0 = 3, death is an absorbing constant sensor

n = 1, 2: lit wins (1.58, 2.32 vs 0). n >= 3: both 0, the agent is indifferent between certain death in
the lit room and living in the dark room. Folding cost into the dynamics removes lam but the horizon then
decides, and empowerment cannot rank "alive in the dark" above "dead", because both are zero control.
This matches Salge et al. (sec. 4.3.1): "vanishing empowerment is equivalent to the agent's death".

### D. Estimated empowerment (agent learns the channel from N samples per action, n = 1)

| sensor K | N | noise: plug-in | noise: Dirichlet a0=1 | noise: a0=1/K | control: plug-in | control: a0=1 | control: a0=1/K |
|---|---|---|---|---|---|---|---|
| 8 | 10 | 0.342 | 0.077 | 0.241 | 1.585 | 0.586 | 1.318 |
| 64 | 10 | 1.519 | 0.053 | 1.262 | 1.585 | 0.143 | 1.417 |
| 1024 | 10 | 1.585 | 0.003 | 1.425 | 1.585 | 0.010 | 1.439 |
| 1024 | 100 | 1.452 | 0.030 | 1.423 | 1.585 | 0.131 | 1.569 |
| 1024 | 1000 | 0.574 | 0.111 | 0.570 | 1.585 | 0.774 | 1.583 |

(Full grid in `output.txt`.) With a large sensor alphabet and few samples, the plug-in estimate of a
white-noise room equals the true empowerment of a fully controllable room (1.585 = log2 3): each action's
few samples land on disjoint symbols, so noise looks like control. The Dirichlet prior concentration a0
decides: a0 = 1/K reproduces the illusion, a0 = 1 suppresses noise but also suppresses true control (0.010
at K=1024, N=10). As N -> infinity every consistent estimator gives 0 for noise, so the exact objective
is clean, but a learning agent's white-noise immunity carries a hidden prior parameter until it has
roughly K samples per action.

### E. Min-max check

For dark, noise, passive, lit, lit+noise and corridor at n = 3: C (Blahut-Arimoto) = max_a KL(p(.|a) || q*)
to 4 decimals (e.g. lit 2.8074 = 2.8074, lit+noise 1.5483 = 1.5483), confirming the PDG reading above.

### F. Control over beliefs instead of actions (passive rooms, order-1 Markov KT learner, T = 4000)

| stream | reducible inconsistency (T log2 K - cumulative log-loss) | final loss rate | steps to < 0.1 bit/step |
|---|---|---|---|
| dark | 11960 bits | 0.001 | 28 |
| white noise | -179 bits | 3.007 | never |
| periodic | 11753 bits | 0.011 | 381 |

Control over belief parameters (what LIR's control mask usually acts on) also rejects noise with no
parameter, but now dark and the learnable stream are both fully reducible and nearly equal; they differ
only in how long learning takes (28 vs 381 steps), i.e. in model complexity. Turning that difference into a
preference needs a time window or a learning-progress rate, which is a parameter again (this is the
compression-progress direction, covered by other keys).

## 4. Hidden parameters found

1. **Horizon n.** Decides whether a dark corridor with an exit is worth anything (part A), whether a
   costly room beats the dark room (part C), and too large an n flattens the landscape (Salge et al. 4.5.4).
2. **Sensor boundary.** Proprioception turns the dark room into the lit ring (part A). Which variables count
   as "sensations" is a modelling choice that decides the dark-room verdict.
3. **Action set / resolution and sensor resolution.** E_n <= min(n log|A|, log|S|); the hash TV shows that
   enlarging either changes the ranking (Salge et al.: "careful when the state-space of either actions or
   sensors is much larger than the other").
4. **Cost exchange rate lam** when costs are external (part B), or the horizon when they are endogenous
   (part C).
5. **Channel-estimation prior** (a0) for an agent that learns the channel (part D).
6. Open- vs closed-loop action sequences (Salge et al. 4.4.3 choose open-loop for simplicity); not varied
   here.

Not a hidden parameter: the white-noise zero. For action-independent noise, I(A;S) = 0 for every n, action
set, sensor and noise level.

## 5. Relation to the LIR attention-and-control split

LIR (Richardson et al. 2026, sec. 3) picks an attention mask phi (which arcs' inconsistency counts) and a
control mask chi (which parameters may move). Sec. 4.7 puts actions in the PDG: choosing A to minimise
inconsistency against a utility "soft constraint" gives expected-utility (beta_p >> beta_b) or max-max
(beta_p << beta_b) decisions, so even there the outcome depends on confidence weights.

"Attend to inconsistency that control can reduce" has two readings, and the toy separates them:

- **Control = actions in the world.** Empowerment is the preference-free version: the inconsistency control
  can force (or, for an unknown target preference, resolve) against any single fixed belief. It is zero
  for noise (parameter-free) and zero for darkness, and also zero for passive structure.
- **Control = belief parameters.** Reducible inconsistency is the regret a learner can remove. It is zero
  for noise (parameter-free), large for both darkness and passive structure, and exhausted after learning.

So the one parameter-free consequence shared by both readings is the white-noise one: inconsistency that no
control can move gets no attention. That is also the logical-induction behaviour for pseudorandom sequences
(abridged Def. 4.2.6 and sec. 4.2: they get their frequency and nothing more). The dark-room consequence is weaker: a
dark room has no inconsistency left to reduce, so it is never *sought* for its own sake, but it is not
*avoided* relative to other zero-value options either, and cost or a horizon breaks the tie.

## 6. What it would mean for a reasoning network

Read the reasoning network's controls as what it can choose to compute (which arcs to update, which proofs
or queries to run) and its "sensors" as its future prices or beliefs. Computational empowerment of a belief
state is then the number of distinguishable future belief states it can reach by choosing what to compute
within budget n. It is zero for settled questions (the dark room: nothing to change) and zero for
undecidable or pseudorandom ones (the white-noise room: prices stay at frequency whatever is computed),
and positive for questions that a choice of computation can decide within budget. This gives an
attention rule that ignores noise with no tuning. Its weak points are the same as in the toy. The budget n
decides what counts as decidable. The choice of which belief variables are "sensed" decides the value.
Any external cost needs an exchange rate. It also prefers questions it can *move* over questions it can
*learn*, which for a reasoner may be the wrong priority (a hash-like question whose answer is cheap to
compute but carries no structure scores as high as a deep one). One possible exception to "the horizon is
a fitted parameter": if n is the network's actual remaining compute rather than a tuned constant, it is
given by the situation, not fitted. Whether that is enough is open; Salge et al.'s flattening argument
says a large real budget makes the drive uninformative.

## Sources

- Klyubin, Polani, Nehaniv (2005), "Empowerment: a universal agent-centric measure of control", Proc. IEEE
  CEC 2005, vol. 1, pp. 128-135. Bibliographic record verified at https://uhra.herts.ac.uk/handle/2299/1114
  (abstract not on the page).
- Salge, Glackin, Polani (2013), "Empowerment - an Introduction", arXiv 1310.1863,
  https://arxiv.org/abs/1310.1863 (full text fetched; quotes above from sec. 4.3.1, 4.4.3, 4.4.6, 4.5.3,
  4.5.4). The brief's "2014" is the book year of Guided Self-Organization: Inception (unverified).
- Friston, Thornton, Clark (2012), "Free-energy minimization and the dark-room problem", Frontiers in
  Psychology 3, https://www.frontiersin.org/articles/10.3389/fpsyg.2012.00130/full (fetched): "Agents that
  predict rich stimulating environments will find the 'dark room' surprising and will leave at the earliest
  opportunity."
- Burda et al. (2018), "Large-Scale Study of Curiosity-Driven Learning", arXiv 1808.04355 (fetched): the
  abstract notes "limitations of the prediction-based rewards in stochastic setups" (the noisy-TV issue is
  in the body, unverified here).
- Mohamed, Rezende (2015), "Variational Information Maximisation for Intrinsically Motivated Reinforcement
  Learning", arXiv 1509.08731 (fetched): scalable variational empowerment.
- van Erven, Harremoës (2014), "Renyi Divergence and Kullback-Leibler Divergence", arXiv 1206.2459
  (fetched): "extend the known equivalence between channel capacity and minimax redundancy". Gallager 1979
  and Ryabko 1979 as the original sources are unverified.
- Richardson et al. (2026), Local Inconsistency Resolution, local `lir2026.txt` (sec. 2 eq. 1-3, sec. 3,
  sec. 4.7 Prop. 7). Richardson (2022), local `2202.txt`.

## Skeptic (2026-10-03)

Script: `empowerment-and-control/skeptic.py` (imports `toy.py`), output `empowerment-and-control/skeptic-output.txt`.
Re-run: `cd math/darkroom/empowerment-and-control && python3 skeptic.py`.

The researcher already named horizon, sensor boundary, action/sensor resolution, lam and the estimator prior
as dark-room parameters, and the toy confirms each one flips (parts A-D reproduce). What the original
verdicts missed is that the white-noise verdict was only tested on the easiest noise room, and the
dark-room verdict was not tested on the brief's own toy.

**S1. The brief's toy, where the learnable stream is passive.** Dark (cost 0), i.i.d. noise (0.5),
learnable periodic stream (0.5): E_3 = 0, 0, 0. lam = 0 gives a three-way tie; lam = 1e-9, 1e-3 or 1 all
pick the dark room. No value of lam avoids it, so this is not "solved with a parameter" but **not solved**:
empowerment has nothing to say about learnable-but-uncontrollable structure, and any cost at all sends the
agent to the dark room. The dark room is avoided only when the alternative is a room the agent can steer.

**S2. Burda et al.'s noisy TV, taken literally.** Burda et al. (2018, sec. "Limitation of prediction error
based curiosity", Fig. 6) describe "a TV that randomly changes channels when an action is taken" and add
"an action to change the channel". That noise is action-*dependent*: pressing gives a uniform image over C
channels, not pressing keeps the current one. Exact empowerment: C = 2: 0.32 bits; 4: 0.56; 16: 0.83;
256: 0.98; 4096: 0.998 (limit 1 bit), the same at n = 1, 2, 3. So the canonical noisy TV beats the dark room
and every passive learnable stream (both 0) with **no parameter**, and noisier TVs are preferred (E rises
with log C). Against a controllable room with sensor noise eps = 0.25 (E_1 = 0.95), the TV wins once C >= 16.
The original claim "noise ... always lowers it" holds only for noise added on top of a fixed controllable
channel; noise the agent can switch on is counted as control. This is the white-noise problem, unsolved.

**S3. Band remote (pure noise, three buttons).** Keep / redraw from the lower half of the screen / redraw
from the upper half. E = 1.17, 1.41, 1.56, 1.583 bits for C = 4, 16, 256, 4096, constant in n. At n = 1 the
pure-noise remote approaches the lit ring (log2 3 = 1.585); for n >= 2 the lit ring wins (2.32, 2.81). The
horizon decides whether structure is preferred to a remote for noise; a remote with more buttons than the
ring has reachable states wins at every n (action resolution, the researcher's parameter 3).

**S4. External cost flips the white-noise room exactly as part B flips the dark room.** Noise (cost 0.5) vs
lit ring (1.0), no dark room on offer: lit for lam <= 5.5, noise for lam >= 5.7, switch at lam* = 5.615.
The verdict asymmetry in the original (noise "solved", dark "solved with a parameter") is not supported:
on the same tests both rooms flip on lam.

**S5. Sensor boundary flips the white-noise room too.** Noise room with proprioception: E = 1.585, 2.32,
2.81 (= lit ring), against 0 for the noise room alone.

**What survives.** One parameter-free statement: for noise that no action can influence, the exact
channel gives E = 0 at every n, action set and sensor, so such noise is never strictly preferred to a
room the agent can steer when costs are equal. That is a statement about uncontrollability, not about
learnability, and it does not extend to (i) noise whose occurrence the agent controls (S2), (ii) choices
with any cost difference (S1, S4), or (iii) a learning agent's estimated channel (part D, prior a0).

**Hidden parameters, final list.** Horizon n (A, C, S3); sensor boundary (A, S5); action and sensor
resolution (A hash TV, S3); cost exchange rate lam (B, S1, S4); channel-estimation prior a0 (D); and the
model-class choice of which variables count as "actions" (an action that redraws noise counts as control,
S2). None of these is a unit-setting scale: each one changes the winner.

**Citations checked.**
- Salge, Glackin, Polani 2013 (local `src/salge2013.txt`): quotes from sec. 4.3.1 ("vanishing empowerment is
  equivalent to the agent's death", line 251), 4.5.3 ("which state has the highest empowerment might depend
  on an agent-sensor configuration", lines 842-843), 4.5.4 ("something just beyond the horizon", "Tragedy of
  the Greek Gods", "much larger than the other", lines 876-897), 4.4.3 open-loop (line 505), eq. 4.8 (line
  528), eq. 4.15 (line 617), Blahut-Arimoto (line 626): all verified.
- Friston, Thornton, Clark 2012, Front. Psychol. 3:130: quote verified by fetch (original uses double quotes
  around "dark room").
- van Erven and Harremoës, arXiv 1206.2459: "we extend the known equivalence between channel capacity and
  minimax redundancy to continuous channel inputs" verified; Gallager/Ryabko 1979 remain unverified.
- Burda et al. 2018, arXiv 1808.04355: abstract phrase "stochastic setups" verified; the noisy-TV passage
  (now quoted in S2) verified from the PDF text. Correction to the original's framing: the noisy TV there is
  action-triggered, so it is not the action-independent "white noise" room of part A.
- Klyubin, Polani, Nehaniv 2005, Proc. IEEE CEC vol. 1, pp. 128-135: record verified at
  https://uhra.herts.ac.uk/handle/2299/1114.
- LIR 2026 (local `lir2026.txt`): sec. 4.7 Decision Making and Proposition 7 (a) beta_p = inf -> expected
  utility, (b) beta_b = inf -> max-max, verified (lines 821-900). Logical induction Def. 4.2.6 (pseudorandom
  sequence) at `abridged.txt` line 704, verified.
- Mohamed and Rezende 2015 not re-checked.
