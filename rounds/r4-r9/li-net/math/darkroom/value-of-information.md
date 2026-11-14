# Value of information (and the goal as a prior component) vs the dark room and the white-noise room

Key: `value-of-information`. Seventh candidate, from Kelvin (2026-10-03): "the value of learning information should
mostly depend on whether the information is going to be useful for some goals", sharpened the same day to "information
learned matters ONLY insofar as it helps the goal, where the goal is a prior component". Researcher subagent, 2026-10-03.
Scripts: `math/darkroom/value-of-information/toy.py` (output `output.txt`, ~20 s) and `pdg_goal.py` (output
`pdg-output.txt`, ~7 s). Re-run: `cd math/darkroom/value-of-information && python3 toy.py > output.txt && python3 pdg_goal.py > pdg-output.txt`.

## Short version

- **Dark room: solved with no fitted weight, at equal cost, and only while the goal is still open.** Kelvin's objective
  is: the value of a stream is the expected drop in the inconsistency between a preference edge and the rest of the
  agent's beliefs. It gives the dark room exactly 0, because a constant stream says nothing about the goal. It gives the
  room that bears on the goal more than 0. The value is capped by the goal's prior surprisal −log P(success). Once the
  goal is settled, every room is worth 0. A "rest" edge then sends the agent back to the dark room. No exchange rate
  enters, because both sides are in nats, but the rest edge's content decides. A drifting world keeps the value positive
  (0.12 bits/step in S).
- **White-noise room: solved with no fitted weight.** Known-fair noise, noise with an unknown bias, a learnable stream
  about something else, and noisy labels of a goal-irrelevant variable all score **exactly 0**. Their information
  content does not matter: N_side has 0.57 nats of curiosity value and 0.0000 of goal value. Noise that does bear on the
  goal scores more than 0 (N_goal 0.154 nats), and that is correct.
- **Is a weight needed?** With a hard preference edge, "the preferred event W = 1" at any finite confidence β_pref,
  the β drops out. The value is E log P(success | x) − log P(success). This is Richardson's surprisal of the preferred
  event (2202 Prop. 2), and it is parameter-free (N_goal 0.6320 at β_pref = 0.25 and at 100). A **soft** preference,
  "succeed with probability 0.9", brings in two hidden parameters, p and β_pref/β_belief. They reorder the
  goal-relevant rooms (S/N_goal goes from 0.62 to 0.12 to −13.8), and they punish knowing that you will succeed more often
  than p. They never move D or N off 0.
- **The honest cost.** Remove the goal and VOI is identically 0 in every room, so cost decides and the dark room wins.
  Replace the goal with "everything about my own stream" (all latents, or all sensory sentences as queries) and the noisy
  TV returns: N1 pays (1/2) log2 n with no cap, and noise moves prices by 0.5 per step forever. The goal is not a scalar
  weight. It is a **choice of variable**, and it must be about the world, not about one's own sensations. Making the goal
  "forecast my own next symbol" sends VOI straight to the dark room (VOI 0.5 for D against 0.445 for S).
- **Which of the three satisfies Kelvin's condition exactly:** only the preference-attributable inconsistency
  reduction (GAP). Plain VOI agrees on all the zeros, but it needs an explicit decision. It also returns 0 for evidence
  that changes the reconciliation without changing the act. Plain curiosity fails: it pays for the dark room (1 bit), for
  unknown-bias noise (unbounded) and for irrelevant structure. Raw PDG inconsistency, which includes the data's
  surprisal, is free energy, and it picks the dark room for n ≥ 4.

## 1. Framing in plain language

Howard (1966) defined the value of information as the most a decision maker should pay for it. It is the expected
utility of acting after seeing the information, minus the expected utility of acting now. Two consequences matter here.
First, information that cannot change any action is worth exactly 0, however surprising or informative it is (Raiffa &
Schlaifer's EVSI; Good 1967: free evidence never has negative expected value). Second, there is nothing to compute
without a decision. Lindley (1956) and Bernardo (1979) close the loop with curiosity. If the "decision" is to report a
distribution over a latent under the log score, then VOI equals the expected information gain about that latent. So
curiosity is VOI for one particular goal, "know this latent", and the choice of latent is the whole question.

Kelvin's sharper version puts the goal inside the beliefs. The agent's PDG has a preference edge, "the preferred world
holds", at ordinary confidence, next to its perception and world-model edges. Learning X is worth the expected drop in the
part of the PDG's inconsistency that the preference edge causes. Learning that leaves that part unchanged is worth
exactly 0.

The self-market reading (logical induction, the `logical-inductor-occam` candidate) is that information is worth what it
moves a price by. For a trader who will trade one share of query q at today's price, the value of an observation is
exactly E|P_after(q) − P_now(q)|, checked in the toy. So "attend to what would move a price" is VOI, once someone says
which queries.

## 2. Formal objectives

Stream X from the chosen room. World latent Z = f*, one of the 16 boolean functions of two bits. Goal query
q = f*(1,1) (pdg_goal.py) or f*(a,b) at a random input (toy.py).

- **VOI** (Howard): E_x[max_A E U(A, Z | x)] − max_A E U(A, Z), with 0-1 or log utility.
- **G0, no decision**: one action, so VOI ≡ 0.
- **CUR**, plain curiosity: I(latents of the room; X), the expected KL on whatever the room's latent is.
- **Self-market**: E|ΔP(q)| over a query set Q; also E ΔP², E KL and signed E ΔP (= 0, a martingale).
- **E1**, the goal as a prior component, KL form: I(Z; X) for a preferred variable Z with an ordinary prior
  p(f) ∝ exp(κ·[f = f0]).
- **PDG** (Kelvin's sharper form). The edges are: p(Z) and p(X|Z) at β_b = 1; the observation x (hard); A, the act; the
  structural edge W = 1[A = f*(q)]; and the preference pref(W) at β_p. Splitting μ(Z) = μ(W)·p(Z | W, x) gives
  Inc(x, A) = β_b(−log p(x)) − (β_b + β_p) log Σ_w p_W^A(w|x)^a pref(w)^(1−a), with a = β_b/(β_b + β_p).
  **INC** = Inc(prior) − E Inc(x), the whole inconsistency, including the data's surprisal.
  **GAP** = the same with the surprisal term removed: Gap = min_A Inc(with preference) − Inc(without preference),
  which is what Kelvin asked for. With a hard preference (pref(W=1) = 1), Gap = −β_b log max_A P(W=1 | x), and β_p drops out.

## 3. Toy

Rooms (binary streams; the agent's prior for each room is correct, so every number is exact Bayes, by Monte Carlo or
enumeration):

| room | stream | latent |
|---|---|---|
| D | constant c | c ~ U{0,1} |
| N0 | iid fair coin, law known | none |
| N1 | iid Bern(θ) | θ ~ U(0,1) |
| S | x_t = f*(x_{t−2}, x_{t−1}) xor flip(0.05) | the world rule f* |
| S_irr | same, driven by an independent g | g |
| N_rel / N_goal | noisy labels of f* (flip 0.4 at random inputs / 0.3 at q) | f* |
| N_side | noisy labels of f*(0,0), flip 0.3 | f*(0,0), independent of f*(1,1) under the uniform prior |

Key output (full tables in `output.txt`, `pdg-output.txt`):

```
A. VOI, own-room goal (bet on own next symbol, 0-1)   n=16:  D 0.500  N0 0.000  N1 0.226  S 0.435   -> dark first
   VOI, world goal (predict f*(a,b)), n=16:                  D 0  N0 0  N1 0  S_irr 0  S 0.346  N_rel 0.152
   marginal VOI per step in S: 0.099 (n<=4) 0.025 0.006 0.002 0.0002 (n~256); capped by EVPI 0.5
B. no decision: VOI = 0 everywhere.  CUR (bits) n=4096: D 1.0  N0 0  N1 5.41 (grows (1/2)log2 n)  S 4.0 (cap)
C. scale x7: same ranking.  dark costs 0, S costs c/step: leave iff stakes m > 20.2 c
   two goals w*own + (1-w)*world: dark wins iff w > 0.842
D. trade-at-price payoff 0.3005 vs E|dP| 0.3054 (MC); signed E dP -0.005
   Q_world n=16 L1/L2/KL: S .341/.167/.648  N_rel .152/.035/.107  D,N0,S_irr 0
   Q_own_bits, total |dP| over 256 steps: D 0.5  N0 128.0  S 27.4   (noisy TV)
E1 EIG on Z=f*, n=16, kappa 0/1/3/6: S 2.63/2.55/1.85/0.26 bits  N_rel 0.43/0.44/0.31/0.04  D,N0,N1,S_irr 0.000
E2 Z drifts (hazard 0.02): S 0.122 bits/step steady state; D, N 0
E4 Friston preference updated by Bayes (Beta strength s): dark room after k* = 0.2 s forced dark steps, absorbing
F. PDG, kappa=0, beta 1/1, pref(W=1)=0.9, n=16 (nats):
            CUR     VOI     INC      GAP
   D       0.693   0       -0.693   0
   N0      0       0      -11.09    0
   N1      1.050   0       -9.05    0
   S_irr   1.800   0       -5.94    0
   N_side  0.573   0      -10.35    0
   S       1.808   0.303   -5.81    0.087
   N_goal  0.573   0.450  -10.19    0.154        argmax CUR S, VOI N_goal, INC D, GAP N_goal
   GAP vs beta_pref (n=16):  0.25: S .047 N_goal .076 | 1: .089 .154 | 4: .020 .158 | 100: -1.16 .084 ; D,N 0 always
   pref p=1 (hard), n=16: S .42 N_goal .632 at beta_pref 0.25 and 100 (beta drops out; S differs by MC noise)
   weak goal evidence (flip .45, kappa=2), n=1,2: VOI 0, GAP -0.003/-0.005 (hard pref)  -> small ostrich effect
   rest edge pref(rest)=0.6: costs 0.405 nats; S's GAP 0.07-0.11 -> stays in the dark room
```

## 4. Verdicts

**Dark room: solved with no fitted parameter, given a goal about the world, at equal costs; a transient otherwise.**
- Under a world goal, D's VOI and GAP are exactly 0 for every n, every β and every prior κ. A constant stream carries at
  most H(c) = 1 bit about anything, and here it carries nothing about f*. Any room bearing on the goal beats it, and
  scaling the goal changes nothing (×7 test).
- When rooms cost different amounts, VOI compares stakes with cost. The threshold m > 20 c is problem data, not a
  fitted rate. In the PDG form, cost is one more edge (pref(rest) = 0.6), in the same nats, so no exchange rate appears.
  The rest edge's probability still decides: with it, S's 0.07-0.11 nats lose to 0.405, and the agent stays dark.
- Value is bounded by the goal's surprisal (EVPI 0.5; GAP ≤ −log P(success)). It decays (marginal VOI 0.099 down to
  0.0002 per step). After the goal is settled every room is worth 0, as in learning progress and the Occam bounds. Only
  a goal whose world keeps changing (E2) keeps the agent out of the dark room for good.
- If the goal is about the agent's own sensations (G1: bet on your next symbol), VOI ranks dark first (0.5 against
  0.435). The full Richardson inconsistency, which keeps the data's surprisal, does the same (INC argmax D for n ≥ 4).
  Both are Friston's dark room.

**White-noise room: solved with no fitted parameter, with the same proviso about the goal.**
- N0, N1, S_irr and N_side all get exactly 0 under VOI, E1 and GAP. That holds whatever their entropy (N0: 1 bit/step)
  or their own information content (S_irr: 1.8 nats; N_side: 0.57 nats). Relevance decides, not learnability: noisy
  but relevant labels (N_rel, N_goal) are worth a lot.
- The noisy TV comes back exactly when the goal is widened to cover the stream itself. With Z = all latents, N1 grows
  without a cap and beats S after n ≈ 256. With Q = all sentences about one's own bits, noise moves prices by 128
  against S's 27.4 over 256 steps.

**Remove the decision:** VOI ≡ 0 in every room, so cost decides and the dark room wins. This is the honest cost of the
framing: VOI needs a goal, and with no goal it is silent, not wrong.

**Which of the three satisfies Kelvin's condition exactly** ("matters only insofar as it helps the goal; learning that
leaves the reconciliation unchanged is worth exactly zero"):
1. **GAP** satisfies it by construction, and the toy confirms it. It is zero for N_side, whose 0.57 nats of
   information bear only on f*(0,0), and for S_irr (1.8 nats). With a hard preference it is also free of weights.
   Caveats: it can be slightly negative for weak goal evidence that cannot change the act (−0.003 to −0.005 nats; Jensen,
   because log P is concave). So it does not inherit Good's theorem, which says that VOI is never negative.
2. **VOI** has the same zeros. But it needs an explicit decision rather than a prior component. It is also zero for
   evidence that shifts P(success) without flipping the act, where GAP is nonzero. With a hard preference, GAP is VOI
   with the utility log P(success) in place of P(success).
3. **Curiosity** fails: it pays D 1 bit, N1 without bound, and S_irr and N_side for irrelevant learning.

## 5. Hidden parameters

- **The goal itself: a choice, not a weight.** One goal needs no scalar (scale-invariant; β_pref drops out under a hard
  preference). The choice that does the work is *which variable*. A world variable avoids both rooms; one's own
  sensory stream reproduces the dark room (single-symbol forecast) or the noisy TV (all latents or all sensory queries).
- **Several goals need weights.** In the own-forecast + world-forecast mix, the dark room wins iff w > 0.842. A
  measure over queries is a vector of weights.
- **A soft preference adds p and β_pref/β_belief.** They reorder the goal-relevant rooms and can make learning the goal
  well *bad* (S at β_pref = 100: −1.16), because believing you will succeed more often than p is itself inconsistent.
  "Prefer W" should be a hard event at any confidence, not a probability.
- **"Ordinary confidence" must mean a PDG edge that stays and is reconciled, not a belief updated by Bayes on one's own
  outcomes.** If the preference is updated like a belief (E4: Beta(0.6 s, 0.4 s) over the rate of change), then after
  k* = 0.2 s forced dark steps the agent expects darkness, prefers it, and never leaves. Avoidance then needs a
  strength that grows with the time spent in the dark, which is a tuned weight. In the PDG form the edge is never
  updated. The reconciled belief bends instead (wishful thinking: μ_W(success) 0.75 against a belief of 0.5 at β 1/1),
  and Gap is computed with the belief edges, so the wishful μ does not leak into the value.
- **Myopia and horizon.** One step of VOI is 0 for n ≤ 2 in S, and the marginal value jumps at n = 3-4 (VOI is not
  concave, as Radner & Stiglitz showed). A one-step greedy VOI agent sees nothing in S's first two steps (not simulated further). The look-ahead horizon is one
  more choice, the same one that metareasoning faces.
- **Independence is in the model.** D and N get 0 because the agent's model says they are independent of f*. An agent
  that wrongly thinks noise is about the goal (astrology) will value it.

## 6. What it would mean for a reasoning network (li-net)

- The parameter-free reasoner rule is: **attend to the computation or observation with the largest expected change in
  log price of the goal sentences** (GAP with a hard preference edge on the goal), or the largest E|ΔP| on the goal
  queries for a trade-at-price trader. This is Russell & Wefald's value of computation, with the network's own market
  as the belief state. Signed price moves have zero expectation (a martingale), so the attention score must be E|ΔP|, or
  KL, or Δlog P of the goal. Those choices are scoring rules, and they agree on the zeros (all three were 0 for D, N0,
  S_irr in Part D).
- It fixes the LIR attention split (`lir2026.txt`). "Attend to inconsistency that control can reduce" becomes "attend
  to the inconsistency that the *goal edge* adds and that this step can reduce". That kills both failure modes found in
  `logical-inductor-occam`: self-posed puzzles in a dark "think" room, and the trickle of noise frequencies. Neither
  moves a goal sentence.
- What it does not give: a reason to keep thinking after the goal sentences are priced near 0 or 1, or a goal. The
  network needs a supply of goal sentences about the world, either posed or drifting (E2), and its query set must
  exclude sentences about its own sensory record. Otherwise the noisy TV returns through the back door.

## Sources

- Howard, "Information Value Theory", IEEE Trans. Systems Science and Cybernetics SSC-2(1):22-26, 1966,
  doi 10.1109/TSSC.1966.300074. The DOI resolves (302) to https://ieeexplore.ieee.org/document/4082064/; the page text
  was not retrievable, so this entry is bibliographic.
- Raiffa & Schlaifer, *Applied Statistical Decision Theory*, Harvard, 1961 (EVSI). Bibliographic only, not fetched.
- Lindley, "On a Measure of the Information Provided by an Experiment", Ann. Math. Statist. 27(4):986-1005, 1956,
  https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-27/issue-4/On-a-Measure-of-the-Information-Provided-by-an-Experiment/10.1214/aoms/1177728069.full (fetched).
- Bernardo, "Expected Information as Expected Utility", Ann. Statist. 7(3):686-690, 1979,
  https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-3/Expected-Information-as-Expected-Utility/10.1214/aos/1176344689.full
  (fetched): maximising expected information is expected utility with a suitable decision space and utility.
- Good, "On the Principle of Total Evidence", Brit. J. Phil. Sci. 17(4):319-321, 1967 (publisher page returned 403;
  bibliographic): free evidence has non-negative expected value.
- Radner & Stiglitz, "A Nonconcavity in the Value of Information", 1984 (not fetched, bibliographic).
- Kapoor, Horvitz & Basu, "Selective Supervision: Guiding Supervised Learning with Decision-Theoretic Active Learning",
  IJCAI 2007, pp. 877-882 (via search; Semantic Scholar page
  https://www.semanticscholar.org/paper/56ba05f2d72ed3bf1388a5294ce23d7c3e80128d): VOI for active learning.
- Russell & Wefald, "Principles of Metareasoning", Artificial Intelligence 49:361-395, 1991,
  doi 10.1016/0004-3702(91)90015-c (via search, not fetched): value of computation.
- Local: `2202.txt` (Richardson 2022: Inc definition; Prop. 2, inconsistency of p with X = x is the surprisal; Lemma 1,
  monotonicity), `abridged.txt` (logical induction; No Expected Net Update), `lir2026.txt`; sibling candidates
  `learning-progress.md`, `logical-inductor-occam.md`, `expected-free-energy.md`.
