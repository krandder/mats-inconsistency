# Löbian cooperation and PrudentBot in a neural, non-circuit system: what can be implemented

For Kelvin, from opus-mats (lob-neural lane, fleet request pool-wtyljg). Written 2026-11-01, 23:00-23:45Z, from
`work/lob-neural/synthesis.md` (21:59Z). Every finding of the citation check (`check_citations.md`, checks
22:00-22:40Z) and of the critique (`critique.md`, checks 22:10-23:00Z) is applied or answered; §8 lists each one. Two
results that the sibling lane energy-prudentbot (pool-22ssuc) finished at 22:35Z and 22:45Z are also used. Paths are
under `/home/agent/workspaces/mats/`, except `inconsistency-search/…` (repo HEAD 0a36abb).

Status: report done, not reviewed as a whole. Nothing was trained. The only compute behind it is the CPU checks of the
synthesis, the citation check and the critique, plus my URL checks (23:10-23:18Z). No decision is needed from you. The
toys in §5 are proposals; none has been run.

Tags:
- [known: S]: source S says it. Every external source was opened today by this lane; §7 says by which file and how
  much of it was read.
- [known: in-house F]: a proof or measurement in our own files. "Re-checked" means a second subagent checked it
  today; "unreviewed" means nobody did.
- [inference]: my reasoning. Short proofs are given in place.
- [conjecture]: untested.

## TL;DR

1. Implementable: if every action belief carries a depth index (depth 0 cooperates by default, and "proved" means "held at every smaller depth"), the energy has exactly one zero, that zero plays PrudentBot's full table once play stops changing, and the update that reaches it is a contraction, so it fits our anchored block design.
2. The catch: this is provability logic written in by hand, and the depth-0 default does all the selecting (set it to "defect" and the unique zero becomes mutual defection), so inconsistency minimisation can host Löbian cooperation but does not produce it.
3. Not implementable as things stand: without the depth index (our PDG and atom energies, the LI-network blocks, logical-inductor prices) mutual cooperation and mutual defection are equally consistent, and only ε-grounding (exploitable with probability ε), a utility term, or the search's starting point or anchor picks between them.
4. The exact guarantees need opponents given as readable templates, read to a depth set by their syntax, with exact values: at any fixed depth a deeper trickster exploits the agent; with noisy values cooperation decays with depth (one noise-tolerant variant holds it near 0.98, but whether its play always settles is open); opponents given as raw weights have no known sound reader short of simulating them; and no learned PrudentBot exists in any source checked.
5. Key open question: can a learned, fixed-size network acquire the depth structure and PrudentBot's DefectBot probe from payoffs or the energy alone, and stay unexploitable against opponents it never trained with? §5 lists four CPU toys (estimated under an hour each) that test parts of it.

## 1. Ngo's post: the claims relevant here

Richard Ngo, "Explaining Knightianism on one foot", Mind the Future, 2 September 2026 (updated 17 September), with a
LessWrong crosspost. `post.md` read the full body through the Substack API and diffed it against the crosspost; the
citation check re-fetched both of Ngo's posts the same way. Quotes are verbatim [known].

1. Other agents are the post's example of a Knightian region, a part of the world "which we can’t model or control (e.g.
   other agents smarter than us)". They break the third-person view because they are "modeling you back".
2. Schelling points and fences "involve this infinite recursion of expectations", and game theory "can’t talk about how
   agents reach equilibria, only what they do once they’re already there".
3. Reflective oracles are the most direct attack so far, but "appealing to a reflective oracle seems to be dodging the
   hard part of the problem".
4. The proposal: "combining something like Garrabrant induction with something like Lobian cooperation". Garrabrant
   induction is "a way for agents to form beliefs about (potentially self-referential) mathematical facts, like the
   outputs of other agents"; Lobian cooperation is "a way for agents to “cut through” the infinite recursion involved
   in modeling each other".
5. "The immediate blocker is that Garrabrant inductors have no way of taking actions". He takes "some steps towards
   resolving this" with his belief-webs framework.
6. An action is "a belief which you expect an external actuator to be “watching” and trying to make come true".
7. Predictability is partly chosen: "when you’re partially transparent to your environment, how you make a decision
   can change which action is utility-maximizing".

Apart from the two uses of "Lobian cooperation", the post never mentions PrudentBot, FairBot, Löb's theorem, modal agents
or program equilibrium, and it gives no construction, theorem or example for the combination [known: post.md §2.1, a
search of the post, its 4 Substack comments, the crosspost and its 28 LessWrong comments]. PrudentBot enters through a link chain: the post links "Agents
as webs of beliefs", which links arXiv:1401.5577, the paper that defines PrudentBot.

The linked belief-webs post adds three points used below [known: post.md §2.4, read in full]:
- Its one passage tying Löb to the framework concerns a single agent. Acting may need higher-order beliefs such as "if I
  believe that ‘if I believe X, then X will happen’, then if I believe X, then X will happen", and Ngo writes: "This
  feels quite Lobian, and makes me wonder whether something like Lobian cooperation will be necessary to ground it all
  out." (§4.5)
- Goals act as "drives" that pull goal credences up, "anchors" hold empirical credences in place, and the web
  equilibrates when the two balance (§4.4).
- Belief webs "might just get stuck in local equilibria" and, unlike FixDT, have no way to jump to the best one.

[inference] The proposal is an instance of an open problem that arXiv:1401.5577 itself lists: "What happens if we apply
probabilistic reasoning rather than provability logic?"

## 2. FairBot, PrudentBot, and what Löb's theorem does

Source: Bárász, Christiano, Fallenstein, Herreshoff, LaVictoire and Yudkowsky 2014 (A2), arXiv:1401.5577v2, read in full
by `post.md` and `lit.md`; the statements below were re-opened by the synthesis and the citation check [known].

The game is a one-shot Prisoner's Dilemma between programs that read each other's source code, with payoffs
T > R > P > S. [X(Y)] means "X cooperates when it plays Y", □φ means "PA proves φ", and PA+1 is PA + Con(PA), that is,
PA + ¬□⊥.

- CooperateBot (CB) always cooperates; DefectBot (DB) always defects.
- FairBot (FB) cooperates iff PA ⊢ [X(FB)]. Modal form: [FB(X)] ↔ □[X(FB)].
- PrudentBot (PB) cooperates iff PA ⊢ [X(PB)] and PA+1 ⊢ ¬[X(DB)]. Modal form: [PB(X)] ↔ □[X(PB)] ∧ □(¬□⊥ → ¬[X(DB)]).
- Löb's theorem for PA (A2 Thm 1.1 with S = PA): if PA ⊢ □φ → φ, then PA ⊢ φ.

Two FairBots with different code cooperate (Thm 3.1). With A = [FB₁(FB₂)] and B = [FB₂(FB₁)], the code gives
PA ⊢ □A → B and PA ⊢ □B → A, hence PA ⊢ □(A ∧ B) → (A ∧ B), and Löb gives PA ⊢ A ∧ B. FairBot is unexploitable if PA
is sound, and it also cooperates with CooperateBot.

What Löb does here is choose between outcomes. Without the boxes, "A ↔ B, B ↔ A" is satisfied by both CC and DD. The
paper's footnote 3: "the symmetry between mutual cooperation and mutual defection is broken by the positive criterion
for action". With bounded proof search, "cooperate unless proof of opponent defection" (CUPOD) gets (D, D) against
itself while the bounded FairBot (DUPOC) gets (C, C) [known: Critch, Dennis and Russell 2022 (B2), Thms 3.4 and 3.7].

Levels. Play between modal agents reduces to closed formulas of the provability logic GL, and A2 derives outcomes
"using the properties of Kripke semantics" (§4, Thm 4.7) [known]. On the chain of levels n = 0, 1, 2, …, □φ holds at
level n iff φ holds at every level below n, so every □φ holds at level 0. A closed formula is constant from some level
on, and that constant is its truth value when PA is sound [known: in-house formal_logic Prop 2(b), proved from Boolos's
results via Buehler's notes; re-checked in verify_math]. In this picture Löb's theorem is induction from a vacuous base:
p ↔ □p is true at level 0, hence at every level, while CUPOD's c ↔ ¬□¬c is false at level 0, hence at every level
[inference: direct evaluation].

Why the DefectBot probe runs in PA+1. [FB(DB)] is □⊥, so "FB defects against DB" is Con(PA), which PA cannot prove. A
probe run in PA would fail against FB and against PB itself, and PB would defect against both (A2, remark after
Thm 3.2) [known]. In levels, ¬□⊥ holds exactly from level 1, so PA+1 means "the probe skips level 0", where every
FairBot cooperates with DB vacuously [known: in-house formal_dynamics Thm 1(d)]. Evaluated this way, PB–PB and PB–FB
cooperate at every level, PB–CB defects from level 2 (A2: "PA+2 ⊢ [PB(CB) = D]"), and a probe that reads level 0 makes
PB defect against FB and against itself from level 1 [known: synthesis §6 check 1, rerun independently by the citation
check and the critique].

Why a third-party probe at all. "Any modal agent X of rank 0 such that PA⊢[X(FB)=C] must also have PA⊢[X(CB)=C]"
(Thm 4.10) [known]. Rank 0 means the agent only asks how the opponent plays against the agent itself. To defect against
CooperateBot while cooperating with FairBot, an agent has to ask how the opponent plays against someone else.

§3 scores six parts: (a) access to the opponent, (b) the self-referential fixed point, (c) the selection of CC, (d) the
DefectBot probe, (e) soundness and the PA/PA+1 stagger, (f) learning any of it.

## 3. Component by component

### 3.0 Terms

"Implements PrudentBot" can mean three things; §3.1 scores each mechanism against all three:
- the table target: the same outcome as A2 for every pair of modal agents;
- the equilibrium target: PrudentBot's properties in A2 Thm 3.2, cooperating with itself and with FB and never being
  suckered;
- the exploitation target: defecting against CB.

Exploitability is measured at actual play, meaning what each side outputs in the match as implemented: the action at
the read-off level it uses, or the sampled action.
- Exact: no opponent in the class gets (C, D) against me.
- Probabilistic: for every opponent in the class, P(I play C and the opponent plays D) ≤ ε.

"Neural, non-circuit" is taken operationally:
- (N1) behaviour is the end state of an iterated update (a recurrent pass, or descent on an energy), not a written
  program;
- (N2) values are soft or noisy;
- (N3) compute per pass is fixed in advance;
- (N4) opponents may arrive as weights rather than as readable templates;
- (N5) the update or the decision rule is trained.

The post "What if not Circuits?" motivates N2 with "noisily writing algorithms to noisily operate on noisy
data", but its noise is parameter drift during training, and it says nothing about self-reference. Applying it to noise
at inference time is my reading [inference].

### 3.1 The table

| Part | PrudentBot | Best substitute without proofs | Under N1-N5 | Evidence | What breaks |
|---|---|---|---|---|---|
| (a) access | two provability queries about X's code: X against me, X against DB | run X's program; or instantiate X's readable clause template against me and against DB | works for templates and runnable programs; for opponents given as weights (N4), simulating X's searcher is the only reader known to be sound | C1-C3 [known]; formal_logic §1.1 [known: in-house] | an energy with several zeros does not fix behaviour; a similarity score has no slot for a third party [inference] |
| (b) fixed point | quining; play reduces to a closed GL formula | a joint fixed point of edges that read each other | exists; unique with a level index or a contraction, and the level sweep is itself a contraction | Brouwer [inference]; formal_logic Prop 1 [known: in-house, re-checked] | without levels, CC and DD are both zeros |
| (c) selecting CC | Löb plus the positive criterion | exact: level index, cooperative base at level 0, □ = "every lower level"; approximate: ε-grounding; otherwise a utility edge or the starting point | the exact version runs as a recurrent pass once the base is written in; no learned or level-free system produces the selection | formal_logic Prop 3 [known: in-house, re-checked]; D2, C1 [known]; Lemma 1 [inference] | level-free inconsistency minimisation, LI prices and naive learning select nothing |
| (d) DB probe | PA+1 ⊢ ¬[X(DB)] | instantiate X's template with DB; or run X against DB with the probe grounded separately, or test X's exact probability against θ ≤ 1−ε | works with templates or exact simulation; a single-sample grounded probe leaks | A2 Thm 4.10 [known]; the 1/(2−ε) leak [known: in-house, re-checked] | no learned probe in the sources checked |
| (e) soundness | PA sound; probe one level stronger | probe from level ≥ 1; a larger proof budget (B3); θ ≤ 1−ε; for LI, finitistic consistency | exact only for template opponents, read at a depth from their syntax, with crisp values; fixed depth (N3) and noise (N2) break it | formal_logic Prop 5 [known: in-house, re-checked]; B3 Lean theorems [known]; LI Thms 4.9.2-4.9.3 [known] | any fixed read-off depth N is suckered by a deeper trickster |
| (f) learning | none: written by hand | opponent-aware learning with imposed grounding (E1); pretraining (E2); coupling built into the environment (E3) | cooperation with the co-trained partner only; nothing tested off the training pair | E1-E3 [known] | no learned PrudentBot |

Against the three targets of §3.0 [inference, from the sections cited]:
- The level energy, read at N ≥ K with crisp values against template opponents, meets all three (§3.6).
- ε-grounded FairBot (εGFB, NicerBot) meets the equilibrium target up to ε and misses the exploitation target, since it
  cooperates with CB. The table target does not apply, because it plays programs and not modal formulas (§3.3).
- ε-grounded PrudentBot with an exact-probability probe meets the equilibrium target up to ε and the exploitation
  target; with a sampled probe it misses the equilibrium target, since it cooperates with its copy with probability
  about ½ (§3.5).
- A similarity score gives cooperation among near-copies, with unexploitability only as good as the score; it has no
  probe for the exploitation target (§3.2).
- Level-free energies and LI prices admit both CC and DD, so they meet no target without an extra selector (§3.3,
  §3.4).
- Learned agents (E1-E3) are shown to cooperate with their co-trained partner; no target has been tested off the
  training pair (§3.7).

### 3.2 (a) Access to the opponent

PrudentBot gets exact source code but uses it only through two provability queries; modal agents are "behavioral",
their action provably depending only on the opponent's behaviour (A2 Thm 4.8) [known]. A substitute has to say what X
does against me and what X does against DB (X with a substituted opponent), and both answers must be true of X's
actual play [inference, from A2 Alg. 5].

- Simulation answers both, exactly up to private randomness [known: C1, C3]. ε-grounding makes the mutual recursion stop
  after O(1/ε) calls with probability 1 [known: E1 §1].
- A readable energy definition (your "access to each other's 'energy function definitions'"). Write each agent's energy as
  a clause template with a slot for its opponent, and instantiate it against me and against DB [known: in-house
  formal_logic §1.1]. This fixes behaviour only when the zero is unique and does not depend on the searcher. With a level
  index it is unique (Prop 1, §3.3). Without levels the definition admits both CC and DD, and two FairBots with opposite
  tie-breaks end at (C, D) [known: in-house formal_logic Prop 5(d)]. Reading the definition is then not enough; one also
  needs the other's searcher, which is source code again [inference].
- A similarity score. E2's abstract: it "allows for the same set of cooperative outcomes as the full transparency
  setting", and "cooperation can be learned using simple ML methods" [known: E2]. The score is a function of the pair
  actually playing, with no slot for a third party, so it cannot answer the DefectBot query and has no PrudentBot
  analogue [inference].
- A learned predictor (LI prices, or E5's "similarity inference"). The queries become questions to the predictor. LI's
  calibration holds along efficiently computable sequences of sentences [known: LI Thm 4.3.3], which says nothing about
  one game [inference]. E5's dependence measure is a belief about the other agent, not a verified conditional
  [inference, from E5's definitions].
- A network reading a network. Full transparency "requires at least one of the models to receive as input a model that
  has at least as many parameters as itself" [known: E2 §1]. A network cannot take its own weights as input at full
  size. It can take a compressed self-description (architecture and seed, or the program that generates the weights),
  which is how programs quine [inference]. In the sources checked, policy-conditioned policies exist only as code read
  by an LLM, and only for coordination games [known: E6 abstract].
- For opponents given as weights, running the opponent's searcher with ε-grounding is a sound reader (the simulation
  route). No sound reader is known that avoids simulating the opponent's search (Q3) [inference].

### 3.3 (b) The self-referential fixed point

Edges that read each other turn the self-reference into one joint constraint. Continuous edges on a product of
simplices have a joint fixed point [inference: Brouwer]; reflective oracles answer some queries randomly to avoid
diagonalisation for the same reason [known: D1]. Uniqueness depends on the setting.

- Level-free: beliefs read beliefs at the same time. FB–FB, PB–PB and PB–FB each have CC and DD as zero-energy states
  [known: in-house formal_logic Prop 4(a), formal_dynamics §0.1]. With an entropy term (γ > 0), FB–FB's minimiser gives
  each player P(C) = ½ and P(CC) = P(DD), 0.4944 each at ε = 0.01 and γ = 0.05, tending to ½CC + ½DD as the edges become
  hard. PrudentBot pairs are not symmetric and lean to DD: P(CC) = 0.375 and P(DD) = 0.565 for two differently written
  PrudentBots (PB–PB2) at η = 0.05
  [known: in-house formal_logic Prop 4(b), formal_market T2]. A soft threshold edge x = σ((y − ½)/τ) between copies has
  three symmetric zeros iff τ < ¼, since its slope at ½ is 1/(4τ); at τ = 0.1 they are 0.0072, 0.5 and 0.9928
  [inference; roots recomputed by the synthesis and the citation check].
- Levels: level n reads only levels below n. The energy has one zero for every choice of positive weights, computed by
  one forward pass [known: in-house formal_logic Prop 1; proof re-checked line by line in verify_math].
- Contraction. For NicerBot or εGFB copies, x = ε + (1−ε)y is a (1−ε)-contraction with the unique fixed point (1, 1)
  [inference: Banach; the algebra is in D2]. The uniqueness comes from the contraction: with a threshold response
  x = ε + (1−ε)·[x > t], both x = ε and x = 1 are fixed points whenever ε ≤ t < 1 [inference: substitution].

Levels and contraction are not alternatives: the level sweep is itself a contraction (§3.4).

### 3.4 (c) Selecting cooperation

The exact substitute is a level structure [known: in-house formal_logic Prop 3, formal_dynamics Thm 1; both directions
of the Löb lemma re-checked in verify_math]. Index the action beliefs by level, let □χ at level n mean "χ at every lower level" (true at
level 0), and read off at or above the level where play stops changing. Löb's role is then played by "if □χ → χ holds
at every level, χ holds at every level", proved by induction from the vacuous base. The zero-energy table is CC for
FB–FB, PB–PB and PB–FB. Ablations: with no levels there are two minima, with cyclic levels 2 or 4, and with nothing
certified at level 0 the table is DD.

The base does the selecting. Replace the level-0 rule by "an empty AND is false" and every uniqueness and
search-independence theorem still holds, with the unique zero DD for FB–FB, PB–FB and PB–PB [known: in-house
formal_logic recorded output, "base=F"; verify_math problem 3]. Uniqueness is therefore no evidence of cooperation. The
cooperative choice is the level-0 unit clause. Level 0 plays the role of GL's bottom world, where every □φ holds (□⊥
included), and the induction up from that world is the semantic content of Löb's theorem. An inconsistency energy can carry this choice once
it is written in; minimising inconsistency does not supply it [inference].

The sweep is a contraction [inference; the proof is the critique's F1 and the citation check's P1]. Take the whole
level table as the state, and let one block set every entry to its clause value computed from the current table's lower
levels, with min, max and 1 − x as connectives and □ = min over lower levels (1 at level 0). Each level-n entry is
1-Lipschitz in the entries below n. So for every λ in (0, 1), in the norm ‖d‖_λ = max over atoms q and levels n of
λⁿ|d(q, n)|, the block is a λ-contraction, and the base is a constant inside it: an anchor. Its fixed point is the
Kripke table, reached exactly after N + 1 sweeps from any start. Measured: 7 matches with 30 starts each reach the exact
table within 13 sweeps at N = 12, and the largest ratio at λ = ½ is 0.500 [known: critique check 2, unreviewed].

Lemma 1, restated after the critique (F7) [inference; proofs here]. Let my probability of cooperating be x = r(q), a
continuous function of the opponent's probability q of cooperating against me, with private coins.
1. Exact unexploitability in A2's sense forces r ≡ 0. Against an unconditional opponent that cooperates with probability
   q, P(I play C, it plays D) = r(q)(1 − q), so r = 0 on [0, 1) and, by continuity, at 1. The only exactly
   unexploitable continuous rule of this kind is DefectBot, and such rules can reach at best ε-unexploitability.
2. Take the weaker "never out-earned": U_me − U_opp = (T − S)(q − x) ≥ 0 for every q, that is, r(q) ≤ q. Then (0, 0)
   is a fixed point, and the simultaneous iteration x ← r(y), y ← r(x) never reaches or approaches (1, 1) from any other
   start, because min(x, y) never increases.
3. Reading the opponent's cooperation d against DefectBot does not help. "Never out-earned" forces r(0, d) = 0 for
   every d, so a copy's cooperation against DB is r(0, 0) = 0, and (0, 0) is still a self-play fixed point.

The obstruction is a rule that reads only current probabilities, with no level structure. ε-grounding escapes it by
setting r(0) = ε, at ε-exploitability. The level model escapes it because the rule at level n reads the levels below n.

Other selectors:
- ε-grounding: a unique cooperative fixed point, at ε-exploitability [known: D2, C1].
- A utility edge. If utility may move only my own action (the control mask), CC is each agent's unique minimiser.
  Without the mask, FairBot cooperates with DefectBot once λ > β_DB·logit(1−η)/(R−P), and selecting by one's own
  preference is exploitable by a spiteful FairBot-like opponent [known: in-house formal_market T5 and §2.4, unreviewed].
  The literature has Pareto selection among reflective-oracle fixed points (D3, D4; D4 omits its proofs) [known] and
  FixDT's utility choice, which Demski calls "little better than what EDT offers" for lack of a learning story [known:
  Demski 2023b].
- The starting point or anchor. In a level-free energy the search's start decides: for the soft threshold edge of §3.3
  the zeros near 0 and near 1 attract and the one at ½ repels [inference: li.md §5.1]; E2's best-response learners need
  pretraining, and E3's jointly learned semantics a warm start (§3.7) [known].

Reaching the selected zero is a separate problem. On the level energy, whose zero is unique, full mean-field gradient
descent reaches the exact table from 0-13.5% of random starts in FB–FB, PB–FB and PB–PB, and single-flip descent from
0-21.3%. Updates that move each entry toward its clause value with the lower levels held fixed (target-only updates, the
control mask) reach it from 100% [known: in-house sibling toy, energy-prudentbot/toy_result.md §0, run 22:38-22:45Z,
unreviewed]. The crisp energy has strict single-flip local minima, 2(N − 1) of them at energy 1 for FB–FB [known:
in-house, re-checked in verify_math]. So with one belief per atom and level, a searcher reaches the unique zero only if
its updates follow the level order [inference].

Logical induction and markets:
- Price FairBot between copies is a truth-teller sentence, φ ↔ P(φ) > p. Its limits 0 and 1 both leave every trader
  without profit, and no LI theorem covers it [inference: li.md §5.1, formal_market §2.7].
- Garrabrant claims, informally, that the logical-inductor EDT agent with ε-exploration will "defect in a prisoner's
  dilemma against a similar power agent that is trying to imitate it", because "random defection will be uncorrelated
  with the opponent's prediction" [known: D7]. The PD behaviour of other LI decision rules is open; for policy
  selection Demski writes "I don't know what space of behaviors is really possible in this case" [known: Demski 2017].
  In repeated games, LI agents of similar strength converge to correlated equilibria under an independence assumption
  [known: Diffractor 2018], and in the PD the only correlated equilibrium is DD, since D strictly dominates C (standard).
- Löb fails for probabilistic belief [known: Demski 2023a; Armstrong 2013]. Payor's lemma, "If ⊢ □(□x → x) → x then
  ⊢ x", needs only "necessitation and distributivity" [known: Critch 2023], but its probabilistic version rests on
  assumptions its author has not confirmed [known: Demski 2023a], and an LI lacks exact necessitation at finite n,
  since a provable φ gets P_n(φ) → 1 only in the limit [inference].
- Garrabrant on moving from proofs to predictions: it "lost the silver lining of Lob's theorem: the Lobian handshakes
  which powered modal combat" [known: D5].

### 3.5 (d) The DefectBot probe

- Necessity among modal agents: A2 Thm 4.10 (§2) [known].
- Energy: instantiate the opponent's template with DB; the DB sub-matches become extra atoms [known: in-house
  formal_logic §2.2].
- Simulation. A naive ε-grounded PrudentBot that samples X(DB) once inherits X's grounding coin, and cooperates with its
  own copy with probability 1/(2 − ε): 0.526 at ε = 0.1, against Monte Carlo values of 0.525 and 0.528 [known: in-house
  formal_market §0.7, derived and Monte Carlo-checked; the algebra re-checked in verify_math]. Two fixes: ground only the
  main recursion [known: in-house, unreviewed]; or test the exact probability P(X(DB) = D) against a threshold
  θ ≤ 1 − ε, which at ε = 0.1 gives self-cooperation 1 for θ = 0.85 and 0.1 for θ = 0.95 [inference, by analogy with C2
  Prop 10]. The exact probability is needed: C2 obtains it inside PA by enumerating random bits, and a single-sample
  simulator cannot.
- C2 Prop 10 itself concerns the proof-based PBθ against εGFB: the two cooperate for θ below 1 − ε. At the boundary
  θ = 1 − ε the statement and the proof disagree (the statement puts it on the defect side). C2's compatibility theorem
  covers DUPOC, CIMCIC, PBθ and εGFB [known: C2 p.6, checked by the citation check].
- LI. For an unbounded FairBot the probe sentence "X(DB) = D" is Con(PA) [known: A2], and Non-Dogmatism keeps
  P∞(Con(PA)) < 1 [known: LI Thm 4.6.2], so an LI PrudentBot needs θ < 1. How exploitable that leaves it depends on the
  inductor's prices on false sentences it has not yet refuted, not on 1 − θ alone [conjecture].
- Learned probes: none in the sources checked [known: absence over E1-E7]. Garrabrant's ProbabilisticPrudentBot is the
  only proof-free version with a stated, if informal, analysis; he calls the post "very hand-waivey" [known: D5].
  Oesterheld describes adding a CooperateBot check to εGFB and calls such checks "a trick or some hacky thing" [known:
  C6, AXRP 49 near 01:37]. A utility edge plus a dependence edge read from the other's definition gives PrudentBot's
  rule ("C iff the other answers C with C and D with D") against fixed-rule opponents, and becomes circular against
  agents of its own kind [known: in-house formal_dynamics Prop 10(b)-(c); (b) re-checked in verify_math].
- Limits even in the ideal case. Oesterheld calls PrudentBot "much more hacky" and says its check "is quite different
  from this 'does my defection make the opponent defect?'"; Filan: "it's both the wrong counterfactual and it's a
  little bit less strategic" [known: C6]. JustBot is exploitable but no modal agent exploits it, and no agent is optimal
  [known: A2 §5]. With modelling costs, among the four agents CB, DB, FB and PB, PrudentBot's exploitation of CB is what
  supports a mixed equilibrium; the author's own simulation finds it unstable, and he marks the post as a draft for
  others to check [known: B5].

### 3.6 (e) Soundness, and PA against PA+1

PrudentBot's unexploitability "is again immediate from the definition of PrudentBot and the assumption that PA is
sound" (A2, proof of Thm 3.2) [known]. Each substitute moves the probe one step up:
- Levels: "PA+k proves it" becomes "stable from level k", and PA+1 becomes "skip level 0". Without the skip, PB–FB and
  PB–PB end in DD [known: in-house formal_dynamics Thm 1(d)].
- Bounded proofs. In B3's Lean formalisation (github.com/ColombanD/open-source-game-theory at commit 5a1bc15),
  same-budget PrudentBot against itself is (D, D) at every budget, because "a same-strength prover cannot certify its own
  failed search"; `PrudentBot2 k (4k+100)` against itself is (C, C) for all large k; PrudentBot against CooperateBot is
  (D, C); and `PrudentBot (2k+64)` against `DupocBot k` is eventually (C, C) [known: B3 theorem statements; no `sorry`,
  `admit`, `axiom` or similar in the 29-file import closure; the library was not built]. The calculus takes the bounded
  derivability conditions as primitive rules with costs from "Critch's literal cost model", so the result holds for
  those postulated costs; whether they match proof lengths in bounded PA is not shown [known: critique F20]. For bounded
  PA the question is open (B2 OP9), and the published bounded Löb proof has a reported gap [known: B4].
- ε-grounding: θ ≤ 1 − ε, with exact probabilities (§3.5).
- LI: P_n(Con(Γ)("f(n)")) ≂ 1 for every computable f, and likewise for any consistent stronger theory [known: LI
  Thms 4.9.2-4.9.3]. This is only finitistic, since P∞(Con(PA)) < 1 [known: LI Thm 4.6.2].

In every case unexploitability is only as good as the evaluator is faithful: a sound proof checker, exact simulation
(up to ε), or the level read-off [inference].

Exactness needs three conditions [known: in-house formal_logic Props 5(a), 5(b) and 6(a), formal_dynamics Prop 9(b);
critique F6]:
1. the opponent is a modal formula given as a readable template;
2. the read-off depth comes from its syntax: at least K, the number of distinct boxed subformulas, or until the vector of
   box values repeats (both in formal_logic's encoding, where □₁ψ is □(¬□⊥ → ψ); with □₁ as a primitive, the repeat
   test must start at level 1 or it stops too early [known: in-house verify_math problem 1, toy_result.md §0 item 3]);
3. values are crisp, so that a repeated box vector certifies that play has stopped changing.

Under these conditions, an agent of the form □p_me ∧ ψ (FairBot and PrudentBot have this form) read at N ≥ K is never
suckered by any modal opponent, at whatever level the opponent reads [known: in-house formal_logic Prop 5(a); proof
re-checked in verify_math; no sucker case in 500 or in 1,000 random match-ups]. The guarantee covers only that form:
CooperateBot is definable and exploitable (Prop 5(c)). Each condition fails in the neural setting:
- A pass of fixed length (N3) breaks condition 2. The trickster Z_N, "cooperate iff □ᴺ⊥", cooperates below level N and
  defects from level N on, so FairBot read at level N plays C while Z_N plays D, for every fixed N [known: in-house
  Prop 5(b); re-checked for N = 1..5]. εGFB has no fixed top level, because it randomises its depth, and it stays
  ε-exploitable [known: in-house formal_dynamics Prop 9(b); C1]. At a fixed depth, then, the level read-off's worst case
  (suckered with probability 1) is worse than εGFB's (ε); a comparison at equal compute is open (Q8).
- Soft or sampled values (N2) break condition 3; see the noise results below.
- Opponents given as weights (N4) break condition 1.

Noise:
- Sampled noise. If each entry flips with probability ε after it is computed, the AND box makes one flip permanent, and
  cooperation decays with depth. For FB–FB at ε = 0.01, P(CC) is 0.942 at N = 2, 0.359 at N = 50 and 0.018 at N = 200;
  one player's P(C) is 0.961, 0.372 and 0.028 [known: exact computation in critique check 5 and citation check P3].
  PB–FB is at 0.844 at N = 5 and PB–PB at 0.711 at N = 8 [known: in-house formal_logic recorded output].
- Deterministic soft gates x_n = σ(β(□_n − ½)) at β = 10 decay with a product AND (0.993 to 0.007 by level 50) and do
  not decay with a min AND (0.993 through level 200) [known: synthesis §6 check 4, rerun by the citation check]. The
  decay comes from sampling or from a non-idempotent conjunction.
- A thresholded soft read-off can be exploited by an opponent that publishes FairBot's formula but keeps a higher private
  flip rate ε_Z; it gains up to ε_Z(T − R) per game [known: in-house verify_math problem 2, unreviewed].
- Other boxes, under sampled noise at ε = 0.01 [known: critique checks 4-6, unreviewed]. "AND over the last w lower
  levels" decays like AND (within 0.001 for w = 2 and 8). "Held at a fraction ≥ ρ of lower levels" keeps □ᵏ⊥ true for
  2ᵏ − 1 levels at ρ = ½, so any fixed read-off is suckered by a deep enough trickster. The windowed vote "held at
  ≥ ρ of the last w lower levels" at (w, ρ) = (8, ½) keeps P(CC) near 0.98 from N = 8 for FB–FB, PB–PB and PB–FB, keeps
  PrudentBot's defection against CB, and was never suckered at its stabilisation level in 1,800 random pairs. But 88 of
  those pairs had no stable read-off by level 280, and 198 stabilised to an outcome different from GL's.

A soundness lemma for the whole family (the critique's F14) [inference; proof here]. Call a box B admissible if (P1) it
is true at level 0, (P2) it is true at level n whenever χ held at every lower level, and (P3) it is false from some level
on whenever χ is false from some level on. AND, windows, fraction boxes and windowed votes are all admissible. Claim: if
FB (or PB, through its first conjunct) and a modal opponent Z both stabilise, and FB's stable action is C, then Z's
stable action is C. Proof: if Z's stable action were D, "Z cooperates with FB" would be false from some level on; by P3
FB's box would be false from some level on, so FB's stable action would be D. ∎ The Löb lemma's induction uses only P1
and P2, so every admissible box also keeps FairBot copies cooperating [inference]. For agents of FairBot's and
PrudentBot's form, soundness at joint stabilisation is therefore settled for the whole family. Open: whether pairs
stabilise, how deep, and whether stabilisation can be certified. Windowed operators have finite state (the last w
values of each boxed argument), so a repeated state certifies a cycle; fraction boxes keep counters and have no finite
certificate.

### 3.7 (f) Learning

- E1 [known: arXiv:2012.02671, re-opened by the critique and the citation check]. The grounding is imposed: "we require
  that both p_P,A ≤ 1 − ε and p_P,B ≤ 1 − ε", with ε = 10⁻³. Play is computed in closed form from four probabilities
  per agent. Plain gradient learning fails ("simple gradient learners will always learn mutual defection"), while with
  LOLA or SOS "the combination of mutually transparent decision-making and opponent-aware learning robustly leads to
  mutual cooperation in a single-shot prisoner's dilemma". The learned policies "end up more forgiving than
  εGroundedFairBot (i.e., both have Pr[C|D] > 0)", "the learning dynamics have not reached an equilibrium", and no
  learned policy is evaluated against an opponent outside its pair. Against DefectBot such a policy cooperates with probability
  (1 − Pr[S])·Pr[C|¬S] + Pr[S]·Pr[C|D], where Pr[S] is the probability of responding to the simulated opponent; this
  exceeds ε whenever Pr[C|D] is not small [inference].
- E2 [known: arXiv:2211.14468 §6.3 and App. G]. With alternating best-response learning, 0 of 26 runs reach partial
  cooperation without "Cooperate against Copies and Defect against Random" (CCDR) pretraining, and 26 of 28 do with it.
  LOLA "sometimes succeeds" without pretraining, though the authors add that "none of our LOLA results are nearly as
  robust". The learned policies are thresholds on the similarity score.
- E3 [known: arXiv:2606.27068 Thm 3 and §3.3]. With fixed sigmoid semantics p_i = σ(θ_i + γθ_{−i}), selfish gradient
  ascent from θ = 0 moves toward cooperation iff γ > γ* = (T + P − R − S)/(R + T − P − S), which is 3/7 for
  (T, R, P, S) = (5, 3, 1, 0). With jointly learned semantics a cold start does not reliably get there: "the threshold
  is not merely a representational condition, but also acts as an optimization barrier". The coupling is supplied by
  the environment; no player can refuse it or test the other's, so unexploitability is not addressed [inference].

So naive learners defect unless the environment hard-wires enough coupling. Learned cooperation so far is cooperation
with a co-trained partner, the one εGFB-like result had the grounding imposed, and no learned PrudentBot exists in the
sources checked [inference, from the three items above].

## 4. Our logical-inductor-like mechanism: what it would need

### 4.1 The mechanism as written down

Sources: li.md §§1.2-1.3, which quotes `outbox/research-direction-note-v2-2026-10-16.md` lines 14, 47 and 51, and
`inconsistency-search/rounds/r4-r9/li-net/math/block-dynamics.md` R4 and R9 [known: in-house].
- A world model defines an inconsistency energy E(μ): PDG edges, axioms at infinite weight, a utility edge and an
  entropy term −γH(μ). In games, each agent's action carries the KL to its soft best response.
- A learned, weight-tied block moves μ toward low E. With γ > 0 the minimiser is unique, so the search changes speed and
  not the target.
- Depth extrapolation. R9(i) recommends injecting the anchor q₀ "so the target is one fixed point". R9(ii): if the
  anchored block is a ρ-contraction, the error after any T ≥ 20 blocks is at most the 20-block error plus 2ρ²⁰ times
  the start's distance from the fixed point. R9(iv): an end-only loss at 20 blocks does not determine block 500. The
  un-anchored self-referential block (R4) has every coherent point as a fixed point.
- Your ruling, quoted via li.md §1.1: "I'd rather the thing evolves to find the same fixed point than for traders to
  actually have to attend to past days."
- No document gives a falsifiable LI-type statement of the hypothesis (li.md §1.3).

### 4.2 What it hosts as written

- In the current multi-agent energy (KL to soft best responses), every zero is a quantal response equilibrium. In the
  one-shot PD, Q(D) − Q(C) = (T − R)q + (P − S)(1 − q) > 0 for every opponent mix q, so at every zero each player
  defects with probability above ½ [inference: lit.md §4 T1, re-derived in li.md §5.1]. Giving each agent the other's
  energy definition adds information but no strategies, and strict dominance does not depend on information, so this
  alone changes nothing [inference]. The program and oracle results all rest on agents choosing conditional policies
  that the other side reads; E5 and D6 reach one-shot cooperation through believed similarity instead [inference].
- With a reader edge of that kind, ε-grounded FairBot is a (1 − ε)-contraction with the unique zero CC, compatible with
  the unique-minimiser target and inside R9(ii)'s sufficient condition. An ε-grounded PrudentBot needs its probe grounded separately or tested
  on an exact probability (§3.5). Both are ε-exploitable [inference, from §3].

### 4.3 What exact Löbian selection would need

From in-house formal_logic §4, corrected as §8 lists (F1, P1, F6) [known: in-house; corrections are inference]:
1. each agent's definition as a clause template that can be instantiated against the other agent and against DB;
2. a level index on action beliefs, one belief per action atom per level 0..N;
3. □ = "held at every lower level", true at level 0: the cooperative base, written in;
4. the DB probe reading levels ≥ 1;
5. a read-off depth taken from the match's syntax, N ≥ K;
6. crisp values, or an admissible box (§3.6) with certified stabilisation (Q2);
7. a searcher whose updates follow the level order (§3.4).

The levels can sit in two places, which behave differently in our block design (critique F1) [inference]:

| | (A) levels in the state | (B) levels as blocks |
|---|---|---|
| state | one belief per action atom and level; grows with N | fixed size, plus a running AND over blocks |
| target | one zero for every weight, independent of the search | greatest fixed point from an optimistic start; depends on the start and the update |
| one block | anchored λ-contraction for any λ < 1; exact after N + 1 sweeps | monotone; not a contraction |
| R9(ii) | applies, to more blocks at a fixed N | does not apply |
| memory of past blocks | none; each block reads the current table | the running AND |
| extrapolation means | a bigger state: more levels, larger closures | more blocks at a fixed size |

So the block design can host (A) and ε-grounding inside R9(ii)'s sufficient condition, since their exact updates are
contractions, and it can host (B) only outside it [inference]. The synthesis concluded that only ε-grounding fits
because it identified levels with blocks, which is (B).

What (A) costs: N + 1 copies of every action atom; a hand-written base that does all the selecting; and, whenever N is
fixed in advance, exploitability by tricksters deeper than N (§3.6). The extrapolation that matters is to deeper
opponents, which needs a bigger state. That is the same kind of test as our sudoku and N-Queens size transfer, on an
easier problem: each level is a function of the levels below it, with no branching or backtracking [inference]. In (A)
the levels are certification depths held in the current state, not past days, so I read it as compatible with your
ruling; you may read the level copies as history.

### 4.4 What the logical-induction side adds

- LI gives beliefs, not actions. The paper says it is "not a decision-making algorithm" (full.txt:233-234), and Ngo
  calls this "the immediate blocker" [known].
- Prices alone do not choose between the copy limits 0 and 1, the truth-teller of §3.4 [inference].
- In a finite-trader market built to the synthesis's X3 specification, the price convention decides. Taking the lowest
  valid price gives the copy-day limit 0 in 30 of 30 runs, the highest gives 1 in 30 of 30, the valid price nearest ½
  holds it at ½, and plain bisection gives a 2-cycle with cooperation on half the copy days [known: critique check 7,
  unreviewed]. Selection then comes from the market maker's convention, the "house view" of li.md §2.4 [inference].
- An LI PrudentBot needs θ < 1 (§3.5), and its exploitation slack is not derived [conjecture].
- Ngo's combination. The one worked toy is the sibling lane's inductive market, with LI-style prices over deliberation
  days. Its lesson is that "the Löbian part lives in the gate and the base, not in the market's coherence", and "with
  exact edges the market layer is idle (prices equal targets)" [known: in-house formal_market §2.7 and §4, unreviewed].
  The market matters only with soft edges and outside traders (formal_market §2.5).
- Drives and anchors [inference]. Ngo's drives pull goal credences up and his anchors hold empirical credences. On our
  edges, strong anchors do the job of the control mask (utility moves my action and not my beliefs about the opponent),
  under which CC is each agent's unique minimiser. Weak anchors give the unmasked edge, under which FairBot cooperates
  with DefectBot once the utility weight is large enough (formal_market T5). Ngo gives no functional form, so this
  matches roles only, with anchor strength as the free parameter.

### 4.5 One agent's action beliefs

Ngo's own Löbian passage concerns one agent and beliefs of the form "if I believe X, X will happen" (§1). In the level
model [inference: direct evaluation]:
- The actuator clause "I do a iff I am certain I do a", a ↔ □a, has the unique zero a = 1 at every level. The clause "I
  do a unless I am certain I don't", a ↔ ¬□¬a, has a = 0 at every level. As with FairBot and CUPOD, the positive form
  of the clause picks the self-fulfilling belief.
- The same base produces the spurious choice in 5-and-10. Take an agent that takes 5 iff it can prove "taking 10 yields
  0". PA proves that taking 10 yields 10, so "A = 10 → U = 0" is provably equivalent to "A = 5", and the agent's clause
  is a ↔ □a with a = [A = 5]. Its unique zero is a = 1: the agent takes 5. This is the standard Löbian argument for 5-and-10, written in
  levels [known, for the argument: Embedded Agency, via li.md §4.1; Troll Bridge is a variant].

Level-indexed action beliefs therefore reproduce spurious Löbian proofs and do not block them, because they are GL's
semantics. Any mechanism that gets FairBot's cooperation from a vacuous base gets this agent's spurious choice from the
same base (Q7).

## 5. Minimal toy experiments

None has been run. All are CPU-only, at nice 19 and ionice idle, with criteria fixed before any run. Each is estimated
at under an hour [conjecture: not timed].

Already measured today, so not proposed again:
- The synthesis's X2 (a noise-tolerant box): the critique measured most of it (checks 4-6, §3.6). Every box in its
  family fails the cooperation bar or is suckered by deep tricksters, except the windowed vote.
- The synthesis's X3 (price FairBot) as specified: the price convention decides (check 7, §4.4).
- Unlearned solvers on the level energy: one zero in all 9 exhaustive enumerations; target-only search reaches it from
  every start; full gradient descent mostly does not (sibling toy, §3.4).

### Toy 1. Learned grounded simulators, off their training pair and with a DefectBot probe (parts a, d, f; Q6)

Question: does a learned FairBot-like policy stay unexploitable against opponents it did not train with? And when the
opponent's play against DefectBot is supplied as an input, does payoff-driven learning find PrudentBot's
discrimination?

Stage A, reproduction and off-pair test:
- Class and play as in E1: Pr[S] ≤ 1 − ε, Pr[C|¬S], Pr[C|C] and Pr[C|D], with ε = 10⁻³, and outcomes from E1's closed
  form. Naive-gradient and LOLA pairs, 20 co-trained seeds each.
- Each learned policy then plays DB, CB, εGFB at ε ∈ {10⁻³, 10⁻², 10⁻¹}, unconditional mixers with q ∈ {0.1, …, 0.9},
  and every other seed's policy.
- Harness check: E1's qualitative results must reappear (naive pairs defect, LOLA pairs cooperate, Pr[C|D] > 0);
  otherwise the implementation is wrong.
- Reading fixed in advance: learned grounded cooperation transfers off the pair iff, for at least 90% of LOLA seeds,
  P(C | DB) ≤ 10ε and cross-seed P(CC) ≥ 0.9. If P(C | DB) ≥ 0.1 for most seeds, E1's result is cooperation with the
  co-trained partner only.

Stage B, a probe input:
- Add d, the opponent's exact probability of cooperating against DB, computed in closed form by the environment, the
  way E1's transparency computes play against me. The responses become Pr[C | a, d] = σ(w_{a,0} + w_{a,1}·d) for the
  opponent's simulated action a ∈ {C, D} against me. The class contains εGFB, and an ε-grounded PrudentBot with a
  threshold on d as its large-weight limit.
- Co-trained pairs that also face a fixed population (CB, DB, εGFB, ε-grounded PB, mixers), naive and LOLA, 20 seeds
  each.
- Tests: held-out opponents (εGFB at other ε, mixers, other seeds) and an adversarial search over the opponent's
  parameters that maximises P(learner plays C, opponent plays D).
- Control arm: the probe as a sampled action instead of a probability. Prediction [inference from §3.5]: a policy
  grounded like εGFB that defects against CB cooperates with its own copy with probability 1/(2 − ε), about ½.
- Pass ("a PrudentBot-like rule is learned"), for at least 50% of LOLA seeds: P(D | CB) ≥ 0.9, cross-seed P(CC) ≥ 0.9,
  P(C | DB) ≤ 10ε, and an adversarial sucker probability ≤ 10ε.
- Readings. A pass means that, with the probe information supplied, learning finds the discrimination. That is a weak
  form of Q6: the information is given and only its use is learned. If P(D | CB) stays low, the learners leave the probe
  unused, which is Thm 4.10's rank-0 limit showing up in practice. A high adversarial sucker probability means the
  learned rule is unsound off its training distribution.
- Cost: a closed form with 4 or 6 parameters per agent; seconds to minutes per seed, under an hour for both stages.

### Toy 2. A learned, label-free level sweep (parts b, c; size transfer), as a second chain inside toy-search pick 2

Question: does a weight-tied block trained only on the level energy learn the exact clause rule, and does it stay exact
on deeper and larger matches? The selection is already in the energy, so a pass shows a learned local rule. It does
not show learned selection, and it does not test Q1.
- Data: random modal matches from formal_logic §7's generator, deduplicated by canonical form. Train at read-off depth
  N ≤ 4 with closures of at most 4 atoms; test up to 3N and 8 atoms. N comes from the match's syntax: N = K, the number
  of distinct boxed subformulas, counted in formal_logic's encoding where □₁ψ is written □(¬□⊥ → ψ). With □₁ as a
  primitive node, K can fall below the level where play stops changing (verify_math problem 1). The cited generator
  gives few deep matches (117 of 3,000 stabilise at levels 4-8; critique check 3), so generate until at
  least 500 test matches change their actions after the training depth.
- Energy: E(s) = Σ over atoms q and levels n of (s(q, n) − Φ_q(lower levels))², with min, max and 1 − x (Kleene-Zadeh
  connectives) and □ = min over lower levels, 1 at level 0.
- Block: one weight-tied message-passing block over atom-by-level nodes, at most 20k parameters, applied T times from
  s = ½, with T swept as a compute budget; numpy with hand-written backprop.
- Arms: label-free (loss E(s_T) + 0.1·Σ_t E(s_t)); supervised on the exact table with the same network; a product-□
  variant.
- Scoring: agreement with the exact table only on test matches whose actions change after the training depth, reported
  next to three baselines: the exact sweep (must score 100%, otherwise the harness is wrong), always-defect, and
  freezing the action at the training depth.
- Exact-rule check: discretise the learned update and compare it with the exact clause rule on every local input
  configuration.
- Adversary: enumerate all opponent formulas up to a stated size, with a graded objective (the learned read-off minus ½
  on matches where the exact table says the opponent defects).
- Pass, all required: the exact-rule check holds; agreement ≥ 99% on the changing matches at every tested depth; no
  sucker case in the enumeration.
- Readings. Pass: label-free training found forward chaining, and depth transfer follows from the rule being exact.
  Label-free fails while supervised passes: the objective is the cause, the comparison toy-search plan §4.1 makes for
  pick 2. Both fail: architecture or budget. Scores pass but the exact-rule check fails: the score reflects the
  generator's distribution.
- Cost: tens of minutes.

### Toy 3. The windowed-vote box: stabilisation, certification and noise (parts c, e; Q2)

Question: for □ = "held at ≥ ρ of the last w lower levels", which pairs stabilise, how deep, can a repeated state
certify it, and does cooperation survive per-entry noise from the certified depth on? Soundness at joint stabilisation
is already proved (§3.6), so it is not sampled.
- Noise-free: random modal pairs (deduplicated), plus the named agents and tricksters Z_k for k ≤ 30, at
  (w, ρ) ∈ {4, 8, 16} × {½, 0.75}. Run until the finite state (the last w values of every boxed argument) repeats, and
  record the cycle length and the depth where the cycle starts, against K.
- Read-off rule: cooperate only on a certified cycle of length 1 whose action is C. By the lemma of §3.6 this keeps
  soundness.
- Noise: per-entry flips at ε ∈ {10⁻³, 10⁻²}; P(CC) for FB–FB, PB–PB and PB–FB from the certified depth to 200; sucker
  probability measured against the opponent's actual noisy play.
- Adversary: all opponent formulas up to a stated size, for each (w, ρ).
- Pass, at one (w, ρ): (i) at least 99% of random pairs reach a length-1 cycle within 10K + 20 levels; (ii) P(CC) ≥ 0.95
  at ε = 0.01 for the three pairs at every tested depth; (iii) P(I play C and the opponent plays D) ≤ 2ε against every
  enumerated opponent; (iv) PB defects against CB with probability ≥ 0.95.
- Readings. A pass means a noisy version with fixed memory keeps PrudentBot's properties with a certificate, which
  answers Q2 for this family. If (i) fails, agents must defect on uncertified pairs, and the cooperation lost is the price
  of noise tolerance. Known already: at (8, ½), 88 of 1,800 random pairs had no stable read-off by level 280.
- Cost: minutes.

### Toy 4 (lower priority). Price FairBot with explicit conventions (Q4)

Question: for a stated trader class, does the copy-day limit depend on the class, or only on the market's conventions?
- Conventions become explicit factors: the choice among valid prices, the tie-break at p = ½, and the day-1 prior.
- No type labels: the opponent's type is settled only by a lagged deductive process (o_n is revealed after d days).
- Traders: every ramp trader on a grid of thresholds and features, with LI-style budgets.
- The synthesis's criteria C1 and C2 become claims about this class only. Its sucker bound is dropped unless derived
  for a stated position rule.
- Reading fixed in advance: if some class gives the same limit under every convention, that class selects; otherwise the
  class does not determine selection, which check 7 leads me to expect [conjecture].
- Cost: seconds to minutes. A finite trader class is not a logical inductor, so this cannot answer Q4 for LI itself.

## 6. Open questions

- Q1. Levels from a fixed-size learned state. Is there a learned update on a fixed-size state (one belief per action atom
  plus a bounded memory) that reproduces the level-n values for every pair of modal agents and extrapolates in n? Known:
  monotone iteration from an optimistic start with a running AND suffices [known: in-house formal_dynamics
  Thm 1(b)-(c)]; such an update is not a contraction, so R9(ii) does not apply, and R9(iv) shows that an end-only loss
  at 20 blocks does not fix block 500. Open: whether training on payoffs or on the energy produces the optimistic start
  and the running AND. The level-free energy gives no signal for the base, so a test needs payoffs and adversarial
  opponents; I have no minimal design for it yet [inference].
- Q2. Noise. Settled: soundness at joint stabilisation for every admissible box, for agents of FairBot's form (§3.6).
  Open: the rate, depth and certification of stabilisation, and noise tolerance from the certified depth on (Toy 3).
- Q3. Opponents given as weights. Is there a reader R(θ_opp, θ_me) that runs in polynomial time, does not simulate the
  opponent's searcher, and is sound against an adversary with budget B? Without a template, "actual play" is defined
  only once the opponent's searcher is fixed [inference].
- Q4. What a logical inductor selects. For an inductor whose day-n action is "cooperate iff P_n(o_n) > t" against an
  exact copy, which limit points occur? Is there a condition on the trader class that forces the limit 1 against copies
  while P_n(o_n) → 0 against DefectBot? Conjecture: the limits 0, 1 and t all occur, depending on the construction
  [conjecture: formal_market §2.7]. In the searches run, no paper shows two LI agents cooperating in a one-shot PD
  [known: li.md §5.2].
- Q5. A bounded PrudentBot in bounded PA. With Critch's hypothesis as corrected (B4), does PrudentBot_{k, g(k)}
  cooperate with itself and with FairBot_k for some computable stagger g, and what is the least g? B3 shows
  g(k) = 4k + 100 for self-play in a calculus with primitive modal rules; no explicit cooperation threshold is known
  even for FairBot_k [known: B4].
- Q6. Deriving the probe. Is there one decision rule that, given readable opponents, exploits CooperateBot as a
  consequence rather than through a written-in DefectBot check? This is open for cooperative oracles [known: D4]; a
  utility edge with a dependence edge derives it only against fixed-rule opponents [known: in-house formal_dynamics
  Prop 10]; Toy 1's stage B tests a weak form.
- Q7. One agent. Is there a level structure or an actuator form that keeps cooperation between agents and blocks the
  spurious 5-and-10 choice within one agent (§4.5)?
- Q8. Equal compute. At a fixed compute budget per decision, which of εGFB with a depth cap, the level read-off at a
  fixed N, and B3's budget stagger has the smallest worst-case exploitability? At fixed N, Z_N suckers the level
  read-off with probability 1, while εGFB stays ε-exploitable (§3.6).

## 7. Sources

URL check: every URL below returned HTTP 200 with a matching page title on 2026-11-01 between 23:10 and 23:18Z (curl
from General), except where the row says otherwise. "Read" says how much this lane read and in which file: `lit.md`,
`li.md` and `post.md` (19:50-20:55Z), the synthesis's re-checks (21:20-22:00Z), the citation check (22:00-22:40Z) and
the critique (22:10-23:00Z).

| ID | Source | URL | Read |
|---|---|---|---|
| Ngo 2026a | R. Ngo, "Explaining Knightianism on one foot", Mind the Future, 2026-09-02 | https://www.mindthefuture.info/p/explaining-knightianism-on-one-foot ; crosspost https://www.lesswrong.com/posts/pYFBD2SnqiWkuNns5/explaining-knightianism-on-one-foot | full text via the Substack API, diffed with the crosspost (post.md); re-fetched by the citation check |
| Ngo 2026b | R. Ngo, "Agents as webs of beliefs", 2026-06-27 | https://www.mindthefuture.info/p/agents-as-webs-of-beliefs | full (post.md) |
| not-circuits | CarolusRenniusVitellius, "What if not Circuits?", LessWrong, 2026-09-21 | https://www.lesswrong.com/posts/mMERyrvEJ4xbiozie/what-if-not-circuits | full, by the not-circuits lane (`work/not-circuits/post.md`) |
| A2 | Bárász, Christiano, Fallenstein, Herreshoff, LaVictoire, Yudkowsky, "Robust Cooperation in the Prisoner's Dilemma: Program Equilibrium via Provability Logic" (2014; v2 2021) | https://arxiv.org/abs/1401.5577 ; checker https://github.com/machine-intelligence/provability | full (lit.md, post.md); re-opened by the synthesis and the citation check, which also read the checker's `src/Modal/Formulas.hs` |
| B1 | A. Critch, "Parametric Bounded Löb's Theorem and Robust Cooperation of Bounded Agents" (2016); J. Symbolic Logic 84(4), 2019 | https://arxiv.org/abs/1602.04184 ; https://doi.org/10.1017/jsl.2017.42 | §§5-6 (lit.md) |
| B2 | Critch, Dennis, Russell, "Cooperative and uncooperative institution designs: Surprises and problems in open-source game theory" (2022) | https://arxiv.org/abs/2208.07006 | theorem statements, open problems, §§6.1-6.2 (lit.md); PDF (citation check) |
| B3 | Duclaux, Formenti, Cobben, Schölkopf, Jin, "Proving Your Way to Cooperation: Formalizing Proof-Based Open Source Game Theory in Lean", ICML 2026 AI4Math workshop | code https://github.com/ColombanD/open-source-game-theory (commit 5a1bc15) ; paper https://openreview.net/forum?id=Wc5TAIUC8k | theorem files and the 29-file import closure at 5a1bc15 (synthesis, critique). OpenReview served a bot check at 22:00-22:40Z and again at 23:12Z (page and API), so the abstract's claims are not used here |
| B4 | L. Levine, "Math for AI safety: an invitation for mathematicians" (2026) | https://arxiv.org/abs/2609.15289 | §2 and acknowledgements (lit.md); pp. 8 and 28 (citation check) |
| B5 | S. Garrabrant, "Prisoners' Dilemma with Costs to Modeling" (2018; a draft) | https://www.alignmentforum.org/posts/XjMkPyaPYTf7LrKiT/prisoners-dilemma-with-costs-to-modeling | full (lit.md) |
| C1 | C. Oesterheld, "Robust program equilibrium", Theory and Decision 86(1):143-159 (2019) | https://doi.org/10.1007/s11238-018-9679-3 | abstract only. The publisher page served a bot check at 23:11Z; title, journal, pages and author confirmed through the Crossref API at 23:12Z |
| C2 | C. Oesterheld, "A Note on the Compatibility of Different Robust Program Equilibria of the Prisoner's Dilemma" (2022) | https://arxiv.org/abs/2211.05057 | full (lit.md); Thm 1 and Prop 10 re-opened |
| C3 | Cooper, Oesterheld, Conitzer, "Characterising Simulation-Based Program Equilibria", AAAI-25 | https://arxiv.org/abs/2412.14570 | §§1-3 |
| C6 | AXRP episode 49, "Caspar Oesterheld on Program Equilibrium" (2026-02-18) | https://axrp.net/episode/2026/02/18/episode-49-caspar-oesterheld-program-equilibrium.html | transcript 01:18-01:43 |
| D1 | Fallenstein, Taylor, Christiano, "Reflective Oracles: A Foundation for Classical Game Theory" (2015) | https://arxiv.org/abs/1508.04145 | abstract |
| D2 | J. Taylor's NicerBot comment on S. Armstrong, "Reflexive Oracles and superrationality: prisoner's dilemma" (2015) | https://www.alignmentforum.org/posts/5bd75cc58225bf067037505e/reflexive-oracles-and-superrationality-prisoner-s-dilemma | comment and thread |
| D3 | S. Garrabrant, "Cooperative Oracles: Introduction" (2017) | https://www.alignmentforum.org/posts/5bd75cc58225bf0670375419/cooperative-oracles-introduction | introduction |
| D4 | Diffractor, "Cooperative Oracles" (2018) | https://www.alignmentforum.org/posts/SgkaXQn3xqJkGQ2D8/cooperative-oracles | definitions and theorem statements; the author omits the proofs |
| D5 | S. Garrabrant, "Prediction Based Robust Cooperation" (2017) | https://www.alignmentforum.org/posts/5bd75cc58225bf067037536c/prediction-based-robust-cooperation | full |
| D7 | S. Garrabrant, "Two Major Obstacles for Logical Inductor Decision Theory" (2017) | https://www.alignmentforum.org/posts/5bd75cc58225bf06703753d4/two-major-obstacles-for-logical-inductor-decision-theory | full |
| LI | Garrabrant, Benson-Tilsen, Critch, Soares, Taylor, "Logical Induction" (2016) | https://arxiv.org/abs/1609.03543 | local full text (`/home/agent/workspaces/eurisko-study/li-net/full.txt`); Thms 4.3.3, 4.6.2, 4.9.2-4.9.3 and 4.11.1-4.11.2 read by line |
| E1 | A. Hutter, "Learning in two-player games between transparent opponents" (2020) | https://arxiv.org/abs/2012.02671 | §§1, 3 and 4.2 (critique, citation check) |
| E2 | Oesterheld, Treutlein, Grosse, Conitzer, Foerster, "Similarity-based cooperative equilibrium", NeurIPS 2023 | https://arxiv.org/abs/2211.14468 | §§1-6 and App. G |
| E3 | Todorov, ten Napel, Müller, "Parametric Open Source Games" (2026) | https://arxiv.org/abs/2606.27068 | §§2-4 and App. A |
| E5 | Meulemans et al., "A game theory for foundation models shows new paths to rational cooperation through similarity inference" (2026) | https://arxiv.org/abs/2608.03958 | abstract, opening section, definitions |
| E6 | Lin et al., "Policy-Conditioned Policies for Multi-Agent Task Solving" (2025) | https://arxiv.org/abs/2512.21024 | abstract |
| Demski 2017 | A. Demski, "Policy Selection Solves Most Problems" | https://www.greaterwrong.com/posts/5bd75cc58225bf067037550c/policy-selection-solves-most-problems | fetched; quotes string-matched (li.md) |
| Diffractor 2018 | "Logical Inductors Converge to Correlated Equilibria (Kinda)" | https://www.greaterwrong.com/posts/5bd75cc58225bf0670375569/logical-inductors-converge-to-correlated-equilibria-kinda | fetched |
| Demski 2023a | A. Demski, "Probabilistic Payor Lemma?" | https://www.lesswrong.com/posts/ZWhJcHPmRaXAPAK5k/probabilistic-payor-lemma | fetched |
| Demski 2023b | A. Demski, "FixDT", and his comments on it | https://www.greaterwrong.com/posts/8dbimB7EJXuYxmteW/fixdt | post and comments |
| Critch 2023 | A. Critch, "Modal Fixpoint Cooperation without Löb's Theorem" | https://www.alignmentforum.org/posts/2WpPRrqrFQa6n2x3W/modal-fixpoint-cooperation-without-loeb-s-theorem | lemma statement and proof rules |
| Armstrong 2013 | S. Armstrong, "Probabilistic Löb theorem" | https://www.lesswrong.com/posts/oBDTMnEzptBidvmw7/probabilistic-loeb-theorem | the two sections on Löb failing (citation check) |
| Troll Bridge | A. Demski, "Troll Bridge" (2019) | https://www.lesswrong.com/posts/hpAbfXtqYC2BrpeiC/troll-bridge-5 | fetched (li.md, via GreaterWrong; that mirror rate-limited my check, so the LessWrong URL is given) |
| Embedded Agency | A. Demski, S. Garrabrant, "Embedded Agency" (2019) | https://arxiv.org/abs/1902.09469 | abstract and the prose chapter on decisions (li.md) |

In-house files used (none has been reviewed by a person):
- this lane: `work/lob-neural/{synthesis,check_citations,critique,lit,li,post}.md`;
- the sibling lane: `work/energy-prudentbot/formal_logic.md` (L), `formal_dynamics.md` (D), `formal_market.md` (M),
  `verify_math.md` (its line-by-line check of L, D and M) and `toy_result.md`;
- `work/toy-search/plan.md`;
- `inconsistency-search/rounds/r4-r9/li-net/math/block-dynamics.md` at 0a36abb;
- `outbox/research-direction-note-v2-2026-10-16.md`.

## 8. What changed after the critique

No finding was rejected. Two were applied with a qualification, noted in their rows. Three kinds of material were added
beyond the two reviews: the sibling lane's verify_math and toy results (the base result and the target-only search
result in §3.4; the soft read-off exploit and the read-off encoding caveat in §3.6); the 5-and-10 derivation (§4.5); and
the URL checks (§7). Findings
that concern only `lit.md` are corrected here wherever this report uses the statement. `lit.md` itself still carries
them, because this task writes only this report.

Citation check (`check_citations.md`):

| Finding | Change |
|---|---|
| P1 (with critique F1): a level index was said to conflict with a fixed, search-independent target | Reversed. With levels in the state, the target is unique, defined by the energy and independent of the search, and the block is an anchored contraction (§3.4, §4.3). The tension applies only to levels as blocks, (B) |
| P2 (with P8 and F2): E3 contradicts "naive learners defect"; E2 and E3 do not learn εGFB-like policies | §3.7 rewritten: coupling hard-wired above γ* gives ascent toward cooperation; E1's grounding is imposed; E2 and E3 learn thresholds and semantics parameters |
| P3 (with F8): the noise values were one player's P(C) | P(CC) values used: 0.942, 0.359, 0.018 (§3.6). The X2 prediction they fed is replaced by Toy 3 |
| P4: Löb's theorem misstated in lit.md | Stated for S = PA with □ = PA-provability (§2) |
| P5: "bounded PrudentBot is open" out of date | Open for bounded PA (B2 OP9); proved in B3's calculus (§3.6, Q5) |
| P6: C2 Prop 10 cited for the simulation fix | Prop 10 cited only for PBθ against εGFB, with its boundary discrepancy; the simulation fix is tagged inference by analogy, with the exact-probability requirement (§3.5) |
| P7: D7 generalised to "LI decision rules" | Restricted to the LI-EDT agent with ε-exploration, an informal claim; other rules open (§3.4) |
| P9 (with F16): ½CC + ½DD overstated | Marginals ½ and P(CC) = P(DD) for FB–FB; ½CC + ½DD only in the hard-edge limit; PB pairs lean to DD (§3.3) |
| P10: E2 does not always need pretraining | LOLA sometimes succeeds without CCDR (§3.7) |
| P11: "every positive result reads a conditional policy" | Restricted to the program and oracle results, with E5 and D6 named as exceptions (§4.2) |
| P12: C2's compatibility result covers εGFB only | Stated as DUPOC, CIMCIC, PBθ and εGFB (§3.5) |
| P13 (with F24): B5 holds among four agents | Restricted to CB, DB, FB and PB; instability and draft status stated (§3.5) |
| P14: three inferences tagged as known | Retagged [inference]: the threshold roots, the similarity score's lack of a probe, E5's dependence measure (§§3.2-3.3) |
| P15: "the only proof-free PrudentBot analogue" | Now "the only one with a stated, if informal, analysis", with Oesterheld's CooperateBot check mentioned (§3.5) |
| P16: in-house εG-PrudentBot "checked only by Monte Carlo" | "Derived and Monte Carlo-checked" (§3.5) |
| P17: scope conditions dropped | Read-off at or above stabilisation (§3.4); Prop 5(a) only for agents of the form □p_me ∧ ψ, with CooperateBot exploitable (§3.6); the same scope on the lemma (§3.6, Q2) |
| P18: R4 is the un-anchored block | Called the un-anchored self-referential block (§4.1) |
| P19, P20: lit.md's cross-system remark and its "all B2 results via B1" | Not used in this report |
| U1: B3's abstract unverifiable | Abstract claims not used; OpenReview still served a bot check at 23:12Z (§7) |

Critique (`critique.md`):

| Finding | Change |
|---|---|
| F1: "only ε-grounding fits our block design" does not follow | Replaced by the (A)/(B) comparison and the conclusion that the design hosts (A) with R9(ii)'s guarantee (§4.3). Qualified: F1's "the base costs no exploitability" holds only when the read-off is at or above the match's stabilisation level; at a fixed N, Z_N exploits it (§3.6). F1 calls depth extrapolation in (A) the same kind of problem as sudoku and N-Queens, while F3 calls that a loose analogy; §4.3 keeps both points: the same kind of test, on an easier problem |
| F2: E1 misread | As P2 and P8. Added Toy 1's stage A to measure off-pair behaviour (§5) |
| F3: X1 could not test learned selection or Q1 | Rebuilt as Toy 2, run inside toy-search pick 2: depth from syntax, scoring on changing matches only, trivial baselines, a supervised arm, an exact-rule check, the degenerate control dropped. It no longer claims to test Q1 or learned selection |
| F4: X2's outcome was predictable; a fixed population cannot answer Q2 | Replaced by Toy 3 on the windowed vote, aimed at stabilisation and certification, with an adversarial enumeration; the critique's measurements are reported in §3.6 and the window and fraction-box predictions dropped |
| F5: X3 is decided by conventions and is not an LI | Demoted to Toy 4 (lower priority), with conventions as factors, no type labels, a closed trader class and no unproved sucker bound; the measured convention dependence is reported in §4.4 |
| F6: exactness holds only under three conditions | Stated in §3.6 and the TL;DR; the "exact against ε" comparison is restricted to those conditions, and the equal-compute comparison became Q8 |
| F7: Lemma 1 used a non-standard notion | Restated with A2's notion (only DefectBot), the iteration form (with a simpler proof: min(x, y) never increases), the rank-0 assumption dropped, and the false dichotomy removed (§3.4) |
| F8 | As P3 |
| F9: "no sound reader exists for energies given as weights" | "No sound reader is known that avoids simulating the opponent's searcher" (§3.2, Q3) |
| F10: the market layer was credited with the result | The market adds nothing in the exact case; it matters only with soft edges and outside traders (§4.4) |
| F11: no stated target; four exploitability notions | Definitions in §3.0: the table, equilibrium and exploitation targets, exact and probabilistic exploitability at actual play; Toy 3 measures against actual noisy play |
| F12: no experiment on a learned decision rule | Toy 1's stage B: a learned rule with a DefectBot probe input, population training, held-out opponents and an adversary (§5). Qualified: the probe information is supplied by the environment, so this tests whether learning uses it, a weaker question than Q6 |
| F13: Ngo's own Löb passage concerns one agent; drives without anchors | Added §4.5 (the actuator clause and the 5-and-10 derivation) and Q7; the drives mapping now has anchor strength as its parameter (§4.4) |
| F14: soundness holds at stabilisation for the whole box family | Lemma stated and proved in §3.6; Q2 and Toy 3 aimed at stabilisation, depth and certification |
| F15: Gödel connectives misnamed | Named Kleene-Zadeh (min, max, 1 − x) in Toy 2 |
| F16 | As P9 |
| F17: "a network cannot take itself as input" | "Cannot take its own weights as input at full size"; compressed self-descriptions noted (§3.2) |
| F18: the threshold fixed points need a condition | ε ≤ t < 1 added (§3.3) |
| F19: the LI probe's slack was not derived | Tagged [conjecture], with what it depends on (§3.5) |
| F20: the B3 check was narrow; the gap is semantic | B3 described as a calculus with primitive modal rules and costs from Critch's model; match to bounded PA not shown (§3.6) |
| F21: the adversary was random search | Exhaustive enumeration with a graded objective (Toys 2 and 3) |
| F22: "fresh instances" does not exclude memorisation or trivial policies | Deduplication by canonical form and trivial baselines (Toy 2) |
| F23: universal claims from absence | "In the sources checked" added (§3.2, §3.5, §3.7) |
| F24 | As P13 |
