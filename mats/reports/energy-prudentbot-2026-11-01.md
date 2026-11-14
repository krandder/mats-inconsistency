# PrudentBot as the lowest-inconsistency state of a level energy (1 November 2026)

For Kelvin, on your question of 1 November (fleet request pool-22ssuc). Written by a read-only subagent of opus-mats,
finished at about 23:30Z. Status: unreviewed.

Built from three working notes in `/home/agent/workspaces/mats/work/energy-prudentbot/`:
- `synthesis.md`: a judged merge of three independent formalisations, `formal_logic.md` (L), `formal_dynamics.md` (D)
  and `formal_market.md` (M), all built on `context.md` (CTX);
- `verify_math.md`: an adversarial check of the synthesis, with code and recorded output;
- `toy_result.md`: one CPU run of the toy specified in `toy_spec.md` (code and output in `toy/`).

Every finding of the two checks is applied or answered; §3.5 lists each one and what I did with it. I re-ran the
numbers those corrections rest on with my own code (§7.3). The input notes were not modified.

Tags: **[proven]** a proof is given here or in the named note, and the synthesis, the verifier or I checked it line by
line; **[checked numerically]** a script reproduced it, and the text says which; **[inference]** argued, not proved;
**[conjecture]** stated, not shown.

## TL;DR

1. Yes, in a narrow but exact model: if each agent's energy unrolls both published definitions over levels of certification, it has one zero for any weights, and read where it stabilises, that zero behaves exactly like PrudentBot against CooperateBot, DefectBot, FairBot and itself, and cannot be suckered [proven; checked numerically].
2. Löb's role is played by induction over the levels: "certified" means "held at every lower level", which is vacuously true at level 0, so cooperation starts at the bottom and climbs; without levels no energy can select CC over DD [proven].
3. The main catch: that vacuous base does all the selecting; if level 0 certifies nothing, every uniqueness and search result still holds and the zero is DD, and our PDG/atom energy has neither the levels nor the base today [proven; checked numerically].
4. Generic search misses the zero: in FB–FB, PB–FB and PB–PB, full-gradient and single-flip descent reach it from at most 21% of random starts, and two FairBots each doing single-flip descent end C against D in half the matches; updates that move only each edge's own target (the LIR mask) always reach it [checked numerically].
5. Soft versions lose cooperation with reasoning depth, and a thresholded read-off lets a noisier copy exploit you unless noise levels are published and agents sample at a non-vacuous level [proven; checked numerically]; nothing was learned, and §6 proposes a learning toy.

## 1. The question and our energy framework

Your question: "On the inconsistency objective, I wonder whether we could in theory have a 'world-model' that defines
the energy/inconsistency in a way that the 'lowest-inconsistency' state could implement an algorithm that is akin to
the PrudentBot maybe? How to formalize that? Instead of having access to each other source codes, maybe we could have
two agents each having access to each other's 'energy function definitions' or something, and mayve they both realize
that they both minimize their energies by cooperating in this case? Very confusing but very interesting."

The two agents, as Bárász et al. define them (arXiv:1401.5577v2, Algorithms 4 and 5, checked in the PDF): FairBot
cooperates with X iff PA ⊢ [X(FairBot) = C], and PrudentBot cooperates with X iff PA ⊢ [X(PrudentBot) = C] and
PA+1 ⊢ [X(DefectBot) = D], where PA+1 is PA plus Con(PA). Their Theorem 3.2: "PrudentBot is unexploitable, mutually
cooperates with itself and with FairBot, and defects against CooperateBot." FairBot–FairBot cooperation comes from
Löb's theorem (Thm 3.1), and their footnote 3 says what breaks the tie: "the symmetry between mutual cooperation and
mutual defection is broken by the positive criterion for action."

Our framework (CTX §0–§3). A belief state μ is a distribution over the world model's variables. The world model
supplies atoms q(X_i | x_S), each with a precision β, and the energy is the PDG inconsistency
F(μ) = Σ β · E_μ KL(μ(X_i | x_S) ‖ q(X_i | x_S)) − γH(μ), convex in μ; the clean target is the limit γ → 0⁺, the
maximum-entropy minimiser of the inconsistency. The lowest-inconsistency state is the minimiser μ*. Search
approximates μ* and should change the speed, never the target (your R1, R12, R18). LIR's control masks say what a term
may move: an edge moves its target and never its context, and utility moves actions and never beliefs. Your other
constraints (CTX §6): no recursion or max in θ (R19), no regulariser in the target (R3), precision not set by the
search (R9), one generic agent search (R20).

The gap (CTX §7): nothing in this energy played the role of provability or of Löb's theorem. A FairBot clause added as
an edge between the two players' actions says c₁ = c₂, which CC and DD satisfy equally well.

Your phrase "they both minimize their energies by cooperating" has two exact readings, and §3 treats both. The
consistency reading: each agent's energy, built from both definitions, has a single zero, and it says CC. The
preference reading: given what each agent reads about the other, its own preference inconsistency is lowest when it
cooperates.

## 2. The formal model

The base model is L's (`formal_logic.md` §1), in the synthesis's notation (§2), with two corrections from the checks
(§3.5, rows 1 and 3).

**Game.** One-shot PD with T > R > P > S and 2R > T + S; the numbers are (T, R, P, S) = (5, 3, 1, 0).

**What an agent publishes.** Its energy definition D_A = (φ_A, H_A). H_A is a finite set of reference opponents of
lower rank (here ∅ or {DB}). φ_A is a fully modalised formula (every letter under a box) over the letters p_me ("my
opponent plays C against me") and p_H ("my opponent plays C against H"), built from ⊤, ⊥, ¬, ∧, ∨ and □. The box with
a skip, □_k ψ, abbreviates □(¬□^k⊥ → ψ), Bárász et al.'s "PA+k proves ψ". A template in this language is a restricted
source code.

| agent | φ_A | H_A |
|---|---|---|
| CooperateBot CB | ⊤ | ∅ |
| DefectBot DB | ⊥ | ∅ |
| FairBot FB | □p_me | ∅ |
| PrudentBot PB | □p_me ∧ □₁¬p_DB | {DB} |
| PB_PA (DefectBot probe in PA) | □p_me ∧ □¬p_DB | {DB} |
| CUPOD (FB with C and D swapped) | ¬□¬p_me | ∅ |
| FB⁺ (FB that also demands Con(PA)) | □p_me ∧ ¬□⊥ | ∅ |
| Z_k (cooperates below level k) | □^k⊥ | ∅ |

Reading the other's energy definition means being able to instantiate its template against any opponent, in
particular against oneself and against its references such as DefectBot. The template and the read-off rule are
public. Weights and solvers stay private: the weights do not matter (A(i)), and the solvers matter unless they are
target-only (§5.2). In the soft model the flip rate ε must be public too (§5.3).

**The unrolled world model.** The match closure Q(X, Y) is the smallest set of ordered pairs that contains (X, Y) and
(Y, X) and, with each (A, B), also (B, A) and (B, H) for every H in H_A. For PB vs CB it is {(PB, CB), (CB, PB),
(CB, DB), (DB, CB)}. The variables are c^n_AB ∈ {0, 1}, "A plays C against B, as held at level n", for (A, B) in Q and
n = 0, …, N. The body β_AB is φ_A with p_me ↦ c_BA and p_H ↦ c_BH, evaluated level by level:

    ⟦c⟧_n = c^n,   connectives pointwise,   ⟦□ψ⟧_n = AND over m < n of ⟦ψ⟧_m   (at n = 0 the empty AND is 1)

Hence ⟦□^k⊥⟧_n = 1 iff n < k, and ⟦□_k ψ⟧_n = AND over k ≤ m < n of ⟦ψ⟧_m. This is the Kripke semantics of the
provability logic GL on the chain in which level n sees every lower level, and it is the rule Bárász et al.'s own
checker uses (`handleBox = scanl (&&) True`, checked in `src/Modal/Formulas.hs` of machine-intelligence/provability).
The empty AND at level 0 is the vacuous base.

**The energy.** Each agent's energy contains every clause of Q (its own, the opponent's and the references'), with its
own weights.
- Crisp: E_w(s) = Σ over (A, B) in Q and n ≤ N of w_{AB,n} · 1[s(AB, n) ≠ ⟦β_AB⟧_n(s)], with every w > 0.
- PDG form (CTX §2.5): one edge per (AB, n), with target c^n_AB and context the lower-level variables its body reads;
  cpd q(c^n_AB = 1 | ctx) = 1 − ε if the body is true and ε if not; precision β_{AB,n} > 0;
  F(μ) = Σ β · E_μ KL(μ(c^n_AB | ctx) ‖ q(· | ctx)) − γH(μ), with γ → 0⁺. Rows with constant bodies (CB, DB) may be
  clamped.

**Read-off.** Let v_n be the values at level n of the distinct boxed subformulas of the bodies, counted after unfolding
every □_k, so that the clock subformulas □⊥, …, □^k⊥ are among them, and let n0 be the least n with v_{n+1} = v_n.
Equivalently, with □_k kept as a primitive, n0 is the least n ≥ k_max with v_{n+1} = v_n, where k_max is the largest
skip [proven, one line: the clock bit □^{n+1}⊥ changes between levels n and n+1 exactly when n < k_max; the two rules
agreed in 3,000 of 3,000 random match-ups, §7.3]. In the crisp model an agent plays its own entry at some level
N_A ≥ n0, by default n0, which both agents compute from the two definitions. In the soft model it samples its entry,
rather than thresholding it, at the first level ≥ n0 where its query about the opponent is not vacuous: level 1 for
FB in FB–FB, level n0 = 1 for PB against FB or PB.

**Search.** A target-only update of entry (AB, n) uses only that entry's own edge: the repair move
s(AB, n) := ⟦β_AB⟧_n(s), or a gradient step on that edge's term with its context held fixed. This is LIR with one focus
per clause edge, attention on the edge and control on its target (CTX §2.1, §4). A full update uses every term that
contains the entry, including the higher-level edges that read it; plain gradient descent on F is a full update.

**Preference layer** (used only in Theorem E). A preference flag with P(F = 1 | outcome) = exp(λ(u − u_max)), held at
1, whose term may move only the agent's own action (the v7 control mask), and a dependence model r: {C, D} → {C, D},
the opponent's action given mine, read from lower levels.

## 3. Results

### 3.1 Proven

The letters follow the synthesis's theorems, so the two documents can be read side by side.

**A. One lowest-inconsistency state.**

(i) For every N and every w > 0, E_w has exactly one zero s*. It does not depend on w, E_w(s) ≥ min w for every
s ≠ s*, and the zero for N is the restriction of the zero for N + 1. Proof: bodies at level n read only lower levels,
so the forward pass s(·, n) := ⟦β⟧_n(s) for n = 0, 1, … defines s*; any other table has a lowest level where it
differs from s*, its bodies there agree with those of s*, so one of its clauses is violated. [proven, L Prop 1.
Checked numerically: one zero in every exhaustive enumeration by L, the verifier and the toy, and in mine under both
bases, §7.3.]

(ii) Uniqueness comes from guardedness. With d(s, s') = 2^(−n), where n is the first level at which s and s' differ,
the clause map is a ½-contraction, so s* is its only fixed point (Banach), and n iterations from any table are correct
below level n [proven, one line; a pilot found no exception in up to 200 random pairs for each of 9 cases].

(iii) Löb lemma for energy minima. For every table s and formula χ: if ⟦□χ → χ⟧_n(s) = 1 at every level, then
⟦χ⟧_n(s) = 1 at every level. This holds for all tables iff the relation "level n reads level m" is conversely
well-founded. Inside s*, with χ = c_{FB1,FB2} ∧ c_{FB2,FB1}, it gives CC at every level, which is the "Löbian circle"
of Bárász et al.'s proof of Thm 3.1. The lemma needs the vacuous base: if level 0 certifies nothing, χ = ⊥ satisfies
the premise at every level. [proven, L Prop 3(a)–(b); the remark on the base is mine, one line. The soft form,
P(χ fails at some level ≤ n) = Σ over m ≤ n of P(□χ holds and χ fails at m), was re-run: 0.440418 on both sides.]

(iv) Stabilisation and agreement with PA. With n0 as in §2: n0 ≤ K, the number of distinct boxed subformulas after
unfolding; s* is constant from n0 on; from there s*(AB, n) = 1 iff A cooperates with B in Bárász et al.'s sense (PA
assumed sound); and a value is constant from level k on iff PA+k decides it. [proven from Bárász et al. Thm 4.7 and
the trace lemmas of Boolos ch. 7, as L Prop 2 and M T3 give it; L, D and M checked 600, 25 and 19 pairs; n0 ≤ K held
in all 3,000 of my random match-ups.] Example: PB vs CB stabilises at level 2, and the proof of Bárász et al.'s
Thm 3.2 says "PA+2 ⊢ [PB(CB) = D]" [checked in the source].

**B. PrudentBot's properties.**

(i) Outcomes read at N ≥ n0 [checked numerically by L, D and M independently, by the toy's S0, S1, S2 and S4, and by
my own evaluator]:

| match | n0 | outcome (row, column) |
|---|---|---|
| PB vs PB | 1 | CC |
| PB vs FB | 1 | CC |
| PB vs DB | 1 | DD |
| PB vs CB | 2 | DC: PB exploits CB |
| FB vs FB | 0 | CC |
| FB vs CB | 0 | CC |
| FB vs DB | 1 | DD |
| PB_PA vs FB | 3 | DD |
| PB_PA vs PB_PA | 2 | DD |
| CUPOD vs CUPOD | 0 | DD |

In a round robin with self-play, the row player's totals are PB 12, FB 10, DB 8 and CB 6.

(ii) Unexploitability. Let φ_A = □p_me ∧ ψ, as for FB and PB, and N_A ≥ n0. For every opponent Z in the modal
language and every level N_Z, s*(AZ, N_A) = 1 implies s*(ZA, N_Z) = 1. Proof: by stabilisation A's entry is 1 at every
level ≥ n0, so its first conjunct, the AND of s*(ZA, m) over m < n, is 1 for arbitrarily large n, and Z cooperates
with A at every level. [proven, L Prop 5(a). Checked numerically, with no sucker outcome in any of: L's 4 × 30,500
(opponent, N_A, N_Z) triples; the verifier's 1,000 match-ups with □₁ and □₂ opponents; the toy's 500 opponents ×
{FB, PB} × opponent levels 0–40; my 3,000 match-ups with □_k opponents, k ≤ 3, at levels 0–60.]

The proof needs stabilisation from n0, which the synthesis's literal rule does not give: under it, Z = □₁⊥ suckers
both FB and PB, and my random □_k opponents sucker FB 211 times and PB 210 times in 1,500 match-ups each. The read
level matters in a second way: FB read at a fixed level N against Z_N = □^N⊥ plays C while Z_N defects [proven;
checked, L §4], and the toy found 526 (FB) and 520 (PB) cooperating read-offs below n0 against opponents that defect at
some level. So each agent must compute n0 from both definitions. This also answers D Prop 9(b)'s "level-aware"
opponent, which assumes a read level fixed in advance.

(iii) PA+1. At level 0 every □ holds, so every agent whose formula is positive in □ cooperates with DB there. A
DefectBot probe that reads level 0 (PB_PA) therefore fails against every such opponent, and PB_PA ends in DD with FB
and with itself. This is Bárász et al.'s remark after Thm 3.2: the DefectBot check must run in a stronger system, or
"the resulting variant of PrudentBot would lose the ability to cooperate with itself" [proven; checked by L, D, M, the
toy (P6) and me; quote checked in the source].

(iv) An opponent whose definition is outside the language gives nothing to evaluate. Defecting against it keeps (ii)
and gives up cooperation [inference, L §5.2].

**C. Without levels, no energy reproduces Löbian cooperation.**

(i) Replace every □ by the identity. FB vs FB and CUPOD vs CUPOD then have the same clause, c₁ = c₂, yet their level
energies select CC and DD. So no selection rule that is a function of the flat energy, up to renaming atoms,
reproduces Löbian outcomes [proven, M T2(d); the toy confirmed that the two flat edge tables are identical].

(ii) If an energy over distributions is strictly convex and invariant under swapping C and D in every atom, its
minimiser gives every atom P(C) = ½. For the flat FB pair with soft copy edges the minimiser over (CC, CD, DC, DD) is
((1−δ)/2, δ/2, δ/2, (1−δ)/2), and as two separate agents the pair is an exact potential game with Φ(CC) = Φ(DD)
[proven, D Prop 4; the toy got 0.4950, 0.0050, 0.0050, 0.4950 at ε = 0.01].

(iii) A third-party probe is needed: a flat agent that reads only its opponent and reaches CC against FB also
cooperates with CB [proven, M §2.1]. Bárász et al. state the modal version in §4, summarised in their outline as
"modal agents which do not use third parties cannot achieve mutual cooperation with FairBot unless they also cooperate
with CooperateBot" [checked in the source, p. 3].

What C shows is that something beyond the flat clauses is needed. Levels with a vacuous base are one sufficient choice,
and the one that reproduces PA exactly; ε-grounding (F) and preference edges with one joint selection (M T5, §5.5) are
others, each with a cost [inference].

**D. Search.**

(i) Repair moves. From any table, any sequence of repair moves that repairs every entry infinitely often reaches s* and
stays there; after k complete sweeps in any order, levels 0 to k−1 equal s*, and one level-ascending sweep is enough.
Proof: level-0 bodies are constants, so a repaired level-0 entry is final; then induct on levels. [proven; checked,
toy P3.]

(ii) Masked descent on per-variable beliefs, μ = ⊗ Bern(p_{AB,n}), each logit moved only by its own edge:
u ← u + η(L(2π − 1) − u), with L = ln((1−ε)/ε) and π the probability that the body holds. For 0 < η < 2 it converges
from every start to one fixed point, computed level by level [proven: the system is triangular; checked, toy S2: 202
of 202 starts in 11 matches]. That fixed point decays with depth: in FB vs FB at ε = 0.01 its credence is 0.9532 at
level 10, 0.5946 at 14 and 0.1631 at 15, so it thresholds to s* only up to N = 14 [checked numerically, the verifier
and §7.3].

(iii) Full updates can stop elsewhere. The crisp energy has strict single-flip local minima, all at energy 1: 2(N−1)
of them for FB vs FB at N = 2 to 6, 2 for PB vs FB at N = 2, 2 and 4 for FB vs CB at N = 3 and 5, and none for PB vs
CB at N = 2 [checked by exhaustive enumeration, three times: pilot, verifier, toy]. How often searches end there is in
§4.

**E. Where the shape of PrudentBot's clause comes from.** Take any PD and a deterministic dependence model r.
Minimising one's own preference inconsistency λ(u_max − u(a, r(a))) over one's own action a gives a = C iff r(C) = C
and r(D) = D [proven by enumerating the four r: R > P picks C for r = (C, D), and T > R, P > S and T > S pick D for the
other three; D Prop 10(b) also checked 4·10⁴ cases]. With a factorised model, in which the opponent's action does not
depend on mine, the minimiser is D for every belief [proven, D Prop 10(a)]. Reading r(C) = C as □[the opponent
cooperates with me] and r(D) = D as □₁[the opponent defects against DefectBot] gives PB's clause. The second reading
uses DefectBot as a stand-in for "me, defecting", and that is PrudentBot's design, put in by hand: E explains the
conjunction (cooperate iff my cooperation is both sufficient and necessary for theirs), and the probe has to be
supplied. Against an agent of the same kind, r is circular without levels [inference, D Prop 10(c)].

**F. Contraction is the common mechanism in copy systems.** Two NicerBots (FairBots grounded with probability ε;
Oesterheld's εGroundedFairBot puts the same constant at a random recursion depth) set x = ε + (1−ε)y and
y = ε + (1−ε)x: a (1−ε)-contraction whose only fixed point is x = y = 1, and against DB x = ε [proven]. The level
clause map is a ½-contraction whose constant part is level 0 (A(ii)). In copy systems the selected outcome is the
contraction's constant, and replacing the constant by D selects DD in both (CUPOD; ε-mass on D for NicerBot)
[proven]. Outside copy systems this fails: FB vs DB has the constant C at level 0 and plays D [proven]. The costs
differ: the level energy costs nothing against DB but needs the read level n0 and the PA+1 skip; ε-grounding
cooperates with DB with probability ε; a naive ε-grounded PB cooperates with itself only with probability 1/(2 − ε),
and grounding only the main recursion restores 1 [proven; M's Monte Carlo, re-run by the synthesis].

**G. Soft versions.**

(i) The γ → 0⁺ minimiser of the PDG form is the Bayes net in which every entry copies its body and flips with
probability ε; for ε = 0 it is the point mass on s*; precisions do not matter [proven, L §1.6 and Prop 6(d)]. P(C)
falls with the read level, because each table "both cooperate below level j, both defect from j on" has energy 2.
FB vs FB at ε = 0.01: 0.99, 0.9802, 0.96099, 0.94207, 0.92352 and 0.90534 at levels 0–5, then 0.820 at 10, 0.372 at 50
and 0.028 at 200. PB against FB, with the DB row clamped: 0.99, 0.9802, 0.95148 and 0.92352 at levels 0–3. [proven;
checked by exact enumeration by the verifier, the toy and me, and by a two-state recursion of mine, §7.3.] These
values belong to the maximum-entropy choice. The inconsistency does not constrain how the level-0 entries are coupled:
a joint that gives both level-0 entries one shared coin, and applies the cpds below, has inconsistency 0 and
P(C) = 0.9705 at level 2 [proven: every edge's cpd holds in every context; checked by the verifier and me].

(ii) Gates on credences: a gate opens iff every lower-level credence of its literal is ≥ 1 − θ, and the entry's
credence is then 1 − η, and η otherwise. For η < θ < ½ every credence is exactly 1 − η or η, so the table is the crisp
one at every level and nothing decays [proven, M §1.3; checked, toy P8: 0.95 at levels 1 and 50 for ε = 0.05].

(iii) Thresholds can be exploited by mixing. An opponent that cooperates with probability 1 − θ passes a gate that
opens at 1 − θ and gains θ(T − R) per game over R; tolerating noise η needs θ > η, so the gain is at least η(T − R)
[proven, D Prop 9(d)]. A linear copy, cooperating with probability ε + (1 − ε)q where q is the opponent's probability,
cannot be exploited by mixing iff 2R ≥ T + S, at ε = 0 [proven, D Prop 9(e); the verifier re-derived
u_Z − R = (1 − q)[(P − R) + q(T + S − R − P)]].

(iv) Ratchet: if each level's observation falls below the gate threshold independently with probability r > 0, the gate
closes almost surely and never reopens [proven, D Prop 7(b), M §2.5].

**H. What each agent computes.** One forward pass of at most n0 + 1 ≤ K + 1 levels, with O(total body size) Boolean
operations per level, stopping at the first repeat of v. Neither agent runs the other's minimiser; each evaluates the
other's clauses at lower levels, so there is no simulation regress. Private weights reach the same table. Without
levels, deciding whether a zero exists is NP-complete for general clause systems [proven, L Prop 6, M §2.6].

### 3.2 What plays Löb's role

Three of the statements above say the same thing.
- The Löb lemma, A(iii): an energy minimum that satisfies □χ → χ at every level satisfies χ, because every level
  reads only lower ones and □ is vacuously true at level 0.
- The contraction, A(ii): the clause map shrinks distances in the "agree up to level n" metric, so it has one fixed
  point, and the constant it injects is level 0, where every □ holds.
- Guarded recursion: Birkedal et al. (LMCS 8(4:1), 2012) prove that in the topos of trees a predicate whose recursion
  variable occurs only under the "later" modality ▷ has a unique fixed point (§2.5, via their internal Banach
  theorem, Thm 2.9), and that the Löb rule (▷p → p) → p holds (Thm 2.7(2)); their ▷ raises a truth value n to n + 1,
  so ▷p holds at the first stage for every p [checked in the source]. On a downward-closed truth set p, the level □
  equals ▷: {n : every m < n is in p} = {0} ∪ {m + 1 : m ∈ p} [proven, one line]. The match is exact when every box's argument
  has a downward-closed truth set, which holds when the closure's formulas contain no negation (FB with CB, DB and FB).
  PrudentBot's DefectBot probe is negated, and there the chain semantics of GL is needed. Positive letters alone are
  not enough: □(p ∧ ¬□⊥) against CB is true on {0} only, while ▷ of its argument is true on {0, 2, 3, …} [proven by
  evaluation].

In plain words: each level may cite only lower levels, and the bottom level has nothing to cite, so everything counts as
certified there; FairBot's clause therefore says "cooperate" at level 0, and the lemma carries cooperation up.
Uniqueness and selection are separate facts. Uniqueness needs only guardedness; selection needs the base.

### 3.3 Checked numerically, not proven

- The search rates in §4: full-gradient and single-flip descent on the level energy miss s* from most random starts,
  while masked descent and repair moves never miss [toy, verifier, pilots].
- On the flat energy, target-only search ends in CC from 46–55% of starts in FB–FB (34–57% in PB–FB, 21–30% in PB–PB)
  and in DD otherwise; two agents searching independently play CD or DC in 49–54% of FB–FB matches, 45–49% of PB–FB
  and 33–39% of PB–PB [toy].
- Two FairBots, each running single-flip descent on its own level energy, end C against D in about half the matches
  [checked numerically, §4 and §7.3].

### 3.4 Conjectures

C1 (a learned searcher on the masked energy generalises in reasoning depth), C2 (a soft PrudentBot without the mixing
exploit), C3 (levels as the step index of an unrolled search), C4 (bounded agents) and C5 (logical induction with an
optimistic day 0) are in §6, with what would test them.

### 3.5 What the two checks changed

Every finding of `verify_math.md` and `toy_result.md`, and two of my own. "Verifier n" is its problem n (§0) and "fix n"
its §3.

| # | finding | source | what I did |
|---|---|---|---|
| 1 | With □_k as a primitive, "first n with v_{n+1} = v_n" stops too early: PB plays C against CB, and Z = □₁⊥ suckers FB and PB | verifier 1; toy §0 item 3, P1 | Applied: n0 counts the clock subformulas, or equivalently starts at k_max (§2). Re-checked with my evaluator |
| 2 | Hence A(iv), B(ii), H and "defecting at a level raises n0 past it" are false as stated | verifier 1 | Applied: all four hold with the corrected n0, and the last now follows from B(ii) |
| 3 | A copy with a noisier cpd exploits the soft read-off; thresholding at ½ is a credence gate with θ = ½; "read at n0" picks FB's vacuous level | verifier 2, fix 2 | Applied (§2, §5.3). Added: if both agents sample at level 1, the noise costs the noisy agent as much as its partner |
| 4 | The selection is the vacuous base: with base 0 every theorem of A and D holds, with DD | verifier 3, fix 3 | Applied (TL;DR line 3, §5.1). Re-checked by brute force |
| 5 | Where a full search decides, the start decides it | verifier 3 | Applied (§4) |
| 6 | "Reached the table" and "played the table" were mixed | verifier 4, fix 4; toy §0 item 5 | Applied: both measures in §4; D(iii) is stated as "does not reach s*" |
| 7 | Target-only search uses the level order, so "none of the solvers sees the level order" is false | verifier 4, fix 4 | Applied (§5.2) |
| 8 | The masked mean-field fixed point falls below ½ from level 15 | verifier 4 | Applied (D(ii)); re-checked |
| 9 | A(iii)'s marginals came from a stalled solve | verifier 5, fix 5; toy P2 | Applied: exact values in G(i), re-checked by enumeration. The minimiser claim stands: the toy's solves started at the Bayes net stay within 4.1e-5 of it |
| 10 | Certify the joint solve against F(BN) or a KKT residual | verifier fix 5 | F(BN) applied. The KKT residual is rejected as a certificate: the toy's 14 solves on 10–12-bit joints had residuals ≤ 4.5e-7 and still sat 1.9e-4 to 9.0e-3 from the Bayes net, because the inconsistency is flat along couplings of the level-0 entries (curvature O(γ)) |
| 11 | γ → 0⁺ sets the soft marginals above level 1 | verifier 5; toy §3 | Applied (G(i), §5.3); re-checked |
| 12 | Level □ = ▷ needs box arguments without negation; positive letters are not enough | verifier 6 | Applied (§3.2); counterexample re-evaluated by hand |
| 13 | "The selected outcome is the contraction's constant" holds only for copy systems | verifier 6 | Applied (F) |
| 14 | Only agents positive in □ cooperate with DB at level 0 | verifier 6 | Applied (B(iii)) |
| 15 | Theorem E puts the DefectBot probe in by hand | verifier 6 | Applied (E) |
| 16 | Toy P4's 50% bar fails in FB–FB because its n0 = 0 read-off is lenient; no rate reaches the 90% refutation bar | toy §0 item 5, §2 | Applied (§4) |
| 17 | Full search also fails on CUPOD, whose answer is DD | toy §3 | Applied (§4) |
| 18 | Plain-gradient failures are not yet stationary; some S5 failures are plateaus; S3's precision choice shifts the FB–FB rates | toy §2, §6 | Recorded in §4; no claim here rests on them |
| 19 | Synthesis item 8 calls the levels "necessary by item 2" | mine | Narrowed: Theorem C shows that something beyond the flat energy is necessary (C) |
| 20 | Synthesis §2.2: private "weights, solvers and randomness" do not matter | mine | Narrowed: weights do not; solvers do unless target-only (§5.2); in the soft model ε must be public (§5.3) |

## 4. The toy and its result

**What it is.** `toy_spec.md` specifies one CPU run. Energy-based FB and PB play the one-shot PD against CB, DB, FB
and PB. Each agent builds its energy from both published definitions and solves it with its own seed and private
precisions, using one of five solvers, with the forward pass (S0) as reference:
- S1, a full-joint convex solve of the PDG form (softmax logits, L-BFGS-B);
- S2, masked mean-field descent (target-only);
- S3, full mean-field gradient descent (plain and natural gradient, private and unit precisions);
- S4, crisp repair moves (level-ascending, level-descending and random order);
- S5, crisp single-flip descent accepting ΔE ≤ 0 (unit and 2^n weights).

It also runs ablations (PB_PA, CUPOD, an FB whose □ is false at level 0, and the flat energy) and a sweep over 500
random modal opponents, and scores eight pass criteria, each a claim of the synthesis. N = 4 and ε = 0.01 unless noted.

**How it ran.** 2026-11-01 from 22:38:03 to 22:45:28Z on General: 445 s at a load of 31–43 on 16 cores, CPU only, nice
19, ionice idle, one thread. The script `toy/guarded_pd_toy.py` executes L's definitions verbatim; the output is in
`toy/out.txt` and `toy/out.json`.

| criterion | verdict | measured |
|---|---|---|
| P1 outcomes, payoffs | pass with the corrected n0; fail with the literal rule | S0, S1, S2 (202 starts per agent) and S4 (900 runs per agent) give the B(i) table and PB 12, FB 10, DB 8, CB 6. Under the literal rule PB plays C against CB in all four solvers |
| P2 joint solve = Bayes net | fail as specified | KKT residual ≤ 4.5e-7 in all 18 solves, but the marginals sit 1.9e-4 to 9.0e-3 from the Bayes net on the 14 joints of 10–12 bits; started at the Bayes net, all 18 stay within 4.1e-5 with F below F_BN |
| P3 repair moves | pass | 100% reach s* in every order: ascending sweep 10 moves (FB–FB) and 20 (PB–FB); descending at most 42 and 84; random order mean 52 and 117 |
| P4 full search fails | fails as written, not refuted | table below |
| P5 flat energy | pass | P(C) = 0.5000 and μ(CC) = μ(DD) = 0.4950; identical flat edge tables for FB–FB and CUPOD–CUPOD, level outcomes CC and DD |
| P6 PA+1 | pass | DD for PB_PA–FB, PB_PA–PB_PA and the pessimistic FB against FB, in S0, S1, S2 and S4, from every start |
| P7 unexploitability | pass | 0 sucker outcomes at n0 under both rules. Its opponents use plain □ only; against them the literal rule suckered no one, but PB failed to exploit 39 of the 500 |
| P8 soft gates | pass | ε = 0.05, PB vs FB: the Bayes net gives 0.9050 at n0 = 1 and 0.0504 at N = 50; credence gates give 0.95 at both |

**The search result.** Each cell gives, over random starts and over both agents and all variants of the solver, the
share of final states that play the right outcome at n0, then the share equal to s*:

| search | FB vs FB | PB vs FB | PB vs PB |
|---|---|---|---|
| S2 masked mean-field | 100% / 100% | 100% / 100% | 100% / 100% |
| S4 repair moves | 100% / 100% | 100% / 100% | 100% / 100% |
| S3 full mean-field, 4 variants | 43–82% / 7–14% | 8–21% / 2–8% | 2–10% / 0–3% |
| S5 single-flip descent, 2 weightings | 46–52% / 3–21% | 15–33% / 1–9% | 14–22% / 0–4% |

How to read it:
- "Equal to s*" stays at or below 21.3% for every full search, so D(iii) holds as "full search does not reach s*".
- "Plays the right outcome" is much higher in FB–FB because the read-off is lenient there: n0 = 0, so whenever both
  level-0 entries of a state are C, v_1 = v_0 and the read-off is CC whatever the higher levels say [proven, one
  line]. From the uniform start, full descent ends at C D D D D in both rows and still "plays CC" (verifier §I).
- Where a full search does find the cooperative table, its start chose it: full mean-field descent on FB vs FB reaches
  the whole table from 100% of all-C starts and from 0% of the starts at u = 0, at u = −6 and from N(0, 0.5²) (verifier
  §F).
- The failure is not tied to cooperative answers. On CUPOD vs CUPOD, whose answer is DD, full descent reaches s* in
  5–16.5% of runs, and the realised profile is right in 41–48.5% of matches. In FB vs CB it reaches s* in 34.5–35%
  and plays right in 69.5–73%. In the four matches whose energy-based answer is D (PB–CB, PB–DB, FB–DB, PB_PA–FB) it
  plays right in 94.5–100%.
- The failed S3 runs with natural gradient end at stationary points (|∇F| ≤ 2e-16) with F ≈ 4.61–4.63 at unit
  precision, about one violated clause, against F = 0.08–0.16 when full descent is warm-started at the masked fixed
  point. Warm-started runs play the table. So under per-variable beliefs the low-energy state is the right one, and the
  random-start search misses it.
- Unexploitability does not survive a missed zero. In my check, two FairBots each ran S5 on their own copy of the FB–FB
  energy from independent random tables (3,000 matches): the realised profiles were CC 26.4%, C against D 25.1%,
  D against C 24.4% and DD 24.1% with unit weights, and 24.6%, 25.8%, 25.7% and 23.9% with 2^n weights [checked
  numerically, §7.3]. Each FairBot is the sucker in about a quarter of the matches.

**Limits.** One run, unreviewed. Only the spec's agents, at N = 4 (2–5 for S1, 40 for the sweep) and ε = 0.01 (0.05 in
P8). The S3 precision choice is a judgment call: unit precision gives FB–FB read-off rates of 43–49%, private
precisions 66–82%. Plain-gradient failures were not yet stationary after 6000 steps (gradient up to 3.6e-4), and with
unit weights 20% (PB–FB) and 33% (PB–PB) of the failed S5 runs end on plateaus rather than at strict minima. Nothing was
learned. E, F, G(iii)–(iv) and H's costs were not tested.

## 5. What fails and why

### 5.1 The cooperation is written in, in the form Löb uses

For agents positive in □, the level-0 clauses are unit clauses: their bodies are the constant 1, so they say
"c⁰ = C". The proofs of uniqueness (A(i)–(ii)) and of search (D(i)–(ii)) use only that level-0 bodies are constants.
They therefore hold unchanged if level 0 certifies nothing ("the empty AND is 0"), and then the unique zero of FB vs FB,
PB vs FB and PB vs PB is DD [proven by inspecting the proofs; checked numerically: one zero under either base, and DD
under base 0, §7.3]. Uniqueness and independence from the search say nothing about cooperation; the base alone decides
between CC and DD.

The base is what "held at every lower level" means when nothing is lower, and level 0 is GL's bottom world, where □⊥
holds; Löb's theorem rests on the same fact. So the model reproduces Bárász et al. exactly, but inconsistency
minimisation does not produce the cooperation: with base 0 the energy is just as consistent. In D's terms the base is the
optimistic grounding G1, written as a hard term instead of an initialisation. The synthesis says this in its item 8.
Its item 4, which claims to resolve the tension in `lob-neural/li.md` §5.1 (policy independence needs one minimiser,
Löbian cooperation seemed to need a choice among several), takes the first horn that li.md already names: "the
selection is written into the energy". Its R3 passage calls the base "not a tunable term", which is true, and the base
still decides the outcome.

This matters for your question because our PDG/atom energy has no level index. A FairBot clause added to it is the
flat case of Theorem C, whose minimiser with γ > 0 is ½ CC + ½ DD.

### 5.2 Generic search misses the zero, and the search that finds it is forward chaining

Full updates also infer against the edges. They revise lower-level beliefs ("did the other really cooperate at
level 0?") to fit a higher-level defection, and end in a self-consistent story, one clause short of the zero, in which
cooperation is explained away [inference; the local minima are proven by enumeration, D(iii)]. Target-only updates
infer only along the edges, from context to target. Their stop-gradient needs each edge's split into target and
context, and in the level energy the context is exactly the lower levels, so target-only search is forward chaining
along the level DAG. The synthesis and the toy spec say that none of the solvers sees the level order; S2 and S4 do.
In our framework the split is part of the PDG (edges have directions) and of LIR's control mask, so the design is
legitimate, but the result is "forward chaining finds the zero", which is weaker than "any search finds it".

Two consequences for a neural searcher [inference]. Gradient training straight on this energy gives full updates, so
the mask has to be built into the architecture or into the update rule. And B(ii) protects only the zero: a searcher
that stops elsewhere has no guarantee, as the two-FairBot check in §4 shows. Your R18 condition, a "complete
information trace" that every full-support policy visits over and over, is met by the unrolled table; it suffices for
target-only search (D(i)) and not for full search (D(iii)).

### 5.3 Soft versions: decay, a tie-break that sets the numbers, and an exploitable read-off

Decay. Under the Bayes net, P(C) for FB vs FB at ε = 0.01 falls from 0.99 at level 0 to 0.372 at level 50 (G(i)), and
the masked mean-field fixed point falls below ½ from level 15 (D(ii)). A noisy reasoner that reasons deeper is more
likely to "disprove" cooperation, and the running AND in □ makes the disproof permanent [inference, L §2.5]. Read
shallow, at n0 or just above it.

The tie-break. The decay curve belongs to the maximum-entropy coupling of the level-0 entries (G(i)). In the soft form
the γ → 0⁺ tie-break therefore sets part of the target, which bears on your R3. In the crisp form it sets nothing.

The read-off. The synthesis read the soft state at n0 by thresholding at ½. In FB vs FB, n0 = 0, where FB's clause is
vacuous: its entry is "⊤ plus noise" and does not depend on the opponent. A copy Z that publishes FB's formula but
keeps a larger flip rate ε_Z then gains. Payoffs (FB, Z) at ε = 0.01, with Z sampling its level-0 entry independently
of FB [verifier §G; recomputed, §7.3]:

| ε_Z | FB samples level 0 | FB thresholds at ½ | FB samples level 1, ε_Z public | FB samples level 1, ε_Z private |
|---|---|---|---|---|
| 0.01 (exact copy) | 2.99, 2.99 | 2.97, 3.02 | 3.01, 2.96 | 3.01, 2.96 |
| 0.1 | 2.72, 3.17 | 2.70, 3.20 | 2.91, 2.87 | 2.74, 3.14 |
| 0.3 | 2.12, 3.57 | 2.10, 3.60 | 2.62, 2.60 | 2.13, 3.54 |
| 0.45 | 1.67, 3.87 | 1.65, 3.90 | 2.35, 2.34 | 1.68, 3.83 |

Thresholding is G(iii) with θ = ½: Z gains ε_Z(T − R). The one mixing-proof read-off in the table samples at a level
where FB's query about Z is live, with ε_Z public. FB's credence is then ε + (1 − 2ε)q, a linear copy of Z's
probability q, and Z's best reply is q = 1 [checked numerically, §7.3]. If both agents sample at level 1, their
cooperation probabilities are equal, x = y = 1 − ε − ε_Z + 2εε_Z, so the noise costs the noisy agent as much as its
partner [proven, one line]. With ε_Z private there is no protection.

Credence gates avoid the decay (G(ii)), but they bring back the mixing exploit, at a gain of θ(T − R) ≥ η(T − R)
(G(iii)), and the ratchet (G(iv)).

### 5.4 What the model assumes

- A shared modal language with exact evaluation. Each agent must instantiate the other's template against DefectBot and
  trust the answer; for a learned energy this is the unmodelled hard part (M §4.3).
- Near-copies lose cooperation, as in PA: PB vs PB_PA and FB vs FB⁺ end in DD [checked numerically, §7.3]. Nobody is
  exploited, but the cooperation is gone.
- Two players, one shot, nothing learned. The levels encode PA-provability, not resource bounds, and a bounded
  PrudentBot is open (Critch, Dennis and Russell 2022, OP9, second-hand via `lob-neural/lit.md` B2); the level model
  says nothing about it.

### 5.5 The preference reading needs the dependence it is meant to explain

E reproduces PB's clause against fixed-rule opponents (CB, DB, FB) with no recursion, but only given a dependence edge
read from the other's definition, and only with the DefectBot probe put in by hand. Against an agent of the same kind
the dependence is "it minimises its energy given its model of me", which is circular without levels and a base
(D Prop 10(c)). Choosing among several zeros by one's own preference is exploitable: a spiteful FB-like opponent picks
DD while you pick CC, so you play C against D. Cooperative oracles avoid this with one joint, stratified selection
[proven, M T5; Garrabrant 2017 second-hand].

Your phrase, checked against the definitions: in the consistency reading it holds at the unique zero of the level
energy, and it holds there because of the base; without levels it holds equally at DD (C(ii)). In the preference
reading it holds against fixed-rule opponents given a dependence edge, and against an agent like itself that edge
needs the levels again.

## 6. Open questions worth your time

1. Can the levels come from something we already have? Everything here hinges on a level index with a vacuous base.
   Two candidates, both [conjecture]: the step index of an unrolled search whose initial state is "everything
   certified", read at the first repeat of its own state (C3); and the days of a logical-inductor-style market whose
   traders' gates read only earlier days, with an optimistic day 0 (C5; with concurrent prices the same inductor
   would implement FairBot's constraint but not Löb's selection). Either would make the level index come from our
   own machinery, for instance search depth or aggregation order. Neither removes the base: it becomes the search's
   initial state or day 0, which puts the selection into the search's start, where your R12 says the target should
   not live [inference]. Open whether some reading of the base avoids this.
2. A learning toy, which could also serve as your "learning search and generalising" example (C1, [conjecture]).
   Train a small recurrent network without labels to minimise the masked level energy on random modal matches with
   K ≤ K_train, and test the read-off on matches with K up to 2·K_train. The conjecture: it learns the running AND
   and the vacuous base and extrapolates in reasoning depth, and trained with full gradients it does not. It fits on a
   CPU, the forward pass scores it exactly (the scores are not used in training), and the random-match generator
   exists (L's `rnd_mod`, the toy's sweep). Refuted if masked training fails to extrapolate in K, or if full-gradient
   training extrapolates as well. Not part of `toy_spec.md`, and untested.
3. Do we accept the vacuous base in a world model? It is the truth value of an empty universal, not a weight or a
   regulariser, so R3 does not obviously exclude it [inference]; but it decides the outcome. If you accept it, the
   construction is complete for modal agents. If not, the known alternatives cost something: ε-grounding cooperates
   with DefectBot with probability ε (F), and preference edges are safe only with one joint selection (§5.5). This is
   your call.
4. Can a search find the zero without being given the level order? Full gradients fail, and the mask that works is the
   level order. Open: whether a search that sees only the energy (for example annealed single-flip descent) finds s*
   reliably, or whether a learned searcher needs the DAG made explicit. The answer sets how much structure a neural
   searcher must be handed.
5. A soft PrudentBot without the exploits (C2, [inference] from G(iii) and M's GPBp; unchecked). PB_lin plays C with
   probability 1[the □₁ DefectBot probe passes] · (ε + (1 − ε)q). It should cooperate with itself and with NicerBot
   with probability 1, cooperate with DB with probability ε, defect against CB, and resist mixing in its self-query.
   Its probe is still a threshold, so an opponent could cooperate with DB with probability just below θ. Open: a
   DefectBot probe that is a linear copy too.

## 7. Sources

### 7.1 In-house inputs (read, not modified)

- `/home/agent/workspaces/mats/work/energy-prudentbot/`: `synthesis.md`, `verify_math.md` (with its recorded output)
  and `toy_result.md`, read in full; `toy_spec.md`; `context.md` §0–§3 and §6–§7; `formal_logic.md` §0–§6;
  `formal_dynamics.md` §0 and Props 9–10; `formal_market.md` §0, T2, T5 and T6; `toy/out.txt` (the S3 and S5 lines)
  and the keys of `toy/out.json`.
- `/home/agent/workspaces/mats/work/lob-neural/li.md` §5.

### 7.2 Literature I checked today (2026-11-01, about 23:05Z)

- M. Bárász, P. Christiano, B. Fallenstein, M. Herreshoff, P. LaVictoire, E. Yudkowsky, "Robust Cooperation in the
  Prisoner's Dilemma: Program Equilibrium via Provability Logic", arXiv:1401.5577v2 (dated 6 April 2021). PDF text
  extracted with ghostscript. Read: Algorithms 4 (FairBot) and 5 (PrudentBot); Thm 3.1; Thm 3.2 and its proof,
  including "PA+2 ⊢ [PB(CB) = D]"; the remark after Thm 3.2; footnote 3; the definition of unexploitable; the outline
  on third parties (p. 3); the PA+n tower (§2). The extraction lost the section 4 theorem numbers, so Thm 4.7 and the
  third-party theorem are cited as L and M number them.
- L. Birkedal, R. E. Møgelberg, J. Schwinghammer, K. Støvring, "First steps in synthetic guarded domain theory:
  step-indexing in the topos of trees", LMCS 8(4:1), 2012, arXiv:1208.3596. PDF text: §1 (truth values on constant
  sets are downward-closed sets of worlds; ▷ "contains n+1 if ϕ contains n"; guarded recursive predicates are well
  defined; "an associated Löb rule for induction, (✄ϕ → ϕ) → ϕ"); §2.2 (▷ maps n to min(m, n+1)); Thm 2.7(2);
  Thm 2.9; §2.5.
- github.com/machine-intelligence/provability, `src/Modal/Formulas.hs`: `handleBox = scanl (&&) True` and
  `handleDia = scanl (||) False`.

### 7.3 Checks I ran (2026-11-01, 22:58–23:15Z; CPU, nice 19, ionice idle, one thread; inline scripts, not saved)

- Crisp (17.6 s), with my own evaluator, not L's, the verifier's or the toy's code. Read-off level under the literal,
  clocked and unfolded rules for 14 matches (PB vs CB: literal n0 = 0 plays CC, corrected n0 = 2 plays DC; FB and PB
  vs □₁⊥: literal n0 = 0 plays CC, corrected n0 = 3 plays DD). Base 0: DD at every level for FB–FB, PB–FB and PB–PB.
  Brute force: exactly one zero under base 1 and under base 0 for FB–FB (N = 4), PB–FB (N = 2), PB–CB (N = 2) and
  PB–PB (N = 1). 1,500 random opponents using □_k with k ≤ 3, each against FB and PB, levels 0–60: corrected and
  unfolded n0 agree in 3,000 of 3,000; n0 ≤ K in 3,000 of 3,000; sucker outcomes 0 (corrected rule) against 211 (FB)
  and 210 (PB) under the literal rule.
- Soft (0.4 s). Exact Bayes-net enumeration: FB–FB 0.99, 0.9802, 0.96099, 0.94207, 0.92352, 0.90534; PB–FB with DB
  clamped 0.99, 0.9802, 0.95148, 0.92352. Shared-coin joint: 0.9705 at level 2. A two-state recursion for FB–FB:
  0.81973 at level 10, 0.37237 at 50 and 0.02777 at 200. Masked mean-field fixed point: first level below ½ is 15. The
  payoff table of §5.3, and Z's best reply against the level-1 linear copy (q = 1, payoff 2.97).
- Two FairBots, each running S5 independently on its own FB–FB energy: N = 4, 3,000 matches, 4,000 proposals, unit and
  2^n weights (profiles in §4).
- Near-copies: PB vs PB_PA plays CC, CD, then DD from level 2; FB vs FB⁺ plays CD, DC, then DD from level 2.

### 7.4 Second-hand (taken from the formal notes or `lob-neural/lit.md`; not re-read by me)

R. J. Buehler's notes on Boolos, *The Logic of Provability*, ch. 7; the SEP entry "Provability Logic"; Critch,
arXiv:1602.04184; Oesterheld 2019 (*Theory and Decision* 86) and arXiv:2211.05057; Taylor's NicerBot comment;
Garrabrant, "Cooperative Oracles" (2017); Critch, Dennis and Russell 2022 (OP9); Garrabrant 2018, "Prisoners' Dilemma
with Costs to Modeling".
