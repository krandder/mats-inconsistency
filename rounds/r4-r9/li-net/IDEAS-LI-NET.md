# Logical induction network: what the math says and what to build

eurisko, 2026-10-02. A summary of five math notes, each attacked and edited by a skeptic pass. The notes tag every claim as proved, known, conjectured or refuted; this page keeps the conclusions and marks conjectures. Paths are at the end.

**TL;DR.** Your PDG idea works: it projects the price table onto the fair (arbitrage-free) tables. Add a house view (a prior over worlds) and its fixed point is exactly what LI asks of a market told everything at block 0: coherent, with some weight on every possible world. The block that follows is a market of traders who each hold one complete world; its prices are coherent at every block, and thinking time goes into finding worlds. **Decision for you:** whether to run the first experiment in section 4 (about a GPU-day, estimated).

## 1. What the math settled

- Your PDG idea is a fair-price projection. With the axioms at infinite confidence, minimising the inconsistency moves the table to the nearest coherent table (in log score). The answer is unique, entailed sentences go to 1, refuted ones to 0, and every identity holds.
- The inconsistency is the most risk-free profit one arbitrageur can take, in one shot, from per-sentence LMSR market makers who ignore the axioms; its trades leave prices exactly at the projected table. The arbitrageur is the other half of the optimisation, so the construction needs neither history nor an adversary.
- With everything given at block 0, the LI criterion shrinks to one fact: a market that stays put is safe from every trader, however clever, exactly when it is coherent and gives every possible world some weight (for sentences that do not refer to prices). The projection supplies the first half; a house view supplies the second. The rest of LI (convergence, theorems priced before proofs, no expected drift, self-trust) is about the path through the blocks and has to be built or trained in.
- Coherence cannot pick frequencies, because every coherent table is fully consistent. A house view (a prior over worlds) fixes them, and in PDG terms learning it and fitting the labels with cross-entropy are one objective.
- The hard part moves to search. Checking coherence is probabilistic SAT (NP-complete). Up to about 12-16 atoms a block can list the worlds and, given a house view over whole worlds, answer in one step; beyond that it must search for worlds, and that is what thinking time is for.
- A loss only at block 20 does not control block 500. A free block can hide a clock (error 0 at block 20, 0.86 at 500), and a proven solver (Hildreth) given a learned step size and momentum beat the safe setting at block 20 (error 0.013) and reached 10^35 at block 500.

## 2. The architecture: a market of world-holders

Each trader holds one complete world, a true/false value for every atom. A sentence's price is the share of traders whose world makes it true. Such prices cannot break a logical identity, because every world obeys all of them. The network's job is to find the right worlds.

**One block** (several layers, the same weights in every block):
1. Propose: message passing over atoms, axioms and sentences; each of N seats suggests a move to a nearby world. Learned.
2. Check: a clerk tests the new world against every axiom and refuses it if one fails. Exact, linear time, fixed.
3. Accept: a Metropolis coin decides whether the seat moves, so that over many blocks each world is held in proportion to the house view.
4. Price: the seats' vote is folded into a running average. An average of coherent tables is coherent, so prices are coherent at every block.

Between blocks the state is the price table, each seat's world, the block-0 intuition table and the block number. Nothing attends to past blocks. A seat's world is a holding, the kind of state the LI proofs appear to need (conjecture R7, no-history note).

**Fixed point:** the PDG answer for three claims: the house view, the axioms at infinite confidence, and optionally the intuition table at finite confidence, which tilts the house view toward it (convergence with the tilt is open, section 5 item 8). With the intuition off it is the model frequency: entailed 1, refuted 0, anything else the house-view share of possible worlds where it holds.

**Variants.** For small theories the seats are all the worlds, the clerk is a mask, and one attention step is exact. Your projection on its own (reconciling a per-sentence intuition table with the axioms) is then exponential weights over the worlds, with a proved rate; on a 6-atom toy it took 178-260 blocks to reach 0.001 (a step 4 times larger was 4 times faster but cycled on 9 of 20 extreme tables). In the arbitrageur variant (Frank-Wolfe) the proposer seeks the checked world where the market's surest bets lose against the intuition table. Prices then reach the projection of that table at about 1/t with exact proposals, and each block certifies how far it still is from it.

| Learned | Fixed |
|---|---|
| proposer: where each seat looks next | clerk: the axiom check |
| house view: a scorer of whole worlds | price = share of held worlds, running average |
| intuition head: the block-0 guess, used until a world passes the clerk | Metropolis accept rule |
| per-sentence liquidity: how much logic must pay to move an intuition price | weights shared across blocks |
| step sizes, only inside the proven range (built into the parametrisation) | |

**From 20 to 500 blocks.** Nothing learned changes with depth; each block only adds checked worlds to the average, so error should fall roughly like 1/√(independent worlds seen). The skeptics' results add two guards. Cap steps by construction (step = cap × sigmoid(a)), since a free step plus momentum acts as a hidden clock. Put the loss at a random block from 1 to 20 instead of only at 20: each block's price is then trained to be the best guess given what that many blocks can see, and those guesses form a martingale over depth (proved for ideal predictors; to be measured for a trained net). What limits the design is search: near the satisfiability threshold the possible worlds split into far-apart clusters (recalled, unverified), seats can stick in one, and then more blocks stop helping.

**Which LI properties it can have.**

| LI property | Here | Reason |
|---|---|---|
| Coherence (identities, sums, Fréchet bounds) | yes, every block, any size | prices are a vote of checked worlds |
| Entailed 1, refuted 0 | yes, from the first checked world | every possible world agrees |
| Never certain without proof | only once seats cover the worlds | a sentence true in every world found so far is priced 1. LI keeps this property at finite time and gets coherence only in the limit; this design does the reverse. Blending in the intuition table trades one for the other |
| Convergence | yes, built in, with capped steps | traders that see only today's prices cannot force it (proved) |
| Frequency on independent sentences | yes, under the house view | LI's frequency along a sequence becomes calibration over the training data |
| Theorems priced before proofs finish | only what the data teaches, within the net's compute budget | inside one run nothing forces it |
| No expected drift, self-trust, persistence | ensemble forms, from the random-depth loss | martingale over blocks |
| Paradox resistance | only for sentences about the net's own block-t price; settles within δ below the threshold after about 1/δ blocks | needs continuous traders; other traders shift the point; "exactly at the threshold" is refuted |
| No single trader can profit | no, a fixed point protects only the traders' sum | needs wealth carried across blocks |
| Learning from facts that arrive later | no, by your design | nothing arrives during the blocks |
| Exactly the right table on big theories | not guaranteed; if P ≠ NP, no fast block can in general | coherence holds; the right point depends on search |

## 3. Ideas worth trying

1. **World seats with a learned proposer** (section 2). You'd like it because it meets all three of your constraints (no attention to past blocks, no adversarial market maker, no enumeration of traders) and gives thinking time one clear job. Exciting result: zero coherence violations from block 1 to 500, error still falling after block 20, and the learned proposer beating plain random-walk seats on 24-40 atom theories.
2. **Rule traders in a ReLU layer.** Each hidden unit trades one rule, say "P(A and B) ≤ P(B)". Its activation is its position, carried to the next block: it adds to the position while the rule is broken, and the ReLU keeps it from going short. Prices are the intuition table minus the positions' combined push. This is exactly Hildreth's method, and it reached the nearest rule-abiding table in 7-9 sweeps on the toy (a sweep is a few layers, one per group of rules that share no sentence). You'd like it because these are traders living inside a layer, with holdings as their state. Limit: local rules reach a relaxation, 0.23-0.24 off coherent on a 6-atom toy. Exciting result: a learned unit that picks extra conjunctions to price (the junction-tree trick) closes that gap on structured theories without listing worlds. Side bet (my conjecture): run beside the seats, the rule traders give an outside price and the seats an inside one; LI judges finite-time prices against the outside, and the gap may work as a visible band of logical uncertainty.
3. **House view and liquidity.** Learn the house view as a scorer of whole worlds, plus a liquidity per sentence (how much logic must pay to move its intuition price). You'd like it because it is one knob with a market meaning, and cross-entropy training already learns the house view. Exciting result: on data from a hidden non-uniform prior, end prices match exact model frequencies, and liquidity comes out low where intuition is unreliable.
4. **Clock-proof training.** Loss at a random block from 1 to 20, steps capped by construction, and a small penalty on the change between consecutive blocks. You'd like it because it closes both clock loopholes the skeptics found, and the martingale it creates brings no-expected-drift and self-trust (ensemble forms) as a by-product. Exciting result: no fresh theory gets worse from block 20 to 500, and the average price change from block t to t+k is zero within noise in every price bin.
5. **Arbitrage meter.** Three numbers per block: the table's inconsistency (exact for small theories; for big ones bracketed between a rule-based lower bound and a found-worlds upper bound); the profit of held-out attackers, including traders that see the block number (allowed, since it is not history); and the sum of squared price moves, the profit of a trader that forecasts the block's own update, which grows without bound exactly when that trader exploits the market. You'd like it because it tests the LI criterion itself; a zero fixed-point residual certifies that the traders' sum cannot gain in any world, with no list of worlds. Exciting result: attackers' profit stays flat out to block 500.
6. **Timeliness probes.** A family where a visible pattern implies entailment (a chain of equivalences with a true anchor) and one where it takes real work (a chain of XORs with random constants), scored by derivation depth at blocks 20, 100 and 500. You'd like it because it is LI's "before the proof finishes" in block terms, and it separates learned patterns from bluffing. Exciting result: long-derivation theorems priced near 1 at block 20 in the pattern family; in the XOR family, the base rate until derived, or better, a shortcut that beats step-by-step deduction (parity takes about log(length) blocks by pointer doubling).
7. **Wealth for paradoxes.** Sentences about the net's own price ("my block-t price of this is below 0.8") and traders of different sharpness, with wealth carried in the state. You'd like it because it is the most LI-specific property and needs no history. Exciting result: the price settles just below 0.8, within the sharpest trader's resolution, while the coarse traders go broke. The static part is proved; the wealth flow has only been simulated.

## 4. The first experiment

**Question.** On theories too big to list inside the block, does the world-seat block stay coherent and keep improving out to 500 blocks, where a free block of the same size does not?

**Data.** The "logic worlds" generator planned on 10-01. Random 3-clause theories over 16 atoms (3.0-4.2 clauses per atom, satisfiable only), with 24 queries each: 8 literals, 12 random formulas over 2-3 atoms (and, or, not, implies), and 2 entailed and 2 refuted where the theory has them, with derivation depth recorded. Answers come from listing all 65,536 worlds offline: 1, 0, or the share of models where the sentence holds (uniform house view this first time). Tests: held-out 16-atom theories, and 24-atom theories (16.8M worlds, brute force offline or an existing model counter) as the harder set. About 100k training theories, 1-2k per test set.

**Arms.** One shared backbone: message passing over atoms, axioms and sentences, 4 layers per block, width 256, weights shared across blocks.
- A. Free block with a price head, loss only at block 20 (the 10-01 plan).
- B. Same, loss at a random block from 1 to 20 (isolates the training fix).
- C. World seats: 64 seats, the backbone as proposer, clerk, Metropolis accept, running-average price, and an intuition head (trained like B) until a world passes the clerk. The proposer learns without labels, rewarded for large accepted moves.
- R. Reference: C with plain single-flip random-walk seats (nothing learned, same moves per block), plus the exact answers.

**Metrics** at blocks 1, 5, 20, 100 and 500, split into entailed, refuted and undecided:
- Brier score and log loss (prices clipped to [0.001, 0.999]);
- coherence violation: exact distance to the coherent set at 16 atoms; largest broken identity (negation, inclusion-exclusion, Fréchet bounds, entailed = 1) at 24;
- overconfidence: share of undecided sentences priced above 0.999 or below 0.001;
- extrapolation: share of test theories worse at 500 than at 20 by more than 0.01, and mean price change from 100 to 500;
- search (C, R): distinct checked worlds and acceptance rate by block;
- timeliness: error by derivation depth at blocks 20, 100 and 500.

**Success.** C and R show zero violations after the first checked world, their error keeps falling from 20 to 500 on the 24-atom set, and no theory gets worse; C beats R per block; A shows violations and stalls or degrades after 20. Exciting: C at block 500 beats A at its best block on the 24-atom set.

**Failure, and what it would mean.**
- C no better than R: the market supplies the properties, but the learned proposer adds nothing yet, so the proposer is the next work.
- C overconfident, with poor log loss at 24 atoms: the coherence-versus-certainty trade in section 2 bites. Turn on the intuition blend and measure it.
- A or B as coherent as C and still improving at 500: a free block learns this unaided, and the structure is worth only its guarantees.
- Many 24-atom theories with no checked world by block 500: the set is too hard; lower the clause ratio.

**Cost** (estimate, not measured): answers take minutes at 16 atoms and a few CPU hours at 24; training is a GPU-day or less.

## 5. Unresolved math, ranked

1. Do capped steps plus the random-depth loss guarantee that a block trained at 20 behaves at 500? Proved for contractions, refuted for free steps, conjecture for capped templates. The 20/100/500 plan rests on it. (block-dynamics R9)
2. Coherent at every block versus never certain without proof, when worlds cannot be listed: how much overconfidence remains after N found worlds, and what is the best price between inside (found worlds) and outside (derived rules)? New here; no math yet.
3. Block budgets: no a-priori rate for the arbitrageur near prices 0 and 1, no constant for the exponential-weights rate, unknown seat mixing times near the clause threshold. (pdg-propositional R6, block-dynamics R5)
4. Quantitative timeliness: do rule-trader sweeps grow with an entailed sentence's derivation length, and is there an explicit error bound at block t against traders limited to t blocks? (block-dynamics, li-criterion-finite)
5. Does a trained net behave like the ideal predictor at each depth, so that its prices form a martingale across 20, 100 and 500? Needs a usable definition of "what N blocks can see"; the obvious one is degenerate. (frequency-and-prior R8, R9)
6. LI without history: a full proof that traders seeing only today's prices cannot force convergence (open for traders whose bets change sign), and whether every trader in the LI proofs needs only today's prices, its holdings and its wealth. (no-history R6, R7, R8b)
7. Beyond about 16 atoms: which relaxation level suits a given block budget, and whether any fast approximation guarantee exists beyond tree-like theories (conjecture: no). (pdg-propositional R6)
8. Does the seat market converge when the house view is tilted by the intuition table and the tilt is updated as it goes? Conjecture. (this page)
9. Paradoxes: can one fixed point be randomised so a self-referential sentence is true at the right frequency, and can a block do it? (no-history)
10. Conditional prices (the price of B given A) as conditional LMSR markets: expected, not derived. (pdg-propositional)

## Where the detail is

`/home/agent/workspaces/eurisko-study/li-net/math/`, each note with scripts, outputs and a closing skeptic section: `pdg-propositional.md` (projection, LMSR duality, hardness, Frank-Wolfe), `block-dynamics.md` (which updates converge, rule traders as ReLU layers, clock counterexamples), `li-criterion-finite.md` (what MarketMaker guarantees, the residual test, the static collapse), `no-history-traders.md` (what history-free traders keep and lose, martingales, paradox traders), `frequency-and-prior.md` (why a house view, cross-entropy as the same objective, timeliness from data).

## Kelvin, 2026-10-05: conditional settlement by induction (hypothesis), proposer bets on acceptance
- Rejects inverse-propensity payouts. Hypothesis: a position on p(A|B) taken when p(B) is low is settled to the then-current
  market price of p(A|B) whenever p(B) has doubled since the position was opened; successive doublings anchor the
  prediction until some position is opened at high enough p(B) to settle on B itself.
- Proposer pays by betting on the proposal being accepted (a forecast of where traders will end up), never on the
  conditional; traders alone own p(outcome | plan). The proposer must not trade its own proposals' conditionals.
- Markets are not assumed useful: start from the Predictor; adopt markets only if the predictor exploration finds them.
