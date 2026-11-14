# The full loop on trained world models: rollout policy, amplification, distillation, and an audit the searcher steers

Model: Claude Fable 5.2 (`claude-fable-5-2[1m]`). 2026-11-10, 16:30-18:45Z. Key `loop`, wf13. CPU only. Words follow
`wf9/FRAME.md` section 5: demonstration (the terminal state), score, target T (the score normalised), rollout policy
(the GFlowNet), amplification (the search on top of rollouts), distillation.

## 0. In plain words

Context. On 10 Nov you said the GFlowNet is the rollout policy, like AlphaZero's policy network; the search is the
amplification on top of many rollouts; the rollout policy is distilled from the amplification and improves over time;
network errors should cost only efficiency. Nobody had run that loop. I ran it in the two worlds with trained world
models: 6x6 Sudoku and the parity world (20 hidden assumptions, 30 noisy checks). The amplification has two forms:
weighted counting of K finished rollouts, and a tree in which the rollout policy chooses what to open while the
specification keeps the books and returns an interval around its answer.

1. Your safety claim held. For every rollout policy I tried (untrained, trained, trained toward a wrong goal, one that
   refuses part of the space) the exact answer of the score was inside the interval: no failure in about 11 million
   checked intervals. Bad policies cost openings or width, never the answer. (Three false alarms were a rounding error
   in my own bookkeeping; fixed and re-checked.)
2. The protection comes from the books, not from the rollouts. Weighted counting alone, fed by the policy that refuses
   part of the space, is off by 0.3 to 0.4 at any number of rollouts and nothing in its output shows it.
3. The loop did not compound. The first round of distillation gives almost all the gain (openings to close a hard
   Sudoku: 785 and 1,796 become 73 and 95; error of weighted counting in the parity world: 0.44 becomes 0.04 to 0.07).
   Rounds 2 to 5 add nothing, and in Sudoku the cost climbs back to about 170. Training once on what the untrained
   search found, with the same puzzles and steps, did as well in one seed and worse in the other.
4. A hand-made order is still cheaper: fewest-values-left closes hard puzzles in 43 openings against 73 for the best
   learned round; in the parity world the narrowest interval comes from the specification searching alone.
5. Why, as far as I can tell: finding the score and proving that nothing else is there want different orders. Your
   target says which values, never which order (every order of a demonstration scores the same), and almost all the
   cost is in branches whose completions score nothing, where distillation has no target. Started at the hand rule's
   level, each round of distillation made the Sudoku tree dearer (46, 67, 86, 86, 86, 95 openings) while the rollouts
   got better. Your "efficiency" target needs a second signal: the cost the search measured. That last sentence is a
   conjecture; I did not build it.
6. What the learned policy does buy: it finds the score sooner, and its rollouts get better every round (Sudoku
   rollouts ending in the solution: 0, 50, 52, 54, 61, 61 percent). Its own one-pass belief stays worse than one pass of M.
7. At 40 assumptions the certain interval is still 0.2 to 0.9 wide after 16,384 openings; the interval from the
   specification's own draws is at most 0.04 wide there. Your probabilistic bound is the one that survives size.

I don't know: whether an order trained on measured cost would compound; a world where the budget keeps binding (9x9) or
with no hand order; a trust head on top (not built: answers are compared with the score's exact answer); decisions.

