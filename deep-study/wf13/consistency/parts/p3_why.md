## 3. Why, and what kind of world would make it help

- **What an order-against-order condition can carry (PROVEN, elementary).** For checks X, Y and context C, the two
  larger-context numbers P(Y given C, X) and P(X given C, Y) fix the joint of (X, Y) given C, hence both
  smaller-context numbers: P(X=1 given C) / P(X=0 given C) = [P(X=1 given C, y) / P(X=0 given C, y)] x [P(y given C,
  X=0) / P(y given C, X=1)]. The converse fails: the two smaller-context numbers leave the dependence free. So the
  condition carries knowledge **from larger contexts down to smaller ones**, and it adds to real data only where
  real outcomes score the larger-context questions and not the smaller-context ones.
- **In this world real data need no such transport.** One real check vector scores every question about itself, of
  every context size. The condition can then only tie together estimates the predictor would otherwise fit
  separately, and a small head with the same weights for every question fits nothing separately. Hence no gain in
  2b, and a large one in 2e, where real outcomes score one family only.
- **At small compute the measured mismatch is sampling noise (2c), and the cheapest way to remove sampling noise is
  to stop reading the samples (2d).** That is the October failure on the amplified predictor: the amplification is
  informative but noisy, one pass of M is stable but says little, and the loss drags the first toward the second.
- **Why the anchor helps and where it stops.** (ii-s) is (ii-o) plus the variance between terminal vectors, which is
  zero only if the amplified joint is proportional to the score. That part is real information (the score's own law,
  the same thing as outcomes sampled from M) and needs no real outcome, so it helps where real outcomes are few. The
  (ii-o) part is the same noise penalty, so a large weight drags too; and it can teach only what the score knows.
- **Worlds where your claim should hold** (item 1 CHECKED post hoc in two worlds; the rest CONJECTURE). (1) Reality
  scores only some kinds of question, for instance only the last step with everything else revealed, and forecasts
  are needed for the others. (2) The amplified predictor has many free numbers per question (your distilled
  one-pass head, params_amp), not a 124k-parameter head on M's numbers. (3) Compute is large enough that sampling
  noise is below the order effect. (4) The variables asked about are strongly dependent given typical contexts, so
  the trivially consistent answers (one half; one pass of an M that says little) are far from right.
- **How I would word the requirement.** As a condition on the EXPECTED amplified prediction: the mean over trajectory
  draws of amplified P(X) P(Y given X) equals that of P(Y) P(X given Y). A loss on single draws also penalises the
  sampling noise, and I do not have a working estimator without it (2f).
