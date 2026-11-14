# shadow.bend: reference traders in shadow mode (P2.1)

Hand-written forecasters that see what a soup trader sees for each question
(the market's price and the packed 32-outcome history, age k in bit k) and are
settled in their own ledger. They never replicate, never die, never move prices.

Per live question per epoch each trader either abstains or trades ONE share
(65536 Q16) toward its belief at the market's forecast for that event, the
post-trade marginal quote `qv`. Profit = demand x (outcome - price) in Q16
ticks: buying at p pays 65536 - p on a 1 and loses p on a 0; selling mirrors it.

| column | trader | signal |
|---|---|---|
| `sh_c50` | constant 0.5 | buy below 0.5, sell above, abstain at 0.5 |
| `sh_mom` | momentum | believe the last outcome repeats (buy if it was 1 and p < 1) |
| `sh_pat2/4/8` | pattern-k, k = 2, 4, 8 | find the most recent j in 1..32-k where ages j..j+k-1 equal ages 0..k-1, predict the outcome at age j-1; abstain when none |
| `sh_arb` | coherence arbitrageur | per (A, B, A and B, A or B) block: if P(A and B) > min(P(A), P(B)) sell A and B, buy the cheaper parent; if P(A or B) < max(P(A), P(B)) buy A or B, sell the dearer parent. One share per leg. |

Gating: a trader trades a question only when it is live (not held out); the
arbitrageur trades a block only when all four questions are live.

Each column in metrics.csv is the trader's cumulative profit (gains minus
losses, Q16 ticks, since epoch 0). A trader whose cumulative profit keeps
growing is finding an edge the soup's prices leave on the table; the Phase 2
gate reads `sh_arb` on compounds (should decay) and `sh_pat*` on periodic
families.

Known limit: at epoch 0 the packed history is all zeros (no outcomes yet), so
momentum and pattern-k treat the empty past as zeros for the first few epochs.
Tests: tests/shadow_test.bend (pattern-k checked against a Python oracle on 525
cases, arbitrageur on all 16 outcome combinations).
