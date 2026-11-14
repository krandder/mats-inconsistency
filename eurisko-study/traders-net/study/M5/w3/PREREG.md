# PREREG (M5 worker 3): pattern-2/4/8 shadow traders against three markets

Written 2026-09-28 ~02:05Z, before any run. Code: `w3/pattern.py` (class), `w3/run_w3.py` (driver),
`w3/summarize.py`. CPU only, 1 thread per job, at most 3 of my jobs at once.

**Question.** Does a class of pattern-k shadow traders (k = 2, 4, 8) keep extracting money from
a network's prices (E_C(t) linear, invasion fitness > 0), and on which families? Same for the
per-question logistic model and the soup (reference class).

**Class C_w3 (9 traders, shadow: wealth -> 0, no price impact, so traded price = clearing price;
no fixed point is needed because nothing in the market reads the shadow).** For k in {2,4,8}:
- `soup-k`: the soup's rule (logical-soup/docs/shadow.md): in the last 32 outcomes, find the most
  recent j in 1..32-k with ages j..j+k-1 equal to ages 0..k-1, predict the outcome at age j-1;
  abstain if none. Belief 0/1, so it is scored only in the one-share currency.
- `kt-k`: per-question table over the last k bits, KT counts (n1+1/2)/(n+1); price-blind.
- `corr-k`: price reader, belief q = sigmoid(l_p + delta[q, ctx]), delta learnt online by
  log-loss SGD (lr 0.05), clipped to |delta| <= 4, init 0 (a "humble" corrector that starts at the
  market).
All traders see only outcomes already revealed (bits before the predicted one).

**Currencies.**
- Kelly/log: E_T(t) = sum over scored positions <= t and questions of log q_T(y) - log p(y)
  (kt, corr; beliefs clipped to [1e-4, 1-1e-4]). This is T1 lemma 5's E_T at the clearing price.
- One share (soup's currency): d = sign(q - p) (0 on abstain), P&L = d * (y - p). All 9 traders.
Units: per question per position (divide by Q), so numbers compare with soup slopes / live Q.

**Markets (arms).**
1. `net-P4`, `net-P16`: S0 worker 1's prequential trade-time logits z (wealth-net Net d=64 L=2 H=4,
   Adam 1e-3, B=4 or 16), wealth-net streams Q=64 (pi family present), 64000 positions
   (`/tmp/traders-net/S0/w1/P{4,16}-s{1,2,3}.npz`). Scored on all positions.
2. `log-P4`, `log-P16`: the same runs' per-question logistic baseline zb.
3. `wn-8k`, `wn-20k`, `wn-final`: wealth-net train-s{1,2,3} checkpoints (step 8000, 20000, final
   ~31.1k) frozen, priced on their own held-out positions [split, n_pos) = [19174, 23968) (never
   trained); traders learn from the full public history from position 0 but are scored only there.
4. `soup`: logical-soup runs/phase3 `sh_pat{2,4,8}` columns (one-share, Q16 ticks); slope over
   the last 1000 epochs divided by 2^F (questions per cell), plus first-half / second-half slopes.
   Read-only; nothing recomputed.

**Seeds.** 1, 2, 3 (stream seed = model seed as in S0/wealth-net). Mean ± sample sd.

**Metrics.**
- E_C(t) = max over C_w3 of E_T(t) per currency (and per k).
- Slopes (least squares, per question-position): first half and second half of the scored window;
  invasion fitness phi_T = slope over the last 20% of the scored window.
- Per family: phi_T by family for the best trader.
- Brier/BCE of the market on the scored window, for context.

**Predictions (written before running).**
- P1: net prices: phi(kt-8, log) > 0 on periodic in at most 1/3 seeds (the net learns periodic);
  on lfsr16 phi(kt-8) <= 0 (8 bits do not predict a 16-bit LFSR).
- P2: logistic prices are exploitable by pattern traders on `and`/`or`/`ca110` (phi > 0, 3/3 seeds),
  because the per-question linear model cannot represent XOR-like contexts.
- P3: corr-k phi >= kt-k phi on every market (reading the price is weakly better).
- P4: E_C(one-share) second-half slope < 0.2 x first-half slope ("sublinear") for net-P4 in 3/3 seeds;
  not for logistic.

**Stop rule.** Fixed budget: every arm x seed runs once; no reruns except crashes; any job past
60 min wall is stopped and reported partial. No tuning of lr/clip after seeing results; any
change goes in an amendment below with its reason.

**Out of scope for w3.** D-delay / D-prove families (w2 arbitrageurs), calibration bins (w1),
MLP probes (w4). Invasion fitness for pattern traders on LI streams is left to the merger.
