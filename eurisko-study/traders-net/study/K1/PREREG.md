# K1 pre-registration (index, written by the lead 2026-09-28T04:45Z after the runs; not a new prereg)

K1's preregistration is three worker files, each written before the runs it covers (timestamps in `STATUS.md`):
- `PREREG-w1.md` (01:59Z): engine, niches on/off, scale 1k/10k/100k, Q in {1, 64, 4096}, T=10000, seeds 1-3.
- `w2/PREREG.md` (02:00Z, amendments 02:05Z and 02:15Z): mid-run shift on/off; compute-price sweep c/c*.
- `w3/PREREG.md` (~02:00Z, amendments ~02:10Z and ~02:20Z): funds of funds, buy-and-hold vs rebalancing.

Common to all: the plan's control (beliefs shuffled among traders, in lockstep on the same stream and population),
3 seeds (1, 2, 3) with mean ± sd, primary metrics survivors, Gini against the null, and market Brier.
Stop rule (plan section 3): if every arm is a monoculture (top clade share >= 0.9 at T in >= 2/3 seeds), report
Blume-Easley and stop. The lead added no arm and changed no metric; `check_k1.py` evaluates the stop rule over all 49 arms.
