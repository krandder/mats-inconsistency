# K1 worker1 pre-registration: economy core, niches on/off, scale (written 2026-09-28 before any logged run)

Engine: `K1/econ.py` (module docstring = full rules). Tests: `K1/tests/test_econ.py`.
Fixed defaults: T=10000 steps, batch B=16 questions per step, c=1e-4 per question traded, subsidy 0.01
(maker stakes 1% of initial total wealth per step, belief 0.5), full Kelly (stake=1), split at w>=4, die at w<0.01,
families periodic/markov1/markov3/parity3/coin/noise uniform, no shift (shift is worker2's), no funds (worker3's).

## Arms (3 seeds each: 1, 2, 3)
- N (trader capacity) in {1000, 10000, 100000}
- Q in {1, 64, 4096}; niches on/off (Q=1: one arm, on == off)
- control in every run, in lockstep on the same stream and population: beliefs shuffled among the traders on each
  question before pricing and settlement (the null).
Grid: N x (Q1, Q64 on, Q64 off, Q4096 on, Q4096 off) = 15 cells x 3 seeds = 45 runs. If 100k niches-off cells take
more than 3 h per seed on this box, they run at T=5000 (declared here, logged as an amendment if used).

## Metrics (end of run; window metrics = mean over the last 10% of logs)
- survivors: alive traders, clades (founder lineages) holding >=1% of wealth, effective clades 1/sum(share^2),
  top clade share, top species (type x k) share;
- Gini of alive wealth vs the shuffled null (gini_null_traders);
- Brier (and BCE) of the market price, prequential, vs the oracle Brier from the true P, and vs the null.
Report mean +- sd over 3 seeds per cell.

## Monoculture definition and stop rule (plan: "Monoculture in every arm: report Blume-Easley and stop")
Monoculture = top clade share >= 0.9 at T. If every cell is a monoculture in >=2/3 seeds, report Blume-Easley
market selection and stop (no further sweeps from worker1).
Predictions (written before running): Q=1 and niches-off drift toward few clades (market selection); niches-on
coexist (eff clades >> 1) with Gini above the null; niches-off with Q>1 underperform niches-on on Brier because a
pooled learner cannot fit per-question rules; Q=4096 at N=1k is subsidy-dominated (most questions have no trader).
