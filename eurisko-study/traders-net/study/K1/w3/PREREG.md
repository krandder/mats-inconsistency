# K1 worker 3 PREREG: funds of funds, fixed fractions vs rebalancing (written 2026-09-28 ~02:00Z, before any run)

Question: in the Kelly-pool economy (K1/econ.py, worker1's engine: wealth-weighted price, maker subsidy
inflow, compute-price sink, split and die), does grouping traders into funds change the market?
T1 lemma 6 (study/T1/THEORY-part3.md) predicts: buy-and-hold fixed fractions = the flat economy exactly;
constant rebalancing = a flat CRP (moves prices); only a rebalancing rule adds power.

## Arms (all on the same stream, same leaf population, same seed per seed)
- **flat**: no funds (worker1's default economy).
- **bh**: every leaf is owned by one fund (groups of m=8), fixed fractions bought at t=0, never traded.
  Leaf wealth is unchanged; the fund value is the sum. Prediction: price path identical to flat.
- **crp**: each fund rebalances its value to fixed equal fractions over its live members every step,
  after settlement and before sink/split/die. Children of a split join the parent's fund; the dead
  leave it (fractions renormalised over live members).
- **eg**: as crp, but the fractions follow online exponentiated gradient (eta = 0.05, Helmbold et al. 1998,
  unverified recollection) on each member's wealth relative.
- Fund topology, two levels: **within** (the 8 members trade the same question: a fund = a pooled
  opinion on one niche) and **cross** (8 members on random questions: a diversified fund).
- Control: **null** = worker1's beliefs-shuffled economy, run with flat and crp.
- Scale: N = 1k leaves (all arms); N = 10k for flat/crp/eg within (if time and CPU allow).
  Q = 64 niches (the S0 stream), T per worker1's default (target >= 20k steps).

## Metrics (per run, logged every 250 steps to K1/runs/w3-<arm>-<topo>-N<n>-s<seed>/metrics.csv, V1 wide schema)
- market Brier (price before the outcome) per family and overall, second half of the run (primary);
- effective strategies per question 1/sum(share^2) of question wealth, mean over questions (coexistence);
- survivors (alive leaves), births, bankruptcies, dead_frac; Gini of leaf wealth and its null;
- fund growth: mean over funds of log(V_T/V_0) minus log of the same members' buy-and-hold value
  (the "rebalancing premium"; 0 by construction in bh).
- identity check: max |price_bh - price_flat| over all (q,t).

## Seeds: 1, 2, 3 (stream seed = economy seed). Report mean ± sd.

## Pre-registered predictions and decision rule
- P1 (theorem check): max |price_bh - price_flat| < 1e-9 in 3/3 seeds.
- P2: crp-within changes the market: |Brier(crp) - Brier(flat)| > 2 sd of the flat arm in 3/3 seeds
  (direction not predicted), and effective strategies per question are higher under crp (rebalancing
  keeps losing members alive).
- P3: rebalancing premium of crp and eg is reported with sign; "rebalancing adds power" is claimed only
  if the premium is > 0 by more than 2 sd in 3/3 seeds, for that topology.
- Stop rule (K1 table): if every arm is a monoculture (effective strategies per question < 1.1 in the
  second half), report Blume-Easley (market selection) and stop adding arms.
- Compute: at most 3 concurrent single-thread CPU jobs, checked with uptime; no GPU unless 10k/100k is
  too slow on CPU (then via G1's queue).

## Amendment 1 (2026-09-28 ~02:10Z, after one 1000-step smoke run at /tmp/traders-net/K1w3/smoke, before any logged grid run)
- Engine facts: worker1's econ.py has its own stream (families periodic/markov1/markov3/parity3/coin/noise, Q=64,
  B=16 per step), T=10000; I use its defaults (c=1e-4, subsidy 0.01, split 4, die 0.01).
- Design: phi = 0.5 of the groups are active funds (rule applied); the other half are virtual buy-and-hold groups of the
  same size/topology in the same market (no transfers). The **bh** arm (all groups inert) is the flat economy exactly
  (the hook does nothing), so bh = flat; its premium is the random-split baseline. P1 holds by construction here and
  is reported as such (T1 C6c already checked it with a non-trivial implementation).
- Primary premium metric changed: mean over funds of log V is dominated by dead funds (floored at 1e-12; the smoke
  run showed half the crp funds dead by t=1000). Primary = aggregate log wealth ratio
  premium_agg = logit(share_fund at T) - logit(share_fund at t=first log), i.e. log(V_funds/V_bhgroups) growth.
  Secondary: fund survival fraction (V >= die_at) vs bh-group survival, median log V difference.
- Claim rule for P3 now uses premium_agg (> 0 by > 2 sd in 3/3 seeds).

## Amendment 2 (2026-09-28 ~02:20Z, after the N=1000 grid; new arm, not a change to the earlier arms)
The N=1000 grid showed every leaf-level rebalancing arm losing to buy-and-hold groups in the same market
(out/summary_N1000.csv). A real fund of funds is two-level; add arm **fof**: subfunds = within-niche groups of 8,
held buy-and-hold (selection inside a niche continues); top funds = 4 random subfunds (cross-niche), rebalanced
to equal fractions every step (top-level CRP; members of a subfund are scaled together). Same phi = 0.5 split into
active top funds vs inert top groups of the same 32 leaves. Run name w3-fof-cross-N*-s*. Prediction written now:
premium_agg of fof is higher than crp-cross and crp-within (selection preserved inside niches); sign not predicted.
Same claim rule (> 0 by > 2 sd in 3/3 seeds).
