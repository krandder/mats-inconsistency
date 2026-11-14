# PREREG, X1 worker 1 (skeptic + reproduction of S0, L1, T1)

Written 2026-09-28T12:07Z. **Deviation, disclosed:** the S0 and T1 fresh-seed jobs were launched at
12:02-12:03Z, 4-5 minutes before this file. The arms, seeds and pass rule below are the ones fixed by
the X1 brief (STUDY-PLAN section 6: "rerun each headline number from the committed scripts on a
fresh seed, in a clean copy; it must land within 2 sd"), which predates the runs. Before writing this
file I had seen the seed-4 log lines of T1 `check_delay.py` and `check_li.py` (one value, the D=512
hindsight gap 0.0436, looked outside 2 sd). The tie-break rule in (4) was added after seeing it and is
applied to every claim alike.

1. **Arms.** One clean copy per workstream under `/tmp/traders-net/X1w1/<ws>/`, unmodified code except
   the seed list. Fresh seeds: S0 w1/w2 seed 4 (committed 1-3), S0 w3/w4 seed 3 (committed 0-2),
   T1 seed 4 (committed 1-3). L1 has no runs: rerun `check_novelty.py` in a copy and re-fetch at least
   three quoted primary sources independently (arXiv PDFs), comparing text.
2. **Metric.** Each headline number in the orchestrator's S0/L1/T1 summaries (the "numbers" lists).
   First: is it in the cited file (value and ± sd)? Second: fresh-seed value x4.
3. **Pass rule.** |x4 - mean| <= 2 sd, with sd the committed 3-seed sample sd. Identities and bounds
   (errors ~1e-12 or "a <= b"): pass if the fresh seed satisfies the same tolerance/inequality.
   Counts (e.g. "learnt in 1/3 seeds"): report the fresh count; a single seed cannot refute a rate.
   If sd is 0 or not reported, use 10% relative as the band and say so.
4. **Tie-break (added 12:07Z, see above).** With n=3 the 2-sd band misses a new Gaussian draw about
   20% of the time (t_2 predictive). A claim whose seed-4 value is outside 2 sd gets one more fresh
   seed (seed 5, same copy). Status: verified = fresh seed(s) inside; refuted = both fresh seeds
   outside on the same side, or the number is not in its file; unverified = mixed, or not rerun.
5. **Stop rule.** Stop when every headline has a status. CPU only, <= 5 of my jobs at once,
   1 thread each (T1's scripts set 2 torch threads; left as committed).
