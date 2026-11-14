p = '/home/agent/workspaces/mats/outbox/unrestricted-estimators-2026-10-31.md'
s = open(p).read()
rep = {
"""**Short answer.** The adaptive deductive estimator I built (a certified branch-and-bound) beats sampling only at
width 3, and its cost grows exponentially with width: that part of the progress prong failed.""":
"""**Short answer.** The adaptive deductive estimator I built (a certified branch-and-bound) catches up with RQMC only
at width 3, and its cost grows exponentially with width: that part of the progress prong failed.""",
"""  out of reach at 8 and above), against 130-2,000 for RQMC, and a cell costs 46-170 forward passes. Nothing in this
  family can reach 10M+ parameters.""":
"""  out of reach at 8 and above), against 130-2,000 for RQMC, and a cell costs 46-170 forward passes. Extrapolating
  that growth (HEURISTIC), nothing in this family can reach 10M+ parameters.""",
"""- Centred unscrambled Sobol points did 2.7-4.5x better than scrambled Sobol at widths 3-8 (geometric means), but
  they are sampling at fixed points, and the gain of any
  such design over radius-exact MC is capped by the low-degree content of the net's angular function: the heuristic""":
"""- Centred unscrambled Sobol points did 2.7-4.5x better than scrambled Sobol at widths 3-8 (geometric means), but
  they are sampling at fixed points, and the gain of any such design over radius-exact MC is capped by the
  low-degree content of the net's angular function: the heuristic""",
"""   infinite-width limit); polynomials in the weights (the mean's Hermite degree grows with depth, about 2, 7 and 17 at
   depths 4, 16 and 32; plug-in estimates put degree-8 polynomials at a few to 17 MC samples at depths 16-32);""":
"""   infinite-width limit); polynomials in the weights (the mean's Hermite degree grows with depth: finite-difference
   proxies of about 2, 7 and 17 at depths 4, 16 and 32; plug-in estimates put degree-8 polynomials at 6-17 MC samples
   at depths 16-32);""",
"""- `lib/` (nets, the branch-and-bound), `tests/` (all pass: `logs/test_bnb.log`), `scratch/` (runs and analysis),
  `runs/`, `results/`, `logs/`, `notes/` (questions, literature, analogies).""":
"""- `lib/` (nets, the branch-and-bound, the multilevel sampler), `tests/` (all pass: `logs/test_bnb.log`,
  `logs/test_mlmc.log`), `scratch/` (runs and analysis), `runs/`, `results/`, `logs/`, `notes/` (questions,
  literature, analogies).""",
}
for a, b in rep.items():
    assert a in s, a[:60]
    s = s.replace(a, b)
open(p, 'w').write(s)
print('ok')
