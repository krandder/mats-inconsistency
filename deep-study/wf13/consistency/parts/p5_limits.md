## 5. Limits, what was not run, files

- **Compute is the main limit.** K = 1 and 4 only. At K = 4 the amplified predictor removes a quarter of one pass's
  error on large contexts and almost nothing elsewhere; wf10's K = 256 predictor is a different object. Whether the
  order variance falls below the sampling noise at K = 64 or 256 is UNKNOWN and needs the L40.
- Two worlds, one network each, two real subsets per n, one initialisation per arm. A second initialisation moves
  real-outcomes-only by up to 0.004 nats at n = 64, so gains under about 10 % at n <= 256 are inside run noise.
- Uniform context sizes make most questions easy; the mean is moved mostly by the 11-13 % with 20-29 revealed checks.
  The weights were chosen on a dev world where the amplifier did not work and kept as registered: scale guesses. The
  score reference of (ii-s) is a 128-trajectory estimate; a better one might change its high-weight result.
- Sections 2e, 2f and the mechanism numbers were designed after seeing world 101's registered results; 2e was then
  repeated on world 102 unchanged. Changed after the first registration, evaluation side only: `cexact.py`, `cU.py`
  (cached table), `creport.py` (bad-check rule, score-estimate numbers), `mktables.py`, `cdirect.py` (800 steps, on
  this host; on the cluster it blocked the queue and job 57889 was cancelled for that after its 114 runs finished).
- NOT RUN: worlds 103, 311; K >= 16; consistency contexts from real check vectors; another searcher; decisions.
- **Jobs** (cluster account k.santos, CPU-only, 10 cores): 57881 benchmark; 57885 (the GPU accident, cancelled);
  57886 dev world; 57889 world 101; 57904 world 102. L40 jobs 57879, 57880: cancelled in the queue, never ran.
- **Files.** `PREREG.md`, `PREREG.sha256` (17:26:32Z; amendment 17:30:10Z: its text says 17:35Z, and a cut pipe left
  only `PREREG.md` and `cgen.py` in its hash entry; the other arm files are unchanged from the first entry and the
  cluster task logs print their hashes). Arms, no exact inference (`check_firewall.py`, `.out`): `cgen.py`,
  `cmodel.py`, `ctrain.py`, `mkjobs.py`, `cdirect.py`; post hoc `ctrain_ph.py`, `cgen_c1b.py`. Evaluation: `cexact.py`,
  `cU.py`, `creport.py`, `mktables.py`, `mknote_tables.py`, `mkfigs.py`, `cmech.py`. `cluster/` (tasks as run).
  Outputs: `out/w101/`, `out/w102/` (every run's forecasts, logs, `summary.json`), `out/tables.md`, `out/w10*_ph25/`,
  `out/w101_phdb/`, `out/mechanism_w10*.json`, `out/pilot_w7/` (dev), `logs/`, `fig/`, `RESULT-work.json`.
  Trajectories: `data/w101/`, `data/w102/` (29 MB each; 4.2 million network calls per world).
