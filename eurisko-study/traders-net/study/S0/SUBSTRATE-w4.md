# SUBSTRATE, worker 4 part: metric panel, run log, tests (2026-09-27)

Files (all under `study/S0/substrate/`, numpy only except the integration tests):
- `panel.py`: the panel every run reports.
  - `scores(p, y, fam, names)`: prequential BCE, Brier, ECE (10 equal-width bins, count-weighted),
    overall and per family. `p` must be the price recorded before training on that label.
  - `violations(P[T,Q], constraints, unresolved[T,Q], tol=0.01)`: AND/OR by Frechet bounds
    (v=0 iff a joint distribution fits), NOT, EQUAL (D-delay twins), TAUT (A or not A,
    (A and B)->A), CONTRA (A and not A), EXHAUSTIVE (|sum-1|). A (t, constraint) counts only
    while every question in it is unresolved. `constraints_from_li(kind, ops, group)` builds the
    list from worker 3's LI stream; the result equals `li_families.violations` to 1e-9
    (`tests/test_integration.py::test_panel_violations_match_li_families`).
  - `ec_slopes(t, E)`: log-log exponent, last-half linear slope and final value of an
    exploitation curve. E_C itself is M5's; this is only the summariser.
  - `gini(w)`: sum|wi-wj| / (2 n sum|wi|); the usual Gini for w>=0, in [0,1] for signed wealth.
    `gini_null(wealth_fn, y, n_null, groups=...)`: z and p of the observed Gini against labels
    permuted (optionally within family); the caller re-runs its books on each permutation.
  - `dead_bankrupt`, `conservation_residual`, `FlopCounter`, `net_forward_flops` (2 x MACs of
    wealth-net's Net, LN/softmax ignored).
  - `panel_row(...)`: one flat dict for runlog; `panel_from_prequential(res, fam, names, stream)`
    turns worker 1's `run_prequential` output into that row.
- `runlog.py`: `RunLog(out_dir, run_id, meta)` writes `<run_id>.csv` in long format
  `run_id,step,wall_s,metric,group,value` (group = family, unit or `all`), flushed per write for
  V1, plus `<run_id>.meta.json` (config, `code_hash`). `read_run(path)` reads it back.
  `PrequentialGuard(stress=False)`: raises if a key is trained before it was evaluated, evaluated
  after it was trained, or trained twice in single-pass mode; in stress mode it counts
  in-sample evaluations instead.
- Tests: `tests/test_panel.py` (11: hand values), `tests/test_runlog.py` (3),
  `tests/test_integration.py` (6: seeded determinism of streams and streams_ext, prequential
  test-then-train and delayed-label timing, panel scores = worker 1's per_family to 1e-5, panel
  violations = worker 3's). Run: `PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib python3 -m pytest -q study/S0/substrate/tests`
  -> 38 passed (all four workers' tests) at 22:20Z.

## Validation (PREREG-w4.md)
- Arm 1 (toy values): all exact to 1e-12 (`tests/test_panel.py`).
- Arm 2 (Gini-null calibration, 50 traders x 2000 examples, 30 replicates x 30 permutations,
  seeds 0-2; `w4/null_calib.json`, `w4/null_calib.csv`):
  - null (signals independent of labels): z mean -0.19 ± 0.19, z sd 1.03 ± 0.28 over seeds,
    0 of 90 replicates with z > 3. Passes the prereg bounds.
  - positive control (5 of 50 traders right 60% of the time): z mean 2.03 ± 0.18, and only
    23 ± 3% of replicates reach z > 3. The prereg expected z > 3; this **fails**. The
    reason is that exponential bet sizes dominate the Gini of these P&Ls, so the test has little power for an
    edge held by a few traders. Readers should report the z and the null sd, not the raw Gini,
    and should not read a low z as "no skill".
- Arm 3 (runlog round trip): exact (`tests/test_runlog.py`).

## Example CSV for V1
`w4/demo/demo-s{0,1,2}.csv` (+ `.meta.json`): a tiny prequential run (streams_ext Q=16+2
held-out, T=2400, regime shift every 800, Net d=32 H=2 L=1, B=8) logged every 25 steps with
per-family BCE/Brier/ECE (current family after regime shifts), `n` and `flops_cum`; final row `final_*` (grouped by initial family). 2 s per seed single-threaded; set torch.set_num_threads(1), since the default 23 threads thrashed on this loaded box. It shows the schema only and
is too small to be a result.

## Caveats
- ECE uses fixed bins, so it is biased upward for small n per family. Compare arms at equal n.
- `violations` default tol 0.01; li_families' default is 0.05. Pass tol explicitly when comparing.
- Wealth Gini, dead/bankrupt fraction and conservation residual need books (M1's `books_v2`);
  the panel takes their arrays but S0 does not compute wealth.
