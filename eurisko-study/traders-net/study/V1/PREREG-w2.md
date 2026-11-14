# V1 worker 2 pre-registration (panels conservation / E_C / price + check script)

Written 2026-09-27 before the first check run. V1 is a view: no arms, no seeds of its own.
- Object: `grafana/panels_w2.py` (5 panels: conservation residual, E_C, price per family,
  held-out BCE per family, Brier per family) and `check.py`.
- Metric (check.py, one JSON result per run of the check, `check/check-<UTC>.json`):
  1. hermetic: serve.py on 127.0.0.1 with a temp copy of a fixture run returns 200 for
     `/runs.json` and the CSV; a row appended to the file appears on the next request
     (re-read on each refresh); a half-written last row is skipped; `nan` served as empty.
  2. exporter: every run in the live `/runs.json` answers 200 with >= 1 data row.
  3. Grafana: the provisioned dashboard `traders-net` exists; panel 1's query, run through
     Grafana's `/api/ds/query` (Infinity datasource) for every run, returns a frame; for every
     panel, its `filterFieldsByName` regex matches >= 1 numeric non-empty field besides `step`
     in >= 1 run. A panel with no matching column in any run is PENDING (upstream named), not
     PASS; a query error or a non-numeric matched field is FAIL.
- Seeds: the three wealth-net seeds are all checked (train-s1..s3), not averaged.
- Stop rule: done when the check reports no FAIL on the live Grafana; E_C (M5) and price (S0
  runlog) may stay PENDING until those workstreams write runs. 3 failed cluster attempts: stop and
  report the exporter-only result.
