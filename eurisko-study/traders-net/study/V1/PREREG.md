# V1 worker 1 pre-registration (live wealth view)

No experimental arms: V1 is a view. What is registered is the acceptance check.
- Sources: every `study/<ws>/runs/<run>/metrics.csv` and every wealth-net snapshot series
  (`wealth-net/results/train-s{1,2,3}`, 3 seeds, read-only).
- Metric: for each run served, HTTP 200 on `/runs/<id>/metrics.csv`, >= 1 data row, and every
  panel's column regex matches >= 1 column for the wealth-net runs; last `step` in the served CSV
  equals the newest snapshot on disk (freshness, re-read per request).
- Seeds: the 3 wealth-net seeds are shown side by side; no mean +- sd here (the view shows raw
  per-seed series; summary numbers belong to the experiment workstreams).
- Stop rule: done when the check passes on the live Grafana (through the Infinity datasource) for
  all 3 seeds; if the cluster route fails 3 times, stop and report the served endpoint only.
