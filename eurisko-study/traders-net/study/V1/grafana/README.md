# Grafana: traders-net dashboard

[removed] (folder "traders-net", uid `traders-net`). Pick a run; panels
plot columns (regex-selected, ../SCHEMA.md) against `step`, refresh every minute, re-reading the source.

## Data path

    study/<ws>/runs/<run>/metrics.csv            (run id <ws>.<run>)
    wealth-net/results/<tag>/snapshots/*.json    (run id wealth-net.<tag>, derived, read-only)
      -> serve.py on General [ip-removed]:8767 (GET /runs.json, /runs/<id>/metrics.{csv,json})
      -> GKE observability/[removed], container socat-8767
      -> Service observability/traders-runs:8767
      -> Grafana Infinity datasource "traders-net runs" (uid traders-net-runs)

The dashboard reads metrics.json: Infinity types JSON numbers as numbers but CSV columns as strings.
Keep-alive: user cron on General (`crontab -l | grep traders-net-grafana-export`) runs ensure-serve.sh
every 5 min and at boot. Log: serve.log.

## Files
- `build_dashboard.py` -> `traders-net.json` (edit the script). Appends worker 2's `panels_w2.py` panels.
- `apply.sh [--dashboard-only]`: rebuild + apply via deploy-host kubectl (`apply.py`), idempotent.
- `datasource.yaml`, `provider.yaml`: entries appended to ConfigMap `grafana-provisioning`.

## In the cluster (namespace observability), added 2026-09-27 by traders-net V1
ConfigMap `grafana-traders-net-dashboard`, Service `traders-runs`, container `socat-8767` in
Deployment `[removed]`, datasource + provider entries in ConfigMap `grafana-provisioning`,
volume/mount `traders-net-dashboard` in Deployment `grafana` (/var/lib/grafana/dashboards-traders-net).
Live patches: if observability manifests are re-applied from a repo lacking them, rerun `./apply.sh`.

Rollback: remove the volume/mount from Deployment grafana, the two entries from grafana-provisioning,
socat-8767 from [removed], delete Service traders-runs and the ConfigMap, drop the cron lines,
stop serve.py.
