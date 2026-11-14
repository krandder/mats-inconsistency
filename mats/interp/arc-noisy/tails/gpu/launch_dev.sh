#!/usr/bin/env bash
# launch_dev.sh <run-id> [timeout_s]: one on-demand L4 VM running a job loop (VM deleted at the end or on STOP).
# Jobs: push.sh / job.sh put scripts in /root/jobs/ on the VM; each runs from /root/soup, log in /root/runs/<rid>/.
set -euo pipefail
RID=$1; TMO=${2:-36000}
HERE=/home/agent/workspaces/mats/work/arc-noisy
STAGE=/tmp/arcnoisy_stage/$RID
mkdir -p "$STAGE/tails"
cp "$HERE"/anet.py "$HERE"/samplers.py "$HERE"/kerdock129.npy "$HERE"/rtable.json "$STAGE"/
cp "$HERE"/tails/*.py "$STAGE"/tails/
cp "$HERE"/tails/gpu/*.py "$STAGE"/tails/ 2>/dev/null || true
(cd "$STAGE" && sha256sum anet.py rtable.json tails/*.py > code_sha256.txt)
LOOP="/root/venv/bin/pip install -q scipy; mkdir -p /root/jobs /root/runs/$RID; while [ ! -f /root/jobs/STOP ]; do for j in /root/jobs/*.sh; do [ -e \"\$j\" ] || continue; b=\$(basename \$j .sh); mv \"\$j\" \"\$j.run\"; (cd /root/soup && bash \"\$j.run\" > /root/runs/$RID/\$b.log 2>&1; echo \"JOB-EXIT \$?\" >> /root/runs/$RID/\$b.log); mv \"\$j.run\" \"\$j.done\"; done; sleep 3; done"
echo "$(date -u +%FT%TZ) LAUNCH rid=$RID provisioning=STANDARD machine=g2-standard-8(L4) job-loop timeout=${TMO}s (tails part 2)" >> "$HERE/GPU-LOG.md"
export GPU_ZONES="us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c us-east1-c us-east1-d us-west1-a us-west1-b us-west4-a europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c asia-southeast1-b asia-southeast1-c"
rc=0
GPU_SOURCE_DIR="$STAGE" GPU_PROVISIONING=STANDARD GPU_RUN_TIMEOUT=${TMO}s \
  /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "$LOOP" || rc=$?
grep -h "$RID" /tmp/l4runs/$RID/GPU-LOG.md 2>/dev/null | sed 's/^/    /' >> "$HERE/GPU-LOG.md" || true
echo "$(date -u +%FT%TZ) END rid=$RID rc=$rc (VM deleted by gpu-run.sh cleanup; verify with gcloud list)" >> "$HERE/GPU-LOG.md"
