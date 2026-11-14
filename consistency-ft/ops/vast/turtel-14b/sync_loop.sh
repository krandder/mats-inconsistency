#!/bin/bash
# Pull the run directory from the GPU box to General every sync_interval_seconds; refresh the watchdog lease only on
# a successful sync (same contract as data/vast-wave1/sync_loop.sh). Usage: sync_loop.sh <lane_dir>
LANE=$1
cd "$LANE" || exit 1
HOST=$(python3 -c "import json; print(json.load(open('instance.json'))['ssh_host'])")
PORT=$(python3 -c "import json; print(json.load(open('instance.json'))['ssh_port'])")
KEY=$(python3 -c "import json; print(json.load(open('manifest.json'))['ssh_key'])")
REMOTE=$(python3 -c "import json; print(json.load(open('manifest.json'))['remote_root'])")
EVERY=$(python3 -c "import json; print(json.load(open('manifest.json')).get('sync_interval_seconds', 300))")
mkdir -p results
while true; do
  # checkpoints are large (optimizer state for the LoRA); pull everything except the rolling ones being written
  if rsync -az --exclude 'checkpoints/checkpoint-*/rng_state*' -e "ssh -i $KEY -p $PORT -o ConnectTimeout=20" \
      root@$HOST:$REMOTE/runs/ results/ > sync.log 2>&1; then
    echo "{\"phase\":\"synced\",\"at\":$(date +%s)}" > lease.json
    date -u +%FT%TZ > last-sync.txt
  fi
  if [ -f results/turtel-14b-grpo/status.txt ] && grep -qE "RUN_DONE|RUN_FAILED|TRAINER EXIT=" results/turtel-14b-grpo/status.txt; then
    echo SYNC_FINAL > sync-final.txt; break
  fi
  sleep "$EVERY"
done
