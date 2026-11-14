#!/usr/bin/env bash
# One-command launch of the Turtel 14B replication on a rented 4xH100 (Vast.ai), unattended end to end:
#   preflight -> rent -> setup (bootstrap) -> start run.sh -> watchdog + sync loop -> finish (verify, destroy).
#
#   ops/vast/turtel-14b/launch.sh --go            # rent and run; everything else is automatic
#   ops/vast/turtel-14b/launch.sh --go --resume   # second lease: same data, trainer resumes from the synced checkpoint
#   ops/vast/turtel-14b/launch.sh                 # preflight + offer search only, no rental, no spend
#
# The lane directory (manifest.lane_dir, default /home/agent/data/vast-turtel-14b) holds manifest.json,
# instance.json, lease.json, results/, logs. It refuses to run while a previous instance is not destroyed.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
GO=0; RESUME=0
for a in "$@"; do case $a in --go) GO=1;; --resume) RESUME=1;; *) echo "unknown arg $a"; exit 2;; esac; done
LANE=$(python3 -c "import json; print(json.load(open('$HERE/manifest.template.json'))['lane_dir'])")
mkdir -p "$LANE"
if [ ! -f "$LANE/manifest.json" ]; then cp "$HERE/manifest.template.json" "$LANE/manifest.json"; fi
if [ -f "$LANE/instance.json" ] && [ ! -f "$LANE/destroyed.json" ]; then
  echo "instance.json exists without destroyed.json in $LANE: a previous rental is still open; finish it first"; exit 3
fi
if [ "$RESUME" = 1 ]; then
  if [ ! -d "$LANE/results/turtel-14b-grpo/checkpoints" ]; then echo "--resume but no synced checkpoints in $LANE/results"; exit 4; fi
  mkdir -p "$LANE/archive" && for f in instance.json destroyed.json budget-stop.json finish.log setup-result.json start.json export-manifest.json; do
    [ -f "$LANE/$f" ] && mv "$LANE/$f" "$LANE/archive/$(date +%s)-$f"; done
fi

# preflight (no API calls)
API_KEY=$(python3 -c "import json; print(json.load(open('$LANE/manifest.json'))['api_key_file'])")
SSH_KEY=$(python3 -c "import json; print(json.load(open('$LANE/manifest.json'))['ssh_key'])")
[ -f "$API_KEY" ] || { echo "missing Vast API key $API_KEY"; exit 5; }
[ -f "$SSH_KEY" ] || { echo "missing ssh key $SSH_KEY"; exit 5; }
DATA=$REPO/data/turtel-prompts-20260922
[ -f "$DATA/train.jsonl" ] && [ -f "$DATA/validation.jsonl" ] || { echo "run scripts/build_turtel_prompts.py first"; exit 6; }
TEST=/home/agent/data/forecasting-benchmarks/LightningRodLabs--outcome-rl-test-dataset/data/train-00000-of-00001.parquet
[ -f "$TEST" ] || { echo "missing test parquet $TEST"; exit 6; }
(cd "$REPO" && python3 -m unittest test_turtel_reward -q 2>&1 | tail -1)

# tarballs: code (training/, scripts/, tests) and data (prompts + manifest + test parquet)
(cd "$REPO" && tar -czf "$LANE/code.tar.gz" training/turtel_grpo.py training/turtel_reward.py training/turtel_prompt.py \
   training/requirements-grpo.in training/__init__.py test_turtel_reward.py scripts/build_turtel_prompts.py)
TMP=$(mktemp -d); cp "$DATA/train.jsonl" "$DATA/validation.jsonl" "$DATA/manifest.json" "$TMP/"; cp "$TEST" "$TMP/test.parquet"
(cd "$TMP" && tar -czf "$LANE/data.tar.gz" train.jsonl validation.jsonl manifest.json test.parquet); rm -r -- "$TMP"
cp "$HERE/bootstrap.sh" "$HERE/run.sh" "$HERE/api.py" "$HERE/rent.py" "$HERE/setup.py" "$HERE/start.py" "$HERE/watchdog.py" "$HERE/sync_loop.sh" "$HERE/finish.py" "$HERE/progress.py" "$LANE/"
(cd "$REPO" && git rev-parse HEAD > "$LANE/code-git-sha.txt" 2>/dev/null || true)
sha256sum "$LANE/code.tar.gz" "$LANE/data.tar.gz" > "$LANE/tarballs.sha256"
echo "lane: $LANE"; cat "$LANE/tarballs.sha256"

if [ "$GO" != 1 ]; then
  echo "preflight OK; searching offers only (no rental):"
  python3 "$LANE/rent.py" "$LANE" --dry-search
  exit 0
fi

# unattended lifecycle in the background; the log is the handoff
LOG=$LANE/launch.log
nohup bash -c "
  set -e
  cd '$LANE'
  echo \"\$(date -u +%FT%TZ) RENT\" >> '$LOG'
  python3 rent.py '$LANE' --go
  echo \"\$(date -u +%FT%TZ) SETUP\" >> '$LOG'
  python3 setup.py '$LANE'
  echo \"\$(date -u +%FT%TZ) START\" >> '$LOG'
  python3 start.py '$LANE' $([ "$RESUME" = 1 ] && echo --resume)
  nohup python3 watchdog.py '$LANE' > watchdog.log 2>&1 &
  nohup bash sync_loop.sh '$LANE' > sync_loop.out 2>&1 &
  nohup python3 progress.py '$LANE' > progress.log 2>&1 &
  echo \"\$(date -u +%FT%TZ) FINISH (waiting for the run)\" >> '$LOG'
  python3 finish.py '$LANE'
  echo \"\$(date -u +%FT%TZ) LIFECYCLE_DONE rc=\$?\" >> '$LOG'
" >> "$LOG" 2>&1 &
echo "launched in background (pid $!); follow: tail -f $LOG ; progress: tail -f $LANE/progress.log"
