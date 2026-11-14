#!/bin/bash
# One rsync of runs/ and logs from the GPU box; refresh the lease only on success.
D=/home/agent/data/qwen06b-structured-20260927; source $D/ops/conn.env
mkdir -p $D/results
if rsync -az -e "ssh -i $KEY -p $PORT -o ConnectTimeout=20" $HOST:/workspace/consistency/runs/ $D/results/ > $D/ops/sync.log 2>&1; then
  echo "{\"phase\":\"synced\",\"at\":$(date +%s)}" > $D/ops/lease.json
  date -u +%FT%TZ > $D/ops/last-sync.txt
fi
