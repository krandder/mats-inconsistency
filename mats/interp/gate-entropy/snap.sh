#!/bin/bash
# quick progress snapshot (gate-entropy)
cd /home/agent/workspaces/mats/work/gate-entropy
date -u +%H:%M:%S; cat logs/queue.log
for f in logs/*.log; do b=$(basename $f .log); [ "$b" = queue ] || [ "$b" = xargs ] && continue; echo "$b: $(grep -cE ' E3 ' $f) layers done; last: $(tail -1 $f | cut -c1-120)"; done
