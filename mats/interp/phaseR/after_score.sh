#!/usr/bin/env bash
# wait until the scorecard queue has drained, then run the follow-up queue with 3 parallel jobs
cd "$(dirname "$0")"
while pgrep -f "runq.sh jobs_score.txt" > /dev/null; do sleep 60; done
./runq.sh jobs_after.txt 3
