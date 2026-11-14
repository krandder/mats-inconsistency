#!/usr/bin/env bash
# wait for the three orphaned d=64 scorecard jobs, then the reduced scorecard queue, then the follow-up queue (3 slots each)
cd "$(dirname "$0")"
while pgrep -f "score.py 64 8 tanh" > /dev/null; do sleep 30; done
./runq.sh jobs_score2.txt 3
./runq.sh jobs_after.txt 3
