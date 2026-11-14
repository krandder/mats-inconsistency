#!/usr/bin/env bash
cd "$(dirname "$0")"
while [ "$(ls logs/surr_d256_L8_*.log 2>/dev/null | wc -l)" -lt 4 ] || ps -eo args | grep -v grep | grep -q "python surr.py 256 8"; do sleep 10; done
while ps -eo args | grep -v grep | grep -q "python skipmerge.py"; do sleep 10; done
./runq.sh jobs_surr_rerun.txt 2
