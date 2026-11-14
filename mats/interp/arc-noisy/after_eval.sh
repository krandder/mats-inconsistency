#!/usr/bin/env bash
cd "$(dirname "$0")"
while pgrep -f "runq.sh jobs_kprop.txt" > /dev/null; do sleep 30; done
./runq.sh jobs_kprop_dev.txt 3
