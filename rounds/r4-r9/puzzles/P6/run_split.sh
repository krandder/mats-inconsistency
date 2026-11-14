#!/bin/bash
# Start this VM's share of the main plan (two-VM split, see queue_p6.py): run_split.sh 1|2
cd /root/job || exit 1
case "$1" in
  1) export P6_ONLY='^(train|eval)_(EC|R0)_s[0-9]|^(train|eval)_(PBT|EMA)_s[0-3]_|^numerics$' P6_TAG='' ;;
  2) export P6_ONLY='^(train|eval)_(F|DTL|R1)_s[0-9]|^(train|eval)_(PBT|EMA)_s4_' P6_TAG='-2' ;;
  *) echo "usage: run_split.sh 1|2" >&2; exit 2 ;;
esac
pgrep -f '[q]ueue_p6.py main' >/dev/null && { echo "queue already running" >&2; exit 1; }
setsid nohup /opt/venv/bin/python queue_p6.py main </dev/null >>"/root/out/queue-main$P6_TAG.out" 2>&1 &
echo "started split $1"
