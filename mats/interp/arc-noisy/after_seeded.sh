#!/usr/bin/env bash
# wait for the seeded oracles, then run oracle replicates and the noiseless kprop controls (3 at a time)
cd "$(dirname "$0")"
while pgrep -f "scratch/seeded_oracle.py" > /dev/null; do sleep 20; done
{
  echo "nice -n 15 /home/agent/.venvs/arc-noisy/bin/python scratch/edgeworth_oracle.py 101 0.001 18 31 > logs/edge_oracle_101_31.log 2>&1"
  echo "nice -n 15 /home/agent/.venvs/arc-noisy/bin/python scratch/edgeworth_oracle.py 102 0.001 18 31 > logs/edge_oracle_102_31.log 2>&1"
  cat jobs_kprop_noiseless.txt
} > jobs_after_seeded.txt
./runq.sh jobs_after_seeded.txt 3
