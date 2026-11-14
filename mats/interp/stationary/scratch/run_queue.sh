#!/bin/bash
# Runs a job file with at most 3 concurrent jobs, each under nice 19 + idle IO, single-threaded BLAS.
cd /home/agent/workspaces/mats/work/stationary
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
JOBS=$1; LOG=$2
echo "queue $JOBS start $(date -u +%H:%M:%SZ)" >> $LOG
xargs -P 3 -I{} bash -c 'echo "start $(date -u +%H:%M:%SZ) {}" >> '"$LOG"'; nice -n 19 ionice -c3 {} > /dev/null 2>> '"$LOG"'.err; echo "end $? $(date -u +%H:%M:%SZ) {}" >> '"$LOG" < $JOBS
echo "queue $JOBS done $(date -u +%H:%M:%SZ)" >> $LOG
