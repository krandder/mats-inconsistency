#!/usr/bin/env bash
# run a job file (one shell command per line) with at most $2 parallel jobs
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
xargs -P "$2" -I CMD bash -c CMD < "$1"
