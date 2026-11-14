#!/usr/bin/env bash
# run a job file with at most $2 parallel jobs (one line = one shell command), from this directory
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
xargs -P "$2" -I CMD bash -c CMD < "$1"
