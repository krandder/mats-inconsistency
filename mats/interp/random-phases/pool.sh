#!/usr/bin/env bash
# usage: pool.sh JOBFILE NWORKERS
# Workers pop the first line of JOBFILE (under flock) and run it until the file is empty; more workers can be started later
# on the same file. Start and end lines go to JOBFILE.done. Run from this directory with OMP_NUM_THREADS=1.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
f="$1"; n="${2:-1}"
pop() {
  flock "$f.lock" bash -c 'l=$(head -n1 "$0"); if [ -n "$l" ]; then sed -i 1d "$0"; fi; printf "%s" "$l"' "$f"
}
worker() {
  while true; do
    line=$(pop)
    [ -z "$line" ] && break
    echo "$(date -u +%H:%M:%S) start $line" >> "$f.done"
    bash -c "$line"
    echo "$(date -u +%H:%M:%S) end rc=$? $line" >> "$f.done"
  done
}
for i in $(seq "$n"); do worker & done
wait
