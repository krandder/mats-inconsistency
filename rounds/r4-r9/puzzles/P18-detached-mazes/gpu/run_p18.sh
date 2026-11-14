#!/bin/bash
# P18 runner on a trader-dds-* VM (copied from P13/gpu/run_p13.sh).
# ~/p18/out/queue.txt: one job per line "<name> <script.py> <args...>" (comments with #); a line "END" means no more jobs
# will be added. Up to $(cat out/SLOTS) jobs run at once (default 2). Finished jobs get out/<name>.done; a job that exits
# nonzero gets out/<name>.fail.<k>; a name "A@B" means job A waits for out/B.done (A fails if B failed twice) and is retried once (resume from its own checkpoint); exit code 3 is "killed on
# purpose" (unit test (e)) and is not retried. Every job resumes from its own checkpoint, so rerunning this is safe.
# Writes out/HEARTBEAT every 20 s and out/RUNNER_DONE when END is reached and everything is done or failed twice.
cd ~/p18 || exit 1
mkdir -p out/logs
# data gate (PREREG): the frozen inputs must match their sha256 (frozen/SHA256SUMS.inputs), else no job runs
if ! sha256sum -c --quiet SHA256SUMS.inputs > out/data_gate.txt 2>&1; then echo "$(date -u +%FT%TZ)" > out/GATE_DATA_FAIL; echo "$(date -u +%FT%TZ) GATE_DATA_FAIL" >> out/runner.log; exit 1; fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
declare -A PIDS
start_job() {
  local n=$1; shift
  ( python3 "$@" >> out/logs/$n.log 2>&1 < /dev/null; rc=$?
    if [ $rc = 0 ]; then touch out/$n.done; else k=$(ls out/$n.fail.* 2>/dev/null | wc -l); touch out/$n.fail.$((k+1)); echo "rc=$rc" >> out/logs/$n.log; fi ) &
  PIDS[$n]=$!
  echo "$(date -u +%FT%TZ) START $n pid ${PIDS[$n]}" >> out/runner.log
}
while true; do
  [ -f out/STOP ] && { echo "$(date -u +%FT%TZ) STOP" >> out/runner.log; exit 0; }
  P=$(cat out/SLOTS 2>/dev/null || echo 2)
  for n in "${!PIDS[@]}"; do kill -0 ${PIDS[$n]} 2>/dev/null || { echo "$(date -u +%FT%TZ) EXIT $n" >> out/runner.log; unset PIDS[$n]; }; done
  pending=0; ended=0
  touch out/queue.txt
  while read -r f s rest; do
    [ -z "$f" ] && continue; [[ "$f" == \#* ]] && continue
    [ "$f" = END ] && { ended=1; continue; }
    n=${f%%@*}; dep=""; [[ "$f" == *@* ]] && dep=${f#*@}
    [ -f out/$n.done ] && continue
    if [ -n "$dep" ] && [ ! -f out/$dep.done ]; then
      if [ "$(ls out/$dep.fail.* 2>/dev/null | wc -l)" -ge 2 ]; then touch out/$n.fail.1 out/$n.fail.2; echo "dep $dep failed" >> out/logs/$n.log; continue; fi
      pending=$((pending+1)); continue
    fi
    nf=$(ls out/$n.fail.* 2>/dev/null | wc -l); [ "$nf" -ge 2 ] && continue
    [ -n "${PIDS[$n]}" ] && continue
    pending=$((pending+1))
    if [ ${#PIDS[@]} -lt "$P" ]; then start_job $n $s $rest; fi
  done < out/queue.txt
  echo "$(date -u +%FT%TZ) running=${#PIDS[@]} pending=$pending" > out/HEARTBEAT
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> out/HEARTBEAT 2>/dev/null
  if [ $ended = 1 ] && [ $pending = 0 ] && [ ${#PIDS[@]} = 0 ]; then echo "$(date -u +%FT%TZ)" > out/RUNNER_DONE; exit 0; fi
  sleep 20
done
