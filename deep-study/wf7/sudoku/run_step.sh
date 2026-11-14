#!/bin/bash
# run_step.sh OUTFILE CMD...  : run one step at low priority, log to OUTFILE, append its exit code; stop the chain on failure
out="$1"; shift
nice -n 19 ionice -c3 /home/agent/.venvs/aggext/bin/python -u "$@" > "$out" 2>&1
rc=$?; echo "exit $rc" >> "$out"
if [ $rc -ne 0 ]; then echo "$(date -u +%H:%M:%S) FAILED: $* -> $out" >> out/CHAIN_FAILURES; exit 1; fi
echo "$(date -u +%H:%M:%S) ok: $*" >> out/CHAIN_LOG
