#!/bin/bash
# On-VM driver: PLAN=<A|B|C|D> bash run_vm.sh. bank.py skips finished units (resume after preemption).
# Plan A runs the gates first (PREREG amendment 1, noise 0.01):
#   1. quick gate: group 0 on val blocks 0-1 at depth 64 only (out/gateq) -> gate.py --quick (G1, G2)
#   2. group 0 on val blocks 0-1 at depth 1024 (the real bank) -> gate.py (G1, G2, G3)
# A failed gate writes GATE_FAILED and stops; the General watchdog then syncs and deletes every P17 VM.
set -u
cd /root/job
export PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1
O=/root/out; mkdir -p $O/bank $O/logs
PLAN=${PLAN:-A}
NOISE=${NOISE:-0.01}
A=/root/eqr-art
hb() { while true; do touch $O/HEARTBEAT; sleep 60; done; }
hb & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
run_bank() { local out=$1 log=$2; shift 2
  python bank.py --eqr $A/eqr --ckpt $A/eqr_sudoku.pth --data $A/data --out $out --compile 1 --noise_scale $NOISE "$@" >>$log 2>&1; }
bank() { run_bank $O/bank $O/logs/bank_$PLAN.log "$@"; }
gate_fail() { touch $O/GATE_FAILED; echo "$(date -u +%FT%TZ) GATE FAILED ($1)"; exit 0; }
[[ -f $O/GATE_FAILED ]] && { echo "gate failed earlier; nothing to do"; exit 0; }
case $PLAN in
  A)
    if [[ ! -f $O/GATEQ.json ]] || ! grep -q '"PASS": true' $O/GATEQ.json; then
      run_bank $O/gateq $O/logs/gateq.log --blocks 0-1 --groups 0 --depth_scale 0.0625 || { echo "quick bank failed" >>$O/FAILED_STEPS; exit 1; }
      python gate.py --quick 1 --bank $O/gateq --log $O/logs/gateq.log --out $O/GATEQ.json >>$O/logs/gate.log 2>&1 || gate_fail quick
    fi
    bank --blocks 0-1 --groups 0 || { echo "bank failed" >>$O/FAILED_STEPS; exit 1; }
    python gate.py --bank $O/bank --log $O/logs/bank_A.log --out $O/GATE.json >>$O/logs/gate.log 2>&1 || gate_fail full
    bank --blocks 0-2 --groups 0-6 ;;
  B) bank --blocks 3-5 --groups 0-6 ;;
  C) bank --blocks 6-8 --groups 0-6 ;;
  D) bank --blocks 9-11 --groups 0-6 ;;
esac
rc=$?
[[ $rc == 0 ]] && touch $O/DONE_$PLAN || echo "$PLAN rc=$rc $(date -u +%FT%TZ)" >>$O/FAILED_STEPS
echo "$(date -u +%FT%TZ) plan $PLAN rc=$rc"
