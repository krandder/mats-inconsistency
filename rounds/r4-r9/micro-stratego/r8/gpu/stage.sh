#!/bin/bash
# stage.sh : build /tmp/r8e3/src (rsynced to the VM by gpu-run.sh): r5 modules, r6 code and r7lib (all unchanged), r8 code,
# run_vm.sh, and hard-linked copies of /tmp/r5e2/data and /tmp/r5e2/ckpt (no duplicate disk use). Old trees go to the trash.
set -eu
MS=/home/agent/workspaces/mats/work/micro-stratego
[[ -e /tmp/r8e3/src.new ]] && trash-put /tmp/r8e3/src.new
mkdir -p /tmp/r8e3/src.new/r5 /tmp/r8e3/src.new/r6 /tmp/r8e3/src.new/r7 /tmp/r8e3/src.new/r8 /tmp/r8e3/src.new/gpu /tmp/r8e3/src.new/ckpt
cp $MS/r5/env6.py $MS/r5/msgame6.py $MS/r5/model6.py $MS/r5/train6.py $MS/r5/eval6.py $MS/r5/agents6.py /tmp/r8e3/src.new/r5/
cp $MS/r6/e3lib.py $MS/r6/run_e3.py /tmp/r8e3/src.new/r6/
cp $MS/r7/r7lib.py /tmp/r8e3/src.new/r7/
cp $MS/r8/r8lib.py $MS/r8/run_r8e3.py $MS/r8/expl_d1_es.py /tmp/r8e3/src.new/r8/
cp $MS/r8/gpu/run_vm.sh $MS/r8/gpu/run_vm_es.sh /tmp/r8e3/src.new/gpu/
cp -al /tmp/r5e2/data /tmp/r8e3/src.new/data
for s in 101 102 103; do mkdir -p /tmp/r8e3/src.new/ckpt/s$s; cp -al /tmp/r5e2/ckpt/s$s/primary /tmp/r8e3/src.new/ckpt/s$s/primary; done
[[ -e /tmp/r8e3/src ]] && trash-put /tmp/r8e3/src
mv /tmp/r8e3/src.new /tmp/r8e3/src
du -sh /tmp/r8e3/src
