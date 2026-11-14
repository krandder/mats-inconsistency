#!/bin/bash
# stage.sh : build the source tree that gpu-run.sh rsyncs to the VM (/tmp/r5e2/src): r5 code + gpu/run_vm.sh +
# a hard-linked copy of the data (no duplicate disk use).
set -eu
R5=/home/agent/workspaces/mats/work/micro-stratego/r5
SRC=/tmp/r5e2/src
rm -rf "$SRC.new"; mkdir -p "$SRC.new/gpu"
cp $R5/env6.py $R5/msgame6.py $R5/model6.py $R5/train6.py $R5/eval6.py $R5/agents6.py $R5/audit6_vm.py "$SRC.new/"
cp $R5/gpu/run_vm.sh "$SRC.new/gpu/"
cp -al /tmp/r5e2/data "$SRC.new/data"
[[ -d $SRC ]] && trash-put "$SRC"
mv "$SRC.new" "$SRC"
du -sh "$SRC"; ls "$SRC" "$SRC/data"
