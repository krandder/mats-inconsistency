#!/bin/bash
# Stage the frozen source tree for the VM: li-net/r5e1 code + smallnet/netdef.py (copied unchanged) -> /tmp/r5e1/src.
# Writes /tmp/r5e1/src/MANIFEST.sha256 (and appends it to li-net/r5e1/results/stage-manifests.txt with a timestamp).
set -eu
R5=/home/agent/workspaces/eurisko-study/li-net/r5e1
SN=/home/agent/workspaces/eurisko-study/li-net/smallnet
S=/tmp/r5e1/src
mkdir -p $S/smallnet $R5/results
rsync -a --delete --include='*.py' --include='*.sh' --exclude='*' $R5/ $S/
mkdir -p $S/smallnet
cp $SN/netdef.py $S/smallnet/netdef.py
( cd $S && sha256sum *.py *.sh smallnet/netdef.py > MANIFEST.sha256 )
{ echo "## $(date -u +%FT%TZ)"; cat $S/MANIFEST.sha256; } >> $R5/results/stage-manifests.txt
