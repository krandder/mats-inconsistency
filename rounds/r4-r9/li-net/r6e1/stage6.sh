#!/bin/bash
# Stage the frozen source tree for the R6-E1 VM: r6e1 code + the r5e1 modules it imports (copied unchanged) + smallnet/netdef.py
# -> /tmp/r6e1/src (flat). Writes MANIFEST.sha256 and appends it, timestamped, to li-net/r6e1/results/stage-manifests.txt.
set -eu
R6=/home/agent/workspaces/eurisko-study/li-net/r6e1
R5=/home/agent/workspaces/eurisko-study/li-net/r5e1
SN=/home/agent/workspaces/eurisko-study/li-net/smallnet
S=/tmp/r6e1/src
mkdir -p $S/smallnet $R6/results
rsync -a --delete --include='*.py' --include='*.sh' --exclude='*' $R6/ $S/
for f in env6.py wm.py rollouts.py run.py partA2.py; do cp $R5/$f $S/$f; done
mkdir -p $S/smallnet
cp $SN/netdef.py $S/smallnet/netdef.py
( cd $S && sha256sum *.py *.sh smallnet/netdef.py > MANIFEST.sha256 )
{ echo "## $(date -u +%FT%TZ)"; cat $S/MANIFEST.sha256; } >> $R6/results/stage-manifests.txt
