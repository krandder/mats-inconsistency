#!/bin/bash
# Stage the frozen source tree for the R8-E1 VM -> /tmp/r8e1/src (flat): r8e1 code, r7e1/e1v4.py, r6e1/e1v3.py and the r5e1
# modules they import (copied unchanged), smallnet/netdef.py, and the three R7-E1 theta_real checkpoints (npz + json) under
# ckpt/s<seed>/ (e1v5.py checks each npz's sha256 against its json before use). Writes MANIFEST.sha256 and appends it,
# timestamped, to li-net/r8e1/results/stage-manifests.txt.
set -eu
LI=/home/agent/workspaces/eurisko-study/li-net
R8=$LI/r8e1
S=/tmp/r8e1/src
mkdir -p $S/smallnet $R8/results
rsync -a --delete --include='*.py' --include='*.sh' --exclude='*' $R8/ $S/
cp $LI/r7e1/e1v4.py $S/e1v4.py
cp $LI/r6e1/e1v3.py $S/e1v3.py
for f in env6.py wm.py rollouts.py run.py partA2.py; do cp $LI/r5e1/$f $S/$f; done
cp $LI/smallnet/netdef.py $S/smallnet/netdef.py
rsync -a --delete --include='s*/' --include='theta_real_v2.npz' --include='theta_real_v2.json' --exclude='*' $LI/r7e1/ckpt/ $S/ckpt/
( cd $S && sha256sum *.py *.sh smallnet/netdef.py ckpt/s*/theta_real_v2.npz > MANIFEST.sha256 )
{ echo "## $(date -u +%FT%TZ)"; cat $S/MANIFEST.sha256; } >> $R8/results/stage-manifests.txt
