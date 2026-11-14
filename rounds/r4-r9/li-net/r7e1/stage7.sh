#!/bin/bash
# Stage the frozen source tree for the R7-E1 VM -> /tmp/r7e1/src (flat): r7e1 code, r6e1/e1v3.py and the r5e1 modules it
# imports (copied unchanged), smallnet/netdef.py. With STAGE_CKPT=1 it also stages li-net/r7e1/ckpt/s*/theta_real_v2.{npz,json}
# (recovery after a lost VM: remote7.sh then loads them instead of retraining); otherwise /tmp/r7e1/src/ckpt is emptied.
# Writes MANIFEST.sha256 and appends it, timestamped, to li-net/r7e1/results/stage-manifests.txt.
set -eu
R7=/home/agent/workspaces/eurisko-study/li-net/r7e1
R6=/home/agent/workspaces/eurisko-study/li-net/r6e1
R5=/home/agent/workspaces/eurisko-study/li-net/r5e1
SN=/home/agent/workspaces/eurisko-study/li-net/smallnet
S=/tmp/r7e1/src
EMPTY=/tmp/r7e1/empty
mkdir -p $S/smallnet $S/ckpt $R7/results $EMPTY
rsync -a --delete --include='*.py' --include='*.sh' --exclude='*' $R7/ $S/
cp $R6/e1v3.py $S/e1v3.py
for f in env6.py wm.py rollouts.py run.py partA2.py; do cp $R5/$f $S/$f; done
cp $SN/netdef.py $S/smallnet/netdef.py
if [ "${STAGE_CKPT:-0}" = "1" ]; then
  rsync -a --delete --include='s*/' --include='theta_real_v2.npz' --include='theta_real_v2.json' --exclude='*' $R7/ckpt/ $S/ckpt/
else
  rsync -a --delete $EMPTY/ $S/ckpt/
fi
( cd $S && sha256sum *.py *.sh smallnet/netdef.py > MANIFEST.sha256 && { ls ckpt/s*/theta_real_v2.npz >/dev/null 2>&1 && sha256sum ckpt/s*/theta_real_v2.npz >> MANIFEST.sha256 || true; } )
{ echo "## $(date -u +%FT%TZ) STAGE_CKPT=${STAGE_CKPT:-0}"; cat $S/MANIFEST.sha256; } >> $R7/results/stage-manifests.txt
