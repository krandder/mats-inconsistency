#!/bin/bash
# up.sh RUNID   (RUNID e1-*): named spot L4 rental in budget group r4-e1 (cap $15), minimal startup, then torch install.
# Writes gpu/vm-RUNID (name zone ip). Teardown: down.sh RUNID (gpu-down.sh --only NAME --no-sync).
set -u
RUN=$1
[[ $RUN == e1-* ]] || { echo "run id must start with e1-" >&2; exit 2; }
D=/home/agent/workspaces/mats/work/micro-stratego/gpu
E=/home/agent/workspaces/eurisko-study/li-net/eca
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
ZONES=${MS_ZONES:-"us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c us-east1-b us-east1-c us-east1-d us-west1-a us-west1-b us-west4-a"}
mkdir -p $D/logs
L=$D/logs/up-$RUN.log
echo "$(date -u +%FT%TZ) UP $RUN group_spent=$(spend_usd r4-e1)" | tee -a $L
GPU_ZONES="$ZONES" GPU_STARTUP=minimal GPU_PROVISIONING=${MS_PROV:-SPOT} GPU_BUDGET_GROUP=r4-e1 GPU_GROUP_BUDGET_USD=15 GPU_RUN_TIMEOUT=3h \
  $E/gpu/gpu-up-zones.sh --run-id $RUN >> $L 2>&1
rc=$?
[[ $rc == 0 ]] || { echo "$(date -u +%FT%TZ) UP-FAILED $RUN rc=$rc" | tee -a $L; exit $rc; }
read -r NAME ZONE < "$STATE/vm-$RUN"
IP=$(vm_ip "$NAME" "$ZONE")
echo "$NAME $ZONE $IP" > $D/vm-$RUN
echo "$(date -u +%FT%TZ) READY $RUN $NAME $ZONE $IP" | tee -a $L
vm_ssh "$IP" 'sudo bash -s' >> $L 2>&1 <<'REMOTE'
mkdir -p /root/ms /root/runs/ms /root/logs
[ -f /usr/include/python3.12/Python.h ] || DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3.12-dev > /root/apt.log 2>&1
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --break-system-packages -q
fi
python3 -m pip install -q --target /root/pylib torch numpy scipy matplotlib > /root/pip.log 2>&1 || tail -5 /root/pip.log
PYTHONPATH=/root/pylib python3 -c "import torch; print('TORCH', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
REMOTE
echo "$(date -u +%FT%TZ) SETUP-DONE $RUN" | tee -a $L
