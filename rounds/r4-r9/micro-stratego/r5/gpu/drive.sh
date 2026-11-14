#!/bin/bash
# drive.sh SEED : local driver for infra/gpu-run.sh --driver (gets GPU_VM_IP/NAME/ZONE). Installs torch on the VM,
# starts gpu/run_vm.sh SEED there, polls, and syncs back ONLY *.csv/*.json/*.png into r5/runs/sSEED/ (checkpoints and
# logs stay on the VM; on failure the remote log tails are printed into this driver's own log). gpu-run.sh deletes
# the VM when this script exits.
set -u
SEED=$1
MODE=${2:-full}
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
IP=$GPU_VM_IP
R5=/home/agent/workspaces/mats/work/micro-stratego/r5
LOCAL=$R5/runs/s$SEED
mkdir -p $LOCAL
RSH="ssh ${SSH_OPTS[*]}"
ts() { date -u +%FT%TZ; }
echo "$(ts) DRIVER start seed=$SEED vm=$GPU_VM_NAME ip=$IP"
vm_ssh "$IP" 'sudo bash -s' <<'REMOTE'
mkdir -p /root/runs/r5e2 /root/pylib
[ -f /usr/include/python3.12/Python.h ] || DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3.12-dev > /root/apt.log 2>&1
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --break-system-packages -q
fi
python3 -m pip install -q --target /root/pylib torch numpy scipy matplotlib > /root/pip.log 2>&1 || tail -5 /root/pip.log
PYTHONPATH=/root/pylib python3 -c "import torch; print('TORCH', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
REMOTE
echo "$(ts) DRIVER deps done"
vm_ssh "$IP" "sudo bash -c 'cd /root/soup && nohup bash gpu/run_vm.sh $SEED $MODE > /root/runs/r5e2/run_vm_s$SEED.out 2>&1 < /dev/null &'"
echo "$(ts) DRIVER launched run_vm.sh"
sync_back() {
  rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='*/' --include='*.csv' --include='*.json' --include='*.png' \
    --exclude='*' "$SSH_USER@$IP:/root/runs/r5e2/s$SEED/" "$LOCAL/" 2>&1 | tail -2
}
fails=0; n=0
while true; do
  sleep 60
  n=$((n+1))
  if st=$(vm_ssh "$IP" "sudo cat /root/runs/r5e2/s$SEED/all_done.json 2>/dev/null; echo PING" 2>/dev/null); then
    fails=0
  else
    fails=$((fails+1)); echo "$(ts) DRIVER ssh failure $fails"
    if (( fails >= 5 )); then
      gcloud compute instances describe "$GPU_VM_NAME" --zone "$GPU_VM_ZONE" --project GCP_PROJECT --format='value(status)' 2>&1 | head -2
      echo "$(ts) DRIVER giving up: VM unreachable (preempted?)"; exit 3
    fi
    continue
  fi
  if [[ -f /tmp/r5e2/GO_EVAL && ! -f /tmp/r5e2/pushed-$GPU_VM_NAME ]]; then
    rsync -az --rsync-path='sudo rsync' -e "$RSH" /tmp/r5e2/src/ "$SSH_USER@$IP:/root/soup/" 2>&1 | tail -2 \
      && vm_ssh "$IP" "sudo touch /root/soup/data/READY_ALL" && touch /tmp/r5e2/pushed-$GPU_VM_NAME \
      && echo "$(ts) DRIVER pushed late data + code, READY_ALL set"
  fi
  if (( n % 5 == 0 )); then
    sync_back
    vm_ssh "$IP" "sudo bash -c 'for a in primary fallback; do echo == \$a; tail -n 2 /root/runs/r5e2/s$SEED/\$a/*.log 2>/dev/null | cut -c1-300; done; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'" 2>/dev/null
  fi
  if [[ $st == *utc* ]]; then echo "$(ts) DRIVER pipeline finished: $st"; break; fi
done
sync_back
vm_ssh "$IP" "sudo bash -c 'for a in primary fallback; do echo == \$a; cat /root/runs/r5e2/s$SEED/\$a/done.json; tail -n 25 /root/runs/r5e2/s$SEED/\$a/*.log | cut -c1-400; done'" 2>/dev/null
ok=$(cat $LOCAL/primary/done.json $LOCAL/fallback/done.json 2>/dev/null | grep -c '"exit": 0')
echo "$(ts) DRIVER done ok_arches=$ok"
[[ $ok == 2 ]] && exit 0 || exit 4
