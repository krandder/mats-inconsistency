#!/bin/bash
# drive.sh main : local driver for infra/gpu-run.sh --driver (GPU_VM_IP/NAME/ZONE). Installs torch on the VM, starts
# gpu/run_vm.sh detached with the redirect INSIDE sudo (the fix logged in R6-E3: the first r6 driver redirected as the
# unprivileged user and the run never started), checks within 3 minutes that the seeds really started (and starts them
# again once if not), polls, and syncs back ONLY *.csv/*.json into r7/runs/. gpu-run.sh deletes the VM when this exits.
set -u
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
IP=$GPU_VM_IP
LOCAL=/home/agent/workspaces/mats/work/micro-stratego/r7/runs
mkdir -p $LOCAL
RSH="ssh ${SSH_OPTS[*]}"
ts() { date -u +%FT%TZ; }
echo "$(ts) DRIVER start vm=$GPU_VM_NAME ip=$IP"
vm_ssh "$IP" 'sudo bash -s' <<'REMOTE'
mkdir -p /root/runs/r7e3 /root/pylib
[ -f /usr/include/python3.12/Python.h ] || DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3.12-dev > /root/apt.log 2>&1
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --break-system-packages -q
fi
python3 -m pip install -q --target /root/pylib torch numpy scipy > /root/pip.log 2>&1 || tail -5 /root/pip.log
PYTHONPATH=/root/pylib python3 -c "import torch; print('TORCH', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
REMOTE
echo "$(ts) DRIVER deps done"
start_run() {
  timeout 60 ssh "${SSH_OPTS[@]}" -n "$SSH_USER@$IP" "sudo bash -c 'setsid nohup bash /root/soup/gpu/run_vm.sh >> /root/runs/r7e3/run_vm.out 2>&1 < /dev/null &'" >> /tmp/r7e3/drive-launch.log 2>&1
  echo "$(ts) DRIVER launched run_vm.sh"
}
started() {
  vm_ssh "$IP" "sudo bash -c 'ls /root/runs/r7e3/s101/run.log /root/runs/r7e3/s102/run.log /root/runs/r7e3/s103/run.log >/dev/null 2>&1 && pgrep -f [r]un_r7e3.py >/dev/null && echo RUNNING'" 2>/dev/null | grep -q RUNNING
}
start_run
sleep 90
if started; then
  echo "$(ts) DRIVER verified: 3 seed logs exist and run_r7e3.py is running"
elif vm_ssh "$IP" "sudo bash -c 'pgrep -f [r]un_vm.sh >/dev/null && echo ALIVE'" 2>/dev/null | grep -q ALIVE; then
  echo "$(ts) DRIVER run_vm.sh alive but seeds not seen yet; waiting 60 s (no second start)"
  sleep 60
  if started; then echo "$(ts) DRIVER verified"; else echo "$(ts) DRIVER seeds still not running; giving up"; exit 5; fi
else
  echo "$(ts) DRIVER run not seen after 90 s and run_vm.sh not alive; starting once more"
  start_run
  sleep 60
  if started; then echo "$(ts) DRIVER verified on the second start"; else echo "$(ts) DRIVER run still not running; giving up"; exit 5; fi
fi
sync_back() {
  rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='*/' --include='*.csv' --include='*.json' --exclude='*' \
    "$SSH_USER@$IP:/root/runs/r7e3/" "$LOCAL/" 2>&1 | tail -2
}
fails=0; n=0
while true; do
  sleep 60
  n=$((n+1))
  if st=$(vm_ssh "$IP" "sudo cat /root/runs/r7e3/all_done.json 2>/dev/null; echo PING" 2>/dev/null); then
    fails=0
  else
    fails=$((fails+1)); echo "$(ts) DRIVER ssh failure $fails"
    if (( fails >= 5 )); then echo "$(ts) DRIVER giving up: VM unreachable"; exit 3; fi
    continue
  fi
  if (( n % 3 == 0 )); then
    sync_back
    vm_ssh "$IP" "sudo bash -c 'for s in 101 102 103; do echo == s\$s; grep -v -E \"Warning|detach|hist.append|nanmean\" /root/runs/r7e3/s\$s/run.log 2>/dev/null | tail -n 2 | cut -c1-260; done; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'" 2>/dev/null
  fi
  if [[ $st == *utc* ]]; then echo "$(ts) DRIVER pipeline finished: $st"; break; fi
done
sync_back
vm_ssh "$IP" "sudo bash -c 'for s in 101 102 103; do echo == s\$s; cat /root/runs/r7e3/s\$s/exit.json; grep -v -E \"Warning|detach|hist.append\" /root/runs/r7e3/s\$s/run.log | tail -n 30 | cut -c1-600; done'" 2>/dev/null
ok=$(cat $LOCAL/s101/exit.json $LOCAL/s102/exit.json $LOCAL/s103/exit.json 2>/dev/null | grep -c '"exit": 0')
echo "$(ts) DRIVER done ok_seeds=$ok"
[[ $ok == 3 ]] && exit 0 || exit 4
