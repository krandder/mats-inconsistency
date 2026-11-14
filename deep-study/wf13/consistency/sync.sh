#!/bin/bash
# sync.sh WORLD [BAD] : copy one world's arm forecasts and logs from the cluster, then build the report here.
W=$1
cd /home/agent/workspaces/mats/work/deep-study/wf13/consistency
R=user@[ip-removed]; SSH="ssh -i $[ssh-key-path-removed] -o BatchMode=yes"
mkdir -p out/w$W logs
rsync -a -e "$SSH" --exclude '*.pt' $R:/cluster/home/user/wave13/consistency/out/w$W/ out/w$W/
rsync -a -e "$SSH" $R:/cluster/home/user/wave13/consistency/logs/ logs/cluster/
rsync -a -e "$SSH" $R:/cluster/home/user/wave13/consistency/'runner-*.log' logs/cluster/
rsync -a -e "$SSH" $R:/cluster/home/user/wave13/consistency/data/w$W/'gen_info_*.json' out/w$W/ 2>/dev/null
[ -f out/w$W/exact.npz ] && nice -n 19 /home/agent/.venvs/aggext/bin/python -B creport.py out/w$W "$2" > logs/report_w$W.log 2>&1; tail -3 logs/report_w$W.log | cut -c1-200
