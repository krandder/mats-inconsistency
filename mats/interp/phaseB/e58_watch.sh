#!/usr/bin/env bash
# E58 after the VM (sixth agent): wait until the retry script (PID given) has exited, find the run id whose VM ran, collect its
# runs into runs/e58/, then queue one e56_eval.py job per net and the summary (evq_loop.sh). usage: e58_watch.sh RETRY_PID
cd /home/agent/workspaces/mats/work/cm-interp-B
while kill -0 "$1" 2>/dev/null; do sleep 60; done
RID=$(grep -o 'attempt [0-9]* (cmb-e58-[0-9]*): VM was created' GPU-LOG.md | tail -1 | grep -o 'cmb-e58-[0-9]*')
echo "$(date -u +%TZ) retry script exited; rid=$RID" >> /tmp/s6/e58_collect.log
[ -n "$RID" ] || exit 1
./gpu/collect_b5.sh "$RID" >> /tmp/s6/e58_collect.log 2>&1
./evq_loop.sh e58 e56_eval.py 22 30 "/home/agent/.venvs/aggext/bin/python e58_summ.py > results/e58_summ.out 2>&1 && touch /tmp/s6q/e58.done" 32
