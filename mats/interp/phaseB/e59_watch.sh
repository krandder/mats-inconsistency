#!/usr/bin/env bash
# E59 scheduling (sixth agent): after E53's summary (/tmp/s6q/e53.done), queue one e59_run.py job per arm (priority 25, ahead of
# E56's trainings), then, once all five outputs exist, queue e59_summ.py (priority 26).
cd /home/agent/workspaces/mats/work/cm-interp-B
until [ -f /tmp/s6q/e53.done ]; do sleep 60; done
for a in ctrl uni unif unifnoln_mag unifnoln_sq; do
  echo "/home/agent/.venvs/aggext/bin/python e59_run.py $a > results/e59_run_$a.log 2>&1" > /tmp/s6q/pending/25_e59_$a
done
until [ "$(ls results/e59/*.npz 2>/dev/null | wc -l)" -ge 5 ]; do sleep 60; done
echo "/home/agent/.venvs/aggext/bin/python e59_summ.py > results/e59_summ.out 2>&1 && touch /tmp/s6q/e59.done" > /tmp/s6q/pending/26_e59_summ
