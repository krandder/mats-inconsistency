#!/usr/bin/env bash
# Copy finished runs (those with a test eval) from the live VMs into local/<rid>/ for early analysis (read-only on
# the VMs). Usage: fetch.sh
cd /home/agent/workspaces/mats/work/scale-free-exp
for v in "sfx-b3 [ip-removed]" "sfx-b4 [ip-removed]" ${B5:+"sfx-b5 $B5"}; do
  set -- $v
  K=/tmp/l4runs/$1/infra/state/soup_ed25519
  O="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20"
  mkdir -p local/$1
  ssh -i $K $O soup@$2 "sudo bash -c 'cd /root/runs/$1 && for d in */; do d=\${d%/}; if [ -f \$d/eval_test_model.json ] || [ -f \$d/eval_test.json ]; then echo \$d; fi; done | tar czf /tmp/fin.tgz -T - \$(ls *.log) 2>/dev/null; chmod 644 /tmp/fin.tgz'" 2>/dev/null
  scp -q -i $K $O soup@$2:/tmp/fin.tgz /tmp/fin_$1.tgz 2>/dev/null && tar xzf /tmp/fin_$1.tgz -C local/$1 && echo "$1: $(ls local/$1 | grep -v '\.log' | wc -l) finished runs"
done
