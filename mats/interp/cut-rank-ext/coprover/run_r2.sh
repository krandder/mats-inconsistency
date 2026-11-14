#!/bin/bash
# gpt-6-astra co-prover, round 2: lemma exchange (read-only sandbox).
cd /home/agent/workspaces/mats/work/cut-rank-ext
echo "r2 start $(date -u +%H:%M:%SZ)" >> coprover/status.txt
timeout 5400 nice -n 19 codex-any exec --skip-git-repo-check -s read-only -m gpt-6-astra -c model_reasoning_effort='"xhigh"' -C /home/agent/workspaces/mats/work/cut-rank-ext "$(cat coprover/prompt_r2.md)" < /dev/null > coprover/astra_r2.raw 2>&1
echo "r2 exit $? $(date -u +%H:%M:%SZ)" >> coprover/status.txt
