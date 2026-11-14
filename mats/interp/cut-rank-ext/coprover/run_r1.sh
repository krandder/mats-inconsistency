#!/bin/bash
# gpt-6-astra co-prover, round 1: independent attempt from the lemma-free statement (read-only sandbox).
cd /home/agent/workspaces/mats/work/cut-rank-ext
echo "r1 start $(date -u +%H:%M:%SZ)" >> coprover/status.txt
timeout 5400 nice -n 19 codex-any exec --skip-git-repo-check -s read-only -m gpt-6-astra -c model_reasoning_effort='"xhigh"' -C /home/agent/workspaces/mats/work/cut-rank-ext "$(cat coprover/prompt_r1.md)" < /dev/null > coprover/astra_r1.raw 2>&1
echo "r1 exit $? $(date -u +%H:%M:%SZ)" >> coprover/status.txt
