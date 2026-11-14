#!/bin/bash
# gpt-6-astra co-prover, round 2: lemma exchange and the two analogies (read-only sandbox, xhigh effort).
cd /home/agent/workspaces/mats/work/unrestricted
echo "r2 start $(date -u +%H:%M:%SZ)" >> coprover/status.txt
timeout 5400 codex-any exec --skip-git-repo-check -s read-only -m gpt-6-astra -c model_reasoning_effort='"xhigh"' -C /home/agent/workspaces/mats/work/unrestricted --add-dir /home/agent/workspaces/mats/outbox "$(cat coprover/prompt_r2.md)" < /dev/null > coprover/astra_r2.raw 2>&1
echo "r2 exit $? $(date -u +%H:%M:%SZ)" >> coprover/status.txt
