#!/bin/bash
# GPT-6 second-round review (fresh session, high effort, read-only) of the revised stationary-numerics report.
cd /home/agent/workspaces/mats/work/stationary-numerics
echo "review r3 start $(date -u +%H:%M:%SZ)" >> review/review_status.txt
timeout 3600 codex-any exec --skip-git-repo-check -s read-only -m gpt-6-astra -c model_reasoning_effort='"high"' -C /home/agent/workspaces/mats/work/stationary-numerics --add-dir /home/agent/workspaces/mats/outbox --add-dir /home/agent/workspaces/mats/work/stationary "$(cat review/prompt_gpt6_r3.md)" < /dev/null > review/review_gpt6_r3.raw 2>&1
echo "review r3 exit $? $(date -u +%H:%M:%SZ)" >> review/review_status.txt
