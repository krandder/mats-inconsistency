#!/bin/bash
# GPT-6 review (fresh session, high effort, read-only) of the stationary-regime report.
cd /home/agent/workspaces/mats/work/stationary
echo "review start $(date -u +%H:%M:%SZ)" >> review/review_status.txt
timeout 3600 codex-any exec --skip-git-repo-check -s read-only -m gpt-6-astra -c model_reasoning_effort='"high"' -C /home/agent/workspaces/mats/work/stationary --add-dir /home/agent/workspaces/mats/outbox "$(cat review/prompt_gpt6.md)" < /dev/null > review/review_gpt6.raw 2>&1
echo "review exit $? $(date -u +%H:%M:%SZ)" >> review/review_status.txt
