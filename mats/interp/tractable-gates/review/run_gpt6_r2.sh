#!/bin/bash
# Second-round GPT-6 review: were the 19 first-round findings applied, and is the revised summary supported?
cd /home/agent/workspaces/mats/work/tractable-gates
echo "r2 start $(date -u +%H:%M:%SZ)" >> review/review_status.txt
timeout 3000 codex-any exec --skip-git-repo-check -s read-only -m gpt-6-astra -c model_reasoning_effort='"high"' \
  -C /home/agent/workspaces/mats/work/tractable-gates \
  --add-dir /home/agent/workspaces/mats/outbox --add-dir /home/agent/workspaces/mats/work/cut-rank \
  "$(cat review/prompt_gpt6_r2.md)" < /dev/null > review/review_gpt6_r2.raw 2>&1
echo "r2 exit $? $(date -u +%H:%M:%SZ)" >> review/review_status.txt
