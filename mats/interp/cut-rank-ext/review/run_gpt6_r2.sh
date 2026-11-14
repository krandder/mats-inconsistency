#!/bin/bash
# Focused GPT-6 re-check of the revised cut-rank-ext report (read-only sandbox, fresh session).
cd /home/agent/workspaces/mats/work/cut-rank-ext
echo "r2 start $(date -u +%H:%M:%SZ)" >> review/review_status.txt
timeout 3600 nice -n 19 codex-any exec --skip-git-repo-check -s read-only -m gpt-6-astra -c model_reasoning_effort='"high"' -C /home/agent/workspaces/mats/work/cut-rank-ext --add-dir /home/agent/workspaces/mats/outbox --add-dir /home/agent/workspaces/mats/work/cut-rank "$(cat review/prompt_gpt6_r2.md)" < /dev/null > review/review_gpt6_r2.raw 2>&1
echo "r2 exit $? $(date -u +%H:%M:%SZ)" >> review/review_status.txt
