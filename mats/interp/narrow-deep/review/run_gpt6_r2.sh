#!/bin/bash
cd /home/agent/workspaces/mats/work/narrow-deep
timeout 3600 codex-any exec --skip-git-repo-check -s read-only -c model_reasoning_effort='"high"' -C /home/agent/workspaces/mats/work/narrow-deep "$(cat review/prompt_gpt6_r2.md)" < /dev/null >> review/review_gpt6_r2.raw 2>&1
echo "gpt6 r2 exit $? $(date -u +%H:%M:%SZ)" >> review/review_status.txt
