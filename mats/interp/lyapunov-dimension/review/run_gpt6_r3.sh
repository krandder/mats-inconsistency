#!/bin/bash
cd /home/agent/workspaces/mats/work/lyapunov-dimension
timeout 1800 codex-any exec --skip-git-repo-check -s read-only -c model_reasoning_effort='"high"' -C /home/agent/workspaces/mats/work/lyapunov-dimension "$(cat review/prompt_gpt6_r3.md)" < /dev/null >> review/review_gpt6_r3.raw 2>&1
echo "gpt6 r3 exit $?" >> review/review_status.txt
