#!/bin/bash
# GPT-6 adversarial review of the narrow-deep report (read-only sandbox).
cd /home/agent/workspaces/mats/work/narrow-deep
timeout 3600 codex-any exec --skip-git-repo-check -s read-only -c model_reasoning_effort='"high"' -C /home/agent/workspaces/mats/work/narrow-deep "$(cat review/prompt_gpt6.md)" < /dev/null >> review/review_gpt6.raw 2>&1
echo "gpt6 exit $?" >> review/review_status.txt
