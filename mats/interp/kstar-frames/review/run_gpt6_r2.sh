#!/bin/bash
cd /home/agent/workspaces/mats/work/kstar-frames
echo "r2 start $(date -u +%H:%M:%SZ)" >> review/review_status.txt
timeout 2400 codex-any exec --skip-git-repo-check -s read-only -m gpt-6-astra -c model_reasoning_effort='"high"' -C /home/agent/workspaces/mats/work/kstar-frames "$(cat review/prompt_gpt6_r2.md)" < /dev/null > review/review_gpt6_r2.raw 2>&1
echo "gpt6 r2 exit $? $(date -u +%H:%M:%SZ)" >> review/review_status.txt
