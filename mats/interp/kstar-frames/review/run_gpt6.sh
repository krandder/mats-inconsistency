#!/bin/bash
# GPT-6 (gpt-6-astra) adversarial review of the kstar-frames report and code (read-only sandbox).
cd /home/agent/workspaces/mats/work/kstar-frames
echo "start $(date -u +%H:%M:%SZ)" >> review/review_status.txt
timeout 3600 codex-any exec --skip-git-repo-check -s read-only -m gpt-6-astra -c model_reasoning_effort='"high"' -C /home/agent/workspaces/mats/work/kstar-frames "$(cat review/prompt_gpt6.md)" < /dev/null > review/review_gpt6.raw 2>&1
echo "gpt6 exit $? $(date -u +%H:%M:%SZ)" >> review/review_status.txt
