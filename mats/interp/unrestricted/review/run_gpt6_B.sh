#!/bin/bash
# Final GPT-6 review B (whole report), fresh session, read-only.
cd /home/agent/workspaces/mats/work/unrestricted
echo "B start $(date -u +%H:%M:%SZ)" >> review/review_status.txt
timeout 3600 codex-any exec --skip-git-repo-check -s read-only -m gpt-6-astra -c model_reasoning_effort='"high"' -C /home/agent/workspaces/mats/work/unrestricted --add-dir /home/agent/workspaces/mats/outbox "$(cat review/prompt_gpt6_B.md)" < /dev/null > review/review_gpt6_B.raw 2>&1
echo "B exit $? $(date -u +%H:%M:%SZ)" >> review/review_status.txt
