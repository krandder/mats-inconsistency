#!/bin/bash
D=/home/agent/workspaces/mats/work/hardness
cd $D/review
codex-any exec --skip-git-repo-check -s read-only -c model_reasoning_effort='"high"' -C $D "$(cat PROMPT_R3.md)" < /dev/null > review_gpt6_r3.md 2>&1
echo "R3 exit $?" >> review_status.txt
