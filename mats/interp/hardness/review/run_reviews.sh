#!/bin/bash
# GPT-6 adversarial reviews (codex-any exec), read-only, in parallel (remote models; local CPU light)
D=/home/agent/workspaces/mats/work/hardness
cd $D/review
mkdir -p /tmp/hardness_review_r1 /tmp/hardness_review_r2
( codex-any exec --skip-git-repo-check -s read-only -c model_reasoning_effort='"xhigh"' -C $D "$(cat PROMPT_R1.md)" < /dev/null > review_gpt6_r1.md 2>&1; echo "R1 exit $?" >> review_status.txt ) &
( codex-any exec --skip-git-repo-check -s read-only -c model_reasoning_effort='"xhigh"' -C $D "$(cat PROMPT_R2.md)" < /dev/null > review_gpt6_r2.md 2>&1; echo "R2 exit $?" >> review_status.txt ) &
wait
echo ALLDONE >> review_status.txt
