#!/bin/bash
cd /home/agent/workspaces/mats/work/hardness/review
timeout 3600 agy --model gemini-4-argon-high -p "$(cat PROMPT_G1.md)" < /dev/null > review_gemini_g1.md 2>&1
echo "G1 exit $?" >> review_status.txt
