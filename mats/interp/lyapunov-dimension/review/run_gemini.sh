#!/bin/bash
cd /home/agent/workspaces/mats/work/lyapunov-dimension/review
timeout 3600 agy --model gemini-4-argon-high -p "$(cat prompt_gemini.md)" < /dev/null >> review_gemini.md 2>&1
echo "gemini exit $?" >> review_status.txt
