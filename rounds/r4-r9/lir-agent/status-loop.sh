#!/bin/bash
cd /home/agent/workspaces/eurisko-study/lir-agent
while [ ! -f REPORT.md ]; do python3 status.py >> STATUS.log 2>&1; sleep 300; done
echo "$(date -u +%FT%TZ) REPORT.md exists; status loop exiting" >> STATUS.log
