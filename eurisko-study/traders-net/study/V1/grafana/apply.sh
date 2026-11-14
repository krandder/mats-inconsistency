#!/bin/sh
# Usage: ./apply.sh [--dashboard-only]   (rebuilds traders-net.json, then applies via deploy-host's kubectl)
set -e
cd "$(dirname "$0")"
python3 build_dashboard.py >/dev/null
{ python3 -c 'import json;print(json.dumps({"dashboard":open("traders-net.json").read(),"datasource":open("datasource.yaml").read(),"provider":open("provider.yaml").read(),"host":"[ip-removed]"}))'; cat apply.py; } \
  | ssh deploy-host "python3 -c 'import sys; line=sys.stdin.readline(); code=sys.stdin.read(); import io; sys.stdin=io.StringIO(line); sys.argv=[\"apply.py\"]+sys.argv[1:]; exec(compile(code,\"apply.py\",\"exec\"))' $*"
