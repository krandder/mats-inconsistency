#!/bin/sh
# Start serve.py unless it is already listening. Run from cron every 5 minutes (see README).
here=$(cd "$(dirname "$0")" && pwd)
bind=${TRADERS_EXPORT_BIND:-[ip-removed]}
port=${TRADERS_EXPORT_PORT:-8767}
exec 9>"$here/.serve.lock"
flock -n 9 || exit 0
if curl -s -m 5 -o /dev/null "http://$bind:$port/runs.json"; then exit 0; fi
echo "$(date -u +%FT%TZ) starting serve.py on $bind:$port"
nohup python3 "$here/serve.py" --bind "$bind" --port "$port" >>"$here/serve.log" 2>&1 9>&- &
