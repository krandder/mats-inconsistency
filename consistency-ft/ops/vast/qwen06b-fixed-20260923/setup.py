"""Wait for the instance, record ssh host/port in conn.env, upload code+data, bootstrap."""
import json, pathlib, subprocess, time
from api import call
P = pathlib.Path(__file__).parent
x = json.loads((P / 'instance.json').read_text()); iid = x['new_contract']
KEY = '[ssh-key-path-removed]'
for _ in range(80):
    i = call(f'instances/{iid}/').get('instances') or {}
    ports = (i.get('ports') or {}).get('22/tcp')
    if i.get('actual_status') == 'running' and i.get('public_ipaddr') and ports: break
    time.sleep(15)
else: raise SystemExit('instance never running: ' + str(i.get('actual_status')))
host, port = i['public_ipaddr'].strip(), int(ports[0]['HostPort'])  # ssh_direct runtype
(P / 'conn.env').write_text(f'HOST=root@{host}\nPORT={port}\nKEY={KEY}\n')
ssh = ['ssh', '-i', KEY, '-p', str(port), '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=15', 'root@' + host]
scp = ['scp', '-i', KEY, '-P', str(port), '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=15']
for _ in range(40):
    if subprocess.run(ssh + ['true'], capture_output=True, timeout=30).returncode == 0: break
    time.sleep(15)
else: raise SystemExit('ssh never came up')
(P / 'lease.json').write_text(json.dumps({'phase': 'setup', 'at': time.time()}))
subprocess.run(ssh + ['mkdir -p /workspace/consistency'], check=True)
for f in ['code.tar.gz', 'data.tar.gz', 'bootstrap.sh', 'remote_timing.sh', 'remote_queue.sh']:
    subprocess.run(scp + [str(P / f), f'root@{host}:/workspace/consistency/{f}'], check=True)
r = subprocess.run(ssh + ['cd /workspace/consistency && bash bootstrap.sh > bootstrap.log 2>&1; echo rc=$?; tail -3 bootstrap.log'],
                   timeout=1800, capture_output=True, text=True)
(P / 'setup-result.json').write_text(json.dumps({'out': r.stdout[-800:], 'at': time.time()}))
(P / 'lease.json').write_text(json.dumps({'phase': 'setup-done', 'at': time.time()}))
print(r.stdout[-800:])
