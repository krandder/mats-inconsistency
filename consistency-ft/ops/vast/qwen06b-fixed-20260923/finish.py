"""Unattended finish: wait for QUEUE_DONE, final sync, sha256-verify against the box, destroy, confirm gone."""
import json, pathlib, subprocess, time, hashlib
from api import call
P = pathlib.Path(__file__).parent; D = P.parent
x = json.loads((P / 'instance.json').read_text()); IID = x['new_contract']
conn = dict(l.split('=', 1) for l in (P / 'conn.env').read_text().split())
K, H, PORT = conn['KEY'], conn['HOST'], conn['PORT']
PROG = pathlib.Path('/home/agent/workspaces/calibrated-ai/qwen06b-progress.log')
def ssh(cmd, t=600): return subprocess.run(['ssh', '-i', K, '-p', PORT, '-o', 'ConnectTimeout=20', H, cmd], capture_output=True, text=True, timeout=t).stdout
def log(msg): (P / 'finish.log').open('a').write(time.strftime('%F %T ') + msg + '\n'); print(msg, flush=True)
def progress(msg): PROG.open('a').write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' qwen06b-fixed ' + msg + '\n')
while 'QUEUE_DONE' not in ssh('cat /workspace/consistency/runs/status.txt'):
    time.sleep(60)
spend = json.loads((P / 'spend.json').read_text()) if (P / 'spend.json').exists() else {}
progress(f"training end: QUEUE_DONE on instance {IID}; est spend ${spend.get('spent_est', 0):.2f}; starting final sync")
subprocess.run(['pkill', '-f', 'qwen06b-fixed-20260923/ops/sync_loop.sh'])
(P / 'lease.json').write_text(json.dumps({'phase': 'final-sync', 'at': time.time()}))
ok = True
for src, dst in (('/workspace/consistency/runs/', 'results/'), ('/workspace/consistency/runs-timing/', 'results-timing/'),
                 ('/workspace/consistency/environment.json', 'results/environment.json'),
                 ('/workspace/consistency/pip-freeze.txt', 'results/pip-freeze.txt'),
                 ('/workspace/consistency/model-sha256.txt', 'results/model-sha256.txt'),
                 ('/workspace/consistency/bootstrap.log', 'results/bootstrap.log')):
    r = subprocess.run(['rsync', '-az', '-e', f'ssh -i {K} -p {PORT} -o ConnectTimeout=20', f'{H}:{src}', str(D / dst)],
                       capture_output=True, text=True, timeout=3600)
    log(f'rsync {src}: rc={r.returncode}'); ok &= r.returncode == 0
if not ok: log('SYNC FAILED; not destroying'); raise SystemExit(1)
bad = n = 0; lines = []
for rdir, ldir in (('runs', 'results'), ('runs-timing', 'results-timing')):
    remote = ssh(f'cd /workspace/consistency/{rdir} && find . -type f | sort | xargs -d "\\n" sha256sum', t=1800)
    for line in remote.splitlines():
        h, path = line.split(None, 1); n += 1; lines.append(f'{h}  {rdir}/{path[2:]}')
        local = D / ldir / path[2:]
        if not local.exists() or hashlib.sha256(local.read_bytes()).hexdigest() != h:
            bad += 1; log('MISMATCH ' + rdir + '/' + path)
log(f'verified {n} files, {bad} mismatches')
if bad or n == 0: log('not destroying'); raise SystemExit(1)
(D / 'SHA256SUMS-remote.txt').write_text('\n'.join(lines) + '\n')
r = call(f'instances/{IID}/', 'DELETE'); log('destroy response: ' + json.dumps(r))
for _ in range(20):
    time.sleep(15)
    alive = [i['id'] for i in call('instances/')['instances']]
    if IID not in alive: break
credit = call('users/current/')['credit']
(P / 'destroyed.json').write_text(json.dumps({'at': time.time(), 'response': r, 'still_listed': IID in alive, 'credit_after': credit}))
log(f'instance {IID} still listed: {IID in alive}; credit now ${credit:.2f}')
progress(f"destroy: instance {IID} deleted, still listed={IID in alive}; {n} files sha256-verified; credit ${credit:.2f}")
