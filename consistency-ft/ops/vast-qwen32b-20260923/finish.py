"""Final sync of /workspace/q32/{out,out-*.log,smoke,environment}, sha256 verify, destroy, confirm via API."""
import hashlib, json, pathlib, subprocess, time
from api import call
P = pathlib.Path(__file__).parent; R = P.parent
x = json.loads((P / 'instance.json').read_text()); IID = x['new_contract']
K = '[ssh-key-path-removed]'; H = 'root@[host-removed]'; PORT = '34742'
def ssh(cmd, t=600): return subprocess.run(['ssh', '-n', '-i', K, '-p', PORT, '-o', 'ConnectTimeout=20', H, cmd], capture_output=True, text=True, timeout=t).stdout
def log(m): (P / 'finish.log').open('a').write(time.strftime('%F %T ') + m + '\n'); print(m, flush=True)
(P / 'stop-sync').write_text('1')
ssh('cd /workspace/q32 && python3 -m pip freeze > pip-freeze.txt 2>/dev/null; nvidia-smi > nvidia-smi.txt')
dst = R / 'remote-final'; dst.mkdir(exist_ok=True)
r = subprocess.run(['rsync', '-az', '--exclude', 'models/', '--exclude', 'code.tar.gz', '--exclude', 'data.tar.gz', '--exclude', 'data/', '--exclude', 'code/',
                    '-e', f'ssh -i {K} -p {PORT} -o ConnectTimeout=20', f'{H}:/workspace/q32/', str(dst) + '/'], capture_output=True, text=True, timeout=3600)
log(f'rsync rc={r.returncode} {r.stderr[-300:]}')
if r.returncode: log('SYNC FAILED; not destroying'); raise SystemExit(1)
remote = ssh('cd /workspace/q32 && find . -type f -not -path "./models/*" -not -path "./data/*" -not -path "./code/*" -not -name "*.tar.gz" | sort | xargs sha256sum', 1200)
bad = n = 0
for line in remote.splitlines():
    h, path = line.split(None, 1); n += 1
    f = dst / path[2:]
    if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != h: bad += 1; log('MISMATCH ' + path)
log(f'verified {n} files, {bad} mismatches')
if bad or n == 0: log('not destroying'); raise SystemExit(1)
(R / 'export-manifest.json').write_text(json.dumps({'files': n, 'remote_sha256': remote, 'at': time.time()}, indent=1))
inst = call(f'instances/{IID}/')['instances']
cost = {'dph_total': inst.get('dph_total'), 'start_date': inst.get('start_date'), 'end_at': time.time()}
r = call(f'instances/{IID}/', 'DELETE'); log('destroy response: ' + json.dumps(r))
time.sleep(20)
alive = [i['id'] for i in call('instances/')['instances']]
credit = call('users/current/')['credit']
(P / 'destroyed.json').write_text(json.dumps({'at': time.time(), 'response': r, 'still_listed': IID in alive, 'credit_after': credit, **cost}))
log(f'instance {IID} still listed: {IID in alive}; credit now ${credit:.2f}')
