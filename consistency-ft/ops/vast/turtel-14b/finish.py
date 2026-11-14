#!/usr/bin/env python3
"""Unattended finish (mirrors data/vast-wave1/finish.py): wait for the run to end, final sync, verify, destroy.

Usage: finish.py <lane_dir>
1. poll status.txt on the box until RUN_DONE / RUN_FAILED / TRAINER EXIT= appears (lease refreshed while waiting);
2. stop the sync loop; rsync runs/, environment.json, pip-freeze.txt, bootstrap.log, vllm-server.log;
3. verify: remote sha256 of every non-log file under runs/ equals the local copy; on any mismatch, do NOT destroy;
4. destroy the instance, confirm it is gone from the instance list, record credit.
A run that stopped early (watchdog, crash) still gets synced and verified; the checkpoint directory is what a
second lease resumes from (launch.sh --resume).
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from api import call  # noqa: E402


def main(argv):
    lane = pathlib.Path(argv[1])
    manifest = json.loads((lane / 'manifest.json').read_text())
    inst = json.loads((lane / 'instance.json').read_text())
    iid = inst['response']['new_contract']
    key, host, port = manifest['ssh_key'], inst['ssh_host'], str(inst['ssh_port'])
    remote = manifest['remote_root']
    markers = manifest['done_markers']

    def ssh(cmd, t=600):
        return subprocess.run(['ssh', '-i', key, '-p', port, '-o', 'ConnectTimeout=20', f'root@{host}', cmd],
                              capture_output=True, text=True, timeout=t).stdout

    def log(msg):
        with (lane / 'finish.log').open('a') as handle:
            handle.write(time.strftime('%H:%M:%S ') + msg + '\n')
        print(msg, flush=True)

    def lease(phase):
        (lane / 'lease.json').write_text(json.dumps({'phase': phase, 'at': time.time()}))

    while True:
        if (lane / 'budget-stop.json').exists():
            log('watchdog stopped the instance; proceeding to sync what exists')
            break
        status = ssh(f'cat {remote}/runs/turtel-14b-grpo/status.txt 2>/dev/null | tail -3')
        if any(m in status for m in markers):
            log('run ended: ' + ' | '.join(status.strip().splitlines()[-2:]))
            break
        lease('waiting-run')
        time.sleep(120)

    subprocess.run(['pkill', '-f', f'sync_loop.sh {lane}'])
    lease('final-sync')
    pairs = [(f'{remote}/runs/', 'results/')] + [(f'{remote}/{f}', f'results/{f}') for f in
                                                 ('environment.json', 'pip-freeze.txt', 'bootstrap.log', 'run.out')]
    for src, dst in pairs:
        r = subprocess.run(['rsync', '-az', '-e', f'ssh -i {key} -p {port} -o ConnectTimeout=20', f'root@{host}:{src}',
                            str(lane / dst)], capture_output=True, text=True, timeout=7200)
        log(f'rsync {src}: rc={r.returncode}')
        if r.returncode != 0 and src.endswith('runs/'):
            log('SYNC FAILED; not destroying')
            return 1

    remote_hashes = ssh(f'cd {remote}/runs && find . -type f -not -name "*.log" | sort | xargs -r sha256sum', t=3600)
    bad = n = 0
    for line in remote_hashes.splitlines():
        digest, path = line.split(None, 1)
        n += 1
        local = lane / 'results' / path[2:]
        if not local.exists() or hashlib.sha256(local.read_bytes()).hexdigest() != digest:
            bad += 1
            log('MISMATCH ' + path)
    log(f'verified {n} files, {bad} mismatches')
    if bad or n == 0:
        log('not destroying')
        return 1
    (lane / 'export-manifest.json').write_text(json.dumps({'files': n, 'remote_sha256': remote_hashes, 'at': time.time()}))

    r = call(f'instances/{iid}/', 'DELETE')
    log('destroy response: ' + json.dumps(r))
    time.sleep(20)
    alive = [i['id'] for i in call('instances/').get('instances', [])]
    credit = call('users/current/').get('credit')
    (lane / 'destroyed.json').write_text(json.dumps({'at': time.time(), 'response': r, 'still_listed': iid in alive,
                                                    'credit_after': credit}))
    log(f'instance {iid} still listed: {iid in alive}; credit now ${credit}')
    return 0 if iid not in alive else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
