#!/usr/bin/env python3
"""Wait for the rented instance's SSH, upload code/data/bootstrap, run bootstrap.sh, record the result.

Usage: setup.py <lane_dir>
Reads instance.json (contract id) and asks the API for ssh_host/ssh_port, then writes them into instance.json so
watchdog/sync/finish never need the API for addressing. Mirrors data/vast-wave1/setup.py.
"""
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from api import call  # noqa: E402


def lease(lane, phase):
    (lane / 'lease.json').write_text(json.dumps({'phase': phase, 'at': time.time()}))


def ssh_endpoint(iid, attempts=60):
    for _ in range(attempts):
        for inst in call('instances/').get('instances', []):
            if inst.get('id') == iid and inst.get('ssh_host') and inst.get('ssh_port'):
                return inst['ssh_host'], int(inst['ssh_port']), inst.get('actual_status')
        time.sleep(15)
    raise SystemExit('instance never published an ssh endpoint')


def main(argv):
    lane = pathlib.Path(argv[1])
    manifest = json.loads((lane / 'manifest.json').read_text())
    inst = json.loads((lane / 'instance.json').read_text())
    iid = inst['response']['new_contract']
    host, port, status = ssh_endpoint(iid)
    inst.update(ssh_host=host, ssh_port=port, status_at_setup=status)
    (lane / 'instance.json').write_text(json.dumps(inst, indent=1))
    key = manifest['ssh_key']
    ssh = ['ssh', '-i', key, '-p', str(port), '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=10',
           f'root@{host}']
    scp = ['scp', '-i', key, '-P', str(port), '-o', 'ConnectTimeout=15']
    for _ in range(60):
        if subprocess.run(ssh + ['true'], capture_output=True, timeout=30).returncode == 0:
            break
        time.sleep(15)
    else:
        raise SystemExit('ssh never came up')
    lease(lane, 'setup')
    remote = manifest['remote_root']
    subprocess.run(ssh + [f'mkdir -p {remote}'], check=True, timeout=60)
    for name in ('code.tar.gz', 'data.tar.gz', 'bootstrap.sh', 'run.sh', 'manifest.json'):
        subprocess.run(scp + [str(lane / name), f'root@{host}:{remote}/{name}'], check=True, timeout=1800)
    result = subprocess.run(ssh + [f'cd {remote} && bash bootstrap.sh > bootstrap.log 2>&1; echo rc=$?; tail -5 bootstrap.log'],
                            timeout=3600, capture_output=True, text=True)
    (lane / 'setup-result.json').write_text(json.dumps({'out': result.stdout[-2000:], 'err': result.stderr[-500:],
                                                        'at': time.time(), 'ssh_host': host, 'ssh_port': port}))
    print(result.stdout[-2000:])
    if 'BOOTSTRAP_OK' not in result.stdout:
        lease(lane, 'setup-failed')
        raise SystemExit('bootstrap did not print BOOTSTRAP_OK; instance left running for inspection')
    lease(lane, 'setup-done')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
