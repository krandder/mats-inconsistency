#!/usr/bin/env python3
"""Start (or resume) run.sh on the rented box under nohup and confirm the trainer process exists.

Usage: start.py <lane_dir> [--resume]
"""
import json
import pathlib
import subprocess
import sys
import time


def main(argv):
    lane = pathlib.Path(argv[1])
    manifest = json.loads((lane / 'manifest.json').read_text())
    inst = json.loads((lane / 'instance.json').read_text())
    key, host, port = manifest['ssh_key'], inst['ssh_host'], inst['ssh_port']
    ssh = ['ssh', '-i', key, '-p', str(port), '-o', 'ConnectTimeout=20', f'root@{host}']
    remote = manifest['remote_root']
    extra = '--resume' if '--resume' in argv else ''
    subprocess.run(ssh + [f'cd {remote} && nohup bash run.sh {extra} > run.out 2>&1 & echo started'], check=True,
                   timeout=60)
    time.sleep(20)
    check = subprocess.run(ssh + ['pgrep -fa "turtel_grpo.py|vllm-serve|vllm.entrypoints" | head -5; tail -3 ' + remote + '/runs/turtel-14b-grpo/status.txt'],
                           capture_output=True, text=True, timeout=60)
    print(check.stdout)
    (lane / 'lease.json').write_text(json.dumps({'phase': 'running' if not extra else 'resumed', 'at': time.time()}))
    (lane / 'start.json').write_text(json.dumps({'at': time.time(), 'resume': bool(extra), 'check': check.stdout[-1500:]}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
