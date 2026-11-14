#!/usr/bin/env python3
"""Budget/time/lease watchdog (local, mirrors data/vast-wave1/watchdog.py). Usage: watchdog.py <lane_dir>

Stops the instance (state=stopped, so GPU billing ends; storage keeps billing until finish.py destroys it) when:
  - max_hours from the manifest have elapsed since creation,
  - elapsed x quoted_dph exceeds budget_usd,
  - lease.json has not been refreshed for lease_stale_minutes (sync loop dead or box unreachable).
Exits quietly when destroyed.json appears. Writes budget-stop.json when it acts.
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from api import call  # noqa: E402


def main(argv):
    lane = pathlib.Path(argv[1])
    manifest = json.loads((lane / 'manifest.json').read_text())
    inst = json.loads((lane / 'instance.json').read_text())
    iid = inst['response']['new_contract']
    created = inst['created_at']
    deadline = created + manifest['max_hours'] * 3600
    stale = manifest.get('lease_stale_minutes', 30) * 60
    reason = None
    while True:
        time.sleep(30)
        if (lane / 'destroyed.json').exists():
            return 0
        now = time.time()
        if now > deadline:
            reason = f'max_hours {manifest["max_hours"]} reached'
        elif (now - created) / 3600 * inst['quoted_dph'] > manifest['budget_usd']:
            reason = f'budget {manifest["budget_usd"]} reached at ${inst["quoted_dph"]}/h'
        else:
            lease = lane / 'lease.json'
            last = lease.stat().st_mtime if lease.exists() else created
            if now - last > stale:
                reason = f'lease stale for {(now - last) / 60:.0f} min'
        if reason:
            break
    for attempt in range(10):
        try:
            response = call(f'instances/{iid}/', 'PUT', {'state': 'stopped'})
            (lane / 'budget-stop.json').write_text(json.dumps({'at': time.time(), 'reason': reason, 'response': response}))
            return 0
        except Exception as error:
            (lane / 'watchdog-error.txt').write_text(f'{type(error).__name__}: {error}\n')
            time.sleep(30)
    return 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
