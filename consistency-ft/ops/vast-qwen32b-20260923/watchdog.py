"""Stop the instance at 8 h, at $30 spent, or after 30 min without a lease refresh."""
import json, pathlib, time
from api import call
P = pathlib.Path(__file__).parent
x = json.loads((P / 'instance.json').read_text()); iid = x['new_contract']; t0 = x['created_at']
def log(m): (P / 'watchdog.log').open('a').write(time.strftime('%F %T ') + m + '\n')
log(f'watching {iid}')
reason = None
while reason is None:
    time.sleep(30)
    if (P / 'destroyed.json').exists(): log('destroyed; exit'); raise SystemExit()
    dph = x['quoted_dph']
    try:
        inst = call(f'instances/{iid}/').get('instances') or {}
        dph = inst.get('dph_total') or dph
    except Exception as e:
        log('api ' + type(e).__name__)
    hours = (time.time() - t0) / 3600
    lease = P / 'lease.json'
    silent = time.time() - (lease.stat().st_mtime if lease.exists() else t0)
    if hours >= 8: reason = '8h cap'
    elif hours * dph >= 30: reason = f'$30 cap ({hours*dph:.2f})'
    elif silent > 1800: reason = f'lease silent {silent/60:.0f} min'
for attempt in range(10):
    try:
        r = call(f'instances/{iid}/', 'PUT', {'state': 'stopped'})
        (P / 'budget-stop.json').write_text(json.dumps({'at': time.time(), 'reason': reason, 'response': r})); log('STOPPED: ' + reason); break
    except Exception as e:
        log('stop failed ' + type(e).__name__); time.sleep(30)
