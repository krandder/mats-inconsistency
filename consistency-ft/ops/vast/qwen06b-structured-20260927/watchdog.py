"""Budget guard. Triggers: max_hours (instance.json), spend >= budget_usd (min($35, credit-$5) at rent), live credit < $5,
or 30 min without a lease refresh. Budget/time/credit triggers: 10-min best-effort rsync, then DESTROY
(a stopped instance still bills; the account must never reach zero with an instance alive).
Lease silence (sync broken): stop, keeping the disk for recovery."""
import json, pathlib, subprocess, time
from api import call
P = pathlib.Path(__file__).parent
x = json.loads((P / 'instance.json').read_text()); iid = x['new_contract']; t0 = x['created_at']
cap, floor = x['budget_usd'], x['credit_floor']
def log(m): (P / 'watchdog.log').open('a').write(time.strftime('%F %T ') + m + '\n')
log(f'watching {iid} cap ${cap:.2f} floor ${floor:.2f}')
reason = None; dph = x['quoted_dph']; spent_prev = 0.0; last = t0
while reason is None:
    time.sleep(30)
    if (P / 'destroyed.json').exists(): log('destroyed; exit'); raise SystemExit()
    credit = None
    try:
        dph = (call(f'instances/{iid}/').get('instances') or {}).get('dph_total') or dph
        credit = call('users/current/')['credit']
    except Exception as e:
        log('api ' + type(e).__name__)
    now = time.time(); spent_prev += (now - last) / 3600 * dph; last = now
    (P / 'spend.json').write_text(json.dumps({'at': now, 'spent_est': spent_prev, 'dph': dph, 'credit': credit}))
    lease = P / 'lease.json'
    silent = now - (lease.stat().st_mtime if lease.exists() else t0)
    if (now - t0) / 3600 >= x["max_hours"]: reason = f"destroy:{x['max_hours']}h cap"
    elif spent_prev >= cap: reason = f'destroy:${cap:.2f} cap (spent {spent_prev:.2f})'
    elif credit is not None and credit < floor: reason = f'destroy:credit {credit:.2f} below floor'
    elif silent > 1800: reason = f'stop:lease silent {silent/60:.0f} min'
log('TRIGGER ' + reason)
if reason.startswith('destroy'):
    try:
        subprocess.run(['bash', str(P / 'sync_once.sh')], timeout=600)
    except Exception as e:
        log('emergency sync ' + type(e).__name__)
for attempt in range(10):
    try:
        if reason.startswith('destroy'):
            r = call(f'instances/{iid}/', 'DELETE')
        else:
            r = call(f'instances/{iid}/', 'PUT', {'state': 'stopped'})
        (P / 'budget-stop.json').write_text(json.dumps({'at': time.time(), 'reason': reason, 'response': r})); log('DONE: ' + reason); break
    except Exception as e:
        log('action failed ' + type(e).__name__); time.sleep(30)
