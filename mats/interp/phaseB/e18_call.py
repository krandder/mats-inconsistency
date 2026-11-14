"""E18 runner (PREREG-B amendment 25): asks one arm's model every registered item, appending one JSON line per call to
results/e18_calls.jsonl. Resumable; paced (1 s between calls); a failed call waits 30 s and is retried up to 3 times; an arm stops
after 10 consecutive failures (to avoid loading the shared proxy).
usage: python e18_call.py ARM
"""
import json, os, sys, time
import numpy as np
import e18_llm as L

ARMS = {  # arm: (model key, framing, number of pairs, samples per prompt)
    'S_haiku': ('haiku', 'stated', 60, 2), 'S_sonnet': ('sonnet', 'stated', 60, 2), 'S_opus': ('opus', 'stated', 60, 2),
    'S_gpt': ('gpt', 'stated', 60, 1), 'R_opus': ('opus_high', 'stated', 30, 1), 'U_sonnet': ('sonnet', 'unstated', 60, 1)}
LOG = 'results/e18_calls.jsonl'


def done_keys():
    out = set()
    if os.path.exists(LOG):
        for line in open(LOG):
            r = json.loads(line)
            if r.get('ok') and L.parse(r.get('text')) is not None:      # addendum 3: re-parse stored text
                out.add((r['arm'], r['pair'], r['version'], r['sample']))
    return out


def main():
    arm = sys.argv[1]; key, framing, npairs, nsamp = ARMS[arm]
    items = [it for it in json.load(open('results/e18_items.json')) if it['pair'] < npairs]
    done = done_keys(); fails = 0
    for smp in range(nsamp):
        order = np.random.default_rng(7 + smp).permutation(len(items))
        for i in order:
            it = items[i]; k = (arm, it['pair'], it['version'], smp)
            if k in done:
                continue
            for attempt in range(4):
                try:
                    r = L.ask(key, it[framing])
                except Exception as e:
                    r = dict(text='', ok=False, err=f'{type(e).__name__}: {str(e)[:300]}')
                ans = L.parse(r.get('text'))
                ok = bool(r.get('ok')) and ans is not None and not r.get('used_tools', False)
                rec = dict(arm=arm, key=key, framing=framing, pair=it['pair'], version=it['version'], sample=smp, attempt=attempt, answer=ans,
                           ok=ok, text=(r.get('text') or ''), t=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                           **{kk: v for kk, v in r.items() if kk not in ('text', 'ok')}, harness_ok=bool(r.get('ok')))
                with open(LOG, 'a') as fh:
                    fh.write(json.dumps(rec) + '\n')
                if ok:
                    fails = 0; break
                fails += 1
                if fails >= 10:
                    print(arm, 'stopping after 10 consecutive failures', flush=True); return
                time.sleep(30)
            time.sleep(1)
    print(arm, 'done', flush=True)


if __name__ == '__main__':
    main()
