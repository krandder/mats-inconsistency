"""E45 runner (PREREG-B amendment 53): single-pass LLM answers on sign-negated pairs. Reuses E18b's call functions unchanged
(e18b_call.ask: claude -p with thinking off, a 16-token cap and E18b's SYS; codex-any gpt-5.6-sol at reasoning "none" with tools
disabled; local Qwen3-1.7B, greedy). Arms: S_ (stated) and U_ (unstated) x haiku45, sonnet46, opus48, gpt56 (2 samples per
prompt) and qwen17 (1 sample; greedy). Each call is appended to results/e45_calls.jsonl; resumable; the answer is the leading number;
non-compliant responses are retried up to 3 times; an arm stops after 10 consecutive failures.
usage: python e45_call.py ARM
"""
import json, os, sys, time
import numpy as np

import e18b_call as C

ARMS = {f'{f}_{m}': (m, fr, 1 if m == 'qwen17' else 2)
        for m in ('haiku45', 'sonnet46', 'opus48', 'gpt56', 'qwen17') for f, fr in (('S', 'stated'), ('U', 'unstated'))}
LOG = 'results/e45_calls.jsonl'


def done_keys():
    out = set()
    if os.path.exists(LOG):
        for line in open(LOG):
            r = json.loads(line)
            if r['ok'] and C.parse(r['text']) is not None:
                out.add((r['arm'], r['pair'], r['version'], r['sample']))
    return out


def main():
    arm = sys.argv[1]; model, framing, nsamp = ARMS[arm]
    items = json.load(open('results/e45_items.json')); done = done_keys(); fails = 0
    for smp in range(nsamp):
        for i in np.random.default_rng(45 + smp).permutation(len(items)):
            it = items[i]; k = (arm, it['pair'], it['version'], smp)
            if k in done:
                continue
            for attempt in range(4):
                try:
                    r = C.ask(model, it[framing])
                except Exception as e:
                    r = dict(text='', ok=False, err=f'{type(e).__name__}: {str(e)[:200]}')
                ans = C.parse(r.get('text')); ok = bool(r.get('ok')) and ans is not None
                rec = dict(arm=arm, model=model, framing=framing, pair=it['pair'], version=it['version'], sample=smp, attempt=attempt,
                           answer=ans, ok=ok, text=r.get('text') or '', t=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                           **{kk: v for kk, v in r.items() if kk not in ('text', 'ok')}, harness_ok=bool(r.get('ok')))
                with open(LOG, 'a') as fh:
                    fh.write(json.dumps(rec) + '\n')
                if ok:
                    fails = 0; break
                fails += 1
                if fails >= 10:
                    print(arm, 'stopping after 10 consecutive failures', flush=True); return
                time.sleep(5)
            if model != 'qwen17':
                time.sleep(1)
    print(arm, 'done', flush=True)


if __name__ == '__main__':
    main()
