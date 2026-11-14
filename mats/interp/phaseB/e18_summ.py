"""E18 analysis (PREREG-B amendment 25): do pretrained LLMs give a source with one big miss more influence than one with the same
squared total spread out, when the stated model makes the two equivalent?
usage: python e18_summ.py -> results/e18.json, results/e18_tables.md
"""
import json
import numpy as np

NBOOT = 1000
ARMS = ['S_haiku', 'S_sonnet', 'S_opus', 'S_gpt', 'R_opus', 'U_sonnet']


def main():
    items = json.load(open('results/e18_items.json'))
    by = {(it['pair'], it['version']): it for it in items}
    npair = max(it['pair'] for it in items) + 1
    calls = [json.loads(l) for l in open('results/e18_calls.jsonl')]
    import e18_llm as LL
    for c in calls:                                                     # addendum 3: the answer is the leading number
        c['answer'] = LL.parse(c.get('text'))
    good = [c for c in calls if c['ok'] and c['answer'] is not None]
    ans = {}
    for c in good:
        ans.setdefault((c['arm'], c['pair'], c['version']), {})[c['sample']] = c['answer']     # last good answer per sample
    out = dict(arms={}, predictions={}, spend={})

    def arm_stats(arm, pairs):
        rows = []
        for k in pairs:
            b, s = ans.get((arm, k, 'big')), ans.get((arm, k, 'spread'))
            if not b or not s:
                continue
            ib, isp = by[(k, 'big')], by[(k, 'spread')]
            src = ib['src']; L = isp['readings'][src] - isp['bayes_mean']; Wb = isp['bayes_w'][src]
            rows.append(dict(pair=k, mb=np.mean(list(b.values())), ms=np.mean(list(s.values())), L=L, W=Wb,
                             db=ib['bayes_mean'] - isp['bayes_mean'], sdb=np.std(list(b.values())) if len(b) > 1 else np.nan,
                             sds=np.std(list(s.values())) if len(s) > 1 else np.nan))
        return rows

    def shift(rows, idx=None):
        rr = rows if idx is None else [rows[i] for i in idx]
        num = sum((r['mb'] - r['ms']) * np.sign(r['L']) for r in rr); den = sum(r['W'] * abs(r['L']) for r in rr)
        return num / den

    rng = np.random.default_rng(0)
    for arm in ARMS:
        n_arm = 30 if arm == 'R_opus' else npair
        rows = arm_stats(arm, range(n_arm))
        if not rows:
            continue
        bs = [shift(rows, rng.integers(0, len(rows), len(rows))) for _ in range(NBOOT)]
        # reliability use over all answered prompts of the arm: slope of (m - a) on (mu_B - a), a = mean of the readings
        x, y = [], []
        for (a_, k, v), dd in ans.items():
            if a_ != arm or k >= n_arm:
                continue
            it = by[(k, v)]; a = np.mean(it['readings'])
            x.append(it['bayes_mean'] - a); y.append(np.mean(list(dd.values())) - a)
        x, y = np.array(x), np.array(y)
        slope = float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))
        mad = float(np.mean([abs(np.mean(list(dd.values())) - by[(k, v)]['bayes_mean']) for (a_, k, v), dd in ans.items() if a_ == arm and k < n_arm]))
        sd = [r['sdb'] for r in rows if not np.isnan(r['sdb'])] + [r['sds'] for r in rows if not np.isnan(r['sds'])]
        out['arms'][arm] = dict(n_pairs=len(rows), shift=float(shift(rows)), ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                                bayes_shift=float(sum(r['db'] * np.sign(r['L']) for r in rows) / sum(r['W'] * abs(r['L']) for r in rows)),
                                n_pos_pairs=int(sum((r['mb'] - r['ms']) * np.sign(r['L']) > 0 for r in rows)),
                                n_neg_pairs=int(sum((r['mb'] - r['ms']) * np.sign(r['L']) < 0 for r in rows)),
                                reliability_slope=slope, mean_abs_dev_from_bayes=mad,
                                within_prompt_sd=float(np.mean(sd)) if sd else None, rows=rows)
    A = out['arms']
    S = [a for a in ('S_haiku', 'S_sonnet', 'S_opus', 'S_gpt') if a in A]
    ev = [a for a in S if A[a]['reliability_slope'] >= 0.3]
    # pooled over evaluable models: mean of per-model shifts, bootstrap over pairs jointly
    common = sorted(set.intersection(*[set(r['pair'] for r in A[a]['rows']) for a in ev])) if ev else []
    def pooled(idx_pairs):
        vals = []
        for a in ev:
            rr = {r['pair']: r for r in A[a]['rows']}
            vals.append(shift([rr[k] for k in idx_pairs]))
        return float(np.mean(vals))
    pb = [pooled([common[i] for i in rng.integers(0, len(common), len(common))]) for _ in range(NBOOT)] if common else []
    out['predictions']['P115'] = dict(evaluable=ev, n_pos=int(sum(A[a]['shift'] > 0 for a in ev)),
                                      pooled=pooled(common) if common else None, pooled_ci=[float(np.quantile(pb, 0.025)), float(np.quantile(pb, 0.975))] if pb else None,
                                      testable=len(ev) >= 2)
    p = out['predictions']['P115']
    p['holds'] = bool(p['testable'] and p['n_pos'] >= len(ev) - 1 and p['pooled_ci'][0] > 0)
    if 'R_opus' in A and 'S_opus' in A:
        lo = shift([r for r in A['S_opus']['rows'] if r['pair'] < 30])
        out['predictions']['P116'] = dict(high=A['R_opus']['shift'], low_same_pairs=float(lo), holds=bool(abs(A['R_opus']['shift']) < 0.5 * abs(lo)))
    if 'U_sonnet' in A and 'S_sonnet' in A:
        out['predictions']['P117'] = dict(unstated=A['U_sonnet']['shift'], stated=A['S_sonnet']['shift'],
                                          holds=bool(A['U_sonnet']['shift'] > A['S_sonnet']['shift']))
    for arm in ARMS:
        cc = [c for c in calls if c['arm'] == arm]
        if cc:
            out['spend'][arm] = dict(calls=len(cc), ok=int(sum(c['ok'] for c in cc)), cost_usd=float(sum(c.get('cost') or 0 for c in cc)),
                                     gpt_tokens=int(sum(c.get('tokens') or 0 for c in cc)), seconds=float(sum(c.get('sec') or 0 for c in cc)),
                                     tool_use=int(sum(bool(c.get('used_tools')) for c in cc)))
    json.dump(out, open('results/e18.json', 'w'), indent=1, default=float)
    for k, v in out['predictions'].items():
        print(k, v)
    T = ['| arm | model / effort | framing | pairs | shift [95% CI] | pairs + / − | Bayes shift (rounding) | reliability slope | mean abs dev from Bayes | within-prompt sd |',
         '|---|---|---|---|---|---|---|---|---|---|']
    desc = {'S_haiku': 'Haiku 5.5 / low', 'S_sonnet': 'Sonnet 5.5 / low', 'S_opus': 'Opus 5.5 / low', 'S_gpt': 'GPT-6 / low',
            'R_opus': 'Opus 5.5 / high', 'U_sonnet': 'Sonnet 5.5 / low'}
    for arm, v in A.items():
        T.append(f"| {arm} | {desc[arm]} | {'stated' if arm != 'U_sonnet' else 'unstated'} | {v['n_pairs']} | {v['shift']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | "
                 f"{v['n_pos_pairs']} / {v['n_neg_pairs']} | {v['bayes_shift']:+.3f} | {v['reliability_slope']:.2f} | {v['mean_abs_dev_from_bayes']:.3f} | "
                 f"{'' if v['within_prompt_sd'] is None else '%.3f' % v['within_prompt_sd']} |")
    T += ['', '| arm | calls | ok | cost (USD, Claude list price) | GPT tokens | seconds | calls that used tools |', '|---|---|---|---|---|---|---|']
    for arm, v in out['spend'].items():
        T.append(f"| {arm} | {v['calls']} | {v['ok']} | {v['cost_usd']:.2f} | {v['gpt_tokens']} | {v['seconds']:.0f} | {v['tool_use']} |")
    open('results/e18_tables.md', 'w').write('\n'.join(T) + '\n')


if __name__ == '__main__':
    main()
