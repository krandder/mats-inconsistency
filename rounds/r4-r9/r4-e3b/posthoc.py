"""Post-hoc checks (not pre-registered): the mandate-only reflex baseline ('BUY iff R2', ignores belief) and how
often each arm's greedy action equals it; the faith-swap flip rate the reflex itself would score."""
import json, sys
import numpy as np
sys.path.insert(0, '.')
from lir_rl import load, eval_q, payoff, preferred
test = load('test'); qs = eval_q(test); y = {it['id']: it['y'] for it in test}
out = {}
for arm in ('fused', 'tokmask', 'lir'):
    for s in (101, 102, 103):
        d = json.load(open(f'runs/{arm}-s{s}/eval-post.json'))
        mu = {'YES': dict(zip(d['ids'], d['mu_yes'])), 'NO': dict(zip(d['ids'], d['mu_no']))}
        eps = [(i, m) for i in d['ids'] for m in ('K7', 'R2')]
        pb = np.array(d['p_buy'])
        reflex = np.array([m == 'R2' for _, m in eps])
        pay = np.array([payoff(m, qs[i], y[i]) for i, m in eps])
        prefs = np.array([preferred(m, round(mu['YES' if m == 'K7' else 'NO'][i], 2), qs[i]) == 'BUY' for i, m in eps])
        # reflex flip score on the faith swap (mu from the note-YES pass, as in eval_action)
        fl = []
        for i in d['ids']:
            m_ = round(mu['YES'][i], 2)
            pp, pm = preferred('K7', m_, qs[i]), preferred('R2', m_, qs[i])
            if pp != pm:
                fl.append(pp == 'PASS' and pm == 'BUY')
        out[f'{arm}-s{s}'] = {'reward_reflex': float((reflex * pay).mean()),
                              'reward_threshold': float((prefs * pay).mean()),
                              'agent_reward': float((pb * pay).mean()),
                              'greedy_equals_reflex': float(((pb > 0.5) == reflex).mean()),
                              'greedy_equals_threshold': float(((pb > 0.5) == prefs).mean()),
                              'reflex_swap_flip_rate': float(np.mean(fl)) if fl else None}
json.dump(out, open('runs/posthoc.json', 'w'), indent=1)
for k, v in out.items():
    print(k, {a: round(b, 3) if b is not None else None for a, b in v.items()})
