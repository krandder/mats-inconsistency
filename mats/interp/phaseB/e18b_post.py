"""E18b post hoc (written after the registered E18b results): pooled shift over the four API models (Qwen excluded, whose answers
were near noise, mean absolute deviation from Bayes 5-7), bootstrap over common pairs; and a check that every accepted Claude call
used zero thinking tokens. usage: python e18b_post.py -> results/e18b_post.json
"""
import json
import numpy as np


def main():
    o = json.load(open('results/e18b.json')); A = o['arms']; rng = np.random.default_rng(0); out = {}
    shift = lambda rows: sum((r['mb'] - r['ms']) * np.sign(r['L']) for r in rows) / sum(r['W'] * abs(r['L']) for r in rows)
    for fr in ('S', 'U'):
        ev = [f'{fr}_{m}' for m in ('haiku45', 'sonnet46', 'opus48', 'gpt56')]
        common = sorted(set.intersection(*[set(r['pair'] for r in A[a]['rows']) for a in ev]))
        pooled = lambda ks: float(np.mean([shift([{r['pair']: r for r in A[a]['rows']}[k] for k in ks]) for a in ev]))
        bs = [pooled([common[i] for i in rng.integers(0, len(common), len(common))]) for _ in range(1000)]
        out[fr] = dict(pooled=pooled(common), ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], n_pairs=len(common))
    rows = [json.loads(l) for l in open('results/e18b_calls.jsonl')]
    th = [r.get('think_tok') for r in rows if r['model'] in ('haiku45', 'sonnet46', 'opus48') and r['ok']]
    out['claude_accepted_calls'] = len(th); out['claude_accepted_with_thinking'] = int(sum(1 for t in th if t))
    json.dump(out, open('results/e18b_post.json', 'w'), indent=1); print(out)


if __name__ == '__main__':
    main()
