"""Tables for the deviation-prediction and DAG sections (runs/*/devpred.json, runs/*/dag.json)."""
import json, os
import numpy as np

B = os.path.dirname(os.path.abspath(__file__))
RUNS = [('L2_s0.05', '2L seed 4'), ('L2seed1_s0.05', '2L seed 1'), ('L2seed2_s0.05', '2L seed 2'), ('L3_s0.05', '3L seed 13')]


def load(r, f):
    p = os.path.join(B, 'runs', r, f)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    print('### R² of predicted vs actual deviation (mean / log-sd / per-episode KL), 4,000 test episodes, K = 8\n')
    print('| model | deviation sd (mean / log-sd / KL) | noise floor | network = Bayes (0) | untrained net | best single-site surrogate (R² mean / log-sd / KL) [site:family] | MLP sites all LR16 | first attention site LRattn256 | cheap chain (LR4 + fixed pattern) |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r, name in RUNS:
        J = load(r, 'devpred.json')
        if J is None: continue
        rows = {x['config']: x for x in J['rows']}
        f3 = lambda d: ' / '.join('%.2f' % d[k] for k in ('mean', 'logsd', 'kl'))
        g3 = lambda x: ' / '.join('%.2f' % x[f'R2_{k}'] for k in ('mean', 'logsd', 'kl')) if x else '–'
        single = [x for c, x in rows.items() if ':' in c]
        best = max(single, key=lambda x: x['R2_mean'])
        cl = [c for c in rows if c.startswith('cum-cheap-to-')]; cheap_last = cl[-1] if cl else None
        print(f"| {name} | {J['dev_sd']['mean']:.3f} / {J['dev_sd']['logsd']:.3f} / {J['dev_sd']['kl']:.3f} | {f3(J['noise_floor'])} | {f3(J['zero'])} | "
              + ' / '.join('%.0f' % J['untrained'][k] for k in ('mean', 'logsd', 'kl')) + f" | {g3(best)} [{best['config']}] | {g3(rows.get('mlp-only-near'))} | "
              f"{g3(rows.get('a1:LRattn256'))} | {g3(rows.get(cheap_last)) if cheap_last else '–'} |")
    print('\n### Single-site surrogates: R²(mean) per site and family (seed 4; other seeds in the JSON)\n')
    J = load('L2_s0.05', 'devpred.json')
    if J:
        bysite = {}
        for x in J['rows']:
            if ':' in x['config'] and not x['config'].startswith('cum'):
                s, f = x['config'].split(':'); bysite.setdefault(s, []).append(f"{f} {x['R2_mean']:.2f}")
        for s, v in bysite.items(): print(f"- {s}: " + '; '.join(v))
    print('\n### Scenario predictions (fresh 32,768 episodes per seed); measured statistic and hit (Y/N)\n')
    keys = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']
    stat = {'S1': lambda s: f"KL×{s['KL_ratio']:.2f}, dlogsd {s['mean_dlogsd']:+.3f}", 'S2': lambda s: f"KL×{s['KL_ratio']:.2f}, dlogsd {s['mean_dlogsd']:+.3f}",
            'S3': lambda s: f"RMS×{s['RMS_ratio']:.2f}, KL×{s['KL_ratio']:.2f}", 'S4': lambda s: f"KL×{s['KL_ratio']:.2f}", 'S5': lambda s: f"KL×{s['KL_ratio']:.2f}",
            'S6': lambda s: f"RMS×{s['RMS_ratio']:.2f}", 'S7': lambda s: f"agree {s['sign_agree']:.2f} (base {s['baseline_sign_agree']:.2f})",
            'S8': lambda s: f"signed {s['mean_signed']:+.3f}, RMS×{s['RMS_ratio']:.2f}", 'S9': lambda s: f"KL×{s['KL_ratio']:.2f}"}
    print('| scenario | ' + ' | '.join(n for _, n in RUNS) + ' |'); print('|---|' + '---|' * len(RUNS))
    for k in keys:
        cells = []
        for r, _ in RUNS:
            J = load(r, 'devpred.json')
            if J is None or 'scenarios' not in J: cells.append('–'); continue
            s = J['scenarios'][k]; cells.append(f"{stat[k](s)} ({'Y' if s['hit'] else 'N'})")
        print(f"| {k} | " + ' | '.join(cells) + ' |')
    print('\n### Lateral arm: sparse surrogate DAGs (KL over the DAG, deviation R², treewidth)\n')
    print('| model | variant | KL (nats) | R² mean / log-sd / KL | scenario hit agreement with the network | treewidth | largest clique dim | attention edges |'); print('|---|---|---|---|---|---|---|---|')
    for r, name in RUNS:
        J = load(r, 'dag.json')
        if J is None: continue
        for x in J['rows']:
            r2 = ' / '.join('%.2f' % x[f'R2_{k}'] for k in ('mean', 'logsd', 'kl')) if 'R2_mean' in x else f"noise floor {J['noise_floor']['mean']:.2f} / {J['noise_floor']['logsd']:.2f} / {J['noise_floor']['kl']:.2f}"
            print(f"| {name} | {x['variant']} | {x['KL']:.1f} | {r2} | {x.get('scenario_hit_agreement', float('nan')):.2f} | {x['treewidth']} | {x['max_clique_dim']} | {x['attention_edges']} |")


if __name__ == '__main__':
    main()
