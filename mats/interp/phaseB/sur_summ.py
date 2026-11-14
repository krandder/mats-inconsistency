"""Tables for the surrogate-chain section (results/sur_small_s*.json, runs/L2_s*/sur.json)."""
import json, os, glob
import numpy as np

B = os.path.dirname(os.path.abspath(__file__))


def small():
    for s in ('0.5', '0.3', '0.2'):
        p = os.path.join(B, 'results', f'sur_small_s{s}.json')
        if not os.path.exists(p): continue
        J = json.load(open(p))
        print(f'\n**Small net, σ {s}** — per-layer KL (nats) [parameters]; exact layer = 72 params (40 for layer 1)\n')
        fams = ['LR1', 'LR2', 'LR3', 'LR4', 'LR6', 'LIN8', 'TC16k2', 'TC64k4']
        print('| layer | ' + ' | '.join(fams) + ' |'); print('|---|' + '---|' * len(fams))
        for L in J['layers']:
            cells = []
            for f in fams:
                r = L['rows'].get(f) or (L['rows'].get('LIN4') if f == 'LIN8' else None)
                cells.append('–' if r is None else f"{r['KL']:.2f} [{r['cost']}]")
            print(f"| {L['layer']} | " + ' | '.join(cells) + ' |')
        print('\nDV two-sided bounds (nats-based KL), truth from 2^20 real paths; width / CI = DV width over plain sampling 95% CI at equal passes\n')
        print('| configuration | KL (nats) | params | squared error: truth; DV [lo, hi]; width/CI | output mean: truth; DV [lo, hi]; width/CI |'); print('|---|---|---|---|---|')
        for c in J['configs']:
            a, m = c['sqerr'], c['mean']
            print(f"| {c['config']} | {c['KL']:.3f} | {c['cost']} | {a['truth']:.3f}; [{a['lo']:.3f}, {a['hi']:.3f}]; {a['width_over_ci']:.0f} | {m['truth']:.3f}; [{m['lo']:.3f}, {m['hi']:.3f}]; {m['width_over_ci']:.0f} |")
        if J.get('merges'):
            print('\nSkip-merge (layers l, l+1 as one block, Gaussian kernel with sigma^2 (I + J J^T)):\n')
            print('| block | Gaussian check KL (mean / max, nats) | expansion | exact-merge KL [params] | best merged LIN-r KL [params] | separate LIN-r (sum of two layers) |'); print('|---|---|---|---|---|---|')
            for mg in J['merges']:
                bm = min(((k, v) for k, v in mg['merged'].items() if k != 'exact-merge'), key=lambda kv: kv[1]['KL'])
                sep = ', '.join(f"{k} {v:.2f}" for k, v in mg['separate_LIN'].items())
                print(f"| {mg['layers']} | {mg['gauss_check_KL_mean']:.3f} / {mg['gauss_check_KL_max']:.3f} | {mg['expansion']:.2f} | {mg['merged']['exact-merge']['KL']:.3f} [{mg['merged']['exact-merge']['cost']}] | {bm[0]} {bm[1]['KL']:.2f} [{bm[1]['cost']}] | {sep} |")
            print('\nDP over cut points (objective = KL + beta * params):\n')
            for d in J['dp']:
                print(f"- beta {d['beta']:g}: objective {d['objective']:.2f}; segments {d['segments']}")


def pfn():
    for s in ('0.05', '0.1', '0.183'):
        p = os.path.join(B, 'runs', f'L2_s{s}', 'sur.json')
        if not os.path.exists(p): continue
        J = json.load(open(p))
        print(f'\n**PFN 2L, σ {s}** — per-site KL (nats per episode) [parameters]\n')
        print('| site | surrogates: KL [params] |'); print('|---|---|')
        for site, rows in J['sites'].items():
            print(f"| {site} | " + '; '.join(f"{k} {v['KL']:.1f} [{v['cost']}]" for k, v in rows.items() if k != 'exact') + f" (exact: {rows['exact']['cost']}) |")
        best = sorted(J['configs'], key=lambda c: c['KL'])[:6]
        print('\nDV bounds for the lowest-KL configurations (mean NLL and mean predictive mean):\n')
        print('| configuration | KL | NLL truth; DV [lo, hi]; width/CI | mean truth; DV [lo, hi]; width/CI |'); print('|---|---|---|---|')
        for c in best + [c for c in J['configs'] if c['config'] == 'all-moderate']:
            a, m = c['nll'], c['mean']
            print(f"| {c['config']} | {c['KL']:.1f} | {a['truth']:.3f}; [{a['lo']:.2f}, {a['hi']:.2f}]; {a['width_over_ci']:.0f} | {m['truth']:.3f}; [{m['lo']:.2f}, {m['hi']:.2f}]; {m['width_over_ci']:.0f} |")
        if J.get('merges'):
            for mg in J['merges']:
                bm = min(((k, v) for k, v in mg['merged'].items() if k != 'exact-merge'), key=lambda kv: kv[1]['KL'])
                bs = min(mg['separate'].items(), key=lambda kv: kv[1]['KL'] + 1e-3 * kv[1]['cost'])
                print(f"- MLP block {mg['block']}: Gaussian moment check {mg['gauss_moment_KL_mean_per_token']:.3f} nats/token; exact-merge {mg['merged']['exact-merge']['KL']:.1f} nats [{mg['merged']['exact-merge']['cost']}]; "
                      f"best merged LR {bm[0]} {bm[1]['KL']:.0f} [{bm[1]['cost']}]; separate e.g. {bs[0]} {bs[1]['KL']:.1f} [{bs[1]['cost']}]")


if __name__ == '__main__':
    small(); pfn()
