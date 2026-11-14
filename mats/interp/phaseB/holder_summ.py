"""Tables for the Hölder / Rényi section (results/holder_*.json)."""
import json, math, os, glob, sys
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
A = [2, 4, 8]


def best_bound(r, a, oracle=False):
    """Best (smallest) log2 bound among out-level and through-kernel for order a, capped at 0."""
    sfx = '_oracle' if oracle else ''
    return min(0.0, r[f'out{sfx}_a{a}'], r[f'tk{sfx}_a{a}'])


def main():
    for f in sorted(glob.glob(os.path.join(R, 'holder_s*_S*.json'))):
        J = json.load(open(f)); print(f"\n### sigma {J['sigma']}, S = {J['S']} (chain + local divergences {J['t_chain']:.0f} s; equal-time sampling n = {J['n_eq']:.1e})\n")
        print('| layer | KL | D_2 | D_4 | D_8 | D_64 | D_inf (search) |'); print('|---|---|---|---|---|---|---|')
        for l in J['local']:
            e = l['est']; print(f"| {l['layer']} | {e['1.0']:.2f} | {e['2.0']:.2f} | {e['4.0']:.2f} | {e['8.0']:.2f} | {e['64.0']:.2f} | {l['dinf_search']:.2f} |")
        print('\n| depth l | composed B(l,2) / B(l,4) / B(l,8) | naive sum of local d(2) / d(4) / d(8) | D_inf chain | direct D_2 / D_4 / D_8 (p_l vs q_l) |')
        print('|---|---|---|---|---|')
        for c, d in zip(J['composed'], J['direct']):
            print(f"| {c['layer']} | {c['B2']:.1f} / {c['B4']:.1f} / {c['B8']:.1f} | {c['naive_sum']['2']:.1f} / {c['naive_sum']['4']:.1f} / {c['naive_sum']['8']:.1f} | "
                  f"{c['Binf']:.1f} | {d['2.0']:.2f} / {d['4.0']:.2f} / {d['8.0']:.2f} |")
        # rare events: fraction of rarity bits kept, per depth and alpha (certified-style and oracle-D)
        print('\nRare events: fraction of true rarity bits certified, best of output-level / through-kernel, per depth (rows) and alpha; '
              'in brackets the same with the directly estimated D_alpha (no composition); last column: sampling at equal time.\n')
        print('| depth | log2 p | a = 2 | a = 4 | a = 8 | D_inf chain | sampling |'); print('|---|---|---|---|---|---|---|')
        for ev in J['events']:
            for r in ev['rows']:
                if r['log2p'] > -10 or int(round(-r['log2p'])) not in (10, 20, 30, 40): continue
                cells = []
                for a in A:
                    b, bo = best_bound(r, a), best_bound(r, a, True)
                    cells.append(f"{b / r['log2p']:.2f} ({bo / r['log2p']:.2f})")
                print(f"| {ev['layer']} | {r['log2p']:.0f} | " + ' | '.join(cells) + f" | {min(0, r['tk_dinf']) / r['log2p']:.2f} | {r['samp_log2'] / r['log2p']:.2f} |")
        print('\nCommon events (p in [0.05, 0.5]): q_l(A)/p and the best Hölder bound / p per alpha (oracle-D in brackets).\n')
        print('| depth | p | q_l(A)/p | a = 2 | a = 4 | a = 8 |'); print('|---|---|---|---|---|---|')
        for ev in J['events']:
            for r in ev['rows']:
                if r['p'] < 0.04: continue
                cells = [f"{2 ** best_bound(r, a) / r['p']:.2f} ({2 ** best_bound(r, a, True) / r['p']:.2f})" for a in A]
                print(f"| {ev['layer']} | {r['p']:.2f} | {r['q_l'] / r['p']:.2f} | " + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
