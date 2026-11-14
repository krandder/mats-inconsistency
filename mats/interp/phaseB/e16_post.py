"""E16 post-hoc descriptives (written after seeing the registered E16 results): saved so each reported number has a source.
usage: python e16_post.py -> results/e16_post.json
"""
import json
import numpy as np
from scipy.stats import spearmanr
import e16_summ as S


def main():
    out = {}
    for s in (5, 6):
        L = S.series(s)
        f = lambda c, k=3: S.first(L, c, k)
        o = dict(
            t_slope06=f(lambda d: d['slope'] >= 0.6), t_spec=f(lambda d: d['spec_large'] > 0.05), t_gap010=f(lambda d: d['gap'] < 0.10),
            t_gap015=f(lambda d: d['gap'] < 0.15),
            middle_first_sustained_negative=f(lambda d: d['rom_middle'] < 0 and d['step'] > 5000, 5),
            code_r2_first_08=f(lambda d: d['code_r2'] >= 0.8),
            code_r2_final=[d['code_r2'] for d in L][-1], code_r2_max=float(max(d['code_r2'] for d in L)),
        )
        idx = [i for i, d in enumerate(L) if d['step'] == o['t_spec']][0]
        o['around_onset'] = [dict(step=d['step'], gap=d['gap'], slope=d['slope'], rom_large=d['rom_large'], spec_large=d['spec_large'],
                                  rom_middle=d['rom_middle'], tot_large=d['tot_large']) for d in L[max(0, idx - 3): idx + 3]]
        post = [d for d in L if d['step'] >= o['t_spec']]
        o['spearman_gsph_step_post'] = float(spearmanr([d['step'] for d in post], [d['gsph_large'] for d in post])[0])
        o['gsph_large_at_onset_and_final'] = [post[0]['gsph_large'], post[-1]['gsph_large']]
        o['rom_large_at_onset_peak_final'] = [post[0]['rom_large'], float(max(d['rom_large'] for d in post)), post[-1]['rom_large']]
        o['rom_middle_at_onset_peak_final'] = [post[0]['rom_middle'], float(max(d['rom_middle'] for d in post)), post[-1]['rom_middle']]
        # constant-lr phase only (onset .. 80,000): the trend without the cooldown
        cl = [d for d in post if d['step'] <= 80000]
        o['spearman_rom_large_step_constant_lr'] = float(spearmanr([d['step'] for d in cl], [d['rom_large'] for d in cl])[0])
        out[s] = o
    json.dump(out, open('results/e16_post.json', 'w'), indent=1, default=float)
    for s, o in out.items():
        print(s, {k: v for k, v in o.items() if k != 'around_onset'})
        for d in o['around_onset']:
            print('   ', {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()})


if __name__ == '__main__':
    main()
