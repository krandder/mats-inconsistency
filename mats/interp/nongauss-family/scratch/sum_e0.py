"""Summarize an e0 JSON: python scratch/sum_e0.py runs/.../e0_*.json"""
import sys, json
for f in sys.argv[1:]:
    d = json.load(open(f))
    print(f, 'n', d['n'], 'L', d['L'], 'N', d['N'])
    print('  l  spread  xl   PR   top3              sdlogr  pcskew(1-4)          pckurt(1-4)            sq2 med/max  nb_t  t_nb sk/ku  t_all sk/ku  nb_z  z_nb sk/ku  z_all sk/ku  R2 k=2/4/8/16')
    for l, r in sorted(d['layers'].items(), key=lambda x: int(x[0])):
        r2 = r.get('r2_topk', {})
        print(f"{int(l):3d} {r['spread']:.3f} {r['spread_x_l']:5.2f} {r['pr']:5.1f} {' '.join(f'{x:.2f}' for x in r['top_share'][:3])}  "
              f"{r['logrho_sd']:.3f}  {' '.join(f'{x:+.2f}' for x in r['pc_skew'][:4])}  {' '.join(f'{x:+.2f}' for x in r['pc_kurt'][:4])}  "
              f"{r['sqcorr_top5_med']:.3f}/{r['sqcorr_top5_max']:.3f}  {r['nb_count_t']:3d}  "
              f"{r['t_nb_abs_skew_med'] or 0:.2f}/{r['t_nb_kurt_med'] or 0:+.2f}  {r['t_all_abs_skew_med']:.2f}/{r['t_all_kurt_med']:+.2f}  "
              f"{r['nb_count_z']:3d}  {r['z_nb_abs_skew_med'] or 0:.2f}/{r['z_nb_kurt_med'] or 0:+.2f}  {r['z_all_abs_skew_med']:.2f}/{r['z_all_kurt_med']:+.2f}  "
              + '/'.join(f'{r2[k]:.2f}' for k in sorted(r2, key=int)))
