"""Summarize explore_shape logs: python scratch/sum_shape.py LOG"""
import sys, json
for line in open(sys.argv[1]):
    p = line.split(' ', 1)
    if len(p) < 2 or not p[0].isdigit():
        print(line.strip()); continue
    l = int(p[0]); d = json.loads(p[1].rsplit('}', 1)[0] + '}')
    if l == 1:
        continue
    print(l, 'spread %.3f x l %.2f pr %.1f top %s lrho sd %.3f sk %.2f ku %.2f | pcsk %s pcku %s | sq2 %.3f/%.3f | '
          'lr-abs %s lr-lin %s | nb_t %d sk %.2f ku %.2f all %.2f/%.2f | nb_z %d sk %.2f ku %.2f all %.2f/%.2f' % (
              d['spread'], d['spread_x_l'], d['pr'], [round(x, 3) for x in d['top_share'][:3]], d['logrho_sd'],
              d['logrho_skew'], d['logrho_kurt'], d['pc_skew'][:4], d['pc_kurt'][:4], d['sqcorr_top5_med'],
              d['sqcorr_top5_max'], [round(x, 2) for x in d['corr_logrho_abs']], [round(x, 2) for x in d['corr_logrho_lin']],
              d['nb_count_t'], d['t_nb_abs_skew_med'], d['t_nb_kurt_med'], d['t_all_abs_skew_med'], d['t_all_kurt_med'],
              d['nb_count_z'], d['z_nb_abs_skew_med'], d['z_nb_kurt_med'], d['z_all_abs_skew_med'], d['z_all_kurt_med']))
