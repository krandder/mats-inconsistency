"""E63 plumbing (PREREG-B amendment 71; disclosed, not results): builds synthetic cohort files for e63_summ.py from the explored nets'
e63_run.py output (/tmp/e63_plumb/plumb.json, E43 newinit seeds 0, 1, 2, 4, 11, which the registered tests exclude): ctrl, fix and ext
are copies of those five nets (ext tags renamed to E43-style tags of distinct clusters), own copies them with the summary-head fields
removed and both heads as pooling heads, traj copies them at two steps. Then e63_summ.py must run end to end on /tmp/e63_plumb.
usage: python e63_plumb.py
"""
import copy, json

P = '/tmp/e63_plumb'


def main():
    base = json.load(open(P + '/plumb.json'))
    json.dump(base, open(P + '/ctrl.json', 'w')); json.dump(base, open(P + '/fix.json', 'w'))
    ext = {}
    for i, (t, o) in enumerate(base.items()):
        for j in range(3):
            ext['e43_br%dk_s%d' % (j + 2, 20 + i)] = o
    json.dump(ext, open(P + '/ext.json', 'w'))
    own = {}
    for t, o in base.items():
        x = copy.deepcopy(o)
        for k in ('P_h', 'self_h', 'osum_h', 'olab_h', 'C_h', 'C_max', 'consensus', 'n_pool', 'leak_mean', 'leak_q90', 'leak_q99', 'leak_spearman_maxcos'):
            x.pop(k, None)
        x['pool_heads'] = [0, 1]; x['variant'] = 'own'
        if len(x['L_own_h']) == 2 and 'Kodd_mean_h' in x:
            pass
        own[t.replace('e43_newinit', 'e60_own')] = x
    json.dump(own, open(P + '/own.json', 'w'))
    traj = {}
    for t, o in base.items():
        for st in (0, 90000):
            x = {k: o[k] for k in ('C_max', 'n_pool', 'type')}; x['step'] = st; traj['%s@%d' % (t, st)] = x
    json.dump(traj, open(P + '/traj.json', 'w'))
    print('ok')


if __name__ == '__main__':
    main()
