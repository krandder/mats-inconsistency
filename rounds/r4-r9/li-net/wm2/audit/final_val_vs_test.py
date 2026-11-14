"""wm2 audit, phase 2: the final runs' own VALIDATION numbers next to their TEST numbers (champion minus rbtd8, paired by
seed), and the loop's validation numbers for the same seeds. Shows whether the TEST result at large N is reproduced on
validation by the final runs (run noise / winner's curse) or is TEST-specific.
  PYTHONDONTWRITEBYTECODE=1 python3 audit/final_val_vs_test.py   (only after results/test-t1-final.json exists)"""
import glob, json, math, os
import numpy as np
RES = '/home/agent/workspaces/eurisko-study/li-net/wm2/results'
assert os.path.exists(os.path.join(RES, 'test-t1-final.json')), 'phase 1: TEST closed'
TQ = {3: 4.303, 5: 2.776}
def load(path):
    out = {}
    for p in glob.glob(path):
        r = json.load(open(p)); out[(r['n'], r['seed'])] = r
    return out
F_c, F_b = load('/tmp/wm2/runs/F/t1/F-WMr-D-rbtd8/*.json'), load('/tmp/wm2/runs/F/t1/F-B-rbtd8/*.json')
L_b = load('/tmp/wm2/runs/W1/t1/B-rbtd8/*.json')
L_c = load('/tmp/wm2/runs/W2/t1/WMr-D-rbtd8/*.json'); L_c.update({k: v for k, v in load('/tmp/wm2/runs/RQ2/t1/WMr-D-rbtd8/*.json').items()
                                                              if k in [(3000, 2), (10000, 0), (10000, 1), (10000, 2)]})
def ci(d):
    d = np.asarray(d); return d.mean(), TQ[len(d)] * d.std(ddof=1) / math.sqrt(len(d))
for n in (100, 300, 1000, 3000, 10000):
    s5 = range(5)
    tk = [1e3 * (F_c[(n, s)]['test']['kl'] - F_b[(n, s)]['test']['kl']) for s in s5]
    vk = [1e3 * (F_c[(n, s)]['val']['kl'] - F_b[(n, s)]['val']['kl']) for s in s5]
    vl = [1e4 * (F_c[(n, s)]['val']['logloss'] - F_b[(n, s)]['val']['logloss']) for s in s5]
    lk = [1e3 * (L_c[(n, s)]['val']['kl'] - L_b[(n, s)]['val']['kl']) for s in range(3)]
    print(f"N={n}: final TEST dKL x1e3 {ci(tk)[0]:+.3f} +- {ci(tk)[1]:.3f} {[round(x, 3) for x in tk]} | final runs' VAL dKL {ci(vk)[0]:+.3f} +- {ci(vk)[1]:.3f} "
          f"{[round(x, 3) for x in vk]}, VAL dLL x1e4 {ci(vl)[0]:+.2f} +- {ci(vl)[1]:.2f} | loop VAL dKL (seeds 0-2) {[round(x, 3) for x in lk]}")
