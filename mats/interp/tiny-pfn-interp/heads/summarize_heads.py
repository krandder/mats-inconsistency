"""Amendment 5 tables: KL_sys per arm, decision rules (i)-(iv)."""
import json, os, sys
import numpy as np
from scipy.stats import ttest_ind

OUT = sys.argv[1] if len(sys.argv) > 1 else 'heads/hout'
ARMS = ['base', 'h3', 'h4', 'w64', 'L3']
main_sys = np.stack([np.load(f'out/budget_main_s{s}_perep.npz')['N_idprotpy'] for s in range(5)]).mean(0)
rows = {}
print('| arm | params | seeds kept (excluded: gap) | test NLL gap | KL raw | KL_sys | nuisance share | sys: mean / var / shape (nats) | corr of per-episode KL_sys with main seeds | R vs base |')
print('|---|---|---|---|---|---|---|---|---|---|')
ksys = {}
for a in ARMS:
    R = []
    for s in range(10, 15):
        f = f'{OUT}/{a}_s{s}.json'
        if os.path.exists(f):
            r = json.load(open(f)); r['perep'] = np.load(f.replace('.json', '_perep.npz')); r['seed'] = s; R.append(r)
    kept = [r for r in R if r['nll_gap_test'] <= 0.05]; excl = [f"s{r['seed']}: {r['nll_gap_test']:.3f}" for r in R if r['nll_gap_test'] > 0.05]
    if not kept:
        print(f'| {a} | - | 0 ({", ".join(excl)}) | | | | | | | |'); continue
    ks = np.array([r['all']['kl'] for r in kept]); ksys[a] = ks
    kr = np.array([r['none']['kl'] for r in kept])
    mp = np.array([r['all']['kl'] - r['all']['kl_meanfix'] for r in kept]); vp = np.array([r['all']['kl_meanfix'] - r['all']['kl_mvfix'] for r in kept]); sp = np.array([r['all']['kl_mvfix'] for r in kept])
    cc = np.mean([np.corrcoef(r['perep']['all'], main_sys)[0, 1] for r in kept])
    rows[a] = ks.mean()
    Rv = 1 - ks.mean() / ksys['base'].mean() if 'base' in ksys else float('nan')
    print(f"| {a} | {kept[0]['params']} | {len(kept)} ({', '.join(excl) or 'none'}) | {np.mean([r['nll_gap_test'] for r in kept]):.4f} | {kr.mean():.4f} ± {kr.std(ddof=1) / np.sqrt(len(kr)) if len(kr) > 1 else 0:.4f} | "
          f"{ks.mean():.4f} ± {ks.std(ddof=1) / np.sqrt(len(ks)) if len(ks) > 1 else 0:.4f} | {1 - ks.mean() / kr.mean():.0%} | {mp.mean():.4f} / {vp.mean():.4f} / {sp.mean():.4f} | {cc:.2f} | {Rv:+.2f} |")
    print(f"|  | per seed KL_sys | {', '.join(f'{x:.4f}' for x in ks)} | | | | | | | |")
Rr = {a: 1 - ksys[a].mean() / ksys['base'].mean() for a in ksys}
p = ttest_ind(ksys['h4'], ksys['base'], equal_var=False, alternative='less').pvalue if 'h4' in ksys and len(ksys['h4']) > 1 else float('nan')
print(f"\nR: {', '.join(f'{a} {v:+.2f}' for a, v in Rr.items())}; Welch one-sided p (h4 < base) = {p:.3f}")
h4, h3, w, l3 = Rr.get('h4', np.nan), Rr.get('h3', np.nan), Rr.get('w64', np.nan), Rr.get('L3', np.nan)
i = h4 >= 0.30 and h3 >= 0.15 and h4 >= h3 and p < 0.05
print('(i) two-head bottleneck supported:', i)
print('(ii) head count, not capacity:', i and w < 0.5 * h4 and l3 < 0.5 * h4)
print('(iii) capacity, not head count:', h4 >= 0.15 and (w >= 0.75 * h4 or l3 >= 0.75 * h4))
print('(iv) head count not the limit (R_h4 < 0.15):', h4 < 0.15, '; then capacity reductions w64 / L3:', f'{w:+.2f} / {l3:+.2f}')
print(f"Reference: CPU-trained main seeds 0-4 KL_sys mean {np.mean([json.load(open(f'out/budget_main_s{s}.json'))['kl']['N:id+rot+y']['kl'] for s in range(5)]):.4f}")
