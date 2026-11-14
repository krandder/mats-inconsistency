"""Extra checks for the explainer page:
1. own/other per-token attention ratio (diag_fa.attention_binding) for the pilot report-token checkpoints, T = 32;
2. model NLL minus history-blind Bayes NLL per T for the four main pilot arms (from diag.json);
3. history value for the curriculum option: N = 2 (anti pair or independent, M = 2) and N = 3 (anti pair, M = 3)."""
import json, math, os, sys
WF = '/home/agent/workspaces/mats/work/world-fa'
sys.path[:0] = [WF, '/home/agent/workspaces/mats/work/worlds']
import numpy as np, torch
import fa_core as C, harness as H
from diag_fa import attention_binding
torch.set_num_threads(2)
npy = lambda p: tuple(x.detach().cpu().numpy().astype(np.float64) for x in p)
root = os.path.join(WF, 'runs/fa-pilot-1028b')
out = {}
for arm in ('anti_report_s0', 'anti_report_s1', 'null_report_s0', 'null_report_s1'):
    model, world, meta = H.load_run(os.path.join(root, arm), 'best.pt')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4242)
        b = C.gen(512, 4, 32, world.c)
    X, M = C.tokens(b, world.c)
    a = attention_binding(model, X, M, 4, 32)
    out[arm] = dict(ratio_by_layer=a['ratio_by_layer'], ratio_by_layer_head=a['ratio_by_layer_head'],
                    history_mass_by_layer=a['history_mass_by_layer'])
    print(arm, 'own/other ratio by layer', np.round(a['ratio_by_layer'], 4).tolist(),
          'min/max over heads', [(round(min(r), 3), round(max(r), 3)) for r in a['ratio_by_layer_head']],
          'history mass', np.round(a['history_mass_by_layer'], 3).tolist())
for arm in ('anti_report_s0', 'anti_report_s1', 'anti_column_s0', 'anti_column_s1'):
    d = json.load(open(os.path.join(root, arm, 'diag.json')))['cells']
    out[arm + '_minus_T0'] = {T: d[T]['nll_model'] - d[T]['nll_bayes_T0'] for T in d}
    print(arm, 'model - bayes_T0:', {T: round(v, 4) for T, v in out[arm + '_minus_T0'].items()})
cur = {}
for name, N, tpl in (('N2 anti-or-independent', 2, [{"blocks": [[2, 0.8, 1]], "marg": 1.0}, {"blocks": [], "marg": 1.0}]),
                     ('N3 anti pair', 3, [{"blocks": [[2, 0.8, 1]], "marg": 1.0}]),
                     ('N4 anti pair', 4, [{"blocks": [[2, 0.8, 1]], "marg": 1.0}])):
    c = C.full_cfg({"prior": "f", "f_templates": tpl})
    M_ = len(C.family(N, C._tkey(c['f_templates']))[1])
    row = {}
    for T in (8, 32):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(99 + T)
            b = C.gen(6000, N, T, c)
        y = b['v'].numpy()
        d = H.nll(npy(C.bayes(b, c, False)), y) - H.nll(npy(C.bayes(b, c, True)), y)
        row[T] = (float(d.mean()), float(d.std() / math.sqrt(len(d))))
    cur[name] = dict(M=M_, hv=row)
    print(name, 'M', M_, {T: f'{m:.3f} +- {s:.3f}' for T, (m, s) in row.items()})
out['curriculum'] = cur
json.dump(out, open('extra_checks.json', 'w'), indent=1)
