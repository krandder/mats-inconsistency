"""Pick the trained models to analyse (PREREG Part 4): from runs/p4/eval_all.json choose, per (kind, size, sigma), the
best seed (RNN: accuracy with noise on; MLP: nMSE with noise on, finite). For RNN runs cut before the end, the last
checkpoint is used. Prints "KIND PATH" lines."""
import json, os, re, sys

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ev = json.load(open(os.path.join(W, 'runs', 'p4', 'eval_all.json')))
OUTD = os.path.join(W, 'train', 'gpu_out')
groups = {}
for name, r in ev.items():
    if r['kind'] == 'rnn':
        m = re.match(r'rnn_T(\d+)_s(\d+)_sig([\d.]+?)(_ck(\d+))?\.npz', name)
        if not m: continue
        T, seed, sig, ck = int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(5) or 10 ** 9)
        groups.setdefault(('rnn', T, sig), []).append((ck, r['acc_on'], name))
    else:
        m = re.match(r'mlp_(res|plain)_L(\d+)_s(\d+)_sig([\d.]+?)(_ck(\d+))?\.npz', name)
        if not m or m.group(5): continue
        if not (r['nmse_on'] == r['nmse_on']) or r['nmse_on'] > 1e3: continue
        groups.setdefault(('mlp-' + m.group(1), int(m.group(2)), float(m.group(4))), []).append((0, -r['nmse_on'], name))
for key, lst in sorted(groups.items()):
    # latest checkpoint per seed first, then best score
    best = {}
    for ck, score, name in lst:
        seed = re.search(r'_s(\d+)_', name).group(1)
        if seed not in best or ck > best[seed][0]:
            best[seed] = (ck, score, name)
    ck, score, name = max(best.values(), key=lambda t: t[1])
    kind = 'rnn' if key[0] == 'rnn' else 'mlp'
    print(kind, os.path.join(OUTD, name))
