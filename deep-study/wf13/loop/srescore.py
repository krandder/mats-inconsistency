"""srescore.py -- rerun the Sudoku SCORING for a finished loop (same networks, same seeds, same puzzles) after the fix of
sk.books (PREREG-ADDENDA A1). The loop and its networks are untouched. Usage: srescore.py SEED -> out/s_s{SEED}.json
(the first scoring is kept as out/s_s{SEED}_prefix.json)."""
import json, shutil, sys, time
import numpy as np, torch
sys.dont_write_bytecode = True
import sk, spol, sscore
HERE = sscore.HERE
seed = int(sys.argv[1])
name = 's_s%d' % seed
shutil.copy(HERE + 'out/%s.json' % name, HERE + 'out/%s_prefix.json' % name)
res = json.load(open(HERE + 'out/%s.json' % name))
res.pop('arms', None)
cfg = res['cfg']
sd = torch.load(HERE + 'ckpt/%s.pt' % name, map_location='cpu')
nets = {}
for k, v in sd.items():
    n = spol.SNet(cfg['d'], cfg['L']); n.load_state_dict(v); n.eval()
    nets[k if k == 'oneshot' else int(k)] = n
torch.set_num_threads(2)
rules = sk.Rules(np.load(HERE + 'out/sudoku_rules.npz')['allowed'])
logf = open(HERE + 'logs/%s.log' % name, 'a')


def log(s):
    logf.write('[%s] %s\n' % (time.strftime('%H:%M:%S'), s)); logf.flush()
log('RESCORE after the fix of sk.books (addendum A1)')
res['rescored'] = True
sscore.score_all(seed, cfg, rules, nets, res, name, log)
