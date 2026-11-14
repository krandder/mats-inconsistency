"""E76 plumbing check (ninth agent): the embedding (inp.*) and block 1 (blocks.0.*) of every registered reader's model.pt and model_final.pt
are bit-identical to its source's model.pt, and the reader parameters (blocks.1.*, ln.*, out.*) differ.
usage: python e76x_frozen.py -> prints one line per reader and a summary
"""
import glob, json, os
import torch

bad, n = 0, 0
for run in sorted(glob.glob('runs/e76/*_s*/')):
    meta = json.load(open(os.path.join(run, 'meta.json')))
    src = torch.load(os.path.join(meta['reader_src'], 'model.pt'), map_location='cpu')
    for wf in ('model.pt', 'model_final.pt'):
        sd = torch.load(os.path.join(run, wf), map_location='cpu'); n += 1
        frozen = [k for k in sd if k.startswith('inp.') or k.startswith('blocks.0.')]
        reader = [k for k in sd if not (k.startswith('inp.') or k.startswith('blocks.0.'))]
        same = all(torch.equal(sd[k], src[k]) for k in frozen)
        moved = sum(not torch.equal(sd[k], src[k]) for k in reader)
        ok = same and (moved > 0 or (meta['reader_mode'] == 'continue' and meta['best_step'] == 0 and wf == 'model.pt'))
        bad += not ok
        print(os.path.basename(run.rstrip('/')), wf, 'frozen identical' if same else 'FROZEN CHANGED', '%d of %d reader tensors differ' % (moved, len(reader)),
              len(frozen), 'frozen tensors')
print('checked', n, 'files;', bad, 'failures')
