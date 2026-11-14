"""E71 (PREREG-B amendment 79): pfn2's baseline net (width 32, depth 2, mixture head) with H1 = meta['b1_heads'] heads of dimension
DH1 = meta['b1_dh'] in block 1; block 2 keeps pfn2's 2 heads of dimension 16 (meta['heads'] = 2, so the config name that seeds the
episode stream is ctrl's).
  build(meta)                  the architecture (pfn2.PFN with block 1 replaced by pfn2.Block(32, H1, dh=DH1));
  paired_init(ref, model, s)   copies the 2-head ctrl initialisation ref of seed s into model:
    - every parameter with the same name and shape (embedding, LayerNorms, MLPs, block 2, output head) is copied;
    - block 1's qkv map: for each of q, k, v, the rows of head h < 2, dimension i < DH1 are ctrl's rows of head h, dimension i
      (weights and biases; ctrl's head dimension is 16, so DH1 = 8 keeps the first 8 dimensions of each ctrl head);
    - every head h >= 2 is drawn fresh from torch.Generator seeded 71000 + 16 s + h, in a fixed order (q, k, v weights; q, k, v
      biases; output-map columns), U(-1, 1) times the PyTorch default bound (1/sqrt(32) for the qkv map, 1/sqrt(H1 DH1) for the
      output map), so a new head's draws do not depend on H1;
    - block 1's output map has fan-in H1 DH1: the columns of the kept ctrl dimensions are ctrl's columns, and the output bias is
      ctrl's, both rescaled by sqrt(32 / (H1 DH1)) so that they follow the default bound 1/sqrt(H1 DH1).
  Nesting: h2w8's heads are E70's b1h4 heads 0 and 2 in q, k and v (b1h4 split each ctrl head into two of dimension 8); h3w16 is
  ctrl plus head 2; h4w16 is h3w16 plus head 3 (the output-map scale aside).
  install()                    makes pfn2.build build these nets for any meta with 'b1_heads' (for loading at evaluation); call it
                               after modules that patch pfn2.build themselves (e.g. e63_run) have been imported.
"""
import math, sys

import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402

CTRL_H, CTRL_DH, D = 2, 16, 32


def build(meta):
    assert meta['width'] == D and meta['depth'] == 2 and meta['heads'] == CTRL_H and not meta.get('l2_heads')
    m = P2.PFN(P2.fin_of(meta['ds']), meta['depth'], meta['width'], meta['heads'], meta['attn_only'], mixk=meta.get('mixk', 0))
    m.blocks[0] = P2.Block(meta['width'], meta['b1_heads'], 2, meta['attn_only'], meta['b1_dh'])
    return m


def paired_init(ref, model, seed):
    blk = model.blocks[0]; H1, DH1 = blk.h, blk.dh
    assert DH1 <= CTRL_DH and ref.blocks[0].h == CTRL_H and ref.blocks[0].dh == CTRL_DH
    rs, ns = ref.state_dict(), model.state_dict()
    new = {}
    for k, v in ns.items():
        if k in ('blocks.0.qkv.weight', 'blocks.0.qkv.bias', 'blocks.0.o.weight', 'blocks.0.o.bias'):
            continue
        assert rs[k].shape == v.shape, k
        new[k] = rs[k].clone()
    Wr, br = rs['blocks.0.qkv.weight'], rs['blocks.0.qkv.bias']             # (3 * 32, 32), (96,)
    Or, obr = rs['blocks.0.o.weight'], rs['blocks.0.o.bias']                 # (32, 32), (32,)
    W = torch.empty(3 * H1 * DH1, D); b = torch.empty(3 * H1 * DH1); O = torch.empty(D, H1 * DH1)
    fan_o = H1 * DH1; bq = 1.0 / math.sqrt(D); bo = 1.0 / math.sqrt(fan_o); sc = math.sqrt(D / fan_o)
    for h in range(H1):
        cols = slice(h * DH1, (h + 1) * DH1)
        if h < CTRL_H:
            for part in range(3):                                            # q, k, v sections
                src = slice(part * CTRL_H * CTRL_DH + h * CTRL_DH, part * CTRL_H * CTRL_DH + h * CTRL_DH + DH1)
                dst = slice(part * H1 * DH1 + h * DH1, part * H1 * DH1 + (h + 1) * DH1)
                W[dst] = Wr[src]; b[dst] = br[src]
            O[:, cols] = Or[:, h * CTRL_DH:h * CTRL_DH + DH1] * sc
        else:
            g = torch.Generator().manual_seed(71000 + 16 * seed + h)
            ws = [(torch.rand(DH1, D, generator=g) * 2 - 1) * bq for _ in range(3)]
            bs = [(torch.rand(DH1, generator=g) * 2 - 1) * bq for _ in range(3)]
            for part in range(3):
                dst = slice(part * H1 * DH1 + h * DH1, part * H1 * DH1 + (h + 1) * DH1)
                W[dst] = ws[part]; b[dst] = bs[part]
            O[:, cols] = (torch.rand(D, DH1, generator=g) * 2 - 1) * bo
    new['blocks.0.qkv.weight'] = W; new['blocks.0.qkv.bias'] = b
    new['blocks.0.o.weight'] = O; new['blocks.0.o.bias'] = obr.clone() * sc
    model.load_state_dict(new)
    return model


def install():
    orig = P2.build

    def build_any(meta):
        return build(meta) if meta.get('b1_heads') else orig(meta)
    P2.build = build_any
