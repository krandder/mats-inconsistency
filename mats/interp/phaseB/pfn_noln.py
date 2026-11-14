"""E50 architecture variant (PREREG-B amendment 58): pfn2.py's PFN with block 1's pre-attention LayerNorm (blocks[0].ln1)
replaced by the identity, so block 1's queries, keys and values are affine in each token's input embedding (no per-token
normalisation before the first pooling). Everything else, including block 1's MLP LayerNorm (ln2), block 2 and the final
LayerNorm, is unchanged. A LayerNorm's initialisation is the constant (1, 0) and draws no random numbers, so with
torch.manual_seed(seed) every remaining parameter starts exactly as in the baseline net of that seed.
build_any(meta, P2, fin): the variant when meta['variant'] == 'noln1', otherwise pfn_pre.build_any (baseline or 'pre').
"""
import torch.nn as nn
import pfn_pre as PP


def build_any(meta, P2, fin):
    if meta.get('variant') == 'noln1':
        m = P2.PFN(fin, meta['depth'], meta['width'], meta['heads'], mixk=meta.get('mixk', 0))
        m.blocks[0].ln1 = nn.Identity()
        return m
    return PP.build_any(meta, P2, fin)
