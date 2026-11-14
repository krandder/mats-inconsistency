"""Loader fix (eighth agent, found during E72): e63_run.py replaces pfn2.build by pfn_own.build_any, whose fallback builds pfn2.PFN
without l2_heads / l2_dh, so a net with a different block-2 head layout (E70's b1h4: 4 heads in block 1, 2 heads of dimension 16 in
block 2) loads with block 2 split into 4 heads of dimension 8 (same shapes, a different function). install() makes pfn2.build build
such nets (meta with l2_heads, no variant, no b1_heads) with pfn2.PFN's own l2_heads / l2_dh arguments, and (second fix, found in
E72's review) E64's variants (nocons, codeln, both) with pfn_e64.build_any, which e63_run's chain does not reach (they loaded as the
plain pfn2 net). Call it after e63_run (and pfn_b1hw.install()) have been imported.
"""
import sys

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402


def install():
    import pfn_e64 as PE
    orig = P2.build

    def build_any(meta):
        if meta.get('variant') in PE.VARIANTS:                 # E64's nocons, codeln, both (masks / code channel), not dispatched by e63_run
            return PE.build_any(meta, P2, P2.fin_of(meta['ds']))
        if meta.get('l2_heads') and not meta.get('variant') and not meta.get('b1_heads'):
            return P2.PFN(P2.fin_of(meta['ds']), meta['depth'], meta['width'], meta['heads'], meta['attn_only'], mixk=meta.get('mixk', 0),
                          l2_heads=meta['l2_heads'], l2_dh=meta['l2_dh'])
        return orig(meta)
    P2.build = build_any
