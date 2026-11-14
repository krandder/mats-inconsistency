"""E60 per-net evaluation (PREREG-B amendment 68): e48_eval_w1.py's measures (test KL; rom and type; block-2 labelled attention,
xq; the sign-flip cost for signed formats), with the model builder patched so that nets whose meta.json names a pfn_own variant
(own, ownf), a pfn_uni variant, noln1 or pre are built as that variant (baseline nets otherwise).
usage: python e60_eval.py RUNDIR OUTDIR I0 I1
"""
import sys

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_own as PO  # noqa: E402

P2.build = lambda meta: PO.build_any(meta, P2, P2.fin_of(meta['ds']))

if __name__ == '__main__':
    import e48_eval_w1 as EV
    EV.main()
