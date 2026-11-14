"""E64 per-net evaluation (PREREG-B amendment 72): e48_eval_w1.py's measures (test KL; rom and type; block-2 labelled attention, xq; the
sign-flip cost), with the model builder patched so that nets whose meta.json names a pfn_e64 variant (nocons, codeln, both) are built
as that variant (pfn_own / pfn_uni variants and baseline nets otherwise).
usage: python e64_eval.py RUNDIR OUTDIR I0 I1
"""
import sys

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_e64 as P64  # noqa: E402

P2.build = lambda meta: P64.build_any(meta, P2, P2.fin_of(meta['ds']))

if __name__ == '__main__':
    import e48_eval_w1 as EV
    EV.main()
