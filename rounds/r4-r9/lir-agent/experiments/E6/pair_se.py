"""Post-hoc (written after reading results.json): pair-level standard error of SSI at step 200 per run,
same pair definition as analyze.py. Writes pair_se.json."""
import json, math, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
import analyze as A
out = {}
for X in ("c", "d"):
    for s in (0, 1, 2):
        ex = A.load(os.path.join(A.RUNS, f"{X}-s{s}", "e6eval.jsonl"))
        sp = np.array(ex[200]["support_logp"])
        d = sp[A.t_I] - sp[A.t_J]; m = np.abs(d) > A.LOG2
        a = np.where(d[m] > 0, A.t_I[m], A.t_J[m]); b = np.where(d[m] > 0, A.t_J[m], A.t_I[m])
        for name, path in ((X, f"{X}-s{s}"), (f"b_{X}", f"b{X}-s{s}")):
            ev = A.load(os.path.join(A.RUNS, path, "e6eval.jsonl"))
            s0, st = np.array(ev[0]["score"]), np.array(ev[200]["score"])
            ds = (st[a] - st[b]) - (s0[a] - s0[b])
            out[f"{name}-s{s}"] = {"ssi": float(ds.mean()), "pair_se": float(ds.std(ddof=1) / math.sqrt(len(ds))), "n": int(len(ds))}
json.dump(out, open(os.path.join(HERE, "pair_se.json"), "w"), indent=1)
for k, v in out.items():
    print(k, "%.4f +- %.4f (n=%d)" % (v["ssi"], v["pair_se"], v["n"]))
