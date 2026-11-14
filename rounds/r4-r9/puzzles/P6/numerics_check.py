"""Pre-registered numerics gate: CPU vs GPU (fp32, TF32 off) for R0 and E-clone seed 0 at
(L=128, D=512) and (L=512, D=2048). Pass: exact-match counts within 1 example and nats/bit
within 1e-3 relative. Writes <out>/numerics.json."""
import argparse, json, os, time
import torch
import eval_ladder as E

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(1)
res, ok_all = [], True
for name in ("R0_s0", "EC_s0"):
    for L, D in ((128, 512), (512, 2048)):
        row = dict(model=name, L=L, D=D)
        for dev in ("cpu", "cuda"):
            t0 = time.time()
            arm, M = E.load_model(os.path.join(a.out, "models", name + ".pt"), dev)
            var = "plain" if arm in ("R0", "DTL") else "auction"
            x, y, m = E.test_set(L, E.n_for(L), dev)
            lg, _ = E.logits_at(arm, M, x, y, m, [D], var)
            ok, nats = E.per_example(lg[D], y, m)
            row[dev] = dict(exact=int(ok.sum()), nats=float(nats.mean()), sec=round(time.time() - t0, 1))
        c, g = row["cpu"], row["cuda"]
        row["pass"] = abs(c["exact"] - g["exact"]) <= 1 and abs(c["nats"] - g["nats"]) <= 1e-3 * max(abs(c["nats"]), 1e-12)
        ok_all &= row["pass"]
        res.append(row)
        print(json.dumps(row), flush=True)
json.dump(dict(all_pass=ok_all, rows=res, torch=torch.__version__), open(os.path.join(a.out, "numerics.json"), "w"))
