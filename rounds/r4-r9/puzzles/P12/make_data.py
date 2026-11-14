"""Generate every P12 maze set once (CPU, nice'd, <= 5 worker processes). Output: data/*.npz + data/genstats.json.
Seeds are fixed per set so the files are reproducible; the training pool is 5 chunks of 20k (seeds 1000..1004)."""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maze

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

SETS = {  # name: (n, p, num, seed, weighted)
    "val_W25": (25, 0.3, 200, 2000, True), "gate_W9": (9, 0.3, 500, 3109, True),
    "test_W9": (9, 0.3, 500, 3009, True), "test_W15": (15, 0.3, 300, 3015, True),
    "test_W25": (25, 0.3, 300, 3025, True), "test_W33": (33, 0.3, 200, 3033, True),
    "test_W49": (49, 0.3, 100, 3049, True), "test_W65": (65, 0.3, 100, 3065, True),
    "shift_p0.1_W33": (33, 0.1, 200, 4001, True), "shift_p0.5_W33": (33, 0.5, 200, 4005, True),
    "doc_U1_9": (9, 0.0, 200, 5009, False), "doc_U1_33": (33, 0.0, 100, 5033, False),
    "doc_U1_65": (65, 0.0, 50, 5065, False), "doc_U2_33": (33, 0.1, 100, 5133, False),
}


def gen(args):
    name, (n, p, num, seed, w) = args
    t = time.time()
    d, meta = maze.make_set(n, p, num, seed, w)
    meta["seconds"] = round(time.time() - t, 1)
    meta["L_mean"] = float(d["L"].mean()); meta["L_p50"] = float(np.percentile(d["L"], 50))
    meta["L_p99"] = float(np.percentile(d["L"], 99)); meta["L_max"] = int(d["L"].max())
    meta["open_edges_per_cell"] = float(d["open"].sum() / (2 * num * n * n))
    return name, d, meta


def main(which):
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    if which in ("all", "pool"):
        jobs += [(f"poolchunk_W9_{i}", (9, 0.3, 20000, 1000 + i, True)) for i in range(5)]
        jobs += [(f"poolchunk_U1_9_{i}", (9, 0.0, 10000, 1100 + i, False)) for i in range(5)]
    if which in ("all", "eval"):
        jobs += list(SETS.items())
    stats = {}
    sp = os.path.join(OUT, "genstats.json")
    if os.path.exists(sp):
        stats = json.load(open(sp))
    with Pool(5) as pool:
        for name, d, meta in pool.imap_unordered(gen, jobs):
            maze.save(os.path.join(OUT, name + ".npz"), d, meta); stats[name] = meta
            print(name, meta, flush=True)
    for fam, k in (("W9", 5), ("U1_9", 5)):
        chunks = [os.path.join(OUT, f"poolchunk_{fam}_{i}.npz") for i in range(k)]
        if all(os.path.exists(c) for c in chunks):
            ds = [maze.load(c) for c in chunks]
            d = {key: np.concatenate([x[key] for x in ds]) for key in ds[0]}
            maze.save(os.path.join(OUT, f"pool_{fam}.npz"), d)
            stats[f"pool_{fam}"] = {"num": int(len(d["L"])), "L_mean": float(d["L"].mean()),
                                    "L_p99": float(np.percentile(d["L"], 99)), "L_max": int(d["L"].max()),
                                    "tie_rate": float(np.mean([stats[f"poolchunk_{fam}_{i}"]["tie_rate"] for i in range(k)])),
                                    "SG_tries": float(np.mean([stats[f"poolchunk_{fam}_{i}"]["SG_placement_tries_per_accept"] for i in range(k)]))}
            for c in chunks:
                os.remove(c)   # chunk files are intermediates we just wrote (merged into pool_*.npz)
    json.dump(stats, open(sp, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
