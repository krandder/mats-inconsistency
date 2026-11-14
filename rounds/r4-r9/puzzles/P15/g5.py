"""P15 gate G5 and the P-pub arm.
  python g5.py ref            our port vs Bansal's own classes on the published weights (logits, fp64)
  python g5.py pubprefix      published prefix net again, inputs centred (x-0.5) as Bansal's loader does
  python g5.py pub            published checkpoints on easy-to-hard 512-bit strings and 59x59 mazes (peak T)
  python g5.py retrain SEED   Bansal's prefix recipe (32 bits, width 400), then 512-bit eval
  python g5.py ours           P-pub (published + retrained prefix; published maze) on our val/test sets
Writes gates/G5_*.json and results/P-pub_*.json.  Data: $P15_DATA (default /root/ext/data)."""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p15  # noqa: E402
import p15_dtrecall as dtr  # noqa: E402

DATA = os.environ.get("P15_DATA", "/root/ext/data")
EXT = os.environ.get("P15_EXT", "/root/ext")
CK_PREFIX = os.path.join(EXT, "p15ckpt/training-roupy-Ambr/model_best.pth")
CK_MAZE = os.path.join(EXT, "p15ckpt/training-rusty-Tayla/model_best.pth")
GD = os.path.join(HERE, "gates")
N_PREFIX_512 = int(os.environ.get("P15_N512", 2000))
N_MAZE_59 = int(os.environ.get("P15_N59", 300))


def save(name, d):
    os.makedirs(GD, exist_ok=True)
    json.dump(d, open(os.path.join(GD, name + ".json"), "w"), indent=1, default=float)
    print(name, json.dumps(d, default=float)[:800], flush=True)


def prefix_file(bits, num):
    X = torch.load(os.path.join(DATA, "prefix_sums_data", f"{bits}_data.pth"), weights_only=False)[:num]
    Y = torch.load(os.path.join(DATA, "prefix_sums_data", f"{bits}_targets.pth"), weights_only=False)[:num]
    X, Y = X.numpy().astype(np.float32), Y.numpy().astype(np.int64)
    return {"x": X[..., None], "y": Y, "valid": np.ones(Y.shape, np.float32)}


def eval_prefix_512(net, Ts):
    d = prefix_file(512, N_PREFIX_512)
    g = dtr.dt_greedy(net.to(p15.DEV).eval(), "prefix", d, Ts)
    ex = np.array(g["exact"][0])
    return {"num": N_PREFIX_512, "Ts": g["Ts"], "exact": g["exact"][0], "peak": float(ex.max()),
            "peak_T": int(g["Ts"][int(ex.argmax())]), "maxlogit": g["maxlogit"][0]}


def cmd_ref():
    out = {"prefix_max_abs_logit_diff": dtr.ref_logit_check(CK_PREFIX, 1, os.path.join(EXT, "p15dt")),
           "maze_max_abs_logit_diff": dtr.ref_logit_check(CK_MAZE, 2, os.path.join(EXT, "p15dt"))}
    out["pass"] = bool(max(out.values()) < 1e-4)
    save("G5_ref", out)


def cmd_pub():
    t0 = time.time()
    net = dtr.load_pub(CK_PREFIX, 1)
    pre = eval_prefix_512(net, list(range(5, 601, 5)) + [237])
    print("prefix 512", pre["peak"], pre["peak_T"], f"{time.time() - t0:.0f}s", flush=True)
    t1 = time.time()
    p = os.path.join(DATA, "maze_data_test_59")
    img = np.load(os.path.join(p, "inputs.npy"), mmap_mode="r")[:N_MAZE_59].astype(np.float32)
    sol = np.load(os.path.join(p, "solutions.npy"), mmap_mode="r")[:N_MAZE_59].astype(np.float32)
    net2 = dtr.load_pub(CK_MAZE, 2).to(p15.DEV).eval()
    Ts = sorted(set(list(range(100, 2001, 100)) + [984]))
    g = dtr.dt_greedy(net2, "maze", {"img": img, "sol": sol}, Ts, pixels=True)
    ex = np.array(g["exact"][0])
    mz = {"num": N_MAZE_59, "Ts": g["Ts"], "exact": g["exact"][0], "peak": float(ex.max()),
          "peak_T": int(g["Ts"][int(ex.argmax())]), "at_984": float(ex[g["Ts"].index(984)]),
          "maxlogit": g["maxlogit"][0], "wall_s": time.time() - t1}
    print("maze 59", mz["peak"], mz["peak_T"], f"{time.time() - t1:.0f}s", flush=True)
    lo = lambda k, n: float(__import__("scipy.stats", fromlist=["beta"]).beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0  # noqa
    out = {"prefix_512": pre, "maze_59": mz,
           "published": {"prefix_512": 0.9712, "maze_59": 0.9730},
           "thresholds": {"prefix_512": 0.921, "maze_59": 0.923},
           "prefix_cp95": [lo(round(pre["peak"] * pre["num"]), pre["num"]), None],
           "maze_cp95_lo": lo(round(mz["peak"] * mz["num"]), mz["num"])}
    out["pass"] = bool(pre["peak"] >= 0.921 and mz["peak"] >= 0.923)
    save("G5_pub", out)


def cmd_pubprefix():
    """Re-run the published prefix net on 512 bits with Bansal's input centring (load_pub sets xshift=0.5).
    The first g5 pub run fed raw 0/1 bits and scored 0.0 (chance); its maze half is unaffected."""
    t0 = time.time()
    pre = eval_prefix_512(dtr.load_pub(CK_PREFIX, 1), list(range(5, 601, 5)) + [237])
    lo = lambda k, n: float(__import__("scipy.stats", fromlist=["beta"]).beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0  # noqa
    pre.update({"published": 0.9712, "threshold": 0.921, "cp95_lo": lo(round(pre["peak"] * pre["num"]), pre["num"]),
                "pass": bool(pre["peak"] >= 0.921), "wall_s": time.time() - t0, "xshift": 0.5})
    save("G5_pub_prefix", pre)


def cmd_retrain(seed, centre=True):
    """Retrain P-pub prefix (centred inputs, as Bansal) or, with centre=False, the uncentred ablation.
    The 512-bit evals are serialised with a file lock: three at once ran out of GPU memory."""
    import fcntl
    seed = int(seed)
    ck = os.path.join(p15.CKDIR, "ppub")
    os.makedirs(ck, exist_ok=True)
    t0 = time.time()
    net = dtr.pub_retrain(seed, os.path.join(DATA, "prefix_sums_data"), ck, centre=centre,
                          log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
    tw = time.time() - t0
    with open("/tmp/p15_g5eval.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        r = eval_prefix_512(net, list(range(5, 601, 5)))
    r["train_wall_s"] = tw
    r["wall_s"] = time.time() - t0
    r["centred"] = centre
    r["pass_92"] = bool(r["peak"] >= 0.92)
    save(f"G5_retrain_s{seed}" if centre else f"G5_retrain_raw_s{seed}", r)


def cmd_ours(parts=("prefix", "maze")):
    """P-pub on our val/test sets with the same c x size grid; c frozen on validation later (analyze.py).
    Prefix elements: the published net plus our centred retrains (ppub_c_s*.pt). The uncentred retrains are
    an ablation and are not P-pub elements. Until all 3 centred retrains exist the prefix file is written as
    P-pub_prefix_D16_pubonly.json, so a VM without them cannot overwrite the full file at sync."""
    if "prefix" in parts:
        cmd_ours_prefix()
    if "maze" in parts:
        cmd_ours_maze()


def cmd_ours_prefix():
    nets = [("pub", dtr.load_pub(CK_PREFIX, 1))]
    for s in range(3):
        f = os.path.join(p15.CKDIR, "ppub", f"ppub_c_s{s}.pt")
        if os.path.exists(f):
            ckd = torch.load(f, map_location="cpu", weights_only=False)
            if ckd["epoch"] < 150:
                continue
            n = dtr.DT(1, 400, 1)
            n.load_state_dict(ckd["net"])
            n.xshift = 0.5
            nets.append((f"retrain_s{s}", n))
    task = p15.Task("prefix", 16)
    res = {"arm": "P-pub", "kind": "prefix", "D": 16, "elements": [n for n, _ in nets], "val": [], "test": []}
    for which in ("val", "test"):
        for size, num in p15.eval_sizes(task, which):
            d = task.eval_set(which, size, num)
            Ts = p15.t_grid(task.size_T(size), p15.CS)
            per = [dtr.dt_greedy(net.to(p15.DEV).eval(), "prefix", d, Ts) for _, net in nets]
            res[which].append({"size": size, "num": num, "Ts": per[0]["Ts"],
                               **{k: [p[k][0] for p in per] for k in ("exact", "logloss", "maxlogit")}})
            print("P-pub prefix", which, size, [np.round(p["exact"][0], 2).tolist() for p in per], flush=True)
    name = "P-pub_prefix_D16.json" if len(nets) == 4 else "P-pub_prefix_D16_pubonly.json"
    json.dump(res, open(os.path.join(p15.RESDIR, name), "w"))


def cmd_ours_maze():
    # mazes: the published maze model (trained on easy-to-hard '9x9' = 9 blocks = 5x5 cells) on our mazes as pixels
    task = p15.Task("maze", 32)
    net = dtr.load_pub(CK_MAZE, 2).to(p15.DEV).eval()
    res = {"arm": "P-pub", "kind": "maze", "D": 32, "elements": ["pub"], "val": [], "test": []}
    for which in ("val", "test"):
        for size, num in p15.eval_sizes(task, which):
            d = task.eval_set(which, size, num)
            img = p15.cells_to_pixels(d["x"], size)
            sol = p15.path_pixels(d["x"], d["y"], size)
            Ts = p15.t_grid(task.size_T(size), p15.CS)
            g = dtr.dt_greedy(net, "maze", {"img": img, "sol": sol, "ycell": d["y"]}, Ts, cap=3000, pixels=True)
            res[which].append({"size": size, "num": num, "Ts": g["Ts"], "exact_pixels": [g["exact"][0]],
                               "exact": [g["exact_cells"][0]], "logloss": [g["logloss"][0]],
                               "maxlogit": [g["maxlogit"][0]]})
            print("P-pub maze", which, size, np.round(g["exact"][0], 2).tolist(),
                  np.round(g["exact_cells"][0], 2).tolist(), flush=True)
    json.dump(res, open(os.path.join(p15.RESDIR, "P-pub_maze_D32.json"), "w"))


if __name__ == "__main__":
    c = sys.argv[1]
    {"ref": cmd_ref, "pub": cmd_pub, "pubprefix": cmd_pubprefix, "ours": cmd_ours,
     "oursprefix": lambda: cmd_ours(("prefix",)), "oursmaze": lambda: cmd_ours(("maze",)),
     "retrainraw": lambda: cmd_retrain(sys.argv[2], centre=False)}.get(c, lambda: cmd_retrain(sys.argv[2]))()
