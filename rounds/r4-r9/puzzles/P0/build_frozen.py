"""Build the frozen P0 data (run once on General, CPU).

Mazes are stored at lattice level (uint8 codes 0 wall / 1 open / 2 start / 3 goal and
0/1 solution bits); check.expand() rebuilds the exact pixel tensors (the round trip is
asserted for every maze). Prefix sets are stored as uint8 tokens (0, 1, PAD=2).
Writes frozen/*.npz and frozen/manifest.json (sha256 over each array's C-order bytes).

  python build_frozen.py E2H_DIR
"""
import hashlib, json, math, os, sys, time
import numpy as np
import torch
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "../../market-resnet"))
from check import to_lattice, bfs_len, deadend_rounds  # noqa: E402
from mresnet import prefix_batch, PAD  # noqa: E402  (read-only import)

E2H = sys.argv[1] if len(sys.argv) > 1 else "/tmp/p0src/data"
OUT = os.path.join(HERE, "frozen")
os.makedirs(OUT, exist_ok=True)
manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}, "sets": {}}


def sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def maze_entry(code, sol, source, extra):
    d = {"code": code.numpy() if torch.is_tensor(code) else code,
         "sol": np.packbits(np.asarray(sol, np.uint8), axis=-1),
         "sp_len": bfs_len(code), "deadend_rounds": deadend_rounds(code)}
    d["depth_iters"] = np.ceil(2 * d["deadend_rounds"] / 5).astype(np.int32)
    n = d["code"].shape[-1]
    meta = {"kind": "maze", "source": source, "n": int(d["code"].shape[0]), "maze_size": int(n),
            "pixels": 2 * n + 6, **extra,
            "sha256": {k: sha(v) for k, v in d.items()}, "shape": {k: list(v.shape) for k, v in d.items()}}
    return d, meta


def save(fname, sets):
    arrs = {}
    for name, (d, meta) in sets.items():
        for k, v in d.items():
            arrs[f"{name}/{k}"] = v
        meta["file"] = fname
        manifest["sets"][name] = meta
    path = os.path.join(OUT, fname)
    np.savez_compressed(path, **arrs)
    manifest["files"][fname] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(fname, os.path.getsize(path) // 1024, "KB", flush=True)


def e2h(size, sl):
    x = np.load(f"{E2H}/maze_data_test_{size}/inputs.npy", mmap_mode="r")
    y = np.load(f"{E2H}/maze_data_test_{size}/solutions.npy", mmap_mode="r")
    code, sol = to_lattice(np.array(x[sl]), np.array(y[sl]))
    return code, sol


# ------------------------------------------------------------ easy-to-hard mazes
sets = {}
for name, size, a, b, tcap in [("m9", 9, 0, 1000, 300), ("m13", 13, 0, 1000, 300),
                               ("m33", 33, 0, 500, 1000), ("v33", 33, 500, 600, 1000), ("m59", 59, 0, 100, 3000)]:
    t = time.time()
    code, sol = e2h(size, slice(a, b))
    sets[name] = maze_entry(code, sol, f"easy-to-hard-data maze_data_test_{size}",
                            {"slice": [a, b], "T_cap": tcap, "perfect": True})
    print(name, round(time.time() - t, 1), "s", flush=True)
# training data: 50k 9x9, deep-thinking split random_split([40000, 10000], Generator().manual_seed(42))
x = np.load(f"{E2H}/maze_data_train_9/inputs.npy")
y = np.load(f"{E2H}/maze_data_train_9/solutions.npy")
code, sol = to_lattice(x, y)
del x, y
perm = torch.randperm(50000, generator=torch.Generator().manual_seed(42)).numpy()
train_idx, val_idx = perm[:40000].astype(np.int32), perm[40000:].astype(np.int32)
# check against torch.utils.data.random_split itself
rs = torch.utils.data.random_split(range(50000), [40000, 10000], generator=torch.Generator().manual_seed(42))
assert list(rs[0].indices) == train_idx.tolist() and list(rs[1].indices) == val_idx.tolist()
d, meta = maze_entry(code[val_idx[:1000]], sol[val_idx[:1000]], "easy-to-hard-data maze_data_train_9",
                     {"slice": "val split indices [0:1000]", "T_cap": 300, "perfect": True})
sets["v9"] = (d, meta)
save("mazes_e2h.npz", sets)
tr = {"code": code.numpy(), "sol": np.packbits(sol.numpy(), axis=-1), "train_idx": train_idx, "val_idx": val_idx}
save("maze_train9.npz", {"train9": (tr, {"kind": "maze_train", "source": "easy-to-hard-data maze_data_train_9",
                                         "n": 50000, "maze_size": 9, "split": "random_split 40000/10000 seed 42",
                                         "sha256": {k: sha(v) for k, v in tr.items()}})})
del code, sol, tr

# ------------------------------------------------------------ prefix sums
psets = {}


def pset(name, n, lmin, lmax, base, tcap, note=""):
    seed = base + 1000 * lmin + lmax
    x, y, m = prefix_batch(n, lmin, lmax, torch.Generator().manual_seed(seed))
    L = m.sum(1).long().numpy().astype(np.int32)
    d = {"x": x.numpy().astype(np.uint8), "L": L,
         "depth_dt": np.ceil(L / 5).astype(np.int32), "depth_mr": np.ceil(L / 2).astype(np.int32)}
    psets[name] = (d, {"kind": "prefix", "source": "mresnet.prefix_batch" + note, "seed": seed, "n": n,
                       "lmin": lmin, "lmax": lmax, "T_cap": tcap,
                       "sha256": {k: sha(v) for k, v in d.items()}})


pset("fL128", 500, 128, 128, 5_000_000, 1000)
pset("fL256", 500, 256, 256, 5_000_000, 1000)
pset("fL512", 500, 512, 512, 5_000_000, 3000)
pset("v_id", 1000, 8, 32, 6_000_000, 300)
pset("vL128", 200, 128, 128, 6_000_000, 1000)
pset("vL512", 200, 512, 512, 6_000_000, 3000)
EVALSETS = {"id_8_32": (8, 32, 2000, 64), "L32": (32, 32, 1000, 64), "L48": (48, 48, 1000, 96),
            "mix_8_64": (8, 64, 2000, 128), "L64": (64, 64, 1000, 128), "L128": (128, 128, 500, 256)}
for k, (lmin, lmax, n, tc) in EVALSETS.items():
    pset("legacy_" + k, n, lmin, lmax, 777, tc, " (market-resnet evaluate.py EVALSETS)")
xd = torch.load(f"{E2H}/prefix_sums_data/512_data.pth")[:200].long()
xt = torch.load(f"{E2H}/prefix_sums_data/512_targets.pth")[:200].long()
assert torch.equal(torch.cumsum(xd, 1) % 2, xt)
L = np.full(200, 512, np.int32)
d = {"x": xd.numpy().astype(np.uint8), "L": L, "depth_dt": np.ceil(L / 5).astype(np.int32),
     "depth_mr": np.ceil(L / 2).astype(np.int32)}
psets["e2h512"] = (d, {"kind": "prefix", "source": "easy-to-hard-data prefix_sums_data/512_data.pth", "slice": [0, 200],
                       "n": 200, "lmin": 512, "lmax": 512, "T_cap": 3000,
                       "sha256": {k: sha(v) for k, v in d.items()}})
save("prefix.npz", psets)

# ------------------------------------------------------------ maze-dataset shift grid and stretch (Knutson get_mazes)
if os.environ.get("P0_SKIP_MD") != "1":
    sys.path.insert(0, "/tmp/p0src")
    from gen_md import get_mazes
    shift = {}
    for n in (9, 19, 33):
        for p in (0.0, 0.01, 0.05):
            for de in (True, False):
                t = time.time()
                name = f"sh_n{n}_p{p:g}_de{int(de)}"
                try:
                    xi, so = get_mazes(n, 100, p, de)
                except Exception as e:  # e.g. percolated 9x9 maze with no dead end for a dead-end start
                    manifest.setdefault("skipped_sets", {})[name] = f"{type(e).__name__}: {str(e)[:120]}"
                    print(name, "SKIPPED", type(e).__name__, flush=True)
                    continue
                code, sol = to_lattice(xi, so)
                shift[name] = maze_entry(code, sol, "maze-dataset 1.4.2 via Knutson get_mazes(dfs_perc)",
                                         {"grid_n": (n + 1) // 2, "p": p, "deadend_start": de, "seed": 42,
                                          "T_cap": 300, "perfect": p == 0.0})
                print(name, round(time.time() - t, 1), "s", flush=True)
    save("mazes_shift.npz", shift)
    t = time.time()
    xi, so = get_mazes(201, 20, 0.0, True)
    code, sol = to_lattice(xi, so)
    save("maze_stretch.npz", {"m201": maze_entry(code, sol, "maze-dataset 1.4.2 via Knutson get_mazes(dfs_perc)",
                                                  {"grid_n": 101, "p": 0.0, "deadend_start": True, "seed": 42,
                                                   "T_cap": 3000, "perfect": True})})
    print("m201", round(time.time() - t, 1), "s")

manifest["params"] = {"dt2d_w64": 208656, "dt2d_w128": 783504, "dt1d_w90_in3": 159840, "mresnet_w128": 158209}
old = {}
mp = os.path.join(OUT, "manifest.json")
if os.path.exists(mp):
    old = json.load(open(mp))
    for k in ("files", "sets"):
        for kk, vv in old.get(k, {}).items():
            manifest[k].setdefault(kk, vv)
json.dump(manifest, open(mp, "w"), indent=1)
print("manifest", len(manifest["sets"]), "sets")
