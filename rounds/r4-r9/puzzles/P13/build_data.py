"""Build P13's frozen eval/dev maze sets (CPU on General, run once).

Sets (lattice codes, check.py layout H = 2n + 6):
  e19   easy-to-hard maze_data_test_19 eval [200:400]            (P1 MAZE_SPLITS 19 eval, first 200)
  e33   easy-to-hard maze_data_test_33 [400:500] + [600:800]     (P1 33 eval, 300 mazes, skipping [500:600] = P0 v33,
                                                                 a dev set; deviation, see PREREG.md)
  e59   easy-to-hard maze_data_test_59 eval [500:700]            (P1 59 eval, first 200)
  d99   P1 generator (make_mazes_upstream, seed0 99000, chunk 10) of 350, dev [0:50]   (selection only)
  e99   same 350, eval [150:350]                                 (P1 99 eval, all 200)
  e201  make_mazes_upstream, 30 mazes, seed 13001                (stretch)
Dev sets come from P0 frozen (v9 = 1000 of the 9x9 val split, m13[0:200] = P1 13 dev, v33, m59 = P1 59 dev).

  python build_data.py /tmp/p13e2h
"""
import hashlib, json, os, random, subprocess, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check import to_lattice, bfs_len  # noqa: E402

E2H = sys.argv[1] if len(sys.argv) > 1 else "/tmp/p13e2h"
os.makedirs(E2H, exist_ok=True)
URL = "https://www.cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2/maze_data_test_{}.tar.gz"


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def get(n):
    d = f"{E2H}/maze_data_test_{n}"
    if not os.path.exists(f"{d}/solutions.npy"):
        subprocess.check_call(["curl", "-sSL", "--retry", "3", "-o", f"{E2H}/t{n}.tgz", URL.format(n)])
        subprocess.check_call(["tar", "-C", E2H, "-xzf", f"{E2H}/t{n}.tgz"])
        os.remove(f"{E2H}/t{n}.tgz")
    return np.load(f"{d}/inputs.npy", mmap_mode="r"), np.load(f"{d}/solutions.npy", mmap_mode="r")


def gen(num, n, seed):
    import contextlib, io
    import make_mazes_upstream as mm
    random.seed(seed); np.random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        inp, _, _, sol = mm.gen_dataset(num, (n + 1) // 2)
    return inp.transpose(0, 3, 1, 2).astype(np.float32), sol[..., 0].astype(np.float32)


sets, meta = {}, {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sets": {}}


def add(name, x, y, src):
    code, sol = to_lattice(x, y)
    code = code.numpy(); sol = sol.numpy()
    sp = bfs_len(code)
    sets[f"{name}/code"] = code; sets[f"{name}/sol"] = np.packbits(sol, axis=-1); sets[f"{name}/sp_len"] = sp
    meta["sets"][name] = {"source": src, "n": int(code.shape[0]), "maze_size": int(code.shape[-1]),
                          "pixels": int(2 * code.shape[-1] + 6),
                          "sha256": {"code": sha(code), "sol": sha(sol), "sp_len": sha(sp)}}
    print(name, code.shape, time.strftime("%H:%M:%S"), flush=True)


for n, sl, name in [(19, [slice(200, 400)], "e19"), (33, [slice(400, 500), slice(600, 800)], "e33"),
                    (59, [slice(500, 700)], "e59")]:
    x, y = get(n)
    xs = np.concatenate([np.asarray(x[s], np.float32) for s in sl]); ys = np.concatenate([np.asarray(y[s], np.float32) for s in sl])
    add(name, xs, ys, f"easy-to-hard-data maze_data_test_{n} {[(s.start, s.stop) for s in sl]}")
    del x, y
# 99: exactly P1's write_generated(root, 99, 350, seed0=99000, chunk=10): chunk k uses seed 99000 + k
t = time.time()
outs = [gen(10, 99, 99000 + k) for k in range(35)]
x99 = np.concatenate([o[0] for o in outs]); y99 = np.concatenate([o[1] for o in outs])
add("d99", x99[:50], y99[:50], "P1 make_mazes_upstream seed0 99000 chunk 10, [0:50] (dev)")
add("e99", x99[150:350], y99[150:350], "P1 make_mazes_upstream seed0 99000 chunk 10, [150:350] (P1 99 eval)")
print("gen99 sec", round(time.time() - t), flush=True)
t = time.time()
x, y = gen(30, 201, 13001)
meta["gen201_cpu_sec"] = round(time.time() - t, 1)
add("e201", x, y, "make_mazes_upstream 30 mazes seed 13001 (stretch)")
path = os.path.join(HERE, "frozen", "p13_eval.npz")
np.savez_compressed(path, **sets)
meta["file_sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
json.dump(meta, open(os.path.join(HERE, "frozen", "manifest.json"), "w"), indent=1)
print(json.dumps(meta)[:2000])
