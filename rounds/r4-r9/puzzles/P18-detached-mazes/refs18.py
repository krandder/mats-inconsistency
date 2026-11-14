"""P18 references outside equal compute (PREREG section 3; reported, not tested).

  python refs18.py order     verify that CostNet+BB's per-maze arrays (followups-batch2 item 1, e2h_maze_33/59 = the
                             first 1,000 raw official test mazes, in file order) line up with P13's e33/e59 slices:
                             stream the official tarballs (no extraction to disk), rebuild the lattice codes of the
                             e33/e59 slices and compare their sha256 with frozen/manifest.json.
                             Writes results/costnet_order_check.json.
  python refs18.py collect   gather M-ref (this experiment's ev_mref), CostNet+BB on the exact e33/e59 mazes (only if
                             the order check passed), P13 DTR-eq and dt-baseline numbers into results/references.json.
Reads other experiments' directories; writes only inside P18.
"""
import hashlib, io, json, os, sys, tarfile
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
PZ = os.path.dirname(D)
CN = os.path.join(PZ, "followups-batch2", "item1-costnet")
RAW = os.path.join(CN, "code", "bansal_data")
SL = {"e33": (33, [(400, 500), (600, 800)]), "e59": (59, [(500, 700)])}


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read_npy_rows(fobj, rows):
    """stream a .npy from a file object; return the concatenated row slices (sorted, non-overlapping)"""
    ver = fobj.read(8)
    assert ver[:6] == b"\x93NUMPY", ver
    hl = int.from_bytes(fobj.read(2 if ver[6] == 1 else 4), "little")
    hdr = eval(fobj.read(hl).decode("latin1"))
    dt, shape = np.dtype(hdr["descr"]), hdr["shape"]
    assert not hdr["fortran_order"]
    row = int(np.prod(shape[1:])) * dt.itemsize
    pos, out = 0, []
    for a, b in rows:
        skip = (a - pos) * row
        while skip > 0:
            k = fobj.read(min(skip, 1 << 24)); assert k; skip -= len(k)
        buf = fobj.read((b - a) * row); assert len(buf) == (b - a) * row
        out.append(np.frombuffer(buf, dt).reshape((b - a,) + tuple(shape[1:])))
        pos = b
    return np.concatenate(out), shape


def order():
    sys.path.insert(0, D)
    from check import to_lattice
    man = json.load(open(os.path.join(D, "frozen", "manifest.json")))["sets"]
    res = {"what": "CostNet per-maze arrays index the official raw files in order; P13 e33/e59 are raw slices", "sets": {}}
    for name, (n, rows) in SL.items():
        got = {}
        with tarfile.open(os.path.join(RAW, f"maze_data_test_{n}.tar.gz"), "r|gz") as tf:
            for m in tf:
                base = os.path.basename(m.name)
                if base in ("inputs.npy", "solutions.npy"):
                    got[base] = read_npy_rows(tf.extractfile(m), rows)
                if len(got) == 2:
                    break
        (x, xs), (y, ys) = got["inputs.npy"], got["solutions.npy"]
        code, sol = to_lattice(x.astype(np.float32), y.astype(np.float32))
        code, sol = code.numpy(), sol.numpy()
        ok = sha(code) == man[name]["sha256"]["code"] and sha(sol) == man[name]["sha256"]["sol"]
        res["sets"][name] = {"raw_shape": list(xs), "rows": rows, "code_sha_match": sha(code) == man[name]["sha256"]["code"],
                             "sol_sha_match": sha(sol) == man[name]["sha256"]["sol"], "pass": bool(ok)}
        print(name, res["sets"][name], flush=True)
    # CostNet evaluated the first 1,000 of each raw test file in order (code/bansal.py SETS)
    src = open(os.path.join(CN, "code", "bansal.py")).read()
    res["costnet_sets_first_1000_in_order"] = ('"e2h_maze_33": ("maze_data_test_33", (0, 1000))' in src and
                                                '"e2h_maze_59": ("maze_data_test_59", (0, 1000))' in src)
    res["pass"] = all(v["pass"] for v in res["sets"].values()) and res["costnet_sets_first_1000_in_order"]
    json.dump(res, open(os.path.join(D, "results", "costnet_order_check.json"), "w"), indent=1)
    print("PASS" if res["pass"] else "FAIL")


def ci_t(v):
    from scipy import stats
    v = np.asarray(v, float)
    if len(v) < 2:
        return [None, None]
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return [float(v.mean() - h), float(v.mean() + h)]


def collect():
    refs = {"exact12n": {}, "notes": {}}
    # M-ref (released official checkpoint), this experiment's Phase-1 eval
    f = os.path.join(D, "results", "vm", "trader-dds-1", "out", "ev_mref.json")
    if os.path.exists(f):
        s = json.load(open(f))["sets"]
        refs["exact12n"]["M-ref (released official DT-Recall checkpoint)"] = {k: s[k]["exact@12n"] for k in s}
    # CostNet+BB on the same mazes (valid only if the order check passed)
    oc = os.path.join(D, "results", "costnet_order_check.json")
    if os.path.exists(oc) and json.load(open(oc))["pass"]:
        per = {"e33": [], "e59": []}
        seeds = []
        for s in range(200, 220):
            z = np.load(os.path.join(CN, "vmout", "eval1b", "test", f"bansal_s{s}.npz"))
            e33 = np.concatenate([z["e2h_maze_33/exact"][400:500], z["e2h_maze_33/exact"][600:800]])
            per["e33"].append(float(e33.mean())); per["e59"].append(float(z["e2h_maze_59/exact"][500:700].mean()))
            seeds.append(s)
        refs["exact12n"]["CostNet+BB (20 seeds, same mazes; not a DT step count)"] = {
            k: float(np.mean(v)) for k, v in per.items()}
        refs["notes"]["costnet"] = {"seeds": seeds, "per_seed": per, "ci95": {k: ci_t(v) for k, v in per.items()},
                                    "source": "followups-batch2/item1-costnet/vmout/eval1b/test/bansal_s200..219.npz"}
    else:
        refs["notes"]["costnet"] = "order check missing or failed: quote followups-batch2 item1 RESULTS (first 1,000 mazes)"
    extra = os.path.join(D, "results", "references_extra.json")    # P13 DTR-eq and dt-baseline, filled by hand-checked reads
    if os.path.exists(extra):
        e = json.load(open(extra))
        refs["exact12n"].update(e.get("exact12n", {})); refs["notes"].update(e.get("notes", {}))
    json.dump(refs, open(os.path.join(D, "results", "references.json"), "w"), indent=1)
    print(json.dumps(refs["exact12n"], indent=1))


if __name__ == "__main__":
    {"order": order, "collect": collect}[sys.argv[1]]()
