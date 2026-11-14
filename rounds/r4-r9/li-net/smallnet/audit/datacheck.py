"""Audit: cached val/test/train episodes and exact posteriors vs fresh regeneration from the harness; seed-range and
episode overlap. Run from anywhere: LIR_HARNESS=<harness> PYTHONDONTWRITEBYTECODE=1 python3 audit/datacheck.py (~30 s, numpy).
Reads the cache in /tmp/smallnet/data (bulky; regenerable with data.py)."""
import sys, time, hashlib, numpy as np
sys.path.insert(0, "/home/agent/workspaces/eurisko-study/li-net/smallnet")
import data as D
from env.shiphunt import placements
t0 = time.time()
def h(*arrs):
    m = hashlib.sha256()
    for a in arrs: m.update(np.ascontiguousarray(a).tobytes())
    return m.hexdigest()[:16]
te = dict(np.load("/tmp/smallnet/data/test-n5000.post.npz")); va = dict(np.load("/tmp/smallnet/data/val-n3000.post.npz"))
print("test seeds", te["seeds"].min(), te["seeds"].max(), len(te["seeds"]), "val seeds", va["seeds"].min(), va["seeds"].max(), len(va["seeds"]))
for name, d, split, n in (("test", te, "test", 5000), ("val", va, "val", 3000)):
    assert (d["seeds"] == D.seeds_for(split, n)).all()
    g = D.episodes(D.seeds_for(split, n))
    same = all((g[k] == d[k]).all() for k in ("occ", "pc", "pr"))
    print(name, "regenerated episodes identical to cache:", same, "hash(occ,pc,pr)", h(d["occ"], d["pc"], d["pr"]))
# posterior recomputation on all test boards and val boards
for name, d in (("test", te), ("val", va)):
    post, lt = D.exact_posteriors(d["seeds"])
    print(name, "posterior max |diff| vs cache:", float(np.abs(post - d["post"]).max()), "ltrue max |diff|:", float(np.abs(lt - d["ltrue"]).max()), f"{time.time()-t0:.0f}s")
# training caches
tr = {}
for s in range(5):
    z = dict(np.load(f"/tmp/smallnet/data/train-n30000-s{s}.npz")); z10 = dict(np.load(f"/tmp/smallnet/data/train-n10000-s{s}.npz"))
    assert (z["seeds"] == D.seeds_for("train", 30000, s)).all()
    nested = all((z[k][:10000] == z10[k]).all() for k in ("occ", "pc", "pr"))
    g = D.episodes(z["seeds"][:2000])
    regen = all((g[k] == z[k][:2000]).all() for k in ("occ", "pc", "pr"))
    print(f"train s{s}: seeds {z['seeds'].min()}..{z['seeds'].max()} nested10k={nested} regen_first2000={regen}")
    tr[s] = z
# seed-range disjointness
allseeds = np.concatenate([tr[s]["seeds"] for s in range(5)])
print("train seed ranges disjoint from val/test:", not (set(allseeds) & set(te["seeds"])) and not (set(allseeds) & set(va["seeds"])), "train seeds unique:", len(set(allseeds)) == len(allseeds))
# episode-level overlap: identical (board, probe trajectory) between any train set (all 30k x 5 seeds) and test/val
def keys(d):
    return {d["occ"][i].tobytes() + d["pc"][i].tobytes() + d["pr"][i].tobytes() for i in range(len(d["occ"]))}
ktest, kval = keys(te), keys(va)
ktr = set()
for s in range(5): ktr |= keys(tr[s])
print("identical full episodes train∩test:", len(ktest & ktr), " train∩val:", len(kval & ktr), " val∩test:", len(kval & ktest))
# evidence-set overlap at stage 6 (unordered set of (cell, report)) -- same inputs at the last stage
def evkeys(d, k=6):
    out = []
    for i in range(len(d["occ"])):
        out.append(tuple(sorted(zip(d["pc"][i][:k].tolist(), d["pr"][i][:k].tolist()))))
    return out
etr = set()
for s in range(5): etr |= set(evkeys({k: v[:10000] for k, v in tr[s].items()}))
et = evkeys(te)
print("test boards whose stage-6 evidence set appears in some train set (seeds 0-4, first 10k):", sum(e in etr for e in et), "of", len(et))
P = placements(D.CFG)
print("placements", len(P), "test boards with a placement seen in train s0 first 10k:", np.mean([o.tobytes() in {x.tobytes() for x in tr[0]["occ"][:10000]} for o in te["occ"]]).round(3))
print(f"done {time.time()-t0:.0f}s")
