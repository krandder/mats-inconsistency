"""Progress of the loop: results home per wave and tag (counts of N x seed), from /tmp/smallnet/runs."""
import collections
import glob
import json
import os
import sys

RUNS = "/tmp/smallnet/runs"
waves = sys.argv[1:] or sorted(d for d in os.listdir(RUNS) if d.startswith("W"))
for w in waves:
    c = collections.defaultdict(dict)
    for p in glob.glob(f"{RUNS}/{w}/*/*.json"):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        c[r["tag"]][(r["n"], r["seed"])] = (r["val"]["kl"], r.get("stop_reason", "?"))
    tot = sum(len(v) for v in c.values())
    print(f"{w}: {tot} results")
    for t in sorted(c):
        byn = collections.defaultdict(list)
        for (n, s), (kl, sr) in c[t].items():
            byn[n].append(kl)
        cells = " ".join(f"{n}:{1000*sum(v)/len(v):.2f}({len(v)})" for n, v in sorted(byn.items()))
        dl = sum(1 for (kl, sr) in c[t].values() if sr == "deadline")
        print(f"  {t:18s} {cells}" + (f"  [{dl} stopped by deadline]" if dl else ""))
