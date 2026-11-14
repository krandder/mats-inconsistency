"""Write P15 queue files (reproducible).  python make_queues.py pilot|tune|full [frozen.json]
pilot : Bl and A (3 seeds, full B_D, one mid config) at both primary cells + arm C (G0 part)
tune  : spec grids at the primary cells, 3 seeds, 50% of B_D, final eval on validation only
full  : frozen configs (from analyze.py freeze -> frozen.json), seeds per the spec"""
import json
import sys

PRIM = (("prefix", 16), ("maze", 32))


def J(id_, arm, kind, D, seeds, configs, **kw):
    j = {"id": id_, "arm": arm, "kind": kind, "D": D, "seeds": list(seeds), "configs": configs}
    j.update(kw)
    return j


def pilot():
    q = []
    for kind, D in PRIM:
        for arm in ("Bl", "A"):
            q.append(J(f"pilot_{arm}_{kind}_D{D}", arm, kind, D, range(3), [{"lr": 3e-3, "tau": 1.0}], frac=1.0))
        q.append(J(f"pilot_C_{kind}_D{D}", "C", kind, D, range(3), [{"lr": 3e-3, "tau": 1.0}], frac=1.0))
    return q


def grid(arm):
    lrs = (1e-3, 3e-3)
    if arm in ("A", "A-prog", "A-ds"):
        return [{"lr": lr, "tau": t} for t in (0.5, 1.0, 2.0) for lr in lrs]
    if arm == "A-1step":
        return [{"lr": lr, "tau": t} for t in (0.5, 1.0) for lr in lrs]
    return [{"lr": lr, "tau": 1.0} for lr in lrs]


TUNE_ARMS = ("Bl", "Bl-local", "Bl-ac", "O", "C", "A", "A-prog", "A-ds", "A-1step", "A-soft-prog")


def tune(seeds=(0, 1, 2)):
    q = []
    for kind, D in PRIM:
        for arm in TUNE_ARMS:
            q.append(J(f"tune_{arm}_{kind}_D{D}", arm, kind, D, seeds, grid(arm), frac=0.5, evals=["val"]))
        for w in (16, 32, 64):
            q.append({"id": f"tune_P-eq_w{w}_{kind}_D{D}", "arm": "P-eq", "kind": kind, "D": D, "seeds": list(seeds),
                      "width": w, "frac": 0.5, "lr": 1e-3, "evals": ["val"]})
    # longest first (maze Bl-family and A-family with many configs)
    q.sort(key=lambda j: (j["kind"] != "maze", j["arm"] not in ("Bl", "Bl-ac", "Bl-local", "O")))
    return q


SEEDS10, SEEDS5 = list(range(100, 110)), list(range(100, 105))   # disjoint from the tuning seeds 0-2
PRIM10 = ("Bl", "Bl-ac", "Bl-local", "A", "A-prog", "A-ds", "A-1step", "A-soft-prog")
PRIM5 = ("O", "C", "Bl-shuffle")
OTHER_D = {"prefix": (8, 32), "maze": (8, 16)}
PRIM_D = dict(PRIM)


def cfg_of(fz, cell, arm):
    return fz[cell][{"Bl-shuffle": "Bl", "Bl-fan": "Bl"}.get(arm, arm)]["config"]


def cost_key(j):
    """rough cost for longest-first ordering: cells^2-ish x depth x seeds"""
    n = {"prefix": j["D"], "maze": {8: 4, 16: 6, 32: 9}[j["D"]] ** 2}[j["kind"]]
    return -(n * j["D"] * len(j["seeds"]) * (3 if j["arm"].startswith("Bl") or j["arm"] == "O" else 2))


# Budget simplification (decided 2026-09-29 22:30Z after the pilot, before any full-grid run; stated in RESULTS.md):
# the maze pilot cost ~23 min of L4 eval per 3 elements (n=59 rollouts to 4 n^2 = 13924 levels) and every arm scored 0
# beyond n=9, so on the maze (i) the primary cell runs 5 seeds per arm instead of 10, and (ii) the n=37 and n=59 test
# sets hold 50 mazes instead of 100 (n=19 keeps 200).  Prefix is unchanged.  Maze D8 is dropped (pre-registered
# descoping step 1).
MAZE_SEEDS_PRIM = SEEDS5
TEST_SIMPLE = {"prefix": {"mult": (4, 16, 64), "num": (1000, 500, 200)},
               "maze": {"sizes": (19, 37, 59), "num": (200, 50, 50)}}


def full(fz, fan=(), drop=()):
    """fan: tasks where the pilot showed geometric |lam| growth (adds Bl-fan); drop: e.g. ("maze_D8",)"""
    q = []
    for kind, D in PRIM:
        cell = f"{kind}_D{D}"
        s10 = MAZE_SEEDS_PRIM if kind == "maze" else SEEDS10
        mz = {"TEST": TEST_SIMPLE} if kind == "maze" else {}
        for arm in PRIM10 + PRIM5 + (("Bl-fan",) if kind in fan else ()):
            kw = {"extra_c": [16]} if (kind == "prefix" and arm in ("Bl", "A", "A-prog", "A-ds")) else {}
            q.append(J(f"full_{arm}_{cell}", arm, kind, D, SEEDS5 if arm in PRIM5 else s10,
                       [cfg_of(fz, cell, arm)], **kw, **mz))
        pe = fz[cell]["P-eq"]
        q.append({"id": f"full_P-eq_{cell}", "arm": "P-eq", "kind": kind, "D": D, "seeds": s10,
                  "width": pe["width"], "lr": pe.get("lr", 1e-3), **mz})
    for kind, Ds in OTHER_D.items():
        cell = f"{kind}_D{PRIM_D[kind]}"
        for D in Ds:
            if f"{kind}_D{D}" in drop:
                continue
            for arm in ("Bl", "A", "A-prog", "A-ds"):
                kw = {"extra_test": [[8192, 50]]} if (kind == "prefix" and D == 8) else {}
                if kind == "maze":
                    kw["TEST"] = TEST_SIMPLE
                q.append(J(f"full_{arm}_{kind}_D{D}", arm, kind, D, SEEDS5, [cfg_of(fz, cell, arm)], **kw))
    for arm in ("Bl", "Bl-ac", "A-prog", "A-ds", "O"):
        q.append(J(f"full_{arm}_prefix_D16_mod3", arm, "prefix", 16, SEEDS5, [cfg_of(fz, "prefix_D16", arm)], mod=3))
    q.sort(key=lambda j: cost_key(j) if j["arm"] != "P-eq" else cost_key(dict(j, arm="A")))
    return q


def timing(fz):
    """200-step single-seed runs per arm and D for the equal-wall-clock S1 numbers (run with NPAR=1)."""
    q = []
    for kind in ("prefix", "maze"):
        cell = f"{kind}_D{PRIM_D[kind]}"
        for D in (8, 16, 32):
            for arm in ("Bl", "A", "A-prog", "A-ds", "A-1step"):
                q.append(J(f"timing_{arm}_{kind}_D{D}", arm, kind, D, [0], [cfg_of(fz, cell, arm)], max_steps=200,
                           no_curve=True, final=False))
    return q


if __name__ == "__main__":
    which = sys.argv[1]
    if which in ("full", "timing"):
        fz = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "frozen.json"))
        fan = [a for a in sys.argv[3:] if not a.startswith("drop=")]
        drop = [a[5:] for a in sys.argv[3:] if a.startswith("drop=")]
        q = full(fz, fan, drop) if which == "full" else timing(fz)
    else:
        q = {"pilot": pilot, "tune": tune}[which]()
    json.dump(q, open(f"queue_{which}.json", "w"), indent=0)
    print(which, len(q), "jobs")
