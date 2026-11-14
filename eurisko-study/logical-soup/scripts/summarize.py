#!/usr/bin/env python3
"""Summarize a run matrix: one row per cell, mean±sd over seeds.

usage: python3 scripts/summarize.py runs/<matrix>/ [--out DIR] [--plot]
       python3 scripts/summarize.py runs/a-k12-f12-s1 runs/a-k12-f12-s2 ... --out DIR
       [--window 50] [--takeoff 50] [--price 0.22] [--hold 50] [--slope 1000]

Inputs: a matrix directory (every subdirectory with a metrics.csv is a run) or
run directories. A run's cell is its directory name without the seed: runs
named <prefix>k<a>-f<b>-s<seed>[-<variant>] group as k<a>-f<b>[-<variant>];
other names group by config.json (K, F). Writes summary.md and summary.csv to
--out (default: the matrix directory); --plot adds brier_vs_K.png and
brier_vs_F.png there (PYTHONPATH=.pylib for matplotlib).

Per run (docs/matrix.md has the definitions):
  brier_<fam>   final Brier per family: mean of the last --window epochs, in 1.0 units
  takeoff       first epoch with births >= --takeoff (blank if never)
  ttp           held-out family only: epochs from release until its Brier stays
                below --price for --hold consecutive epochs (blank if never or
                no held-out family)
  slope_<sh>    least-squares slope of each shadow trader's cumulative profit
                over the last --slope epochs, shares per epoch (Q16 ticks / 65536)
  coh_and, coh_or  final coherence violation (last --window mean, 1.0 units)
  genotypes, gini  last epoch
  epochs, wall_s   rows in metrics.csv, sum of timing.csv wall seconds
Per cell: mean and sample sd over seeds of each column; n = seeds; ttp_n and
takeoff_n count the seeds where the event happened (means are over those).
"""
import argparse, csv, json, math, os, re, statistics, sys

FAMS = ["periodic", "lfsr", "ca30", "ca110", "prime", "pi", "and", "or"]
SHADOWS = ["c50", "mom", "pat2", "pat4", "pat8", "arb"]
NAME_RE = re.compile(r"^(?:.*?-)?(k\d+-f\d+)-s(\d+)(?:-(.+))?$")


def num(s):
    try:
        return float(s)
    except ValueError:
        return math.nan


def load_metrics(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r]
    head, data = rows[0], [r for r in rows[1:] if r[0][:1].isdigit()]
    return {h: [num(r[i]) for r in data] for i, h in enumerate(head)}


def fmean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else math.nan


def slope(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if not math.isnan(y)]
    if len(pts) < 2:
        return math.nan
    mx = sum(p[0] for p in pts) / len(pts); my = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx if sxx else math.nan


def takeoff(epochs, births, threshold):
    for e, b in zip(epochs, births):
        if b >= threshold:
            return e
    return math.nan


def time_to_price(epochs, brier, release, price, hold):
    """Epochs from release to the start of the first run of `hold` consecutive
    epochs with Brier < price (nan epochs break the run)."""
    run_start, run_len = None, 0
    for e, b in zip(epochs, brier):
        if e < release:
            continue
        if not math.isnan(b) and b < price:
            if run_len == 0:
                run_start = e
            run_len += 1
            if run_len >= hold:
                return run_start - release
        else:
            run_len = 0
    return math.nan


def run_stats(run, a):
    m = load_metrics(os.path.join(run, "metrics.csv"))
    try:
        conf = json.load(open(os.path.join(run, "config.json")))
    except (OSError, ValueError):
        conf = {}
    ep = m["epoch"]
    w = slice(-a.window, None)
    s = {"epochs": len(ep)}
    for f in FAMS:
        s[f"brier_{f}"] = fmean(m[f"brier_{f}"][w]) / 1e6
    s["takeoff"] = takeoff(ep, m["births"], a.takeoff)
    held, rel = conf.get("held_out_family", "none"), conf.get("release", "none")
    s["ttp"] = math.nan
    if held not in ("none", None) and rel not in ("none", None):
        fam = FAMS[int(held)] if str(held).isdigit() else held
        br = [b / 1e6 for b in m[f"brier_{fam}"]]
        s["ttp"] = time_to_price(ep, br, float(rel), a.price, a.hold)
    last = slice(-a.slope, None)
    for sh in SHADOWS:
        s[f"slope_{sh}"] = slope(ep[last], m[f"sh_{sh}"][last]) / 65536
    s["coh_and"] = fmean(m["coh_and_ppm"][w]) / 1e6
    s["coh_or"] = fmean(m["coh_or_ppm"][w]) / 1e6
    s["genotypes"] = m["genotypes"][-1]
    s["gini"] = m["gini_ppm"][-1] / 1e6
    wall = 0.0
    try:
        with open(os.path.join(run, "timing.csv")) as f:
            wall = sum(float(r["wall_s"]) for r in csv.DictReader(f))
    except (OSError, KeyError, ValueError):
        wall = math.nan
    s["wall_s"] = wall
    return s, conf


def cell_of(run, conf):
    name = os.path.basename(os.path.normpath(run))
    mt = NAME_RE.match(name)
    if mt:
        return mt.group(1) + (f"-{mt.group(3)}" if mt.group(3) else ""), int(mt.group(2))
    logk = conf.get("logk", int(math.log2(conf.get("K", 1))))
    logq = conf.get("logq", int(math.log2(conf.get("F", 1))))
    return f"k{logk}-f{logq}", conf.get("seed", name)


COLS = [f"brier_{f}" for f in FAMS] + ["takeoff", "ttp"] + [f"slope_{s}" for s in SHADOWS] + \
       ["coh_and", "coh_or", "genotypes", "gini", "epochs", "wall_s"]
FMT = {"takeoff": "{:.0f}", "ttp": "{:.0f}", "genotypes": "{:.0f}", "epochs": "{:.0f}",
       "wall_s": "{:.0f}", "gini": "{:.3f}", "coh_and": "{:.4f}", "coh_or": "{:.4f}"}


def aggregate(runs):
    cells = {}
    for cell, seed, s in runs:
        cells.setdefault(cell, []).append((seed, s))
    out = []
    for cell in sorted(cells, key=cell_sort):
        seeds = cells[cell]
        row = {"cell": cell, "n": len(seeds), "seeds": " ".join(str(x[0]) for x in sorted(seeds))}
        for c in COLS:
            xs = [s[c] for _, s in seeds if not math.isnan(s[c])]
            row[c] = statistics.fmean(xs) if xs else math.nan
            row[c + "_sd"] = statistics.stdev(xs) if len(xs) > 1 else math.nan
            if c in ("takeoff", "ttp"):
                row[c + "_n"] = len(xs)
        out.append(row)
    return out


def cell_sort(c):
    mt = re.match(r"k(\d+)-f(\d+)(?:-(.+))?$", c)
    return (0, int(mt.group(1)), int(mt.group(2)), mt.group(3) or "") if mt else (1, 0, 0, c)


def fmt(row, c):
    v, sd = row[c], row.get(c + "_sd", math.nan)
    if math.isnan(v):
        return "-"
    f = FMT.get(c, "{:.3f}" if c.startswith("brier") else "{:+.3g}")
    txt = f.format(v)
    if not math.isnan(sd):
        txt += "±" + f.replace("+", "").format(sd)
    if c + "_n" in row and row[c + "_n"] < row["n"]:
        txt += f" ({row[c + '_n']}/{row['n']})"
    return txt


def write(rows, out, a):
    os.makedirs(out, exist_ok=True)
    fields = ["cell", "n", "seeds"]
    for c in COLS:
        fields += [c, c + "_sd"] + ([c + "_n"] if c in ("takeoff", "ttp") else [])
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: ("" if isinstance(r[k], float) and math.isnan(r[k]) else r[k]) for k in fields})
    short = {f"brier_{f}": f for f in FAMS}
    short.update({f"slope_{s}": f"Δsh_{s}" for s in SHADOWS})
    lines = [f"Final Brier: mean of the last {a.window} epochs. takeoff: first epoch with births >= {a.takeoff}. "
             f"ttp: epochs after release until the held-out family's Brier < {a.price} for {a.hold} epochs. "
             f"Δsh: shadow profit slope over the last {a.slope} epochs, shares/epoch. "
             "mean±sd over seeds; (k/n): seeds where the event happened.", "",
             "| cell | n | " + " | ".join(short.get(c, c) for c in COLS) + " |",
             "|" + "---|" * (len(COLS) + 2)]
    for r in rows:
        lines.append(f"| {r['cell']} | {r['n']} | " + " | ".join(fmt(r, c) for c in COLS) + " |")
    md = "\n".join(lines) + "\n"
    open(os.path.join(out, "summary.md"), "w").write(md)
    return md


PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = {}
    for r in rows:
        mt = re.match(r"k(\d+)-f(\d+)$", r["cell"])
        if mt:
            base[(int(mt.group(1)), int(mt.group(2)))] = r
    if not base:
        print("summarize: no base k<a>-f<b> cells to plot", file=sys.stderr)
        return
    ks = sorted({k for k, _ in base}); fs = sorted({f for _, f in base})
    for name, xs, panels, key in (("K", ks, fs, lambda x, p: (x, p)), ("F", fs, ks, lambda x, p: (p, x))):
        pname = "F" if name == "K" else "K"
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels) + 1.6, 3.6), squeeze=False, sharey=True)
        for ax, p in zip(axes[0], panels):
            for fam, color in zip(FAMS, PALETTE):
                pts = [(x, base[key(x, p)]) for x in xs if key(x, p) in base]
                ys = [r[f"brier_{fam}"] for _, r in pts]
                sd = [0 if math.isnan(r[f"brier_{fam}_sd"]) else r[f"brier_{fam}_sd"] for _, r in pts]
                ax.errorbar([x for x, _ in pts], ys, yerr=sd, color=color, linewidth=2, marker="o",
                            markersize=5, capsize=3, label=fam)
            ax.axhline(0.25, color="#999", linestyle="--", linewidth=1)
            ax.set_title(f"{pname} = 2^{p}", fontsize=10)
            ax.set_xlabel(f"log2 {name}"); ax.set_xticks(xs)
            ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
        axes[0][0].set_ylabel("final Brier (0.25 = constant 0.5)")
        axes[0][-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.suptitle(f"Brier per family vs {name} (mean ± sd over seeds)", fontsize=11)
        fig.tight_layout()
        path = os.path.join(out, f"brier_vs_{name}.png")
        fig.savefig(path, dpi=110); plt.close(fig)
        print(f"wrote {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("paths", nargs="+")
    p.add_argument("--out"); p.add_argument("--plot", action="store_true")
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--takeoff", type=float, default=50)
    p.add_argument("--price", type=float, default=0.22)
    p.add_argument("--hold", type=int, default=50)
    p.add_argument("--slope", type=int, default=1000)
    a = p.parse_args(argv)
    runs = []
    for path in a.paths:
        if os.path.isfile(os.path.join(path, "metrics.csv")):
            dirs = [path]
        else:
            dirs = sorted(os.path.join(path, d) for d in os.listdir(path)
                          if os.path.isfile(os.path.join(path, d, "metrics.csv")))
        runs += dirs
    if not runs:
        sys.exit("summarize: no runs (directories with metrics.csv) found")
    out = a.out or (a.paths[0] if len(a.paths) == 1 and runs[0] != a.paths[0] else None)
    if not out:
        sys.exit("summarize: several run directories given; pass --out DIR")
    stats = []
    for r in runs:
        s, conf = run_stats(r, a)
        cell, seed = cell_of(r, conf)
        stats.append((cell, seed, s))
    rows = aggregate(stats)
    print(write(rows, out, a), end="")
    if a.plot:
        plot(rows, out)


if __name__ == "__main__":
    main()
