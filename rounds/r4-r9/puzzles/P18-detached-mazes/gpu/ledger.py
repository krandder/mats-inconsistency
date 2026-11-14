"""Render the P18 section of trader-economy/GPU-LEDGER.md from gpu/spend-state.tsv (name, created, last seen, zone;
kept by watchdog.sh) and gpu/deleted.tsv (name, time, reason). Replaces only the P18 section; writes only on change."""
import datetime as dt, os
G = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(G, "..", "..", "..", "GPU-LEDGER.md")
HEAD = "## P18 detached mazes (trader-dds-*, cap $40)"
RATE = 0.71


def ts(s):
    return dt.datetime.fromtimestamp(int(s), dt.timezone.utc).strftime("%Y-%m-%d %H:%M")


def main():
    rows, dele = [], {}
    if os.path.exists(os.path.join(G, "deleted.tsv")):
        for l in open(os.path.join(G, "deleted.tsv")):
            p = l.rstrip("\n").split("\t")
            if len(p) >= 3:
                dele.setdefault(p[0], (p[1][:16].replace("T", " "), p[2]))
    live = set()
    try:
        live = {l.split()[0] for l in open("/tmp/p18gpu/wd_vms.txt") if l.strip()}
    except OSError:
        pass
    tot = 0.0
    for l in open(os.path.join(G, "spend-state.tsv")):
        p = l.split()
        if len(p) < 3:
            continue
        n, c, last = p[0], int(p[1]), int(p[2])
        z = p[3] if len(p) > 3 else ""
        h = (last - c + 300) / 3600
        tot += h * RATE
        if n in live:
            d = "running"
        elif n in dele:
            d = f"{dele[n][0]} ({dele[n][1]})"
        else:
            d = f"{ts(last)} (last seen; gone)"
        rows.append((c, f"| {ts(c)} | {n} | {z} | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | {d} | {h:.2f} | {h * RATE:.2f} |"))
    rows.sort()
    sec = [HEAD,
           f"Rate ceiling $0.71/h (g2-standard-4 + 1x L4 at on-demand list price + 100 GB disk; spot is lower; spend = (last seen - created + 5 min) x rate). "
           f"Watchdog: puzzles/P18-detached-mazes/gpu/watchdog.sh (5-min loop; deletes all at est. spend >= $40, any VM at 5.8 h or 20 min idle). "
           f"Controller: puzzles/P18-detached-mazes/gpu/ctl.sh (3-min sha256-verified sync, resume on -rN after preemption). Est. total ${tot:.2f}.",
           "| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |", "|---|---|---|---|---|---|---|"] + [r for _, r in rows]
    txt = open(LEDGER).read()
    lines = txt.split("\n")
    if HEAD in lines:
        i = lines.index(HEAD)
        j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("## ")), len(lines))
        new = lines[:i] + sec + [""] + lines[j:] if j < len(lines) else lines[:i] + sec + [""]
    else:
        new = lines + ([""] if lines and lines[-1] else []) + sec + [""]
    out = "\n".join(new)
    if out != txt:
        with open(LEDGER + ".p18tmp", "w") as f:
            f.write(out)
        os.replace(LEDGER + ".p18tmp", LEDGER)
    print(f"total ${tot:.2f}, {len(rows)} VMs")


if __name__ == "__main__":
    main()
