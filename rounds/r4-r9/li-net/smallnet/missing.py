"""List jobs of a wave whose result JSON has not come home, and write them to a new job file.

  python3 missing.py --wave W1 [--out jobs/W1-retry.txt] [--split K]   (K files when --split > 1)
A job counts as done when /tmp/smallnet/runs/<path under /root/runs/smallnet/> exists.
"""
import argparse
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = "/tmp/smallnet/runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--exclude-vm", default="", help="comma list of vm indices still running (their jobs are not missing)")
    a = ap.parse_args()
    excl = {int(x) for x in a.exclude_vm.split(",") if x}
    todo = []
    for f in sorted(glob.glob(os.path.join(HERE, "jobs", f"{a.wave}-vm*.txt"))):
        vm = int(re.search(r"-vm(\d+)\.txt$", f).group(1))
        if vm in excl:
            continue
        for line in open(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out = line.split("--out ")[1].split()[0]
            local = os.path.join(RUNS, os.path.relpath(out, "/root/runs/smallnet"))
            if not os.path.exists(local):
                todo.append(line)
    print(len(todo), "missing")
    if a.out and todo:
        k = max(1, a.split)
        for i in range(k):
            p = a.out if k == 1 else a.out.replace(".txt", f"-vm{i}.txt")
            with open(p, "w") as fh:
                fh.write(f"# missing jobs of {a.wave} (part {i} of {k})\n" + "\n".join(todo[i::k]) + "\n")
            print(p, len(todo[i::k]))


if __name__ == "__main__":
    main()
