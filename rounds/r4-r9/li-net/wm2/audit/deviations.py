"""wm2 audit: evidence for the PREREG deviation entries (read-only).

  PYTHONDONTWRITEBYTECODE=1 python3 audit/deviations.py

 - 14:15Z fix: when the first forecaster on the joint WM started on each wave-1 T1 VM (status files + GPU-LOG RUN time);
 - 15:09Z OOM: the failed WM job and its forecasters; RQ1 job lines vs the wave-1 lines;
 - 16:00Z spot: which job of W3-vm1 never finished (preemption) and whether any job file reruns it;
 - 17:42Z drain: start/end of the jobs on the drained T2 baseline rental; F1b lines vs the drained jobs' arguments;
 - code versions: md5 of wm2.py / job.py / round-1 files in every frozen per-rental source.
"""
import glob
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone

G = "/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md"
J = "/home/agent/workspaces/eurisko-study/li-net/wm2/jobs/"
ST = "/tmp/wm2/runs/status/"


def runs():
    out = {}
    for l in open(G):
        m = re.match(r"- (\S+) RUN vm=(soup-gpu-wm2\S+)", l)
        if m:
            out[m.group(2)] = datetime.strptime(m.group(1).rstrip("Z"), "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
    return out


def status(vm):
    return [json.loads(l) for l in open(ST + vm + ".jsonl") if l.strip()]


def t0_of(vm, R):
    L = status(vm)
    return R[vm] + timedelta(seconds=[l["sec"] for l in L if l.get("event") == "setup"][0])


def vm_of(pattern):
    return [p.split("/")[-1][:-6] for p in glob.glob(ST + pattern) if not p.endswith("-b.jsonl")][0]


def main():
    R = runs()
    print("14:15Z fix: first joint-WM forecaster start per wave-1 T1 VM (RUN time has minute resolution)")
    for p in sorted(glob.glob(ST + "soup-gpu-wm2-w1-t1-*.jsonl")):
        vm = p.split("/")[-1][:-6]
        t0 = t0_of(vm, R)
        js = [l for l in status(vm) if "job" in l and re.search(r"--tag WMc-(D|S)-only", l["args"])]
        st = [l["t"] - l["sec"] for l in js]
        ok = sum(1 for l in js if l["rc"] == 0)
        print(f"  {vm[13:30]}: first start ~{(t0 + timedelta(seconds=min(st))).strftime('%H:%M')}Z, {ok}/{len(js)} exited 0")
    print("15:09Z OOM: non-zero exits on wave-1 VMs")
    for p in sorted(glob.glob(ST + "soup-gpu-wm2-w1-*.jsonl")):
        vm = p.split("/")[-1][:-6]
        for l in status(vm):
            if "job" in l and l["rc"] != 0:
                print(f"  {vm[13:30]} job {l['job']} rc {l['rc']} ended ~{(t0_of(vm, R) + timedelta(seconds=l['t'])).strftime('%H:%M')}Z "
                      f"{l['args'].split('--out ')[1].split('/')[-1]}: {l['tail'][-90:].strip()!r}")
    w1 = sorted(l.strip() for l in open(J + "W1-t1-vm3.txt") if "--n 100 --seed 2" in l and "WMs" in l)
    rq1 = sorted(l.strip() for l in open(J + "RQ1-t1-vm0.txt") if l.startswith("--"))
    print(f"  RQ1 job lines identical to the lost wave-1 lines: {w1 == rq1} ({len(rq1)} lines)")
    print("16:00Z spot: jobs of W3-vm1 that never finished")
    jobs = [l.strip() for l in open(J + "W3-t1-vm1.txt") if l.startswith("--")]
    vm = vm_of("soup-gpu-wm2-w3-t1-vm1-*.jsonl")
    done = {l["args"] for l in status(vm) if "job" in l}
    lost = [j.split("--out ")[1] for j in jobs if j not in done]
    print(f"  lost: {lost}")
    later = [f for f in glob.glob(J + "*.txt") if "W3-t1-vm1" not in f and any(x.split('/')[-1] in open(f).read() for x in lost)]
    print(f"  other job files that rerun it: {later or 'none'}")
    print("17:42Z drain of the T2 final-baseline rental")
    vm = vm_of("soup-gpu-wm2-f1-t2-vm0-*.jsonl")
    t0 = t0_of(vm, R)
    L = status(vm)
    ok = [l for l in L if "job" in l and l["rc"] == 0]
    bad = [l for l in L if "job" in l and l["rc"] != 0]
    print(f"  {len(ok)} jobs exited 0 (last start ~{(t0 + timedelta(seconds=max(l['t'] - l['sec'] for l in ok))).strftime('%H:%M')}Z, "
          f"last end ~{(t0 + timedelta(seconds=max(l['t'] for l in ok))).strftime('%H:%M')}Z); {len(bad)} drained (first ended ~"
          f"{(t0 + timedelta(seconds=min(l['t'] for l in bad))).strftime('%H:%M:%S')}Z, message {bad[0]['tail'].strip()!r})")
    f1b = sorted(x.strip() for x in open(J + "F1b-t2-vm0.txt") if x.startswith("--"))
    vb = vm_of("soup-gpu-wm2-f1b-t2-vm0-*.jsonl")
    print(f"  F1b job lines == drained jobs' arguments: {f1b == sorted(l['args'] for l in bad)}; F1b exit codes: "
          f"{sorted({l['rc'] for l in status(vb) if 'job' in l})}")
    print("code versions in the frozen per-rental sources (md5 prefixes: wm2.py job.py smallnet.py netdef.py data.py bninfer.py)")
    seen = {}
    for d in sorted(glob.glob("/tmp/wm2/src-wm2-*")):
        key = tuple(hashlib.md5(open(f"{d}/{f}", "rb").read()).hexdigest()[:6] for f in
                    ("wm2/wm2.py", "wm2/job.py", "smallnet/smallnet.py", "smallnet/netdef.py", "smallnet/data.py", "smallnet/bninfer.py"))
        seen.setdefault(key, []).append(d.split("src-wm2-")[1].rsplit("-", 1)[0])
    for k, v in seen.items():
        print(f"  {' '.join(k)}: {v}")


if __name__ == "__main__":
    main()
