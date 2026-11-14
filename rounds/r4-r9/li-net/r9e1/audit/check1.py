#!/usr/bin/env python3
"""R9-E1 audit, check 1 (timing and integrity). Read-only; prints evidence lines. Run: python3 check1.py"""
import hashlib
import os
import re
import subprocess
from datetime import datetime, timezone

LI = "/home/agent/workspaces/eurisko-study/li-net"
R9 = f"{LI}/r9e1"
GPULOG = "/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md"
LAUNCH = "/tmp/r9e1/logs/launch-r9e1-main.log"
SPEC = "/home/agent/workspaces/mats/outbox/experiments-round-9-spec.md"
AMEND = "/home/agent/workspaces/mats/outbox/experiments-round-9-amendments.md"


def sha(path=None, data=None):
    h = hashlib.sha256()
    h.update(open(path, "rb").read() if data is None else data)
    return h.hexdigest()


def mt(p, ctime=False):
    st = os.stat(p)
    return datetime.fromtimestamp(st.st_ctime if ctime else st.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


print("now", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
# 1. registration lines and hashes
for f, reg in (("PREREG-R9-E1.md", "results/PREREG-R9-E1-as-registered.md"),
               ("PREREG-R9-E1-AMENDMENTS.md", "results/PREREG-R9-E1-AMENDMENTS-as-registered.md")):
    line = open(f"{R9}/{f[:-3]}.sha256").read().strip()
    h_reg = line.split()[0]
    print(f"{f[:-3]}.sha256: '{line}' (mtime {mt(f'{R9}/{f[:-3]}.sha256')})")
    print(f"   sha256({reg}) = {sha(f'{R9}/{reg}')} -> matches .sha256: {sha(f'{R9}/{reg}') == h_reg}")
    print(f"   sha256({f}) now = {sha(f'{R9}/{f}')} -> equals registered: {sha(f'{R9}/{f}') == h_reg}")
# 2. first r9e1 GPU-LOG line and first launch-log line
first_gpu = next(l.rstrip() for l in open(GPULOG) if "soup-gpu-r9e1-" in l)
print("first soup-gpu-r9e1- GPU-LOG line:", first_gpu)
print("first launch-log line:", open(LAUNCH).readline().rstrip())
# 3. PREREG differs only in section 5
reg = open(f"{R9}/results/PREREG-R9-E1-as-registered.md").read().splitlines()
cur = open(f"{R9}/PREREG-R9-E1.md").read().splitlines()
i5 = next(i for i, l in enumerate(reg) if l.startswith("## 5. Deviations log"))
same_head = cur[:i5 + 2] == reg[:i5 + 2]
same_tail = cur[-1] == reg[-1]
print(f"PREREG: lines 1..{i5 + 2} (through the section-5 heading) identical: {same_head}; last line identical ('{reg[-1]}'): {same_tail}; "
      f"registered section 5 body: {reg[i5 + 1:-1]}; current section 5 body has {len(cur) - len(reg[:i5 + 2]) - 1} lines")
# 4. staged manifests vs registered hashes (full or 8-char prefixes in the registered PREREG text) vs current files
regtxt = open(f"{R9}/results/PREREG-R9-E1-as-registered.md").read()
pairs = re.findall(r"((?:ckpt/s10\d/)?[\w/]+\.(?:py|sh|npz))\s+([0-9a-f]{8,64})", regtxt)
regh = {}
for f, h in pairs:
    regh.setdefault(f, h)
blocks, cur_b = {}, None
for l in open(f"{R9}/results/stage-manifests.txt"):
    l = l.rstrip()
    if l.startswith("## "):
        cur_b = l[3:]
        blocks[cur_b] = {}
    elif l:
        h, f = l.split(None, 1)
        blocks[cur_b][f] = h
loc = {"e1v5.py": f"{LI}/r8e1/e1v5.py", "e1v4.py": f"{LI}/r7e1/e1v4.py", "e1v3.py": f"{LI}/r6e1/e1v3.py",
       "smallnet/netdef.py": f"{LI}/smallnet/netdef.py"}
for m in ("env6.py", "wm.py", "rollouts.py", "run.py", "partA2.py"):
    loc[m] = f"{LI}/r5e1/{m}"
for s in (101, 102, 103):
    loc[f"ckpt/s{s}/theta_real_v2.npz"] = f"{LI}/r7e1/ckpt/s{s}/theta_real_v2.npz"
print(f"registered hashes found in the PREREG text: {len(regh)} files")
for t, b in blocks.items():
    bad_reg = [f for f in b if f in regh and not b[f].startswith(regh[f])]
    missing = [f for f in regh if f not in b]
    print(f"stage block {t}: {len(b)} files; mismatches vs registered: {bad_reg or 'none'}; registered files missing: {missing or 'none'}")
allsame = all(b == list(blocks.values())[0] for b in blocks.values())
print("all stage blocks identical:", allsame)
first = list(blocks.values())[0]
now_bad = []
for f, h in first.items():
    p = loc.get(f, f"{R9}/{f}")
    if sha(p) != h:
        now_bad.append(f)
print("current files (li-net) vs staged hashes: mismatches", now_bad or "none")
ok = subprocess.run(["sha256sum", "-c", "--quiet", "MANIFEST.sha256"], cwd="/tmp/r9e1/src", capture_output=True, text=True)
print("staged tree /tmp/r9e1/src vs its MANIFEST.sha256: rc", ok.returncode, ok.stdout.strip() or "all OK")
# 5. files changed after registration (mtime or ctime > 19:16:38Z)
regt = datetime(2026, 10, 17, 19, 16, 38, tzinfo=timezone.utc).timestamp()
chg = []
for d in (R9,):
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if x not in ("results", "__pycache__", ".pytest_cache", "audit")]
        for f in files:
            p = os.path.join(root, f)
            st = os.stat(p)
            if st.st_mtime > regt or st.st_ctime > regt:
                chg.append((p.replace(R9 + "/", ""), mt(p), mt(p, True)))
for p in list(loc.values()):
    st = os.stat(p)
    if st.st_mtime > regt or st.st_ctime > regt:
        chg.append((p, mt(p), mt(p, True)))
print("files with mtime or ctime after registration 19:16:38Z (results/ excluded):")
for c in sorted(chg):
    print("   ", c)
# 6. spec and amendments
spec = open(SPEC, "rb").read()
am = open(AMEND, "rb").read()
print("spec sha256", sha(data=spec), "mtime", mt(SPEC))
print("amendments sha256", sha(data=am), "mtime", mt(AMEND))
sl = spec.decode().split("\n")
al = am.decode().split("\n")
print("spec lines 36-41 sha256", sha(data=("\n".join(sl[35:41]) + "\n").encode()), "| lines 73-98", sha(data=("\n".join(sl[72:98]) + "\n").encode()))
print("amendments lines 43-53 sha256", sha(data=("\n".join(al[42:53]) + "\n").encode()))
pre = open(f"{R9}/results/PREREG-R9-E1-as-registered.md").read()
print("spec 1.2 (lines 73-98) verbatim in registered PREREG 2a:", "\n".join(sl[72:98]) in pre,
      "| global protocol (36-41) verbatim:", "\n".join(sl[35:41]) in pre, "| amendments Part B verbatim:", "\n".join(al[42:53]) in pre)
amd = open(f"{R9}/results/PREREG-R9-E1-AMENDMENTS-as-registered.md").read()
print("amendments Part B verbatim in PREREG-R9-E1-AMENDMENTS.md:", "\n".join(al[42:53]) in amd)
print("other round-9 amendment files:", sorted(f for f in os.listdir(os.path.dirname(AMEND)) if "round-9-amend" in f))
