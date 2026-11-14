#!/usr/bin/env python3
"""R8-E1 audit, check 1: registration timing and integrity, staged hashes per launch, post-registration changes, amendments folded in.
Read-only. Run: python3 a1_integrity.py"""
import difflib
import glob
import hashlib
import json
import os
import re
from datetime import datetime, timezone

OUT = os.path.dirname(os.path.abspath(__file__))
LI = "/home/agent/workspaces/eurisko-study/li-net"
R8 = f"{LI}/r8e1"
OB = "/home/agent/workspaces/mats/outbox"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
mt = lambda p: datetime.fromtimestamp(os.stat(p).st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
rep = {"checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}

# 1. registration files and as-registered copies
for base in ("PREREG-R8-E1", "PREREG-R8-E1-AMENDMENTS"):
    line = open(f"{R8}/{base}.sha256").read().split()
    rep[base] = {"sha256_file": line[0], "registered_line": " ".join(line[1:]), "sha256_file_mtime": mt(f"{R8}/{base}.sha256"),
                 "as_registered_copy_sha256": sha(f"{R8}/results/{base}-as-registered.md"), "current_sha256": sha(f"{R8}/{base}.md")}
    rep[base]["as_registered_matches"] = rep[base]["as_registered_copy_sha256"] == line[0]
    rep[base]["current_matches"] = rep[base]["current_sha256"] == line[0]
a = open(f"{R8}/results/PREREG-R8-E1-as-registered.md").read().splitlines()
b = open(f"{R8}/PREREG-R8-E1.md").read().splitlines()
sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
ops = [(t, i1, i2, j1, j2) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal"]
sec5 = next(i for i, l in enumerate(a) if l.startswith("## 5. Deviations log"))
rep["PREREG_diff_ops"] = [{"op": t, "registered_lines": [i1 + 1, i2], "current_lines": [j1 + 1, j2],
                           "removed": a[i1:i2], "inside_section_5": i1 >= sec5} for t, i1, i2, j1, j2 in ops]
rep["PREREG_only_section5_changed"] = all(i1 >= sec5 for t, i1, i2, j1, j2 in ops) and a[-1] == b[-1]
rep["deviation_entries"] = re.findall(r"^(\d+)\. \*\*(\S+Z), ([^*]+)\*\*", open(f"{R8}/PREREG-R8-E1.md").read(), re.M)

# 2. timing: first soup-gpu-r8e1- GPU-LOG line, launch log, VM name stamp
gl = [l.rstrip() for l in open(f"{LI}/../logical-soup/GPU-LOG.md") if "soup-gpu-r8e1-" in l]
rep["first_gpu_log_r8e1_line"] = gl[0]
lau = open("/tmp/r8e1/logs/launch-r8e1-main.log").read().splitlines()[0]
rep["first_launch_log_line"] = lau
rep["registration_precedes_launch_s"] = (datetime.strptime(lau.split()[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                                         - datetime.strptime("2026-10-17T16:07:15Z", "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)).total_seconds()

# 3. stage manifests per launch vs registration
blocks, cur = {}, None
for l in open(f"{R8}/results/stage-manifests.txt"):
    if l.startswith("## "):
        cur = l[3:].strip()
        blocks[cur] = {}
    elif l.strip():
        h, f = l.split()
        blocks[cur][f] = h
reg = blocks["2026-10-17T16:07:15Z"]
rep["prereg_listed_hash_prefixes_match_registration_block"] = all(
    any(reg[f].startswith(pfx) for f in reg if f.endswith(name)) for name, pfx in re.findall(r"(\S+\.(?:py|sh|npz)) ([0-9a-f]{8})", open(f"{R8}/PREREG-R8-E1.md").read()))
rep["stage_blocks"] = {}
for t, blk in blocks.items():
    rep["stage_blocks"][t] = {"changed_vs_registration": sorted(f for f in blk if f in reg and blk[f] != reg[f]),
                              "added_vs_registration": sorted(f for f in blk if f not in reg), "missing": sorted(f for f in reg if f not in blk)}
# current files vs the last staged block (the scored run's)
last = blocks["2026-10-17T16:37:15Z"]
paths = {f: f"{R8}/{f}" for f in last if os.path.exists(f"{R8}/{f}")}
paths.update({"e1v4.py": f"{LI}/r7e1/e1v4.py", "e1v3.py": f"{LI}/r6e1/e1v3.py", "smallnet/netdef.py": f"{LI}/smallnet/netdef.py"})
for f in ("env6.py", "wm.py", "rollouts.py", "run.py", "partA2.py"):
    paths[f] = f"{LI}/r5e1/{f}"
for s in (101, 102, 103):
    paths[f"ckpt/s{s}/theta_real_v2.npz"] = f"{LI}/r7e1/ckpt/s{s}/theta_real_v2.npz"
rep["current_vs_scored_run_stage"] = {f: ("same" if sha(p) == last[f] else f"CHANGED (now {sha(p)[:8]}, mtime {mt(p)})") for f, p in sorted(paths.items())}
# registered e1v5.py = current minus the deviation-3 block?
L = open(f"{R8}/e1v5.py", "rb").read().split(b"\n")
rep["e1v5_registered_hash_reproduced_by_removing_lines_785_791"] = hashlib.sha256(b"\n".join(L[:784] + L[791:])).hexdigest() == reg["e1v5.py"]
rep["e1v5_lines_785_791"] = [x.decode() for x in L[784:791]]

# 4. amendments and spec
am = open(f"{OB}/experiments-round-8-amendments.md", "rb").read()
spec = open(f"{OB}/experiments-round-8-spec.md", "rb").read()
first45 = b"\n".join(am.split(b"\n")[:45]) + b"\n"
rep["amendments_file"] = {"mtime": mt(f"{OB}/experiments-round-8-amendments.md"), "first45_sha256": hashlib.sha256(first45).hexdigest(),
                          "first45_verbatim_in_registered_prereg": first45.decode().strip("\n") in "\n".join(a),
                          "other_round8_amendment_files": sorted(os.path.basename(p) for p in glob.glob(f"{OB}/experiments-round-8-amendments*"))}
rep["spec_file"] = {"mtime": mt(f"{OB}/experiments-round-8-spec.md"), "sha256": hashlib.sha256(spec).hexdigest()}
sl = spec.decode().split("\n")
i0 = next(i for i, l in enumerate(sl) if l.startswith("### 1.2 "))
ex = "\n".join(sl[i0:62]) + "\n"          # spec lines 48-62 (section 1.2 up to the rule separator)
rep["spec_1.2_excerpt_sha256"] = hashlib.sha256(ex.encode()).hexdigest()
rep["spec_1.2_verbatim_in_registered_prereg"] = ex.strip("\n") in "\n".join(a)

# 5. run 1 vs the scored run: identical experimental configuration? test-board numbers in run 1's logs?
c1 = json.load(open(f"{R8}/results/main-run1/s101/summary.json"))["cfg"]
c2 = json.load(open(f"{R8}/results/main/s101/summary.json"))["cfg"]
rep["run1_vs_scored_cfg_differences_s101"] = {k: (c1.get(k), c2.get(k)) for k in set(c1) | set(c2) if c1.get(k) != c2.get(k)}
r1 = {}
for s in (101, 102, 103):
    t1 = [l for l in open(f"/tmp/r8e1/logs/main-run1/s{s}/log.txt") if "roots: KL" in l or "K=0 identity" in l]
    t2 = [l for l in open(f"/tmp/r8e1/logs/main/s{s}/log.txt") if "roots: KL" in l or "K=0 identity" in l]
    r1[s] = {"run1_test_board_lines": [l[:150].rstrip() for l in t1],
             "identical_to_scored_run_after_timestamp": [x.split("] ", 1)[1] for x in t1] == [x.split("] ", 1)[1] for x in t2],
             "retry_log_has_traceback": "Traceback" in open(f"/tmp/r8e1/logs/main-run1/s{s}/stdout-retry.txt").read(),
             "retry_log_last_line": open(f"/tmp/r8e1/logs/main-run1/s{s}/stdout-retry.txt").read().strip().splitlines()[-1][:110],
             "scored_dir_stdout_retry_identical_to_run1": open(f"/tmp/r8e1/logs/main/s{s}/stdout-retry.txt", "rb").read()
             == open(f"/tmp/r8e1/logs/main-run1/s{s}/stdout-retry.txt", "rb").read(),
             "scored_rc_json": open(f"{R8}/results/main/s{s}/rc.json").read().strip()}
rep["run1"] = r1
print(json.dumps(rep, indent=1))
json.dump(rep, open(os.path.join(OUT, "a1_integrity.json"), "w"), indent=1)
