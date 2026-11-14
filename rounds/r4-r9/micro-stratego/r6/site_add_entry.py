"""Add (or replace) the R6-E3 entry on the MATS /experiments site, staged like R4-E3a's (lir-agent/results-r2/
site_add_entry.py): one EXP row (above the round-4 E-MS row), one MD_NAMES entry, the name in the 'overnight' sub-stream
of 'Search over a learned world model' (right after the round-4 E-MS name) in calibrated-ai/experiments_page.py, and one
side entry in exp-pages-side.json. Takes .bak-r6e3-20261017 copies first; re-reads each file right before writing
(other agents edit them too). Input: r5/site-entry.json (date, name, status, oneline, led_to, page).
Does NOT publish: build with `python3 experiments_page.py` and link-check locally; sync_economy_tracks.sh is not run.
   python3 r5/site_add_entry.py"""
import json
import os
import shutil

SITE = "/home/agent/workspaces/calibrated-ai"
PY = os.path.join(SITE, "experiments_page.py")
SIDE = os.path.join(SITE, "exp-pages-side.json")
MD = "/home/agent/workspaces/mats/work/micro-stratego/r6/RESULTS-R6-E3.md"
E = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "site-entry.json")))
SLUG = "micro-stratego-r6-e3"
R4 = "Round 5 E-MS-6x6: 6x6 Barrage micro-Stratego with preference-guided search (stopped at the gate)"

for f in (PY, SIDE):
    b = f + ".bak-r6e3-20261017"
    if not os.path.exists(b):
        shutil.copy2(f, b)

s0 = open(PY).read()
s = "".join(l for l in s0.splitlines(True) if ("md:" + MD) not in l)
row = "    (%r, %r, %r, %r, %r),\n" % (E["date"], E["name"], E["status"], E["oneline"], "md:" + MD)
anchor = s.index("    ('17 Oct', " + repr(R4))
s = s[:anchor] + row + s[anchor:]
if "'micro-stratego/r6/RESULTS-R6-E3.md'" not in s:
    s = s.replace("MD_NAMES = {  # side-track write-ups outside DOCS -> published file stem\n",
                  "MD_NAMES = {  # side-track write-ups outside DOCS -> published file stem\n"
                  "    'micro-stratego/r6/RESULTS-R6-E3.md': 'micro-stratego-r6-e3-results',\n", 1)
sub_anchor = "                      %r,\n" % R4
assert s.count(sub_anchor) == 1, "sub-stream anchor not unique"
if ("                      %r,\n" % E["name"]) not in s:
    s = s.replace(sub_anchor, sub_anchor + ("                      %r,\n" % E["name"]), 1)
assert open(PY).read() == s0, "experiments_page.py changed while building the edit; rerun"
open(PY, "w").write(s)

t0 = open(SIDE).read()
d = json.loads(t0)
d["experiments"] = [x for x in d["experiments"] if x.get("slug") != SLUG]
t = json.dumps(d, indent=1, ensure_ascii=False) if len(d["experiments"]) != len(json.loads(t0)["experiments"]) else t0
entry = json.dumps({"slug": SLUG, "title": E["name"], "led_to": E.get("led_to", []), "page": E["page"]}, indent=1,
                   ensure_ascii=False)
entry = "\n".join("  " + l for l in entry.splitlines())
k = t.rfind("\n ]")
t2 = t[:k] + ",\n" + entry + t[k:]
assert json.loads(t2)["experiments"][-1]["slug"] == SLUG
assert open(SIDE).read() == t0, "exp-pages-side.json changed while building the edit; rerun"
open(SIDE, "w").write(t2)
print("ok: EXP row, MD_NAMES, STREAMS(overnight) and side entry", SLUG)
