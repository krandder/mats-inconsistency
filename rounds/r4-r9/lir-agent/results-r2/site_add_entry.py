"""Add (or replace) the R4-E3a entry on the MATS /experiments site, in the same format as the E-CA entries.
Input: results-r2/site-entry.json with keys date, name, status, oneline, page (why/what/result/sowhat/decided/chart/details).
Edits calibrated-ai/experiments_page.py (EXP row, MD_NAMES, STREAMS sub-stream 'lir') and exp-pages-side.json (side entry),
after taking .bak-r4e3a-20261016 copies; re-reads each file right before writing (other agents edit them too)."""
import json
import os
import re
import shutil
import sys

SITE = "/home/agent/workspaces/calibrated-ai"
PY = os.path.join(SITE, "experiments_page.py")
SIDE = os.path.join(SITE, "exp-pages-side.json")
MD = "/home/agent/workspaces/eurisko-study/lir-agent/RESULTS-R2.md"
E = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "site-entry.json")))
SLUG = "lir-r4-e3a"

for f in (PY, SIDE):
    b = f + ".bak-r4e3a-20261016"
    if not os.path.exists(b):
        shutil.copy2(f, b)

s = open(PY).read()
row = "    (%r, %r, %r, %r, %r),\n" % (E["date"], E["name"], E["status"], E["oneline"], "md:" + MD)
# drop an older version of our row, then insert ours right above the first LIR round-1 row
s = "".join(l for l in s.splitlines(True) if ("md:" + MD) not in l)
anchor = s.index("    ('3 Oct', 'Walling reward off from beliefs under RL")
s = s[:anchor] + row + s[anchor:]
if "'lir-agent/RESULTS-R2.md'" not in s:
    s = s.replace("MD_NAMES = {  # side-track write-ups outside DOCS -> published file stem\n",
                  "MD_NAMES = {  # side-track write-ups outside DOCS -> published file stem\n    'lir-agent/RESULTS-R2.md': 'lir-r4-e3a-results',\n", 1)
sub_anchor = "                      'Walling reward off from beliefs under RL (LIR E1, ShipHunt)',\n"
assert s.count(sub_anchor) == 1, "sub-stream anchor not unique"
if ("                      %r,\n" % E["name"]) not in s:
    s = s.replace(sub_anchor, ("                      %r,\n" % E["name"]) + sub_anchor, 1)
cur = open(PY).read()   # re-read: abort if someone changed it while we built ours
assert cur == open(PY).read()
open(PY, "w").write(s)

t = open(SIDE).read()
assert SLUG not in [x.get("slug") for x in json.loads(t)["experiments"]], "entry exists: restore the .bak first"
entry = json.dumps({"slug": SLUG, "title": E["name"], "led_to": E.get("led_to", []), "page": E["page"]},
                   indent=1, ensure_ascii=False)
entry = "\n".join("  " + l for l in entry.splitlines())
k = t.rfind("\n ]")
t2 = t[:k] + ",\n" + entry + t[k:]
assert json.loads(t2)["experiments"][-1]["slug"] == SLUG
open(SIDE, "w").write(t2)
print("ok: EXP row, MD_NAMES, STREAMS(lir) and side entry", SLUG)
