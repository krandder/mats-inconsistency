"""Add the E-MS (R4-E1 micro-Stratego) experiment to the MATS /experiments site (additive only): one EXP row, one MD_NAMES
entry and the name in the 'Overnight search experiments' sub-stream of the 'Search over a learned world model' stream in
calibrated-ai/experiments_page.py, and one side entry in exp-pages-side.json. Refuses to run unless the
.bak-20261016-ms backups exist; idempotent (re-running replaces this experiment's row and entry).
   python3 site_patch.py site-entry.json"""
import json
import pathlib
import sys

CA = pathlib.Path("/home/agent/workspaces/calibrated-ai")
PY, SIDE = CA / "experiments_page.py", CA / "exp-pages-side.json"
for p in (PY, SIDE):
    assert pathlib.Path(str(p) + ".bak-20261016-ms").exists(), f"backup missing for {p}"
e = json.loads(pathlib.Path(sys.argv[1]).read_text())
name, slug = e["title"], e["slug"]
MDP = "/home/agent/workspaces/mats/work/micro-stratego/RESULTS-MS.md"
src = PY.read_text()
row = repr((e["date"], name, e["status"], e["oneline"], "md:" + MDP))
lines = src.split("\n")
lines = [l for l in lines if ("md:" + MDP) not in l]
i = lines.index("EXP = [")
lines.insert(i + 1, f"    {row},")
src = "\n".join(lines)
key = "'micro-stratego/RESULTS-MS.md': 'micro-stratego-results',"
if key not in src:
    anchor = "MD_NAMES = {  # side-track write-ups outside DOCS -> published file stem\n"
    assert anchor in src
    src = src.replace(anchor, anchor + "    " + key + "\n", 1)
# the sub-stream list: insert our name after the E-CA round 3 entry inside dict(key='overnight', ...)
ov = src.index("dict(key='overnight', title='Overnight search experiments',")
nm_start = src.index("names=[", ov)
nm_end = src.index("]),", nm_start)
block = src[nm_start:nm_end]
if repr(name) not in block:
    anchor = "'Overnight E-CA round 3: a zero-overlap CA grid and a depth gate',\n"
    assert anchor in block, "overnight sub-stream anchor not found"
    block = block.replace(anchor, anchor + "                      " + repr(name) + ",\n", 1)
    src = src[:nm_start] + block + src[nm_end:]
PY.write_text(src)
txt = SIDE.read_text()
ent = {"slug": slug, "title": name, "led_to": e.get("led_to", []), "page": e["page"]}
body = "\n".join("  " + l for l in json.dumps(ent, indent=1, ensure_ascii=False).split("\n"))
start = txt.find('\n  {\n   "slug": "' + slug + '",')
if start >= 0:
    end = txt.index("\n]\n}", start)
    txt = txt[:start].rstrip().rstrip(",") + txt[end:]
    txt = txt.rstrip()
    assert txt.endswith("]\n}")
tail = "\n]\n}"
k = txt.rstrip().rindex(tail)
txt = txt[:k] + ",\n" + body + tail + "\n"
json.loads(txt)
SIDE.write_text(txt)
print("patched", PY, SIDE)
