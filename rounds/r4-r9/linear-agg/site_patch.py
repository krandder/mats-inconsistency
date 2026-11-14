"""Add the linear-agg experiment to the MATS /experiments site (additive only): one EXP row, one MD_NAMES entry and the
track list entry in calibrated-ai/experiments_page.py, and one side entry in exp-pages-side.json. Refuses to run unless
the .bak-20261016-la backups exist; idempotent (re-running replaces the linear-agg row/entry).
   python3 site_patch.py site-entry.json"""
import json
import pathlib
import sys

CA = pathlib.Path("/home/agent/workspaces/calibrated-ai")
PY, SIDE = CA / "experiments_page.py", CA / "exp-pages-side.json"
for p in (PY, SIDE):
    assert pathlib.Path(str(p) + ".bak-20261016-la").exists(), f"backup missing for {p}"
e = json.loads(pathlib.Path(sys.argv[1]).read_text())
name, slug = e["title"], e["slug"]
src = PY.read_text()
row = repr((e["date"], name, e["status"], e["oneline"], "md:" + e["results_md"]))
lines = src.split("\n")
lines = [l for l in lines if "md:/home/agent/workspaces/mats/work/linear-agg/RESULTS-LA.md" not in l]
i = lines.index("EXP = [")
lines.insert(i + 1, f"    {row},")
src = "\n".join(lines)
key = "'linear-agg/RESULTS-LA.md': 'linear-agg-results', "
if key not in src:
    src = src.replace("MD_NAMES = {  # side-track write-ups outside DOCS -> published file stem\n",
                      "MD_NAMES = {  # side-track write-ups outside DOCS -> published file stem\n    " + key.rstrip() + "\n", 1)
anchor = "'Overnight E-CA round 3: a zero-overlap CA grid and a depth gate', "
tr_start = src.index("    ('linet', ")
tr_end = src.index("]),", tr_start)
block = src[tr_start:tr_end]
if repr(name) not in block:
    assert anchor in block, "linet track anchor not found"
    block = block.replace(anchor, anchor + repr(name) + ", ", 1)
    src = src[:tr_start] + block + src[tr_end:]
PY.write_text(src)
txt = SIDE.read_text()
ent = {"slug": slug, "title": name, "led_to": e.get("led_to", []), "page": e["page"]}
body = "\n".join("  " + l for l in json.dumps(ent, indent=1, ensure_ascii=False).split("\n"))
start = txt.find('\n  {\n   "slug": "' + slug + '",')
if start >= 0:                                     # replace our previous entry (it is always the last one we appended)
    end = txt.index("\n ]\n}", start)
    txt = txt[:start].rstrip().rstrip(",") + txt[end:]
    txt = txt.rstrip()
    assert txt.endswith("]\n}")
tail = "\n ]\n}"
k = txt.rstrip().rindex(tail)
txt = txt[:k] + ",\n" + body + tail + "\n"
json.loads(txt)
SIDE.write_text(txt)
print("patched", PY, SIDE)
