"""Add R7-E1 (E-INFO-V4) to the MATS /experiments site, additive only (same pattern as micro-stratego/site_patch.py): one EXP
row, one MD_NAMES entry and the name in the 'Rollout aggregation' sub-stream of the 'Search over a learned world model' stream
in calibrated-ai/experiments_page.py, and one side entry in exp-pages-side.json. Refuses to run unless the
.bak-r7e1-20261017 backups exist; idempotent (re-running replaces this experiment's row and entry).
   python3 site_patch.py site-entry.json
Do not run sync_economy_tracks.sh from here; build locally with experiments_page.py and link-check (see RESULTS section 13)."""
import json
import pathlib
import sys

CA = pathlib.Path("/home/agent/workspaces/calibrated-ai")
PY, SIDE = CA / "experiments_page.py", CA / "exp-pages-side.json"
for p in (PY, SIDE):
    assert pathlib.Path(str(p) + ".bak-r7e1-20261017").exists(), f"backup missing for {p}"
e = json.loads(pathlib.Path(sys.argv[1]).read_text())
name, slug = e["title"], e["slug"]
MDP = "/home/agent/workspaces/eurisko-study/li-net/r7e1/RESULTS-R7-E1.md"
src = PY.read_text()
row = repr((e["date"], name, e["status"], e["oneline"], "md:" + MDP))
lines = [l for l in src.split("\n") if ("md:" + MDP) not in l]
i = lines.index("EXP = [")
lines.insert(i + 1, f"    {row},")
src = "\n".join(lines)
key = "'li-net/r7e1/RESULTS-R7-E1.md': 'r7-e1-results',"
if key not in src:
    anchor = "MD_NAMES = {  # side-track write-ups outside DOCS -> published file stem\n"
    assert anchor in src
    src = src.replace(anchor, anchor + "    " + key + "\n", 1)
ag = src.index("dict(key='aggregation', title='Rollout aggregation',")
nm_start = src.index("names=[", ag)
nm_end = src.index("]),", nm_start)
block = src[nm_start:nm_end]
if repr(name) not in block:
    anchor = "names=[\n"
    assert block.startswith(anchor), "aggregation sub-stream anchor not found"
    block = anchor + "                      " + repr(name) + ",\n" + block[len(anchor):]
    src = src[:nm_start] + block + src[nm_end:]
PY.write_text(src)
txt = SIDE.read_text()
data = json.loads(txt)
assert json.dumps(data, indent=1, ensure_ascii=False) == txt.rstrip("\n"), "side JSON is not a plain indent=1 dump; refusing"
ent = {"slug": slug, "title": name, "led_to": e.get("led_to", []), "page": e["page"]}
data["experiments"] = [x for x in data["experiments"] if x.get("slug") != slug] + [ent]
SIDE.write_text(json.dumps(data, indent=1, ensure_ascii=False))
print("patched", PY, SIDE)
