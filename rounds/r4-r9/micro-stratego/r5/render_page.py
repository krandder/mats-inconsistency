"""Render RESULTS-R5-E2.md to /home/agent/memo-serve/lir-agent/r5-e2.html with tools/render.py, fix the <title>
and the meta line, embed the three report figures (copied to r5-e2-files/).  python3 r5/render_page.py"""
import datetime
import pathlib
import shutil
import subprocess

MS = pathlib.Path("/home/agent/workspaces/mats/work/micro-stratego")
OUT = pathlib.Path("/home/agent/memo-serve/lir-agent")
FILES = OUT / "r5-e2-files"
tmp = pathlib.Path("/tmp/r5e2/r5-e2.html")
subprocess.run(["python3", "/home/agent/workspaces/eurisko-study/tools/render.py", str(MS / "RESULTS-R5-E2.md"), str(tmp)],
               check=True)
s = tmp.read_text()
s = s.replace("<title>EURISKO Study Report</title>", "<title>R5-E2: 6x6 micro-Stratego search (E-MS-6x6)</title>", 1)
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
s = s.replace('<p class="meta">Rendered from REPORT.md (2026-09-27). Numbers in brackets link to the Sources list.</p>',
              f'<p class="meta">Rendered {now} from micro-stratego/RESULTS-R5-E2.md by r5/render_page.py. Every number comes '
              'from the files named next to it.</p>', 1)
FILES.mkdir(parents=True, exist_ok=True)
figs = [("d0_by_turn.png", "One-pass KL to the exact posterior per hidden slot by turn, both nets (3-seed means), with the "
         "constraint-only oracle's distance and the 0.05 gate bar (data r5/report/d0_by_turn.csv)."),
        ("d1_aggregators.png", "EXPLORATORY D1: policy sensitivity (8-slot belief, value) and belief KL to the K = 128 reference "
         "per aggregator, primary net, 3-seed means with 95% intervals (data r5/report/d1_aggregators.csv)."),
        ("train_curves.png", "Calibration-split losses of theta_real during training, 3 seeds x 2 nets (data r5/report/train_curves.csv).")]
html = ""
for f, cap in figs:
    shutil.copy2(MS / "r5" / "report" / f, FILES / f)
    html += (f'<figure><img src="r5-e2-files/{f}" alt="{cap[:150]}" loading="lazy" style="max-width:100%;height:auto;'
             f'background:#fff;border:1px solid #ddd;border-radius:6px;padding:4px"><figcaption style="font-size:13.5px">'
             f'{cap}</figcaption></figure>\n')
anchor = '<section id="figures">\n<h2>Figures</h2>\n'
assert anchor in s
s = s.replace(anchor, anchor + html, 1)
assert "EURISKO Study Report" not in s
(OUT / "r5-e2.html").write_text(s)
print("published", OUT / "r5-e2.html", len(s), "bytes;", len(figs), "figures")
