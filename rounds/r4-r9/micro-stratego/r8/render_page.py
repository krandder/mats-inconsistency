"""Render r8/RESULTS-R8-E3.md to /home/agent/memo-serve/lir-agent/r8-e3.html with tools/render.py, fix the
<title> and the meta line, embed the two report figures (copied to r8-e3-files/).  python3 r8/render_page.py"""
import datetime
import pathlib
import shutil
import subprocess

R8 = pathlib.Path("/home/agent/workspaces/mats/work/micro-stratego/r8")
OUT = pathlib.Path("/home/agent/memo-serve/lir-agent")
FILES = OUT / "r8-e3-files"
tmp = pathlib.Path("/tmp/r8e3/r8-e3.html")
subprocess.run(["python3", "/home/agent/workspaces/eurisko-study/tools/render.py", str(R8 / "RESULTS-R8-E3.md"), str(tmp)],
               check=True)
s = tmp.read_text()
s = s.replace("<title>EURISKO Study Report</title>", "<title>R8-E3: current-turn anchor safety (E-MS-SAFE-ANCHOR)</title>", 1)
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
s = s.replace('<p class="meta">Rendered from REPORT.md (2026-09-27). Numbers in brackets link to the Sources list.</p>',
              f'<p class="meta">Rendered {now} from micro-stratego/r8/RESULTS-R8-E3.md by r8/render_page.py. Every number '
              'comes from the files named next to it.</p>', 1)
FILES.mkdir(parents=True, exist_ok=True)
figs = [("e8_d1.png", "E3-D1: KL to the exact posterior against the number of rollouts K, every aggregator anchored at the "
         "current-turn one-pass belief (K = 0); left the learned and scaled aggregators, right the two naive baselines "
         "(log scale); 3-seed means with 95% intervals (data r8/report/figures.csv)."),
        ("e8_d2.png", "E3-D2: action-value MSE against Q* on clean determinization bags (16 rollouts) and on mixed bags "
         "(10 clean + 6 epistemically noisy rollouts) per pooling rule; 3-seed means with 95% intervals "
         "(data r8/report/figures.csv).")]
html = ""
for f, cap in figs:
    shutil.copy2(R8 / "report" / f, FILES / f)
    html += (f'<figure><img src="r8-e3-files/{f}" alt="{cap[:150]}" loading="lazy" style="max-width:100%;height:auto;'
             f'background:#fff;border:1px solid #ddd;border-radius:6px;padding:4px"><figcaption style="font-size:13.5px">'
             f'{cap}</figcaption></figure>\n')
anchor = '<section id="figures">\n<h2>Figures</h2>\n'
assert anchor in s
s = s.replace(anchor, anchor + html, 1)
assert "EURISKO Study Report" not in s
(OUT / "r8-e3.html").write_text(s)
print("published", OUT / "r8-e3.html", len(s), "bytes")
