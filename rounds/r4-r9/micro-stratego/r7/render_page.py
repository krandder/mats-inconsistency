"""Render r7/RESULTS-R7-E3.md to /home/agent/memo-serve/lir-agent/r7-e3.html with tools/render.py, fix the
<title> and the meta line, embed the two report figures (copied to r7-e3-files/).  python3 r7/render_page.py"""
import datetime
import pathlib
import shutil
import subprocess

R7 = pathlib.Path("/home/agent/workspaces/mats/work/micro-stratego/r7")
OUT = pathlib.Path("/home/agent/memo-serve/lir-agent")
FILES = OUT / "r7-e3-files"
tmp = pathlib.Path("/tmp/r7e3/r7-e3.html")
subprocess.run(["python3", "/home/agent/workspaces/eurisko-study/tools/render.py", str(R7 / "RESULTS-R7-E3.md"), str(tmp)],
               check=True)
s = tmp.read_text()
s = s.replace("<title>EURISKO Study Report</title>", "<title>R7-E3: trust gating and salience keys (E-MS-TRUST-SEARCH)</title>", 1)
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
s = s.replace('<p class="meta">Rendered from REPORT.md (2026-09-27). Numbers in brackets link to the Sources list.</p>',
              f'<p class="meta">Rendered {now} from micro-stratego/r7/RESULTS-R7-E3.md by r7/render_page.py. Every number '
              'comes from the files named next to it.</p>', 1)
FILES.mkdir(parents=True, exist_ok=True)
figs = [("e7_d1.png", "E3-D1: key cosine gap within vs across flanks for R6-E3's K4 and this run's K3 and K3c (bar 0.28), "
         "and belief JSD between left- and right-preferring search and KL to the exact posterior per pooling rule; 3-seed "
         "means with 95% intervals (data r7/report/figures.csv)."),
        ("e7_d2.png", "E3-D2: action-value MSE against Q* on mixed-reliability bags (8 clean + 8 exploratory rollouts per "
         "action) for uniform, stiffness (gamma 2), trust-weighted, trust-gated and trust-resampled pooling, with the "
         "clean-only mean as a reference; 3-seed means with 95% intervals (data r7/report/figures.csv).")]
html = ""
for f, cap in figs:
    shutil.copy2(R7 / "report" / f, FILES / f)
    html += (f'<figure><img src="r7-e3-files/{f}" alt="{cap[:150]}" loading="lazy" style="max-width:100%;height:auto;'
             f'background:#fff;border:1px solid #ddd;border-radius:6px;padding:4px"><figcaption style="font-size:13.5px">'
             f'{cap}</figcaption></figure>\n')
anchor = '<section id="figures">\n<h2>Figures</h2>\n'
assert anchor in s
s = s.replace(anchor, anchor + html, 1)
assert "EURISKO Study Report" not in s
(OUT / "r7-e3.html").write_text(s)
print("published", OUT / "r7-e3.html", len(s), "bytes")
