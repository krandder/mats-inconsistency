"""Render r6/RESULTS-R6-E3.md to /home/agent/memo-serve/lir-agent/r6-e3.html with tools/render.py, fix the
<title> and the meta line, embed the two report figures (copied to r6-e3-files/).  python3 r6/render_page.py"""
import datetime
import pathlib
import shutil
import subprocess

R6 = pathlib.Path("/home/agent/workspaces/mats/work/micro-stratego/r6")
OUT = pathlib.Path("/home/agent/memo-serve/lir-agent")
FILES = OUT / "r6-e3-files"
tmp = pathlib.Path("/tmp/r6e3/r6-e3.html")
subprocess.run(["python3", "/home/agent/workspaces/eurisko-study/tools/render.py", str(R6 / "RESULTS-R6-E3.md"), str(tmp)],
               check=True)
s = tmp.read_text()
s = s.replace("<title>EURISKO Study Report</title>", "<title>R6-E3: marginal window rollouts (E-MS-MARGINAL)</title>", 1)
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
s = s.replace('<p class="meta">Rendered from REPORT.md (2026-09-27). Numbers in brackets link to the Sources list.</p>',
              f'<p class="meta">Rendered {now} from micro-stratego/r6/RESULTS-R6-E3.md by r6/render_page.py. Every number '
              'comes from the files named next to it.</p>', 1)
FILES.mkdir(parents=True, exist_ok=True)
figs = [("e3_d0_d1.png", "E3-D0 gate KLs on calibration positions (4-plies-old anchor, 16 marginal rollouts, every hidden slot "
         "scored, one pass at the current turn) and E3-D1 belief JSD between left- and right-preferring search and KL to "
         "the exact posterior per pooling rule; 3-seed means with 95% intervals (data r6/report/figures.csv)."),
        ("e3_d2.png", "E3-D2 action-value MSE against Q* for uniform, entropy-weighted and trust-weighted pooling of 16 "
         "rollouts per action; 3-seed means with 95% intervals (data r6/report/figures.csv).")]
html = ""
for f, cap in figs:
    shutil.copy2(R6 / "report" / f, FILES / f)
    html += (f'<figure><img src="r6-e3-files/{f}" alt="{cap[:150]}" loading="lazy" style="max-width:100%;height:auto;'
             f'background:#fff;border:1px solid #ddd;border-radius:6px;padding:4px"><figcaption style="font-size:13.5px">'
             f'{cap}</figcaption></figure>\n')
anchor = '<section id="figures">\n<h2>Figures</h2>\n'
assert anchor in s
s = s.replace(anchor, anchor + html, 1)
assert "EURISKO Study Report" not in s
(OUT / "r6-e3.html").write_text(s)
print("published", OUT / "r6-e3.html", len(s), "bytes")
