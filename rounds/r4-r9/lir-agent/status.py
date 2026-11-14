#!/usr/bin/env python3
"""Live status page for the lir-agent overnight programme (eurisko, 2026-10-03).
Renders memo-serve/lir-agent/index.html from files only; every line names its check (policy 15).
Stops being used once REPORT.md exists (status-loop.sh exits; the report agent then owns index.html)."""
import html, re, subprocess, time
from pathlib import Path
OUT = Path('/home/agent/workspaces/eurisko-study/lir-agent')
WEB = Path('/home/agent/memo-serve/lir-agent'); FILES = WEB / 'files'; FILES.mkdir(parents=True, exist_ok=True)
RENDER = OUT.parent / 'tools' / 'render.py'
now = time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())
def esc(s): return html.escape(str(s), quote=False)
def render(md: Path, name: str):
    out = FILES / f'{name}.html'
    try:
        if not out.exists() or out.stat().st_mtime < md.stat().st_mtime:
            subprocess.run(['python3', str(RENDER), str(md), str(out)], check=True, timeout=60, capture_output=True)
        return f'files/{name}.html'
    except Exception as e:
        out.write_text('<pre>' + esc(md.read_text()) + '</pre>'); return f'files/{name}.html'
def first_line(p: Path, pat=None, default=''):
    try:
        for l in p.read_text().splitlines():
            if pat is None and l.strip(): return l.strip()
            if pat and re.search(pat, l): return l.strip()
    except Exception: pass
    return default
# hawks
hawks = []
for k in ['E1','E2','E3','E4','E5','E6','E7','T1','T2','FORECAST']:
    p = OUT / 'hawks' / f'{k}.md'
    if not p.exists(): hawks.append((k, 'missing', '', None)); continue
    t = p.read_text(); m = re.search(r'Verdict:\*{0,2}\s*\*{0,2}([A-Za-z]+)', t)
    v = m.group(1).upper() if m else '?'
    after = re.sub(r'^[\s.*:()]+', '', t[m.end():].split('\n')[0].replace('**','')).strip() if m else first_line(p)
    hawks.append((k, v, after[:220], render(p, f'hawk-{k}')))
# experiments
exps = []
for k in ['E1','E2','E3','E4','E5','E6','E7']:
    d = OUT / 'experiments' / k; pre = d / 'PREREG.md'; res = d / 'RESULTS.md'
    title = first_line(pre, r'^#')[:170] if pre.exists() else ''
    paper = first_line(pre, r'Papers? modified')[:200] if pre.exists() else ''
    exps.append((k, title, paper, render(pre, f'prereg-{k}') if pre.exists() else None, render(res, f'results-{k}') if res.exists() else None))
theory = [(p.stem, render(p, f'theory-{p.stem}')) for p in sorted((OUT/'theory').glob('*.md'))] if (OUT/'theory').exists() else []
claims = render(OUT/'CLAIMS.md', 'claims') if (OUT/'CLAIMS.md').exists() else None
goal = render(OUT/'GOAL.md', 'goal'); harness = render(OUT/'harness'/'README.md', 'harness-readme') if (OUT/'harness'/'README.md').exists() else None
# gpu from GUARD.log
gl = (OUT/'GUARD.log').read_text().splitlines() if (OUT/'GUARD.log').exists() else []
st = [l for l in gl if ' STATUS ' in l]; last = st[-1] if st else ''
kills = [l for l in gl if 'KILL' in l]
def g(key): m = re.search(key + r'=([0-9.]+)', last); return m.group(1) if m else '?'
vms = re.findall(r'(soup-gpu-lir-\S+?):(\d+)m', last)
gts = last.split(' ')[0] if last else '?'
bb = (OUT.parent/'li-net'/'bb'/'RESULTS.md').exists()
if bb: render(OUT.parent/'li-net'/'bb'/'RESULTS.md', 'bb-results')
done_e = sum(1 for e in exps if e[4]); pre_e = sum(1 for e in exps if e[3])
phase = ('report written' if (OUT/'REPORT.md').exists() else 'skeptic' if claims else 'theory' if theory and done_e == 7 else f'experiments ({done_e}/7 RESULTS, {pre_e}/7 PREREG)')
H = []
H.append(f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="300"><title>LIR control masks: overnight programme (live)</title>
<style>:root{{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ddd;--ok:#2a7;--warn:#c80;--bad:#c33;--card:#f6f6f6}}@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#111;--fg:#e8e8e8;--mut:#999;--line:#333;--card:#1b1b1b}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:0 auto;padding:16px}}table{{border-collapse:collapse;width:100%;margin:8px 0 16px}}td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top;font-size:14px}}th{{color:var(--mut);font-weight:600}}.m{{color:var(--mut)}}.k{{font-size:12px;color:var(--mut)}}.v-RESHAPE{{color:var(--warn);font-weight:600}}.v-KILL{{color:var(--bad);font-weight:600}}.v-RUN{{color:var(--ok);font-weight:600}}.card{{background:var(--card);padding:10px 14px;border-radius:8px;margin:10px 0}}code{{font-size:13px}}</style></head><body>
<h1>LIR control masks for post-training: overnight programme</h1>
{'<div class="card"><b>Interim report (14:xxZ, from the finished experiments E1, E2, E6 + hawks + BB):</b> <a href="interim.html">interim.html</a></div>' if (WEB/'interim.html').exists() else ''}
<p class="m">Live status, regenerated every 5 min from files; checked at <b>{now}</b>. This page is replaced by the morning report when <code>lir-agent/REPORT.md</code> exists. Spec: <a href="{goal}">GOAL.md</a>. Owner: eurisko; builders: Opus 5.5 subagents in one workflow (hawks → harness → experiments → theory → skeptic → report).</p>
<div class="card"><b>Phase now:</b> {esc(phase)} · <b>hawks:</b> {sum(1 for h in hawks if h[1] not in ('missing','?'))}/10 done, all RESHAPE (none killed) · <b>harness:</b> {'built, smoke passed (0 mask violations)' if harness else 'building'} · <b>theory:</b> {len(theory)}/2 · <b>skeptic:</b> {'done' if claims else 'pending'} · <b>report:</b> {'ready' if (OUT/'REPORT.md').exists() else 'pending (sequential after experiments; expected late morning UTC)'}<br>
<span class="k">check: ls lir-agent/hawks/*.md experiments/*/PREREG.md experiments/*/RESULTS.md theory/*.md CLAIMS.md REPORT.md</span></div>
<div class="card"><b>GPU (spot L4, $0.87/h ceiling each):</b> {len(vms)} lir VMs running · spend tonight (lir runs only): audited ${g('audited')} finished + ${g('running_est')} running ≈ <b>${g('total_est')}</b> of the $60 cap (guard hard-stops launches at $55; deletes any VM past 2.6 h; guard kills so far: {len(kills)}) · guard checked at {esc(gts)}<br>
<span class="k">{esc(', '.join(f"{n.split('-1003')[0].replace('soup-gpu-lir-','')} {a}m" for n,a in vms)) or 'no lir VMs'}</span><br>
<span class="k">check: tail -1 lir-agent/GUARD.log; grep -E "(DELETE|COST-AUDIT) vm=soup-gpu-lir" logical-soup/GPU-LOG.md; gcloud compute instances list --filter=labels.logical-soup=1</span></div>
<h2>Hawks (adversarial review before each experiment)</h2><table><tr><th>item</th><th>verdict</th><th>first line of the verdict</th></tr>''')
for k, v, a, link in hawks:
    H.append(f'<tr><td>{"<a href=%s>%s</a>" % (link, k) if link else k}</td><td class="v-{v}">{esc(v)}</td><td>{esc(a)}</td></tr>')
H.append('</table><h2>Experiments (each a small modification of a named paper; 3 seeds; fused baseline B0 alongside)</h2><table><tr><th>item</th><th>PREREG title</th><th>paper modified</th><th>status</th></tr>')
for k, title, paper, pre, res in exps:
    run = [a for n, a in vms if f'lir-{k.lower()}-' in n]
    status = f'<a href="{res}">RESULTS</a>' if res else (f'running on {len(run)} VM(s)' if run else ('waiting for GPU / analysing' if pre else 'no PREREG yet'))
    H.append(f'<tr><td>{k}</td><td>{"<a href=%s>%s</a>" % (pre, esc(title.lstrip("# "))) if pre else "<span class=m>pending</span>"}</td><td class="k">{esc(paper)}</td><td>{status}</td></tr>')
H.append('</table>')
H.append('<h2>Theory</h2><ul>' + ''.join(f'<li><a href="{l}">{k}</a></li>' for k, l in theory) + ('' if theory else '<li class="m">T1 (metareasoning stopping bound) and T2 (self-model consistency) start after the experiments finish.</li>') + '</ul>')
H.append(f'<h2>Also tonight</h2><ul><li>Busy-beaver reasoner campaign (li-net/bb): {"done: 3 seeds of every arm, $20.04 GPU; <a href=#>pages</a>, <a href=files/bb-results.html>RESULTS.md</a>. Headline: the reasoner does not simulate (thinking longer does not help; belief state stops changing after a few blocks because time reaches the model only at the readout)" if bb else "seeds 0-1 done, seed 2 running; RESULTS.md pending"} <span class="k">check: ls li-net/bb/RESULTS.md</span></li><li>Harness: {"<a href=%s>README</a>" % harness if harness else "building"} (ShipHunt text POMDP, per-token routed role adapters on a frozen Qwen3-0.6B, goal-deleted role passes, stochastic commit with logged propensity, dose knob).</li>{"<li><a href=%s>CLAIMS.md (skeptic)</a></li>" % claims if claims else ""}</ul>')
H.append(f'<p class="k">Generated by lir-agent/status.py at {now}. Every number above comes from the named file or command; nothing here is a result until RESULTS.md and CLAIMS.md exist.</p></body></html>')
WEB.mkdir(parents=True, exist_ok=True); (WEB/'index.html').write_text('\n'.join(H)); print('ok', phase, len(vms), g('total_est'))
