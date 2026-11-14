#!/usr/bin/env python3
"""Inline SVG figures for IDEAS-NET (reads committed study files; no deps)."""
import csv, json, base64, html
from pathlib import Path
S = Path(__file__).resolve().parents[2]  # study/

def svg_open(w, h, label):
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(label)}" '
            f'style="width:100%;max-width:{w}px;height:auto;font:12px system-ui,sans-serif">')

def fig_toy():
    rows = list(csv.DictReader(open(S/'M1/lead/results/toy_closed_form.csv')))
    summ = json.load(open(S/'M1/lead/results/toy_summary.json'))['closed_form']
    lo, hi = summ['disagree_interval']
    W, H, L, R, T, B = 640, 300, 52, 16, 14, 40
    xs = [float(r['w']) for r in rows]
    x0, x1, y0, y1 = min(xs), max(xs), -1.05, 0.2
    X = lambda x: L + (x - x0) / (x1 - x0) * (W - L - R)
    Y = lambda y: T + (y1 - y) / (y1 - y0) * (H - T - B)
    o = [svg_open(W, H, 'Toy: gradient credit vs ablation value')]
    o.append(f'<rect x="{X(lo):.1f}" y="{T}" width="{X(hi)-X(lo):.1f}" height="{H-T-B}" fill="#e0a030" opacity="0.18"/>')
    o.append(f'<text x="{(X(lo)+X(hi))/2:.1f}" y="{T+14}" text-anchor="middle" fill="var(--warn)">credit &lt; 0 &lt; value</text>')
    for yv in (-1, -0.5, 0):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" stroke="var(--line)"/>'
                 f'<text x="{L-6}" y="{Y(yv)+4:.1f}" text-anchor="end" fill="var(--muted)">{yv:g}</text>')
    for xv in range(0, 5):
        o.append(f'<text x="{X(xv):.1f}" y="{H-B+16}" text-anchor="middle" fill="var(--muted)">{xv}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-6}" text-anchor="middle" fill="var(--muted)">readout weight w (true coefficient 1.0)</text>')
    for key, col, name in (('credit', '#c0392b', 'gradient credit C = -w L\'(w)'), ('value', '#2457c5', 'ablation value V = L(0) - L(w)')):
        pts = ' '.join(f'{X(float(r["w"])):.1f},{Y(float(r[key])):.1f}' for r in rows)
        o.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2"/>')
    o.append(f'<text x="{L+10}" y="{Y(-0.8):.1f}" fill="#c0392b">gradient credit C = -w L\'(w)</text>')
    o.append(f'<text x="{L+10}" y="{Y(-0.92):.1f}" fill="#2457c5">ablation value V = L(0) - L(w)</text>')
    o.append('</svg>')
    return ''.join(o)

def fig_census():
    want = [('books_cum', 'books, all-time'), ('books_hl10000', 'books, half-life 10k'),
            ('books_hl1000', 'books, half-life 1k'), ('books_win2000', 'books, window 2k'),
            ('settle_loo_heldout', 'LOO settlement, held-out'), ('settle_loo_train', 'LOO settlement, train')]
    d = {}
    for r in csv.DictReader(open(S/'M1/lead/results/sign_census_summary.csv')):
        if r['source'] == 'wealth-net final' and r['unit'] == 'neuron' and r['unit_set'] in ('all', 'top5pct'):
            d[(r['currency'], r['unit_set'])] = (float(r['frac_neg_mean']), float(r['frac_neg_sd']))
    W, rowh, L = 640, 34, 190
    H = 30 + rowh * len(want) + 24
    sx = lambda v: L + v * (W - L - 20)
    o = [svg_open(W, H, 'Share of neurons with negative account')]
    for v in (0, 0.25, 0.5):
        o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="22" y2="{H-24}" stroke="var(--line)"/>'
                 f'<text x="{sx(v):.1f}" y="{H-8}" text-anchor="middle" fill="var(--muted)">{v:.0%}</text>')
    o.append(f'<rect x="{L}" y="4" width="12" height="10" fill="#c0392b"/><text x="{L+16}" y="13" fill="var(--fg)">top 5% by oracle (26 neurons)</text>'
             f'<rect x="{L+210}" y="4" width="12" height="10" fill="var(--muted)" opacity="0.55"/><text x="{L+226}" y="13" fill="var(--fg)">all 512 neurons</text>')
    for i, (k, name) in enumerate(want):
        y = 28 + i * rowh
        o.append(f'<text x="{L-8}" y="{y+17}" text-anchor="end" fill="var(--fg)">{name}</text>')
        m, sd = d[(k, 'top5pct')]; ma, _ = d[(k, 'all')]
        o.append(f'<rect x="{L}" y="{y+2}" width="{sx(m)-L:.1f}" height="12" fill="#c0392b"/>')
        o.append(f'<line x1="{sx(max(0,m-sd)):.1f}" x2="{sx(m+sd):.1f}" y1="{y+8}" y2="{y+8}" stroke="var(--fg)"/>')
        o.append(f'<text x="{sx(m+sd)+5:.1f}" y="{y+12}" fill="var(--fg)">{m:.3f}</text>')
        o.append(f'<rect x="{L}" y="{y+16}" width="{max(1,sx(ma)-L):.1f}" height="9" fill="var(--muted)" opacity="0.55"/>')
    o.append('</svg>')
    return ''.join(o)

def fig_spearman():
    want = [('cum', 'all-time gradient books'), ('gradnorm', 'gradient norm (baseline)'),
            ('settle_loo_train', 'LOO settlement, train'), ('settle_loo_heldout', 'LOO settlement, held-out')]
    d = {r['currency']: (float(r['mean']), float(r['sd'])) for r in csv.DictReader(open(S/'M3/results/currency_summary.csv'))}
    W, rowh, L = 640, 30, 190
    H = 10 + rowh * len(want) + 26
    sx = lambda v: L + (v + 0.05) / 0.75 * (W - L - 30)
    o = [svg_open(W, H, 'Spearman with oracle importance')]
    for v in (0, 0.2, 0.4, 0.6):
        o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="6" y2="{H-26}" stroke="var(--line)"/>'
                 f'<text x="{sx(v):.1f}" y="{H-10}" text-anchor="middle" fill="var(--muted)">{v:.1f}</text>')
    for i, (k, name) in enumerate(want):
        y = 10 + i * rowh; m, sd = d[k]
        col = '#2457c5' if k.startswith('settle') else '#c0392b'
        o.append(f'<text x="{L-8}" y="{y+14}" text-anchor="end" fill="var(--fg)">{name}</text>')
        o.append(f'<rect x="{sx(0):.1f}" y="{y+2}" width="{sx(m)-sx(0):.1f}" height="16" fill="{col}"/>')
        o.append(f'<line x1="{sx(m-sd):.1f}" x2="{sx(m+sd):.1f}" y1="{y+10}" y2="{y+10}" stroke="var(--fg)"/>')
        o.append(f'<text x="{sx(m+sd)+5:.1f}" y="{y+14}" fill="var(--fg)">{m:.3f} ± {sd:.3f}</text>')
    o.append('</svg>')
    return ''.join(o)


def fig_signstrip():
    import numpy as np
    WN = '/home/agent/workspaces/eurisko-study/wealth-net/results'
    o = np.load(f'{WN}/decay_oracle_s1.npy')
    cur = {'books': np.load(f'{WN}/decay_currencies_s1.npz')['cum'],
           'settle': np.load(f'{WN}/settle_currencies_s1_step31100.npz')['loo_heldout']}
    order = np.argsort(-o); n = o.size; ntop = int(round(0.05 * n))
    W, H, L, R, T = 640, 230, 10, 10, 20
    mid = T + 90; bw = (W - L - R) / n
    panels = []
    for key, name in (('books', 'all-time gradient books'), ('settle', 'held-out LOO settlement')):
        v = cur[key][order]; sc = np.median(np.abs(v)) + 1e-12
        y = np.sign(v) * np.log1p(np.abs(v) / sc); y = y / (np.abs(y).max() + 1e-12) * 85
        neg_top = int((v[:ntop] < 0).sum()); neg_all = int((v < 0).sum())
        g = [f'<g class="sp sp-{key}"' + ('' if key == 'books' else ' style="display:none"') + '>']
        g.append(f'<rect x="{L}" y="{T}" width="{ntop*bw:.1f}" height="180" fill="var(--tldr)"/>')
        for i, yy in enumerate(y):
            col = '#c0392b' if yy < 0 else '#2457c5'
            y0, h = (mid - yy, yy) if yy >= 0 else (mid, -yy)
            g.append(f'<rect x="{L+i*bw:.2f}" y="{y0:.1f}" width="{max(bw,0.6):.2f}" height="{max(h,0.4):.1f}" fill="{col}"/>')
        g.append(f'<text x="{L+ntop*bw+6:.1f}" y="{T+12}" fill="var(--fg)">{name}: negative in top 26 = {neg_top}/26, all = {neg_all}/512</text>')
        g.append('</g>'); panels.append(''.join(g))
    s = [svg_open(W, H, 'Per-neuron account sign, sorted by oracle importance')]
    s.append(f'<line x1="{L}" x2="{W-R}" y1="{mid}" y2="{mid}" stroke="var(--muted)"/>')
    s += panels
    s.append(f'<text x="{L}" y="{H-8}" fill="var(--muted)">most important (oracle ablation) at left; shaded = top 5%; bar height = signed log magnitude; red = negative</text></svg>')
    btn = ('<div style="display:flex;gap:8px;margin:6px 0">'
           '<button type="button" data-k="books" class="spb" style="padding:6px 12px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--fg);font:inherit;cursor:pointer">Gradient books (in-sample)</button>'
           '<button type="button" data-k="settle" class="spb" style="padding:6px 12px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);font:inherit;cursor:pointer">LOO settlement (prequential, held-out)</button></div>')
    js = ('<script>document.querySelectorAll(".spb").forEach(function(b){b.addEventListener("click",function(){'
          'var k=b.getAttribute("data-k");document.querySelectorAll(".sp").forEach(function(g){g.style.display=g.classList.contains("sp-"+k)?"":"none"});'
          'document.querySelectorAll(".spb").forEach(function(x){x.style.background=x===b?"var(--panel)":"var(--bg)"})})});</script>')
    return btn + ''.join(s) + js

def png(rel, alt):
    b = base64.b64encode((S/rel).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="{html.escape(alt)}" style="width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff">'

def figure(body, cap):
    import re
    body = re.sub(r' (fill|stroke)="(var\(--[a-z-]+\))"', r' style="\1:\2"', body)
    return f'<figure style="margin:16px 0">{body}<figcaption class="meta">{cap}</figcaption></figure>'

FIGS = {
 'signstrip': figure(fig_signstrip(), 'Demo. Wealth-net seed 1, final step 31100, all 512 hidden neurons sorted by oracle importance (loss increase when zeroed). Toggle between the in-sample gradient books and prequential LOO settlement. Source: wealth-net/results/decay_oracle_s1.npy, decay_currencies_s1.npz (cum), settle_currencies_s1_step31100.npz (loo_heldout); the same arrays feed study/M1/lead/sign_census.py (seed-1 rows: books_cum top-5% 0.308, settle held-out 0.000).'),
 'toy': figure(fig_toy(), 'Figure 1. M1 toy (closed form): gradient credit and ablation value disagree in sign for w in (1, 2.1786), 23 of 23 grid points. Source: study/M1/lead/results/toy_closed_form.csv, toy_summary.json. X1: M1-a1 verified.'),
 'census': figure(fig_census(), 'Figure 2. Sign census on wealth-net (3 seeds, mean ± sd): share of the 26 most important neurons (by ablation oracle) whose account is negative. Thin grey bar: same share over all 512 neurons. Source: study/M1/lead/results/sign_census_summary.csv. X1: M1-b-books, M1-b-settle verified; M1-lock-why-b refuted (see text).'),
 'spearman': figure(fig_spearman(), 'Figure 3. M3-A: Spearman rank correlation of each currency with oracle importance (3 seeds, mean ± sd). Source: study/M3/results/currency_summary.csv. X1: cum, gradnorm, settle_loo_train verified by fresh seeds; held-out is M3\'s table (upper bound, not a separate X1 row).'),
 'postshift': figure(png('M3/w2/results/postshift.png', 'Fixed share vs Kelly after a regime shift'), 'Figure 4. M3-B: after a shift, Kelly wealth stays locked on the old winner; fixed share 1e-3 recovers in about 120 steps. Source: study/M3/w2/results/postshift.png, summary.csv. X1: M3-B verified. This is Herbster-Warmuth fixed share, not new.'),
 'lottery': figure(png('M9/w1/lottery_fit.png', 'Soup Brier vs K/Q with the lottery law fit'), 'Figure 6. M9-a: the soup follows a lottery law, Brier = 0.25 - D(1-(1-pi)^(K/Q)), R2 0.79-0.94 on priced families, shuffled null p=0.001. Source: study/M9/w1/lottery_fit.png, lottery_fit.csv. X1: M9a-lottery verified.'),
 'flops': figure(png('M9/b/flops_to_brier.png', 'Soup vs net on one compute axis'), 'Figure 7 (UNVERIFIED). M9-b: soup vs net on one FLOP axis. Source: study/M9/b/flops_to_brier.png. X1: M9b-flops is unverified (no fresh-seed rerun); read it as a picture of the setup, not a result.'),
 'k1': figure(png('K1/w1/trajectories.png', 'Kelly economy: top-clade share and Gini vs null'), 'Figure 8. K1: thousands of Kelly traders. With niches on, every slot stays alive; with niches off one species takes >= 99.7% of wealth. Gini 0.44 vs 0.74 for the beliefs-shuffled null. Source: study/K1/w1/trajectories.png, cells.csv. X1: K1-niches-on, K1-niches-off, K1-gini verified; K1-shift refuted.'),
 'maxload': figure(png('M7/w1/results/max_load_trace.png', 'MoE expert max load over training'), 'Figure 5. M7: max expert load over training for each router arm (E8 and E64). Source: study/M7/w1/results/max_load_trace.png. X1: M7-5 (Exp3 collapses, max load 0.775) and M7-6 (global prior max load 1.00) verified.'),
}

if __name__ == '__main__':
    import sys
    p = Path(sys.argv[1]); t = p.read_text()
    t = t.replace('<title>EURISKO Study Report</title>', '<title>Traders Inside a Net</title>')
    t = t.replace('Rendered from REPORT.md (2026-09-27).', 'Rendered from study/D1/IDEAS-NET.md (2026-09-28) with tools/render.py; figures by study/D1/w1/figs.py from committed result files.')
    n = 0
    for k, v in FIGS.items():
        tag = f'<p>FIG:{k}</p>'
        if tag in t: t = t.replace(tag, v); n += 1
    p.write_text(t); print('figures', n, 'bytes', len(t))
