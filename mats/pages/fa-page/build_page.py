"""Build index.html for mats.ekelvin.com/fa/ from example.json, pop_checks.json, extra_checks.json and the binding-arm
logs in bind_logs/ (fetch_bind.sh). Every number on the page that comes from data is filled in here.
usage: python3 build_page.py"""
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
EX = json.load(open(os.path.join(HERE, 'example.json')))
PC = json.load(open(os.path.join(HERE, 'pop_checks.json')))
XC = json.load(open(os.path.join(HERE, 'extra_checks.json')))
WF = '/home/agent/workspaces/mats/work/world-fa'
DIAG = {a: json.load(open(os.path.join(WF, 'runs/fa-pilot-1028b', a, 'diag.json')))['cells']
        for a in PC['pilot']}


def f(x, d=3, plus=False):
    """number -> string with a real minus sign."""
    s = f'{x:+.{d}f}' if plus else f'{x:.{d}f}'
    if s.startswith('-'):
        s = '−' + s[1:]
    if s in ('−0.' + '0' * d,):
        s = s[1:]
    return s


def sci(x):
    """small positive number -> 'm×10<sup>−k</sup>'."""
    m, e = f'{x:.1e}'.split('e')
    k = int(e)
    return f'{m}×10<sup>−{-k}</sup>' if k < 0 else f'{x:.1f}'


def fa(x, d=3):
    """ASCII-minus number for KaTeX."""
    return f'{x:.{d}f}'


# ------------------------------------------------------------------ the example episode
pair = [i + 1 for i in EX['true_pair']]                       # 1-indexed (1, 3)
PA = set(EX['true_pair'])
eh, rh, vh, r, v = EX['e_hist'], EX['r_hist'], EX['v_hist'], EX['r'], EX['v']
T, N = EX['T'], EX['N']
assert pair == [1, 3]


def hist_table():
    hl = lambda i: ' class="hl"' if i in PA else ''
    head = ('<tr><th rowspan="2">round</th><th rowspan="2">value \\(v_t\\)</th><th colspan="4" class="grp">reports \\(r_{t,i}\\), trader</th>'
            '<th colspan="4" class="grp">errors \\(e_{t,i}=r_{t,i}-v_t\\), trader</th><th rowspan="2">\\(e_{t,1}+e_{t,3}\\)</th></tr><tr class="sub">'
            + ''.join(f'<th>{i + 1}</th>' for i in range(N)) + ''.join(f'<th{hl(i)}>{i + 1}</th>' for i in range(N)) + '</tr>')
    rows = []
    for t in range(T):
        rows.append(f'<tr><td>{t + 1}</td><td>{f(vh[t], 2)}</td>' + ''.join(f'<td>{f(rh[t][i], 2)}</td>' for i in range(N))
                    + ''.join(f'<td{hl(i)}>{f(eh[t][i], 2)}</td>' for i in range(N))
                    + f'<td class="hl">{f(eh[t][0] + eh[t][2], 2)}</td></tr>')
    return '<div class="tbl"><table class="num">' + head + ''.join(rows) + '</table></div>'


A = EX['answers']
nets = EX['nets']
net = nets['anti_report_s0']


def answers_table():
    rows = [
        ('Plain average of the four reports', 'query reports', f(A['plain_average']['mean']), '', ''),
        ('Correlation-blind Bayes', 'query reports; treats the four errors as independent, variance 1',
         f(A['corr_blind']['mean']), f(A['corr_blind']['sd']), f(A['corr_blind']['nll'])),
        ('History-blind Bayes', 'query reports and the six-pair prior', f(A['history_blind']['mean']),
         f(A['history_blind']['sd']), f(A['history_blind']['nll'])),
        ('Exact Bayes', 'history, query reports and the prior', f(A['bayes']['mean']), f(A['bayes']['sd']), f(A['bayes']['nll'])),
        ('Trained network', 'all 37 tokens (pilot, report layout, seed 0)', f(net['mean']), f(net['sd']), f(net['nll'])),
    ]
    h = '<tr><th>forecast</th><th class="txt">uses</th><th>mean</th><th>sd</th><th>log loss at \\(v\\)</th></tr>'
    body = ''.join(f'<tr><td>{a}</td><td class="txt">{b}</td><td>{c}</td><td>{d}</td><td>{e}</td></tr>' for a, b, c, d, e in rows)
    return '<div class="tbl"><table class="num ans">' + h + body + '</table></div>'


def pairs_table():
    hb, fb = A['history_blind']['comps'], A['bayes']['comps']
    rows = []
    for j, p in enumerate(EX['family_pairs']):
        a, b = p[0] + 1, p[1] + 1
        pf = fb['w'][j]
        pfs = '&gt; 0.9999999' if pf > 0.9999999 else (sci(pf) if pf < 1e-3 else f(pf))
        cls = ' class="hl"' if set(p) == PA else ''
        rows.append(f'<tr{cls}><td>{a}&ndash;{b}</td><td>{f(hb["m"][j])}</td><td>{f(hb["w"][j], 2)}</td>'
                    f'<td>{f(EX["pair_score"][j], 1, plus=True)}</td><td>{pfs}</td></tr>')
    h = ('<tr><th>pair</th><th>candidate mean \\(m_{ab}\\)</th><th>probability, query reports only</th>'
         '<th>history score</th><th>probability, history and query</th></tr>')
    return '<div class="tbl"><table class="num">' + h + ''.join(rows) + '</table></div>'


def token_table():
    tk = EX['tokens']
    def key(x):
        return '(' + ', '.join(f(y, 2) for y in x[:2]) + ', &hellip;)'
    def row(name, t):
        z = all(abs(y) < 1e-12 for y in t['id']) and all(abs(y) < 1e-12 for y in t['rid'])
        idk = '0' if z else key(t['id'])
        rk = '0' if z else key(t['rid'])
        fl = f"{int(t['hist'])} {int(t['qrep'])} {int(t['query'])}"
        return (f'<tr><td>{name}</td><td>{idk}</td><td>{rk}</td><td>{f(t["r"])}</td><td>{f(t["v"])}</td><td>{f(t["e"])}</td>'
                f'<td>{fl}</td><td>{f(t["logN"])}</td><td>{f(t["logT"])}</td></tr>')
    h = ('<tr><th>token</th><th>trader key</th><th>round key</th><th>\\(r\\)</th><th>\\(v\\)</th><th>\\(e\\)</th>'
         '<th>flags</th><th>\\(\\log N\\)</th><th>\\(\\log(1+T)\\)</th></tr>')
    body = (row('round 1, trader 1', tk['hist_t1_i1']) + row('round 1, trader 2', tk['hist_t1_i2'])
            + row('round 2, trader 1', tk['hist_t2_i1']) + row('query report, trader 1', tk['qrep_i1'])
            + row('query report, trader 2', tk['qrep_i2']) + row('query token', tk['query']))
    return '<div class="tbl"><table class="num tok">' + h + body + '</table></div>'


def grid():
    xs = [96, 176, 256, 336]; w, h = 72, 24
    o = ['<text x="132" y="14" class="t b" text-anchor="middle">(a)</text>']
    o.append('<text x="80" y="34" class="t2" text-anchor="end">cells: e</text>')
    for i, x in enumerate(xs):
        o.append(f'<text x="{x + w // 2}" y="34" class="t b{" acc-t" if i in PA else ""}" text-anchor="middle">trader {i + 1}</text>')
    for t in range(T):
        y = 44 + 30 * t
        o.append(f'<text x="80" y="{y + 16}" class="t2" text-anchor="end">round {t + 1}</text>')
        for i, x in enumerate(xs):
            o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" class="{"cell-acc" if i in PA else "cell"}"/>')
            o.append(f'<text x="{x + w // 2}" y="{y + 16}" class="t2" text-anchor="middle">{f(eh[t][i], 2)}</text>')
    ybot = 44 + 30 * (T - 1) + h
    o.append(f'<rect x="{xs[0] - 4}" y="40" width="{w + 8}" height="{ybot + 4 - 40}" rx="6" class="dash"/>')
    yb = 44 + 30 * 3
    o.append(f'<rect x="{xs[0] - 7}" y="{yb - 4}" width="{xs[3] + w - xs[0] + 14}" height="{h + 8}" rx="6" class="dash"/>')
    o.append(f'<text x="{xs[3] + w + 14}" y="{yb + 16}" class="t b">(b)</text>')
    yq = ybot + 32
    o.append(f'<text x="80" y="{yq + 16}" class="t2" text-anchor="end">query: r</text>')
    for i, x in enumerate(xs):
        o.append(f'<rect x="{x}" y="{yq}" width="{w}" height="{h}" rx="4" class="cell-q"/>')
        o.append(f'<text x="{x + w // 2}" y="{yq + 16}" class="t q-t" text-anchor="middle">{f(r[i], 2)}</text>')
    for i in sorted(PA):
        xc = xs[i] + w // 2
        o.append(f'<line x1="{xc}" y1="{ybot + 6}" x2="{xc}" y2="{yq - 3}" class="ln-acc" marker-end="url(#ah2)"/>')
    o.append(f'<text x="{xs[2] + w // 2 + 10}" y="{ybot + 24}" class="t b">(d)</text>')
    yt = yq + h + 26
    o.append(f'<rect x="{xs[0]}" y="{yt}" width="{xs[2] + w - xs[0]}" height="28" rx="6" class="box"/>')
    o.append(f'<text x="{(xs[0] + xs[2] + w) // 2}" y="{yt + 18}" class="t2" text-anchor="middle">query token: reads all {N * (T + 1)} report tokens</text>')
    o.append(f'<line x1="{xs[2] + w + 4}" y1="{yt + 14}" x2="{xs[3] + 20}" y2="{yt + 14}" class="ln" marker-end="url(#ah2g)"/>')
    o.append(f'<text x="{xs[3] + 24}" y="{yt + 18}" class="t2">forecast of v</text>')
    return '\n'.join(o)


# ------------------------------------------------------------------ pilot numbers
MAIN = ['anti_report_s0', 'anti_report_s1', 'anti_column_s0', 'anti_column_s1']
NULLS = ['null_report_s0', 'null_report_s1', 'null_column_s0', 'null_column_s1']
LBL = {'anti_report_s0': 'report layout, seed 0', 'anti_report_s1': 'report layout, seed 1',
       'anti_column_s0': 'column layout, seed 0', 'anti_column_s1': 'column layout, seed 1'}


def pilot_table():
    h = ('<tr><th>main-family arm</th><th>gap at T&nbsp;=&nbsp;0</th><th>gap at T&nbsp;=&nbsp;8&nbsp;/&nbsp;32&nbsp;/&nbsp;128</th><th>fraction recovered at T&nbsp;=&nbsp;32</th>'
         '<th>above history-blind, worst&nbsp;T</th></tr>')
    rows = []
    for a in MAIN:
        c = DIAG[a]
        hv = ' / '.join(f(c[T_]['history_value']) for T_ in ('8', '32', '128'))
        gp = ' / '.join(f(c[T_]['nll_model'] - c[T_]['nll_bayes']) for T_ in ('8', '32', '128'))
        mx = max(c[T_]['nll_model'] - c[T_]['nll_bayes_T0'] for T_ in c)
        rows.append(f'<tr><td>{LBL[a]}</td><td>{f(c["0"]["nll_model"] - c["0"]["nll_bayes"])}</td><td>{gp}</td>'
                    f'<td>{f(c["32"]["frac_history_recovered"], 2)}</td><td>{f(mx)}</td></tr>')
    return '<div class="tbl"><table class="num">' + h + ''.join(rows) + '</table></div>'


def pilot_facts():
    shuf = max(abs(DIAG[a][T_]['shuf_nll_model'] - DIAG[a][T_]['nll_model']) for a in MAIN for T_ in ('8', '32', '128'))
    relab = max(DIAG[a][T_]['kl_relabel'] for a in PC['pilot'] for T_ in DIAG[a])
    null_loss = max(DIAG[a][T_]['nll_model'] - DIAG[a][T_]['nll_bayes_T0'] for a in NULLS for T_ in DIAG[a])
    null_gap = [DIAG[a][T_]['nll_model'] - DIAG[a][T_]['nll_bayes'] for a in NULLS for T_ in ('8', '32', '128')]
    null_hv = [DIAG[NULLS[0]][T_]['history_value'] for T_ in ('8', '32', '128')]
    att = {a: XC[a] for a in ('anti_report_s0', 'anti_report_s1')}
    rmin = min(min(x) for a in att for x in [att[a]['ratio_by_layer']])
    rmax = max(max(x) for a in att for x in [att[a]['ratio_by_layer']])
    hmin = min(min(h) for a in att for L in att[a]['ratio_by_layer_head'] for h in [L])
    hmax = max(max(h) for a in att for L in att[a]['ratio_by_layer_head'] for h in [L])
    mass = [m for a in att for m in att[a]['history_mass_by_layer']]
    gapmin = min(DIAG[a][T_]['nll_model'] - DIAG[a][T_]['nll_bayes_T0'] for a in MAIN for T_ in DIAG[a])
    gapmax = max(DIAG[a][T_]['nll_model'] - DIAG[a][T_]['nll_bayes_T0'] for a in MAIN for T_ in DIAG[a])
    t0 = [DIAG[a]['0']['nll_model'] - DIAG[a]['0']['nll_bayes'] for a in MAIN]
    return dict(shuf=shuf, relab=relab, null_loss=null_loss, null_gap=(min(null_gap), max(null_gap)), null_hv=null_hv,
                rmin=rmin, rmax=rmax, hmin=hmin, hmax=hmax, mass=(min(mass), max(mass)), gapmin=gapmin, gapmax=gapmax,
                t0=(min(t0), max(t0)))


PF = pilot_facts()


def ladder():
    c = DIAG['anti_report_s0']['32']
    rows = [('Prior \\(\\mathcal N(0,1)\\)', c['nll_prior'], ''), ('Correlation-blind Bayes', c['nll_diag_bayes'], ''),
            ('History-blind Bayes', c['nll_bayes_T0'], ''), ('Trained network', c['nll_model'], ' net'),
            ('Exact Bayes', c['nll_bayes'], '')]
    top = rows[0][1]
    body = ''.join(f'<tr><td>{n}</td><td>{f(x)}</td><td class="barcell"><span class="bar{cl}" style="width:{100 * x / top:.1f}%">'
                   f'</span></td></tr>' for n, x, cl in rows)
    return ('<div class="tbl"><table class="num ladder"><tr><th>forecast</th><th>log loss (nats)</th><th></th></tr>'
            + body + '</table></div>'), c


# ------------------------------------------------------------------ pilot validation chart (inline SVG)
def chart():
    W, H = 640, 270
    x0, x1, y0, y1 = 64, 560, 228, 18
    ymax = 0.6
    X = lambda s: x0 + (x1 - x0) * s / 20000
    Y = lambda g: y0 - (y0 - y1) * g / ymax
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Validation gap to exact Bayes during the pilot, by training step" class="chart">']
    for g in (0, 0.2, 0.4, 0.6):
        out.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(g):.1f}" y2="{Y(g):.1f}" class="grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{Y(g) + 4:.1f}" class="t2" text-anchor="end">{f(g, 1)}</text>')
    for s in (0, 5000, 10000, 15000, 20000):
        out.append(f'<text x="{X(s):.1f}" y="{y0 + 18}" class="t2" text-anchor="middle">{s // 1000}k</text>')
    out.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{H - 4}" class="t2" text-anchor="middle">training step</text>')
    out.append(f'<text x="14" y="{(y0 + y1) / 2:.0f}" class="t2" text-anchor="middle" transform="rotate(-90 14 {(y0 + y1) / 2:.0f})">gap to exact Bayes (nats)</text>')
    for name, lvl, cls in (('main family', PC['val_level_anti']['mean'], 'acc'), ('null family', PC['val_level_null']['mean'], 'ink2')):
        out.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(lvl):.1f}" y2="{Y(lvl):.1f}" class="lvl {cls}"/>')
    for a, arm in PC['pilot'].items():
        pts = ' '.join(f'{X(s):.1f},{Y(g):.1f}' for s, _, g in arm['curve'])
        cls = 'acc' if a.startswith('anti') else 'ink2'
        out.append(f'<polyline points="{pts}" class="crv {cls}"/>')
    la, ln = PC['val_level_anti']['mean'], PC['val_level_null']['mean']
    out.append(f'<text x="{x1 + 6}" y="{Y(la) - 10:.1f}" class="t acc-t">main family:</text>')
    out.append(f'<text x="{x1 + 6}" y="{Y(la) + 5:.1f}" class="t acc-t">4 arms</text>')
    out.append(f'<text x="{x1 + 6}" y="{Y(la) + 20:.1f}" class="t2">level {f(la)}</text>')
    out.append(f'<text x="{x1 + 6}" y="{Y(ln) - 8:.1f}" class="t2">null: 4 arms,</text>')
    out.append(f'<text x="{x1 + 6}" y="{Y(ln) + 7:.1f}" class="t2">level {f(ln)}</text>')
    out.append('</svg>')
    return '\n'.join(out)


# ------------------------------------------------------------------ binding arms (running) from the copied logs
ARM_DESC = {
    'nll_s0': ('control (log loss only)', 60000), 'nll_s1': ('control, seed 1', 60000),
    'nll_ortho_s0': ('orthonormal keys', 60000), 'dense_s0': ('dense target; own GPU', 60000),
    'dense_s1': ('dense, seed 1', 60000), 'dense_ortho_s0': ('dense, orthonormal keys', 60000),
    'dense_bs1024_s0': ('dense, batch 1,024', 15000)}
ORDER = ['nll_s0', 'nll_s1', 'nll_ortho_s0', 'dense_s0', 'dense_s1', 'dense_ortho_s0', 'dense_bs1024_s0']
LINE = re.compile(r'\[(\S+)\] step (\d+) T0gap (\S+) T32gap (\S+) frac32 (\S+) attn \[([^\]]*)\] aux (\S+)')


def parse_logs():
    stamp, arms = None, {}
    for fn in ('fa-bind-1028d.txt', 'fa-bind-1028e.txt'):
        txt = open(os.path.join(HERE, 'bind_logs', fn)).read()
        st = txt.splitlines()[0].strip()
        stamp = st if stamp is None or st > stamp else stamp
        for sec in txt.split('== ')[1:]:
            name = sec.split('.log')[0]
            if fn == 'fa-bind-1028d.txt' and name == 'dense_s0':
                continue                                   # crashed there (out of memory); rerun on fa-bind-1028e
            rows = [m.groups() for m in LINE.finditer(sec)]
            done = 'DONE' in sec or 'TRAINED' in sec
            if rows:
                arms[name] = dict(rows=rows, done=done)
    return stamp, arms


def bind_table():
    stamp, arms = parse_logs()
    h = ('<tr><th>arm</th><th class="txt">setting</th><th>step / planned</th>'
         '<th>fraction recovered at T&nbsp;=&nbsp;32: last (range)</th><th>own/other attention, layers 1&ndash;4</th>'
         '<th>dense-head MSE</th></tr>')
    body, fr_all, att_all, steps = [], [], [], []
    for a in ORDER:
        if a not in arms:
            continue
        rows = arms[a]['rows']
        last = rows[-1]
        fr = [float(x[4]) for x in rows]
        fr_all += fr
        att = [float(y) for y in last[5].split(',')]
        att_all += [float(y) for x in rows for y in x[5].split(',')]
        desc, planned = ARM_DESC[a]
        steps.append((a, int(last[1]), planned))
        aux = '&ndash;' if last[6] == 'None' else f(float(last[6]))
        body.append(f'<tr><td>{a}</td><td class="txt sm">{desc}</td><td>{int(last[1]):,} / {planned:,}</td>'
                    f'<td>{f(fr[-1], 2)} ({f(min(fr), 2)} to {f(max(fr), 2)})</td>'
                    f'<td>{"all " + format(att[0], ".2f") if len(set(round(y, 2) for y in att)) == 1 else " ".join(f"{y:.2f}" for y in att)}</td><td>{aux}</td></tr>')
    tab = '<div class="tbl"><table class="num">' + h + ''.join(body) + '</table></div>'
    return tab, stamp, dict(fr=(min(fr_all), max(fr_all)), att=(min(att_all), max(att_all)), steps=steps)


BIND_TAB, STAMP, BIND = bind_table()
D0 = json.load(open(os.path.join(HERE, 'bind_logs', 'dense_s0.diag.json')))['cells']
stamp_hm = STAMP[11:16] + 'Z'


# ------------------------------------------------------------------ assemble
def tmpl():
    return open(os.path.join(HERE, 'template.html')).read()


def main():
    ladder_html, c32 = ladder()
    share = (c32['nll_prior'] - c32['nll_bayes_T0']) / (c32['nll_prior'] - c32['nll_bayes'])
    hb = A['history_blind']['comps']
    pop = EX['population_T8']
    cur = XC['curriculum']
    rep = {
        '@@HIST@@': hist_table(), '@@ANSWERS@@': answers_table(), '@@PAIRS@@': pairs_table(), '@@TOKENS@@': token_table(),
        '@@PILOT@@': pilot_table(), '@@LADDER@@': ladder_html, '@@CHART@@': chart(), '@@BIND@@': BIND_TAB,
        '@@STAMP@@': stamp_hm, '@@GRID@@': grid(),
        '@@D0_T0@@': f(D0['0']['gap_model']), '@@D0_F8@@': f(D0['8']['frac_history_recovered']),
        '@@D0_F32@@': f(D0['32']['frac_history_recovered']), '@@D0_F64@@': f(D0['64']['frac_history_recovered']),
        '@@D0_F128@@': f(D0['128']['frac_history_recovered']),
        '@@D0_ATT@@': ', '.join(f(x) for x in D0['32']['attention']['ratio_by_layer']),
        '@@D0_RMS@@': '&ndash;'.join(f'{100 * x:.2f}' for x in (min(D0[T_]['binding_response_shuffle']['rms_ratio'] for T_ in ('8', '32', '64', '128')),
                                                                    max(D0[T_]['binding_response_shuffle']['rms_ratio'] for T_ in ('8', '32', '64', '128')))),
        '@@S3_E1@@': '  '.join(f(e[0], 2) for e in eh[:3]) + '  …',
        '@@S3_E3@@': '  '.join(f(e[2], 2) for e in eh[:3]) + '  …',
        '@@R1@@': fa(r[0]), '@@R2@@': fa(r[1]), '@@R3@@': fa(r[2]),
        '@@R4K@@': (fa(r[3]) if r[3] < 0 else '+' + fa(r[3])),
        '@@R4T@@': ('− ' + fa(-r[3])) if r[3] < 0 else ('+ ' + fa(r[3])),
        '@@R@@': '(' + ', '.join(f(x) for x in r) + ')', '@@V@@': f(v),
        '@@MIN_M@@': f(min(hb['m'])), '@@MAX_M@@': f(max(hb['m'])),
        '@@PT0_MIN@@': f(min(hb['w']), 2), '@@PT0_MAX@@': f(max(hb['w']), 2),
        '@@HB_MEAN@@': f(A['history_blind']['mean']), '@@HB_SD@@': f(A['history_blind']['sd']),
        '@@B_MEAN@@': f(A['bayes']['mean']), '@@B_SD@@': f(A['bayes']['sd']),
        '@@NET_MEAN@@': f(net['mean']), '@@NET_SD@@': f(net['sd']),
        '@@NET_KL_T0@@': f(net['kl_from_T0'], 4), '@@NET_KL_B@@': f(net['kl_from_bayes'], 2),
        '@@NETS_KL_T0_MAX@@': f(max(n['kl_from_T0'] for n in nets.values()), 4),
        '@@HVLIST@@': ' / '.join(f(DIAG['anti_report_s0'][T_]['history_value']) for T_ in ('8', '32', '128')),
        '@@E1@@': f(r[0] - v, 2, plus=True), '@@E3@@': f(r[2] - v, 2, plus=True), '@@E4@@': f(r[3] - v, 2, plus=True),
        '@@AVG13@@': f((r[0] + r[2]) / 2), '@@AVG13ERR@@': f((r[0] + r[2]) / 2 - v, 2),
        '@@CORR13@@': f(EX['sample_corr'][0][2], 2),
        '@@CORR_OTH_MIN@@': f(min(EX['sample_corr'][i][j] for i in range(4) for j in range(i + 1, 4) if (i, j) != (0, 2)), 2),
        '@@CORR_OTH_MAX@@': f(max(EX['sample_corr'][i][j] for i in range(4) for j in range(i + 1, 4) if (i, j) != (0, 2)), 2),
        '@@MAXSUM@@': f(max(abs(e[0] + e[2]) for e in eh), 2),
        '@@MAX1@@': f(max(abs(e[0]) for e in eh), 1), '@@MAX3@@': f(max(abs(e[2]) for e in eh), 1),
        '@@NOPP@@': str(sum(1 for e in eh if e[0] * e[2] < 0)),
        '@@SCORE13@@': f(EX['pair_score'][1], 1, plus=True),
        '@@SCORE_NEXT@@': f(sorted(EX['pair_score'])[-2], 1, plus=True),
        '@@UNORD@@': f(EX['unordered_loglik'][0], 3),
        '@@POP_MEAN@@': f(pop['mean_kl'], 2), '@@POP_MED@@': f(pop['kl_quartiles'][1], 2),
        '@@POP_P09@@': f'{100 * pop["p_true_gt_09"]:.0f}%', '@@EX_KL@@': f(EX['kl_bayes_vs_T0'], 2),
        '@@EX_PCT@@': f'{100 * EX["kl_percentile"]:.0f}th', '@@SEED@@': str(EX['seed']),
        '@@L_PRIOR@@': f(c32['nll_prior']), '@@L_T0@@': f(c32['nll_bayes_T0']), '@@L_B@@': f(c32['nll_bayes']),
        '@@L_NET@@': f(c32['nll_model']), '@@L_DIAG@@': f(c32['nll_diag_bayes']),
        '@@L_SPAN@@': f(c32['nll_prior'] - c32['nll_bayes'], 2), '@@L_GOT@@': f(c32['nll_prior'] - c32['nll_bayes_T0'], 2),
        '@@L_SHARE@@': f'{100 * share:.0f}%', '@@L_NETGAP@@': f(c32['nll_model'] - c32['nll_bayes_T0'], 3),
        '@@L_HV@@': f(c32['history_value'], 2), '@@SHUF_B@@': f(c32['shuf_nll_bayes_seen'], 2),
        '@@PF_SHUF@@': sci(PF['shuf']), '@@PF_RELAB@@': sci(PF['relab']),
        '@@PF_RMIN@@': f(PF['rmin']), '@@PF_RMAX@@': f(PF['rmax']), '@@PF_HMIN@@': f(PF['hmin']), '@@PF_HMAX@@': f(PF['hmax']),
        '@@PF_MASS_MIN@@': f'{100 * PF["mass"][0]:.0f}', '@@PF_MASS_MAX@@': f'{100 * PF["mass"][1]:.0f}',
        '@@PF_GAPMIN@@': f(PF['gapmin']), '@@PF_GAPMAX@@': f(PF['gapmax']),
        '@@PF_T0MIN@@': f(PF['t0'][0]), '@@PF_T0MAX@@': f(PF['t0'][1]),
        '@@NULL_HV@@': ' / '.join(f(x) for x in PF['null_hv']),
        '@@NULL_GMIN@@': f(PF['null_gap'][0]), '@@NULL_GMAX@@': f(PF['null_gap'][1]), '@@NULL_LOSS@@': f(PF['null_loss']),
        '@@VL_A@@': f(PC['val_level_anti']['mean']), '@@VL_A_SE@@': f(PC['val_level_anti']['se']),
        '@@VL_N@@': f(PC['val_level_null']['mean']), '@@VL_N_SE@@': f(PC['val_level_null']['se']),
        '@@HV8@@': f(DIAG['anti_report_s0']['8']['history_value'], 2), '@@HV128@@': f(DIAG['anti_report_s0']['128']['history_value'], 2),
        '@@COST_A_MC@@': f(PC['cost_rho-0.8']['mc']), '@@COST_A_SE@@': f(PC['cost_rho-0.8']['se']),
        '@@COST_N_MC@@': f(PC['cost_rho0.8']['mc']), '@@COST_N_SE@@': f(PC['cost_rho0.8']['se']),
        '@@COST_A@@': f(PC['cost_rho-0.8']['closed']), '@@COST_N@@': f(PC['cost_rho0.8']['closed']),
        '@@CUR2_8@@': f(cur['N2 anti-or-independent']['hv']['8'][0]), '@@CUR2_32@@': f(cur['N2 anti-or-independent']['hv']['32'][0]),
        '@@CUR3_8@@': f(cur['N3 anti pair']['hv']['8'][0]), '@@CUR3_32@@': f(cur['N3 anti pair']['hv']['32'][0]),
        '@@BFR_MIN@@': f(BIND['fr'][0], 2), '@@BFR_MAX@@': f(BIND['fr'][1], 2),
        '@@BATT_MIN@@': f'{BIND["att"][0]:.2f}', '@@BATT_MAX@@': f'{BIND["att"][1]:.2f}',
    }
    html = tmpl()
    for k, x in rep.items():
        html = html.replace(k, x)
    left = re.findall(r'@@[A-Z0-9_]+@@', html)
    assert not left, left
    open(os.path.join(HERE, 'index.html'), 'w').write(html)
    print('wrote index.html', len(html), 'bytes; run status stamp', STAMP, 'steps', BIND['steps'])


if __name__ == '__main__':
    main()
