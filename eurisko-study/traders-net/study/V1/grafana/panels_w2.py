"""V1 worker 2 panels for the traders-net dashboard: conservation residual, E_C, price per family.

build_dashboard.py (worker 1) calls panels_w2(y0, ds) and appends the result. Every panel reads
the source query of panel 1 through the `-- Dashboard --` datasource and selects its columns by
regex (column names: ../SCHEMA.md), so a run lacking a column family just shows "No data".
Column typing is the source query's job: every column matched here must be typed number there.
"""
DASH = {'type': 'datasource', 'uid': '-- Dashboard --'}

# Fixed colour per question family, so a family keeps its colour across panels.
FAMILY_COLOR = {
    'periodic': '#3274D9', 'lfsr16': '#FF9830', 'lfsr': '#FF9830', 'ca30': '#73BF69', 'ca110': '#F2495C',
    'prime': '#B877D9', 'pi': '#FADE2A', 'prng': '#FADE2A', 'and': '#4FC4CA', 'or': '#AB7B53',
    'heldout': '#8AB8FF', 'd_delay': '#C4162A', 'd_prove': '#1F60C4', 'd_expensive': '#FA6400',
}

# (regex on the raw column name, what it is, where it comes from). check.py imports this list.
PATTERNS = {
    'cons': r'^(cons_resid_.*|cons_sum_resid_max)$',
    'ec': r'^ec_.*$',
    'price': r'^price_.*$',
    'bce': r'^bce_(?!logistic_).*$',
    'bce_logistic': r'^bce_logistic_.*$',
    'brier': r'^brier_.*$',
}


def _fam_overrides(extra=None):
    o = [{'matcher': {'id': 'byName', 'options': f},
          'properties': [{'id': 'color', 'value': {'mode': 'fixed', 'fixedColor': c}}]}
         for f, c in FAMILY_COLOR.items()]
    return o + (extra or [])


def _trend(pid, title, pattern, rename, x, y, w, h, unit='none', desc='', overrides=None, custom=None,
           thresholds=None):
    cust = {'lineWidth': 2, 'showPoints': 'never', 'spanNulls': True, 'fillOpacity': 0, 'axisPlacement': 'auto'}
    cust.update(custom or {})
    defaults = {'unit': unit, 'custom': cust, 'color': {'mode': 'palette-classic'}}
    if thresholds:
        defaults['thresholds'] = {'mode': 'absolute', 'steps': [{'color': 'transparent', 'value': None}]
                                  + [{'color': c, 'value': v} for v, c in thresholds]}
        defaults['custom']['thresholdsStyle'] = {'mode': 'dashed'}
    tr = [{'id': 'filterFieldsByName', 'options': {'include': {'pattern': '^step$|' + pattern}}}]
    for rx, rp in rename:
        tr.append({'id': 'renameByRegex', 'options': {'regex': rx, 'renamePattern': rp}})
    return {
        'id': pid, 'type': 'trend', 'title': title, 'description': desc,
        'gridPos': {'x': x, 'y': y, 'w': w, 'h': h},
        'datasource': DASH,
        'targets': [{'refId': 'A', 'datasource': DASH, 'panelId': 1, 'withTransforms': False}],
        'options': {'xField': 'step', 'legend': {'showLegend': True, 'displayMode': 'list', 'placement': 'bottom'},
                    'tooltip': {'mode': 'multi', 'sort': 'none'}},
        'fieldConfig': {'defaults': defaults,
                        'overrides': [{'matcher': {'id': 'byName', 'options': 'step'},
                                       'properties': [{'id': 'unit', 'value': 'none'}, {'id': 'decimals', 'value': 0},
                                                      {'id': 'custom.scaleDistribution',
                                                       'value': {'type': 'linear'}}]}]
                        + (overrides or [])},
        'transformations': tr,
    }


def panels_w2(y0, ds=None, first_id=100):
    """Return the worker-2 panels laid out from row y0 (24-wide grid). ds is unused (panels reuse panel 1)."""
    p, pid = [], first_id
    p.append(_trend(pid, 'Conservation residual (fraction of |income|)', PATTERNS['cons'],
                    [(r'^cons_resid_(.*)$', '$1'), (r'^cons_sum_resid_max$', 'max over cuts')], 0, y0, 12, 8,
                    unit='percentunit', custom={'scaleDistribution': {'type': 'log', 'log': 10}},
                    desc='How far the books leak: |sum of unit revenues at a cut - model income| / |income|, by '
                         'where the leak enters (total, attention QK, softmax, LayerNorm, other). Near 0 means the '
                         'wealth of a big trader equals the sum of its parts. Log axis.'))
    pid += 1
    p.append(_trend(pid, 'Exploitation E_C(t) by trader class', PATTERNS['ec'], [(r'^ec_(.*)$', '$1')], 12, y0, 12, 8,
                    desc='Best cumulative profit a fixed trader class C makes against the network’s prices '
                         '(M5 harness). A logical inductor keeps every line sublinear: flat or bending over. '
                         'A straight rising line means the class exploits the net.'))
    pid += 1
    p.append(_trend(pid, 'Market price per family (mean probability)', PATTERNS['price'],
                    [(r'^price_(.*)$', '$1')], 0, y0 + 8, 12, 8, unit='percentunit',
                    custom={'axisSoftMin': 0, 'axisSoftMax': 1}, overrides=_fam_overrides(),
                    desc='Mean predicted probability (the market price) on the questions of each family. '
                         'Unlearnable families (PRNG, pi) should sit at their base rate, near 0.5.'))
    pid += 1
    p.append(_trend(pid, 'Held-out BCE per family (nats; dashed = logistic baseline)',
                    r'^bce_.*$',
                    [(r'^bce_logistic_(.*)$', '$1 (logistic)'), (r'^bce_(.*)$', '$1')], 12, y0 + 8, 12, 8,
                    thresholds=[(0.6931, 'red')],
                    overrides=_fam_overrides([{'matcher': {'id': 'byRegexp', 'options': r'.* \(logistic\)$'},
                                               'properties': [{'id': 'custom.lineStyle',
                                                               'value': {'fill': 'dash', 'dash': [6, 4]}},
                                                              {'id': 'custom.lineWidth', 'value': 1}]}]),
                    desc='Cross-entropy per family on held-out (or prequential) data; red dashed line at ln 2 = '
                         'always 0.5. Thin dashed lines: a per-question logistic model on the same inputs.'))
    pid += 1
    p.append(_trend(pid, 'Brier score per family (0.25 = always 0.5)', PATTERNS['brier'], [(r'^brier_(.*)$', '$1')],
                    0, y0 + 16, 24, 8, overrides=_fam_overrides(), thresholds=[(0.25, 'red')],
                    desc='Mean (p - y)^2 per family on held-out (or prequential) data. Lower is better.'))
    return p
