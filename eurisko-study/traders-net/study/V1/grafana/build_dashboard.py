#!/usr/bin/env python3
"""Write traders-net.json, the Grafana dashboard for traders-net runs. Rerun after editing.

Data: serve.py /runs/${run}/metrics.json (JSON so Infinity types the columns as numbers).
Panels pick columns by regex (see ../SCHEMA.md), so runs with different families or scales
all render. Panel 1 fetches once per refresh; every other panel reuses it via -- Dashboard --.
Worker 2's panels come from panels_w2.py (panels_w2(y0, ds) -> list of panel dicts), if present.
"""
import importlib.util, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
DS = {'type': 'yesoreyeram-infinity-datasource', 'uid': 'traders-net-runs'}
DASH = {'type': 'datasource', 'uid': '-- Dashboard --'}
SCALES = ['out', 'head', 'neuron', 'comp']
SCALE_DESC = {'out': 'readout weights (one per question)', 'head': 'attention heads',
              'neuron': 'MLP neurons', 'comp': 'components (embeddings, attention/MLP blocks)'}

source_query = {
    'refId': 'A', 'datasource': DS, 'type': 'json', 'source': 'url', 'parser': 'backend',
    'format': 'table', 'url': '/runs/${run}/metrics.json', 'url_options': {'method': 'GET', 'data': ''},
    'root_selector': '', 'filters': [], 'columns': [],
}
panels, pid = [], [0]
DASHED = [{'id': 'custom.lineStyle', 'value': {'fill': 'dash', 'dash': [6, 4]}}]


def trend(title, pattern, y, x, w=12, h=8, unit='none', desc='', renames=(), overrides=None, source=False,
          decimals=None):
    """pattern: regex over column names (step is always kept); renames: [(regex, replacement)]."""
    pid[0] += 1
    defaults = {'unit': unit, 'custom': {'lineWidth': 2, 'showPoints': 'never', 'spanNulls': True,
                                         'fillOpacity': 0, 'axisPlacement': 'auto'},
                'color': {'mode': 'palette-classic'}}
    if decimals is not None:
        defaults['decimals'] = decimals
    p = {
        'id': pid[0], 'type': 'trend', 'title': title, 'description': desc,
        'gridPos': {'x': x, 'y': y, 'w': w, 'h': h},
        'options': {'xField': 'step', 'legend': {'showLegend': True, 'displayMode': 'list', 'placement': 'bottom'},
                    'tooltip': {'mode': 'multi', 'sort': 'desc'}},
        'fieldConfig': {'defaults': defaults,
                        'overrides': [{'matcher': {'id': 'byName', 'options': 'step'},
                                       'properties': [{'id': 'unit', 'value': 'none'}, {'id': 'decimals', 'value': 0}]}]
                                     + (overrides or [])},
        'transformations': [{'id': 'filterFieldsByName', 'options': {'include': {'pattern': f'^(step|{pattern})$'}}}]
                           + [{'id': 'renameByRegex', 'options': {'regex': a, 'renamePattern': b}} for a, b in renames],
    }
    if source:
        p['datasource'], p['targets'] = DS, [source_query]
    else:
        p['datasource'], p['targets'] = DASH, [{'refId': 'A', 'datasource': DASH, 'panelId': 1, 'withTransforms': False}]
    panels.append(p)
    return p


def by_regex(rx, props):
    return {'matcher': {'id': 'byRegexp', 'options': rx}, 'properties': props}


# Row 0: total wealth per scale (source panel) + runs table
trend('Wealth by scale (sum over units) and total income R', r'wealth_[a-z0-9]+|R_total', 0, 0, w=16, h=9, source=True,
      renames=[(r'^wealth_(.*)$', '$1'), (r'^R_total$', 'R (model income)')],
      overrides=[by_regex('R \\(model income\\)', [{'id': 'color', 'value': {'mode': 'fixed', 'fixedColor': 'text'}}] + DASHED)],
      desc='Cumulative wealth (booked income a*dR/da) summed over every unit at one granularity. Each complete cut should '
           'add up to R, the model’s total income (dashed); the gap at a scale is income paid to biases and '
           'LayerNorm/softmax leaks. Heads can sum below zero: the attention softmax is not degree-one homogeneous.')
pid[0] += 1
panels.append({
    'id': pid[0], 'type': 'table', 'title': 'Runs', 'datasource': DS,
    'description': 'Every run the exporter sees: study/<ws>/runs/<run>/metrics.csv (id <ws>.<run>) and the live '
                   'wealth-net seeds (id wealth-net.<tag>, derived from its 100-step snapshots). Rows = checkpoints.',
    'gridPos': {'x': 16, 'y': 0, 'w': 8, 'h': 9},
    'targets': [{'refId': 'A', 'datasource': DS, 'type': 'json', 'source': 'url', 'parser': 'backend',
                 'format': 'table', 'url': '/runs.json', 'url_options': {'method': 'GET', 'data': ''},
                 'root_selector': '', 'filters': [],
                 'columns': [{'selector': 'run', 'text': 'run', 'type': 'string'},
                             {'selector': 'rows', 'text': 'rows', 'type': 'number'},
                             {'selector': 'updated', 'text': 'updated', 'type': 'timestamp'}]}],
    'fieldConfig': {'defaults': {}, 'overrides': [
        {'matcher': {'id': 'byName', 'options': 'updated'}, 'properties': [{'id': 'unit', 'value': 'dateTimeFromNow'}]}]},
    'options': {'showHeader': True, 'cellHeight': 'sm'},
})

# Fixed family colours (palette-classic reuses green for 'and' and 'pi'); unknown families keep the palette
FAMCOLOR = {'periodic': '#3274D9', 'lfsr16': '#FF9830', 'lfsr': '#FF9830', 'ca30': '#73BF69', 'ca110': '#F2495C',
            'prime': '#B877D9', 'pi': '#FADE2A', 'prng': '#FADE2A', 'and': '#4FC4CA', 'or': '#AB7B53'}
FAMCOL = [{'matcher': {'id': 'byName', 'options': f}, 'properties': [{'id': 'color', 'value': {'mode': 'fixed', 'fixedColor': c}}]}
          for f, c in FAMCOLOR.items()]

# Rows 1-2: wealth by family, one panel per scale
for i, s in enumerate(SCALES):
    trend(f'Wealth by family: {s}', f'wf_{s}_.*', 9 + 8 * (i // 2), 12 * (i % 2),
          renames=[(f'^wf_{s}_(.*)$', '$1')], overrides=FAMCOL,
          desc=f'Cumulative wealth of the {SCALE_DESC[s]}, split by the family of the question that paid it. '
               'Negative = on balance these units pushed the loss up on that family (in-sample books).')

# Row 3: Gini vs null, rank churn
trend('Wealth Gini vs shuffled-label null', r'gini_[a-z0-9_]+', 25, 0,
      renames=[(r'^gini_null_(.*)$', '$1 (null)'), (r'^gini_(?!null)(.*)$', '$1')],
      overrides=[by_regex('.*\\(null\\)', DASHED)], unit='none', decimals=2,
      desc='Gini of positive wealth per scale (0 = equal, 1 = one unit owns everything). Dashed = same run with '
           'shuffled labels (S0 panel.py). No null line = the run did not log one; wealth-net has none.')
trend('Rank churn: 1 - Spearman(wealth now, 1000 steps ago)', r'churn_[a-z0-9]+', 25, 12, decimals=3,
      renames=[(r'^churn_(.*)$', '$1')],
      desc='How much the wealth ranking moved over the last 1000 steps. Cumulative wealth freezes its ranks by '
           'construction (the premise says cumulative rank stability is an artefact), so read this with the turnover panel.')

# Row 4: turnover, births/bankruptcies
trend('Income turnover: Spearman(income this window, previous window)', r'turnover_[a-z0-9]+', 33, 0, decimals=2,
      renames=[(r'^turnover_(.*)$', '$1')],
      desc='Rank correlation of per-unit income between consecutive 100-step windows. Near 1 = the same units keep '
           'earning; near 0 = income is noise at that scale.')
trend('Births, bankruptcies, units in the red', r'births|bankrupt|dead_frac|neg_[a-z0-9]+', 33, 12,
      unit='percentunit', renames=[(r'^neg_(.*)$', '$1 in the red')],
      overrides=[by_regex('births|bankrupt', [{'id': 'unit', 'value': 'short'}, {'id': 'custom.axisPlacement', 'value': 'right'}])],
      desc='Fraction of units with negative cumulative wealth, per scale. births/bankrupt (right axis, counts) '
           'appear only in runs with an economy (K1, M8); wealth-net has neither.')

# Worker 2's panels (conservation residual, E_C, price per family)
w2 = os.path.join(HERE, 'panels_w2.py')
if os.path.exists(w2):
    spec = importlib.util.spec_from_file_location('panels_w2', w2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for p in mod.panels_w2(41, DS):
        pid[0] += 1
        p['id'] = pid[0]
        panels.append(p)

dash = {
    'uid': 'traders-net', 'title': 'traders-net', 'tags': ['traders-net', 'eurisko'],
    'description': 'Wealth of units inside small networks (eurisko-study/traders-net). Source: study/V1/grafana in that workspace.',
    'editable': False, 'graphTooltip': 1, 'refresh': '1m', 'schemaVersion': 41, 'version': 1,
    'time': {'from': 'now-6h', 'to': 'now'}, 'timepicker': {'hidden': True, 'refresh_intervals': ['30s', '1m', '5m', '15m']},
    'templating': {'list': [{
        'name': 'run', 'label': 'Run', 'type': 'query', 'datasource': DS, 'refresh': 2, 'sort': 1,
        'multi': False, 'includeAll': False,
        'current': {'text': 'wealth-net.train-s1', 'value': 'wealth-net.train-s1'},
        'definition': 'runs.json',
        'query': {'queryType': 'infinity', 'query': '', 'refId': 'variable',
                  'infinityQuery': {'refId': 'variable', 'type': 'json', 'source': 'url', 'parser': 'backend',
                                    'format': 'table', 'url': '/runs.json', 'url_options': {'method': 'GET', 'data': ''},
                                    'root_selector': '', 'filters': [],
                                    'columns': [{'selector': 'run', 'text': 'run', 'type': 'string'}]}},
    }]},
    'panels': sorted(panels, key=lambda p: (p['gridPos']['y'], p['gridPos']['x'])),
}
out = os.path.join(HERE, 'traders-net.json')
with open(out, 'w') as f:
    json.dump(dash, f, indent=1)
    f.write('\n')
print(out)
