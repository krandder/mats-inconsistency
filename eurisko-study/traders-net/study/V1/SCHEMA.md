# V1 CSV schema (what the traders-net dashboard reads)

One CSV per run, one row per checkpoint, header in row 1, x-axis column `step`.
Panels select columns by regex, so extra columns are harmless and missing ones leave a panel empty.
`nan` or empty = no value.

Where serve.py looks (re-read on every request):
- `traders-net/study/<ws>/runs/<run>/metrics.csv`  -> run id `<ws>.<run>` (S0 runlog.py output; any ws)
- `wealth-net/results/<tag>/snapshots/step-*.json` -> run id `wealth-net.<tag>` (derived on the fly, read-only)

| column pattern | meaning | panel (owner) |
|---|---|---|
| `step` | training / stream step | x-axis |
| `loss` | training loss (window mean) | wealth (w1) |
| `wealth_<scale>` | total wealth at a scale: out, head, neuron, comp (one per granularity) | wealth by scale (w1) |
| `wf_<scale>_<family>` | wealth of that scale's units earned on questions of that family | wealth by family (w1) |
| `R_total` | total model income (the sum every scale should match) | wealth by scale (w1) |
| `gini_<scale>` | Gini of positive wealth at that scale | Gini vs null (w1) |
| `gini_null_<scale>` | same under shuffled labels (S0 panel.py) | Gini vs null (w1) |
| `churn_<scale>` | 1 - Spearman(wealth now, wealth 1000 steps ago) | rank churn (w1) |
| `turnover_<scale>` | Spearman(income this window, income previous window) | rank churn (w1) |
| `neg_<scale>` | units with negative wealth (fraction) | births/bankruptcies (w1) |
| `births`, `bankrupt`, `dead_frac` | cumulative births / bankruptcies, dead fraction | births/bankruptcies (w1) |
| `cons_resid_<source>` | conservation residual / abs(R), by source | conservation (w2) |
| `ec_<class>` | exploitation E_C(t) of trader class | E_C (w2) |
| `price_<family>` | mean market price (probability) per family | price (w2) |
| `bce_<family>`, `brier_<family>` | held-out / prequential BCE, Brier per family | price (w2) |
