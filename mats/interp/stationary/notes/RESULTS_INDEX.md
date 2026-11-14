# Where the measured numbers are (stationary-regime lane, for the sibling lane stationary-numerics)

All runs: Model S (absolute noise, sigma = 1 unless a file name says otherwise), W_l = lib/st.py W_layer(seed, n, l),
layer 0 = target, output a = relu(W_1 h_0). JSON files in `runs/`, scorecards in `results/`.

| what | file(s) | notes |
|---|---|---|
| Part A scorecard (PA1-PA12, PR1-PR2) and tables | `results/partA_scores.json`, `results/partA_tables.md` | scorer `scratch/analyze_A.py` |
| radius law: scalar chain vs nets (quantiles of Q = \|h\|^2/n) | `runs/radius_n{n}.json` | chain_q, net_q, dufresne_q, Hill indices |
| cloud at layer 0 (covariance spectrum, radial share, skew, gate entropy) | `runs/cloud_n{n}_s{1000..1003}.json` | evC in sigma^2 units; coord_pred = sigma^2/(1-exp(-2(k+1/4)/n)) |
| cloud, fresh nets, two-replica dispersion delta and radial CV (Amendment A3) | `runs/cloudA3_n{n}_s{2000..2003}.json` | delta_pairs, R_cv, X_quantiles |
| radius-history cloud model (post hoc) | `runs/cloudhist_n{n}_s{1000,1001}.json` | |
| births (flips under resampling one layer's noise, binned by Q) | `runs/births_n{n}_s*.json` | |
| age profile (total effect of one layer's noise by age, radial share, PR) | `runs/age_n{n}_s{1000,1001}.json`, `results/age_by_direction.json` | |
| bit cascade after one resampling (flips per later layer; linear-response correlation) | `runs/bitlife_n{16,64,256}_s{1000,1001}.json` | Amendment A2 |
| Lyapunov QR, coupling-from-the-past rates | `runs/lyap_n{n}_s*.json`, `runs/couple_n{n}_s*.json` | |
| Model R sub-study (fixed input, sigma 1e-3..1e-1) | `runs/partR_n{32,128}_s*_sig*.json` | |
| interaction structure (Sobol by layer, mechanisms, effective dimension) | `runs/partI_{curves,pairs,trunc}_S_n{n}_s*.json`, `runs/partI_curves_clamp_*`, `runs/partI_kesten_n{n}.json`, `results/partI_scores.json` | Part I (registered 14:10Z) |
| Part B estimators (truth, MLMC, CV, antithetic, RQMC, tail IS) | `runs/{truth,mlmc,cv,cvfix,anti,rqmc,tail}_n{n}_s{1000,1001}.json`, `results/partB_scores.json` | |
| Part C input regime (Model S with an input, sigma 1e-3..1e-1) | `runs/partC_n{n}_L{L}_sig{s}_s*.json`, `results/partC_scores.json` | |
