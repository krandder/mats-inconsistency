## 4. Trained narrow-deep nets

Trained on one L4 (`gpu/`, `train/`; GPU-LOG.md).
- **RNNs.** ARC's AlgZoo architecture: bias-free ReLU, h_0 = 0, logits W_oh h_T, 2nd argmax of a Gaussian sequence.
  Width 16, T = 64 (4 seeds) and T = 32 (2 seeds), noise sigma = 0 and 0.01. ARC's recipe at batch 4096, 2^28
  sequences (a quarter of ARC's budget).
- **Sorting MLP (my choice of task).** Sort x ~ N(0, I_16). Width-16 bias-free ReLU MLPs:
  - residual (h <- h + V ReLU(W h), V initialised at 1/sqrt(L)), L = 16 and 64;
  - plain He-ReLU with a linear readout, L = 16 and 64;
  - sigma = 0 and 0.01; Adam, 2^15 steps.
- Why sorting: sorting networks are narrow-deep comparator circuits, and ReLU(b - a) is one comparator. The map is
  positively homogeneous like the nets. The literature check found no learned narrow-deep sorting network.

### 4.1 Trainability (MEASURED; finals on 2^18 fresh inputs, `runs/p4/eval_all.json`)

| model | seeds | result, noise on (per seed) | chance / reference | trains? (PREREG criterion) |
|---|---|---|---|---|
| RNN width 16, T = 64, sigma = 0 | 4 | accuracy 44.1%, 34.8%, 35.6%, 42.7% | 1.6% | yes (> 7.8%) |
| RNN width 16, T = 64, sigma = 0.01 | 4 | 31.2%, 34.4%, 29.3%, 33.1% | 1.6% | yes |
| RNN width 16, T = 32, sigma = 0 / 0.01 | 2 / 2 | 38.1%, 40.4% / 32.7%, 37.1% | 3.1% | yes |
| residual sorting MLP, L = 64 | 2 + 2 | nMSE 0.036, 0.039 / 0.034, 0.041 | 1 = predict the mean | yes (<= 0.05) |
| residual sorting MLP, L = 16 | 2 + 2 | nMSE 0.074, 0.082 / 0.075, 0.087 | | no (0.07-0.09) |
| plain sorting MLP, L = 64 | 2 + 1 | nMSE 0.81, 1.14; one run diverged (2e18) | | no |
| plain sorting MLP, L = 16 | 2 + 1 | nMSE 0.32, 0.40 / 0.41 | | no |

- **The RNNs learn but are capacity-bound.** Width 16 reaches 29-44% at T = 64, 19-28x chance. AlgZoo's width 16
  reaches 94.8% at T = 10.
  - T = 64 is not worse than T = 32 (best 44.1% vs 40.4%); the PREREG predicted it would be.
  - The best T = 64 seed was still improving at the end (35.7% -> 39.8% -> 44.1% over the last three checkpoints),
    so more training may help.
  - Noise at sigma = 0.01 costs 2-10 points.
- **Sorting needs depth plus a residual stream.**
  - Residual depth 64 beats depth 16 (nMSE 0.034-0.041 vs 0.074-0.087), but it is approximate: only 15-19% of
    output vectors have every coordinate within 0.1.
  - Plain nets of the same width fail at depth 64 (nMSE 0.81-1.14, i.e. no better than predicting the mean) and
    reach only 0.32-0.41 at depth 16.
  - The failed plain depth-64 net is synchronised exactly like the random ones (4.2): gate disagreement between
    inputs is 0.000 from layer 16 on, its output is a function of |x| alone (direction share of the output variance
    0.12, norm share 0.87), and it cannot represent a sort.

### 4.2 Structure of the trained nets (MEASURED, `runs/p4/an_*.json`, best seed per configuration)

| model | gate disagreement between inputs at layer/step 1 / L/4 / L/2 / L | per-path N_surv(0.01) at L' = 4 / L/2 / L-4 | k*_F(M; 1e-2) | share of output variance: direction / norm |
|---|---|---|---|---|
| residual sort, L = 64 | 0.50 / 0.24 / 0.22 / 0.19 | 16 / 10.8 / 10.7 | 12-16 | 0.77 / 0.20 |
| plain sort, L = 64 (failed) | 0.50 / 0.000 / 0.000 / 0.000 | 5.0 / 1.0 / 1.0 | 1-5 | 0.12 / 0.87 |
| RNN T = 64, sigma = 0 | 0.50 / 0.09 / 0.05 / 0.03 | 11.8 / 10.6 / 8.8 | 11-12 | 0.98 / 0.01 |
| RNN T = 64, sigma = 0.01 | 0.50 / 0.08 / 0.05 / 0.05 | 4.3 / 1.8 / 5.5 | 1-2 | 0.75 / 0.20 |
| RNN T = 32, sigma = 0 | 0.50 / 0.19 / 0.11 / 0.07 | 11.8 / 10.5 / 8.5 | 12-13 | 0.96 / 0.03 |

- **The trained residual sorter keeps every direction.** Path Jacobians keep 11-16 of 16 directions, a fixed block
  needs 12-16, and 19-24% of gates differ between inputs at every depth. A sorter's Jacobian is close to a
  permutation, so this is expected.
- **RNNs.** No input synchronisation: the angle between two sequences' states stays at 0.2-0.4 rad. But the gates
  settle: by step 64 only 3-5% of units differ between input sequences.
  - In the noise-trained RNN (sigma = 0.01) the logits depend on the state at any earlier step through essentially
    one direction: PR(M) = 1.0, and the top direction carries 99-100% of the sensitivity at cuts 32-60.
  - There, a gate-free object finds that direction: the random-mask average captures 96-100% of it, the unmasked
    W_hh power 12-87%.
  - Without noise the top direction carries 51-80%. The random-mask average gets 28-72%, and the marginal-gate
    product matches the optimum at every cut except the earliest.
  - So for one trained net Kelvin's gate-free bound does hold. It holds for that net's specific structure, a single
    integrating direction, not as a general property.

### 4.3 Estimation on trained nets (MEASURED, `runs/p4/est_*.json`)

| model | K2 nMSE (cost in MC paths) | K2-Stein nMSE | best simple sampler at about the K2 budget | best hybrid |
|---|---|---|---|---|
| plain sort, L = 16 | 0.013 (1,183) | 0.013 | RQMC 2.3e-5 at 1,024 | 1.1e-3 (HYB8-st, 1,024) |
| residual sort, L = 64, sigma = 0 | 0.024 (670) | 0.021 | RQMC 8.4e-5 at 1,024 | 6.3e-3 (HYB32-st, 1,024) |
[P4EST]

- On trained nets the Gaussian closure loses to the best sampler by 2-3 orders of magnitude, like ARC's AlgZoo
  models in the algzoo-noisy lane.
- The trained sorter's state is a partially sorted vector, far from Gaussian. The random residual nets of section
  3.1 are close to quadratic maps of a Gaussian, and that is what made K2 work there.
