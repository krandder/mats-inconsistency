# Harness ready (2026-10-28)

`harness.py` is usable now. Its module docstring is the reference: it covers the World interface, the entry
points and the outputs. `junta.py` is a complete world to copy from, and `test_junta.py` shows the oracle tests
expected of a world.

To add a world, write `worlds/<name>.py` with `make_world(cfg)`. The returned object needs:
- `feat_dim` and `target_kind` (`'real'`, or `'cat'` plus `n_classes`)
- `train_policies`, `heldout`, `test_policies`
- `sample(rng, E, N, policy, n=None)`, returning `X`, `mask` and `target` with the query token LAST
- `sample_train(rng, E, nmin, nmax)`
- `exact_bayes(ep)`, returning `(w, m, v)` mixtures or `(E, C)` probabilities
- `references(ep)`, a dict that must include `'bayes'`
- `meta(ep)`
- optional: `splits`, `abilities`, `fit_fixed`/`set_fixed`

Then run:

    python harness.py train   --world <name> --world-cfg '{}' --depth 2 --width 64 --heads 4 --mix 4 [--sink 1] --out runs/X
    python harness.py evalset --world <name> --world-cfg '{}' --Ns 16,64,256,1024,4096 --out runs/es
    python harness.py eval    --run runs/X --evalset runs/es

Library functions: `load_run`, `predict`, `query_attention`, `hidden_states`, `nll`, `kl_bayes_model`, `moments`.
For world-specific constructed-context tests, see `diag_junta.py`.

To run it in the cloud, copy `launch.sh` and `run_pilot.sh` (on-demand L4, code packed as tar+base64; the VM
deletes itself). The harness has been tested with W1 only. Categorical targets are implemented but have not been
exercised by any world yet.

Added 2026-10-28 for W1b; all of these default to off:
- `--sum-head K`: an unnormalised precision head per layer.
- `--nmode batch --tok-budget T`: one N per batch, with batch size T/N.
- `--init RUN`: continue a run from its final weights, keeping its curve.
- Eval summaries now also include `kl_var_part` and `kl_mean_part` (a moment-matched split of the KL).
- `test_harness.py` checks that padding and context order do not change predictions.
