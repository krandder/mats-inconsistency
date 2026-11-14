# Spot-check: astra full-4096 exact Bayes (N=32, a=1.0), opus-mats, 2026-10-24

Verdict: PASS. GPU numbers match the original qagents CPU reference to float64 round-off.

Checked (CPU, torch 1 thread, no GPU, no learned losses/result.json opened):
- Artifact results/explore/astra-pilot-bidir-s5-a1/oracle/full.pt loaded via full_reference.load_artifact (f5ab3ae); data.py unchanged from dbaa7c8 (sha256 a3c0a3f1...).
- GPU file is oracle/gpu/probabilities.json (not gpu/probabilities.json at the run root): status COMPLETE, 4096/4096; its identity (data, metadata, data.py hashes) matches the artifact. 256 contiguous 16-episode chunks.
- Regenerating data.sample(4096, 32, seed 741202, a=1.0) on CPU reproduces the artifact's public x, keys, q, y bit-for-bit. Labels y are consistent with metadata bits and q.
- Env: data.py's U, V, W and Zipf R prior equal q_env.Env's (np.allclose); max |U^T V| = 1.4e-16. N=32 enters only through the sampled agent count; q_env.bayes_latent is N-agnostic.
- 20 episodes, indices 0, 205, ..., 3895 (11 parity / 9 AND; 10 of size 2, 10 of size 3). Recomputed with q_env.Env(a=1.0, nu=1, sigma=0.5).bayes_latent (public x, keys, q) and Env.bayes (true roles, Woodbury form):
  - max |p_gpu - p_cpu|, feasible latent-R Bayes: 1.6e-13 (max |log p diff| 3.5e-13)
  - max |p_gpu - p_cpu|, true-R oracle: 3.6e-15
  Per-episode values and script: /tmp/astra_spotcheck.json, /tmp/astra_spotcheck.py.

Data alignment and leakage:
- The feasible reference uses only x, keys, q (full_reference.probabilities; data.bayes_latent touches no roles, bits or y). OK.
- The simplified oracle (independent bit posteriors with log-odds 2a(U^T x)_j) is exact here because U is orthogonal to V; it agrees with the full Woodbury oracle. OK.
- Episode ordering: the chunk loader enforces contiguous coverage; GPU and CPU indices align. OK.

Remaining (not blocking for the pilot):
1. verify_trainer_reference binds only the first 64 episodes to the trainer's data. Bind the trainer's full final-4096 public digest to data_sha256 dc880e15... before any learned number is scored against these references.
2. full.pt holds privileged metadata (bits, roles) next to the public inputs. Add a test that the scorer and trainer read only x, keys, q, y from it.
