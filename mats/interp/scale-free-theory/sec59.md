### 5.9 Which tasks are untangled, by definition

| Task | D1 fixed cuts | D2 routed | D3 rank | D4 item treewidth | D5 data MI | SFU |
|---|---|---|---|---|---|---|
| Conditionally independent sources, exponential family | exact: infinite (continuous) or d log2(s + 1) bits (d count statistics); to posterior precision about (d/2) log s | same | parameter-observation rank <= d + 2 (log space) | N - 1 | about (d/2) log s | 1 open latent, d numbers, 1 round |
| Same, smooth non-exponential (tiny-PFN) | polylog at tolerance | same | r(delta/N) | N - 1 | log | 1 latent; 48-64 numbers in 1 round, about 4-8 in 2 rounds (N >= 64) |
| G shared-error groups, labels visible | about G r at a positional cut | r (group-aligned) | as D1/D2 | >= group size | log | log G label bits; open = straddling groups |
| G groups, membership hidden, T past questions | functions of about G T latents | no labels (re-route after a pilot) | grows fast | N - 1 | log | 1 round: exponential in the open latents; with a pilot and re-routing: local statistics, if label symmetry is broken |
| Keyed aggregation / binding (World FA) | Omega(N) | O(1) + log N label bits | full rank | high | 0 | untangled with routing |
| k-hop lookup (pointer chasing) | Omega(n) at a cut separating the tables | k sequential rounds: O(k log n) bits; pointer doubling: log k rounds, Omega(n log n) bits per round | high | high | 0 | untangled with k rounds |
| Parity | 1 bit | 1 bit | 1 (+-1 form) | n/a | 0 | untangled; hard for SQ learners |
| GF(2) quadratic on an expander | Theta(N) | Theta(N) | 2^{Theta(N)} (log-rank Theta(N)) | Theta(N) | 0 | tangled; uniform expectation still polynomial |
| Correlated errors, inverse-Wishart (World FA (b)) | full posterior of Sigma: T \|S\| numbers | same | - | dense | - | predictive: O(T^2) additive numbers per cut (time coordinates) |
| One-query retrieval (attention) | volume without broadcast | - | - | - | - | d + 2 numbers after broadcasting the query |
