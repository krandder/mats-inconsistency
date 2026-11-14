# Artifacts kept on General, not in git

Model checkpoints, binary arrays, bulky raw run outputs and reviewer CLI transcripts are kept out of `mats/` (the repo
`.gitignore` also ignores `*.pt`, `*.npz` and `*.npy`). They stay on General. `ARTIFACTS.tsv`, next to this page, lists every
such file with its size, SHA-256 and path on General; this page summarises it.

Snapshot: the source trees were read at 2026-10-28T20:28Z; `work/cm-interp/` again at 22:59Z; `work/world-fa/` again at
2026-10-29T00:54Z (World FA is parked, so its list is final for now); `work/cm-interp-R/` again at 03:29Z; `work/arc-noisy/` at
05:00Z and its `tails/` at 10:10Z, 12:14Z and 14:02Z (final); `work/algzoo-noisy/` again at 06:43Z and 19:30Z (final);
`work/cm-interp-B/` again at 16:43Z (through E17), 19:30Z (through E25), 21:57Z (through E26 and E18b),
2026-10-30T01:08Z (through E36, without E30), 2026-10-31T05:04Z (through E65, without E64, E64b and E66) and
2026-10-31T20:38Z (through E73, without E74 and E75)
and 2026-11-01T16:55Z (through E76, without E77); `work/gate-entropy/` and `work/random-phases/` at 16:43Z,
`work/sparse-codes/` and `work/boundary-family/` at 19:30Z, `work/hardness/`, `work/lyapunov-dimension/`,
`work/nongauss-family/` and `work/scale-free-theory/` at 21:57Z, `work/narrow-deep/` at 2026-10-30T01:08Z and again at
2026-10-31T05:04Z (with its section 10), `work/scale-free-exp/` and `work/scale-free-ngo/` at 2026-10-30T01:08Z,
`work/narrow-deep-m1/`, `work/cut-rank/`, `work/kstar-frames/` and `work/tractable-gates/` at 2026-10-31T05:04Z, and
`work/cut-rank-ext/`, `work/unrestricted/`, `work/stationary/`, `work/stationary-numerics/` and `work/input-bits/` at
2026-10-31T20:38Z (all finished). The ARC and AlgZoo lanes are final. Phase B's E77 was being registered at
2026-11-01T16:55Z and is not listed, so phase B's list may grow. `/home/agent` and `/home/agent/data/home` are the same directory on General.

## Per lane

| Lane (repo folder) | On General (`work/` = `/home/agent/workspaces/mats/work/`) | Kept off-repo | Files | Size |
|---|---|---|---|---|
| `mats/worlds/pfn-dedup/` | `work/pfn-dedup/` | 64 checkpoints | 64 | 101.4 MB |
| `mats/worlds/tiny-pfn/` | `work/tiny-pfn/` | 270 checkpoints, 137 arrays | 407 | 152.8 MB |
| `mats/worlds/worlds/` | `work/worlds/` | 3,098 arrays, 192 checkpoints | 3,290 | 811.7 MB |
| `mats/worlds/world-fa/` | `work/world-fa/` | 62 checkpoints, 60 arrays | 122 | 45.0 MB |
| `mats/worlds/w2-mech/` | `work/w2-mech/` | 2 arrays | 2 | 8.4 MB |
| `mats/interp/phaseA/` | `work/cm-interp/` | 23 arrays | 23 | 217.3 MB |
| `mats/interp/phaseB/` | `work/cm-interp-B/` | 14,654 checkpoints, 3,634 arrays | 18,288 | 3.99 GB |
| `mats/interp/tiny-pfn-interp/` | `work/tiny-pfn-interp/` | 120 checkpoints, 66 arrays | 186 | 55.3 MB |
| `mats/interp/estimability/` | `work/estimability/` | 46 checkpoints | 46 | 14.4 MB |
| `mats/interp/arc-noisy/` | `work/arc-noisy/` | 1,295 raw run outputs (JSON, logs), 855 arrays | 2,150 | 2.72 GB |
| `mats/interp/algzoo-noisy/` | `work/algzoo-noisy/` | 152 arrays, 46 checkpoints | 198 | 3.2 MB |
| `mats/interp/random-phases/` | `work/random-phases/` | 30 arrays | 30 | 5.1 MB |
| `mats/interp/boundary-family/` | `work/boundary-family/` | 1 checkpoint | 1 | 0.0 MB |
| `mats/interp/sparse-codes/` | `work/sparse-codes/` | 58 checkpoints, 10 arrays | 68 | 438.2 MB |
| `mats/interp/lyapunov-dimension/` | `work/lyapunov-dimension/` | 21 arrays | 21 | 12.7 MB |
| `mats/interp/nongauss-family/` | `work/nongauss-family/` | 62 arrays | 62 | 43.8 MB |
| `mats/interp/scale-free-theory/` | `work/scale-free-theory/` | 27 checkpoints | 27 | 1.9 MB |
| `mats/interp/narrow-deep/` | `work/narrow-deep/` | 164 arrays, 6 raw run outputs (JSON) | 170 | 55.5 MB |
| `mats/interp/scale-free-exp/` | `work/scale-free-exp/` | 840 checkpoints, 431 arrays | 1,271 | 487.3 MB |
| `mats/interp/cut-rank/` | `work/cut-rank/` | 23 arrays | 23 | 605.0 MB |
| `mats/interp/kstar-frames/` | `work/kstar-frames/` | 639 arrays | 639 | 10.15 GB |
| `mats/interp/tractable-gates/` | `work/tractable-gates/` | 97 arrays, 36 checkpoints | 133 | 213.0 MB |
| `mats/interp/cut-rank-ext/` | `work/cut-rank-ext/` | 18 arrays, 3 raw run outputs (logs) | 21 | 162.4 MB |
| `mats/interp/stationary-numerics/` | `work/stationary-numerics/` | 8 raw run outputs (JSON), 7 arrays | 15 | 68.0 MB |
| review folders (see below) | `/tmp/arcn_review/`, `/tmp/arcn_review2/`, `/tmp/pbrev/`, `/tmp/prrev/`, `work/algzoo-noisy/`, `work/boundary-family/`, `work/c6/`, `work/cut-rank-ext/`, `work/cut-rank/`, `work/gate-entropy/`, `work/hardness/`, `work/input-bits/`, `work/kstar-frames/`, `work/lyapunov-dimension/`, `work/narrow-deep-m1/`, `work/narrow-deep/`, `work/nongauss-family/`, `work/scale-free-ngo/`, `work/scale-free-theory/`, `work/sparse-codes/`, `work/stationary-numerics/`, `work/stationary/`, `work/tractable-gates/`, `work/unrestricted/`, `work/w2-mech/`, `work/world-critique/` | 80 reviewer CLI transcripts | 80 | 26.1 MB |
| **Total** | | | **27,337** | **20.31 GB** |

Checkpoints are `*.pt`, plus the 46 published AlgZoo weights (`*.pth`) downloaded into `work/algzoo-noisy/ext/weights/`. Arrays
are `*.npz`, one `*.npy` (the Kerdock table in `work/arc-noisy/`) and four `*.pkl` (two in `work/w2-mech/out/`, two in
`work/cm-interp-B/results/`); all were left out
whatever their size. The raw run outputs are the ARC tails study's truth and run files (`work/arc-noisy/tails/runs/`), left
out under the size rule, and the narrow-deep lane's six per-input collapse records (`work/narrow-deep/runs/collapse_plain_*.json`,
5-7 MB each), which its `scratch/an_p2.py` reads; since 31 October every other file over 1 MiB is listed the same way, such as the three
large logs of `work/cut-rank-ext/logs/` and eight run JSON files of `work/stationary-numerics/runs/`. The reviewer CLI transcripts are the stderr logs of the Codex, Kimi and Gemini review runs; they hold
the CLI session, including the fleet bootstrap and memory-recall output, and no research content. The reviews themselves are
in the repo. The transcripts under `/tmp` belong to reviews whose only other copy was in `/tmp`, which is not durable, so
the repo copies of those reviews are the kept ones. The hardness lane's three GPT-6 reviews exist only as such transcripts
(`work/hardness/review/review_gpt6_r*.md`); the final message of each is in the repo as `review_gpt6_r*_final.md`. The same
holds for the review of `narrow-deep` section 10 and the two reviews of `narrow-deep-m1`, whose final messages are in the repo as
`review/*_answer.md`. The co-prover transcripts (`coprover/*.raw`) of `cut-rank`, `tractable-gates`, `cut-rank-ext`,
`unrestricted`, `stationary` and `input-bits` are listed with the reviewers'; their answers are in the repo. One memory-recall printout (`work/nongauss-family/logs/recall1.txt`) is listed
with the transcripts.

## How to check

Columns of `ARTIFACTS.tsv` (tab-separated, one header row): `path` (where the file would sit in this repo), `size` (bytes),
`sha256`, `lane` (repo folder), `general_path` (absolute path on General), `kind` (`checkpoint`, `array`, `raw-output`,
`transcript`). From the repo root on General:

```sh
# prints only the files that changed or are missing
awk -F'\t' 'NR > 1 {print $3 "  " $5}' mats/ARTIFACTS.tsv | sha256sum -c --quiet
# the paths to back up
awk -F'\t' 'NR > 1 {print $5}' mats/ARTIFACTS.tsv
```

## Not listed file by file

- Paper build intermediates in `outbox/binding-plateau-paper-2026-10-28/`: `paper.aux`, `paper.bbl`, `paper.blg`, `paper.log`,
  `paper.out` (rebuilt by the command in its `BUILD.md`).
- The backup page `work/interp-page/index.html.bak-20261028-v3`.
- Every `__pycache__/` and `.pytest_cache/` directory.
- 44 dangling symlinks under `work/worlds/runs_w2b_1028/{a,b}/main/` and `work/worlds/runs_w2c_1028/{a,b}/main/`
  (`es_*`). They point to `/root/soup/w2b/es/...` on VMs that no longer exist. The symlinks that point inside a copied folder
  were kept, rewritten as relative links (for example `runs_w2c_1028/combined/*` and `pfn-dedup/runs_final/*`).
- The VM run directories `/tmp/l4runs/<run-id>/` on General (the checkout synced to each VM, plus what came back; since
  31 October `/tmp/l4runs` is a symlink to `/home/agent/data/l4runs/`, and the kstar-frames runs `kstar1` to `kstar5` are archived
  under `/home/agent/data/archive/mats/l4runs-2026-10-31/`). They were
  not copied or listed file by file. The outputs that matter were copied into the `work/` folders above, and those copies are what
  this repo and this list cover. MATS run ids there (58):
  estd4-1027, estd8-1027, estd8s1-1027, estd8s2-1027, estpilot-1027, estsweep-1027, estsweep-1027b, fa-bind-1028c,
  fa-bind-1028d, fa-bind-1028e, fa-decomp-1028f, fa-pilot-1028a, fa-pilot-1028b, pfndedup-1026, pfndedup-1026b,
  pfndedup-1026c, pfndedup-1027a, pfndedup-1027b, pfndedup-1027c, pfndedup-1027d, pfndedup-1027e, pfndedup-1027f,
  pfndedup-1027g, pfndedupa2-1028a, pfndedupa2-1028b, pfndedupa2-1028c, tinyb1-1028, tinyheads-1028, w1b-1028-s0,
  w1b-1028-s0b, w1b-1028-s1, w1b-1028-s1b, w1b-1028-s2, w1b2-1028-s0, w1b2-1028-s1, w1b2-1028-s1b, w1b2-1028-s2,
  w1b2-1028-s2b, w1junta-1028a, w2b-1028a, w2b-1028b, w2c-1028a, w2c-1028b, w2market-1028a, w3c2-anneal-1028e,
  w3c2-diag-1028f, w3c2-looped-1028, w3c2-looped-1028b, w3c2-s0-1028c, w3c2-s1-1028c, w3c2-s2-1028c,
  w3c2-screen-1028d, w3c2-untied-1028, w3c2-untied-1028b, w3eq-s0-1028g, w3eq-s0-1028h, w3eq-s1-1028g, w3eq-s2-1028g.
- The paper copies in `work/hardness/papers/` (Impagliazzo, Kabanets and Wigderson 2002; Chen and Tell 2021; Wu et al., arXiv
  2605.05179; two arXiv API listings): third-party, cited in `mats/interp/hardness/notes/sources.md`.
- The texts in `work/scale-free-ngo/sources/` (Ngo's posts and comments, Lee, Kondor and Ngo's arXiv 2509.06701, Dietrich and
  List, Ostrovsky 2012, two arXiv API listings): other authors' work, cited in `mats/interp/scale-free-ngo/CITATIONS.md`.
- Unmodified third-party clones: ARC's `mlp_cumulant_propagation` at 93d091a (`work/arc-noisy/ext/` and
  `work/boundary-family/ext/`), AIcrowd's
  `whest-starterkit` at 96e308c and `flopscope` at 8632052 (`work/arc-noisy/ext2/`), and ARC's `alg-zoo` at 4433f50 with ARC's
  notes as PDF and text (`work/algzoo-noisy/ext/`). The AlgZoo weights downloaded there are listed.
- From the review folders in `/tmp`: the reviewers' snapshots of our own code (the reviewed hashes are in each folder's
  `reviewed_sha256.txt`), copies of ARC's code and of our run data, and the reviewers' recomputed pickles.
