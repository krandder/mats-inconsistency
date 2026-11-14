# Repo sync tools

Scripts used to copy the MATS work from `/home/agent/workspaces/mats` on General into `mats/` (PRs #3 onward). Run them from a separate worktree of this repo; they assume General's paths.

- `stage.py PLAN.json MANIFEST.json [--dry]` copies each plan job (`src`, `dst`, optional `files`, `skip_globs`, `artifact_globs`, `transcript_globs`, `keep_globs`, `skip_ext`) into the worktree. It skips caches and backups and does not copy checkpoints, arrays, files over 1 MB (listed as raw outputs unless they match `keep_globs`), `artifact_globs` matches or reviewer CLI transcripts; those go to `MANIFEST.json` with size, SHA-256, repo path and lane. It stops before copying anything if a job's `src` folder or a listed file is missing.
- `scan.py DIR...` is the regex secret scan (keys, tokens, bearer, passwords, URL credentials, `key=` plus 32 hex, bare 32-hex). `KNOWN_SECRETS_FILE` adds literal strings from a file kept outside the repo, such as the gate's guest keys.
- `commit.sh WORKTREE MSGFILE` commits `mats/` only. It force-adds ignored `*.log` files and stops on any other ignored file unless it matches `ALLOW_IGNORED_RE`.
- `artifacts_tsv.py MATS_DIR MANIFEST.json...` replaces, in `ARTIFACTS.tsv`, the rows under the folders and files the sync re-read, and skips files already committed under `mats/`. `artifacts_summary.py MATS_DIR` then rebuilds the per-lane table in `ARTIFACTS.md`; `--check` only verifies it.
