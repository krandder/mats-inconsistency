# Site and document builders (snapshot from the fleet workspaces, 2026-10-18)

- `experiments-page/experiments_page.py` generates the /experiments pages (streams, sub-streams, experiment cards, archive) from the `STREAMS` table. Paths inside are fleet-local (`/home/agent/data/...`).
- `direction-page/build_direction.py` renders the research direction note (markdown with KaTeX and three PDG figures) into `/direction/index.html`; long equations get multi-line forms (`NARROW`).
- `gdoc-note/` rebuilds the Google Doc copy of the note (equations rendered to PNG; `make.sh --push`).

Specs and amendments for rounds 4-10 are in `docs/specs/`; independent audits of the write-ups are in `docs/audits/`.
