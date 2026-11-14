# Build and review

Checked at `2026-10-28T14:01:35Z` with Tectonic 0.17.0 at
`/home/agent/.local/share/astra-tools/bin/tectonic`.

Run these commands from this directory:

```sh
/home/agent/.local/share/astra-tools/bin/tectonic -X compile \
  --only-cached --keep-logs --keep-intermediates --outdir . paper.tex

rg -n 'Output written on paper.xdv' paper.log
if rg -n -i \
  'overfull|underfull|undefined|rerun to get|multiply defined' \
  paper.log paper.blg; then
  exit 1
fi

mkdir -p final-render
gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r120 \
  -sOutputFile=final-render/page-%d.png paper.pdf

sha256sum paper.pdf
```

The build produced 4 pages. The warning scan returned no matches. All four RGB
page renders were checked for clipping, overlap, missing content and layout
breaks; none were found. The figure and its labels are readable at page size.

`paper.pdf` SHA-256:
`47061e608cdfaac9ddac26de90434b47c82f1064d581aa7f2c44deb78d5eb1ee`
