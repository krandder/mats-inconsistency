#!/usr/bin/env bash
# Build the soup's native CPU lane through Bend C emission and clang.
# Run on the provisioned GPU VM; installs nothing and never starts a soup run.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BEND_HOME=${BEND_HOME:-$ROOT/../research/bend2-scratch/bend-2.0.29}
OUT="$ROOT/build/native/soup"
COMPILER=${CC:-}
EMIT_ONLY=0
BEND_SOURCE=${BEND_SOURCE:-src/soup.bend}
usage() {
  cat <<'HELP'
Usage: scripts/build-native.sh [--bend-home DIR] [--out PATH] [--cc CLANG] [--emit-only]

Requires bun and the Bend 2.0.29 checkout. Full build requires clang >=14.
--emit-only checks the Bend program and writes PATH.c without requiring clang.
--cc (or CC) pins a compiler; an invalid explicit compiler fails rather than falling back.
Default output: build/native/soup, with .c, .check.log, .build.log and .build-info.txt.

This is the native CPU lane on the GPU VM. Marked ! cubes execute on CPU here; use build-gpu.sh for CUDA. No packages
are installed, no VM is created, and no benchmark or calibration run is started.

VM example (after provisioning clang and copying the pinned Bend checkout):
  BEND_HOME=/opt/bend-2.0.29 scripts/build-native.sh --cc clang-19
  SOUP_CMD="$PWD/build/native/soup --threads 16" scripts/run.sh --K '2^14' --F '2^12' --epochs 100 --out /path/to/new-output
HELP
}
die() { echo "build-native: $*" >&2; exit 2; }
while (($#)); do
  case "$1" in
    --bend-home|--out|--cc)
      (($# >= 2)) || die "$1 needs a value"
      case "$1" in
        --bend-home) BEND_HOME=$2;; --out) OUT=$2;; --cc) COMPILER=$2;;
      esac
      shift 2;;
    --emit-only) EMIT_ONLY=1; shift;;
    --help|-h) usage; exit 0;;
    *) die "unknown argument: $1";;
  esac
done
command -v bun >/dev/null || die 'bun is required'
[[ -f "$BEND_HOME/bend2/main.ts" ]] || die "missing $BEND_HOME/bend2/main.ts; set BEND_HOME to the pinned Bend checkout"
[[ -n "$OUT" && "$OUT" != *.c ]] || die '--out must name the executable, not its .c file'
# Make paths absolute before passing them to the compiler or recording provenance.
BEND_HOME=$(cd -- "$BEND_HOME" && pwd)
mkdir -p -- "$(dirname -- "$OUT")"
OUT="$(cd -- "$(dirname -- "$OUT")" && pwd)/$(basename -- "$OUT")"
cd -- "$ROOT"
if ! bun "$BEND_HOME/bend2/main.ts" "$BEND_SOURCE" --check-only > "$OUT.check.log" 2>&1; then
  cat "$OUT.check.log" >&2
  die 'Bend check failed'
fi
if ! grep -qx 'All terms check\.' "$OUT.check.log"; then
  cat "$OUT.check.log" >&2
  die 'Bend type/termination check failed or relies on unsafe definitions'
fi
bun "$BEND_HOME/bend2/main.ts" "$BEND_SOURCE" -o "$OUT.c"
# CPU-only compilation deliberately runs marked cubes on the host pool.
if ((EMIT_ONLY)); then
  echo "build-native: checked C emitted to $OUT.c (native compilation not attempted)"
  exit 0
fi
version_of() { "$1" --version 2>/dev/null | sed -nE 's/^(Apple )?([^ ]+ )?clang version ([0-9]+).*/\3/p' | head -1; }
if [[ -n "$COMPILER" ]]; then
  version=$(version_of "$COMPILER") || die "cannot execute compiler: $COMPILER"
  [[ "$version" =~ ^[0-9]+$ && "$version" -ge 14 ]] || die "explicit compiler $COMPILER must be clang >=14"
else
  for candidate in clang clang-22 clang-21 clang-20 clang-19 clang-18 clang-17 clang-16 clang-15 clang-14; do
    version=$(version_of "$candidate") || continue
    if [[ "$version" =~ ^[0-9]+$ && "$version" -ge 14 ]]; then COMPILER=$candidate; break; fi
  done
  [[ -n "$COMPILER" ]] || die 'clang >=14 not found; run this recipe on the provisioned VM (no installation attempted)'
fi
COMPILER=$(command -v -- "$COMPILER")
# Same CPU flags as Bend 2.0.29 cli_build; these pure/file-IO modules require
# pthread and libm, with no window/audio libraries.
"$COMPILER" -std=c11 -O3 "$OUT.c" -lpthread -lm -o "$OUT" > "$OUT.build.log" 2>&1 || {
  cat "$OUT.build.log" >&2
  die "clang failed; see $OUT.build.log"
}
{
  echo 'lane=native-cpu'
  date -u '+checked_at=%Y-%m-%dT%H:%M:%SZ'
  printf 'repository_commit=%s\n' "${SOUP_SOURCE_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
  printf 'bend_home=%s\ncompiler=%s\n' "$BEND_HOME" "$COMPILER"
  "$COMPILER" --version
  printf 'bun_version='; bun --version
  sha256sum "$OUT.c" "$OUT" "$BEND_HOME/bend2/main.ts" src/*.bend
} > "$OUT.build-info.txt"
echo "build-native: built $OUT (native CPU lane; no run started)"
