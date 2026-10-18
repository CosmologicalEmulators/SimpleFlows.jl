#!/usr/bin/env bash
# Launch ELG2 validation cells, ONE FRESH PROCESS PER CELL, sequentially.
# Dry run by default: prints the commands. Pass --run to execute.
#
#   examples/elg2_local/launch_cells.sh [--run] [--seeds "42 43"] [--backends "zygote"] \
#       [--epochs N]  (default 150; smoke default 2) [--timeout SECONDS] [--identity-init true|false]
# New identity-initialization matrix (do not launch before the parent gate):
#   launch_cells.sh --run --identity-init true  --out-root /home/marcobonici/Desktop/work/simpleflows_elg2_validation_20261017_identity
# Raw legacy baseline: --identity-init false [--out-root DIR] [--smoke]
#
# Existing cells are never overwritten (run_cell.jl fails if the directory exists);
# a failed cell does not stop the loop. Logs go to <out-root>/logs/<cell>.log.
set -u
JULIA=/home/marcobonici/.juliaup/bin/julia
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=/home/marcobonici/Desktop/work/simpleflows_elg2_validation_20261017
IDENT=""; RUN=0; SEEDS="42 43"; BACKENDS="zygote"; EPOCHS=""; SMOKE=""   # empty epochs: 150 (full) or 2 (smoke)
die() { echo "launch_cells.sh: $*" >&2; exit 2; }
need() { [ "$1" -ge 2 ] || die "option $2 needs a value"; }
TIMEOUT="${TIMEOUT:-}"      # optional external per-cell limit in seconds, e.g. TIMEOUT=7200
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=1; shift;;
    --seeds) need $# "$1"; SEEDS="$2"; shift 2;;
    --backends) need $# "$1"; BACKENDS="$2"; shift 2;;
    --epochs) need $# "$1"; EPOCHS="$2"; shift 2;;
    --out-root) need $# "$1"; OUT="$2"; shift 2;;
    --timeout) need $# "$1"; TIMEOUT="$2"; shift 2;;
    --identity-init) need $# "$1"; case "$2" in true|false) IDENT="$2";; *) die "--identity-init must be true or false";; esac; shift 2;;
    --smoke) SMOKE="--smoke"; shift;;
    *) die "unknown argument $1";;
  esac
done
[ -f "$HERE/run_cell.jl" ] || die "run_cell.jl not found next to the launcher"
[ -x "$JULIA" ] || die "Julia not found at $JULIA"
for s in $SEEDS; do case "$s" in ''|*[!0-9]*) die "bad seed '$s'";; esac; done
for b in $BACKENDS; do case "$b" in zygote|reactant) ;; *) die "bad backend '$b'";; esac; done
[ -n "$SEEDS" ] && [ -n "$BACKENDS" ] || die "--seeds and --backends must be non-empty"
# Nothing is created in dry-run mode.
[ "$RUN" = 1 ] && { mkdir -p "$OUT/logs" || die "cannot create $OUT/logs"; }
FAILED=()
for seed in $SEEDS; do for backend in $BACKENDS; do for arch in RealNVP NSF MAF; do for norm in minmax whiten; do
  cell="${arch}_${norm}_${backend}_seed${seed}"
  sub=cells; [ -n "$SMOKE" ] && sub=smoke
  cmd=(env JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 ${TIMEOUT:+timeout --kill-after=60 "$TIMEOUT"} "$JULIA" --project="$HERE" "$HERE/run_cell.jl"
       --arch "$arch" --normalization "$norm" --backend "$backend" --seed "$seed" ${EPOCHS:+--epochs "$EPOCHS"} --out-root "$OUT" ${IDENT:+--identity-init "$IDENT"} $SMOKE)
  if [ "$RUN" = 1 ]; then
    d="$OUT/$sub/$cell"; logf="$OUT/logs/${cell}${SMOKE:+_smoke}.log"
    if [ -e "$d" ] || [ -e "$logf" ]; then
      echo "REFUSING $cell: cell directory or log already exists (not overwriting)" >&2
      FAILED+=("$cell(exists)"); continue
    fi
    echo "[$(date '+%F %T')] START $cell"
    "${cmd[@]}" > "$logf" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] END $cell rc=$rc"
    if [ "$rc" -ne 0 ]; then
      FAILED+=("$cell(rc=$rc)")
      # killed/timed-out cells never wrote a terminal marker: leave one (the directory is kept)
      d="$OUT/$sub/$cell"
      if [ -d "$d" ] && [ ! -e "$d/FAILED" ] && [ ! -e "$d/COMPLETE" ]; then
        echo "killed or timed out by the launcher, rc=$rc, $(date '+%F %T')" > "$d/FAILED"
      fi
    fi
  else
    echo "${cmd[*]} > $OUT/logs/${cell}${SMOKE:+_smoke}.log 2>&1"
  fi
done; done; done; done
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "FAILED CELLS: ${FAILED[*]}" >&2
  exit 1
fi
exit 0
