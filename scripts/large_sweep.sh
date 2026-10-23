#!/usr/bin/env bash
# Large-host probe: one process per (cell, engine, protocol) under GNU Parallel.
#
# Usage: scripts/large_sweep.sh [output_root]
#        JOBS=8 RETRY_TIMEOUT=1800 scripts/large_sweep.sh /scratch/lukas/tmp/opencode/large_hosts_v2
#
# The main pass caps every job at 600 s.  Jobs that produced no row are retried
# once with a longer cap (VF2/VF2++ excluded: they are the known exponential
# cases), then the summary censors the remaining timeouts at 600 s.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-/scratch/lukas/tmp/opencode/large_hosts}"
JOBS="${JOBS:-$(nproc)}"
PY="${PY:-.venv/bin/python}"
RETRY_TIMEOUT="${RETRY_TIMEOUT:-1800}"
CORPUS="${CORPUS:-probe}"

mkdir -p "$OUT/rows" "$OUT/logs"

"$PY" -c "import sys; sys.path.insert(0, 'scripts'); from large_cells import corpus_engines, corpus_names; [print(c, e, p) for p in ('one-shot', 'amortized') for c in corpus_names('$CORPUS') for e in corpus_engines('$CORPUS')]" > "$OUT/jobs.txt"

echo "jobs: $(wc -l < "$OUT/jobs.txt")  parallel: $JOBS  output: $OUT"

parallel --will-cite --colsep ' ' --jobs "$JOBS" --timeout 600 \
    --joblog "$OUT/joblog.tsv" \
    "$PY -u scripts/large_cell.py --cell {1} --engine {2} --protocol {3} --output_dir $OUT/rows" \
    :::: "$OUT/jobs.txt" || true

"$PY" -u scripts/large_missing.py --jobs_file "$OUT/jobs.txt" --rows_dir "$OUT/rows" --exclude vf2,vf2pp > "$OUT/retry.txt"
echo "retry jobs: $(wc -l < "$OUT/retry.txt") (cap ${RETRY_TIMEOUT}s)"
if [[ -s "$OUT/retry.txt" ]]; then
    parallel --will-cite --colsep ' ' --jobs "$JOBS" --timeout "$RETRY_TIMEOUT" \
        --joblog "$OUT/joblog_retry.tsv" \
        "$PY -u scripts/large_cell.py --cell {1} --engine {2} --protocol {3} --output_dir $OUT/rows" \
        :::: "$OUT/retry.txt" || true
fi

"$PY" -u scripts/large_summary.py --rows_dir "$OUT/rows" --jobs_file "$OUT/jobs.txt" --output="$OUT/summary.csv"
