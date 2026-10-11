#!/usr/bin/env bash
# Sweeps rank counts and assembles the scaling table that the benchmark checker validates:
#   ranks,time_s,speedup,efficiency,throughput
# It runs ./bench (which prints "ranks,time_s") at 1, 2, 4, ... ranks up to the core
# count, using the single-rank time as the baseline for the speed-up.
#   bench_sweep.sh [n]      (n is the problem size, default 200000000)
# BENCH_MAX_RANKS (environment) caps the width of the sweep. Some benchmarks need memory
# that grows with the rank count, because every rank holds the whole input, and could
# otherwise exhaust a shared machine at high widths.
set -euo pipefail
cd "$(dirname "$0")"
n="${1:-200000000}"
max=$(nproc)
if [ -n "${BENCH_MAX_RANKS:-}" ] && [ "$BENCH_MAX_RANKS" -lt "$max" ]; then
    max="$BENCH_MAX_RANKS"
fi

ranks=(); r=1
while [ "$r" -le "$max" ]; do ranks+=("$r"); r=$((r * 2)); done
[ "${ranks[-1]}" -ne "$max" ] && ranks+=("$max")

echo "ranks,time_s,speedup,efficiency,throughput"
t1=""
for r in "${ranks[@]}"; do
    line=$(mpirun --oversubscribe -np "$r" ./bench "$n" 2>/dev/null)   # prints "ranks,time_s"
    t=$(printf '%s' "$line" | cut -d, -f2)
    [ -z "$t1" ] && t1="$t"
    python3 - "$r" "$t" "$t1" "$n" <<'PY'
import sys
r = int(sys.argv[1]); t = float(sys.argv[2]); t1 = float(sys.argv[3]); n = float(sys.argv[4])
su = (t1 / t) if t > 0 else 0.0
print("%d,%.6f,%.6f,%.6f,%.1f" % (r, t, su, su / r, n / t / 1.0e6))
PY
done
