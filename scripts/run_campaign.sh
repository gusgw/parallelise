#!/usr/bin/env bash
# Runs the masked-versus-unmasked campaign: 4 models x 4 problems x 2 backends x 2 variants,
# 64 cells, each cell being one run for each of the three agents.
#
# The three agents of a cell run at the same time; cells run one after another, so that the
# benchmark timings of one cell are not disturbed by another. The order is model by model,
# with the unmasked and masked variants of the same cell next to each other.
#
# The masking tool that the masked variant uses, and the report tables built from the records,
# are in the subpackage src/parallelise/study/ (see its __init__.py); this script holds no
# Python logic of its own.
#
# Each run is one "parallelise run" inside the cpu container image. The model gateway
# address is read from the environment (PARALLELISE_GATEWAY_URL) and is never stored here.
#
# Settings that may be overridden in the environment:
#   IMAGE      container image                (default parallelise:cpu)
#   BASE       folder holding work/ and results/   (default the current folder)
#   MODELS, PROBLEMS, BACKENDS   space-separated lists, to run a part of the design
set -uo pipefail

: "${PARALLELISE_GATEWAY_URL:?Set PARALLELISE_GATEWAY_URL to the model gateway address.}"
IMAGE="${IMAGE:-parallelise:cpu}"
BASE="${BASE:-$PWD}"
WORK="$BASE/work/campaign-2026-10"
OUT="$BASE/results/campaign-2026-10"
LOGS="$OUT/logs"
mkdir -p "$WORK" "$OUT" "$LOGS"
# The container runs as an ordinary user (identifier 10001), so these must be writable.
chmod 777 "$WORK" "$OUT" "$LOGS"

# Memory discipline. Every container runs under an explicit memory limit with no swap, one
# container runs at a time, and a cell does not start while the host has less than
# MIN_FREE_GB free. The MPI benchmarks of the prefix sum, decay and reservoir problems
# allocate 2 GiB per rank at the production length, so the benchmark length and the rank
# count are capped to fit the limit. These caps change the benchmark conditions from the
# October 2026 campaign, which used the production sizes and no limits.
MEMORY="${MEMORY:-8g}"
MIN_FREE_GB="${MIN_FREE_GB:-12}"
export BENCH_N="${BENCH_N:-20000000}"
export BENCH_MAX_RANKS="${BENCH_MAX_RANKS:-4}"

wait_for_memory() {
    local free_gb tries=0
    while :; do
        free_gb=$(free -g | awk '/^Mem:/ {print $7}')
        if [ "${free_gb:-0}" -ge "$MIN_FREE_GB" ]; then return 0; fi
        tries=$((tries + 1))
        if [ "$tries" -gt 60 ]; then
            say "ABORT: only ${free_gb} GB free after waiting an hour; need $MIN_FREE_GB GB."
            exit 3
        fi
        say "waiting: ${free_gb} GB free, need $MIN_FREE_GB GB"
        sleep 60
    done
}

MODELS="${MODELS:-claude-sonnet-4.5 kimi-k2.5 gpt-oss-120b llama-3.3-70b}"
PROBLEMS="${PROBLEMS:-integrate jacobi scan cg}"
BACKENDS="${BACKENDS:-openmp mpi}"
AGENTS="basic pydantic langgraph"
NOTES="masked campaign, October 2026"
QUANT="not disclosed by the provider"

alias_for() {
    case "$1" in
        integrate) echo kernel_a ;;
        jacobi) echo kernel_b ;;
        scan) echo kernel_c ;;
        cg) echo kernel_d ;;
    esac
}

serving_for() {
    case "$1" in
        claude-sonnet-4.5) id="au.anthropic.claude-sonnet-4-5-20250929-v1:0" ;;
        kimi-k2.5) id="moonshotai.kimi-k2.5" ;;
        gpt-oss-120b) id="openai.gpt-oss-120b-1:0" ;;
        llama-3.3-70b) id="us.meta.llama3-3-70b-instruct-v1:0" ;;
    esac
    echo "Amazon Bedrock, $id, through a LiteLLM gateway"
}

say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }

# run_one MODEL PROBLEM BACKEND VARIANT AGENT
# Each variant and agent has its own work root, because a masked run and the three agents
# would otherwise share and delete one another's folders.
run_one() {
    local model="$1" problem="$2" backend="$3" variant="$4" agent="$5"
    local extra=()
    if [ "$variant" = masked ]; then
        extra=(--masked "$(alias_for "$problem")")
    fi
    local tag="${model}_${problem}_${backend}_${variant}_${agent}"
    docker run --rm \
        --memory="$MEMORY" --memory-swap="$MEMORY" \
        -v "$WORK:/work" -v "$OUT:/results" \
        -e BENCH_N -e BENCH_MAX_RANKS \
        -e PARALLELISE_GATEWAY_URL \
        -e PARALLELISE_MODEL="$model" \
        -e PARALLELISE_TEMPERATURE=0.2 \
        -e PARALLELISE_RESULTS=/results/runs.jsonl \
        "$IMAGE" run "$problem" "$backend" --agent "$agent" "${extra[@]}" \
        --work-root "/work/$variant/$agent" \
        --serving "$(serving_for "$model")" --quantisation "$QUANT" --notes "$NOTES" \
        >"$LOGS/$tag.log" 2>&1
}

total=0
for m in $MODELS; do for p in $PROBLEMS; do for b in $BACKENDS; do total=$((total + 2)); done; done; done
n=0
say "CAMPAIGN START: $total cells, three agents each, one container at a time under a $MEMORY limit."
for model in $MODELS; do
    for problem in $PROBLEMS; do
        for backend in $BACKENDS; do
            for variant in unmasked masked; do
                n=$((n + 1))
                say "CELL $n/$total START model=$model problem=$problem backend=$backend variant=$variant"
                for agent in $AGENTS; do
                    wait_for_memory
                    run_one "$model" "$problem" "$backend" "$variant" "$agent"
                done
                say "CELL $n/$total END   model=$model problem=$problem backend=$backend variant=$variant"
            done
        done
    done
done
say "ALL DONE: $n cells finished. Records are in results/campaign-2026-10/runs.jsonl."
