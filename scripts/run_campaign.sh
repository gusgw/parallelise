#!/usr/bin/env bash
# Runs the masked-versus-unmasked campaign: 4 models x 4 problems x 2 backends x 2 variants,
# 64 cells, each cell being one run for each of the three agents.
#
# The three agents of a cell run at the same time; cells run one after another, so that the
# benchmark timings of one cell are not disturbed by another. The order is model by model,
# with the unmasked and masked variants of the same cell next to each other.
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
        -v "$WORK:/work" -v "$OUT:/results" \
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
say "CAMPAIGN START: $total cells, three agents each."
for model in $MODELS; do
    for problem in $PROBLEMS; do
        for backend in $BACKENDS; do
            for variant in unmasked masked; do
                n=$((n + 1))
                say "CELL $n/$total START model=$model problem=$problem backend=$backend variant=$variant"
                for agent in $AGENTS; do
                    run_one "$model" "$problem" "$backend" "$variant" "$agent" &
                done
                wait
                say "CELL $n/$total END   model=$model problem=$problem backend=$backend variant=$variant"
            done
        done
    done
done
say "ALL DONE: $n cells finished. Records are in results/campaign-2026-10/runs.jsonl."
