#!/usr/bin/env bash
# A test of the whole tool as a new user would meet it, in a fresh copy of the repository.
#
# The script clones the repository into a temporary folder, installs it into a new virtual
# environment, and then, against a scripted stand-in for a language model (tests/fake_model_server.py),
# runs "parallelise health", converts the integrate problem for OpenMP, and runs
# "parallelise check" on the resulting work folder. If Docker is available it also builds the
# CPU image and repeats the health check and the conversion inside it, as the README
# describes; without Docker those steps are skipped with a notice.
#
# Usage:   scripts/smoke-test.sh
# Settings read from the environment:
#   REPO    the repository to clone (default: the repository holding this script)
#   KEEP    set to 1 to keep the temporary folder afterwards
# The script stops at the first failure and exits with a non-zero status. It needs git,
# python3 (3.11 or later), a C compiler and make; the conversion needs the OpenMP library
# that ships with gcc. It needs no real model and no network access beyond installing packages.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/.." && pwd)}"
TMP="$(mktemp -d)"
SERVER_PIDS=()

cleanup() {
    status=$?
    for pid in "${SERVER_PIDS[@]:-}"; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    done
    if [ "${KEEP:-0}" = 1 ]; then
        echo "Kept the temporary folder $TMP."
    else
        rm -rf "$TMP"
    fi
    if [ "$status" -ne 0 ]; then
        echo "SMOKE TEST FAILED (exit status $status). Read the last step above." >&2
    fi
}
trap cleanup EXIT

step() {
    echo
    echo "=== $1"
    echo "    Good result: $2"
}

# start_server NAME SCRIPT [extra options]: start the stand-in model in the background and
# set SERVER_URL to its address.
start_server() {
    local name="$1" script="$2"
    shift 2
    "$TMP/venv/bin/python" "$TMP/clone/tests/fake_model_server.py" --script "$script" \
        --url-file "$TMP/$name.url" "$@" >"$TMP/$name.log" 2>&1 &
    SERVER_PIDS+=("$!")
    for _ in $(seq 1 50); do
        [ -s "$TMP/$name.url" ] && break
        sleep 0.2
    done
    if [ ! -s "$TMP/$name.url" ]; then
        echo "The stand-in model server did not start. Its log:" >&2
        cat "$TMP/$name.log" >&2
        exit 1
    fi
    SERVER_URL="$(cat "$TMP/$name.url")"
}

step "Clone $REPO into a temporary folder" \
     "a folder $TMP/clone holding the repository's files."
git clone --quiet "$REPO" "$TMP/clone"

step "Create a virtual environment and install the package with the dev and report extras" \
     "pip finishes without an error."
python3 -m venv "$TMP/venv"
"$TMP/venv/bin/python" -m pip install --quiet --upgrade pip
"$TMP/venv/bin/python" -m pip install --quiet "$TMP/clone[dev,report]"
PARALLELISE="$TMP/venv/bin/parallelise"
cd "$TMP/clone"

step "Start the stand-in model for the health check" \
     "a web address such as http://127.0.0.1:PORT/v1 is printed."
start_server health health
HEALTH_URL="$SERVER_URL"
echo "    Serving at $HEALTH_URL"

step "parallelise health" \
     "the lines 'Reachable: yes' and 'Test chat call succeeded: yes', and exit status 0."
"$PARALLELISE" health --gateway-url "$HEALTH_URL" --model local

step "Start the stand-in model for the integrate conversion" \
     "a web address is printed. This model writes the known-good solution and then finishes."
start_server convert convert --name integrate --backend openmp \
    --reference-dir "$TMP/clone/problems/integrate/openmp/reference"
CONVERT_URL="$SERVER_URL"
echo "    Serving at $CONVERT_URL"

step "parallelise run integrate openmp --agent basic" \
     "the last line begins 'PASS: integrate on openmp', and exit status 0."
"$PARALLELISE" run problems/integrate openmp --agent basic --model local \
    --gateway-url "$CONVERT_URL" --work-root "$TMP/work" --results "$TMP/runs.jsonl"

WORK_FOLDER="$TMP/work/integrate/openmp/local/basic"
step "parallelise check on the finished work folder" \
     "the last line reads 'Check PASSED (failure stage: none, best speed-up: ...)'."
"$PARALLELISE" check "$WORK_FOLDER" --problem problems/integrate --backend openmp

step "parallelise study report on the new record" \
     "a table that shows one run with one pass."
"$PARALLELISE" study report --results "$TMP/runs.jsonl"

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    step "Build the CPU image" \
         "docker finishes and the image parallelise:smoke exists."
    docker build --quiet --target cpu -t parallelise:smoke "$TMP/clone"

    step "docker run parallelise:smoke health" \
         "the lines 'Reachable: yes' and 'Test chat call succeeded: yes', and exit status 0."
    docker run --rm --network host -e PARALLELISE_GATEWAY_URL="$HEALTH_URL" \
        -e PARALLELISE_MODEL=local parallelise:smoke health

    step "docker run parallelise:smoke run integrate openmp, with the work folders in a mounted folder" \
         "the last line begins 'PASS: integrate on openmp', and exit status 0."
    # The stand-in model plays its scripted conversation once, and the run outside the
    # container has already used it, so start a fresh one for the run inside.
    start_server convert-docker convert --name integrate --backend openmp \
        --reference-dir "$TMP/clone/problems/integrate/openmp/reference"
    DOCKER_CONVERT_URL="$SERVER_URL"
    mkdir -p "$TMP/out"
    # Run as the calling user, so that the files the container writes into the mounted
    # folder belong to that user and can be removed afterwards.
    docker run --rm --network host --user "$(id -u):$(id -g)" -v "$TMP/out:/work" \
        -e PARALLELISE_GATEWAY_URL="$DOCKER_CONVERT_URL" -e PARALLELISE_MODEL=local \
        -e PARALLELISE_RESULTS=/work/runs.jsonl -e PARALLELISE_JUDGE_SPEED \
        parallelise:smoke run integrate openmp --agent basic
    docker rmi --force parallelise:smoke >/dev/null 2>&1 || true
else
    echo
    echo "=== NOTICE: Docker is not available here, so the image build and the two runs inside"
    echo "    the image are skipped. Run this script on a machine with Docker to test them."
fi

echo
echo "SMOKE TEST PASSED."
