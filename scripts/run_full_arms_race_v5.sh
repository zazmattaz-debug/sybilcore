#!/usr/bin/env bash
# run_full_arms_race_v5.sh
#
# Wrapper for the v5-patched 1000-iteration arms-race tournament.
# Designed to be triggered by a scheduled task once the Gemini Pro daily
# quota has reset. Runs the patched ensemble against gemini-pro first,
# then (only on success) cross-confirms against gpt4o and grok.
#
# Secrets:
#   - Sources the workspace master .env so GEMINI_API_KEY / OPENAI_API_KEY
#     / XAI_API_KEY become available to the python process.
#   - Never echoes any secret value. set +x stays in effect for the whole
#     run.
#
# Exit codes:
#   0  -> all 3 model runs completed successfully
#   2  -> primary gemini-pro run failed
#   3  -> gemini-pro succeeded but gpt4o cross-confirm failed
#   4  -> gemini-pro + gpt4o succeeded but grok cross-confirm failed
#   5  -> environment problem (missing .env, missing python, etc.)

set -uo pipefail
# Explicitly do NOT set -x — we never want to echo secret values.

REPO_ROOT="/Users/zazumoloi/Desktop/Claude Code/rooms/engineering/sybilcore"
WORKSPACE_ROOT="/Users/zazumoloi/Desktop/Claude Code"
ENV_FILE="${WORKSPACE_ROOT}/.env"
LOG_DIR="${REPO_ROOT}/logs"
SUMMARY_FILE="${LOG_DIR}/arms_race_v5_full_summary.txt"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: workspace .env not found at ${ENV_FILE}" >&2
    exit 5
fi

# Source secrets without echoing them. `set -a` exports every variable
# the .env defines into the environment of subsequent commands.
set +x
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

# Pick a python interpreter. Prefer a project-local venv if one ever
# appears, otherwise fall back to the system python3.
if [[ -x "${REPO_ROOT}/.venv/bin/python3" ]]; then
    PYTHON="${REPO_ROOT}/.venv/bin/python3"
elif [[ -x "${REPO_ROOT}/venv/bin/python3" ]]; then
    PYTHON="${REPO_ROOT}/venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    echo "ERROR: no python3 interpreter available" >&2
    exit 5
fi

cd "${REPO_ROOT}" || {
    echo "ERROR: failed to cd into ${REPO_ROOT}" >&2
    exit 5
}

TS="$(date -u +%Y%m%d_%H%M%S)"

run_model() {
    local model="$1"
    local log_path="${LOG_DIR}/arms_race_v5_full_${model}_${TS}.log"
    echo "[$(date -u +%FT%TZ)] starting v5-patched 1000-iter run model=${model}" \
        | tee -a "${log_path}"
    "${PYTHON}" scripts/arms_race_smoke.py \
        --iterations 1000 \
        --model "${model}" \
        >> "${log_path}" 2>&1
    local rc=$?
    echo "[$(date -u +%FT%TZ)] model=${model} exit_code=${rc}" \
        | tee -a "${log_path}"
    return ${rc}
}

GEMINI_RC=0
GPT4O_RC=0
GROK_RC=0

run_model "gemini-pro"
GEMINI_RC=$?

if [[ ${GEMINI_RC} -ne 0 ]]; then
    {
        echo "ts=${TS} gemini=FAIL(${GEMINI_RC}) gpt4o=SKIPPED grok=SKIPPED"
    } > "${SUMMARY_FILE}"
    exit 2
fi

run_model "gpt4o"
GPT4O_RC=$?

if [[ ${GPT4O_RC} -ne 0 ]]; then
    {
        echo "ts=${TS} gemini=OK gpt4o=FAIL(${GPT4O_RC}) grok=SKIPPED"
    } > "${SUMMARY_FILE}"
    exit 3
fi

run_model "grok"
GROK_RC=$?

if [[ ${GROK_RC} -ne 0 ]]; then
    {
        echo "ts=${TS} gemini=OK gpt4o=OK grok=FAIL(${GROK_RC})"
    } > "${SUMMARY_FILE}"
    exit 4
fi

{
    echo "ts=${TS} gemini=OK gpt4o=OK grok=OK"
} > "${SUMMARY_FILE}"
exit 0
