#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${BASE_DIR}/.." && pwd)"
OUTPUT_DIR="${BASE_DIR}/generation_outputs"
LOG_DIR="${BASE_DIR}/run_logs"

MODEL="${MODEL:-gemma3:27b}"
QUERY_IDS="${QUERY_IDS:-}"
TOP_K_VALUES="${TOP_K_VALUES:-1,2,3,4,5,6,7,8}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-SDKG_50q_topk1to8_$(date +%Y%m%d_%H%M%S)}"
PID_FILE="${LOG_DIR}/${OUTPUT_PREFIX}.pid"
UNIT_FILE="${LOG_DIR}/${OUTPUT_PREFIX}.unit"
LOG_FILE="${LOG_DIR}/${OUTPUT_PREFIX}.log"
UNIT_NAME="${UNIT_NAME:-sdkg-${OUTPUT_PREFIX//_/-}}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}")"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "A generation run is already active: pid=${old_pid}"
    echo "Log: ${LOG_FILE}"
    exit 1
  fi
fi

if systemctl --user is-active --quiet "${UNIT_NAME}.service" 2>/dev/null; then
  echo "A systemd user service is already active: ${UNIT_NAME}.service"
  exit 1
fi

if ! curl -fsS http://localhost:11434/api/tags >/dev/null; then
  echo "Ollama is not reachable at http://localhost:11434."
  echo "Start Ollama first, then rerun this script."
  exit 1
fi

query_arg=()
if [[ -n "${QUERY_IDS}" ]]; then
  query_arg=(--query-ids "${QUERY_IDS}")
fi

run_command=$(
  printf 'cd %q && set -euo pipefail; python SDKG_official/SDKG_run_50q_18exp_topk1to8.py --model %q --output-prefix %q --top-k-values %q' \
    "${PROJECT_DIR}" "${MODEL}" "${OUTPUT_PREFIX}" "${TOP_K_VALUES}"
)
if [[ -n "${QUERY_IDS}" ]]; then
  run_command+=$(printf ' --query-ids %q' "${QUERY_IDS}")
fi

systemd-run --user --unit="${UNIT_NAME}" --collect bash -lc "${run_command} > ${LOG_FILE@Q} 2>&1" >/dev/null
pid="$(systemctl --user show "${UNIT_NAME}.service" -p MainPID --value 2>/dev/null || true)"
echo "${pid:-}" >"${PID_FILE}"
echo "${UNIT_NAME}.service" >"${UNIT_FILE}"

echo "Started SDKG 50-query top-k generation."
echo "PID: ${pid:-unknown}"
echo "systemd unit: ${UNIT_NAME}.service"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "Query IDs: ${QUERY_IDS:-all 50}"
echo "Top-k values: ${TOP_K_VALUES}"
echo "Model: ${MODEL}"
echo "Outputs: ${OUTPUT_DIR}/${OUTPUT_PREFIX}_{FI-L,...,CI-H}.jsonl"
echo "Log: ${LOG_FILE}"
echo
echo "Check status:"
echo "  bash ${BASE_DIR}/status_SDKG_50q_18exp_topk1to8.sh ${OUTPUT_PREFIX}"
echo
echo "Resume if interrupted:"
echo "  OUTPUT_PREFIX=${OUTPUT_PREFIX} bash ${BASE_DIR}/run_SDKG_50q_18exp_topk1to8.sh"
echo
echo "Stop if needed:"
echo "  systemctl --user stop ${UNIT_NAME}.service"
