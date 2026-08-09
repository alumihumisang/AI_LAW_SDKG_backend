#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${BASE_DIR}/generation_outputs"
LOG_DIR="${BASE_DIR}/run_logs"
OUTPUT_PREFIX="${1:-}"

if [[ -z "${OUTPUT_PREFIX}" ]]; then
  latest_pid="$(ls -t "${LOG_DIR}"/SDKG_50q_topk1to8_*.pid 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest_pid}" ]]; then
    echo "Usage: bash ${BASE_DIR}/status_SDKG_50q_18exp_topk1to8.sh OUTPUT_PREFIX"
    echo "No matching pid files found in ${LOG_DIR}."
    exit 1
  fi
  OUTPUT_PREFIX="$(basename "${latest_pid}" .pid)"
fi

PID_FILE="${LOG_DIR}/${OUTPUT_PREFIX}.pid"
UNIT_FILE="${LOG_DIR}/${OUTPUT_PREFIX}.unit"
LOG_FILE="${LOG_DIR}/${OUTPUT_PREFIX}.log"

echo "Output prefix: ${OUTPUT_PREFIX}"

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "Process: running (pid=${pid})"
  else
    echo "Process: not running (last pid=${pid})"
  fi
else
  echo "Process: no pid file"
fi

if [[ -f "${UNIT_FILE}" ]]; then
  unit_name="$(cat "${UNIT_FILE}")"
  unit_state="$(systemctl --user is-active "${unit_name}" 2>/dev/null || true)"
  echo "systemd: ${unit_state:-unknown} (${unit_name})"
else
  echo "systemd: no unit file"
fi

python - "${OUTPUT_DIR}" "${OUTPUT_PREFIX}" <<'PY'
import csv
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
prefix = sys.argv[2]
exp_names = [
    "FI-L", "FC-L", "IF-L", "CF-L", "IC-L", "CI-L",
    "FI-M", "FC-M", "IF-M", "CF-M", "IC-M", "CI-M",
    "FI-H", "FC-H", "IF-H", "CF-H", "IC-H", "CI-H",
]

total_rows = 0
total_ok = 0
total_err = 0
for exp_name in exp_names:
    path = output_dir / f"{prefix}_{exp_name}.csv"
    if not path.exists():
        print(f"{exp_name}: file not created")
        continue
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ok = sum(1 for r in rows if r.get("run_status") == "ok")
    err = sum(1 for r in rows if r.get("run_status") == "error")
    total_rows += len(rows)
    total_ok += ok
    total_err += err
    last = rows[-1] if rows else {}
    print(
        f"{exp_name}: rows={len(rows)}/400 ok={ok} error={err} "
        f"last=query_id={last.get('query_id','')} top_k={last.get('top_k','')} status={last.get('run_status','')}"
    )
print(f"total rows: {total_rows}/7200 ok={total_ok} error={total_err}")
PY

if [[ -f "${LOG_FILE}" ]]; then
  echo
  echo "Recent log:"
  tail -n 16 "${LOG_FILE}"
else
  echo "Log: file not created yet"
fi
