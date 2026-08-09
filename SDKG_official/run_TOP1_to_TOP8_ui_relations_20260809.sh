#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${BASE_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

for k in 1 2 3 4 5 6 7 8; do
  python SDKG_official/SDKG_query_generate.py \
    --all-queries \
    --exp-id E14 \
    --top-k "${k}" \
    --model gemma3:27b \
    --save \
    --output-stem "SDKG_50q_FCH_TOP${k}_ui_relations_20260809"
done
