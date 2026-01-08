#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "$0")/.."; pwd)
cd "$ROOT_DIR"

echo "Cleaning outputs and temporary files under: $ROOT_DIR"

rm -f report_execution_*.csv report_execution_*.html report_execution_*.json || true
rm -f queries_*.sql dataset_*.csv incremental_dml.sql || true

if [ -d output ]; then
  find output -mindepth 1 -maxdepth 1 -exec rm -rf {} +
fi

if [ -d tmp ]; then
  find tmp -mindepth 1 -maxdepth 1 -exec rm -rf {} +
fi

if [ -d runs ]; then
  find runs -mindepth 1 -maxdepth 1 -exec rm -rf {} +
fi

echo "Done."

