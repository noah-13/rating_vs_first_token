#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
LIMIT="${LIMIT:-}"
PYTHON_BIN="${PYTHON:-python3}"

mkdir -p results

run_dim() {
  local dim="$1"
  local args=(
    --dim "$dim"
    --model "$MODEL"
    --batch_size "$BATCH_SIZE"
    --max_new_tokens "$MAX_NEW_TOKENS"
  )
  if [[ -n "$LIMIT" ]]; then
    args+=(--limit "$LIMIT")
  fi

  "$PYTHON_BIN" llama3_as_judge_with_path.py "${args[@]}"
}

for dim in coherence consistency fluency relevance; do
  run_dim "$dim"
done

"$PYTHON_BIN" analyze_rating_vs_first.py \
  --csv_dir results \
  --csv_stem summeval_results_3tiers \
  --pred_mode expected \
  --out_csv results/metrics_table.csv

"$PYTHON_BIN" analyze_traj.py \
  --input_dir results \
  --out_prefix results/pure_traj_4dims

echo "All done. Results are in ./results"
