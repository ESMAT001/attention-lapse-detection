#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

checkpoints=()
for cell in gru_10fps_10s lstm_10fps_10s gru_uni_attention_10fps_10s lstm_uni_attention_10fps_10s lstm_uni_attention_30fps_10s gru_30fps_10s lstm_30fps_10s; do
  for seed in 42 43 44 52 53 54 55 56; do
    checkpoints+=("${cell}_all_features_s$seed")
  done
done

uv run python scripts/08_evaluate_test.py "${checkpoints[@]}" 2>&1 | tee logs/test.log
