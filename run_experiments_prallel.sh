#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs experiments/results

tmux new-session -d -s experiments
for fps in 10 15 30; do
  for model in gru lstm gru_uni_attention lstm_uni_attention; do
    name=${model}_${fps}fps
    tmux new-window -t experiments -n "$name" \
      "OMP_NUM_THREADS=2 uv run python scripts/experiments_script/05_run_experiments.py --device cuda \
       --fps $fps --model $model --save \
       --out experiments/results/grid_$name.csv 2>&1 | tee logs/expreiments_$name.log; exec bash"
  done
done

echo "12 experiment jobs started. Open: tmux attach -t experiments"
echo 'After all jobs finish: (head -1 experiments/results/grid_gru_10fps.csv; tail -q -n +2 experiments/results/grid_*.csv) > experiments/results/thesis_runs.csv'
