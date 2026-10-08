#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs tuning/results

tmux new-session -d -s search
for fps in 10 15 30; do
  for w in 5 10; do
    name=${fps}_${w}s
    tmux new-window -t search -n "$name" \
      "OMP_NUM_THREADS=2 uv run python tuning/search.py --device cuda \
       --fps $fps --window-seconds $w \
       --out tuning/results/shard_$name.csv 2>&1 | tee logs/search_$name.log; exec bash"
  done
done

echo "6 search jobs started. Open: tmux attach -t search"
