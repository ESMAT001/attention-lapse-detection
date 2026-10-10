"""print the selected model, fps and window length."""

import argparse
from pathlib import Path

import pandas as pd

from attention_lapse_detection.training.selection import select, select_per_model
from attention_lapse_detection.utils.paths import PATHS

RUNS = PATHS.experiments / "results" / "thesis_runs.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS)

    args = parser.parse_args()

    runs = pd.read_csv(args.runs)

    if runs[runs.features == "all_features"].empty:
        raise SystemExit(f"No full-feature rows in {args.runs}.")

    winner = select(runs)
    print(f"winner: {winner[0]} {winner[1]} {winner[2]}")


    print("Test configuration by model:")
    for m, (_, f, w) in select_per_model(runs).items():
        print(f"{m} {f}fps  {w}s")
     
