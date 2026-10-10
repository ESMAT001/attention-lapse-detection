import argparse
import csv
import json
from pathlib import Path

from attention_lapse_detection.utils.csv_log import read_rows
from attention_lapse_detection.utils.hyperparameters import VALUES_JSON
from attention_lapse_detection.utils.paths import PATHS

RESULTS = PATHS.root / "tuning" / "results"
MERGED_CSV = RESULTS / "search_results.csv"
TARGET = VALUES_JSON


def merge_results(results: Path | None = None) -> None:
    merge_shards = results is None and not MERGED_CSV.is_file()

    if merge_shards:
        rows = [
            row
            for path in sorted(RESULTS.glob("shard_*.csv"))
            for row in read_rows(path)
        ]

    else:
        rows = read_rows(results if results is not None else MERGED_CSV)

    if not rows:
        raise ValueError("No search results found.")

    picked = {}
    for row in rows:
        if row["selected"] == "True":
            picked[int(row["window_seconds"]), int(row["fps"]), row["model"]] = row

    records = []
    for window_seconds, fps, model_name in sorted(picked):
        row = picked[window_seconds, fps, model_name]

        records.append(
            {
                "model": model_name,
                "fps": fps,
                "window_seconds": window_seconds,
                "hidden_size": int(row["hidden_size"]),
                "num_layers": int(row["num_layers"]),
                "dropout": float(row["dropout"]),
                "learning_rate": float(f"{float(row['learning_rate']):.6g}"),
                "batch_size": int(row["batch_size"]),
            }
        )
    if not records:
        raise ValueError("No selected settings found.")
    
    payload = json.dumps(records) 

    if merge_shards:
        with MERGED_CSV.open("w", newline="", encoding="utf-8") as final_file:
            writer = csv.DictWriter(final_file, fieldnames=list(rows[0]))

            writer.writeheader()
            writer.writerows(rows)

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge search results and export selected hyperparameters to JSON."
    )
    parser.add_argument("--results", type=Path, help="Export this CSV directly.")
    args = parser.parse_args()

    try:
        merge_results(args.results)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"{error}")
