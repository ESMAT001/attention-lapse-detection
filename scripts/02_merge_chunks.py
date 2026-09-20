"""Merge each build's window chunks into one file per split"""

import argparse
from typing import Any

import numpy as np
import pandas as pd

from attention_lapse_detection.utils.cli import resolve_variants
from attention_lapse_detection.utils.constants import (
    FPS_OPTIONS,
    LABELS_SPLITS,
    WINDOW_METADATA_COLUMNS,
    DEFAULT_WINDOW_SECONDS,
    WINDOW_SECONDS_OPTIONS,
)

from attention_lapse_detection.utils.data_paths import (
    chunk_glob_pattern,
    chunks_split_dir,
    merged_dir,
    merged_npz,
)


def merge_split(
    fps: int,
    split: str,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> None:
    tag = f"{window_seconds}s/{fps}fps/{split}"
    chunk_dir = chunks_split_dir(fps, split, window_seconds)
    chunk_paths = sorted(chunk_dir.glob(chunk_glob_pattern(split)))

    if not chunk_paths:
        print(f"[{tag}] No chunks at {chunk_dir}, skipping.")
        return

    chunks = [np.load(path) for path in chunk_paths]

    X = np.concatenate([chunk["X"] for chunk in chunks])
    y = np.concatenate([chunk["y"] for chunk in chunks])
    metadata = pd.concat(
        [
            pd.DataFrame({key: chunk[key] for key in WINDOW_METADATA_COLUMNS})
            for chunk in chunks
        ],
        ignore_index=True,
    )

    merged_dir(fps, window_seconds).mkdir(parents=True, exist_ok=True)
    out_path = merged_npz(fps, split, window_seconds)
    arrays = {
        "X": X,
        "y": y,
        **{key: metadata[key].to_numpy() for key in WINDOW_METADATA_COLUMNS},
    }
    np.savez_compressed(out_path, **arrays)

    print(
        f"[{tag}] {len(chunk_paths)} chunks -> X {X.shape}, {len(metadata)} rows -> {out_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=None,
        help=(
            f"Window length to merge, in seconds. Omit to merge every variant in {list(WINDOW_SECONDS_OPTIONS)}"
        ),
    )

    args = parser.parse_args()

    window_variants = resolve_variants(args.window_seconds, WINDOW_SECONDS_OPTIONS)

    if any(window < 1 for window in window_variants):
        parser.error("--window-seconds must be >= 1")

    for window_seconds in window_variants:
        for fps in FPS_OPTIONS:
            for split in LABELS_SPLITS:
                merge_split(fps, split, window_seconds)
