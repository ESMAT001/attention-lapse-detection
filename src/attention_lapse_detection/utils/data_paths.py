from pathlib import Path

from attention_lapse_detection.utils.constants import (
    LABELS_SPLITS,
    DEFAULT_WINDOW_SECONDS,
)

from attention_lapse_detection.utils.paths import PATHS


def labels_dir() -> Path:
    return PATHS.processed_data / "labels"


def labels_csv(split: str) -> Path:
    return labels_dir() / LABELS_SPLITS[split]


def labels_file(split: str) -> Path:
    return labels_dir() / LABELS_SPLITS[split]


def fps_tag(fps: int) -> str:
    return f"{fps}fps"


def window_tag(window_seconds: int = DEFAULT_WINDOW_SECONDS) -> str:
    return f"{window_seconds}s"


def chunks_dir(fps: int, window_seconds: int = DEFAULT_WINDOW_SECONDS) -> Path:
    return (
        PATHS.processed_data
        / "window_chunks"
        / window_tag(window_seconds)
        / fps_tag(fps)
    )


def chunks_split_dir(
    fps: int, split: str, window_seconds: int = DEFAULT_WINDOW_SECONDS
) -> Path:
    return chunks_dir(fps, window_seconds) / split


def chunk_glob_pattern(split: str):
    return f"{split}_windows_chunk_*.npz"


def merged_dir(fps: int, window_seconds: int = DEFAULT_WINDOW_SECONDS) -> Path:
    return PATHS.processed_data / "merged" / window_tag(window_seconds) / fps_tag(fps)


def merged_npz(
    fps: int, split: str, window_seconds: int = DEFAULT_WINDOW_SECONDS
) -> Path:
    return merged_dir(fps, window_seconds) / f"{split}_windows.npz"


def clean_dir(
    fps: int,
    features_id: str = "all_features",
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> Path:
    return (
        PATHS.processed_data
        / "clean"
        / window_tag(window_seconds)
        / fps_tag(fps)
        / features_id
    )


def failed_extraction_csv(
    fps: int, split: str, window_seconds: int = DEFAULT_WINDOW_SECONDS
) -> Path:
    return (
        PATHS.processed_data
        / "failed_extractions"
        / window_tag(window_seconds)
        / f"{fps_tag(fps)}_{split}.csv"
    )
