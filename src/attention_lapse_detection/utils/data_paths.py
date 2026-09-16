from pathlib import Path

from attention_lapse_detection.utils.constants import (
    LABELS_SPLITS,
    DEFAULT_WINDOW_SECONDS,
)

from attention_lapse_detection.utils.paths import PATHS


def labels_dir() -> Path:
    return PATHS.processed_data / "labels"


def labels_file(split: str) -> Path:
    return labels_dir() / LABELS_SPLITS[split]


def fps_tag(fps: int) -> str:
    return f"{fps}fps"


def window_tag(window_seconds: int = DEFAULT_WINDOW_SECONDS) -> str:
    return f"{window_seconds}s"

