from typing import Iterable

from attention_lapse_detection.types import RawRows
from attention_lapse_detection.utils.constants import EAR_CLOSED_THRESHOLD

FATIGUE = "likely fatigue"
MIND_WANDERING = "likely mind wandering"
UNKNOWN = "unknown"

MIN_FACE_VALID_RATIO = 0.80
EAR_CLOSED = EAR_CLOSED_THRESHOLD
PERCLOS_MAX = 0.24  # Savas and Becerikli
PROLONGED_CLOSURE_S = 5.0  # Savas and Becerikli
MAR_YAWN = 0.60  # Rohitha et al.
YAWN_MIN_S = 0.50  # Rohitha et al.
BLINK_RATE_MAX = 25.0  # Exploratory cutoff; see STAGE2_CAUSE_CLASSIFICATION.md.


def run_lengths(flags: Iterable[bool]) -> list[int]:
    runs: list[int] = []
    current = 0

    for flag in flags:
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0

    if current:
        runs.append(current)

    return runs


def cause_evidence(rows: RawRows, fps: float) -> dict[str, float]:
    valid = [row["face_present"] == 1 for row in rows]

    valid_frames = sum(valid)
    if not valid_frames:
        return {
            "face_valid_ratio": 0.0,
            "perclos": 0.0,
            "longest_closure_s": 0.0,
            "longest_yawn_s": 0.0,
            "blink_rate": 0.0,
        }

    closed = [ok and row["ear"] <= EAR_CLOSED for row, ok in zip(rows, valid)]
    yawning = [ok and row["mar"] > MAR_YAWN for row, ok in zip(rows, valid)]

    closure_runs = run_lengths(closed)
    blinks = sum(row["blinks"] for row, ok in zip(rows, valid) if ok)

    return {
        "face_valid_ratio": valid_frames / len(rows),
        "perclos": sum(closed) / valid_frames,
        "longest_closure_s": max(closure_runs, default=0) / fps,
        "longest_yawn_s": max(run_lengths(yawning), default=0) / fps,
        "blink_rate": blinks * 60.0 / (valid_frames / fps),
    }


def classify_cause(rows: RawRows, fps: float):
    if not rows:
        return UNKNOWN

    e = cause_evidence(rows, fps)

    if e["face_valid_ratio"] < MIN_FACE_VALID_RATIO:
        return UNKNOWN

    if (
        e["perclos"] > PERCLOS_MAX
        or e["longest_closure_s"] > PROLONGED_CLOSURE_S
        or e["longest_yawn_s"] >= YAWN_MIN_S
    ):
        return FATIGUE

    if e["blink_rate"] > BLINK_RATE_MAX:
        return MIND_WANDERING

    return UNKNOWN
