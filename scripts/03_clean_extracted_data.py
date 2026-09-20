"""Impute, clip and standardize merged windows into the clean arrays training reads."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from attention_lapse_detection.types import FeatureScaler
from attention_lapse_detection.utils.cli import resolve_variants
from attention_lapse_detection.utils.constants import (
    FEATURE_COLUMNS,
    FPS_OPTIONS,
    IMPUTE_ON_ABSENT,
    LABELS_SPLITS,
    WINDOW_METADATA_COLUMNS,
    DEFAULT_WINDOW_SECONDS,
    WINDOW_SECONDS_OPTIONS,
)
from attention_lapse_detection.utils.data_paths import (
    clean_dir,
    features_id_from_drops,
    merged_npz,
)

BINARY_COLUMNS = ["face_present", "blinks"]

QUANTILE_LOW = 0.001
QUANTILE_HIGH = 0.999
STD_CLIP = 5


def clean_split(
    split: str,
    fps: int,
    drop_columns: list[str],
    scaler: FeatureScaler | None,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> FeatureScaler | None:
    """Clean one split, returning the scaler (fit here when split is Train)."""

    npz_path = merged_npz(fps, split, window_seconds)

    if not npz_path.is_file():
        print(f"[{split}] {npz_path} not found, skipping.")
        return scaler

    if split != "Train" and scaler is None:
        print(f"[{split}] no Train scaler yet, skipping.")
        return scaler

    data = np.load(npz_path, allow_pickle=True)

    df, window_size = build_long_dataframe(data, drop_columns)

    windows_in = df["sample_id"].nunique()
    active_features = [c for c in FEATURE_COLUMNS if c not in drop_columns]

    df, scaler = clean_dataframe(df, active_features, scaler)
    X_clean, y_clean, metadata = to_window_arrays(df, window_size, active_features)

    out_dir = clean_dir(fps, features_id_from_drops(drop_columns), window_seconds)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"X_{split}_windows_clean.npy", X_clean)
    np.save(out_dir / f"y_{split}_windows_clean.npy", y_clean)
    metadata.to_csv(out_dir / f"{split}_window_metadata_clean.csv", index=False)

    if split == "Train":
        write_scaler(out_dir, scaler, active_features, window_size, fps)

    print(f"[{split}] {windows_in} windows kept. X {X_clean.shape}, y {y_clean.shape}")
    return scaler


def build_long_dataframe(
    data, drop_columns: list[str]
) -> tuple[pd.DataFrame, int]:
    """Flatten (windows, frames, features) into one row per frame."""
    X = data["X"]
    y = data["y"]
    num_windows, window_size, _ = X.shape

    df = pd.DataFrame(
        X.reshape(num_windows * window_size, len(FEATURE_COLUMNS)),
        columns=FEATURE_COLUMNS,
    )

    if drop_columns:
        df = df.drop(columns=drop_columns)

    for column in WINDOW_METADATA_COLUMNS:
        df[column] = np.repeat(data[column], window_size)

    df["frame_in_window"] = np.tile(np.arange(window_size), num_windows)
    df["engagement"] = np.repeat(y, window_size)

    return df, window_size


def clean_dataframe(
    df: pd.DataFrame, active_features: list[str], scaler: FeatureScaler | None
) -> tuple[pd.DataFrame, FeatureScaler]:
    """Impute, clip and standardize. scaler=None fits one (Train); else it is reused."""
    df = df.copy()

    df = impute_absent_frames(df, active_features)

    scaled_features = [c for c in active_features if c not in BINARY_COLUMNS]

    # ear/mar are ratios that can't go negative: a fixed floor, so it precedes the fit.
    for column in ("ear", "mar"):
        if column in scaled_features:
            df[column] = df[column].clip(lower=0)

    if scaler is None:
        scaler = fit_scaler(df, scaled_features)

    transform_features(df, scaled_features, scaler)

    return df, scaler


def impute_absent_frames(df: pd.DataFrame, active_features: list[str]) -> pd.DataFrame:
    """Forward fill zero filled features on missing-face frames."""

    if "face_present" not in df.columns:
        return df

    impute_columns = [c for c in IMPUTE_ON_ABSENT if c in active_features]
    if not impute_columns:
        return df

    df = df.sort_values(["sample_id", "frame_in_window"], kind="stable")
    df.loc[df["face_present"] == 0, impute_columns] = np.nan

    df[impute_columns] = df.groupby("sample_id", sort=False)[impute_columns].ffill()
    df[impute_columns] = df.groupby("sample_id", sort=False)[impute_columns].bfill()
    return df


def fit_scaler(df: pd.DataFrame, scaled_features: list[str]) -> FeatureScaler:
    """Fit clip bounds and standardization stats on Train only."""

    quantile_low = df[scaled_features].quantile(QUANTILE_LOW)
    quantile_high = df[scaled_features].quantile(QUANTILE_HIGH)
    clipped = df[scaled_features].clip(quantile_low, quantile_high, axis=1)

    return FeatureScaler(
        mean=clipped.mean(),
        std=clipped.std(ddof=0).replace(0, 1),
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )


def transform_features(
    df: pd.DataFrame, scaled_features: list[str], scaler: FeatureScaler
):
    """Quantile-clip, standardize and std-clip in place from fitted stats."""
    df[scaled_features] = df[scaled_features].clip(
        scaler.quantile_low, scaler.quantile_high, axis=1
    )
    df[scaled_features] = (df[scaled_features] - scaler.mean) / scaler.std
    df[scaled_features] = df[scaled_features].clip(-STD_CLIP, STD_CLIP)

    df[scaled_features] = df[scaled_features].fillna(0.0)


def to_window_arrays(df: pd.DataFrame, window_size: int, active_features: list[str]):
    df = df.sort_values(["sample_id", "frame_in_window"], kind="stable")

    counts = df.groupby("sample_id", sort=False).size()
    bad = counts[counts != window_size]
    assert bad.empty, f"windows with != {window_size} frames:{bad}"

    X_clean = (
        df[active_features]
        .to_numpy(dtype=np.float32)
        .reshape(counts.shape[0], window_size, len(active_features))
    )

    grouped = df.groupby("sample_id", sort=False)
    y_clean = grouped["engagement"].first().to_numpy(dtype=np.int64)
    metadata = grouped[WINDOW_METADATA_COLUMNS].first().reset_index(drop=True)

    assert X_clean.shape[0] == y_clean.shape[0] == len(metadata)
    return X_clean, y_clean, metadata


def write_scaler(
    out_dir: Path,
    scaler: FeatureScaler,
    active_features: list[str],
    window_size: int,
    fps: int,
):
    """Persisting the Train-split preprocessing so inference can reproduce it."""

    scaled_features = [c for c in active_features if c not in BINARY_COLUMNS]
    payload = {
        "fps": fps,
        "window_size": window_size,
        "feature_order": active_features,
        "scaled_features": scaled_features,
        "binary_features": [c for c in active_features if c in BINARY_COLUMNS],
        "clip_lower_zero": [c for c in ("ear", "mar") if c in active_features],
        "impute_on_absent": [c for c in IMPUTE_ON_ABSENT if c in active_features],
        "std_clip": STD_CLIP,
        "mean": {c: float(scaler.mean[c]) for c in scaled_features},
        "std": {c: float(scaler.std[c]) for c in scaled_features},
        "quantile_low": {c: float(scaler.quantile_low[c]) for c in scaled_features},
        "quantile_high": {c: float(scaler.quantile_high[c]) for c in scaled_features},
    }

    (out_dir / "scaler.json").write_text(json.dumps(payload, indent=2))
    print(f"[Train] wrote scaler.json ({len(scaled_features)} scaled features)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fps",
        type=int,
        choices=list(FPS_OPTIONS),
        default=None,
        help="Target fps. Omit to run every fps.",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=[],
        choices=FEATURE_COLUMNS,
        metavar="COL",
        help=(
            "Feature columns to drop before cleaning. Choices: "
            + ", ".join(FEATURE_COLUMNS)
            + ". Default: keep all."
        ),
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=None,
        help=(
            f"Window length to clean. Omit to clean every variant in {list(WINDOW_SECONDS_OPTIONS)}."
        ),
    )
    args = parser.parse_args()

    fps_variants = resolve_variants(args.fps, FPS_OPTIONS)
    window_variants = resolve_variants(args.window_seconds, WINDOW_SECONDS_OPTIONS)
    if any(window < 1 for window in window_variants):
        parser.error("--window-seconds must be >= 1")

    for window_seconds in window_variants:
        for fps in fps_variants:
            scaler: FeatureScaler | None = None
            for split in LABELS_SPLITS:
                scaler = clean_split(
                    split, fps, args.drop_columns, scaler, window_seconds
                )
