import numpy as np
import torch
import pandas as pd
from numpy.typing import NDArray
from sklearn.utils import compute_class_weight
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from attention_lapse_detection.utils.constants import (
    DEFAULT_WINDOW_SECONDS,
    FEATURE_COLUMNS,
)
from attention_lapse_detection.utils.data_paths import (
    clean_dir,
    features_id_from_drops,
    load_clean_split,
)


class AttentionLapseDataset(Dataset):
    """Returns one feature window and its label."""

    def __init__(self, X: NDArray[np.float32], y: NDArray[np.int64]) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.X[idx], self.y[idx]


def load_clean_splits(
    fps: int,
    drop_columns: list[str],
    seed: int,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> tuple[
    NDArray[np.float32], NDArray[np.int64], NDArray[np.float32], NDArray[np.int64]
]:
    """Load the cleaned train/validation windows for a feature set."""

    src = clean_dir(fps, features_id_from_drops(drop_columns), window_seconds)
    active_features = [c for c in FEATURE_COLUMNS if c not in drop_columns]

    print(f"Loading from {src}")
    print(f"Features ({len(active_features)}): {active_features}")
    print(f"Window: {window_seconds}s | Seed: {seed}")

    X_train, y_train = load_clean_split(fps, drop_columns, "Train", window_seconds)
    X_val, y_val = load_clean_split(fps, drop_columns, "Validation", window_seconds)

    print(f"Train: X {X_train.shape}, y {y_train.shape}")
    print(f"Validation: X {X_val.shape}, y {y_val.shape}")

    return X_train, y_train, X_val, y_val


def make_loaders(
    X_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    X_val: NDArray[np.float32],
    y_val: NDArray[np.int64],
    generator: torch.Generator,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:

    train_loader = DataLoader(
        AttentionLapseDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        AttentionLapseDataset(X_val, y_val), batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader


def balanced_class_weights(y_train: NDArray[np.int64]) -> torch.Tensor:
    """Weight the rare class up so it is not ignored. Full balancing, about 11:1."""

    weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )

    tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"Class weights: {tensor.tolist()}")

    return tensor


def load_clip_ids(
    fps: int,
    drop_columns: list[str],
    split: str,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> NDArray:
    """The clip each window of a split came from, in .npy row order."""
    
    src = clean_dir(fps, features_id_from_drops(drop_columns), window_seconds)
    meta = pd.read_csv(src / f"{split}_window_metadata_clean.csv")
    return meta.video_id.to_numpy()