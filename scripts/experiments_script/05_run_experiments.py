"""Compare architectures, window lengths and feature sets across seeds.

Use tuned settings and save one CSV row per run. With --ablate, drop each
named feature separately and reuse the settings tuned on all features.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from functools import lru_cache
from attention_lapse_detection.core_models.registry import CLASSIFIERS
from attention_lapse_detection.training.metrics import (
    ap_disengaged,
    best_f1_threshold,
    clip_ap,
)
from attention_lapse_detection.training.train import save_checkpoint
from attention_lapse_detection.training.trainer import Trainer
from attention_lapse_detection.utils.constants import (
    DEFAULT_FPS,
    EPOCHS,
    FPS_OPTIONS,
    NUM_CLASSES,
    PATIENCE,
    WEIGHT_DECAY,
    WINDOW_SECONDS_OPTIONS,
    FEATURE_COLUMNS,
)
from attention_lapse_detection.utils.csv_log import append_row, read_rows
from attention_lapse_detection.utils.data_paths import features_id_from_drops
from attention_lapse_detection.utils.hyperparameters import hyperparameters
from attention_lapse_detection.utils.paths import PATHS
from attention_lapse_detection.training.data import (
    load_clean_split,
    balanced_class_weights,
    load_clip_ids,
    make_loaders,
)
from attention_lapse_detection.utils.seed import set_seed

DEFAULT_SEEDS = [42, 43, 44, 52, 53, 54, 55, 56]

RESULTS_CSV = PATHS.experiments / "results" / "thesis_runs.csv"

FIELDS = [
    "model",
    "fps",
    "window_seconds",
    "features",
    "seed",
    "hidden_size",
    "num_layers",
    "dropout",
    "learning_rate",
    "batch_size",
    "params",
    "epochs_run",
    "best_epoch",
    "val_ap",
    "val_clip_ap",
    "threshold",
    "val_precision",
    "val_recall",
    "val_weighted_f1",
    "val_accuracy",
    "device",
    "checkpoint",
]


@lru_cache(maxsize=1)
def load_data(fps: int, window_seconds: int, drop_columns: tuple[str, ...]):
    X_train, y_train = load_clean_split(
        fps, list(drop_columns), "Train", window_seconds
    )
    X_val, y_val = load_clean_split(
        fps, list(drop_columns), "Validation", window_seconds
    )
    return X_train, y_train, X_val, y_val


@lru_cache(maxsize=1)
def class_weights(fps: int, window_seconds: int, drop_columns: tuple[str, ...]):
    return balanced_class_weights(load_data(fps, window_seconds, drop_columns)[1])


def run_once(
    model_name: str,
    fps: int,
    window_seconds: int,
    drop_columns: list[str],
    seed: int,
    device: torch.device,
    save: bool,
) -> dict:

    X_train, y_train, X_val, y_val = load_data(fps, window_seconds, tuple(drop_columns))
    hp = hyperparameters(model_name, fps, window_seconds)

    generator = set_seed(seed)

    train_loader, val_loader = make_loaders(
        X_train, y_train, X_val, y_val, generator, batch_size=hp.batch_size
    )

    model_kwargs = dict(
        input_size=X_train.shape[2],
        hidden_size=hp.hidden_size,
        num_layers=hp.num_layers,
        num_classes=NUM_CLASSES,
        dropout=hp.dropout,
    )

    model = CLASSIFIERS[model_name](**model_kwargs)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(fps, window_seconds, tuple(drop_columns))
    )

    optimizer = optim.Adam(
        model.parameters(), lr=hp.learning_rate, weight_decay=WEIGHT_DECAY
    )

    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, device=device
    )

    trainer.fit(epochs=EPOCHS, early_stopping=True, patience=PATIENCE)

    clip_ids = load_clip_ids(fps, drop_columns, "Validation", window_seconds)
    threshold, precision, recall = best_f1_threshold(trainer)
    y_pred = np.where(trainer.val_probs[:, 0] >= threshold, 0, 1)

    features_id = features_id_from_drops(drop_columns)
    checkpoint = ""

    if save:
        checkpoint = f"{model_name}_{fps}fps_{window_seconds}s_{features_id}_s{seed}"
        save_checkpoint(
            checkpoint,
            model,
            model_kwargs,
            trainer,
            fps,
            drop_columns,
            window_seconds,
        )

    return {
        "model": model_name,
        "fps": fps,
        "window_seconds": window_seconds,
        "features": features_id,
        "seed": seed,
        "hidden_size": hp.hidden_size,
        "num_layers": hp.num_layers,
        "dropout": hp.dropout,
        "learning_rate": hp.learning_rate,
        "batch_size": hp.batch_size,
        "params": sum(p.numel() for p in model.parameters()),
        "epochs_run": trainer.epochs_run,
        "best_epoch": trainer.best_epoch,
        "val_ap": ap_disengaged(trainer.val_y_true, trainer.val_probs),
        "val_clip_ap": clip_ap(trainer.val_y_true, trainer.val_probs, clip_ids),
        "threshold": threshold,
        "val_precision": precision,
        "val_recall": recall,
        "val_weighted_f1": f1_score(
            trainer.val_y_true, y_pred, average="weighted", zero_division=0
        ),
        "val_accuracy": accuracy_score(trainer.val_y_true, y_pred),
        "device": str(device),
        "checkpoint": checkpoint,
    }


def completed_runs(path: Path) -> set[tuple]:
    """Find completed runs so they can be skipped."""

    return {
        (
            row["model"],
            int(row["fps"]),
            int(row["window_seconds"]),
            row["features"],
            int(row["seed"]),
        )
        for row in read_rows(path)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", nargs="*", choices=list(CLASSIFIERS), default=list(CLASSIFIERS)
    )
    parser.add_argument(
        "--fps", type=int, choices=list(FPS_OPTIONS), default=DEFAULT_FPS
    )
    parser.add_argument(
        "--window-seconds",
        nargs="*",
        type=int,
        choices=list(WINDOW_SECONDS_OPTIONS),
        default=list(WINDOW_SECONDS_OPTIONS),
    )
    parser.add_argument(
        "--ablate",
        nargs="*",
        choices=FEATURE_COLUMNS,
        default=None,
        metavar="COL",
        help=("Drop each named feature in a separate run."),
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--out", type=Path, default=RESULTS_CSV)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save a checkpoint per run to models/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned runs and exit.",
    )
    args = parser.parse_args()

    variants = [[]] if args.ablate is None else [[col] for col in args.ablate]

    done = completed_runs(args.out)
    runs = [
        (model_name, window_seconds, drop_columns, seed)
        for window_seconds in args.window_seconds
        for drop_columns in variants
        for model_name in args.model
        for seed in args.seeds
        if (
            model_name,
            args.fps,
            window_seconds,
            features_id_from_drops(drop_columns),
            seed,
        )
        not in done
    ]

    print(f"Runs to start: {len(runs)} ({len(done)} already recorded)")

    if args.dry_run:
        for model_name, window_seconds, drop_columns, seed in runs:
            features_id = features_id_from_drops(drop_columns)
            print(
                f"Model: {model_name}, window: {window_seconds}s, dropped features: {drop_columns}, seed: {seed}"
            )
            sys.exit(0)

    device = torch.device(args.device)

    print(f"Using device: {device}")

    for model_name, window_seconds, drop_columns, seed in runs:
        row = run_once(
            model_name,
            args.fps,
            window_seconds,
            drop_columns,
            seed,
            device,
            args.save,
        )
        append_row(args.out, FIELDS, row)
        print(
            f"Finished {model_name}: window {window_seconds}s, dropped features {drop_columns}, seed {seed}"
        )
