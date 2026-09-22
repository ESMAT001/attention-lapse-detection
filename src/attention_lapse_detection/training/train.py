import argparse

import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from attention_lapse_detection.core_models.registry import CLASSIFIERS
from attention_lapse_detection.training.data import (
    balanced_class_weights,
    load_clean_splits,
    make_loaders,
)
from attention_lapse_detection.training.report import report_ap
from attention_lapse_detection.training.trainer import Trainer
from attention_lapse_detection.utils.constants import (
    DEFAULT_FPS,
    DEFAULT_WINDOW_SECONDS,
    EPOCHS,
    FEATURE_COLUMNS,
    FPS_OPTIONS,
    NUM_CLASSES,
    PATIENCE,
    SEED,
    WEIGHT_DECAY,
)
from attention_lapse_detection.utils.seed import set_seed


def train_and_report(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
) -> Trainer:
    """Fit with early stopping on val AP, then report."""

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer)
    trainer.fit(epochs=epochs, early_stopping=True, patience=patience)
    report_ap(trainer)


    return trainer


def train_model(
    model_name: str,
    fps: int = DEFAULT_FPS,
    drop_columns: list[str] | None = None,
    seed: int = SEED,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> Trainer:
    """Train one architecture on one data build at its searched hyperparameters.

    The single training path behind script 04 and the experiment runner 07.
    """
    drop_columns = drop_columns or []

    generator = set_seed(seed)
    X_train, y_train, X_val, y_val = load_clean_splits(
        fps, drop_columns, seed, window_seconds
    )
    train_loader, val_loader = make_loaders(
        X_train, y_train, X_val, y_val, generator, batch_size=64
    )

    model_kwargs = dict(
        input_size=X_train.shape[2],
        hidden_size=64,
        num_layers=1,
        num_classes=NUM_CLASSES,
        dropout=0.25,
    )
    model = CLASSIFIERS[model_name](**model_kwargs)

    criterion = nn.CrossEntropyLoss(weight=balanced_class_weights(y_train))
    optimizer = optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY
    )

    trainer = train_and_report(model, train_loader, val_loader, criterion, optimizer)
    return trainer


def create_arg_parser(description: str | None = None):
    """The training argument parser."""

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model",
        dest="model_name",
        choices=list(CLASSIFIERS),
        required=True,
        help="Architecture to train.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        choices=list(FPS_OPTIONS),
        default=DEFAULT_FPS,
        help=f"Target fps (default: {DEFAULT_FPS}).",
    )

    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=[],
        choices=FEATURE_COLUMNS,
        metavar="COL",
        help=(
            "Feature columns that were dropped at clean time. Choices: "
            + ", ".join(FEATURE_COLUMNS)
            + ". Default: none."
        ),
    )

    parser.add_argument(
        "--window-seconds",
        type=int,
        default=DEFAULT_WINDOW_SECONDS,
        help=(
            f"Window length in seconds (default {DEFAULT_WINDOW_SECONDS}), selects which "
            "cleaned build under clean/{n}s/ to train on. Stage 03 must already have produced it."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for Python, NumPy, PyTorch and DataLoader (default: {SEED}).",
    )
    return parser
