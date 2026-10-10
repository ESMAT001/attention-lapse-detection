import json

import torch
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
from attention_lapse_detection.training.metrics import best_f1_threshold
from attention_lapse_detection.training.report import report_ap
from attention_lapse_detection.training.trainer import Trainer
from attention_lapse_detection.types import Checkpoint, SavedScaler
from attention_lapse_detection.utils.constants import (
    DEFAULT_FPS,
    DEFAULT_WINDOW_SECONDS,
    EPOCHS,
    NUM_CLASSES,
    PATIENCE,
    SEED,
    WEIGHT_DECAY,
)
from attention_lapse_detection.utils.data_paths import clean_dir, features_id_from_drops
from attention_lapse_detection.utils.numeric import round4
from attention_lapse_detection.utils.paths import PATHS
from attention_lapse_detection.utils.seed import set_seed


def train_and_report(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    device: str = "cpu",
) -> Trainer:
    """Train with early stopping on validation AP and report results."""

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device=device)
    trainer.fit(epochs=epochs, early_stopping=True, patience=patience)
    report_ap(trainer)

    return trainer


def save_checkpoint(
    name: str,
    model: nn.Module,
    model_kwargs: dict,
    trainer: Trainer,
    fps: int,
    drop_columns: list[str],
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> None:
    """Save one .pt bundle: weights, config, scaler and decision threshold."""
    scaler_path = (
        clean_dir(fps, features_id_from_drops(drop_columns), window_seconds)
        / "scaler.json"
    )

    if not scaler_path.is_file():
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}. Run 03_clean_extracted_data.py for this data build."
        )

    with open(scaler_path) as f:
        scaler: SavedScaler = json.load(f)

    threshold, prec, rec = best_f1_threshold(trainer)

    bundle: Checkpoint = {
        "model_class": type(model).__name__,
        "model_kwargs": model_kwargs,
        # Save weights on CPU so the checkpoint can load on any device.
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "threshold": threshold,
        "fps": fps,
        "drop_columns": drop_columns,
        "window_seconds": window_seconds,
        "val_ap": trainer.best_val_ap,
    }

    PATHS.models.mkdir(parents=True, exist_ok=True)
    out_path = PATHS.models / f"{name}.pt"
    torch.save(bundle, out_path)

    print(
        f"Saved checkpoint to {out_path}\n"
        f"  model: {bundle['model_class']}, threshold: {round4(threshold)} "
        f"(validation precision {round4(prec)}, recall {round4(rec)}) "
        f"validation AP: {round4(trainer.best_val_ap)}"
    )


def train_model(
    model_name: str,
    fps: int = DEFAULT_FPS,
    drop_columns: list[str] | None = None,
    seed: int = SEED,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    device: str = "cpu",
) -> Trainer:
    """Train one model on cleaned feature windows."""
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
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)

    trainer = train_and_report(model, train_loader, val_loader, criterion, optimizer,device=device)
    return trainer
