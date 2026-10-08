"""Tune each model and data build with Optuna.

Search learning rate, dropout, batch size, depth and hidden size using mean
clip-level validation AP over three seeds. Rescore top candidates on separate
seeds and keep the one with the highest confirmed mean.
"""

import argparse
import itertools
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import optuna
import torch
from collections.abc import Callable

from torch import nn
from torch import optim

from attention_lapse_detection.core_models.registry import CLASSIFIERS
from attention_lapse_detection.training.metrics import clip_ap
from attention_lapse_detection.training.trainer import Trainer
from attention_lapse_detection.utils.constants import (
    EPOCHS,
    FPS_OPTIONS,
    PATIENCE,
    SEED,
    WEIGHT_DECAY,
    NUM_CLASSES,
    WINDOW_SECONDS_OPTIONS,
)
from attention_lapse_detection.utils.csv_log import append_row, read_rows
from attention_lapse_detection.utils.hyperparameters import Hyperparameters
from attention_lapse_detection.training.data import (
    balanced_class_weights,
    load_clean_split,
    load_clip_ids,
    make_loaders,
)

from pathlib import Path

from attention_lapse_detection.utils.numeric import round4
from attention_lapse_detection.utils.seed import set_seed

DROP_COLUMNS = []

SEARCH_SEEDS = [42, 43, 44]
CONFIRM_SEEDS = [52, 53, 54, 55, 56]

RESULTS_CSV = Path(__file__).parent / "results" / "search_results.csv"

# Search space
HIDDEN_SIZE = (20, 200)
NUM_LAYERS = (1, 3)
DROPOUTS = (0.0, 0.05, 0.10, 0.25, 0.50)
LEARNING_RATE = (1e-4, 1e-2)
BATCH_SIZES = (8, 16, 32, 64)

FIELDS = [
    "model",
    "fps",
    "window_seconds",
    "stage",
    "hidden_size",
    "num_layers",
    "dropout",
    "learning_rate",
    "batch_size",
    "ap",
    "sem",
    "params",
    "selected",
    "device",
]


@dataclass(frozen=True)
class Build:
    """One cleaned data variant."""

    fps: int
    window_seconds: int

    def __str__(self):
        return f"{self.fps}fps/{self.window_seconds}s"


@dataclass(frozen=True)
class Result:
    """A configuration scored on the confirmation seeds."""

    config: Hyperparameters
    mean: float
    sem: float  # Standard error of the mean
    params: int


def all_builds() -> list[Build]:
    return [
        Build(fps, window)
        for fps, window in itertools.product(FPS_OPTIONS, WINDOW_SECONDS_OPTIONS)
    ]


# Data


@lru_cache(maxsize=1)
def load_data(build: Build) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cache one build's fixed train/validation arrays for reuse across seeds."""
    X_train, y_train = load_clean_split(
        build.fps, DROP_COLUMNS, "Train", build.window_seconds
    )
    X_val, y_val = load_clean_split(
        build.fps, DROP_COLUMNS, "Validation", build.window_seconds
    )
    return X_train, y_train, X_val, y_val


@lru_cache(maxsize=1)
def class_weights(build: Build) -> torch.Tensor:
    return balanced_class_weights(load_data(build)[1])


@lru_cache(maxsize=None)
def validation_clip_ids(build: Build) -> np.ndarray:
    return load_clip_ids(build.fps, DROP_COLUMNS, "Validation", build.window_seconds)


def entry(model_name: str, build: Build, pick: Result) -> str:
    """Format the selected settings for hyperparameters.py."""
    config = pick.config

    return (
        f'    ("{model_name}", {build.fps}, {build.window_seconds}): '
        f"Hyperparameters(hidden_size={config.hidden_size}, "
        f"num_layers={config.num_layers}, dropout={config.dropout}, "
        f"learning_rate={config.learning_rate:.6g}, batch_size={config.batch_size}),"
        f"  # clip-level val AP {round4(pick.mean)}"
    )


def report(model_name: str, build: Build, results: list[Result], pick: Result):
    print(f"\n {model_name} | {build} (confirmed on seeds {CONFIRM_SEEDS})")

    for result in sorted(results, key=lambda result: result.mean, reverse=True):
        print(
            f"  {round4(result.mean)} +/- {round4(result.sem)}  "
            f"{result.params,} params  {result.config}"
        )
    print(
        f"  Selected: {pick.config}  (AP {round4(pick.mean)}, {pick.params:,} parameters)"
    )


def select(results: list[Result]) -> Result:
    """Choose the candidate with the highest confirmed mean AP."""
    return max(results, key=lambda result: result.mean)


def confirm(
    model_name: str, build: Build, configs: list[Hyperparameters], device: torch.device
) -> list[Result]:
    """Rescore top candidates on separate confirmation seeds."""

    results = []
    for config in configs:
        mean, sem = score(model_name, build, config, CONFIRM_SEEDS, device)
        params = sum(
            p.numel() for p in build_model(model_name, build, config).parameters()
        )
        results.append(Result(config, mean, sem, params))
    return results


def build_model(model_name: str, build: Build, config: Hyperparameters) -> nn.Module:
    return CLASSIFIERS[model_name](
        input_size=load_data(build)[0].shape[2],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_classes=NUM_CLASSES,
        dropout=config.dropout,
    )


def train_once(
    model_name: str,
    build: Build,
    config: Hyperparameters,
    seed: int,
    device: torch.device,
) -> float:
    """Train once and return clip-level validation AP."""

    X_train, y_train, X_val, y_val = load_data(build)
    generator = set_seed(seed)

    train_loader, val_loader = make_loaders(
        X_train, y_train, X_val, y_val, generator, batch_size=config.batch_size
    )

    model = build_model(model_name, build, config)
    criterion = nn.CrossEntropyLoss(weight=class_weights(build))

    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=WEIGHT_DECAY
    )

    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, device=device
    )

    trainer.fit(epochs=EPOCHS, early_stopping=True, patience=PATIENCE)
    return clip_ap(trainer.val_y_true, trainer.val_probs, validation_clip_ids(build))


def score(
    model_name: str,
    build: Build,
    config: Hyperparameters,
    seeds: list[int],
    device: torch.device,
) -> tuple[float, float]:
    """Return mean clip-level validation AP and its standard error across seeds."""

    scores = [train_once(model_name, build, config, seed, device) for seed in seeds]
    return float(np.mean(scores)), float(np.std(scores) / np.sqrt(len(scores)))


def suggest(trial: optuna.Trial) -> Hyperparameters:
    return Hyperparameters(
        hidden_size=trial.suggest_int("hidden_size", *HIDDEN_SIZE, log=True),
        num_layers=trial.suggest_int("num_layers", *NUM_LAYERS),
        dropout=trial.suggest_categorical("dropout", DROPOUTS),
        learning_rate=trial.suggest_float("learning_rate", *LEARNING_RATE, log=True),
        batch_size=trial.suggest_categorical("batch_size", BATCH_SIZES),
    )


def search(
    model_name: str,
    build: Build,
    trials: int,
    device: torch.device,
    log_row: Callable[..., None],
) -> list[Hyperparameters]:
    """Rank configurations with sequential TPE trials; confirm the top ones later."""

    def objective(trial: optuna.Trial) -> float:
        config = suggest(trial)
        mean, sem = score(model_name, build, config, SEARCH_SEEDS, device)

        log_row("search", config, mean, sem, None, False)
        return mean

    study = optuna.create_study(
        study_name=f"{model_name}|{build}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=trials)

    scored = [
        (t.value, Hyperparameters(**t.params))
        for t in study.trials
        if t.value is not None
    ]

    ranked = sorted(scored, key=lambda scored_trial: scored_trial[0], reverse=True)
    return list(dict.fromkeys(config for _, config in ranked))


def row_writer(path: Path, model_name: str, build: Build, device: torch.device):
    """Save each trial to CSV as it finishes."""

    def write(
        stage: str,
        config: Hyperparameters,
        ap: float,
        sem: float,
        params: int | None,
        selected: bool,
    ):
        append_row(
            path,
            FIELDS,
            {
                "model": model_name,
                "fps": build.fps,
                "window_seconds": build.window_seconds,
                "stage": stage,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "ap": ap,
                "sem": sem,
                "params": params,
                "selected": selected,
                "device": str(device),
            },
        )

    return write


def completed_studies(path: Path) -> set[tuple[str, int, int]]:
    """Find studies with a saved selection so they can be skipped."""

    return {
        (row["model"], int(row["fps"]), int(row["window_seconds"]))
        for row in read_rows(path)
        if row["selected"] == "True"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        nargs="*",
        choices=list(CLASSIFIERS),
        default=list(CLASSIFIERS),
    )

    parser.add_argument(
        "--fps",
        nargs="*",
        type=int,
        choices=list(FPS_OPTIONS),
        default=list(FPS_OPTIONS),
    )

    parser.add_argument(
        "--window-seconds",
        nargs="*",
        type=int,
        choices=list(WINDOW_SECONDS_OPTIONS),
        default=list(WINDOW_SECONDS_OPTIONS),
    )

    parser.add_argument("--trails", type=int, default=20)
    parser.add_argument(
        "--confirm-top",
        type=int,
        default=2,
        help="Number of top configurations to confirm.",
    )

    parser.add_argument("--out", type=Path, default=RESULTS_CSV)
    parser.add_argument(
        "--device",
        choices=(
            "cpu",
            "cuda",
            "mps",
        ),
        default="cpu",
        help="Training device (default: cpu).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned studies and training run counts, then exit.",
    )

    args = parser.parse_args()

    builds = [
        b
        for b in all_builds()
        if b.fps in args.fps and b.window_seconds in args.window_seconds
    ]

    done = completed_studies(args.out)

    studies = [
        (model_name, build)
        for build in builds
        for model_name in args.model
        if (model_name, build.fps, build.window_seconds) not in done
    ]

    runs_each = args.trails * len(SEARCH_SEEDS) + args.confirm_top * len(CONFIRM_SEEDS)

    print(
        f"{len(studies)} studies to run ({len(done)} already in {args.out}), "
        f"{runs_each} training runs each, {len(studies) * runs_each} total."
    )

    if args.dry_run:
        for name, build in studies:
            print(f"Model: {name}, data: {build}")
        exit()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    picks = []
    for model_name, build in studies:
        log_row = row_writer(args.out, model_name, build, device)
        ranked = search(model_name, build, args.trails, device, log_row)
        results = confirm(model_name, build, ranked[: args.confirm_top], device)
        pick = select(results)

        for result in results:
            log_row(
                "confirm",
                result.config,
                result.mean,
                result.sem,
                result.params,
                result is pick,
            )

        report(model_name, build, results, pick)
        picks.append((model_name, build, pick))

    print(
        "\nAdd these settings to src/attention_lapse_detection/utils/hyperparameters.py:"
    )

    for model_name, build, pick in picks:
        print(entry(model_name, build, pick))
