"""Train a model on cleaned feature windows.

uv run python scripts/04_train.py --model lstm_uni_attention --window-seconds 10
"""
import argparse
from attention_lapse_detection.core_models.registry import CLASSIFIERS
from attention_lapse_detection.utils.constants import (
    FPS_OPTIONS,
    DEFAULT_FPS,
    FEATURE_COLUMNS,
    DEFAULT_WINDOW_SECONDS,
    SEED,
)
from attention_lapse_detection.training.train import train_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description=__doc__, 
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
        help=(
            "Features removed during cleaning. Choices: "
            + ", ".join(FEATURE_COLUMNS)
            + ". Default: none."
        ),
    )

    parser.add_argument(
        "--window-seconds",
        type=int,
        default=DEFAULT_WINDOW_SECONDS,
        help=(
            f"Window length in seconds (default: {DEFAULT_WINDOW_SECONDS}). "
            "Run script 03 for this window length first."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED}).",
    )

    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
        help="Device to use for training (default: cpu).",
    )

    args = parser.parse_args()
    train_model(**vars(args))
