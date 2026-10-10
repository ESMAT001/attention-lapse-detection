"""Evaluate checkpoints on Test. Skip recorded results."""

import argparse
from pathlib import Path

import numpy as np
from attention_lapse_detection.types import WindowArray
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from attention_lapse_detection.training.data import load_clip_ids
from attention_lapse_detection.types import Checkpoint
from attention_lapse_detection.training.metrics import (
    clip_bootstrap_ci,
    clip_ap,
    participant_ids,
    per_clip,
    ap_disengaged,
)
from attention_lapse_detection.core_models.registry import CLASSIFIER_NAMES
from attention_lapse_detection.utils.checkpoints import resolve_checkpoint
from attention_lapse_detection.utils.csv_log import append_row, read_rows
from attention_lapse_detection.utils.data_paths import (
    features_id_from_drops,
    load_clean_split,
)
from attention_lapse_detection.utils.numeric import round4
from attention_lapse_detection.utils.paths import PATHS

RESULTS_CSV = PATHS.experiments / "results" / "test_results.csv"

BATCH_SIZE = 512

FIELDS = [
    "checkpoint",
    "model_class",
    "fps",
    "window_seconds",
    "features",
    "threshold",
    "n_test_windows",
    "n_test_clips",
    "n_pos_clips",
    "test_ap",
    "test_clip_ap",
    "test_clip_ci_lo",
    "test_clip_ci_hi",
    "test_weighted_f1",
    "test_accuracy",
    "test_precision_disengaged",
    "test_recall_disengaged",
    "tp",
    "fn",
    "fp",
    "tn",
]


def predict(model: torch.nn.Module, X: WindowArray):
    model.eval()
    probs = []

    with torch.no_grad():

        for start in range(0, len(X), BATCH_SIZE):

            batch = torch.tensor(X[start : start + BATCH_SIZE], dtype=torch.float32)
            probs.append(torch.softmax(model(batch), dim=1).numpy())

    return np.concatenate(probs)


def evaluate(checkpoint_path: Path, resamples: int, out: Path) -> dict:

    checkpoint: Checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    model = CLASSIFIER_NAMES[checkpoint["model_class"]](**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])

    fps = checkpoint["fps"]
    drop_columns = checkpoint["drop_columns"]
    window_seconds = checkpoint["window_seconds"]
    threshold = checkpoint["threshold"]

    X_test, y_test = load_clean_split(fps, drop_columns, "Test", window_seconds)
    clip_ids = load_clip_ids(fps, drop_columns, "Test", window_seconds)

    probs = predict(model, X_test)

    y_pred = np.where(probs[:, 0] >= threshold, 0, 1)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    ci_lo, ci_hi = clip_bootstrap_ci(y_test, probs, clip_ids, resamples=resamples)
    clips = per_clip(y_test, probs, clip_ids)

    probs_dir = out.parent / "test_probs"
    probs_dir.mkdir(parents=True, exist_ok=True)

    clips.assign(participant=participant_ids(clips.index.to_numpy())).to_csv(
        probs_dir / f"{checkpoint_path.stem}.csv", index_label="clip"
    )

    return {
        "checkpoint": checkpoint_path.stem,
        "model_class": checkpoint["model_class"],
        "fps": fps,
        "window_seconds": window_seconds,
        "features": features_id_from_drops(drop_columns),
        "threshold": threshold,
        "n_test_windows": len(y_test),
        "n_test_clips": len(clips),
        "n_pos_clips": int(clips.y.sum()),
        "test_ap": ap_disengaged(y_test, probs),
        "test_clip_ap": clip_ap(y_test, probs, clip_ids),
        "test_clip_ci_lo": ci_lo,
        "test_clip_ci_hi": ci_hi,
        "test_weighted_f1": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision_disengaged": tp / (tp + fp) if tp + fp else 0.0,
        "test_recall_disengaged": tp / (tp + fn) if tp + fn else 0.0,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints",
        nargs="+",
        help="Checkpoint names or .pt paths.",
    )
    parser.add_argument("--out", type=Path, default=RESULTS_CSV)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()

    scored = {row["checkpoint"] for row in read_rows(args.out)}

    for name in args.checkpoints:
        path = resolve_checkpoint(name)
        if path.stem in scored:
            print(f"Skipping {path.stem}: results already saved in {args.out}")
            continue

        row = evaluate(path, args.resamples, args.out)
        append_row(args.out, FIELDS, row)

        print(
            f"{row['checkpoint']}: clip AP {round4(row['test_clip_ap'])} "
            f"[{round4(row['test_clip_ci_lo'])}, {round4(row['test_clip_ci_hi'])}] | "
            f"weighted F1 {round4(row['test_weighted_f1'])} | "
            f"accuracy {round4(row['test_accuracy'])} | "
            f"disengaged precision {round4(row['test_precision_disengaged'])} "
            f"recall {round4(row['test_recall_disengaged'])} | "
            f"disengaged windows detected: {row['tp']}/{row['tp'] + row['fn']}"
        )
