from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, precision_recall_curve

from attention_lapse_detection.types import LabelArray, WindowArray

if TYPE_CHECKING:
    # trainer.py imports this module, so only import Trainer for type checking.
    from attention_lapse_detection.training.trainer import Trainer


def ap_disengaged(y_true: LabelArray, probs: WindowArray):
    """Compute disengagement AP from class-0 probabilities.

    y_true has shape (N,); probs has shape (N, C).
    """
    is_disengaged = (y_true == 0).astype(np.int64)
    return float(average_precision_score(is_disengaged, probs[:, 0]))


def best_f1_threshold(trainer: "Trainer") -> tuple[float, float, float]:
    """Find the validation F1 cutoff for P(disengaged), with precision and recall."""
    
    y_pos = (trainer.val_y_true == 0).astype(int)
    probs_disengaged = trainer.val_probs[:, 0]

    prec, rec, thr = precision_recall_curve(y_pos, probs_disengaged)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)

    best = int(np.nanargmax(f1s))

    best_thr = thr[min(best, len(thr) - 1)]  # thr is shorter than prec/rec by 1
    return float(best_thr), float(prec[best]), float(rec[best])


def per_clip(
    y_true: LabelArray,
    probs: WindowArray,
    clip_ids: NDArray,
) -> pd.DataFrame:
    """Group windows by clip, keeping its label and mean P(disengaged)."""
    if len(clip_ids) != len(y_true):
        raise ValueError(
            f"Found {len(clip_ids)} clip IDs for {len(y_true)} windows. "
            "Check that the cleaned arrays match their metadata."
        )

    windows = pd.DataFrame(
        {
            "clip": clip_ids,
            "y": (y_true == 0).astype(int),  # disengaged is the positive class
            "p": probs[:, 0],
        }
    )

    labels = windows.groupby("clip")["y"].nunique()

    if (labels > 1).any():
        conflicting = labels[labels > 1].index.tolist()
        raise ValueError(
            f"Found conflicting labels in {len(conflicting)} clips, including "
            f"{conflicting[:3]}. Check that the labels match the window metadata."
        )

    return windows.groupby("clip").agg(y=("y", "first"), p=("p", "mean"))


def clip_ap(
    y_true: LabelArray,
    probs: WindowArray,
    clip_ids: NDArray,
) -> float:
    """Compute AP after averaging window predictions per clip."""

    clips = per_clip(y_true, probs, clip_ids)
    return float(average_precision_score(clips.y, clips.p))


def participant_ids(clip_ids: NDArray):
    return pd.Index(clip_ids).str[:6].to_numpy()


def clip_bootstrap_ci(
    y_true: LabelArray,
    probs: WindowArray,
    clip_ids: NDArray,
    resamples: int = 2000,
    seed: int = 0,
):
    clips = per_clip(y_true, probs, clip_ids)
    y, p = clips.y.to_numpy(), clips.p.to_numpy()
    if y.sum() == 0:
        raise ValueError("Cannot bootstrap AP: this split has no disengaged clips.")

    participants = participant_ids(clips.index.to_numpy())
    groups = [np.flatnonzero(participants == pid) for pid in np.unique(participants)]

    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(resamples):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        if y[idx].sum() == 0:
            continue
        scores.append(average_precision_score(y[idx], p[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))
