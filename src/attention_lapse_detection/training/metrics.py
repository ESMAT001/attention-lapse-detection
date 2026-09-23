from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, precision_recall_curve

if TYPE_CHECKING:
    from attention_lapse_detection.training.trainer import Trainer


def ap_disengaged(y_true: NDArray[np.int64], probs: NDArray[np.float32]):
    """AP for the disengaged class (class 0).

    y_true: (N,) class indices. probs: (N, C) softmax probabilities; column 0
    is P(disengaged), the score we rank by.
    """
    is_disengaged = (y_true == 0).astype(np.int64)
    return float(average_precision_score(is_disengaged, probs[:, 0]))


def best_f1_threshold(trainer: Trainer) -> tuple[float, float, float]:
    """Threshold on P(disengaged) maximising validation F1, with its precision and recall.

    This is the cutoff a saved model uses at inference time.
    """
    y_pos = (trainer.val_y_true == 0).astype(int)
    probs_disengaged = trainer.val_probs[:, 0]

    prec, rec, thr = precision_recall_curve(y_pos, probs_disengaged)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)

    best = int(np.nanargmax(f1s))

    best_thr = thr[min(best, len(thr) - 1)]  # thr is shorter than prec/rec by 1
    return float(best_thr), float(prec[best]), float(rec[best])

def per_clip(
    y_true: NDArray[np.int64],
    probs: NDArray[np.float32],
    clip_ids: NDArray,
) -> pd.DataFrame:
    """Aggregate window predictions to one row per clip.

    Windows of a clip are correlated and all carry the clip's label, so `y` comes
    from the clip and `p` is the mean of its windows P(disengaged).
    """
    if len(clip_ids) != len(y_true):
        raise ValueError(
            f"{len(clip_ids)} clip ids for {len(y_true)} windows; the clean "
            "arrays and their metadata are out of sync."
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
            f"{len(conflicting)} clips carry more than one window label, "
            f"{conflicting[:3]}. Every window inherits its clip's label, so this "
            "means the labels and the window metadata are out of sync."
        )
    
    return windows.groupby("clip").agg(y=("y", "first"), p=("p", "mean"))


def clip_ap(
    y_true: NDArray[np.int64],
    probs: NDArray[np.float32],
    clip_ids: NDArray,
) -> float:
    """AP at one window per clip."""

    clips = per_clip(y_true, probs, clip_ids)
    return float(average_precision_score(clips.y, clips.p))