import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

from attention_lapse_detection.training.metrics import best_f1_threshold
from attention_lapse_detection.training.trainer import Trainer
from attention_lapse_detection.utils.numeric import round4


def report_ap(trainer: Trainer) -> None:
    """AP and best-F1 threshold for the disengaged class."""

    y_val = trainer.val_y_true
    probs_disengaged = trainer.val_probs[:, 0]
    y_pos = (y_val == 0).astype(int)

    ap = average_precision_score(y_pos, probs_disengaged)
    baseline = y_pos.mean()  # Random AP baseline: fraction of disengaged samples.
    print(f"Disengagement AP: {round4(ap)} | random baseline: {round4(baseline)}")

    best_thr, best_prec, best_rec = best_f1_threshold(trainer)
    print(
        f"Best F1 threshold: {round4(best_thr)} | "
        f"precision {round4(best_prec)}, recall {round4(best_rec)}"
    )

    y_pred = np.where(probs_disengaged >= best_thr, 0, 1)
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

    caught, total = cm[0, 0], cm[0].sum()

    print(
        f"Confusion matrix at threshold {round4(best_thr)} "
        f"(rows: actual, columns: predicted) [disengaged, engaged]:\n{cm}\n"
        f"Disengaged windows detected: {caught}/{total}"
    )
