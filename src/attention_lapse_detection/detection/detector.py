"""Predict attention lapses from raw feature windows and a saved checkpoint."""

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from attention_lapse_detection.core_models.registry import CLASSIFIER_NAMES
from attention_lapse_detection.utils.numeric import round4

DISENGAGED, ENGAGED = 0, 1


def fill_absent(
    column: NDArray[np.float64], absent: NDArray[np.bool_]
) -> NDArray[np.float64]:
    valid = np.flatnonzero(~absent)
    if valid.size == 0:
        return np.full_like(column, np.nan)
    source = np.maximum.accumulate(np.where(absent, -1, np.arange(column.size)))
    source[source < 0] = valid[0]
    return column[source]


class Prediction:
    def __init__(self, prob_disengaged: float, threshold: float) -> None:
        self.prob_disengaged = prob_disengaged
        self.threshold = threshold
        self.is_lapse = prob_disengaged >= threshold
        self.label = "disengaged" if self.is_lapse else "engaged"

    def __repr__(self) -> str:
        return (
            f"Prediction(label={self.label!r}, "
            f"p_disengaged={round4(self.prob_disengaged)}, thr={round4(self.threshold)})"
        )


class Detector:
    def __init__(
        self,
        model: nn.Module,
        scaler: dict,
        threshold: float,
        meta: dict | None = None,
    ) -> None:
        self.model = model.eval()
        self.scaler = scaler
        self.threshold = threshold
        self.meta = meta or {}
        self.feature_order: list[str] = scaler["feature_order"]
        self.window_size = int(scaler["window_size"])
        self.impute_on_absent: list[str] = scaler["impute_on_absent"]

    @classmethod
    def load(cls, path: str | Path) -> "Detector":
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)

        model_class = CLASSIFIER_NAMES.get(checkpoint["model_class"])

        if model_class is None:
            raise KeyError(f"Unknown model_class {checkpoint['model_class']}")

        model = model_class(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["state_dict"])

        meta = {
            "model_class": checkpoint["model_class"],
            "fps": checkpoint["fps"],
            "drop_columns": checkpoint["drop_columns"],
            "val_ap": checkpoint["val_ap"],
        }

        return cls(model, checkpoint["scaler"], float(checkpoint["threshold"]), meta)

    def standardize(self, raw_window: NDArray[np.floating]) -> NDArray[np.float32]:
        s = self.scaler
        raw_arr = np.asarray(raw_window, dtype=np.float64)

        if raw_arr.ndim != 2 or raw_arr.shape[1] != len(self.feature_order):
            raise ValueError(
                f"expected (T, {len(self.feature_order)}) in feature order "
                f"{self.feature_order}, got {raw_arr.shape}"
            )

        current_feature_arr = raw_arr.copy()
        absent = (
            current_feature_arr[:, self.feature_order.index("face_present")] == 0
            if "face_present" in self.feature_order
            else np.zeros(current_feature_arr.shape[0], dtype=bool)
        )

        for j, feature in enumerate(self.feature_order):
            if feature in s["binary_features"]:
                continue

            column = current_feature_arr[:, j]

            if feature in self.impute_on_absent:
                column = fill_absent(column, absent)

            if feature in s["clip_lower_zero"]:
                column = np.clip(column, 0.0, None)

            column = np.clip(
                column, s["quantile_low"][feature], s["quantile_high"][feature]
            )

            column = (column - s["mean"][feature]) / s["std"][feature]
            current_feature_arr[:, j] = np.clip(column, -s["std_clip"], s["std_clip"])

        return np.nan_to_num(current_feature_arr, nan=0.0).astype(np.float32)

    @torch.no_grad()
    def predict(self, raw_window: NDArray[np.floating]) -> Prediction:
        tensor = torch.from_numpy(self.standardize(raw_window)).unsqueeze(0)
        probs = torch.softmax(self.model(tensor), dim=1).squeeze(0)

        return Prediction(float(probs[DISENGAGED]), self.threshold)

    def window_from_rows(self, rows: Sequence[dict[str, float]]) -> NDArray[np.float32]:
        return np.array(
            [[float(row[f]) for f in self.feature_order] for row in rows],
            dtype=np.float32,
        )

    def predict_rows(self, rows: Sequence[dict[str, float]]) -> Prediction:
        return self.predict(self.window_from_rows(rows))
