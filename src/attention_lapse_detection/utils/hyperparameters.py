import json
from dataclasses import dataclass
from pathlib import Path

from attention_lapse_detection.types import GridCell
from attention_lapse_detection.utils.constants import (
    DEFAULT_FPS,
    DEFAULT_WINDOW_SECONDS,
)
from attention_lapse_detection.utils.numeric import round4
from attention_lapse_detection.utils.paths import PATHS

VALUES_JSON = PATHS.root / "hyperparameters" / "values.json"


@dataclass(frozen=True)
class Hyperparameters:
    hidden_size: int
    num_layers: int
    dropout: float
    learning_rate: float
    batch_size: int

    def __str__(self) -> str:
        return (
            f"hidden {self.hidden_size}, layers {self.num_layers}, "
            f"dropout {round4(self.dropout)}, lr {round4(self.learning_rate)}, "
            f"batch {self.batch_size}"
        )


def load_hyperparameters(path: Path | None = None) -> dict[GridCell, Hyperparameters]:
    path = VALUES_JSON if path is None else path

    if not path.is_file():
        raise FileNotFoundError(f"{path.name} not found. Run tuning/merge_shards.py.")

    settings = {}

    for record in json.loads(path.read_text(encoding="utf-8")):
        key = (record.pop("model"), record.pop("fps"), record.pop("window_seconds"))
        settings[key] = Hyperparameters(**record)

    return settings


def hyperparameters(
    model_name: str,
    fps: int = DEFAULT_FPS,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> Hyperparameters:
    settings = load_hyperparameters()

    lookup_key = (model_name, fps, window_seconds)

    if lookup_key not in settings:
        raise KeyError(f"No settings for {model_name}, {fps} fps, {window_seconds} s.")

    return settings[lookup_key]
