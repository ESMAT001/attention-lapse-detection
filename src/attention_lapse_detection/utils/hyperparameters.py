"""Hyperparameters adopted by the tuning/search.py, one entry per study."""

from dataclasses import dataclass
from attention_lapse_detection.utils.constants import (
    DEFAULT_FPS,
    DEFAULT_WINDOW_SECONDS,
)
from attention_lapse_detection.utils.numeric import round4


@dataclass(frozen=True)
class Hyperparameters:
    """The five parameters each study searched."""

    hidden_size: int
    num_layers: int
    dropout: float
    learning_rate: float
    batch_size: int

    def __str__(self):
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, "
            f"dropout={round4(self.dropout)}, "
            f"learning_rate={self.learning_rate}, "
            f"batch_size={self.batch_size}"
        )
