"""Settings selected by tuning/search.py, one entry per study."""

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


# Keys: (model, fps, window_seconds). Comments show confirmed clip-level val AP.
# September 2026 sweep on an A40 with CUDA.
SEARCHED = {
    ("gru", 10, 5): Hyperparameters(95, 2, 0.5, 0.00242776, 8),  # 0.2344
    ("gru_uni_attention", 10, 5): Hyperparameters(107, 2, 0.25, 0.00125789, 8),  # 0.2666
    ("lstm", 10, 5): Hyperparameters(179, 2, 0.05, 0.00127649, 64),  # 0.2198
    ("lstm_uni_attention", 10, 5): Hyperparameters(70, 1, 0.0, 0.00697828, 64),  # 0.2543
    ("gru", 15, 5): Hyperparameters(47, 3, 0.0, 0.00539948, 64),  # 0.2178
    ("gru_uni_attention", 15, 5): Hyperparameters(108, 2, 0.05, 0.00063431, 16),  # 0.2658
    ("lstm", 15, 5): Hyperparameters(193, 2, 0.05, 0.000561515, 8),  # 0.2238
    ("lstm_uni_attention", 15, 5): Hyperparameters(70, 1, 0.0, 0.00697828, 64),  # 0.2491
    ("gru", 30, 5): Hyperparameters(136, 1, 0.25, 0.000382348, 8),  # 0.2104
    ("gru_uni_attention", 30, 5): Hyperparameters(108, 2, 0.05, 0.000634310, 16),  # 0.2657
    ("lstm", 30, 5): Hyperparameters(179, 2, 0.05, 0.00127649, 32),  # 0.2043
    ("lstm_uni_attention", 30, 5): Hyperparameters(162, 1, 0.05, 0.00257831, 64),  # 0.2610
    ("gru", 10, 10): Hyperparameters(48, 1, 0.0, 0.00402155, 16),  # 0.2139
    ("gru_uni_attention", 10, 10): Hyperparameters(105, 2, 0.05, 0.000534317, 64),  # 0.2716
    ("lstm", 10, 10): Hyperparameters(120, 2, 0.05, 0.000993052, 64),  # 0.2132
    ("lstm_uni_attention", 10, 10): Hyperparameters(126, 2, 0.05, 0.000304165, 8),  # 0.2831
    ("gru", 15, 10): Hyperparameters(31, 2, 0.1, 0.00225754, 16),  # 0.2146
    ("gru_uni_attention", 15, 10): Hyperparameters(200, 3, 0.05, 0.000245908, 64),  # 0.2676
    ("lstm", 15, 10): Hyperparameters(70, 1, 0.0, 0.00697828, 64),  # 0.2088
    ("lstm_uni_attention", 15, 10): Hyperparameters(136, 2, 0.05, 0.00117828, 16),  # 0.2790
    ("gru", 30, 10): Hyperparameters(40, 1, 0.0, 0.00658629, 16),  # 0.2168
    ("gru_uni_attention", 30, 10): Hyperparameters(200, 3, 0.25, 0.00195779, 64),  # 0.2929
    ("lstm", 30, 10): Hyperparameters(70, 1, 0.0, 0.00697828, 64),  # 0.2101
    ("lstm_uni_attention", 30, 10): Hyperparameters(94, 3, 0.05, 0.000234471, 16),  # 0.2804
}


def hyperparameters(
    model_name: str,
    fps: int = DEFAULT_FPS,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> Hyperparameters:
    lookup_key = (model_name, fps, window_seconds)
    if lookup_key not in SEARCHED:
        raise KeyError(
            f"No search settings for model={model_name} "
            f"fps={fps} window_seconds={window_seconds}. Run:\n"
            f"  uv run python tuning/search.py --model {model_name} "
            f"--fps {fps} --window-seconds {window_seconds}\n"
            "Then copy the printed entry into SEARCHED."
        )
    return SEARCHED[lookup_key]