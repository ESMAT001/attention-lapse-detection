from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from attention_lapse_detection.utils.constants import FEATURE_COLUMNS


@dataclass
class FramePacket:
    """One frame read by FrameReader, with its timestamp and post-downsampling index."""

    frame: np.ndarray
    timestamp_ms: int
    frame_index: int


@dataclass
class FeatureRow:
    """All features extracted from a single frame."""

    frame_index: int
    timestamp_ms: int
    ear: float
    mar: float
    roll: float
    pitch: float
    yaw: float
    perclos: float
    blinks: int
    face_present: int
    gaze_x: float
    gaze_y: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_list(self) -> list[float]:
        """The values the model sees, in FEATURE_COLUMNS order."""
        row = self.to_dict()
        return [float(row[column]) for column in FEATURE_COLUMNS]


@dataclass
class FeatureWindow:
    """A fixed-length slice of consecutive frames, used as one model input."""

    video_id: str
    engagement: int

    window_index: int
    start_frame: int
    end_frame: int
    start_timestamp_ms: int
    end_timestamp_ms: int

    features: np.ndarray

    def sample_id(self, split: str) -> str:
        """Stable key for joining features and metadata."""
        return f"{split}|{self.video_id}|{self.window_index}"

    def to_metadata_dict(self, split: str) -> dict:
        return {
            "sample_id": self.sample_id(split),
            "label_split": split,
            "video_id": self.video_id,
            "engagement": self.engagement,
            "window_index": self.window_index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_timestamp_ms": self.start_timestamp_ms,
            "end_timestamp_ms": self.end_timestamp_ms,
        }


@dataclass
class VideoResult:
    """Every per-frame FeatureRow from one video"""

    video_id: str
    engagement: int
    features: list[FeatureRow]

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                **feature.to_dict(),
                "video_id": self.video_id,
                "engagement": self.engagement,
            }
            for feature in self.features
        ]
        return pd.DataFrame(rows)

    def to_windows(
        self, window_size: int, stride: int | None = None
    ) -> list[FeatureWindow]:
        """Slice the frame rows into fixed-size windows"""
        if stride is None:
            stride = window_size

        windows = []

        # Stop early enough that the last window still has window_size frames
        for window_index, start in enumerate(
            range(0, len(self.features) - window_size + 1, stride)
        ):
            window_rows = self.features[start : start + window_size]

            windows.append(
                FeatureWindow(
                    video_id=self.video_id,
                    engagement=self.engagement,
                    window_index=window_index,
                    start_frame=window_rows[0].frame_index,
                    end_frame=window_rows[-1].frame_index,
                    start_timestamp_ms=window_rows[0].timestamp_ms,
                    end_timestamp_ms=window_rows[-1].timestamp_ms,
                    features=np.array(
                        [row.to_list() for row in window_rows], dtype=np.float32
                    ),
                )
            )

        return windows


@dataclass
class FeatureScaler:
    """Preprocessing stats fit on Train and reused for every other split."""

    mean: pd.Series
    std: pd.Series
    quantile_low: pd.Series
    quantile_high: pd.Series
