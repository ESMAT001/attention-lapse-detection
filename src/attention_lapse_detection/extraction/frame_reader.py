import time
from pathlib import Path
from tkinter.tix import WINDOW
import cv2
from attention_lapse_detection.types import FramePacket
from attention_lapse_detection.utils.constants import DEFAULT_WINDOW_SECONDS


class FrameReader:
    """Reads frames from a video file or webcam, downsampling to target_fps."""

    def __init__(
        self,
        source: str | int = 0,
        mirror: bool = True,
        target_fps: int | None = 10,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ):
        self.source = source
        self.mirror = mirror
        self.target_fps = target_fps
        self.window_seconds = window_seconds

        # Integer source is the webcam
        self.is_live = isinstance(source, int)

        self.capture = cv2.VideoCapture(source if self.is_live else str(source))

        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

        self.source_fps = self.capture.get(cv2.CAP_PROP_FPS)
        if not self.is_live and self.source_fps <= 0:
            raise RuntimeError(f"Failed to get FPS for video source: {source}")

        if self.is_live:
            # A webcam delivers in real time, so we can't skip ahead.
            self.effective_fps = target_fps or 30
            self.frame_skip = 1
        else:
            # Downsample by keeping every Nth frame (30fps -> 10fps -> skip 3).
            self.effective_fps = target_fps or self.source_fps
            self.frame_skip = max(1, round(self.source_fps / self.effective_fps))

        self.window_size = int(self.effective_fps * self.window_seconds)

        self.frame_index = 0  # Frames read from the source
        self.returned_frame_index = 0  # Frames returned after downsampling

    def read(self) -> FramePacket | None:
        """Return the next kept frame, or None at end of stream."""

        while True:
            ok, frame = self.capture.read()
            if not ok:
                return None

            original_frame_index = self.frame_index
            self.frame_index += 1

            if not self.is_live and (original_frame_index % self.frame_skip != 0):
                continue

            if self.mirror:
                frame = cv2.flip(frame, 1)

            # MediaPipe needs monotonically increasing timestamps.
            if self.is_live:
                timestamp_ms = int(time.monotonic() * 1000)
            else:
                timestamp_ms = int((original_frame_index / self.source_fps) * 1000)

            packet = FramePacket(
                frame=frame,
                frame_index=self.returned_frame_index,
                timestamp_ms=timestamp_ms,
            )
            self.returned_frame_index += 1
            return packet

    def release(self) -> None:
        self.capture.release()
