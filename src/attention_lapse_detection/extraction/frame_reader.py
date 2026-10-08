import time
import cv2

from attention_lapse_detection.types import FramePacket
from attention_lapse_detection.utils.constants import DEFAULT_WINDOW_SECONDS


class FrameReader:
    def __init__(
        self,
        source=0,
        mirror: bool = True,
        target_fps: int | None = 10,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ) -> None:
        self.source = source
        self.mirror = mirror
        self.target_fps = target_fps
        self.window_seconds = window_seconds

        self.is_live = isinstance(source, int)

        if self.is_live:
            open_target = source
        else:
            open_target = str(source)
        self.cap = cv2.VideoCapture(open_target)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")

        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.is_live and self.source_fps <= 0:
            raise RuntimeError(f"Could not read FPS from video: {source}")

        if self.is_live:
            if target_fps:
                self.effective_fps = target_fps
            else:
                self.effective_fps = 30
            self.frame_skip = 1
        else:
            if target_fps:
                self.effective_fps = target_fps
            else:
                self.effective_fps = self.source_fps
            fps_ratio = self.source_fps / self.effective_fps
            self.frame_skip = max(1, round(fps_ratio))

        self.window_size = int(self.effective_fps * self.window_seconds)

        self.frame_index = 0
        self.returned_frame_index = 0

    def read(self) -> FramePacket | None:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return None

            raw_index = self.frame_index
            self.frame_index += 1

            if not self.is_live and raw_index % self.frame_skip != 0:
                continue

            if self.mirror:
                frame = cv2.flip(frame, 1)

            # Use increasing timestamps for MediaPipe.
            if self.is_live:
                stamp_ms = int(time.monotonic() * 1000)
            else:
                stamp_ms = int((raw_index / self.source_fps) * 1000)

            packet = FramePacket(
                frame=frame,
                frame_index=self.returned_frame_index,
                timestamp_ms=stamp_ms,
            )
            self.returned_frame_index += 1
            return packet

    def release(self) -> None:
        self.cap.release()
