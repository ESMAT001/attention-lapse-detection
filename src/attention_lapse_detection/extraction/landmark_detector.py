import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from attention_lapse_detection.types import FramePacket


class LandmarkDetector:
    """Wrapper around MediaPipe Face Landmarker."""

    def __init__(self, model_path: str, running_mode: VisionTaskRunningMode):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=1,
            output_facial_transformation_matrixes=False,
            output_face_blendshapes=False,
        )

        self.detector = FaceLandmarker.create_from_options(options)

    def detect(self, packet: FramePacket) -> np.ndarray | None:
        """Return one face's landmarks, or None if not found."""
        # OpenCV gives BGR frames and MediaPipe expects RGB.
        rgb = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        # detect_for_video requires monotonically increasing timestamps.
        result = self.detector.detect_for_video(
            mp_image,
            packet.timestamp_ms,
        )

        if not result.face_landmarks:
            return None

        return np.array(
            [[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float32
        )

    def close(self) -> None:
        self.detector.close()
