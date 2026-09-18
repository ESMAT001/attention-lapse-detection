from third_party.e_candeloro_Driver_State_Detection.eye_detector import EyeDetector
from third_party.Tandon_A_Drowsiness_Detection_Mediapipe_main.inference import (
    mouth_feature,
)

from attention_lapse_detection.features.blink_counter import BlinkCounter
from attention_lapse_detection.features.gaze_detector import GazeDetector
from attention_lapse_detection.features.head_pose import HeadPoseFeature
from attention_lapse_detection.features.perclos import PERCLOS

import numpy as np

from attention_lapse_detection.types import FeatureRow, FramePacket
from attention_lapse_detection.utils.constants import (
    EAR_CLOSED_THRESHOLD,
    PERCLOS_WINDOW_SECONDS,
)
from attention_lapse_detection.utils.numeric import round4


class FeatureExtractor:
    """Builds one FeatureRow per frame from the six feature sources."""

    def __init__(self):
        self.eye_detector = EyeDetector()
        self.blink_counter = BlinkCounter(EAR_CLOSED_THRESHOLD)
        self.gaze_detector = GazeDetector()
        self.head_pose_feature = HeadPoseFeature()
        self.perclos = PERCLOS(PERCLOS_WINDOW_SECONDS)
        self.blink_counter = BlinkCounter(EAR_CLOSED_THRESHOLD)
        self.mar_detector = mouth_feature

    def reset_video_state(self):
        """Clear PERCLOS and blink states"""
        self.perclos.reset()
        self.blink_counter.reset()

    def extract(
        self, frame_packet: FramePacket, landmarks: np.ndarray | None
    ) -> FeatureRow:
        t_now_s = frame_packet.timestamp_ms / 1000.0

        if landmarks is None:
            return FeatureRow(
                frame_index=frame_packet.frame_index,
                timestamp_ms=frame_packet.timestamp_ms,
                ear=0.0,
                mar=0.0,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
                perclos=round4(self.perclos.update(t_now_s, None)),
                blinks=self.blink_counter.update(None),
                face_present=0,
                gaze_x=0.0,
                gaze_y=0.0,
            )

        h, w = frame_packet.frame.shape[:2]

        ear = float(self.eye_detector.get_EAR(landmarks))
        perclos_score = self.perclos.update(t_now_s, ear)
        blink_event = self.blink_counter.update(ear)

        head_pose = self.head_pose_feature.angles(
            frame=frame_packet.frame, landmarks=landmarks, frame_size=(w, h)
        )

        roll, pitch, yaw = head_pose if head_pose is not None else (0.0, 0.0, 0.0)

        mar = self.mar_detector(landmarks)

        gaze_x, gaze_y = self.gaze_detector.extract(landmarks)
        
        return FeatureRow(
            frame_index=frame_packet.frame_index,
            timestamp_ms=frame_packet.timestamp_ms,
            ear=round4(ear),
            mar=round4(mar),
            roll=round4(roll),
            pitch=round4(pitch),
            yaw=round4(yaw),
            perclos=round4(perclos_score),
            blinks=blink_event,
            face_present=1,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
        )
