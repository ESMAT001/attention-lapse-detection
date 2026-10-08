from e_candeloro_Driver_State_Detection.eye_detector import EyeDetector
from Tandon_A_Drowsiness_Detection_Mediapipe_main.inference import (
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
    def __init__(self) -> None:
        self.eye_detector = EyeDetector()
        self.gaze_detector = GazeDetector()
        self.head_pose_estimator = HeadPoseFeature()
        self.mar_detector = mouth_feature
        self.perclos = PERCLOS(
            ear_thresh=EAR_CLOSED_THRESHOLD, time_period_s=PERCLOS_WINDOW_SECONDS
        )
        self.blink_counter = BlinkCounter(EAR_CLOSED_THRESHOLD)

    def reset_video_state(self) -> None:
        self.perclos.reset()
        self.blink_counter.reset()

    def extract(self, packet: FramePacket, landmarks: np.ndarray | None) -> FeatureRow:
        t_now_s = packet.timestamp_ms / 1000.0

        if landmarks is None:
            closed_part = round4(self.perclos.update(t_now_s, None))
            blink_flag = self.blink_counter.update(None)
            return FeatureRow(
                frame_index=packet.frame_index,
                timestamp_ms=packet.timestamp_ms,
                ear=0.0,
                mar=0.0,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
                perclos=closed_part,
                blinks=blink_flag,
                face_present=0,
                gaze_x=0.0,
                gaze_y=0.0,
            )

        frame_h = packet.frame.shape[0]
        frame_w = packet.frame.shape[1]

        ear = float(self.eye_detector.get_EAR(landmarks))
        closed_part = self.perclos.update(t_now_s, ear)
        blink_flag = self.blink_counter.update(ear)

        pose = self.head_pose_estimator.angles(
            frame=packet.frame, landmarks=landmarks, frame_size=(frame_w, frame_h)
        )
        if pose is None:
            roll = 0.0
            pitch = 0.0
            yaw = 0.0
        else:
            roll = pose[0]
            pitch = pose[1]
            yaw = pose[2]

        gaze_x, gaze_y = self.gaze_detector.extract(landmarks)
        mar = self.mar_detector(landmarks)

        return FeatureRow(
            frame_index=packet.frame_index,
            timestamp_ms=packet.timestamp_ms,
            ear=round4(ear),
            mar=round4(mar),
            roll=round4(roll),
            pitch=round4(pitch),
            yaw=round4(yaw),
            perclos=round4(closed_part),
            blinks=blink_flag,
            face_present=1,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
        )
