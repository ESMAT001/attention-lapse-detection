from e_candeloro_Driver_State_Detection.pose_estimation import HeadPoseEstimator

import numpy as np


class HeadPoseFeature:
    def __init__(
        self, camera_matrix=None, dist_coeffs=None, show_axis: bool = False
    ) -> None:
        self._pose_helper = HeadPoseEstimator(camera_matrix, dist_coeffs, show_axis)

    def angles(
        self, frame, landmarks: np.ndarray, frame_size: tuple[int, int]
    ) -> tuple[float, float, float] | None:

        _, roll, pitch, yaw = self._pose_helper.get_pose(
            frame=frame, landmarks=landmarks, frame_size=frame_size
        )

        if roll is None or pitch is None or yaw is None:
            return None

        return float(roll[0]), float(pitch[0]), float(yaw[0])
