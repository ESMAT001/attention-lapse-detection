"""
Two corrections that the vendored code cannot carry, since it stays unmodified:

Order. `get_pose` documents its return as `yaw, pitch, roll`, but the real order
is `roll, pitch, yaw`, which is what this wrapper unpacks.

Sign. Pitch is negative nodding down, positive tilting up, and yaw runs the same
way round. Both are opposite to the head-pose literature, so published bounds
apply only after `to_reference_frame`. Zero means facing the camera axis, not an
anatomical reference. tests/test_head_pose_sign.py pins the convention.
"""

from e_candeloro_Driver_State_Detection.pose_estimation import (
    HeadPoseEstimator,
)
import numpy as np


def to_reference_frame(pitch: float, yaw: float):
    """Pitch and yaw flipped into the convention the literature states bounds in."""
    return -pitch, -yaw


class HeadPoseFeature:
    """Head orientation in degrees, in the convention described in this module."""

    def __init__(self, camera_matrix=None, dist_coeffs=None, show_axis: bool = False):
        self._estimator = HeadPoseEstimator(camera_matrix, dist_coeffs, show_axis)

    def angles(self, frame, landmarks: np.ndarray, frame_size: tuple[int, int]):
        """roll, pitch, yaw for one frame, or None when the PnP solve fails."""
        # get_pose returns each angle as a length-1 array, or a tuple of Nones.
        _, roll, pitch, yaw = self._estimator.get_pose(
            frame=frame, landmarks=landmarks, frame_size=frame_size
        )

        if roll is None or pitch is None or yaw is None:
            return None

        return float(roll[0]), float(pitch[0]), float(yaw[0])
