import cv2 as cv
import numpy as np

from attention_lapse_detection.utils.numeric import round4


class GazeDetector:
    """Iris offsets adapted from alireza787b/Python-Gaze-Face-Tracker."""

    # Pair each iris with the same eye's outer corner.
    RIGHT_EYE_IRIS = [469, 470, 471, 472]
    RIGHT_EYE_OUTER_CORNER = 33
    LEFT_EYE_IRIS = [474, 475, 476, 477]
    LEFT_EYE_OUTER_CORNER = 263

    def vector_position(
        self, point1: np.ndarray, point2: np.ndarray
    ) -> tuple[float, float]:

        x1, y1 = point1.ravel()
        x2, y2 = point2.ravel()
        return x2 - x1, y2 - y1

    def iris_shift_from_corner(
        self,
        points: np.ndarray,
        iris_indices: list[int],
        outer_corner_index: int,
    ) -> tuple[float, float]:
        iris_points = points[iris_indices].astype(np.float32)
        (cx, cy), _radius = cv.minEnclosingCircle(iris_points)
        iris_center = np.array([cx, cy], dtype=np.float32)

        outer_corner = points[outer_corner_index]

        dx, dy = self.vector_position(outer_corner, iris_center)
        return float(dx), float(dy)

    def extract(self, landmark: np.ndarray) -> tuple[float, float]:
        points = landmark[:, :2]

        l_dx, l_dy = self.iris_shift_from_corner(
            points,
            self.LEFT_EYE_IRIS,
            self.LEFT_EYE_OUTER_CORNER,
        )

        r_dx, r_dy = self.iris_shift_from_corner(
            points,
            self.RIGHT_EYE_IRIS,
            self.RIGHT_EYE_OUTER_CORNER,
        )

        gaze_x = (l_dx + r_dx) / 2.0
        gaze_y = (l_dy + r_dy) / 2.0

        return round4(gaze_x), round4(gaze_y)
