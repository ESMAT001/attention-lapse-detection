import cv2 as cv
import numpy as np
from attention_lapse_detection.utils.numeric import round4


class GazeDetector:
    """Gaze x/y from iris and eye-corner landmarks.

    Port of the iris tracking logic from alireza787b/Python-Gaze-Face-Tracker.
    The original works in pixels; here the landmarks arrive normalized.
    """

    # Each iris is paired with its own eye's outer corner. The original project
    # cross-paired them, which leaves a large one-sided baseline; same-eye pairing
    # keeps the two offsets symmetric so the averaged gaze centers near 0.
    RIGHT_EYE_IRIS = [469, 470, 471, 472]
    RIGHT_EYE_OUTER_CORNER = 33
    LEFT_EYE_IRIS = [474, 475, 476, 477]
    LEFT_EYE_OUTER_CORNER = 263

    def _vector_position(self, point1: np.ndarray, point2: np.ndarray):
        x1, y1 = point1.ravel()
        x2, y2 = point2.ravel()
        return x2 - x1, y2 - y1

    def _iris_relative_position(
        self,
        points: np.ndarray,
        iris_indices: list[int],
        outer_corner_index: int,
    ) -> tuple[float, float]:
        """Iris center position relative to the eye's outer corner."""
        iris_points = points[iris_indices].astype(np.float32)
        (cx, cy), _radius = cv.minEnclosingCircle(iris_points)
        iris_center = np.array([cx, cy], dtype=np.float32)

        outer_corner = points[outer_corner_index]

        dx, dy = self._vector_position(outer_corner, iris_center)
        return float(dx), float(dy)

    def extract(self, landmark: np.ndarray) -> tuple[float, float]:
        """Average both eyes relative iris positions into a single gaze x/y."""
        points = landmark[:, :2]

        l_dx, l_dy = self._iris_relative_position(
            points,
            self.LEFT_EYE_IRIS,
            self.LEFT_EYE_OUTER_CORNER,
        )

        r_dx, r_dy = self._iris_relative_position(
            points,
            self.RIGHT_EYE_IRIS,
            self.RIGHT_EYE_OUTER_CORNER,
        )

        gaze_x = (l_dx + r_dx) / 2.0
        gaze_y = (l_dy + r_dy) / 2.0

        return round4(gaze_x), round4(gaze_y)
