import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

from attention_lapse_detection.utils.numeric import round4

# Official MediaPipe Tasks face-mesh layers: tesselation, contours, irises.
MESH_LAYERS = [
    (
        vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        drawing_styles.get_default_face_mesh_tesselation_style,
    ),
    (
        vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        drawing_styles.get_default_face_mesh_contours_style,
    ),
    (
        vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
        drawing_styles.get_default_face_mesh_iris_connections_style,
    ),
    (
        vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
        drawing_styles.get_default_face_mesh_iris_connections_style,
    ),
]

# gaze_x/gaze_y are iris-center offsets relative to the eye corner, centered
# near 0, so the indicator treats 0 as the box center and amplifies the small
# offset by this many box-half-widths per unit to make the dot visible.
GAZE_INDICATOR_GAIN = 30.0
GAZE_BOX_SIZE = 150


class LiveFaceMesh:
    """Debug overlay: face mesh plus gaze box, PERCLOS and blink readouts."""

    def __init__(self, model_path):
        self.latest_result = None
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            result_callback=self._on_result,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def _on_result(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def detect(self, frame, timestamp_ms: int):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.landmarker.detect_async(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
        )

    def draw(self, frame, gaze_x: float, gaze_y: float, perclos: float, blinks: int):
        result = self.latest_result
        if result is None or not result.face_landmarks:
            return

        for connections, style in MESH_LAYERS:
            drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=result.face_landmarks[0],
                connections=connections,
                landmark_drawing_spec=None,
                connection_drawing_spec=style(),
            )

        self._text(frame, f"Gaze x:{gaze_x} y:{gaze_y}", 100, (0, 255, 0))
        self._draw_gaze_indicator(frame, gaze_x, gaze_y)
        perclos_color = (0, 0, 255) if perclos >= 0.2 else (0, 255, 0)
        self._text(frame, f"PERCLOS: {round4(perclos * 100)}%", 130, perclos_color)
        self._text(frame, f"Blinks: {blinks}", 160, (0, 255, 255))

    def _draw_gaze_indicator(self, frame, gaze_x: float, gaze_y: float):
        w = frame.shape[1]
        size, margin = GAZE_BOX_SIZE, 10
        x1, y1 = w - size - margin, margin
        x2, y2 = w - margin, margin + size

        # Translucent backdrop so the box stays readable over the face mesh.
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Border and center crosshair.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
        cx, cy = x1 + size // 2, y1 + size // 2
        cv2.line(frame, (x1, cy), (x2, cy), (80, 80, 80), 1)
        cv2.line(frame, (cx, y1), (cx, y2), (80, 80, 80), 1)

        # Offset 0 sits at the box center; amplify and clamp inside the box.
        half = size / 2
        dot_x = int(cx + max(min(gaze_x * GAZE_INDICATOR_GAIN, 1.0), -1.0) * half)
        dot_y = int(cy + max(min(gaze_y * GAZE_INDICATOR_GAIN, 1.0), -1.0) * half)
        cv2.circle(frame, (dot_x, dot_y), 6, (0, 255, 0), -1)
        cv2.circle(frame, (dot_x, dot_y), 6, (0, 0, 0), 1)

    def _text(self, frame, text: str, y: int, color: tuple[int, int, int]):
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def close(self):
        self.landmarker.close()
