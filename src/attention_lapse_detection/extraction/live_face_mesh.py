import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

from attention_lapse_detection.utils.numeric import round4

# MediaPipe mesh, contours and irises.
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

GAZE_INDICATOR_GAIN = 30.0
GAZE_BOX_SIZE = 150


class LiveFaceMesh:
    def __init__(self, model_path) -> None:
        self.latest_result = None
        base_opts = python.BaseOptions(model_asset_path=str(model_path))
        mesh_opts = vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            result_callback=self.stash_latest_result,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(mesh_opts)

    def stash_latest_result(self, result, output_image, timestamp_ms) -> None:
        self.latest_result = result

    def detect(self, frame, timestamp_ms: int) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.landmarker.detect_async(mp_frame, timestamp_ms)

    def draw(
        self, frame, gaze_x: float, gaze_y: float, perclos: float, blinks: int
    ) -> None:
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

        self.write_label(frame, f"Gaze x:{gaze_x} y:{gaze_y}", 100, (0, 255, 0))
        self.paint_gaze_box(frame, gaze_x, gaze_y)

        if perclos >= 0.2:
            perclos_color = (0, 0, 255)
        else:
            perclos_color = (0, 255, 0)
        perclos_pct = round4(perclos * 100)
        self.write_label(frame, f"PERCLOS: {perclos_pct}%", 130, perclos_color)
        self.write_label(frame, f"Blinks: {blinks}", 160, (0, 255, 255))

    def paint_gaze_box(self, frame, gaze_x: float, gaze_y: float) -> None:
        frame_w = frame.shape[1]
        size = GAZE_BOX_SIZE
        margin = 10
        x1 = frame_w - size - margin
        y1 = margin
        x2 = frame_w - margin
        y2 = margin + size

        shaded = frame.copy()
        cv2.rectangle(shaded, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(shaded, 0.4, frame, 0.6, 0, frame)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
        mid_x = x1 + size // 2
        mid_y = y1 + size // 2
        cv2.line(frame, (x1, mid_y), (x2, mid_y), (80, 80, 80), 1)
        cv2.line(frame, (mid_x, y1), (mid_x, y2), (80, 80, 80), 1)

        half = size / 2

        push_x = min(gaze_x * GAZE_INDICATOR_GAIN, 1.0)
        push_x = max(push_x, -1.0)
        push_y = min(gaze_y * GAZE_INDICATOR_GAIN, 1.0)
        push_y = max(push_y, -1.0)

        dot_x = int(mid_x + push_x * half)
        dot_y = int(mid_y + push_y * half)
        cv2.circle(frame, (dot_x, dot_y), 6, (0, 255, 0), -1)
        cv2.circle(frame, (dot_x, dot_y), 6, (0, 0, 0), 1)

    def write_label(
        self, frame, text: str, y: int, color: tuple[int, int, int]
    ) -> None:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def close(self) -> None:
        self.landmarker.close()
