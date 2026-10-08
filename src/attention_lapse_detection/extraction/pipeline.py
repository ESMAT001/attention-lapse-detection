import cv2
from mediapipe.tasks.python import vision

from attention_lapse_detection.extraction.feature_extractor import FeatureExtractor
from attention_lapse_detection.extraction.frame_reader import FrameReader
from attention_lapse_detection.extraction.landmark_detector import LandmarkDetector
from attention_lapse_detection.extraction.live_face_mesh import LiveFaceMesh
from attention_lapse_detection.types import FeatureRow
from attention_lapse_detection.utils.constants import DEFAULT_WINDOW_SECONDS
from attention_lapse_detection.utils.paths import PATHS
from attention_lapse_detection.utils.numeric import round4

OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Pipeline:
    def __init__(
        self,
        reader: FrameReader,
        detector: LandmarkDetector,
        extractor: FeatureExtractor,
        face_mesh: LiveFaceMesh | None = None,
        show: bool = False,
    ) -> None:
        self.reader = reader
        self.detector = detector
        self.extractor = extractor
        self.face_mesh = face_mesh
        self.show = show

    def run(self) -> list[FeatureRow]:
        rows: list[FeatureRow] = []
        self.extractor.reset_video_state()

        while True:
            packet = self.reader.read()
            if packet is None:
                break

            marks = self.detector.detect(packet)
            row = self.extractor.extract(packet, marks)
            rows.append(row)

            if self.show:
                keep_going = self.paint_overlay(packet.frame, row, packet.timestamp_ms)
                if not keep_going:
                    break

        self.reader.release()
        self.detector.close()
        if self.face_mesh is not None:
            self.face_mesh.close()
        cv2.destroyAllWindows()

        return rows

    def paint_overlay(self, frame, row: FeatureRow, timestamp_ms: int) -> bool:
        # Show the total blink count, not the frame event.
        blinks_so_far = self.extractor.blink_counter.total_blinks

        if self.face_mesh is not None:
            self.face_mesh.detect(frame=frame, timestamp_ms=timestamp_ms)
            self.face_mesh.draw(frame, row.gaze_x, row.gaze_y, row.perclos, blinks_so_far)

        perclos_pct = round4(row.perclos * 100)
        info_line = (
            f"EAR {row.ear}, Pitch {row.pitch}, Roll {row.roll}, Yaw {row.yaw}, "
            f"MAR {row.mar}, PERCLOS {perclos_pct}%, Blinks {blinks_so_far}, "
            f"Gaze: x:{row.gaze_x}, y:{row.gaze_y}"
        )
        cv2.putText(frame, info_line, (10, 30), OVERLAY_FONT, 0.7, (255, 0, 0), 2)
        cv2.imshow("Video features", frame)

        pressed = cv2.waitKey(1)
        return pressed != ord("q")


def build_pipeline(
    source,
    target_fps: int | None = None,
    mirror: bool = False,
    show: bool = False,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> Pipeline:
    reader = FrameReader(
        source=source,
        mirror=mirror,
        target_fps=target_fps,
        window_seconds=window_seconds,
    )
    detector = LandmarkDetector(
        model_path=str(PATHS.face_landmarker),
        running_mode=vision.RunningMode.VIDEO,
    )
    extractor = FeatureExtractor()

    if show:
        face_mesh = LiveFaceMesh(PATHS.face_landmarker)
    else:
        face_mesh = None

    return Pipeline(
        reader=reader,
        detector=detector,
        extractor=extractor,
        face_mesh=face_mesh,
        show=show,
    )
