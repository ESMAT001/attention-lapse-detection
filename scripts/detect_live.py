"""Show attention-lapse predictions on a webcam feed. Press q to quit."""

import argparse

import cv2
from mediapipe.tasks.python import vision

from attention_lapse_detection.extraction.feature_extractor import FeatureExtractor
from attention_lapse_detection.extraction.frame_reader import FrameReader
from attention_lapse_detection.extraction.landmark_detector import LandmarkDetector
from attention_lapse_detection.inference.cause_classifier import classify_cause
from attention_lapse_detection.inference.detector import Detector
from attention_lapse_detection.utils.numeric import round4
from attention_lapse_detection.utils.paths import PATHS, resolve_checkpoint

FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_MODEL = "gru_uni_attention_10fps_10s_all_features_s42"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Checkpoint name or path")
    parser.add_argument("--source", type=int, default=1, help="Webcam index")
    args = parser.parse_args()

    lapse_detector = Detector.load(resolve_checkpoint(args.model))
    window_size = lapse_detector.window_size
    target_fps = lapse_detector.meta.get("fps") or 10
    interval_ms = 1000.0 / target_fps
    window_seconds = window_size // target_fps
    print(
        f"Model: {lapse_detector.meta['model_class']} | window: {window_size} frames "
        f"({window_seconds}s at {target_fps} fps) | threshold: "
        f"{round4(lapse_detector.threshold)} | camera {args.source}. Press q to quit."
    )

    reader = FrameReader(source=args.source, mirror=False)
    landmark_detector = LandmarkDetector(
        model_path=str(PATHS.face_landmarker),
        running_mode=vision.RunningMode.VIDEO,
    )
    extractor = FeatureExtractor()
    extractor.reset_video_state()

    buffer: list[dict[str, float]] = []
    next_sample_ms: float | None = None
    overlay: tuple[str, tuple[int, int, int]] | None = None

    while True:
        packet = reader.read()
        if packet is None:
            break

        # Match the training frame rate.
        if next_sample_ms is None or packet.timestamp_ms >= next_sample_ms:
            next_sample_ms = packet.timestamp_ms + interval_ms
            row = extractor.extract(packet, landmark_detector.detect(packet))
            buffer.append(row.to_dict())

            if len(buffer) == window_size:
                prediction = lapse_detector.predict_rows(buffer)
                label = "Engaged"
                if prediction.is_lapse:
                    label = (
                        "Disengaged - "
                        f"{classify_cause(buffer, target_fps)}"
                    )
                overlay = (
                    f"{label}  P(disengaged): {round4(prediction.prob_disengaged)} "
                    f"cutoff: {round4(prediction.threshold)}",
                    (0, 0, 255) if prediction.is_lapse else (0, 180, 0),
                )
                buffer.clear()

        # Mirror only the preview.
        frame = cv2.flip(packet.frame, 1)

        y = 40
        if overlay is not None:
            text, color = overlay
            cv2.putText(frame, text, (10, y), FONT, 0.8, color, 2)
            y = 75

        seconds = len(buffer) / target_fps
        cv2.putText(
            frame,
            f"Collecting frames: {len(buffer)}/{window_size}  "
            f"({round4(seconds)}/{window_seconds}s)",
            (10, y),
            FONT,
            0.7,
            (200, 200, 200),
            2,
        )

        cv2.imshow("Attention-lapse detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    reader.release()
    landmark_detector.close()
    cv2.destroyAllWindows()


