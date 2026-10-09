"""Print attention-lapse predictions for a dataset video."""

import argparse

from attention_lapse_detection.extraction.pipeline import build_pipeline
from attention_lapse_detection.detection.classify_cause import classify_cause
from attention_lapse_detection.detection.detector import Detector
from attention_lapse_detection.utils.checkpoints import resolve_checkpoint
from attention_lapse_detection.utils.numeric import round4
from attention_lapse_detection.utils.paths import PATHS

DEFAULT_MODEL = "gru_uni_attention_10fps_10s_all_features_s42"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--split", choices=["Train", "Test", "Validation"], default="Test"
    )
    parser.add_argument(
        "--video", default="9877360271", help="Video ID without the extension."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Checkpoint name or path."
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Frames between windows (default: window size).",
    )

    parser.add_argument("--show", action="store_true", help="Show the video overlay.")
    args = parser.parse_args()

    attention_lapse_detector = Detector.load(resolve_checkpoint(args.model))
    window_size = attention_lapse_detector.window_size
    stride = args.stride or window_size
    target_fps = attention_lapse_detector.meta.get("fps") or 10

    print(
        f"Model {attention_lapse_detector.meta['model_class']} | window {window_size} frames "
        f"= {window_size // target_fps}s @ {target_fps} fps | stride {stride} | "
        f"threshold {round4(attention_lapse_detector.threshold)}"
    )

    clip_dir = PATHS.raw_data / "DataSet" / args.split / args.video[:6] / args.video
    video_path = next(clip_dir.glob(f"{args.video}.*"), None)

    if video_path is None:
        parser.error(f"no video file for {args.video} under {clip_dir}")

    pipeline = build_pipeline(source=video_path, target_fps=target_fps, show=args.show)

    rows = [row.to_dict() for row in pipeline.run()]

    if len(rows) < window_size:
        print(f"Clip too short: {len(rows)} frames; need {window_size}.")
        return

    print(
        f"\n{'window':<15}{'start_ms':<15}{'end_ms':<15}{'label':<15}{'p(disengaged)':<15}cause"
    )

    lapse_windows = 0
    total = 0

    for window_index, start in enumerate(range(0, len(rows) - window_size + 1, stride)):
        window = rows[start : start + window_size]
        prediction = attention_lapse_detector.predict_rows(window)
        total += 1
        lapse_windows += int(prediction.is_lapse)
       
        cause = classify_cause(window, target_fps) if prediction.is_lapse else "-"
       
        print(
            f"{window_index:<15}{window[0]['timestamp_ms']:<15}{window[-1]['timestamp_ms']:<15}"
            f"{prediction.label:<15}{prediction.prob_disengaged:<15.3f}{cause}"
        )

    print(
        f"\nDisengaged: {lapse_windows}/{total} windows "
        f"({round4(lapse_windows / total * 100)}%)."
    )


if __name__ == "__main__":
    main()
