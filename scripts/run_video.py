"""Preview a dataset video and print its features."""

import argparse

from attention_lapse_detection.extraction.pipeline import build_pipeline
from attention_lapse_detection.utils.paths import PATHS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=["Train", "Test", "Validation"],
        default="Test",
        help="Dataset split the video belongs to (default: Test).",
    )
    parser.add_argument(
        "--video",
        default="9877360271",
        help="Video ID without .avi, e.g. 9877360271.",
    )
    args = parser.parse_args()
    
    video_path = (
        PATHS.raw_data
        / "DataSet"
        / args.split
        / args.video[:6]
        / args.video
        / f"{args.video}.avi"
    )

    # Mirroring flips roll/yaw signs relative to training.
    # Use mirror=False to match run_live.py.
    pipeline = build_pipeline(source=video_path, mirror=True, show=True)

    for row in pipeline.run():
        print(row.to_dict())