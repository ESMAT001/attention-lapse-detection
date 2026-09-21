"""Preview the feature pipeline on one dataset video and print its feature rows."""

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
        help="Video id (the .avi base name), e.g. 9877360271.",
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

    # mirror=True is this script's default and flips the sign of roll and yaw
    # relative to how the model was trained. Preserved as-is.
    # pass mirror=False to read the signs the way run_live.py prints them.
    pipeline = build_pipeline(source=video_path, mirror=True, show=True)

    for row in pipeline.run():
        print(row.to_dict())