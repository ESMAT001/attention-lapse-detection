"""Preview webcam features with a face mesh overlay."""

import argparse

from attention_lapse_detection.extraction.pipeline import build_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=int, default=1, help="Webcam index.")
    args = parser.parse_args()

    # Keep frames unmirrored to match the training roll/yaw signs.
    build_pipeline(source=args.source, mirror=False, show=True).run()
