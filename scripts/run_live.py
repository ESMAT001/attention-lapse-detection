"""Preview the feature pipeline on a webcam, with the face mesh overlay."""

import argparse

from attention_lapse_detection.extraction.pipeline import build_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=int, default=1, help="Webcam index.")
    args = parser.parse_args()

    # mirror=False so the printed roll/yaw carry the same sign the model was
    # trained on, the preview is un-mirrored to match.
    build_pipeline(source=args.source, mirror=False, show=True).run()
