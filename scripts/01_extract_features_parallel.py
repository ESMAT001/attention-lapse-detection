"""Extract feature windows from every DAiSEE clip, in parallel, for every build."""

import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


from attention_lapse_detection.utils.parallel import default_max_workers

import cv2
import numpy as np
import pandas as pd

from attention_lapse_detection.extraction.pipeline import build_pipeline
from attention_lapse_detection.types import FeatureWindow, VideoResult
from attention_lapse_detection.utils.cli import resolve_variants
from attention_lapse_detection.utils.constants import (
    FPS_OPTIONS,
    LABELS_SPLITS,
    STRIDE_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    WINDOW_SECONDS_OPTIONS,
)

from attention_lapse_detection.utils.data_paths import (
    chunks_split_dir,
    failed_extraction_csv,
    labels_csv,
)

from attention_lapse_detection.utils.paths import PATHS

CHUNK_SIZE = 100
PARTIAL_SUFFIX = ".partial"

INT_METADATA = [
    "window_index",
    "start_frame",
    "end_frame",
    "start_timestamp_ms",
    "end_timestamp_ms",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    test_parser = subparsers.add_parser(
        "test", help="Run get_windows_from_video for a single clip and exit."
    )
    test_parser.add_argument("video_id", help="Clip filename, e.g. 1100011002.avi")
    test_parser.add_argument(
        "split",
        choices=sorted(LABELS_SPLITS.keys()),
        help="Label split the clip belongs to.",
    )
    test_parser.add_argument(
        "--fps",
        type=int,
        default=FPS_OPTIONS[0],
        choices=list(FPS_OPTIONS),
        help=f"Target fps (default {FPS_OPTIONS[0]}).",
    )
    test_parser.add_argument(
        "--window-seconds",
        type=int,
        default=argparse.SUPPRESS,
        help=f"Window length in seconds (default {DEFAULT_WINDOW_SECONDS}).",
    )
    test_parser.add_argument(
        "--stride-seconds",
        type=int,
        default=argparse.SUPPRESS,
        help="Hop between windows in seconds (default: same as --window-seconds).",
    )

    parser.add_argument(
        "--window-seconds",
        type=int,
        default=None,
        help=(
            f"Length of one window in seconds. Omit to build every variant in {list(WINDOW_SECONDS_OPTIONS)} in one run."
        ),
    )
    parser.add_argument(
        "--stride-seconds",
        type=int,
        default=None,
        help=(
            "Hop between consecutive windows in seconds (default: same as --window-seconds)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_max_workers(),
        help="Parallel worker processes (default: cpu_count - 2).",
    )
    return parser


def to_engagement_label(value: int) -> int:
    """Validate a raw engagement label."""
    if value in (0, 1):
        return value
    raise ValueError(f"Unexpected engagement label: {value!r}")


def lookup_engagement(video_id: str, split: str) -> int:
    """Find one clip's binarized engagement label."""
    df = pd.read_csv(labels_csv(split))
    row = df.loc[df["ClipID"].astype(str) == video_id]
    if row.empty:
        raise KeyError(f"{video_id!r} not found in {split} labels")
    return to_engagement_label(int(row["Engagement"].iloc[0]))


def make_video_path(video_id: str, split: str) -> Path:
    """Build the raw video path for one clip."""
    return PATHS.raw_data / "DataSet" / split / video_id[:6] / video_id[:-4] / video_id


def get_windows_from_video(
    video_id: str,
    engagement: int,
    split: str,
    fps: int,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    stride_seconds: int = STRIDE_SECONDS,
) -> list[FeatureWindow]:
    """Extract feature windows from one video at the given fps."""

    pipeline = build_pipeline(
        source=make_video_path(video_id, split),
        target_fps=fps,
        window_seconds=window_seconds,
    )

    video_result = VideoResult(
        video_id=video_id, engagement=engagement, features=pipeline.run()
    )

    return video_result.to_windows(
        window_size=pipeline.reader.window_size, stride=fps * stride_seconds
    )


def run_test(
    video_id: str,
    split: str,
    fps: int,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    stride_seconds: int = STRIDE_SECONDS,
) -> None:
    """Run extraction for one clip."""
    engagement = lookup_engagement(video_id, split)

    print(
        f"Testing {video_id} (split={split}, fps={fps}, window={window_seconds}s, stride={stride_seconds}s, engagement={engagement})"
    )

    windows = get_windows_from_video(
        video_id, engagement, split, fps, window_seconds, stride_seconds
    )

    print(f"Produced {len(windows)} window(s)")


def load_engagements(split: str) -> dict[str, int]:
    path = labels_csv(split)
    if not path.is_file():
        raise FileNotFoundError(f"Binarized labels missing for {split}: {path}. ")

    df = pd.read_csv(path)
    return {
        str(clip_id): to_engagement_label(int(engagement))
        for clip_id, engagement in zip(df["ClipID"], df["Engagement"])
    }


def scan_extracted_chunks(
    fps: int,
    split: str,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> tuple[set[str], int]:
    """Clip ids already written for this bucket, and the highest chunk index used.

    Resume depends on both: the ids say which clips to skip, and the index says
    where to continue numbering.
    """
    directory = chunks_split_dir(fps, split, window_seconds)
    extracted: set[str] = set()
    highest_index = 0

    # Leftovers from a kill
    for stale in directory.glob(f"*{PARTIAL_SUFFIX}"):
        print(f"  discarding incomplete chunk from an earlier run: {stale.name}")
        stale.unlink()

    for path in sorted(directory.glob(f"{split}_windows_chunk_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as chunk:
                extracted.update(str(video_id) for video_id in chunk["video_id"])
        except Exception as exc:
            raise RuntimeError(f"Unreadable chunk {path} ({exc})") from exc

        highest_index = max(highest_index, int(path.stem.rsplit("_", 1)[-1]))

    return extracted, highest_index


def process_one_video(args):
    """Extract from one video and return errors instead of raising."""
    # OpenCV keeps its own thread pool that THREAD_ENV_VARS does not reach.
    cv2.setNumThreads(1)

    video_id, engagement, split, fps, window_seconds, stride_seconds = args
    try:
        windows = get_windows_from_video(
            video_id, engagement, split, fps, window_seconds, stride_seconds
        )
        return video_id, windows, None
    except Exception:
        return video_id, None, traceback.format_exc()


def save_window_chunk(
    windows: list[FeatureWindow],
    chunk_index: int,
    split: str,
    fps: int,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> None:
    """Save one chunk of feature windows for one (window, fps, split) bucket

    Written to a .partial sibling and renamed into place, so a kill or a full
    disk write can never leave a half formed .npz for 02_merge_chunks.py.
    """
    if not windows:
        return

    X = np.stack([window.features for window in windows])
    y = np.array([window.engagement for window in windows], dtype=np.int64)
    metadata = [window.to_metadata_dict(split) for window in windows]

    npz_path = (
        chunks_split_dir(fps, split, window_seconds)
        / f"{split}_windows_chunk_{chunk_index:04d}.npz"
    )

    arrays: dict[str, Any] = {
        "X": X,
        "y": y,
        "sample_id": np.array([m["sample_id"] for m in metadata]),
        "video_id": np.array([m["video_id"] for m in metadata]),
        "label_split": np.full(len(windows), split),
        **{
            key: np.array([m[key] for m in metadata], dtype=np.int64)
            for key in INT_METADATA
        },
    }

    partial_path = npz_path.with_name(npz_path.name + PARTIAL_SUFFIX)
    with open(partial_path, "wb") as handle:
        np.savez_compressed(handle, **arrays)

    os.replace(partial_path, npz_path)

    print(
        f"  [{window_seconds}s/{fps}fps] chunk {chunk_index:04d}: X{X.shape} y{y.shape} -> {npz_path.name}"
    )


def write_failed_extractions(
    failed: list[dict[str, str]],
    fps: int,
    split: str,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> None:
    """Record failures"""

    out_path = failed_extraction_csv(fps, split, window_seconds)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(failed)
    if out_path.is_file():
        previous = pd.read_csv(out_path)
        kept = previous[~previous["video_id"].astype(str).isin(df["video_id"])]
        df = pd.concat([kept, df], ignore_index=True)

    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} failed video(s) to: {out_path}")


def process_split_at_fps(
    fps: int,
    split: str,
    max_workers: int,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    stride_seconds: int = STRIDE_SECONDS,
) -> bool:
    """Extract one (fps, split) into its window chunks.
    clips already present in this bucket's chunks are skipped and new
    chunks continue the existing numbering.
    """
    engagements = load_engagements(split)

    chunks_split_dir(fps, split, window_seconds).mkdir(parents=True, exist_ok=True)

    buffer: list[FeatureWindow] = []
    count = 0
    failed: list[dict[str, str]] = []

    extracted, index = scan_extracted_chunks(fps, split, window_seconds)

    jobs = [
        (clip_id, engagement, split, fps, window_seconds, stride_seconds)
        for clip_id, engagement in engagements.items()
        if clip_id not in extracted
    ]

    total = len(engagements)

    print(f"[{window_seconds}s/{fps}fps/{split}] {len(jobs)} clip(s) to do of {total}")

    if total - len(jobs):
        print(
            f" resuming: {total - len(jobs)} clip(s) already extracted; "
            f"continuing at chunk {index + 1:04d}"
        )

    if not jobs:
        print(f"Finished {split} at {fps}fps.")
        return False

    print(f"using {max_workers} worker process")
    interrupted = False

    with ProcessPoolExecutor(max_workers) as executor:
        futures = [executor.submit(process_one_video, job) for job in jobs]

        try:
            for done_count, future in enumerate(as_completed(futures), start=1):
                video_id, windows, error = future.result()

                if error is not None or windows is None:
                    print(f"  [{done_count}/{len(jobs)}] Failed: {video_id}")
                    failed.append(
                        {"video_id": video_id, "error": error or "no windows returned"}
                    )
                    continue

                print(
                    f"  [{done_count}/{len(jobs)}] Done: {video_id}, windows: {len(windows)}"
                )

                buffer.extend(windows)
                count += 1

                if count >= CHUNK_SIZE:
                    index += 1
                    save_window_chunk(buffer, index, split, fps, window_seconds)
                    buffer.clear()
                    count = 0

        except KeyboardInterrupt:
            # Drop everything still queued so shutdown waits only on the clips
            # already in flight, then fall through and flush what is buffered.
            interrupted = True
            cancelled = sum(1 for future in futures if future.cancel())
            print(
                f" Interrupted: cancelled {cancelled} queued clip(s), waiting for the {max_workers} in flight to finish, then flushing buffers."
            )

    if buffer:
        index += 1
        save_window_chunk(buffer, index, split, fps, window_seconds)

    if failed:
        write_failed_extractions(failed, fps, split, window_seconds)

    if interrupted:
        print(f"Stopped during {split} at {fps}fps; re-run to resume.")
    else:
        print(f"Finished {split} at {fps}fps.")
    return interrupted


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    window_variants = resolve_variants(args.window_seconds, WINDOW_SECONDS_OPTIONS)

    if any(window < 1 for window in window_variants):
        parser.error("--window-seconds must be >= 1")
    if args.stride_seconds is not None and args.stride_seconds < 1:
        parser.error("--stride-seconds must be >= 1")

    def stride_for(window_seconds: int) -> int:
        return window_seconds if args.stride_seconds is None else args.stride_seconds

    if args.command == "test":
        window_seconds = (
            DEFAULT_WINDOW_SECONDS
            if args.window_seconds is None
            else args.window_seconds
        )
        run_test(
            args.video_id,
            args.split,
            args.fps,
            window_seconds,
            window_seconds if args.stride_seconds is None else args.stride_seconds,
        )
        sys.exit(0)

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    for window_seconds in window_variants:
        for fps in FPS_OPTIONS:
            for split in LABELS_SPLITS:
                print(f"window={window_seconds}s, fps={fps}, split={split}")
                interrupted = process_split_at_fps(
                    fps, split, args.workers, window_seconds, stride_for(window_seconds)
                )

                if interrupted:
                    sys.exit(130)
