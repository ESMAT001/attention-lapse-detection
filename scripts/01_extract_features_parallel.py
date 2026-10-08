"""Extract DAiSEE feature windows in parallel. Resume unfinished builds."""

import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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

VideoJob = tuple[str, int, str, int, int, int]
VideoJobResult = tuple[str, list[FeatureWindow] | None, str | None]

INT_METADATA = [
    "window_index",
    "start_frame",
    "end_frame",
    "start_timestamp_ms",
    "end_timestamp_ms",
]


def checked_engagement(value: int) -> int:
    if value == 0 or value == 1:
        return value
    raise ValueError(f"Bad engagement label: {value}")


def find_avi_location(video_id: str, split: str) -> Path:
    person_folder = video_id[:6]
    clip_folder = video_id[:-4]
    return PATHS.raw_data / "DataSet" / split / person_folder / clip_folder / video_id


def slice_clip_into_windows(
    video_id: str,
    engagement: int,
    split: str,
    fps: int,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    stride_seconds: int = STRIDE_SECONDS,
) -> list[FeatureWindow]:

    runner = build_pipeline(
        source=find_avi_location(video_id, split),
        target_fps=fps,
        window_seconds=window_seconds,
    )

    frame_rows = runner.run()

    clip_result = VideoResult(
        video_id=video_id, engagement=engagement, features=frame_rows
    )

    frames_per_window = runner.reader.window_size
    step_frames = fps * stride_seconds

    return clip_result.to_windows(window_size=frames_per_window, stride=step_frames)


def handle_single_clip(job: VideoJob) -> VideoJobResult:
    # OpenCV has a separate thread pool.
    cv2.setNumThreads(1)

    video_id, engagement, split, fps, window_seconds, stride_seconds = job

    try:
        clip_windows = slice_clip_into_windows(
            video_id, engagement, split, fps, window_seconds, stride_seconds
        )
        return video_id, clip_windows, None

    except Exception:
        return video_id, None, traceback.format_exc()


def write_chunk_to_disk(
    windows: list[FeatureWindow],
    chunk_index: int,
    split: str,
    fps: int,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
):
    if not windows:
        return

    feature_list = []
    label_list = []
    info_rows = []

    for window in windows:
        feature_list.append(window.features)
        label_list.append(window.engagement)
        info_rows.append(window.to_metadata_dict(split))

    X = np.stack(feature_list)
    y = np.array(label_list, dtype=np.int64)

    sample_ids = []
    video_ids = []

    for row in info_rows:
        sample_ids.append(row["sample_id"])
        video_ids.append(row["video_id"])

    arrays = {}

    arrays["X"] = X
    arrays["y"] = y
    arrays["sample_id"] = np.array(sample_ids)
    arrays["video_id"] = np.array(video_ids)
    arrays["label_split"] = np.full(len(windows), split)

    for key in INT_METADATA:
        column_vals = []

        for row in info_rows:
            column_vals.append(row[key])

        arrays[key] = np.array(column_vals, dtype=np.int64)

    chunk_folder = chunks_split_dir(fps, split, window_seconds)

    file_name = f"{split}_windows_chunk_{chunk_index:04d}.npz"

    npz_path = chunk_folder / file_name
    half_done_path = npz_path.with_name(npz_path.name + PARTIAL_SUFFIX)

    with open(half_done_path, "wb") as handle:
        np.savez_compressed(handle, **arrays)

    os.replace(half_done_path, npz_path)

    print(
        f"  [{window_seconds}s/{fps}fps] chunk {chunk_index:04d}: "
        f"X{X.shape} y{y.shape} -> {npz_path.name}"
    )


def collect_split_labels(split: str) -> dict[str, int]:
    label_path = labels_csv(split)

    if not label_path.is_file():
        raise FileNotFoundError(
            f"Labels not found for {split}: {label_path}. "
            f"Run 00_binarize_labels.py first."
        )

    label_table = pd.read_csv(label_path)
    labels_by_clip: dict[str, int] = {}

    for clip_id, engagement in zip(label_table["ClipID"], label_table["Engagement"]):
        labels_by_clip[str(clip_id)] = checked_engagement(int(engagement))

    return labels_by_clip


def look_over_saved_chunks(
    fps: int,
    split: str,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> tuple[set[str], int]:

    chunk_folder = chunks_split_dir(fps, split, window_seconds)
    seen_clips: set[str] = set()

    last_number = 0

    for leftover in chunk_folder.glob(f"*{PARTIAL_SUFFIX}"):
        print(f"  Removing incomplete chunk: {leftover.name}")
        leftover.unlink()

    chunk_files = sorted(chunk_folder.glob(f"{split}_windows_chunk_*.npz"))

    for chunk_file in chunk_files:
        try:
            with np.load(chunk_file, allow_pickle=False) as chunk:
                for video_id in chunk["video_id"]:
                    seen_clips.add(str(video_id))

        except Exception as exc:
            raise RuntimeError(
                f"Could not read chunk {chunk_file} ({exc}). It may be cut off from a killed run. Delete it and rerun."
            )

        name_parts = chunk_file.stem.split("_")
        chunk_number = int(name_parts[-1])

        if chunk_number > last_number:
            last_number = chunk_number

    return seen_clips, last_number


def work_on_split(
    fps: int,
    split: str,
    max_workers: int,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    stride_seconds: int = STRIDE_SECONDS,
) -> bool:
    labels_by_clip = collect_split_labels(split)

    chunks_split_dir(fps, split, window_seconds).mkdir(parents=True, exist_ok=True)

    pending_windows: list[FeatureWindow] = []
    clips_in_buffer = 0

    broken_clips: list[dict[str, str]] = []

    seen_clips, chunk_number = look_over_saved_chunks(fps, split, window_seconds)

    jobs: list[VideoJob] = []
    for clip_id, engagement in labels_by_clip.items():
        if clip_id in seen_clips:
            continue

        job = (clip_id, engagement, split, fps, window_seconds, stride_seconds)
        jobs.append(job)

    clip_total = len(labels_by_clip)
    jobs_left = len(jobs)
    already_done = clip_total - jobs_left

    run_tag = f"[{window_seconds}s/{fps}fps/{split}]"
    print(f"{run_tag} {jobs_left}/{clip_total} clips remaining")

    if already_done > 0:
        next_number = chunk_number + 1
        print(f"  Resuming: {already_done} clips saved; next chunk {next_number:04d}")

    if jobs_left == 0:
        print(f"{split} at {fps} fps: already complete.")
        return False

    print(f"  Using {max_workers} workers")

    was_stopped = False
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for job in jobs:
            futures.append(pool.submit(handle_single_clip, job))

        done_so_far = 0
        try:
            for future in as_completed(futures):
                done_so_far += 1
                video_id, clip_windows, error = future.result()

                if error is not None or clip_windows is None:
                    print(f"  [{done_so_far}/{jobs_left}] Failed: {video_id}")

                    if error is None:
                        reason = "no windows returned"
                    else:
                        reason = error

                    broken_clips.append({"video_id": video_id, "error": reason})
                    continue

                n_windows = len(clip_windows)
                print(f"  [{done_so_far}/{jobs_left}] {video_id}: {n_windows} windows")

                pending_windows.extend(clip_windows)
                clips_in_buffer += 1

                if clips_in_buffer >= CHUNK_SIZE:
                    chunk_number += 1

                    write_chunk_to_disk(
                        pending_windows, chunk_number, split, fps, window_seconds
                    )

                    pending_windows.clear()
                    clips_in_buffer = 0

        except KeyboardInterrupt:
            was_stopped = True
            cancelled = 0
            for future in futures:
                if future.cancel():
                    cancelled += 1
            print(
                f"\nStopped: cancelled {cancelled} queued clips. "
                "Waiting for running workers, then saving."
            )

    if pending_windows:
        chunk_number += 1
        write_chunk_to_disk(pending_windows, chunk_number, split, fps, window_seconds)

    if broken_clips:
        note_broken_clips(broken_clips, fps, split, window_seconds)

    if was_stopped:
        print(f"Stopped: {split} at {fps} fps. Rerun to resume.")
    else:
        print(f"Finished: {split} at {fps} fps.")

    return was_stopped


def note_broken_clips(
    failed: list[dict[str, str]],
    fps: int,
    split: str,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
):
    csv_path = failed_extraction_csv(fps, split, window_seconds)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fail_table = pd.DataFrame(failed)

    if csv_path.is_file():
        older = pd.read_csv(csv_path)

        older_ids = older["video_id"].astype(str)

        replaced = older_ids.isin(fail_table["video_id"])

        kept = older[~replaced]
        fail_table = pd.concat([kept, fail_table], ignore_index=True)

    fail_table.to_csv(csv_path, index=False)
    print(f"Saved {len(fail_table)} failed clips to {csv_path}")


def fetch_one_label(video_id: str, split: str) -> int:
    label_table = pd.read_csv(labels_csv(split))
    clip_ids = label_table["ClipID"].astype(str)

    match = label_table.loc[clip_ids == video_id]

    if match.empty:
        raise KeyError(f"{video_id} not found in {split} labels")

    first_score = int(match["Engagement"].iloc[0])
    return checked_engagement(first_score)


def try_single_clip(
    video_id: str,
    split: str,
    fps: int,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    stride_seconds: int = STRIDE_SECONDS,
):
    engagement = fetch_one_label(video_id, split)

    print(
        f"Testing {video_id} (split={split}, fps={fps}, window={window_seconds}s, "
        f"stride={stride_seconds}s, engagement={engagement})"
    )

    clip_windows = slice_clip_into_windows(
        video_id, engagement, split, fps, window_seconds, stride_seconds
    )

    print(f"Extracted {len(clip_windows)} windows")


def choose_stride(window_seconds: int, stride_arg: int | None) -> int:
    if stride_arg is None:
        return window_seconds
    return stride_arg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    test_cmd = subparsers.add_parser("test", help="Extract one clip.")
    test_cmd.add_argument("video_id", help="Clip filename, e.g. 1100011002.avi")
    test_cmd.add_argument(
        "split",
        choices=sorted(LABELS_SPLITS.keys()),
        help="Dataset split.",
    )
    test_cmd.add_argument(
        "--fps",
        type=int,
        default=FPS_OPTIONS[0],
        choices=list(FPS_OPTIONS),
        help=f"Frame rate (default: {FPS_OPTIONS[0]}).",
    )
    test_cmd.add_argument(
        "--window-seconds",
        type=int,
        default=argparse.SUPPRESS,
        help=f"Window length in seconds (default: {DEFAULT_WINDOW_SECONDS}).",
    )
    test_cmd.add_argument(
        "--stride-seconds",
        type=int,
        default=argparse.SUPPRESS,
        help="Window step in seconds (default: window length).",
    )

    all_windows = list(WINDOW_SECONDS_OPTIONS)
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=None,
        help=f"Window length in seconds (default: all {all_windows}).",
    )
    parser.add_argument(
        "--stride-seconds",
        type=int,
        default=None,
        help="Window step in seconds (default: window length).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_max_workers(),
        help="Worker count (default: cpu_count - 2).",
    )
    args = parser.parse_args()

    window_options = resolve_variants(args.window_seconds, WINDOW_SECONDS_OPTIONS)

    for window_len in window_options:
        if window_len < 1:
            parser.error("--window-seconds must be >= 1")

    if args.stride_seconds is not None and args.stride_seconds < 1:
        parser.error("--stride-seconds must be >= 1")

    if args.command == "test":

        if args.window_seconds is None:
            test_window = DEFAULT_WINDOW_SECONDS
        else:
            test_window = args.window_seconds

        test_stride = choose_stride(test_window, args.stride_seconds)
        try_single_clip(args.video_id, args.split, args.fps, test_window, test_stride)
        sys.exit(0)

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    for window_len in window_options:

        stride_len = choose_stride(window_len, args.stride_seconds)

        for fps in FPS_OPTIONS:
            for split in LABELS_SPLITS:

                print(f"[{window_len}s/{fps}fps/{split}]")

                was_stopped = work_on_split(
                    fps, split, args.workers, window_len, stride_len
                )
                
                if was_stopped:
                    sys.exit(130)
