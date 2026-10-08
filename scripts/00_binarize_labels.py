"""Convert official DAiSEE split labels: 0–1 disengaged, 2–3 engaged."""

import pandas as pd

from attention_lapse_detection.utils.data_paths import labels_dir
from attention_lapse_detection.utils.paths import PATHS
from attention_lapse_detection.utils.constants import LABELS_SPLITS


def binarize_split(label_csv: str) -> None:
    src_file = PATHS.raw_data / "Labels" / label_csv

    if not src_file.is_file():
        raise FileNotFoundError(f"Label file not found: {src_file}")

    out_dir = labels_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / label_csv

    tbl = pd.read_csv(src_file, usecols=["ClipID", "Engagement"])
    eng = tbl["Engagement"].astype(int)
    tbl["Engagement"] = (eng >= 2).astype(int)

    tbl.to_csv(dst, index=False)
    print(f"  {label_csv}: {len(tbl)} rows -> {dst}")


if __name__ == "__main__":
    print("Converting DAiSEE labels.")
    for csv_name in LABELS_SPLITS.values():
        binarize_split(csv_name)
