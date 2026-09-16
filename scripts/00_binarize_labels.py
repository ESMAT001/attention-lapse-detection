"""Map DAiSEE engagement levels from (0-3) to binary labels."""

import pandas as pd

from attention_lapse_detection.utils.data_paths import LABELS_SPLITS
from attention_lapse_detection.utils.data_paths import labels_dir
from attention_lapse_detection.utils.paths import PATHS


def binarize_labels_in_split(label_csv: str) -> None:
    src_file = PATHS.raw_data / "labels" / label_csv

    if not src_file.exists():
        raise FileNotFoundError(f"Raw label file not found: {src_file}")

    out_dir = labels_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / label_csv

    # Raw level 0,1 -> disengaged (0) and level 2,3 -> engaged (1)
    df = pd.read_csv(src_file,usecols=["ClipID", "Engagement"])
    df["Engagement"] = df["Engagement"].apply(lambda x: 0 if x in [0, 1] else 1)

    df.to_csv(out_file, index=False)
    print(f"Binarized labels saved to: {out_file}")

def main():
    print("Script 00: Binarize DAiSEE engagement labels")

    for split_csv in LABELS_SPLITS.values():
        binarize_labels_in_split(split_csv)

if __name__ == "__main__":
    main()
    