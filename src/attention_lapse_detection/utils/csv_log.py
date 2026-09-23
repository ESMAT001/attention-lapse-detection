import csv
from pathlib import Path


def append_row(path: Path, fields: list[str], row: dict):
    """Append one row, writing the header first if the file is new."""

    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()

    with open(path, "a", newline="") as infile:
        writer = csv.DictWriter(infile, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def read_rows(path: Path) -> list[dict]:
    """Every row already logged, or [] when the file does not exist yet."""
    if not path.is_file():
        return []
    with open(path, newline="") as infile:
        return list(csv.DictReader(infile))
