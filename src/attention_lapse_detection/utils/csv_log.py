import csv
from pathlib import Path


def append_row(path: Path, fields: list[str], row: dict):
    """Append a row and add a header for a new file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()

    with open(path, "a", newline="") as infile:
        writer = csv.DictWriter(infile, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def read_rows(path: Path) -> list[dict]:
    """Read saved rows, or return [] if the file is missing."""
    if not path.is_file():
        return []
    with open(path, newline="") as infile:
        return list(csv.DictReader(infile))
