from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class ProjectPaths:
    "Shared paths used across this project."

    root = ROOT
    data = ROOT / "Data"
    raw_data = ROOT / "data" / "raw"
    processed_data = ROOT / "data" / "processed"
    processed_clean_data = ROOT / "data" / "processed" / "clean"
    models = ROOT / "models"
    face_landmarker = ROOT / "models" / "face_landmarker.task"

PATHS = ProjectPaths()
