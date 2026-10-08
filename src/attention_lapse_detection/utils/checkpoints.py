from pathlib import Path
from attention_lapse_detection.utils.paths import PATHS


def resolve_checkpoint(name: str) -> Path:
    """models/NAME.pt"""
    path = Path(name)
    if path.suffix == ".pt" or path.is_file():
        return path
    return PATHS.models / f"{name}.pt"
