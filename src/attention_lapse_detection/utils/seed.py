import random

import numpy as np
import torch

from attention_lapse_detection.utils.constants import SEED


def set_seed(seed: int = SEED) -> torch.Generator:
    """Seed Python, NumPy and PyTorch, return a Generator for DataLoaders."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
