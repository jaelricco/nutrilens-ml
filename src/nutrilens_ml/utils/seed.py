"""Deterministic-seed helper.

Enabling `cudnn.deterministic = True` is a deliberate trade — it slows training
~5–15% on conv-heavy models but makes training reruns reproducible, which we
want for release-bar comparisons. Flip this off locally if you're prototyping.
"""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
