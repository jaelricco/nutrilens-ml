import os
import random

from nutrilens_ml.utils.seed import set_global_seed


def test_set_global_seed_is_deterministic() -> None:
    set_global_seed(42)
    a = [random.random() for _ in range(4)]
    set_global_seed(42)
    b = [random.random() for _ in range(4)]
    assert a == b
    assert os.environ["PYTHONHASHSEED"] == "42"
