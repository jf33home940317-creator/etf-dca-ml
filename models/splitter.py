import numpy as np
import pandas as pd
from typing import Iterator


def purged_walk_forward_splits(
    idx: pd.DatetimeIndex,
    n_splits: int = 5,
    purge_gap: int = 20,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    n = len(idx)
    val_size = n // (n_splits + 1)
    for k in range(n_splits):
        val_start = (k + 1) * val_size
        val_end = val_start + val_size
        if val_end > n:
            break
        train_end = max(0, val_start - purge_gap)
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        if len(train_idx) == 0:
            continue
        yield train_idx, val_idx
