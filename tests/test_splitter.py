import pandas as pd
from models.splitter import purged_walk_forward_splits


def test_purge_gap_isolates_train_and_val():
    idx = pd.bdate_range("2020-01-01", periods=500, name="date")
    splits = list(purged_walk_forward_splits(idx, n_splits=3, purge_gap=20))
    assert len(splits) == 3
    for train_idx, val_idx in splits:
        # train ends well before val starts (purge gap)
        gap = (idx[val_idx[0]] - idx[train_idx[-1]]).days
        assert gap >= 20  # at least 20 calendar days separation
        # train strictly precedes val
        assert idx[train_idx[-1]] < idx[val_idx[0]]
