import numpy as np
import pandas as pd
from models.trainer import train_symbol


def test_train_symbol_returns_model_and_meta():
    idx = pd.bdate_range("2020-01-01", periods=600, name="date")
    rng = np.random.default_rng(0)
    feat = pd.DataFrame(rng.normal(0, 1, (600, 5)),
                        columns=["a", "b", "c", "d", "e"], index=idx)
    # Plant a weak signal so XGBoost has something to learn
    feat["label"] = ((feat["a"] + rng.normal(0, 0.5, 600)) > 0).astype(float)
    model, meta = train_symbol(feat, feature_cols=["a", "b", "c", "d", "e"])
    assert hasattr(model, "predict_proba")
    assert "cv_auc_mean" in meta
    assert "n_samples" in meta
    assert meta["n_samples"] == 600
