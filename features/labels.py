import numpy as np
import pandas as pd


def compute_dca_labels(
    df: pd.DataFrame,
    window: int = 30,
    top_q: float = 0.20,
) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    low = out["low"]

    forward_close = close.shift(-window)
    forward_return = forward_close / close - 1

    rolling_future_low = (
        low.shift(-window).rolling(window, min_periods=1).min()
    )
    forward_mdd = (rolling_future_low - close) / close
    forward_mdd = forward_mdd.clip(upper=0)

    perf = forward_return / (np.abs(forward_mdd) + 1e-5)

    ym = out.index.to_period("M")
    perf_with_ym = pd.DataFrame({"perf": perf, "ym": ym})
    rank_pct = perf_with_ym.groupby("ym")["perf"].rank(pct=True)

    valid = forward_return.notna() & forward_mdd.notna()
    label = (rank_pct >= (1.0 - top_q)).astype(float)
    label[~valid] = np.nan

    out["forward_return"] = forward_return
    out["forward_mdd"] = forward_mdd
    out["perf"] = perf
    out["label"] = label
    return out
