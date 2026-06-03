import pandas as pd
from features.indicators import compute_indicators
from features.labels import compute_dca_labels
from data.cleaner import align_macro
import config

FEATURE_COLS = [
    "close_ma20", "close_ma60", "ma20_ma60",
    "rsi14", "macd_hist", "roc5", "roc20",
    "atr_norm", "bb_width20", "rvol20",
    "channel_pos60", "dd_60", "dd_252",
    "vol_norm", "obv_slope",
    "day_of_week",
    "vix", "vix_ma200_ratio", "vix_chg5",
    "tnx", "tnx_slope",
    "dxy", "dxy_chg5", "smh_roc1",
]


def build_features_for_symbol(
    sym_df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    tech = compute_indicators(sym_df)
    merged = align_macro(tech, macro_df)
    return compute_dca_labels(
        merged, window=config.LABEL_WINDOW, top_q=config.LABEL_TOP_Q,
    )
