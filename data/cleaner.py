import pandas as pd

def drop_invalid(df: pd.DataFrame) -> pd.DataFrame:
    # Indices like ^VIX, ^TNX, DX-Y.NYB report zero (or sporadic) volume.
    # Only apply the volume filter for instruments that actually report
    # volume on the majority of bars, otherwise the whole frame would be wiped.
    if len(df) > 0 and (df["volume"] > 0).mean() >= 0.5:
        df = df[df["volume"] > 0].copy()
    else:
        df = df.copy()
    # Drop bars with non-positive prices (corrupt yfinance rows, e.g. 0050.TW
    # has a handful of zero-open bars that would divide-by-zero downstream).
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df = df[df[col] > 0]
    daily_ret = df["close"].pct_change().abs()
    df["_extreme_move"] = daily_ret > 0.5
    return df.drop(columns=["_extreme_move"], errors="ignore")

def is_last_trading_day_of_month(df: pd.DataFrame, day: pd.Timestamp) -> bool:
    day = pd.Timestamp(day).normalize()
    if day not in df.index.normalize():
        return False
    same_month = df.index[(df.index.year == day.year) & (df.index.month == day.month)]
    if len(same_month) == 0:
        return False
    return same_month.max().normalize() == day

def is_first_trading_day_of_month(df: pd.DataFrame, day: pd.Timestamp) -> bool:
    day = pd.Timestamp(day).normalize()
    if day not in df.index.normalize():
        return False
    same_month = df.index[(df.index.year == day.year) & (df.index.month == day.month)]
    return same_month.min().normalize() == day

def align_macro(symbol_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    s = symbol_df.sort_index().copy()
    m = macro_df.sort_index().copy()
    s["_date"] = s.index
    m["_date"] = m.index
    merged = pd.merge_asof(
        s.reset_index(drop=True), m.reset_index(drop=True),
        on="_date", direction="backward",
    )
    merged.index = pd.DatetimeIndex(merged["_date"], name="date")
    return merged.drop(columns=["_date"])
