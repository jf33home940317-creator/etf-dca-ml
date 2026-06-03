import numpy as np
import pandas as pd

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _macd_hist(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["close"]

    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    out["close_ma20"] = c / ma20
    out["close_ma60"] = c / ma60
    out["ma20_ma60"] = ma20 / ma60

    out["rsi14"] = _rsi(c, 14)
    out["macd_hist"] = _macd_hist(c)
    out["roc5"] = c.pct_change(5)
    out["roc20"] = c.pct_change(20)

    atr = _atr(out, 14)
    out["atr_norm"] = atr / c
    std20 = c.rolling(20).std()
    out["bb_width20"] = (4 * std20) / c.rolling(20).mean()
    out["rvol20"] = c.pct_change().rolling(20).std() * np.sqrt(252)

    high60 = out["high"].rolling(60).max()
    low60 = out["low"].rolling(60).min()
    out["channel_pos60"] = (c - low60) / (high60 - low60).replace(0, np.nan)

    out["dd_60"] = c / high60 - 1
    out["dd_252"] = c / out["high"].rolling(252).max() - 1

    vol_ma = out["volume"].rolling(20).mean()
    out["vol_norm"] = out["volume"] / vol_ma
    obv = _obv(out)
    out["obv_slope"] = obv.diff(20) / 20

    out["day_of_month"] = out.index.day
    out["day_of_week"] = out.index.dayofweek

    return out
