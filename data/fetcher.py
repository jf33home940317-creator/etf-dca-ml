import yfinance as yf
import pandas as pd
from pathlib import Path
import config

def fetch_symbol(symbol: str, start: str = config.HISTORY_START) -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start,
        auto_adjust=True,
        progress=False,
        actions=False,
    )
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })[["open", "high", "low", "close", "volume"]]
    df.index.name = "date"
    return df

def fetch_and_cache(symbol: str) -> pd.DataFrame:
    config.STORAGE_RAW.mkdir(parents=True, exist_ok=True)
    path = config.STORAGE_RAW / f"{symbol.replace('^', '').replace('-', '_')}.parquet"
    df = fetch_symbol(symbol)
    df.to_parquet(path)
    return df

def load_or_fetch(symbol: str) -> pd.DataFrame:
    path = config.STORAGE_RAW / f"{symbol.replace('^', '').replace('-', '_')}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return fetch_and_cache(symbol)
