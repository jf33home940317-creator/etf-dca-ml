from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

import config
from data.fetcher import fetch_and_cache
from data.cleaner import drop_invalid
from live.ledger import read_signals, rewrite_signals

PREDICTIONS = config.STORAGE_LIVE / "predictions.jsonl"


def resolve_fill(signal_date: pd.Timestamp, prices: pd.DataFrame) -> float | None:
    signal_date = pd.Timestamp(signal_date).normalize()
    idx = prices.index.normalize()
    if signal_date in idx:
        return float(prices.loc[prices.index[idx == signal_date][0], "open"])
    return None


def run_once():
    signals = read_signals(PREDICTIONS)
    if not signals:
        print("[ledger_settle] no signals to settle")
        return

    pending = [s for s in signals if s.get("status") == "PENDING"]
    symbols_needed = {s["symbol"] for s in pending}
    price_cache = {}
    for sym in symbols_needed:
        df = fetch_and_cache(sym)
        price_cache[sym] = drop_invalid(df)

    changed = 0
    for s in signals:
        if s.get("status") != "PENDING":
            continue
        prices = price_cache.get(s["symbol"])
        if prices is None:
            continue
        fill = resolve_fill(pd.Timestamp(s["signal_date"]), prices)
        if fill is not None:
            s["fill_price"] = fill
            s["status"] = "FILLED"
            s["settled_at"] = datetime.now(timezone.utc).isoformat()
            changed += 1

    if changed:
        rewrite_signals(PREDICTIONS, signals)
    print(f"[ledger_settle] settled {changed} signals")


if __name__ == "__main__":
    run_once()
