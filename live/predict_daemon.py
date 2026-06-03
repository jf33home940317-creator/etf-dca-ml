import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
from pandas.tseries.offsets import BDay

import config
from data.fetcher import fetch_and_cache
from data.cleaner import drop_invalid
from features.macro import build_macro_frame
from features.builder import build_features_for_symbol, FEATURE_COLS
from models.trainer import load_model
from live.notifier import send_discord
from live.ledger import (
    append_signal, read_budget, update_budget, write_budget,
)

PREDICTIONS = config.STORAGE_LIVE / "predictions.jsonl"
BUDGET_STATE = config.STORAGE_LIVE / "budget_state.json"
HEARTBEAT = config.STORAGE_LIVE / "heartbeat.json"


def decide_action(
    *, symbol: str, prob: float, threshold: float,
    budget_left: float, monthly_budget: float,
    max_per_trade_ratio: float, is_last_day: bool,
) -> dict | None:
    if prob >= threshold and budget_left > 0:
        amount = min(monthly_budget * max_per_trade_ratio, budget_left)
        return {"reason": "BUY", "amount": amount,
                "new_budget": budget_left - amount}
    if is_last_day and budget_left > 0:
        return {"reason": "FORCED_BUY", "amount": budget_left,
                "new_budget": 0.0}
    return None


def _fetch_with_retry(symbol: str, retries: int = 3, delay: int = 600):
    for attempt in range(retries):
        try:
            df = fetch_and_cache(symbol)
            df = drop_invalid(df)
            if len(df) < 10:
                raise ValueError(f"Too few rows for {symbol}: {len(df)}")
            return df
        except Exception as e:
            print(f"[{symbol}] fetch attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    raise RuntimeError(f"Failed to fetch {symbol} after {retries} retries")


def _emoji(reason: str | None) -> str:
    return {"BUY": "🟢", "FORCED_BUY": "🔴"}.get(reason, "⚪")


def _reason_zh(reason: str | None) -> str:
    return {"BUY": "進場", "FORCED_BUY": "月底強制"}.get(reason, reason or "")


def _is_calendar_month_last_business_day(today: pd.Timestamp) -> bool:
    """Live-mode month-end check.

    Spec §2.2 says "FORCED_BUY on the last trading day of the month for this
    symbol". In backtest the historical df spans multiple months so
    data.cleaner.is_last_trading_day_of_month works (it compares against the
    max same-month bar). In live, the df is truncated at today, so that
    function would return True every day and drain the budget on every run.

    The pragmatic live-mode check: today is Mon-Fri AND the next business day
    is in the next calendar month. Misses the rare case where the actual last
    trading day is a Thursday because Friday is a market holiday — that month
    just won't see a forced buy, and the 1st-of-month retrain resets budget
    anyway, so the worst-case cost is one missed forced buy per such month.
    """
    today = pd.Timestamp(today).normalize()
    if today.dayofweek >= 5:
        return False
    return (today + BDay(1)).month != today.month


def run_once():
    today_utc = datetime.now(timezone.utc).date()
    print(f"[predict_daemon] starting {today_utc.isoformat()}")

    macro_raw = {
        "VIX": _fetch_with_retry("^VIX"),
        "TNX": _fetch_with_retry("^TNX"),
        "DXY": _fetch_with_retry("DX-Y.NYB"),
        "SMH": _fetch_with_retry("SMH"),
    }
    macro_df = build_macro_frame(macro_raw)

    lines = [f"📅 {today_utc.isoformat()} ETF 定投訊號"]
    for symbol in config.DCA_SYMBOLS:
        try:
            sym_df = _fetch_with_retry(symbol)
            feat = build_features_for_symbol(sym_df, macro_df)
            latest = feat.dropna(subset=FEATURE_COLS).iloc[[-1]]
            if len(latest) == 0:
                lines.append(f"⚠️ {symbol} 無特徵資料")
                continue

            model, _ = load_model(symbol)
            prob = float(model.predict_proba(latest[FEATURE_COLS])[0, 1])

            budget_left = read_budget(BUDGET_STATE, symbol)
            if budget_left is None:
                budget_left = config.MONTHLY_BUDGET[symbol]
                update_budget(BUDGET_STATE, symbol, budget_left)

            today_idx = latest.index[-1]
            # Use calendar-based check for live (the sym_df-based check in
            # data.cleaner is correct for backtest but degenerates to "every
            # day is the last bar of its month" in live mode).
            is_last = _is_calendar_month_last_business_day(
                pd.Timestamp(datetime.now(timezone.utc).date())
            )
            action = decide_action(
                symbol=symbol, prob=prob,
                threshold=config.SIGNAL_THRESHOLD[symbol],
                budget_left=budget_left,
                monthly_budget=config.MONTHLY_BUDGET[symbol],
                max_per_trade_ratio=config.MAX_PER_TRADE_RATIO,
                is_last_day=is_last,
            )

            if action:
                append_signal(PREDICTIONS, {
                    "issued_at": datetime.now(timezone.utc).isoformat(),
                    "signal_date": str(today_idx.date()),
                    "symbol": symbol, "reason": action["reason"],
                    "prob": prob, "amount": action["amount"],
                    "fill_price": None, "status": "PENDING",
                })
                update_budget(BUDGET_STATE, symbol, action["new_budget"])
                lines.append(
                    f"{_emoji(action['reason'])} {symbol:<8} 機率={prob:.2f} "
                    f"{_reason_zh(action['reason'])} ${action['amount']:.0f} "
                    f"(剩餘 ${action['new_budget']:.0f})"
                )
            else:
                lines.append(
                    f"⚪ {symbol:<8} 機率={prob:.2f} 觀望中 "
                    f"(剩餘 ${budget_left:.0f})"
                )
        except Exception as e:
            lines.append(f"❌ {symbol} 錯誤: {e}")

    msg = "\n".join(lines)
    print(msg)
    send_discord(msg)

    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT.write_text(datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    run_once()
