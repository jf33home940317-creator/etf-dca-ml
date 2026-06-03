"""Compute live paper-trading performance from the ledger.

Reads storage/live/predictions.jsonl (FILLED signals only) and computes
per-symbol: total invested, shares owned, current market value, unrealized
PnL, and annualized return. Also prints an overall aggregate.

Usage:
    python -m live.performance_report             # uses today's close
    python -m live.performance_report --discord   # also push to Discord
"""
import argparse
from datetime import datetime, timezone
from collections import defaultdict

import config
from data.fetcher import fetch_and_cache
from data.cleaner import drop_invalid
from live.ledger import read_signals
from live.notifier import send_discord

PREDICTIONS = config.STORAGE_LIVE / "predictions.jsonl"


def _latest_close(symbol: str) -> float | None:
    try:
        df = drop_invalid(fetch_and_cache(symbol))
        return float(df["close"].iloc[-1])
    except Exception as e:
        print(f"[performance_report] cannot fetch latest close for {symbol}: {e}")
        return None


def run_once(push_discord: bool = False):
    signals = read_signals(PREDICTIONS)
    filled = [s for s in signals if s.get("status") == "FILLED" and s.get("fill_price")]

    if not filled:
        msg = "📊 績效報告: 尚無已成交訊號"
        print(msg)
        if push_discord:
            send_discord(msg)
        return

    # Aggregate per symbol
    invested = defaultdict(float)
    shares = defaultdict(float)
    first_date = defaultdict(lambda: None)
    for s in filled:
        sym = s["symbol"]
        amt = float(s["amount"])
        fp = float(s["fill_price"])
        invested[sym] += amt
        shares[sym] += amt / fp
        sd = s.get("signal_date")
        if sd and (first_date[sym] is None or sd < first_date[sym]):
            first_date[sym] = sd

    today = datetime.now(timezone.utc).date()
    lines = [f"📊 績效報告 {today.isoformat()}"]
    total_invested = 0.0
    total_mv = 0.0
    for sym in config.DCA_SYMBOLS:
        if invested[sym] == 0:
            lines.append(f"  {sym}: 尚未有成交")
            continue
        latest = _latest_close(sym)
        if latest is None:
            lines.append(f"  {sym}: 無法取得最新收盤價")
            continue
        mv = shares[sym] * latest
        pnl = mv - invested[sym]
        pnl_pct = pnl / invested[sym]
        # Annualize using days since first fill for this symbol
        if first_date[sym]:
            days = max((today - datetime.fromisoformat(first_date[sym]).date()).days, 1)
            ann = (1 + pnl_pct) ** (365.25 / days) - 1
            ann_str = f"年化={ann*100:+.1f}%"
        else:
            ann_str = "年化=n/a"
        lines.append(
            f"  {sym}: 已投入=${invested[sym]:.0f} 持有={shares[sym]:.4f}股 "
            f"市值=${mv:.0f} 損益=${pnl:+.0f} ({pnl_pct*100:+.1f}%) {ann_str}"
        )
        total_invested += invested[sym]
        total_mv += mv

    if total_invested > 0:
        total_pnl = total_mv - total_invested
        total_pct = total_pnl / total_invested
        lines.append(
            f"💰 總計: 投入=${total_invested:.0f} 市值=${total_mv:.0f} "
            f"損益=${total_pnl:+.0f} ({total_pct*100:+.1f}%)"
        )
        lines.append(f"📝 註: 跨幣別合計（USD + TWD 直接相加），僅供參考")

    msg = "\n".join(lines)
    print(msg)
    if push_discord:
        send_discord(msg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--discord", action="store_true", help="also push to Discord")
    args = p.parse_args()
    run_once(push_discord=args.discord)
