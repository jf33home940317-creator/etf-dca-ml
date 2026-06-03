from datetime import datetime, timezone, timedelta
from collections import defaultdict
import config
from live.notifier import send_discord
from live.ledger import read_signals, read_budget

PREDICTIONS = config.STORAGE_LIVE / "predictions.jsonl"
BUDGET_STATE = config.STORAGE_LIVE / "budget_state.json"

def run_once():
    signals = read_signals(PREDICTIONS)
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    recent = [s for s in signals if datetime.fromisoformat(s["issued_at"]) > cutoff]

    counts = defaultdict(lambda: defaultdict(int))
    amounts = defaultdict(float)
    for s in recent:
        counts[s["symbol"]][s["reason"]] += 1
        amounts[s["symbol"]] += float(s["amount"])

    lines = ["📊 每週定投報告"]
    for sym in config.DCA_SYMBOLS:
        bl = read_budget(BUDGET_STATE, sym)
        bl_str = f"${bl:.0f}" if bl is not None else "n/a"
        c = counts[sym]
        lines.append(
            f"  {sym}: 進場={c.get('BUY', 0)} 月底強制={c.get('FORCED_BUY', 0)} "
            f"已花費=${amounts[sym]:.0f} 剩餘={bl_str}"
        )
    msg = "\n".join(lines)
    print(msg)
    send_discord(msg)

if __name__ == "__main__":
    run_once()
