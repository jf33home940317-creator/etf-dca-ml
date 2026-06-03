from datetime import datetime, timezone
import config
from data.fetcher import fetch_and_cache
from data.cleaner import drop_invalid
from features.macro import build_macro_frame
from features.builder import build_features_for_symbol, FEATURE_COLS
from models.trainer import train_symbol, save_model
from live.notifier import send_discord
from live.ledger import write_budget

BUDGET_STATE = config.STORAGE_LIVE / "budget_state.json"

def run_once():
    macro_raw = {
        "VIX": drop_invalid(fetch_and_cache("^VIX")),
        "TNX": drop_invalid(fetch_and_cache("^TNX")),
        "DXY": drop_invalid(fetch_and_cache("DX-Y.NYB")),
        "SMH": drop_invalid(fetch_and_cache("SMH")),
    }
    macro_df = build_macro_frame(macro_raw)

    lines = [f"✅ Monthly retrain {datetime.now(timezone.utc).date().isoformat()}"]
    for symbol in config.DCA_SYMBOLS:
        try:
            sym_df = drop_invalid(fetch_and_cache(symbol))
            feat = build_features_for_symbol(sym_df, macro_df)
            model, meta = train_symbol(feat, FEATURE_COLS)
            save_model(symbol, model, meta)
            auc = meta.get("cv_auc_mean")
            auc_str = f"{auc:.3f}" if auc is not None else "n/a"
            lines.append(f"  {symbol}: n={meta['n_samples']} CV-AUC={auc_str}")
        except Exception as e:
            lines.append(f"  {symbol}: ❌ {e}")

    write_budget(BUDGET_STATE, dict(config.MONTHLY_BUDGET))
    lines.append(f"💰 Budgets reset: {dict(config.MONTHLY_BUDGET)}")

    msg = "\n".join(lines)
    print(msg)
    send_discord(msg)

if __name__ == "__main__":
    run_once()
