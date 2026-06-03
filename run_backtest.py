"""Train models on full history and run walk-forward backtest vs baselines."""
import argparse
import json
import traceback

import config
from data.fetcher import fetch_and_cache
from data.cleaner import drop_invalid
from features.macro import build_macro_frame
from features.builder import build_features_for_symbol, FEATURE_COLS
from models.trainer import train_symbol, save_model, load_model
from backtest.engine import run_backtest
from backtest.metrics import (
    equity_curve_from_trades, classic_dca_baseline,
    weekly_dca_baseline, summary_stats, cost_basis_vs_monthly_mean,
)


def main(retrain: bool = True):
    macro_raw = {
        "VIX": drop_invalid(fetch_and_cache("^VIX")),
        "TNX": drop_invalid(fetch_and_cache("^TNX")),
        "DXY": drop_invalid(fetch_and_cache("DX-Y.NYB")),
        "SMH": drop_invalid(fetch_and_cache("SMH")),
    }
    macro_df = build_macro_frame(macro_raw)

    summary = {}
    for symbol in config.DCA_SYMBOLS:
        print(f"\n=== {symbol} ===")
        try:
            sym_df = drop_invalid(fetch_and_cache(symbol))
            feat = build_features_for_symbol(sym_df, macro_df)

            if retrain:
                model, meta = train_symbol(feat, FEATURE_COLS)
                save_model(symbol, model, meta)
                print(f"  trained: n={meta['n_samples']} CV-AUC={meta.get('cv_auc_mean')}")
            else:
                model, _ = load_model(symbol)

            trades = run_backtest(
                symbol=symbol, prices=sym_df, features=feat, model=model,
                feature_cols=FEATURE_COLS,
                monthly_budget=config.MONTHLY_BUDGET[symbol],
                threshold=config.SIGNAL_THRESHOLD[symbol],
                max_per_trade_ratio=config.MAX_PER_TRADE_RATIO,
            )
            classic = classic_dca_baseline(symbol, sym_df, config.MONTHLY_BUDGET[symbol])
            weekly = weekly_dca_baseline(symbol, sym_df, config.MONTHLY_BUDGET[symbol])

            strat_eq = equity_curve_from_trades(trades, sym_df)
            classic_eq = equity_curve_from_trades(classic, sym_df)
            weekly_eq = equity_curve_from_trades(weekly, sym_df)

            summary[symbol] = {
                "strategy": summary_stats(strat_eq),
                "classic_dca": summary_stats(classic_eq),
                "weekly_dca": summary_stats(weekly_eq),
                "strategy_cost_basis_vs_mean": cost_basis_vs_monthly_mean(trades, sym_df),
                "n_buys": sum(1 for t in trades if t["reason"] == "BUY"),
                "n_forced": sum(1 for t in trades if t["reason"] == "FORCED_BUY"),
            }
            print(json.dumps(summary[symbol], indent=2, default=str))
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            summary[symbol] = {"error": str(e)}

    out_path = config.STORAGE / "backtest_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-retrain", action="store_true")
    args = p.parse_args()
    main(retrain=not args.no_retrain)
