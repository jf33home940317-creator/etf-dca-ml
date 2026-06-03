# etf-dca-ml

Smart DCA bot: per-symbol XGBoost predicts top-20%-of-month entry days by
forward Calmar. Signals on SPY/QQQ/0050.TW. Discord notification + paper
ledger. Manual order execution.

## Quickstart

```bash
pip install -r requirements.txt
export DCA_DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
python run_backtest.py            # train + backtest end-to-end
python -m live.monthly_retrain    # fresh models
python -m live.predict_daemon     # daily signal
```

## Schedule (Asia/Taipei)

| Time | Job |
|---|---|
| Day-1 06:00 | Monthly retrain |
| Daily 06:30 | Predict + Discord |
| Daily 07:00 | Settle prior signals |
| Sunday 23:00 | Weekly report |

## Layout

- `data/` — yfinance fetch + clean
- `features/` — indicators, macro, labels, builder
- `models/` — trainer + walk-forward CV with purge gap
- `backtest/` — engine + baselines + metrics
- `live/` — daemon, settle, retrain, notifier, ledger
- `deploy/` — Oracle VM install + crontab
- `storage/` — runtime artifacts (gitignored)

## Spec

See `docs/specs/design.md` for full design and §12 for post-Phase-B
diagnostic notes (purge gap, day_of_month removal, Y-randomization).

## Acceptance results

Strategy beat both classic-DCA and weekly-DCA baselines on Sharpe across
all 3 symbols in walk-forward backtest. Honest CV-AUC 0.66 (Y-randomization
gap +0.16 over shuffled null).

| Symbol | Strategy Sharpe | Classic | Weekly |
|---|---:|---:|---:|
| SPY | 1.38 | 1.12 | 1.31 |
| QQQ | 1.45 | 1.16 | 1.35 |
| 0050.TW | 1.51 | 1.10 | 1.26 |

Cost basis vs monthly mean: 0050.TW −3.0%, SPY/QQQ slightly positive at
τ=0.55 (future tuning: threshold grid search).
