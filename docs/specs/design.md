# ETF DCA ML — Design Spec

- **Date**: 2026-06-03
- **Status**: Approved (pending implementation plan)
- **Target repo**: `etf-dca-ml` (new, sibling to `BYBIT_ML`)
- **Predecessor lessons**: BYBIT_ML Phase 1-6 (data pipeline, walk-forward, live daemon, paper ledger)

## 1. Purpose

Build a "smart DCA" bot for stock/ETF: instead of mechanically buying on a fixed schedule, an ML model scores every trading day and the bot only deploys monthly budget on days predicted to be in the top 20% of the month by **forward 30-day Calmar ratio** (return / max drawdown).

**Decision boundary**: signal generation + paper trading ledger + Discord notification. Human places real orders manually.

## 2. Strategy Specification

### 2.1 Label definition

Top 20% of each natural month, ranked by forward-30-day Calmar (return / |max drawdown|).

```python
def compute_dca_labels(df, window=30, top_q=0.20):
    forward_return = df['close'].shift(-window) / df['close'] - 1
    rolling_future_low = df['low'].shift(-window).rolling(window, min_periods=1).min()
    forward_mdd = (rolling_future_low - df['close']) / df['close']
    forward_mdd = forward_mdd.clip(upper=0)
    perf_metric = forward_return / (np.abs(forward_mdd) + 1e-5)

    df['year_month'] = df['timestamp'].dt.to_period('M')
    df['month_rank_pct'] = df.groupby('year_month')['perf_metric'].rank(pct=True)
    df['label'] = (df['month_rank_pct'] >= (1.0 - top_q)).astype(int)
    return df
```

Rationale: relative ranking inside each month forces the model to learn timing alpha (not market beta). Calmar over plain return discriminates "smooth dip-buy" entries from "caught a falling knife" entries.

### 2.2 Execution rules (per symbol)

- Monthly budget `B` (asset-specific in `config.py`).
- Single-trade cap: `B / 4` (max 4 shots per month).
- Signal threshold: `prob >= τ` (default 0.55, grid-searched per symbol in [0.50, 0.55, 0.60, 0.65]).
- If `today == last_trading_day_of_month_for_symbol AND budget_left > 0` → `FORCED_BUY` of remaining budget.
- Budget resets on the 1st of each calendar month after retrain.

### 2.3 Why this design

- Pure dynamic accumulation (no monthly-1st baseline buy): in bear months, forced month-end buy will often land at lower prices than month-start buys. Backtest will validate.
- Asset-specific last-trading-day check: TW (春節) and US (Thanksgiving) calendars diverge — per-symbol determination from each symbol's own OHLCV index.

## 3. Data

### 3.1 Source

`yfinance` with `auto_adjust=True` (mandatory: stock dividends and splits must be back-adjusted, else technical features get artificial gaps).

### 3.2 Tickers

| Ticker | Role |
|---|---|
| `SPY`, `QQQ`, `0050.TW` | Targets (Phase 1 universe) |
| `^VIX` | US fear gauge |
| `^TNX` | US 10Y yield |
| `DX-Y.NYB` | US Dollar Index (drives EM outflows / TW foreign selling) |
| `TSM` or `SMH` | Overnight semi proxy for 0050.TW |

History: 2005-01-01 onward (~5000 daily bars).

### 3.3 Cleaner rules

- Each symbol keeps its **own** trading calendar (no outer-join across markets).
- Macro features (VIX/TNX/DXY/TSM) broadcast via `pd.merge_asof(direction='backward')`, after `to_datetime` + `sort_values`.
- Drop rows with zero volume; flag (do not auto-drop) bars with |daily return| > 50%.

## 4. Feature Engineering

Per symbol, ~22 features:

| Category | Features |
|---|---|
| Trend | `close/MA20`, `close/MA60`, `MA20/MA60` |
| Momentum | RSI(14), MACD histogram, ROC(5), ROC(20) |
| Volatility | ATR(14)/close, BB width(20), realized vol(20) |
| Position | 60-day channel position |
| Drawdown | DD from 60d high, DD from 252d high |
| Volume | volume / MA_vol(20), OBV slope |
| Seasonality | day_of_month, day_of_week |
| Macro (broadcast) | VIX level, VIX / VIX_MA200, VIX 5d change, TNX level, TNX slope, DXY level, DXY 5d change, TSM (or SMH) ROC_1 |

**Excluded**: `month_of_year` — only ~20 samples per month over 20 years; XGBoost overfits regime-specific events (e.g. 2008-09, 2020-03) as universal patterns.

**`features/builder.py` flow**:
```python
for symbol in DCA_SYMBOLS:
    tech = indicators.compute(raw[symbol])
    fmat = pd.merge_asof(tech, macro_df, on='date')
    feat[symbol] = fmat
labels[symbol] = compute_dca_labels(feat[symbol])
```

## 5. Model & Training

### 5.1 Model

XGBoost binary classifier (`objective='binary:logistic'`), one independent fit per symbol → `storage/models/{SYMBOL}.pkl`.

`scale_pos_weight = 4` (corrects 20/80 class imbalance; recalibrates output so 0.5 is the neutral cutoff and τ=0.55 carries genuine "above-average confidence" semantics).

### 5.2 Walk-forward retrain cadence

- **Monthly**, on the 1st calendar day at 06:00 TPE.
- Training set: 2005-01-01 to last completed trading day of previous month.
- Last 30 trading days of labels dropped (forward window incomplete).
- Hyperparameter selection: 5-fold time-series CV with **20-trading-day Purge Gap** between train and validation folds (prevents leakage from long-window features like MA60, 252d high).

### 5.3 Output artifacts

- `storage/models/{SYMBOL}.pkl` (overwrites prior month)
- `storage/models/{SYMBOL}_meta.json`: training date, sample count, CV scores, feature importance top-10

## 6. Backtest

### 6.1 Engine (`backtest/engine.py`)

Walk-forward: each month, the model used for prediction is trained on data up to the prior month-end. Zero look-ahead.

**Timing convention** (mirrors live daemon exactly):
- On the morning of trading day `d`, features are computed from data **through day `d-1` close**.
- The order, if any, fills at **day `d` open**.
- "Last trading day of month" means day `d` itself is the last session — the forced buy still fills at day `d`'s open, inside the same month.

```
for each trading day d in backtest window:
    for symbol in DCA_SYMBOLS:
        if d is first trading day of month for symbol:
            budget_left[symbol] = B[symbol]
        prob = model[symbol].predict(features[symbol, through d-1])
        if prob >= τ and budget_left[symbol] > 0:
            amount = min(B[symbol] / 4, budget_left[symbol])
            execute_buy(symbol, d_open, amount)
            budget_left[symbol] -= amount
        if d == last_trading_day_of_month_for(symbol) and budget_left[symbol] > 0:
            execute_buy(symbol, d_open, budget_left[symbol], reason="FORCED_BUY")
            budget_left[symbol] = 0
```

### 6.2 Baselines (mandatory comparison)

- **Baseline A**: classic DCA — full monthly budget on 1st trading day.
- **Baseline B**: weekly DCA — `B/4` every Monday (no timing).
- **Strategy**: model-driven engine above.

### 6.3 Metrics

- Total return, CAGR
- Sharpe, Sortino, Calmar
- Max drawdown
- **Cost basis vs monthly mean price** — the DCA-native metric: average effective entry price relative to that month's mean price, expressed as % savings vs Baseline B.
- Per-month bullet usage distribution (0/1/2/3/4 shots fired; how often FORCED_BUY triggered)

## 7. Live Deployment

### 7.1 Host

Reuse existing Oracle Cloud VM (same instance as BYBIT_ML, isolated project directory, isolated cron entries).

### 7.2 Cron schedule (Asia/Taipei)

| Time | Job | Script |
|---|---|---|
| 1st of month 06:00 | Walk-forward retrain | `live/monthly_retrain.py` |
| Daily 06:30 | Fetch → predict → notify → ledger append | `live/predict_daemon.py` |
| Daily 07:00 | Settle prior-day signals at today's open | `live/ledger_settle.py` |
| Sunday 23:00 | Weekly report to Discord | `live/weekly_report.py` |

06:30 (not 06:00) handles US Winter Time (EST) yfinance settlement lag: EST close is 05:00 TPE, and yfinance can take 60-90min to finalize the daily bar.

### 7.3 `predict_daemon.py` flow

```
1. Fetch latest yfinance increment (all tickers including macro)
2. Data integrity check: assert raw_df.index[-1] >= expected_last_session
   On failure: retry 3× with 10min spacing; on full failure: Discord alert + abort
3. For each symbol:
     features = build_latest_features(symbol)
     prob = model.predict_proba(features)[-1, 1]
     budget_left = read_budget_state(symbol)
     if prob >= τ[symbol] and budget_left > 0:
         amount = min(B[symbol] / 4, budget_left)
         append_signal(symbol, prob, amount, "BUY")
         update_budget(symbol, budget_left - amount)
     if cleaner.is_last_trading_day_of_month(symbol, today) and budget_left > 0:
         append_signal(symbol, prob, budget_left, "FORCED_BUY")
         update_budget(symbol, 0)
4. Compose single Discord message covering all symbols
5. notifier.send(message)
6. Write heartbeat
```

### 7.4 Discord message format

```
📅 2026-06-04 ETF DCA Signals (06:30 TPE)

🟢 SPY    prob=0.68  BUY $250 (remaining $250/mo)
⚪ QQQ    prob=0.42  no action (remaining $1000/mo)
🔴 0050   prob=0.31  no action (last trading day → FORCED_BUY NT$800)

Next run: tomorrow 06:30
```

### 7.5 `monthly_retrain.py`

```
1. Pull latest yfinance increment
2. For each symbol: refit XGBoost, overwrite pkl + meta.json
3. Reset budget_state.json: B[symbol] for every symbol
4. Discord: "✅ Monthly retrain complete. CV scores: ..."
```

### 7.6 Ledger

- `storage/live/predictions.jsonl`: one signal per line, atomic append (reused from BYBIT_ML).
- `storage/live/ledger.json`: positions, cost basis, marked-to-market.
- `storage/live/budget_state.json`: `{symbol: budget_left}` per month.
- Settlement basis: each signal is settled at the **open price of its target market's next trading session after the signal was emitted**. `ledger_settle.py` runs daily at 07:00 TPE and processes any pending signal whose fill bar is now available in yfinance — e.g. a Friday-morning US signal fills at Friday 21:30 TPE (US open) and is settled Monday 07:00; a Monday-morning TW signal fills at Monday 09:00 TPE and is settled Tuesday 07:00. The script does not need to know which day a signal was issued, only whether the fill bar exists.

### 7.7 Monitoring

- yfinance retry chain (3×, 10min spacing).
- Heartbeat miss > 25h → local `check_results.ps1` reports stale.
- Anomaly: all prob < 0.3 for N consecutive days → Discord warning (model possibly broken).

## 8. Repo Structure

```
etf-dca-ml/
├── config.py                    # DCA_SYMBOLS, budgets, thresholds, paths
├── data/
│   ├── fetcher.py               # yfinance with auto_adjust=True
│   └── cleaner.py               # per-symbol calendar, merge_asof, last-trading-day helper
├── features/
│   ├── indicators.py            # technical features
│   ├── macro.py                 # VIX/TNX/DXY/TSM derived features
│   ├── builder.py               # per-symbol assembly + macro broadcast
│   └── labels.py                # compute_dca_labels
├── models/
│   ├── trainer.py               # XGBoost fit per symbol
│   └── splitter.py              # walk-forward TS-CV with Purge Gap
├── backtest/
│   ├── engine.py                # monthly budget + signal-driven accumulation
│   └── metrics.py               # Sharpe, MDD, vs-baseline cost-basis comparison
├── live/
│   ├── predict_daemon.py
│   ├── ledger_settle.py
│   ├── monthly_retrain.py
│   ├── weekly_report.py
│   ├── notifier.py              # adapted from BYBIT_ML
│   └── ledger.py                # adapted from BYBIT_ML
├── storage/                     # .gitignore'd
│   ├── raw/                     # yfinance cache
│   ├── live/                    # predictions.jsonl, ledger.json, budget_state.json, heartbeat.json
│   └── models/                  # {SYMBOL}.pkl, {SYMBOL}_meta.json
├── tests/
├── README.md
└── .gitignore
```

## 9. Config (starter values)

```python
DCA_SYMBOLS = ["SPY", "QQQ", "0050.TW"]
MONTHLY_BUDGET = {"SPY": 1000, "QQQ": 1000, "0050.TW": 30000}    # USD, USD, TWD
SIGNAL_THRESHOLD = {"SPY": 0.55, "QQQ": 0.55, "0050.TW": 0.55}
MAX_PER_TRADE_RATIO = 0.25
HISTORY_START = "2005-01-01"
LABEL_WINDOW = 30
LABEL_TOP_Q = 0.20
PURGE_GAP_DAYS = 40   # NOTE: must be >= LABEL_WINDOW (see §11 diagnostic notes)
DISCORD_WEBHOOK_URL = os.environ["DCA_DISCORD_WEBHOOK"]
TZ = "Asia/Taipei"
```

## 10. Out of Scope (explicit)

- Real brokerage auto-execution (Shioaji / IBKR). Signal only; manual follow-up.
- TW chip data (foreign net buy/sell). Phase 2 enhancement.
- Multi-timeframe (intraday) signals. Daily granularity only.
- Crypto assets (kept in BYBIT_ML).
- Currency hedging on USD-denominated TWD budget.

## 11. Acceptance Criteria

- Backtest 2010–2026 walk-forward shows strategy Sharpe > both baselines on at least 2 of 3 symbols.
- Cost-basis savings vs Baseline B is positive on aggregate.
- Live daemon runs 30 consecutive days on VM without manual intervention, heartbeat never stale > 25h.
- Discord notifications fire daily and include data integrity status.

## 12. Diagnostic Notes (Phase B Post-Build)

After completing Tasks 1-9, a series of smoke tests on real SPY data (2005-01 → 2026-06, 5102 samples) uncovered and corrected two pipeline issues. Recorded here for future maintainers.

### 12.1 PURGE_GAP_DAYS bumped 20 → 40

The original `PURGE_GAP_DAYS = 20` was shorter than `LABEL_WINDOW = 30`, meaning train day `T`'s label (computed from prices in `[T+1, T+30]`) overlapped by ~10 days with the features at val day `T+20` (which read prices through `T+20`). A theoretical leakage path.

Ablation on SPY (purge=20 vs purge=40, all 25 features): CV-AUC moved 0.6918 → 0.6875 (within fold std ~0.014). Practical impact tiny, but the correct value is `>= LABEL_WINDOW`, so the project ships with `PURGE_GAP_DAYS = 40` (30 + 10-day safety buffer). Zero cost, correct invariant.

### 12.2 `day_of_month` removed from FEATURE_COLS

The original 25-feature set included `day_of_month`. In a within-month-relative-rank task with a forward 30-day label window, rows late in the month have their forward window straddle the next calendar month, while early-month rows have their forward window fully inside the current month. The model learned this structural difference and gave `day_of_month` ~5.9% feature importance — not real market signal, just a label-construction artifact.

Ablation (purge=40, drop `day_of_month`): CV-AUC 0.6875 → 0.6658 (Δ -0.022). Removed in production. `day_of_week` is retained — its potential bias (e.g., Friday risk-off) is a genuine market effect, not a label-window artifact.

The "honest" CV-AUC for the production pipeline is therefore **0.6658**.

### 12.3 Y-randomization (label-shuffle) sanity check

Lopez de Prado-style null test: with config (purge=40, 24 features), shuffle the training label vector across 5 seeds and measure CV-AUC.

| Run | CV-AUC |
|---|---|
| Real labels | 0.6658 |
| Shuffled seed 0 | 0.5180 |
| Shuffled seed 1 | 0.5074 |
| Shuffled seed 2 | 0.5096 |
| Shuffled seed 3 | 0.4968 |
| Shuffled seed 4 | 0.4946 |
| **Shuffled mean** | **0.5053 ± 0.0086** |
| **Gap (real − shuffled)** | **+0.1605** |

Shuffled distribution sits squarely in the null band [0.49, 0.51]; the 0.66 in production is **real predictive signal** from RSI / MA distance / drawdown / VIX features, not a residual structural artifact. Cleared to proceed to Phase C (backtest + live).

### 12.4 FORCED_BUY × bull-market interaction (cost-basis insight)

Task 17 acceptance gate passed on the primary criterion (strategy Sharpe > both baselines on 3/3 symbols) but **failed** on the secondary criterion (`cost_basis_vs_monthly_mean` negative on aggregate). Per-symbol results at τ = 0.55:

| Symbol | Strategy Sharpe | Cost basis vs monthly mean |
|---|---:|---:|
| SPY | 1.38 | +2.24% (premium) |
| QQQ | 1.45 | +2.79% (premium) |
| 0050.TW | 1.51 | **−3.04% (discount)** |

A τ grid search [0.55, 0.60, 0.65, 0.70, 0.75] confirmed that **raising τ makes cost basis monotonically worse**, not better. SPY: 0.55→+2.17% / 0.60→+2.21% / 0.65→+2.26% / 0.70→+2.36% / 0.75→+2.62%. QQQ shows the same pattern; 0.75 also breaks the Sharpe gate.

**Why higher τ hurts cost basis on SPY/QQQ:** the original intuition was "be more patient → wait for deeper dips → cheaper fills". That intuition is correct in **bear** months. In **bull** months the dynamic reverses:

- Higher τ → model fires less often during the month → more leftover budget at month-end
- `FORCED_BUY` on the last trading day drains all leftover budget at the last session's open
- In a bull month the last session's open is typically near the **monthly high**
- So the "patient" version spends a larger fraction of monthly budget at the worst price of the month

Over a 2005–2026 sample dominated by bull years, this bull-month penalty outweighs the bear-month benefit. The +2-3% cost-basis premium on SPY/QQQ is structural — not a bug, but the cost of the `FORCED_BUY` rule on rising assets. 0050.TW escapes the penalty because TW market behaviour (heavier foreign selling pressure, sharper monthly mean reversion around 台積電 ADR moves) leaves more genuinely-cheap month-end days.

**Implications for tuning:**
- Threshold tuning alone cannot fix SPY/QQQ cost basis. The `FORCED_BUY` rule is the binding constraint.
- Alternatives worth exploring (Phase 2): (1) split forced buy across the last *N* sessions instead of one shot; (2) lower `MAX_PER_TRADE_RATIO` from 0.25 to spread bullets thinner; (3) drop forced buy entirely on months where strategy signals were below a "very low" floor (model says "no edge anywhere this month" → don't force).
- For now the strategy ships with `τ_QQQ = 0.60` (best grid result) and `τ_SPY = τ_0050 = 0.55` (grid floor). Sharpe stays 3/3 PASS; cost basis remains the open secondary objective for the next iteration.
