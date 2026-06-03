import pandas as pd
from data.cleaner import (
    is_first_trading_day_of_month,
    is_last_trading_day_of_month,
)

def run_backtest(
    symbol: str,
    prices: pd.DataFrame,
    features: pd.DataFrame,
    model,
    feature_cols: list[str],
    monthly_budget: float,
    threshold: float,
    max_per_trade_ratio: float = 0.25,
) -> list[dict]:
    trades: list[dict] = []
    budget_left = 0.0
    common = prices.index.intersection(features.dropna(subset=feature_cols).index)
    for d in common:
        if is_first_trading_day_of_month(prices, d):
            budget_left = monthly_budget

        row = features.loc[[d], feature_cols]
        prob = float(model.predict_proba(row)[0, 1])

        fill_price = float(prices.loc[d, "open"])
        if prob >= threshold and budget_left > 0:
            amount = min(monthly_budget * max_per_trade_ratio, budget_left)
            trades.append({
                "date": d, "symbol": symbol, "reason": "BUY",
                "prob": prob, "amount": amount, "fill_price": fill_price,
            })
            budget_left -= amount

        if is_last_trading_day_of_month(prices, d) and budget_left > 0:
            trades.append({
                "date": d, "symbol": symbol, "reason": "FORCED_BUY",
                "prob": prob, "amount": budget_left, "fill_price": fill_price,
            })
            budget_left = 0.0
    return trades
