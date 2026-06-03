import numpy as np
import pandas as pd
from data.cleaner import is_first_trading_day_of_month


def classic_dca_baseline(symbol: str, prices: pd.DataFrame, monthly_budget: float):
    out = []
    for d in prices.index:
        if is_first_trading_day_of_month(prices, d):
            out.append({
                "date": d, "symbol": symbol, "reason": "CLASSIC_DCA",
                "amount": monthly_budget, "fill_price": float(prices.loc[d, "open"]),
            })
    return out


def weekly_dca_baseline(symbol: str, prices: pd.DataFrame, monthly_budget: float):
    weekly = monthly_budget / 4.0
    out = []
    for d in prices.index:
        if d.dayofweek == 0:  # Monday
            out.append({
                "date": d, "symbol": symbol, "reason": "WEEKLY_DCA",
                "amount": weekly, "fill_price": float(prices.loc[d, "open"]),
            })
    return out


def equity_curve_from_trades(trades: list, prices: pd.DataFrame) -> pd.Series:
    cash_in = pd.Series(0.0, index=prices.index)
    shares = pd.Series(0.0, index=prices.index)
    cum_shares = 0.0
    cum_cash = 0.0
    for t in trades:
        s = t["amount"] / t["fill_price"]
        cum_shares += s
        cum_cash += t["amount"]
        cash_in.loc[t["date"]] = cum_cash
        shares.loc[t["date"]] = cum_shares
    shares = shares.replace(0.0, np.nan).ffill().fillna(0.0)
    cash_in = cash_in.replace(0.0, np.nan).ffill().fillna(0.0)
    market_value = shares * prices["close"]
    # Equity = market value (cash invested is implicit; fine for relative comparison)
    return market_value


def summary_stats(equity: pd.Series) -> dict:
    ret = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity[equity > 0].iloc[0] - 1 if (equity > 0).any() else 0.0
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-6)
    cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1
    sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
    cummax = equity.cummax()
    dd = (equity / cummax - 1).min()
    calmar = cagr / abs(dd) if dd < 0 else 0.0
    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_dd": float(dd),
        "calmar": float(calmar),
    }


def cost_basis_vs_monthly_mean(trades: list, prices: pd.DataFrame) -> float:
    if not trades:
        return 0.0
    total_amount = sum(t["amount"] for t in trades)
    weighted_price = sum(t["amount"] * t["fill_price"] for t in trades) / total_amount
    monthly_mean = prices.groupby(prices.index.to_period("M"))["close"].mean().mean()
    return float(weighted_price / monthly_mean - 1)
