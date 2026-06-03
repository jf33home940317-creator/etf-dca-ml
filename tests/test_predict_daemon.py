from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from live.predict_daemon import decide_action


class FakeModel:
    def __init__(self, p): self.p = p
    def predict_proba(self, X):
        return np.array([[1 - self.p, self.p]])


def test_decide_action_emits_buy_when_above_threshold():
    feat_row = pd.DataFrame({"x": [1.0]})
    action = decide_action(
        symbol="SPY", prob=0.7, threshold=0.55, budget_left=1000.0,
        monthly_budget=1000.0, max_per_trade_ratio=0.25,
        is_last_day=False,
    )
    assert action["reason"] == "BUY"
    assert action["amount"] == 250.0
    assert action["new_budget"] == 750.0


def test_decide_action_no_action_below_threshold():
    action = decide_action(
        symbol="SPY", prob=0.30, threshold=0.55, budget_left=1000.0,
        monthly_budget=1000.0, max_per_trade_ratio=0.25,
        is_last_day=False,
    )
    assert action is None


def test_decide_action_forces_buy_on_last_day():
    action = decide_action(
        symbol="SPY", prob=0.20, threshold=0.55, budget_left=750.0,
        monthly_budget=1000.0, max_per_trade_ratio=0.25,
        is_last_day=True,
    )
    assert action["reason"] == "FORCED_BUY"
    assert action["amount"] == 750.0
    assert action["new_budget"] == 0.0
