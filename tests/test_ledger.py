import json
import tempfile
from pathlib import Path
import pandas as pd
from live.ledger import append_signal, read_budget, write_budget, read_signals

def test_append_signal_creates_jsonl(tmp_path):
    path = tmp_path / "predictions.jsonl"
    append_signal(path, {"date": "2024-01-10", "symbol": "SPY",
                          "reason": "BUY", "amount": 250, "prob": 0.7,
                          "fill_price": None, "status": "PENDING"})
    append_signal(path, {"date": "2024-01-11", "symbol": "QQQ",
                          "reason": "BUY", "amount": 250, "prob": 0.6,
                          "fill_price": None, "status": "PENDING"})
    sigs = read_signals(path)
    assert len(sigs) == 2
    assert sigs[0]["symbol"] == "SPY"

def test_budget_state_roundtrip(tmp_path):
    path = tmp_path / "budget_state.json"
    write_budget(path, {"SPY": 750.0, "QQQ": 1000.0, "0050.TW": 30000.0})
    assert read_budget(path, "SPY") == 750.0
    assert read_budget(path, "QQQ") == 1000.0

def test_read_budget_returns_none_when_missing(tmp_path):
    assert read_budget(tmp_path / "nope.json", "SPY") is None
