import json
import os
import tempfile
from pathlib import Path

def _atomic_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

def append_signal(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)

def read_signals(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def write_budget(path: Path, state: dict) -> None:
    _atomic_write(path, json.dumps(state, indent=2))

def read_budget(path: Path, symbol: str | None = None):
    if not path.exists():
        return None if symbol else {}
    state = json.loads(path.read_text(encoding="utf-8"))
    return state.get(symbol) if symbol else state

def update_budget(path: Path, symbol: str, new_value: float) -> None:
    state = read_budget(path) or {}
    state[symbol] = new_value
    write_budget(path, state)

def rewrite_signals(path: Path, signals: list[dict]) -> None:
    """Used by ledger_settle to overwrite predictions.jsonl with updated rows."""
    content = "".join(json.dumps(s, default=str) + "\n" for s in signals)
    _atomic_write(path, content)
