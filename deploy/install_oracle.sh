#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$HOME/etf-dca-ml"
PYTHON_BIN="$REPO_DIR/.venv/bin/python"

if [ ! -d "$REPO_DIR" ]; then
  echo "Clone or rsync etf-dca-ml to $REPO_DIR first"
  exit 1
fi

cd "$REPO_DIR"
python3 -m venv .venv
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements.txt

# DCA_DISCORD_WEBHOOK must already be exported in ~/.bashrc (or use systemd EnvironmentFile)
echo "Set DCA_DISCORD_WEBHOOK in ~/.bashrc, then install cron entries from deploy/crontab.txt"
