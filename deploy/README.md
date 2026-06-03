# deploy/

## First-time setup on Oracle VM

1. rsync repo to `~/etf-dca-ml`
2. `bash deploy/install_oracle.sh`
3. Export `DCA_DISCORD_WEBHOOK` in `~/.bashrc`
4. Run an initial `python -m live.monthly_retrain` to populate `storage/models/`
5. Install cron: `crontab deploy/crontab.txt` (edit webhook value first)
6. Verify: `tail -f storage/live/predict.log` next morning at 06:30 TPE

## Timezone

VM must be `Asia/Taipei`. Verify with `timedatectl`. If UTC, either change TZ
or rewrite crontab.txt with UTC times (subtract 8h).

## Manual check

```
python -m live.predict_daemon   # forces a run; sends Discord
python -m live.ledger_settle    # settles any PENDING signals
```
