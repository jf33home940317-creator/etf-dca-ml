import requests

import config


def send_discord(message: str, webhook: str | None = None) -> None:
    url = webhook or config.DISCORD_WEBHOOK_URL
    if not url:
        print(f"[notifier] No webhook configured. Message:\n{message}")
        return
    resp = requests.post(url, json={"content": message[:1900]}, timeout=10)
    if resp.status_code >= 400:
        print(f"[notifier] Discord post failed: {resp.status_code} {resp.text}")
