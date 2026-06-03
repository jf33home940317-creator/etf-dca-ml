from unittest.mock import patch

from live.notifier import send_discord


def test_send_discord_posts_payload():
    with patch("live.notifier.requests.post") as mock:
        mock.return_value.status_code = 204
        send_discord("hello", webhook="https://discord.test/hook")
    args, kwargs = mock.call_args
    assert args[0] == "https://discord.test/hook"
    assert "hello" in kwargs["json"]["content"]
