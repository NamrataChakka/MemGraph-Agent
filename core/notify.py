"""
Push notification helper via ntfy.sh.

Setup:
  1. Install ntfy app on phone (iOS: App Store, Android: Play Store/F-Droid)
  2. Subscribe to your chosen topic in the app
  3. Set NTFY_TOPIC in your .env file

Configure via environment variables:
  NTFY_TOPIC   — your unique topic name (e.g. "myhome-abc123")
  NTFY_SERVER  — ntfy server URL (default: https://ntfy.sh, can self-host)
"""

from __future__ import annotations

import logging
import os
import urllib.request
import urllib.parse

logger = logging.getLogger(__name__)

_NTFY_TOPIC  = os.environ.get("NTFY_TOPIC", "")
_NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")


def is_configured() -> bool:
    return bool(_NTFY_TOPIC)


def send_notification(subject: str, body: str, **_kwargs) -> None:
    """Send a push notification via ntfy.sh.
    No-op (with debug log) if NTFY_TOPIC is not set."""
    if not is_configured():
        logger.debug("ntfy not configured — skipping notification: %s", subject)
        return

    url = f"{_NTFY_SERVER}/{urllib.parse.quote(_NTFY_TOPIC, safe='')}"
    try:
        req = urllib.request.Request(
            url,
            data=body.encode("utf-8"),
            headers={
                "Title": subject,
                "Priority": "default",
            },
        )
        with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310
            resp.read()
        logger.info("ntfy notification sent: %s → %s", subject, _NTFY_TOPIC)
    except Exception as e:
        logger.warning("ntfy notification failed %r: %s", subject, e)
