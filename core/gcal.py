"""
Google Calendar push integration (push-only).

Pushes events created via the agent to a Google Calendar.
Falls back gracefully if credentials are not configured.

Setup:
  1. Google Cloud Console → create project → enable Calendar API
  2. Create OAuth 2.0 credentials (Desktop app type)
  3. Download credentials JSON → save to path set in GOOGLE_CREDENTIALS_FILE
  4. First run opens browser for one-time auth → token saved to GOOGLE_TOKEN_FILE
  5. Subsequent runs use the saved refresh token automatically

Environment variables:
  GOOGLE_CREDENTIALS_FILE — path to OAuth client credentials JSON
                             (default: data/credentials.json)
  GOOGLE_TOKEN_FILE        — path to store the access/refresh token
                             (default: data/google_token.json)
  GOOGLE_CALENDAR_ID       — target calendar (default: primary)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_CREDENTIALS_FILE = Path(os.environ.get("GOOGLE_CREDENTIALS_FILE", "data/credentials.json"))
_TOKEN_FILE       = Path(os.environ.get("GOOGLE_TOKEN_FILE",       "data/google_token.json"))
_CALENDAR_ID      = os.environ.get("GOOGLE_CALENDAR_ID", "primary")

_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def is_configured() -> bool:
    """Return True if Google Calendar credentials file exists."""
    return _CREDENTIALS_FILE.exists()


def _get_service():
    """Build and return an authenticated Google Calendar service object."""
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as e:
        raise RuntimeError(
            "Google Calendar packages not installed. Run: "
            "pip install google-api-python-client google-auth-oauthlib"
        ) from e

    creds = None

    if _TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_FILE), _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(_CREDENTIALS_FILE), _SCOPES
            )
            # Opens browser for one-time auth
            creds = flow.run_local_server(port=0)

        _TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        _TOKEN_FILE.write_text(creds.to_json())

    return build("calendar", "v3", credentials=creds)


def push_event(title: str, date: str, time: str = "", notes: str = "") -> dict:
    """Push an event to Google Calendar.

    Args:
        title: Event summary/title
        date:  Date string in YYYY-MM-DD format
        time:  Optional time in HH:MM format
        notes: Optional description

    Returns:
        dict with 'gcal_id' and 'gcal_link' on success, or 'error' on failure.
    """
    if not is_configured():
        return {"skipped": "Google Calendar not configured"}

    try:
        service = _get_service()

        if time:
            # Timed event
            start = {"dateTime": f"{date}T{time}:00", "timeZone": "local"}
            end   = {"dateTime": f"{date}T{time}:00", "timeZone": "local"}
        else:
            # All-day event
            start = {"date": date}
            end   = {"date": date}

        event_body = {
            "summary":     title,
            "description": notes or "",
            "start":       start,
            "end":         end,
        }

        created = service.events().insert(
            calendarId=_CALENDAR_ID, body=event_body
        ).execute()

        logger.info("Google Calendar event created: %s (%s)", title, created.get("id"))
        return {
            "gcal_id":   created.get("id", ""),
            "gcal_link": created.get("htmlLink", ""),
        }

    except Exception as e:
        logger.warning("Google Calendar push failed for %r: %s", title, e)
        return {"error": str(e)}
