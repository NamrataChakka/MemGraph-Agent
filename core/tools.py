"""
Tool definitions and execution dispatch for the MemGraph agent.

Add new tools here:
  1. Define a schema (WEB_SEARCH_TOOL pattern)
  2. Add it to DEFAULT_TOOLS
  3. Add a branch in execute_tool()

Data storage:
  All persistent data lives in DATA_DIR (default: ./data, override via DATA_DIR env var).
  Files: lists.json, notes.json, events.json, expenses.json
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Storage ───────────────────────────────────────────────────────────────────

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
_lock = threading.Lock()


_LIST_FILES = {"events.json", "expenses.json", "books.json"}

def _load(filename: str) -> Any:
    path = DATA_DIR / filename
    if not path.exists():
        return [] if filename in _LIST_FILES else {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(filename: str, data: Any) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Tool schemas ──────────────────────────────────────────────────────────────

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for up-to-date information. "
            "Use when the user asks about current events, recent news, live data, "
            "or anything that may have changed since your training cutoff."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (1–10, default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

LIST_VIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "list_view",
        "description": "View all items in a named list (e.g. grocery, todo, shopping, watchlist).",
        "parameters": {
            "type": "object",
            "properties": {
                "list_name": {"type": "string", "description": "Name of the list"},
            },
            "required": ["list_name"],
        },
    },
}

LIST_ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "list_add",
        "description": "Add one or more items to a named list.",
        "parameters": {
            "type": "object",
            "properties": {
                "list_name": {"type": "string", "description": "Name of the list"},
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Items to add",
                },
            },
            "required": ["list_name", "items"],
        },
    },
}

LIST_REMOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "list_remove",
        "description": "Remove one or more items from a named list.",
        "parameters": {
            "type": "object",
            "properties": {
                "list_name": {"type": "string", "description": "Name of the list"},
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Items to remove (case-insensitive match)",
                },
            },
            "required": ["list_name", "items"],
        },
    },
}

LIST_CLEAR_TOOL = {
    "type": "function",
    "function": {
        "name": "list_clear",
        "description": "Remove all items from a named list.",
        "parameters": {
            "type": "object",
            "properties": {
                "list_name": {"type": "string", "description": "Name of the list to clear"},
            },
            "required": ["list_name"],
        },
    },
}

LISTS_ALL_TOOL = {
    "type": "function",
    "function": {
        "name": "lists_all",
        "description": "Show all lists and their contents.",
        "parameters": {"type": "object", "properties": {}},
    },
}

NOTE_WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "note_write",
        "description": "Create or overwrite a note by title.",
        "parameters": {
            "type": "object",
            "properties": {
                "title":   {"type": "string", "description": "Note title"},
                "content": {"type": "string", "description": "Note content"},
            },
            "required": ["title", "content"],
        },
    },
}

NOTE_READ_TOOL = {
    "type": "function",
    "function": {
        "name": "note_read",
        "description": "Read a note by title.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Note title"},
            },
            "required": ["title"],
        },
    },
}

NOTE_LIST_TOOL = {
    "type": "function",
    "function": {
        "name": "note_list",
        "description": "List all saved note titles.",
        "parameters": {"type": "object", "properties": {}},
    },
}

NOTE_DELETE_TOOL = {
    "type": "function",
    "function": {
        "name": "note_delete",
        "description": "Delete a note by title.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Note title to delete"},
            },
            "required": ["title"],
        },
    },
}

WEATHER_GET_TOOL = {
    "type": "function",
    "function": {
        "name": "weather_get",
        "description": (
            "Get current weather conditions and forecast for a location. "
            "Use when the user asks about weather, temperature, rain, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, postcode, or coordinates (e.g. 'London', '10001', '48.8566,2.3522')",
                },
            },
            "required": ["location"],
        },
    },
}

EVENT_ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "event_add",
        "description": "Add an event to the shared household calendar.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title"},
                "date":  {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "time":  {"type": "string", "description": "Time in HH:MM format (optional)"},
                "notes": {"type": "string", "description": "Extra notes (optional)"},
            },
            "required": ["title", "date"],
        },
    },
}

EVENTS_LIST_TOOL = {
    "type": "function",
    "function": {
        "name": "events_list",
        "description": "List upcoming household calendar events.",
        "parameters": {
            "type": "object",
            "properties": {
                "days_ahead": {
                    "type": "integer",
                    "description": "How many days ahead to look (default 14)",
                    "default": 14,
                },
            },
        },
    },
}

EVENT_DELETE_TOOL = {
    "type": "function",
    "function": {
        "name": "event_delete",
        "description": "Delete a calendar event by title and date.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title"},
                "date":  {"type": "string", "description": "Date in YYYY-MM-DD format"},
            },
            "required": ["title", "date"],
        },
    },
}

EXPENSE_ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "expense_add",
        "description": "Log a household expense.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount":      {"type": "number",  "description": "Amount spent"},
                "category":    {"type": "string",  "description": "Category (e.g. groceries, utilities, dining, transport, health, other)"},
                "description": {"type": "string",  "description": "Brief description (optional)"},
            },
            "required": ["amount", "category"],
        },
    },
}

EXPENSES_SUMMARY_TOOL = {
    "type": "function",
    "function": {
        "name": "expenses_summary",
        "description": "Summarise household expenses by category for the last N days.",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of past days to include (default 30)",
                    "default": 30,
                },
            },
        },
    },
}

BOOK_ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "book_add",
        "description": (
            "Add a book to the household reading tracker. "
            "ONLY include fields the user explicitly mentioned. "
            "Do NOT guess or make up the author, summary, dates, or any other details the user did not provide. "
            "Leave unknown fields empty."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title":      {"type": "string",  "description": "Book title"},
                "author":     {"type": "string",  "description": "Author name — ONLY if the user said it"},
                "genre":      {"type": "string",  "description": "Genre (e.g. Fiction, Science Fiction, Fantasy, Mystery, Romance, Non-Fiction, Self-Help, Biography, History, Horror, Thriller, Other) — ONLY if the user said it or it's obvious"},
                "rating":     {"type": "number",  "description": "Rating out of 10 — ONLY if the user gave a rating"},
                "start_date": {"type": "string",  "description": "Date started reading (YYYY-MM-DD) — ONLY if the user said it"},
                "end_date":   {"type": "string",  "description": "Date finished reading (YYYY-MM-DD) — ONLY if the user said it"},
                "summary":    {"type": "string",  "description": "Brief book summary — ONLY if the user described the book"},
                "reader":     {"type": "string",  "description": "Who read it — ONLY if the user said who"},
            },
            "required": ["title"],
        },
    },
}

BOOK_LIST_TOOL = {
    "type": "function",
    "function": {
        "name": "book_list",
        "description": "List all books in the reading tracker, optionally filtered by genre or reader.",
        "parameters": {
            "type": "object",
            "properties": {
                "genre":  {"type": "string",  "description": "Filter by genre (optional)"},
                "reader": {"type": "string",  "description": "Filter by reader (optional)"},
            },
        },
    },
}

BOOK_UPDATE_TOOL = {
    "type": "function",
    "function": {
        "name": "book_update",
        "description": "Update a book's details (rating, dates, summary, etc.).",
        "parameters": {
            "type": "object",
            "properties": {
                "title":      {"type": "string",  "description": "Book title to update"},
                "author":     {"type": "string",  "description": "Updated author (optional)"},
                "genre":      {"type": "string",  "description": "Updated genre (optional)"},
                "rating":     {"type": "number",  "description": "Updated rating out of 10 (optional)"},
                "start_date": {"type": "string",  "description": "Updated start date (optional)"},
                "end_date":   {"type": "string",  "description": "Updated end date (optional)"},
                "summary":    {"type": "string",  "description": "Updated summary (optional)"},
                "reader":     {"type": "string",  "description": "Updated reader (optional)"},
            },
            "required": ["title"],
        },
    },
}

BOOK_DELETE_TOOL = {
    "type": "function",
    "function": {
        "name": "book_delete",
        "description": "Remove a book from the reading tracker.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Book title to delete"},
            },
            "required": ["title"],
        },
    },
}

DEFAULT_TOOLS: list[dict] = [
    WEB_SEARCH_TOOL,
    LIST_VIEW_TOOL,
    LIST_ADD_TOOL,
    LIST_REMOVE_TOOL,
    LIST_CLEAR_TOOL,
    LISTS_ALL_TOOL,
    NOTE_WRITE_TOOL,
    NOTE_READ_TOOL,
    NOTE_LIST_TOOL,
    NOTE_DELETE_TOOL,
    WEATHER_GET_TOOL,
    EVENT_ADD_TOOL,
    EVENTS_LIST_TOOL,
    EVENT_DELETE_TOOL,
    EXPENSE_ADD_TOOL,
    EXPENSES_SUMMARY_TOOL,
    BOOK_ADD_TOOL,
    BOOK_LIST_TOOL,
    BOOK_UPDATE_TOOL,
    BOOK_DELETE_TOOL,
]


# ── Dispatch ──────────────────────────────────────────────────────────────────

def execute_tool(name: str, args: dict[str, Any]) -> dict:
    """Dispatch a tool call by name. Raises ValueError for unknown tools."""
    dispatch = {
        "web_search":        lambda: _web_search(str(args.get("query", "")), int(args.get("max_results", 5))),
        "list_view":         lambda: _list_view(str(args["list_name"])),
        "list_add":          lambda: _list_add(str(args["list_name"]), list(args.get("items", []))),
        "list_remove":       lambda: _list_remove(str(args["list_name"]), list(args.get("items", []))),
        "list_clear":        lambda: _list_clear(str(args["list_name"])),
        "lists_all":         lambda: _lists_all(),
        "note_write":        lambda: _note_write(str(args["title"]), str(args["content"])),
        "note_read":         lambda: _note_read(str(args["title"])),
        "note_list":         lambda: _note_list(),
        "note_delete":       lambda: _note_delete(str(args["title"])),
        "weather_get":       lambda: _weather_get(str(args["location"])),
        "event_add":         lambda: _event_add(str(args["title"]), str(args["date"]), str(args.get("time", "")), str(args.get("notes", ""))),
        "events_list":       lambda: _events_list(int(args.get("days_ahead", 14))),
        "event_delete":      lambda: _event_delete(str(args["title"]), str(args["date"])),
        "expense_add":       lambda: _expense_add(float(args["amount"]), str(args["category"]), str(args.get("description", ""))),
        "expenses_summary":  lambda: _expenses_summary(int(args.get("days", 30))),
        "book_add":    lambda: _book_add(str(args["title"]), str(args.get("author", "")), str(args.get("genre", "")), float(args.get("rating", 0)) if args.get("rating") else 0, str(args.get("start_date", "")), str(args.get("end_date", "")), str(args.get("summary", "")), str(args.get("reader", ""))),
        "book_list":   lambda: _book_list(str(args.get("genre", "")), str(args.get("reader", ""))),
        "book_update": lambda: _book_update(str(args["title"]), args),
        "book_delete": lambda: _book_delete(str(args["title"])),
    }
    if name not in dispatch:
        raise ValueError(f"Unknown tool: {name!r}")
    return dispatch[name]()


# ── Web search ────────────────────────────────────────────────────────────────

def _web_search(query: str, max_results: int = 5) -> dict:
    """Web search via DuckDuckGo HTML (no API key or pip dependency)."""
    if not query.strip():
        return {"error": "Empty query"}
    max_results = min(max(max_results, 1), 10)

    # Try the ddgs/duckduckgo-search package first (best results)
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        raw = DDGS().text(query, max_results=max_results)
        results = [
            {"title": r.get("title", ""), "url": r.get("href", ""), "body": r.get("body", "")}
            for r in (raw or [])
        ]
        # Only use package results if they have actual body content
        if results and any(r.get("body") for r in results):
            return {"query": query, "results": results}
    except Exception:
        pass

    # Fallback: DuckDuckGo Lite via POST (also used when package returns empty bodies) (zero dependencies)
    try:
        import urllib.request
        import re as _re
        import html as _html

        post_data = urllib.parse.urlencode({"q": query}).encode()
        req = urllib.request.Request(
            "https://lite.duckduckgo.com/lite/",
            data=post_data,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            page = resp.read().decode("utf-8", errors="replace")

        results = []
        # DDG Lite uses table cells: link td → url td → snippet td
        # Extract all <td> contents and pair links with their snippets
        tds = _re.findall(r'<td\b[^>]*>(.*?)</td>', page, _re.DOTALL)
        i = 0
        while i < len(tds) and len(results) < max_results:
            # Look for a td containing a nofollow link
            link_match = _re.search(
                r'<a[^>]+rel="nofollow"[^>]+href="([^"]+)"[^>]*>\s*(.*?)\s*</a>',
                tds[i], _re.DOTALL,
            )
            if link_match:
                href = link_match.group(1)
                title = _re.sub(r'<[^>]+>', '', link_match.group(2)).strip()
                # Snippet is typically 2 cells later (after the URL cell)
                body = ""
                for j in range(i + 1, min(i + 4, len(tds))):
                    candidate = _re.sub(r'<[^>]+>', '', tds[j]).strip()
                    if len(candidate) > 60 and not candidate.startswith("http"):
                        body = _html.unescape(candidate)
                        break
                if href.startswith("http") and title:
                    results.append({
                        "title": _html.unescape(title),
                        "url": href,
                        "body": body,
                    })
            i += 1

        if results:
            return {"query": query, "results": results}
        return {"query": query, "results": [], "note": "No results found"}
    except Exception as e:
        logger.warning("Web search failed for %r: %s", query, e)
        return {"error": str(e), "query": query}


# ── List tools ────────────────────────────────────────────────────────────────

def _list_view(list_name: str) -> dict:
    with _lock:
        data = _load("lists.json")
    items = data.get(list_name.lower(), [])
    return {"list": list_name, "items": items, "count": len(items)}


def _list_add(list_name: str, items: list) -> dict:
    name = list_name.lower()
    with _lock:
        data = _load("lists.json")
        existing = data.get(name, [])
        added = []
        for item in items:
            item_str = str(item).strip()
            if item_str and item_str.lower() not in [e.lower() for e in existing]:
                existing.append(item_str)
                added.append(item_str)
        data[name] = existing
        _save("lists.json", data)
    if added and name == "grocery":
        items_str = ", ".join(added)
        _notify(
            f"Grocery list updated",
            f"Added to grocery list: {items_str}\n\nFull list: {', '.join(existing)}",
        )
    return {"list": list_name, "added": added, "items": existing}


def _list_remove(list_name: str, items: list) -> dict:
    name = list_name.lower()
    with _lock:
        data = _load("lists.json")
        existing = data.get(name, [])
        to_remove = {str(i).strip().lower() for i in items}
        removed = [e for e in existing if e.lower() in to_remove]
        data[name] = [e for e in existing if e.lower() not in to_remove]
        _save("lists.json", data)
    return {"list": list_name, "removed": removed, "items": data[name]}


def _list_clear(list_name: str) -> dict:
    name = list_name.lower()
    with _lock:
        data = _load("lists.json")
        data[name] = []
        _save("lists.json", data)
    return {"list": list_name, "cleared": True}


def _lists_all() -> dict:
    with _lock:
        data = _load("lists.json")
    return {"lists": {k: v for k, v in data.items()}}


# ── Note tools ────────────────────────────────────────────────────────────────

def _note_write(title: str, content: str) -> dict:
    with _lock:
        data = _load("notes.json")
        data[title] = content
        _save("notes.json", data)
    return {"title": title, "saved": True}


def _note_read(title: str) -> dict:
    with _lock:
        data = _load("notes.json")
    if title not in data:
        # try case-insensitive
        match = next((k for k in data if k.lower() == title.lower()), None)
        if match:
            return {"title": match, "content": data[match]}
        return {"error": f"Note '{title}' not found", "available": list(data.keys())}
    return {"title": title, "content": data[title]}


def _note_list() -> dict:
    with _lock:
        data = _load("notes.json")
    return {"notes": list(data.keys()), "count": len(data)}


def _note_delete(title: str) -> dict:
    with _lock:
        data = _load("notes.json")
        if title not in data:
            match = next((k for k in data if k.lower() == title.lower()), None)
            if not match:
                return {"error": f"Note '{title}' not found"}
            title = match
        del data[title]
        _save("notes.json", data)
    return {"title": title, "deleted": True}


# ── Weather tool ──────────────────────────────────────────────────────────────

def _weather_get(location: str) -> dict:
    """Fetch weather from wttr.in (no API key required)."""
    if not location.strip():
        return {"error": "Empty location"}
    try:
        import urllib.request
        url = f"https://wttr.in/{urllib.parse.quote(location)}?format=j1"
        with urllib.request.urlopen(url, timeout=8) as resp:  # noqa: S310
            raw = json.loads(resp.read().decode())

        current = raw["current_condition"][0]
        area    = raw["nearest_area"][0]
        city    = area["areaName"][0]["value"]
        country = area["country"][0]["value"]

        return {
            "location":    f"{city}, {country}",
            "temp_c":      int(current["temp_C"]),
            "temp_f":      int(current["temp_F"]),
            "feels_like_c": int(current["FeelsLikeC"]),
            "description": current["weatherDesc"][0]["value"],
            "humidity":    int(current["humidity"]),
            "wind_kmph":   int(current["windspeedKmph"]),
            "visibility_km": int(current["visibility"]),
        }
    except Exception as e:
        logger.warning("Weather fetch failed for %r: %s", location, e)
        return {"error": str(e), "location": location}


# Add missing import for weather tool
import urllib.parse  # noqa: E402


# Optional integrations — imported lazily to avoid errors if packages missing
def _notify(subject: str, body: str) -> None:
    try:
        from .notify import send_notification
        send_notification(subject, body)
    except Exception:
        pass


def _gcal_push(title: str, date: str, time: str = "", notes: str = "") -> dict:
    try:
        from .gcal import push_event, is_configured
        if is_configured():
            return push_event(title, date, time, notes)
    except Exception:
        pass
    return {}


# ── Calendar / events tools ───────────────────────────────────────────────────

def _event_add(title: str, date: str, time: str = "", notes: str = "") -> dict:
    with _lock:
        events = _load("events.json")
        event = {
            "title": title,
            "date":  date,
            "time":  time or "",
            "notes": notes or "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        events.append(event)
        events.sort(key=lambda e: (e["date"], e.get("time", "")))
        _save("events.json", events)
    gcal_result = _gcal_push(title, date, time, notes)
    result = {"added": event}
    if gcal_result.get("gcal_link"):
        result["gcal_link"] = gcal_result["gcal_link"]
    return result


def _events_list(days_ahead: int = 14) -> dict:
    with _lock:
        events = _load("events.json")
    today = datetime.now(timezone.utc).date()
    cutoff = today + timedelta(days=days_ahead)
    upcoming = [
        e for e in events
        if today <= datetime.strptime(e["date"], "%Y-%m-%d").date() <= cutoff
    ]
    return {"events": upcoming, "count": len(upcoming), "days_ahead": days_ahead}


def _event_delete(title: str, date: str) -> dict:
    with _lock:
        events = _load("events.json")
        before = len(events)
        events = [
            e for e in events
            if not (e["title"].lower() == title.lower() and e["date"] == date)
        ]
        _save("events.json", events)
    deleted = before - len(events)
    return {"deleted": deleted, "title": title, "date": date}


# ── Expense tools ─────────────────────────────────────────────────────────────

def _expense_add(amount: float, category: str, description: str = "") -> dict:
    with _lock:
        expenses = _load("expenses.json")
        entry = {
            "date":        datetime.now(timezone.utc).date().isoformat(),
            "amount":      round(amount, 2),
            "category":    category.lower().strip(),
            "description": description.strip(),
        }
        expenses.append(entry)
        _save("expenses.json", expenses)
    return {"logged": entry}


def _expenses_summary(days: int = 30) -> dict:
    with _lock:
        expenses = _load("expenses.json")
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
    recent = [e for e in expenses if e["date"] >= cutoff]
    totals: dict[str, float] = {}
    for e in recent:
        totals[e["category"]] = round(totals.get(e["category"], 0) + e["amount"], 2)
    grand_total = round(sum(totals.values()), 2)
    return {
        "period_days":  days,
        "total":        grand_total,
        "by_category":  dict(sorted(totals.items(), key=lambda x: -x[1])),
        "entry_count":  len(recent),
    }


# ── Book tracker tools ────────────────────────────────────────────────────────

_book_cache: dict[str, dict] = {}
_BOOK_CACHE_FILE = DATA_DIR / "book_cache.json"


def _load_book_cache() -> dict:
    if _BOOK_CACHE_FILE.exists():
        try:
            with open(_BOOK_CACHE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_book_cache(cache: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_BOOK_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _book_lookup(title: str) -> dict:
    """Look up book details via Open Library API (free, no key needed). Cached locally."""
    global _book_cache
    if not _book_cache:
        _book_cache = _load_book_cache()

    cache_key = title.lower().strip()
    if cache_key in _book_cache:
        print(f"[DEBUG] Book lookup cache hit: {cache_key}", flush=True)
        return _book_cache[cache_key]

    import urllib.request
    import time

    _OL_HEADERS = {"User-Agent": "MemGraphAgent/1.0"}
    MAX_RETRIES = 2

    for attempt in range(MAX_RETRIES + 1):
        try:
            url = (
                f"https://openlibrary.org/search.json"
                f"?q={urllib.parse.quote(title)}&limit=1"
                f"&fields=title,author_name,subject,first_publish_year,key"
            )
            req = urllib.request.Request(url, headers=_OL_HEADERS)
            with urllib.request.urlopen(req, timeout=12) as resp:  # noqa: S310
                data = json.loads(resp.read().decode())

            doc = data.get("docs", [{}])[0] if data.get("docs") else {}
            if not doc:
                return {}

            author = (doc.get("author_name") or [""])[0]
            subjects = [s.lower() for s in (doc.get("subject") or [])]
            subjects_text = " ".join(subjects)

            genre = ""
            genre_keywords = [
                ("Science Fiction", ["science fiction", "sci-fi", "scifi", "hard sci"]),
                ("Fantasy", ["fantasy", "epic fantasy", "high fantasy"]),
                ("Mystery", ["mystery", "detective", "whodunit"]),
                ("Thriller", ["thriller", "suspense"]),
                ("Romance", ["romance", "love story"]),
                ("Horror", ["horror"]),
                ("Biography", ["biography", "memoir", "autobiography"]),
                ("History", ["history", "historical"]),
                ("Self-Help", ["self-help", "self help", "personal development", "self-actualization"]),
                ("Non-Fiction", ["non-fiction", "nonfiction"]),
            ]
            for genre_name, keywords in genre_keywords:
                if any(kw in subjects_text for kw in keywords):
                    genre = genre_name
                    break

            # Fetch description from works endpoint
            summary = ""
            work_key = doc.get("key", "")
            if work_key:
                try:
                    work_url = f"https://openlibrary.org{work_key}.json"
                    work_req = urllib.request.Request(work_url, headers=_OL_HEADERS)
                    with urllib.request.urlopen(work_req, timeout=12) as work_resp:  # noqa: S310
                        work_data = json.loads(work_resp.read().decode())
                    desc = work_data.get("description", "")
                    if isinstance(desc, dict):
                        desc = desc.get("value", "")
                    if desc:
                        sentences = desc.replace("\r\n", " ").replace("\n", " ").split(". ")
                        summary = ". ".join(sentences[:3]).strip()
                        if not summary.endswith("."):
                            summary += "."
                except Exception:
                    pass

            result = {"author": author, "genre": genre, "summary": summary}
            _book_cache[cache_key] = result
            _save_book_cache(_book_cache)
            print(f"[DEBUG] Book lookup: author='{author}', genre='{genre}', summary={len(summary)} chars", flush=True)
            return result

        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"[DEBUG] Book lookup attempt {attempt+1} failed, retrying: {e}", flush=True)
                time.sleep(2)
            else:
                print(f"[DEBUG] Book lookup failed after {MAX_RETRIES+1} attempts: {e}", flush=True)
                return {}


def _book_add(title: str, author: str, genre: str, rating: float,
              start_date: str, end_date: str, summary: str, reader: str) -> dict:
    # Auto-enrich missing fields via Open Library
    if not author or not genre.strip() or not summary:
        lookup = _book_lookup(title)
        if not author and lookup.get("author"):
            author = lookup["author"]
        if not genre.strip() and lookup.get("genre"):
            genre = lookup["genre"]
        if not summary and lookup.get("summary"):
            summary = lookup["summary"]

    with _lock:
        books = _load("books.json")
        if not isinstance(books, list):
            books = []
        # Check for duplicate (case-insensitive title match)
        for b in books:
            if b["title"].lower() == title.lower():
                return {"error": f"Book '{title}' already exists. Use book_update to modify it."}
        book = {
            "title":      title,
            "author":     author or "",
            "genre":      genre.strip().title() if genre.strip() else "Other",
            "rating":     min(10, max(0, round(rating, 1))),
            "start_date": start_date or "",
            "end_date":   end_date or "",
            "summary":    summary,
            "reader":     reader or "",
            "status":     "pending",
            "added_at":   datetime.now(timezone.utc).isoformat(),
        }
        books.append(book)
        books.sort(key=lambda b: b.get("added_at", ""), reverse=True)
        _save("books.json", books)
    return {"added": book}


def _book_list(genre: str = "", reader: str = "") -> dict:
    with _lock:
        books = _load("books.json")
    if not isinstance(books, list):
        books = []
    if genre:
        books = [b for b in books if b.get("genre", "").lower() == genre.lower()]
    if reader:
        books = [b for b in books if b.get("reader", "").lower() == reader.lower()]
    # Group by genre
    grouped: dict[str, list] = {}
    for b in books:
        g = b.get("genre", "Other")
        grouped.setdefault(g, []).append(b)
    return {"books": grouped, "total": sum(len(v) for v in grouped.values())}


def _book_update(title: str, updates: dict) -> dict:
    with _lock:
        books = _load("books.json")
        if not isinstance(books, list):
            return {"error": "No books found"}
        found = None
        for b in books:
            if b["title"].lower() == title.lower():
                found = b
                break
        if not found:
            return {"error": f"Book '{title}' not found"}
        # Apply updates
        _CLEARABLE_FIELDS = {"reader", "status"}
        for key in ("author", "genre", "rating", "start_date", "end_date", "summary", "reader", "status"):
            if key not in updates:
                continue
            val = updates[key]
            if val in (None, "", "None") and key not in _CLEARABLE_FIELDS:
                continue
            if key == "rating":
                found[key] = min(10, max(0, round(float(val), 1)))
            elif key == "genre":
                found[key] = str(val).strip().title()
            else:
                found[key] = str(val) if val else ""
        _save("books.json", books)
    return {"updated": found}


def _book_delete(title: str) -> dict:
    with _lock:
        books = _load("books.json")
        if not isinstance(books, list):
            return {"error": "No books found"}
        before = len(books)
        books = [b for b in books if b["title"].lower() != title.lower()]
        if len(books) == before:
            return {"error": f"Book '{title}' not found"}
        _save("books.json", books)
    return {"deleted": title}
