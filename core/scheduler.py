"""
Background scheduler for periodic household tasks.

Jobs:
  - Weekly expenses report (every Sunday at 09:00 local time)
    Generates a Markdown report, saves to data/reports/, sends email notification.

Requires: pip install apscheduler
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))


def generate_weekly_report() -> str:
    """Generate a weekly expenses report, save it, and send email notification.
    Returns the report text."""
    from .notify import send_notification

    # Load expenses from last 7 days
    expenses_file = DATA_DIR / "expenses.json"
    if not expenses_file.exists():
        logger.info("Weekly report: no expenses file found, skipping")
        return ""

    with open(expenses_file, encoding="utf-8") as f:
        all_expenses = json.load(f)

    today   = datetime.now(timezone.utc).date()
    cutoff  = (today - timedelta(days=7)).isoformat()
    week_expenses = [e for e in all_expenses if e.get("date", "") >= cutoff]

    # Build report
    week_str  = today.strftime("%Y-%m-%d")
    report_lines = [
        f"# Household Expenses Report — Week ending {week_str}",
        "",
    ]

    if not week_expenses:
        report_lines.append("No expenses recorded this week.")
    else:
        # Totals by category
        totals: dict[str, float] = {}
        for e in week_expenses:
            cat = e.get("category", "other")
            totals[cat] = round(totals.get(cat, 0) + e.get("amount", 0), 2)
        grand_total = round(sum(totals.values()), 2)

        report_lines += [
            f"**Total: {grand_total}**",
            "",
            "## By Category",
            "",
        ]
        for cat, amt in sorted(totals.items(), key=lambda x: -x[1]):
            report_lines.append(f"- {cat.capitalize()}: {amt}")

        report_lines += [
            "",
            "## Transactions",
            "",
            "| Date | Amount | Category | Description |",
            "|------|--------|----------|-------------|",
        ]
        for e in sorted(week_expenses, key=lambda x: x.get("date", "")):
            desc = e.get("description", "")
            report_lines.append(
                f"| {e['date']} | {e['amount']} | {e.get('category', '')} | {desc} |"
            )

    report_text = "\n".join(report_lines)

    # Save report to data/reports/
    reports_dir = DATA_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"expenses_{week_str}.md"
    report_file.write_text(report_text, encoding="utf-8")
    logger.info("Weekly report saved to %s", report_file)

    # Send email notification
    subject = f"Household Expenses — Week ending {week_str}"
    if week_expenses:
        summary_lines = [f"Total this week: {grand_total}", "", "By category:"]
        for cat, amt in sorted(totals.items(), key=lambda x: -x[1]):
            summary_lines.append(f"  {cat.capitalize()}: {amt}")
        email_body = "\n".join(summary_lines)
    else:
        email_body = "No expenses recorded this week."

    send_notification(subject, email_body)

    return report_text


def start_scheduler() -> None:
    """Start the APScheduler background scheduler. Call once at app startup."""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        logger.warning(
            "APScheduler not installed — weekly reports disabled. "
            "Run: pip install apscheduler"
        )
        return

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        generate_weekly_report,
        trigger="cron",
        day_of_week="sun",
        hour=9,
        minute=0,
        id="weekly_expenses_report",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler started — weekly report runs every Sunday at 09:00")
