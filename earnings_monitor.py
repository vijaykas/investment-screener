"""
============================================================
  INVESTMENT INTELLIGENCE — Earnings Calendar Monitor
  Phase 5a: Upcoming Earnings Alerts
  Author: Claude (Vijay's Investment Project)
  Updated: 2026-04-15
============================================================

WHAT THIS DOES:
  Fetches upcoming earnings dates for all watchlist stocks
  using yfinance (no API key required). Flags stocks reporting
  in the next 14 days so you can position before the move.

HOW TO USE:
  Automatically called by daily_monitor.py each run.
  Results appear in the Earnings Calendar section of the report.
============================================================
"""

import yfinance as yf
import json
import os
from datetime import date, datetime, timedelta

# Cache file path — reuse existing data/ folder
_CACHE_FILE = None   # set in fetch_earnings_calendar()
_CACHE_HOURS = 12    # refresh after 12 hours


def _load_cache(cache_path: str) -> dict:
    try:
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            age_h = (datetime.now().timestamp() - mtime) / 3600
            if age_h < _CACHE_HOURS:
                with open(cache_path) as f:
                    return json.load(f)
    except Exception:
        pass
    return {}


def _save_cache(cache_path: str, data: dict):
    try:
        with open(cache_path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def _parse_earnings_date(cal) -> date | None:
    """
    yfinance.Ticker.calendar can return:
      - None / empty
      - A dict: {'Earnings Date': [Timestamp(...)], ...}
      - A DataFrame (older yfinance)
    Returns the earliest future earnings date as a date object, or None.
    """
    if cal is None:
        return None

    today = date.today()

    # ── dict format (yfinance ≥ 0.2.x) ──────────────────────
    if isinstance(cal, dict):
        raw = cal.get("Earnings Date")
        if raw is None:
            return None
        if not hasattr(raw, "__iter__"):
            raw = [raw]
        dates = []
        for item in raw:
            try:
                d = item.date() if hasattr(item, "date") else item
                if isinstance(d, date) and d >= today:
                    dates.append(d)
            except Exception:
                continue
        return min(dates) if dates else None

    # ── DataFrame format (older yfinance) ────────────────────
    try:
        import pandas as pd
        if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            row = cal.loc["Earnings Date"]
            raw = row.values if hasattr(row, "values") else [row]
            dates = []
            for item in raw:
                try:
                    if pd.isnull(item):
                        continue
                    d = item.date() if hasattr(item, "date") else item
                    if isinstance(d, date) and d >= today:
                        dates.append(d)
                except Exception:
                    continue
            return min(dates) if dates else None
    except Exception:
        pass

    return None


def fetch_earnings_calendar(tickers: list, data_dir: str, days_ahead: int = 14) -> list:
    """
    Fetch upcoming earnings for a list of tickers.

    Returns a list of dicts sorted by days_until:
      {
        ticker       : str,
        date         : str   (YYYY-MM-DD),
        days_until   : int,
        eps_estimate : float | None,
        note         : str   ("TODAY" / "TOMORROW" / "In N days"),
      }

    Caches results to data_dir/earnings_cache.json for _CACHE_HOURS.
    Silently skips tickers that fail.
    """
    global _CACHE_FILE
    _CACHE_FILE = os.path.join(data_dir, "earnings_cache.json")
    cached = _load_cache(_CACHE_FILE)

    today  = date.today()
    cutoff = today + timedelta(days=days_ahead)
    today_str = today.strftime("%Y-%m-%d")

    # Re-use cache if same calendar day
    if cached.get("_date") == today_str:
        return cached.get("earnings", [])

    upcoming = []
    for ticker in tickers:
        try:
            t   = yf.Ticker(ticker)
            cal = t.calendar
            ed  = _parse_earnings_date(cal)
            if ed is None or ed > cutoff:
                continue

            days_until = (ed - today).days

            # Try to get EPS estimate
            eps_est = None
            try:
                if isinstance(cal, dict):
                    raw_eps = cal.get("EPS Estimate")
                    if raw_eps is not None:
                        vals = list(raw_eps) if hasattr(raw_eps, "__iter__") else [raw_eps]
                        for v in vals:
                            try:
                                eps_est = float(v)
                                break
                            except (TypeError, ValueError):
                                continue
            except Exception:
                pass

            if days_until == 0:
                note = "🔴 TODAY"
            elif days_until == 1:
                note = "🟠 TOMORROW"
            elif days_until <= 3:
                note = f"🟡 In {days_until} days"
            else:
                note = f"In {days_until} days"

            upcoming.append({
                "ticker":       ticker,
                "date":         ed.strftime("%Y-%m-%d"),
                "days_until":   days_until,
                "eps_estimate": eps_est,
                "note":         note,
            })
        except Exception:
            continue

    upcoming.sort(key=lambda x: x["days_until"])

    # Save cache
    _save_cache(_CACHE_FILE, {"_date": today_str, "earnings": upcoming})

    return upcoming
