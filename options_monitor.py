"""
============================================================
  INVESTMENT INTELLIGENCE — Options Put/Call Monitor
  Phase 6a: Options Flow Signals
  Author: Claude (Vijay's Investment Project)
  Updated: 2026-04-15
============================================================

WHAT THIS DOES:
  Fetches options chain data for watchlist stocks via yfinance
  (no API key required) and computes put/call ratios on both
  volume (current activity) and open interest (persistent positioning).

  A high P/C ratio means traders are buying protection (bearish).
  A low P/C ratio means complacency or bullish positioning.

PUT/CALL RATIO SIGNALS:
  Volume P/C Ratio (today's activity):
    🔴 > 1.5  — Heavy put buying, bearish sentiment
    🟠 1.0–1.5 — Elevated puts, caution
    ⚪ 0.7–1.0 — Neutral
    🔵 0.5–0.7 — Mild call bias, bullish lean
    🟢 < 0.5  — Strong call bias (bullish or overconfident)

  Open Interest P/C Ratio (accumulated positioning):
    Same scale — but reflects institutional positioning over weeks/months.

SQUEEZE SETUP DETECTOR:
  High short interest + low P/C ratio = potential short squeeze setup
  (shorts are exposed with little put protection)

HOW IT'S USED:
  • Added as a signal column in the daily report
  • Extreme readings flagged as events
  • Fed into the Enhanced ML Score as a contrarian/confirming signal
============================================================
"""

import yfinance as yf
import json
import os
from datetime import datetime, date, timedelta

_CACHE_HOURS = 4   # options data: refresh every 4 hours (intra-day moves matter)
_EXPIRY_MIN_DAYS = 7    # minimum days to expiry (avoid 0DTE noise)
_EXPIRY_MAX_DAYS = 45   # maximum days to expiry (focus on near-term positioning)


def _load_cache(path: str) -> dict:
    try:
        if os.path.exists(path):
            age_h = (datetime.now().timestamp() - os.path.getmtime(path)) / 3600
            if age_h < _CACHE_HOURS:
                with open(path) as f:
                    return json.load(f)
    except Exception:
        pass
    return {}


def _save_cache(path: str, data: dict):
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def _pc_signal(ratio: float | None, kind: str = "vol") -> str:
    """Return emoji signal label for a put/call ratio."""
    if ratio is None:
        return "—"
    if   ratio > 1.5:  return f"🔴 {ratio:.2f} (heavy puts)"
    elif ratio > 1.0:  return f"🟠 {ratio:.2f} (elevated puts)"
    elif ratio > 0.7:  return f"⚪ {ratio:.2f} (neutral)"
    elif ratio > 0.5:  return f"🔵 {ratio:.2f} (call bias)"
    else:              return f"🟢 {ratio:.2f} (strong calls)"


def _fetch_ticker_options(ticker: str) -> dict:
    """
    Fetch options chain for a single ticker and compute P/C ratios.
    Returns dict with vol_pc, oi_pc, and signal labels, or {} on failure.
    """
    today = date.today()
    try:
        t = yf.Ticker(ticker)

        # Get expiration dates
        expirations = t.options
        if not expirations:
            return {}

        # Filter to near-term expirations only
        valid_expiries = []
        for exp in expirations:
            try:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                days = (exp_date - today).days
                if _EXPIRY_MIN_DAYS <= days <= _EXPIRY_MAX_DAYS:
                    valid_expiries.append(exp)
            except ValueError:
                continue

        if not valid_expiries:
            # Fallback: use the nearest expiry regardless
            valid_expiries = [expirations[0]] if expirations else []

        if not valid_expiries:
            return {}

        # Aggregate across all valid expirations
        total_call_vol = 0
        total_put_vol  = 0
        total_call_oi  = 0
        total_put_oi   = 0

        for exp in valid_expiries[:3]:   # cap at 3 expirations for speed
            try:
                chain = t.option_chain(exp)
                calls = chain.calls
                puts  = chain.puts

                total_call_vol += int(calls["volume"].fillna(0).sum())
                total_put_vol  += int(puts["volume"].fillna(0).sum())
                total_call_oi  += int(calls["openInterest"].fillna(0).sum())
                total_put_oi   += int(puts["openInterest"].fillna(0).sum())
            except Exception:
                continue

        vol_pc = round(total_put_vol / total_call_vol, 3) if total_call_vol > 0 else None
        oi_pc  = round(total_put_oi  / total_call_oi,  3) if total_call_oi  > 0 else None

        # Squeeze setup: high short interest + low P/C = dangerous for shorts
        squeeze_flag = False
        if vol_pc is not None and vol_pc < 0.5:
            squeeze_flag = True   # confirmed by caller if short interest also high

        return {
            "vol_pc":       vol_pc,
            "oi_pc":        oi_pc,
            "vol_pc_signal":_pc_signal(vol_pc),
            "oi_pc_signal": _pc_signal(oi_pc, "oi"),
            "call_vol":     total_call_vol,
            "put_vol":      total_put_vol,
            "call_oi":      total_call_oi,
            "put_oi":       total_put_oi,
            "squeeze_setup":squeeze_flag,
            "expiries_used":len(valid_expiries[:3]),
        }

    except Exception:
        return {}


def fetch_options_data(tickers: list, data_dir: str) -> dict:
    """
    Fetch options put/call data for all tickers.

    Returns dict: {ticker: {vol_pc, oi_pc, vol_pc_signal, oi_pc_signal,
                             call_vol, put_vol, squeeze_setup}}

    Caches to data_dir/options_cache.json for _CACHE_HOURS hours.
    """
    cache_path = os.path.join(data_dir, "options_cache.json")
    cached     = _load_cache(cache_path)
    today_str  = date.today().strftime("%Y-%m-%d")

    # Use today's cache if fresh
    if cached.get("_date") == today_str and len(cached) > 5:
        print(f"  📈 Options P/C loaded from cache ({len(cached)-1} tickers).")
        return {k: v for k, v in cached.items() if k != "_date"}

    print(f"  📈 Fetching options chains for {len(tickers)} tickers...")
    result = {}
    success = 0

    for i, ticker in enumerate(tickers):
        data = _fetch_ticker_options(ticker)
        if data:
            result[ticker] = data
            success += 1

    print(f"  ✅ Options P/C complete: {success}/{len(tickers)} tickers with data.")

    _save_cache(cache_path, {"_date": today_str, **result})
    return result


def get_options_events(options_data: dict, fundamental_data: dict = None) -> list:
    """
    Scan options data for notable signals to flag as events.
    Returns list of event dicts compatible with the existing events format.
    """
    events = []
    fund = fundamental_data or {}

    for ticker, opt in options_data.items():
        vol_pc = opt.get("vol_pc")
        if vol_pc is None:
            continue

        # Heavy put buying
        if vol_pc > 1.5:
            events.append({
                "ticker": ticker,
                "type":   "OPTIONS",
                "detail": f"P/C ratio {vol_pc:.2f} — heavy put buying, bearish hedge",
            })

        # Strong call bias — could be complacency or genuine bullishness
        elif vol_pc < 0.4:
            short_pct = fund.get(ticker, {}).get("short_pct") or 0
            if short_pct > 0.10:   # High short + low P/C = squeeze setup
                events.append({
                    "ticker": ticker,
                    "type":   "SQUEEZE SETUP",
                    "detail": (f"P/C {vol_pc:.2f} + short {short_pct*100:.1f}% float "
                               f"— potential short squeeze"),
                })
            else:
                events.append({
                    "ticker": ticker,
                    "type":   "OPTIONS",
                    "detail": f"P/C ratio {vol_pc:.2f} — strong call bias / bullish",
                })

    return events
