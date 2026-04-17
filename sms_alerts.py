"""
============================================================
  INVESTMENT INTELLIGENCE — SMS Alert System
  Phase 6d: Critical Event SMS via Twilio
  Author: Claude (Vijay's Investment Project)
  Updated: 2026-04-15
============================================================

WHAT THIS DOES:
  Sends concise SMS alerts for critical market events that can't
  wait until the next email report.

  Alerts are batched and rate-limited (max 5 SMS per run) to
  avoid notification fatigue. Only genuine thresholds trigger.

ALERT TRIGGERS (configurable in config.py SMS_CONFIG):
  📈 Big Move Up      Stock surges > MOVE_UP_PCT (default 6%)
  📉 Big Move Down    Stock drops  > MOVE_DOWN_PCT (default 5%)
  🔴 RSI Extreme      RSI > 82 (very overbought) or < 22 (very oversold)
  🏛️ Insider Buy      Single insider purchase > INSIDER_BUY_THRESHOLD ($500k)
  📅 Earnings Today   Stock reports earnings today
  ⚠️ Squeeze Setup    High short + low P/C ratio (squeeze imminent)
  🌐 Macro Alert      Macro health score < 20 (risk-off emergency)

SETUP:
  1. Create a free Twilio account at twilio.com
  2. Get a free trial phone number (can send SMS)
  3. Add these GitHub Secrets:
       TWILIO_ACCOUNT_SID   — Your Twilio Account SID
       TWILIO_AUTH_TOKEN    — Your Twilio Auth Token
       TWILIO_FROM_NUMBER   — Your Twilio phone number (+1XXXXXXXXXX)
       SMS_TO_NUMBER        — Your mobile number to receive alerts (+1XXXXXXXXXX)

RATE LIMITS:
  Twilio free trial: ~15 SMS per month (more than enough for alerts)
  This module caps at MAX_SMS_PER_RUN (default 5) per run to be safe.
============================================================
"""

import os
import json
from datetime import datetime, date

# ── Configuration defaults (override in config.py SMS_CONFIG) ──
DEFAULT_CONFIG = {
    "move_up_pct":          6.0,    # % surge to trigger alert
    "move_down_pct":        5.0,    # % drop to trigger alert
    "rsi_overbought_alert": 82,     # very overbought
    "rsi_oversold_alert":   22,     # very oversold
    "insider_buy_threshold":500_000, # $500k+ single purchase
    "macro_alert_threshold": 20,    # macro score below this
    "max_sms_per_run":       5,     # hard cap
}


def _get_config() -> dict:
    """Load SMS config from config.py if available, else use defaults."""
    try:
        from config import SMS_CONFIG
        return {**DEFAULT_CONFIG, **SMS_CONFIG}
    except (ImportError, AttributeError):
        return DEFAULT_CONFIG


def _send_sms(account_sid: str, auth_token: str, from_num: str,
              to_num: str, body: str) -> bool:
    """Send a single SMS via Twilio REST API."""
    import requests
    try:
        r = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json",
            auth=(account_sid, auth_token),
            data={
                "From": from_num,
                "To":   to_num,
                "Body": body[:160],   # SMS limit
            },
            timeout=10,
        )
        return r.status_code == 201
    except Exception as e:
        print(f"      ⚠️  SMS send failed: {e}")
        return False


def _format_alert(emoji: str, ticker: str, detail: str) -> str:
    """Format a concise SMS alert message (< 160 chars)."""
    now = datetime.now().strftime("%H:%M")
    msg = f"[{now}] {emoji} {ticker}: {detail}"
    return msg[:160]


def send_sms_alerts(
    today_scores: dict,
    events: list,
    earnings_data: list,
    insider_data: dict,
    options_data: dict,
    macro_data: dict,
    fundamental_data: dict,
) -> list:
    """
    Evaluate all alert triggers and send SMS for critical events.

    Returns list of alert dicts that were sent (for report logging).
    Silently skips if Twilio env vars are not set.
    """
    # ── Read credentials from environment ────
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    auth_token  = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    from_num    = os.environ.get("TWILIO_FROM_NUMBER", "").strip()
    to_num      = os.environ.get("SMS_TO_NUMBER", "").strip()

    if not all([account_sid, auth_token, from_num, to_num]):
        print("  📱 SMS skipped — Twilio credentials not set.")
        return []

    cfg     = _get_config()
    alerts  = []   # list of {ticker, type, message} to send
    sent    = []

    # ── 1. Big Price Moves ────────────────────
    for ticker, s in today_scores.items():
        chg = s.get("day_chg_pct", 0)
        if chg >= cfg["move_up_pct"]:
            alerts.append({
                "ticker":  ticker,
                "type":    "BIG MOVE UP",
                "priority":1,
                "message": _format_alert(
                    "📈", ticker,
                    f"+{chg:.1f}% surge — Score {s['score']}/75 {s['signal']}"
                ),
            })
        elif chg <= -cfg["move_down_pct"]:
            alerts.append({
                "ticker":  ticker,
                "type":    "BIG MOVE DOWN",
                "priority":1,
                "message": _format_alert(
                    "📉", ticker,
                    f"{chg:.1f}% drop — Score {s['score']}/75 {s['signal']}"
                ),
            })

    # ── 2. RSI Extremes ───────────────────────
    for ticker, s in today_scores.items():
        rsi = s.get("rsi", 50)
        if rsi >= cfg["rsi_overbought_alert"]:
            alerts.append({
                "ticker":  ticker,
                "type":    "RSI EXTREME HIGH",
                "priority":2,
                "message": _format_alert("🔴", ticker, f"RSI {rsi:.0f} — very overbought"),
            })
        elif rsi <= cfg["rsi_oversold_alert"]:
            alerts.append({
                "ticker":  ticker,
                "type":    "RSI EXTREME LOW",
                "priority":2,
                "message": _format_alert("🟢", ticker, f"RSI {rsi:.0f} — deeply oversold"),
            })

    # ── 3. Earnings TODAY ─────────────────────
    for e in earnings_data:
        if e.get("days_until", 99) == 0:
            alerts.append({
                "ticker":  e["ticker"],
                "type":    "EARNINGS TODAY",
                "priority":1,
                "message": _format_alert("📅", e["ticker"], "REPORTS TODAY"),
            })

    # ── 4. Large Insider Buys ─────────────────
    for row in insider_data.get("_rows", []):
        if (row.get("type") == "BUY"
                and row.get("value", 0) >= cfg["insider_buy_threshold"]):
            val_str = f"${row['value']/1e6:.1f}M" if row['value'] >= 1e6 else f"${row['value']/1e3:.0f}k"
            alerts.append({
                "ticker":  row["ticker"],
                "type":    "INSIDER BUY",
                "priority":2,
                "message": _format_alert(
                    "🏛️", row["ticker"],
                    f"Insider bought {val_str} — {row['name'][:20]}"
                ),
            })

    # ── 5. Squeeze Setups ─────────────────────
    for ev in events:
        if "SQUEEZE" in ev.get("type", "").upper():
            alerts.append({
                "ticker":  ev["ticker"],
                "type":    "SQUEEZE SETUP",
                "priority":2,
                "message": _format_alert("⚡", ev["ticker"], ev["detail"][:80]),
            })

    # ── 6. Macro Emergency ───────────────────
    macro_score = macro_data.get("macro_score", 100)
    if macro_score < cfg["macro_alert_threshold"]:
        alerts.append({
            "ticker":  "MACRO",
            "type":    "MACRO ALERT",
            "priority":1,
            "message": _format_alert(
                "🌐", "MACRO",
                f"Health score {macro_score}/100 — {macro_data.get('macro_signal','')}"
            ),
        })

    if not alerts:
        print("  📱 SMS: no critical alerts triggered.")
        return []

    # ── Sort by priority and deduplicate ──────
    seen_tickers = set()
    deduped = []
    for a in sorted(alerts, key=lambda x: x["priority"]):
        if a["ticker"] not in seen_tickers or a["ticker"] == "MACRO":
            deduped.append(a)
            seen_tickers.add(a["ticker"])

    # ── Cap at max_sms_per_run ────────────────
    to_send = deduped[:cfg["max_sms_per_run"]]

    print(f"  📱 Sending {len(to_send)} SMS alert(s) (of {len(alerts)} triggered)...")

    for alert in to_send:
        success = _send_sms(account_sid, auth_token, from_num, to_num, alert["message"])
        if success:
            print(f"      ✅ SMS sent: {alert['ticker']} — {alert['type']}")
            sent.append(alert)
        else:
            print(f"      ❌ SMS failed: {alert['ticker']}")

    # If more alerts than cap, note it
    skipped = len(deduped) - len(to_send)
    if skipped > 0:
        summary_msg = _format_alert(
            "📊", "SUMMARY",
            f"+{skipped} more alerts not sent (cap {cfg['max_sms_per_run']}/run)"
        )
        _send_sms(account_sid, auth_token, from_num, to_num, summary_msg)

    return sent
