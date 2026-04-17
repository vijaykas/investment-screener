"""
============================================================
  INVESTMENT INTELLIGENCE — ML Score Enhancer
  Phase 6e: Multi-Signal Ensemble Score Adjustment
  Author: Claude (Vijay's Investment Project)
  Updated: 2026-04-15
============================================================

WHAT THIS DOES:
  Takes the base technical score (0–75) from the daily monitor
  and layers on Phase 5/6 signals to produce an Enhanced Score
  that incorporates fundamental quality, insider sentiment,
  options positioning, news sentiment, and sector rotation.

SCORING LAYERS (max additive adjustment: +30 pts):
  Fundamentals  +0–10 pts  — quality companies score higher
  Insider Net    -5–+5 pts  — insiders buy = bullish, sell = bearish
  Options P/C    -5–+3 pts  — heavy puts = headwind, call bias = tailwind
  News Sentiment -4–+4 pts  — bullish/bearish news flow
  Sector Rotation -5–+5 pts  — rotating into vs out of sector

FINAL ENHANCED SCORE:
  Capped at 100. Values above 75 represent stocks with strong
  technical + fundamental + sentiment alignment.

  ≥ 85 → 🟢 CONVICTION BUY
  ≥ 72 → 🔵 STRONG BUY
  ≥ 58 → 💙 BUY
  ≥ 44 → 🟡 HOLD
  ≥ 30 → 🟠 CAUTION
  < 30  → 🔴 AVOID
============================================================
"""

from __future__ import annotations


# ─────────────────────────────────────────────
#  SECTOR → SECTOR ETF MAP
#  Maps daily_monitor sector labels to sector rotation ETF tickers
# ─────────────────────────────────────────────

_SECTOR_TO_ETF = {
    "Tech":           "XLK",
    "AI/Growth":      "XLK",
    "Finance":        "XLF",
    "Healthcare":     "XLV",
    "Consumer":       "XLY",
    "Energy":         "XLE",
    "Industrial":     "XLI",
    "Dividend":       "XLP",   # staples proxy
    "ETF":            None,
    # GICS sector labels (used when watchlist_mode="sp500")
    "Information Technology":     "XLK",
    "Financials":                  "XLF",
    "Health Care":                 "XLV",
    "Consumer Discretionary":      "XLY",
    "Consumer Staples":            "XLP",
    "Energy":                      "XLE",
    "Industrials":                 "XLI",
    "Materials":                   "XLB",
    "Utilities":                   "XLU",
    "Real Estate":                 "XLRE",
    "Communication Services":      "XLC",
}


def _fundamental_adj(fundamental_data: dict, ticker: str) -> tuple[float, str]:
    """
    Convert fundamental_score (0–25) into a score adjustment (-2 to +10).
    Returns (adjustment, explanation).
    """
    if not fundamental_data or ticker not in fundamental_data:
        return 0.0, ""
    fs = fundamental_data[ticker].get("fundamental_score", 0)
    # Linear map: 0 pts → -2, 12 pts → +5, 25 pts → +10
    adj = round((fs / 25) * 12 - 2, 1)
    adj = max(-2.0, min(10.0, adj))
    return adj, f"Fund {fs}/25 → {adj:+.0f}"


def _insider_adj(insider_data: dict, ticker: str) -> tuple[float, str]:
    """
    Convert insider net signal into a score adjustment (-5 to +5).
    """
    if not insider_data or ticker not in insider_data:
        return 0.0, ""
    signal = insider_data[ticker].get("signal", "")
    adj_map = {
        "🟢 Strong Buy":   5.0,
        "🔵 Buy":          2.5,
        "⚪ Neutral":      0.0,
        "🟠 Sell":        -2.5,
        "🔴 Strong Sell": -5.0,
    }
    for label, adj in adj_map.items():
        if label in signal:
            return adj, f"Insider {label} → {adj:+.0f}"
    return 0.0, ""


def _options_adj(options_data: dict, ticker: str) -> tuple[float, str]:
    """
    Convert P/C ratio into a score adjustment (-5 to +3).
    Heavy puts = headwind. Strong call bias = tailwind.
    """
    if not options_data or ticker not in options_data:
        return 0.0, ""
    vol_pc = options_data[ticker].get("vol_pc")
    if vol_pc is None:
        return 0.0, ""

    if vol_pc > 1.5:
        adj = -5.0
        label = f"P/C {vol_pc:.2f} (heavy puts)"
    elif vol_pc > 1.0:
        adj = -2.5
        label = f"P/C {vol_pc:.2f} (elevated puts)"
    elif vol_pc > 0.7:
        adj = 0.0
        label = f"P/C {vol_pc:.2f} (neutral)"
    elif vol_pc > 0.5:
        adj = 1.5
        label = f"P/C {vol_pc:.2f} (call bias)"
    else:
        adj = 3.0
        label = f"P/C {vol_pc:.2f} (strong calls)"

    return adj, f"Options {label} → {adj:+.0f}"


def _sentiment_adj(sentiment_data: dict, ticker: str) -> tuple[float, str]:
    """
    Convert news sentiment score (-1..+1) into a score adjustment (-4 to +4).
    """
    if not sentiment_data or ticker not in sentiment_data:
        return 0.0, ""
    score = sentiment_data[ticker].get("score", 0)
    label = sentiment_data[ticker].get("label", "")
    # Linear: -1 → -4, 0 → 0, +1 → +4
    adj = round(score * 4, 1)
    adj = max(-4.0, min(4.0, adj))
    return adj, f"Sentiment {label} → {adj:+.0f}"


def _sector_adj(sector_rotation: dict, sector_map: dict, ticker: str) -> tuple[float, str]:
    """
    Add/subtract points based on sector rotation signal.
    ROTATE IN = +5, HOLD = 0, ROTATE OUT = -3, AVOID = -5.
    """
    if not sector_rotation or not sector_map:
        return 0.0, ""

    sector   = sector_map.get(ticker, "")
    etf_key  = _SECTOR_TO_ETF.get(sector)
    if etf_key is None:
        return 0.0, ""

    sectors_data = sector_rotation.get("sectors", {})
    if etf_key not in sectors_data:
        return 0.0, ""

    signal = sectors_data[etf_key].get("signal", "")
    adj_map = {
        "ROTATE IN":  5.0,
        "HOLD":       0.0,
        "NEUTRAL":    0.0,
        "ROTATE OUT": -3.0,
        "AVOID":      -5.0,
    }
    for key, adj in adj_map.items():
        if key in signal.upper():
            return adj, f"Sector ({etf_key}) {signal} → {adj:+.0f}"
    return 0.0, ""


# ─────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────

def compute_enhanced_scores(
    today_scores:     dict,
    sector_map:       dict,
    fundamental_data: dict = None,
    insider_data:     dict = None,
    options_data:     dict = None,
    sentiment_data:   dict = None,
    sector_rotation:  dict = None,
) -> dict:
    """
    Compute an enhanced composite score for each ticker by layering
    Phase 5/6 signals on top of the base technical score.

    Parameters
    ----------
    today_scores      {ticker: {"score": int, "signal": str, ...}}  — from daily_monitor
    sector_map        {ticker: sector_label}
    fundamental_data  {ticker: {"fundamental_score": int, ...}}
    insider_data      {ticker: {"signal": str, ...}, "_rows": [...]}
    options_data      {ticker: {"vol_pc": float, ...}}
    sentiment_data    {ticker: {"score": float, "label": str, ...}}
    sector_rotation   {"sectors": {ETF: {"signal": str, ...}}, ...}

    Returns
    -------
    {ticker: {
        "enhanced_score":   int,         # 0-100
        "base_score":       int,         # original tech score
        "total_adj":        float,       # net adjustment applied
        "enhanced_signal":  str,         # signal label based on enhanced_score
        "factors":          list[str],   # human-readable adjustment factors
    }}
    """
    fundamental_data = fundamental_data or {}
    insider_data     = insider_data     or {}
    options_data     = options_data     or {}
    sentiment_data   = sentiment_data   or {}
    sector_rotation  = sector_rotation  or {}

    result = {}

    for ticker, s in today_scores.items():
        base = s.get("score", 0)
        factors = []

        f_adj, f_lbl = _fundamental_adj(fundamental_data, ticker)
        i_adj, i_lbl = _insider_adj(insider_data, ticker)
        o_adj, o_lbl = _options_adj(options_data, ticker)
        s_adj, s_lbl = _sentiment_adj(sentiment_data, ticker)
        r_adj, r_lbl = _sector_adj(sector_rotation, sector_map, ticker)

        for lbl in (f_lbl, i_lbl, o_lbl, s_lbl, r_lbl):
            if lbl:
                factors.append(lbl)

        total_adj = round(f_adj + i_adj + o_adj + s_adj + r_adj, 1)
        raw_enhanced = base + total_adj
        enhanced = int(min(100, max(0, round(raw_enhanced))))

        # Signal label based on enhanced score
        if   enhanced >= 85: signal = "🟢 CONVICTION BUY"
        elif enhanced >= 72: signal = "🟢 STRONG BUY"
        elif enhanced >= 58: signal = "🔵 BUY"
        elif enhanced >= 44: signal = "🟡 HOLD"
        elif enhanced >= 30: signal = "🟠 CAUTION"
        else:                signal = "🔴 AVOID"

        result[ticker] = {
            "enhanced_score":  enhanced,
            "base_score":      base,
            "total_adj":       total_adj,
            "enhanced_signal": signal,
            "factors":         factors,
        }

    return result
