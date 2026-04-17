"""
============================================================
  INVESTMENT INTELLIGENCE — Fundamental Scoring
  Phase 5c + 5d: Short Interest + Fundamental Layer
  Author: Claude (Vijay's Investment Project)
  Updated: 2026-04-15
============================================================

WHAT THIS DOES:
  Pulls per-ticker fundamental data from yfinance.info and
  builds a composite Fundamental Score (0-25 pts) that
  complements the existing technical score (0-75 pts).

  Also extracts Short Interest (% of float shorted) which
  is used both as a risk flag and an opportunity signal
  for potential short squeezes.

FUNDAMENTAL SCORE BREAKDOWN (0-25 pts):
  Valuation   (0-8 pts)  — Forward P/E vs sector, P/B ratio
  Profitability (0-8 pts) — Net margin, ROE
  Growth      (0-5 pts)  — Revenue & earnings growth YoY
  Balance Sheet (0-4 pts) — Debt-to-equity, current ratio

SHORT INTEREST FLAGS:
  🔴 Very High  → ≥ 20% of float shorted (squeeze risk / bearish)
  🟠 High       → ≥ 10% of float shorted
  ⚪ Normal     → < 10% of float shorted
  🟢 Low        → < 3% (institutional confidence)
============================================================
"""

import yfinance as yf
import json
import os
from datetime import datetime

_CACHE_HOURS = 12    # fundamentals don't change intra-day


# ─────────────────────────────────────────────
#  CACHE HELPERS
# ─────────────────────────────────────────────

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
            json.dump(data, f, default=str)
    except Exception:
        pass


# ─────────────────────────────────────────────
#  SAFE FLOAT HELPER
# ─────────────────────────────────────────────

def _f(val, default=None):
    """Safely convert to float; return default on None/NaN/error."""
    try:
        v = float(val)
        if v != v:   # NaN check
            return default
        return v
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────
#  FUNDAMENTAL SCORE  (0-25 pts)
# ─────────────────────────────────────────────

def _compute_fundamental_score(info: dict) -> int:
    """
    Score fundamentals 0-25.
    All inputs are optional — missing data scores 0 for that component.
    """
    score = 0

    # ── Valuation (0-8) ─────────────────────
    fwd_pe = _f(info.get("forwardPE"))
    pb     = _f(info.get("priceToBook"))

    if fwd_pe is not None:
        if   fwd_pe < 15:  score += 4   # cheap
        elif fwd_pe < 25:  score += 3   # fair value
        elif fwd_pe < 40:  score += 2   # moderate premium
        elif fwd_pe < 60:  score += 1   # expensive
        # above 60 → 0 pts

    if pb is not None:
        if   pb < 1.5:     score += 4   # trading near book (value territory)
        elif pb < 3.0:     score += 3   # reasonable
        elif pb < 6.0:     score += 2   # elevated but acceptable for growth
        elif pb < 10.0:    score += 1
        # above 10 → 0 pts

    # ── Profitability (0-8) ──────────────────
    net_margin = _f(info.get("profitMargins"))
    roe        = _f(info.get("returnOnEquity"))

    if net_margin is not None:
        pct = net_margin * 100
        if   pct > 25:     score += 4   # exceptional margins (e.g. NVDA)
        elif pct > 15:     score += 3   # strong
        elif pct > 8:      score += 2   # decent
        elif pct > 0:      score += 1   # at least profitable
        # negative → 0 pts

    if roe is not None:
        pct = roe * 100
        if   pct > 30:     score += 4   # excellent capital efficiency
        elif pct > 20:     score += 3
        elif pct > 10:     score += 2
        elif pct > 0:      score += 1
        # negative → 0 pts

    # ── Growth (0-5) ─────────────────────────
    rev_growth = _f(info.get("revenueGrowth"))
    eps_growth = _f(info.get("earningsGrowth"))

    growth = max(
        rev_growth if rev_growth is not None else -999,
        eps_growth if eps_growth is not None else -999,
    )
    if growth != -999:
        pct = growth * 100
        if   pct > 30:     score += 5   # hypergrowth
        elif pct > 20:     score += 4
        elif pct > 10:     score += 3
        elif pct > 0:      score += 2
        elif pct > -10:    score += 1   # slight decline but manageable
        # worse → 0 pts

    # ── Balance Sheet (0-4) ──────────────────
    d2e           = _f(info.get("debtToEquity"))
    current_ratio = _f(info.get("currentRatio"))

    if d2e is not None:
        if   d2e < 30:     score += 2   # very low debt (D/E < 0.3)
        elif d2e < 100:    score += 1   # manageable leverage
        # above 100 → 0 pts (yfinance reports D/E * 100)

    if current_ratio is not None:
        if   current_ratio > 2.0:  score += 2   # strong liquidity
        elif current_ratio > 1.5:  score += 1
        # below 1.5 → 0 pts

    return min(score, 25)


# ─────────────────────────────────────────────
#  SHORT INTEREST SIGNAL
# ─────────────────────────────────────────────

def _short_interest_signal(short_pct: float | None) -> str:
    if short_pct is None:
        return "—"
    pct = short_pct * 100
    if   pct >= 20:  return f"🔴 {pct:.1f}% (Very High)"
    elif pct >= 10:  return f"🟠 {pct:.1f}% (High)"
    elif pct >= 5:   return f"⚪ {pct:.1f}%"
    elif pct >= 0:   return f"🟢 {pct:.1f}% (Low)"
    return "—"


# ─────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────

def fetch_fundamentals(tickers: list, data_dir: str) -> dict:
    """
    Fetch fundamental data for all tickers via yfinance.
    Returns dict: {ticker: {fundamental_score, short_pct, short_signal,
                             fwd_pe, pb, net_margin, roe, revenue_growth,
                             debt_to_equity, market_cap_b, sector}}

    Caches to data_dir/fundamentals_cache.json for _CACHE_HOURS hours.
    """
    cache_path = os.path.join(data_dir, "fundamentals_cache.json")
    cached     = _load_cache(cache_path)

    # Check if today's cache is valid
    today_str = datetime.now().strftime("%Y-%m-%d")
    if cached.get("_date") == today_str and len(cached) > 2:
        print(f"  📊 Fundamentals loaded from cache ({len(cached)-1} tickers).")
        return {k: v for k, v in cached.items() if k != "_date"}

    print(f"  📊 Fetching fundamentals for {len(tickers)} tickers...")

    result = {}
    for i, ticker in enumerate(tickers):
        try:
            info = yf.Ticker(ticker).info

            fwd_pe      = _f(info.get("forwardPE"))
            pb          = _f(info.get("priceToBook"))
            net_margin  = _f(info.get("profitMargins"))
            roe         = _f(info.get("returnOnEquity"))
            rev_growth  = _f(info.get("revenueGrowth"))
            eps_growth  = _f(info.get("earningsGrowth"))
            d2e         = _f(info.get("debtToEquity"))
            curr_ratio  = _f(info.get("currentRatio"))
            short_pct   = _f(info.get("shortPercentOfFloat"))
            short_ratio = _f(info.get("shortRatio"))  # days to cover
            mkt_cap     = _f(info.get("marketCap"))
            mkt_cap_b   = round(mkt_cap / 1e9, 1) if mkt_cap else None
            sector      = info.get("sector", "")
            analyst_rec = info.get("recommendationKey", "")

            f_score = _compute_fundamental_score(info)

            result[ticker] = {
                "fundamental_score": f_score,
                "short_pct":         short_pct,
                "short_ratio":       short_ratio,
                "short_signal":      _short_interest_signal(short_pct),
                "fwd_pe":            round(fwd_pe, 1)     if fwd_pe     is not None else None,
                "pb":                round(pb, 2)          if pb         is not None else None,
                "net_margin_pct":    round(net_margin*100, 1) if net_margin is not None else None,
                "roe_pct":           round(roe*100, 1)    if roe        is not None else None,
                "rev_growth_pct":    round(rev_growth*100,1) if rev_growth is not None else None,
                "eps_growth_pct":    round(eps_growth*100,1) if eps_growth is not None else None,
                "debt_to_equity":    round(d2e/100, 2)    if d2e        is not None else None,
                "current_ratio":     round(curr_ratio, 2) if curr_ratio is not None else None,
                "market_cap_b":      mkt_cap_b,
                "sector":            sector,
                "analyst_rec":       analyst_rec,
            }

        except Exception:
            result[ticker] = {"fundamental_score": 0, "short_signal": "—"}

        # Progress dot every 10 tickers
        if (i + 1) % 10 == 0:
            print(f"      {i+1}/{len(tickers)} done...")

    print(f"  ✅ Fundamentals complete for {len(result)} tickers.")

    # Save to cache
    _save_cache(cache_path, {"_date": today_str, **result})
    return result
