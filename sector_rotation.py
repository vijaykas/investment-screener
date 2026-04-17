"""
============================================================
  INVESTMENT INTELLIGENCE — Sector Rotation Model
  Phase 6c: GICS Sector Momentum & Rotation Signals
  Author: Claude (Vijay's Investment Project)
  Updated: 2026-04-15
============================================================

WHAT THIS DOES:
  Analyses the 11 GICS sector ETFs (XL* series) to identify which
  sectors are being rotated INTO vs OUT OF by institutional money.

  Sector rotation follows a predictable economic cycle:
    Early Recovery  → Financials, Consumer Discretionary, Industrials
    Mid Expansion   → Technology, Materials, Energy
    Late Cycle      → Energy, Healthcare, Consumer Staples
    Recession       → Utilities, Consumer Staples, Healthcare

SCORING (0–100 per sector):
  Momentum   40 pts — 1M/3M/6M return vs SPY (relative strength)
  Technical  35 pts — RSI, MACD, 200MA position
  Flow       25 pts — Volume trend (institutional accumulation/distribution)

ROTATION SIGNALS:
  🟢 ROTATE IN  (≥70) — Strong outperformance, add exposure
  🔵 HOLD       (≥50) — In line with market, maintain
  🟡 NEUTRAL    (≥35) — Mixed signals
  🟠 ROTATE OUT (≥20) — Underperforming, reduce
  🔴 AVOID      (<20) — Clear underperformance, exit

CYCLE POSITION:
  Derived from which sectors are leading vs lagging.
  Provides a macro context label for portfolio positioning.
============================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import json
import os
from datetime import datetime, date, timedelta

_CACHE_HOURS = 6

# GICS sector ETFs with full names
SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLV":  "Healthcare",
    "XLY":  "Consumer Disc.",
    "XLP":  "Consumer Staples",
    "XLI":  "Industrials",
    "XLB":  "Materials",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLC":  "Comm. Services",
}

BENCHMARK = "SPY"

# Economic cycle phase → which sectors typically lead
CYCLE_PHASES = {
    "Early Recovery":  ["XLF", "XLY", "XLI", "XLK"],
    "Mid Expansion":   ["XLK", "XLB", "XLE", "XLI"],
    "Late Cycle":      ["XLE", "XLV", "XLP", "XLU"],
    "Contraction":     ["XLU", "XLP", "XLV", "XLRE"],
}


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


def _safe_return(series: pd.Series, days: int) -> float | None:
    """Compute n-day return safely."""
    try:
        if len(series) < days + 1:
            return None
        v0 = float(series.iloc[-days-1])
        v1 = float(series.iloc[-1])
        if v0 == 0:
            return None
        return (v1 / v0 - 1) * 100
    except Exception:
        return None


def _compute_sector_score(df: pd.DataFrame, spy_returns: dict) -> dict:
    """Compute rotation score + signals for a single sector ETF."""
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()
    price = float(close.iloc[-1])

    score = 0

    # ── Momentum vs SPY (0-40 pts) ───────────────────────────────
    ret_1m  = _safe_return(close, 21)
    ret_3m  = _safe_return(close, 63)
    ret_6m  = _safe_return(close, 126)

    spy_1m  = spy_returns.get("1m", 0) or 0
    spy_3m  = spy_returns.get("3m", 0) or 0
    spy_6m  = spy_returns.get("6m", 0) or 0

    rel_1m = (ret_1m - spy_1m) if ret_1m is not None else None
    rel_3m = (ret_3m - spy_3m) if ret_3m is not None else None
    rel_6m = (ret_6m - spy_6m) if ret_6m is not None else None

    # 1M relative (15 pts)
    if rel_1m is not None:
        if   rel_1m >  4: score += 15
        elif rel_1m >  2: score += 12
        elif rel_1m >  0: score += 8
        elif rel_1m > -2: score += 4
        else:             score += 0

    # 3M relative (15 pts)
    if rel_3m is not None:
        if   rel_3m >  6: score += 15
        elif rel_3m >  3: score += 12
        elif rel_3m >  0: score += 8
        elif rel_3m > -4: score += 4
        else:             score += 0

    # 6M relative (10 pts)
    if rel_6m is not None:
        if   rel_6m >  8: score += 10
        elif rel_6m >  4: score += 8
        elif rel_6m >  0: score += 5
        elif rel_6m > -5: score += 2
        else:             score += 0

    # ── Technical (0-35 pts) ─────────────────────────────────────
    rsi = float(ta.momentum.RSIIndicator(close=close, window=14).rsi().iloc[-1])
    # RSI 40-65 is the sweet spot for trending (0-12 pts)
    if   40 <= rsi <= 65: score += 12
    elif 30 <= rsi <  40: score += 9   # oversold recovery
    elif 65 <  rsi <= 75: score += 6
    elif rsi < 30:        score += 5   # deeply oversold
    else:                 score += 1   # overbought (>75)

    # 200MA position (0-8 pts)
    sma200 = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
    if float(close.iloc[-1]) > float(sma200.iloc[-1]):
        score += 8

    # MACD (0-10 pts)
    macd_ind  = ta.trend.MACD(close=close)
    macd_line = macd_ind.macd()
    sig_line  = macd_ind.macd_signal()
    macd_hist = macd_ind.macd_diff()
    if float(macd_line.iloc[-1]) > float(sig_line.iloc[-1]):
        score += 6
    if float(macd_hist.iloc[-1]) > float(macd_hist.iloc[-2]):
        score += 4   # histogram improving

    # ADX trend strength (0-5 pts)
    adx = float(ta.trend.ADXIndicator(high=high, low=low, close=close).adx().iloc[-1])
    if   adx > 30: score += 5
    elif adx > 20: score += 3

    # ── Volume/Flow (0-25 pts) ────────────────────────────────────
    vol_avg20 = float(vol.rolling(20).mean().iloc[-1])
    vol_ratio = float(vol.iloc[-1]) / vol_avg20 if vol_avg20 > 0 else 1.0

    obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=vol).on_balance_volume()
    obv_20ma  = float(obv.rolling(20).mean().iloc[-1])
    obv_curr  = float(obv.iloc[-1])
    obv_trend = float(obv.diff(10).iloc[-1])  # 10-day OBV change (flow direction)

    if obv_curr > obv_20ma: score += 12   # OBV above its MA = accumulation
    if obv_trend > 0:       score += 8    # OBV rising = money flowing in
    if vol_ratio > 1.1:     score += 5    # above-average volume confirms move

    score = min(score, 100)

    # Rotation signal
    if   score >= 70: signal = "🟢 ROTATE IN"
    elif score >= 50: signal = "🔵 HOLD"
    elif score >= 35: signal = "🟡 NEUTRAL"
    elif score >= 20: signal = "🟠 ROTATE OUT"
    else:             signal = "🔴 AVOID"

    return {
        "score":        score,
        "signal":       signal,
        "price":        round(price, 2),
        "rsi":          round(rsi, 1),
        "ret_1m":       round(ret_1m, 2)  if ret_1m  is not None else None,
        "ret_3m":       round(ret_3m, 2)  if ret_3m  is not None else None,
        "ret_6m":       round(ret_6m, 2)  if ret_6m  is not None else None,
        "rel_1m":       round(rel_1m, 2)  if rel_1m  is not None else None,
        "rel_3m":       round(rel_3m, 2)  if rel_3m  is not None else None,
        "rel_6m":       round(rel_6m, 2)  if rel_6m  is not None else None,
        "above_200ma":  float(close.iloc[-1]) > float(sma200.iloc[-1]),
        "adx":          round(adx, 1),
    }


def _infer_cycle_phase(sector_scores: dict) -> str:
    """
    Infer which economic cycle phase we're in based on which sectors
    are leading (score ≥ 60).
    """
    leading = [t for t, d in sector_scores.items() if d.get("score", 0) >= 60]

    phase_matches = {}
    for phase, etfs in CYCLE_PHASES.items():
        overlap = len(set(leading) & set(etfs))
        if overlap > 0:
            phase_matches[phase] = overlap

    if not phase_matches:
        return "Unclear / Transitional"

    best_phase = max(phase_matches, key=phase_matches.get)
    top_sectors = [SECTOR_ETFS.get(t, t) for t in leading if t in CYCLE_PHASES.get(best_phase, [])]

    return f"{best_phase} (led by {', '.join(top_sectors[:3])})"


def fetch_sector_rotation(data_dir: str) -> dict:
    """
    Compute sector rotation scores for all 11 GICS sectors.

    Returns dict:
      {
        "sectors": {XLK: {score, signal, ret_1m, rel_1m, ...}, ...},
        "ranked":  [(XLK, score), ...],   # sorted best to worst
        "cycle_phase": str,
        "rotate_in":   [ticker, ...],
        "rotate_out":  [ticker, ...],
      }

    Caches to data_dir/sector_rotation_cache.json.
    """
    cache_path = os.path.join(data_dir, "sector_rotation_cache.json")
    cached     = _load_cache(cache_path)
    today_str  = date.today().strftime("%Y-%m-%d")

    if cached.get("_date") == today_str:
        print(f"  🔄 Sector rotation loaded from cache.")
        return cached.get("rotation", {})

    print(f"  🔄 Computing sector rotation model...")

    # Fetch SPY as benchmark
    try:
        spy_df    = yf.download(BENCHMARK, period="6mo", auto_adjust=True, progress=False)
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df_columns = spy_df.columns.get_level_values(0)
        spy_close = spy_df["Close"].squeeze()
        spy_returns = {
            "1m": _safe_return(spy_close, 21),
            "3m": _safe_return(spy_close, 63),
            "6m": _safe_return(spy_close, 126),
        }
    except Exception:
        spy_returns = {"1m": 0, "3m": 0, "6m": 0}

    # Score each sector
    sector_scores = {}
    for etf, name in SECTOR_ETFS.items():
        try:
            df = yf.download(etf, period="9mo", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 150:
                continue
            scores = _compute_sector_score(df, spy_returns)
            scores["name"] = name
            sector_scores[etf] = scores
        except Exception:
            continue

    # Rank sectors
    ranked = sorted(sector_scores.items(), key=lambda x: -x[1].get("score", 0))

    # Identify rotate_in / rotate_out
    rotate_in  = [t for t, d in ranked if d.get("score", 0) >= 70]
    rotate_out = [t for t, d in ranked if d.get("score", 0) < 30]

    cycle_phase = _infer_cycle_phase(sector_scores)

    result = {
        "sectors":     sector_scores,
        "ranked":      [(t, d["score"]) for t, d in ranked],
        "cycle_phase": cycle_phase,
        "rotate_in":   rotate_in,
        "rotate_out":  rotate_out,
    }

    _save_cache(cache_path, {"_date": today_str, "rotation": result})

    print(f"  ✅ Sector rotation complete. Cycle: {cycle_phase}")
    if rotate_in:
        print(f"      Rotate IN:  {', '.join(rotate_in)}")
    if rotate_out:
        print(f"      Rotate OUT: {', '.join(rotate_out)}")

    return result
