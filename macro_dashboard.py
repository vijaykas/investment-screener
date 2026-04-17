"""
============================================================
  INVESTMENT INTELLIGENCE — Macro Dashboard
  Phase 5e: FRED + Market Indicators
  Author: Claude (Vijay's Investment Project)
  Updated: 2026-04-15
============================================================

WHAT THIS DOES:
  Pulls macro-economic indicators that move markets.
  Provides context that pure technicals miss:
    — Are we in a risk-on or risk-off environment?
    — Is the yield curve flashing recession risk?
    — Is the dollar strengthening (headwind for equities)?

DATA SOURCES:
  Primary:  FRED API (Federal Reserve Economic Data)
            Set FRED_KEY in config.py or as env var FRED_KEY
            Get a free key: https://fred.stlouisfed.org/docs/api/api_key.html

  Fallback: yfinance — proxies for VIX, treasury yields, dollar
            (works without any API key, always available)

KEY INDICATORS:
  Yield Curve  T10Y2Y   — 10yr minus 2yr treasury spread
                          Negative = inverted (recession signal)
  VIX          VIXCLS   — Fear index; >30 = high fear / market stress
  Fed Funds    FEDFUNDS  — Current interest rate environment
  CPI YoY      CPIAUCSL  — Inflation trend (drives Fed policy)
  Unemployment UNRATE    — Labour market health
  DXY          USD Index — Dollar strength (headwind for US equities)

MACRO HEALTH SCORE (0-100):
  25 pts — Yield curve shape (positive = healthy)
  20 pts — VIX level (low = calm markets)
  20 pts — Inflation trend (falling toward target = healthy)
  20 pts — Employment (low unemployment = healthy)
  15 pts — Dollar trend (stable/weakening = equities friendly)
============================================================
"""

import os
import json
import requests
import yfinance as yf
from datetime import datetime, date, timedelta

_CACHE_HOURS = 6    # macro data: refresh every 6 hours

# FRED series IDs
_FRED_SERIES = {
    "yield_curve":    "T10Y2Y",     # 10yr - 2yr spread (%)
    "vix":            "VIXCLS",     # CBOE VIX
    "fed_funds":      "FEDFUNDS",   # Fed funds effective rate
    "cpi_yoy":        "CPIAUCSL",   # CPI (level — we compute YoY change)
    "unemployment":   "UNRATE",     # US unemployment rate (%)
}

# yfinance fallback tickers
_YF_PROXIES = {
    "vix":        "^VIX",
    "t10y":       "^TNX",          # 10yr yield
    "t2y":        "^IRX",          # ~3-month, closest free proxy
    "dxy":        "DX-Y.NYB",      # US Dollar Index
}


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
            json.dump(data, f)
    except Exception:
        pass


# ─────────────────────────────────────────────
#  FRED FETCH
# ─────────────────────────────────────────────

def _fred_latest(series_id: str, api_key: str, obs: int = 3) -> list[dict]:
    """Fetch the last `obs` observations for a FRED series."""
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}&file_type=json"
        f"&sort_order=desc&limit={obs}"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json().get("observations", [])


def _fred_value(obs_list: list) -> float | None:
    """Return the most recent non-null value from a FRED observation list."""
    for obs in obs_list:
        v = obs.get("value", ".")
        if v != ".":
            try:
                return float(v)
            except ValueError:
                continue
    return None


def _fetch_fred(api_key: str) -> dict:
    """Fetch all macro indicators via FRED API."""
    data = {}

    # Yield curve
    try:
        obs = _fred_latest("T10Y2Y", api_key)
        data["yield_curve"]      = _fred_value(obs)
        data["yield_curve_date"] = obs[0]["date"] if obs else None
    except Exception:
        pass

    # VIX
    try:
        obs = _fred_latest("VIXCLS", api_key)
        data["vix"] = _fred_value(obs)
    except Exception:
        pass

    # Fed funds
    try:
        obs = _fred_latest("FEDFUNDS", api_key)
        data["fed_funds"] = _fred_value(obs)
    except Exception:
        pass

    # CPI — get last 13 months to compute 12-month change
    try:
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id=CPIAUCSL&api_key={api_key}&file_type=json"
            f"&sort_order=desc&limit=13"
        )
        r   = requests.get(url, timeout=10)
        obs = r.json().get("observations", [])
        vals = [_fred_value([o]) for o in obs if _fred_value([o]) is not None]
        if len(vals) >= 13:
            data["cpi_yoy"] = round((vals[0] / vals[12] - 1) * 100, 2)
        elif len(vals) >= 2:
            data["cpi_yoy"] = round((vals[0] / vals[-1] - 1) * 100, 2)
    except Exception:
        pass

    # Unemployment
    try:
        obs = _fred_latest("UNRATE", api_key)
        data["unemployment"] = _fred_value(obs)
    except Exception:
        pass

    return data


# ─────────────────────────────────────────────
#  YFINANCE FALLBACK
# ─────────────────────────────────────────────

def _fetch_yfinance_macro() -> dict:
    """Fetch macro proxies via yfinance — always available, no key needed."""
    data = {}
    try:
        vix = yf.Ticker("^VIX").history(period="5d")
        if not vix.empty:
            data["vix"] = round(float(vix["Close"].iloc[-1]), 2)
    except Exception:
        pass

    try:
        t10 = yf.Ticker("^TNX").history(period="5d")
        if not t10.empty:
            data["t10y_yield"] = round(float(t10["Close"].iloc[-1]), 3)
    except Exception:
        pass

    try:
        t2 = yf.Ticker("^IRX").history(period="5d")
        if not t2.empty:
            # IRX is 13-week yield annualised; use as proxy for short rates
            data["t_short_yield"] = round(float(t2["Close"].iloc[-1]) / 100, 3)
    except Exception:
        pass

    # Compute yield curve proxy if both available
    if "t10y_yield" in data and "t_short_yield" in data:
        data["yield_curve"] = round(data["t10y_yield"] - data["t_short_yield"] * 100, 3)

    try:
        dxy = yf.Ticker("DX-Y.NYB").history(period="30d")
        if not dxy.empty:
            data["dxy"]          = round(float(dxy["Close"].iloc[-1]), 2)
            data["dxy_1m_chg"]   = round(
                (float(dxy["Close"].iloc[-1]) / float(dxy["Close"].iloc[0]) - 1) * 100, 2
            )
    except Exception:
        pass

    return data


# ─────────────────────────────────────────────
#  MACRO HEALTH SCORE  (0-100)
# ─────────────────────────────────────────────

def _compute_macro_score(data: dict) -> int:
    score = 0

    # ── Yield Curve (0-25) ───────────────────
    yc = data.get("yield_curve")
    if yc is not None:
        if   yc >  1.5:  score += 25   # steep / healthy
        elif yc >  0.5:  score += 20
        elif yc >  0.0:  score += 15   # flat but positive
        elif yc > -0.5:  score += 8    # mildly inverted (caution)
        else:            score += 2    # deeply inverted (recession risk)

    # ── VIX (0-20) ───────────────────────────
    vix = data.get("vix")
    if vix is not None:
        if   vix < 15:   score += 20   # complacent / calm
        elif vix < 20:   score += 16
        elif vix < 25:   score += 12
        elif vix < 30:   score += 6
        else:            score += 0    # fear/panic regime

    # ── CPI / Inflation (0-20) ───────────────
    cpi = data.get("cpi_yoy")
    if cpi is not None:
        if   cpi < 2.5:  score += 20   # at/below target
        elif cpi < 3.5:  score += 14
        elif cpi < 5.0:  score += 8
        elif cpi < 7.0:  score += 4
        else:            score += 0    # hot inflation

    # ── Unemployment (0-20) ──────────────────
    ue = data.get("unemployment")
    if ue is not None:
        if   ue < 4.0:   score += 20   # full employment
        elif ue < 5.0:   score += 15
        elif ue < 6.5:   score += 8
        elif ue < 8.0:   score += 4
        else:            score += 0

    # ── Dollar (0-15) ────────────────────────
    dxy_chg = data.get("dxy_1m_chg")
    if dxy_chg is not None:
        if   dxy_chg < -2:   score += 15   # weakening dollar = bullish for equities
        elif dxy_chg < 0:    score += 10
        elif dxy_chg < 1:    score += 7    # stable
        elif dxy_chg < 3:    score += 4
        else:                score += 1    # strong dollar = headwind

    return min(score, 100)


def _macro_signal(score: int) -> str:
    if   score >= 75:  return "🟢 Risk-On (Favours Equities)"
    elif score >= 55:  return "🔵 Moderate (Mixed Signals)"
    elif score >= 40:  return "🟡 Caution (Defensive Bias)"
    elif score >= 25:  return "🟠 Risk-Off (Reduce Exposure)"
    else:              return "🔴 High Risk (Capital Preservation)"


def _yield_curve_label(yc: float | None) -> str:
    if yc is None:
        return "—"
    if   yc >  1.5:  return f"+{yc:.2f}% 🟢 Steep (healthy)"
    elif yc >  0.5:  return f"+{yc:.2f}% 🔵 Normal"
    elif yc >  0.0:  return f"+{yc:.2f}% 🟡 Flat (caution)"
    elif yc > -0.5:  return f"{yc:.2f}% 🟠 Inverted (watch)"
    else:            return f"{yc:.2f}% 🔴 Deeply Inverted (⚠️ recession signal)"


def _vix_label(vix: float | None) -> str:
    if vix is None:
        return "—"
    if   vix < 15:   return f"{vix:.1f} 🟢 Low (calm markets)"
    elif vix < 20:   return f"{vix:.1f} 🔵 Normal"
    elif vix < 25:   return f"{vix:.1f} 🟡 Elevated"
    elif vix < 30:   return f"{vix:.1f} 🟠 High (fear)"
    else:            return f"{vix:.1f} 🔴 Extreme (panic)"


def _cpi_label(cpi: float | None) -> str:
    if cpi is None:
        return "—"
    if   cpi < 2.5:  return f"{cpi:.1f}% 🟢 On Target"
    elif cpi < 3.5:  return f"{cpi:.1f}% 🔵 Mild"
    elif cpi < 5.0:  return f"{cpi:.1f}% 🟠 Elevated"
    else:            return f"{cpi:.1f}% 🔴 Hot"


def _ue_label(ue: float | None) -> str:
    if ue is None:
        return "—"
    if   ue < 4.0:   return f"{ue:.1f}% 🟢 Full Employment"
    elif ue < 5.0:   return f"{ue:.1f}% 🔵 Healthy"
    elif ue < 6.5:   return f"{ue:.1f}% 🟡 Softening"
    else:            return f"{ue:.1f}% 🔴 Weak"


# ─────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────

def fetch_macro_dashboard(data_dir: str, fred_key: str = "") -> dict:
    """
    Fetch macro dashboard data.

    Returns dict with keys:
      yield_curve, yield_curve_label, vix, vix_label, fed_funds,
      cpi_yoy, cpi_label, unemployment, ue_label, dxy, dxy_1m_chg,
      macro_score, macro_signal, source

    Caches to data_dir/macro_cache.json for _CACHE_HOURS hours.
    """
    cache_path = os.path.join(data_dir, "macro_cache.json")
    cached     = _load_cache(cache_path)
    if cached:
        print(f"  🌐 Macro dashboard loaded from cache (score: {cached.get('macro_score', '?')}).")
        return cached

    print("  🌐 Fetching macro indicators...")

    # ── Try FRED first, fallback to yfinance ─
    if fred_key:
        try:
            data   = _fetch_fred(fred_key)
            source = "FRED"
            # Always supplement with yfinance for DXY (not in FRED)
            yf_data = _fetch_yfinance_macro()
            data.setdefault("dxy",        yf_data.get("dxy"))
            data.setdefault("dxy_1m_chg", yf_data.get("dxy_1m_chg"))
            if "vix" not in data or data["vix"] is None:
                data["vix"] = yf_data.get("vix")
        except Exception as e:
            print(f"      ⚠️  FRED fetch failed ({e}), falling back to yfinance.")
            data   = _fetch_yfinance_macro()
            source = "yfinance"
    else:
        data   = _fetch_yfinance_macro()
        source = "yfinance"

    # ── Add labels ───────────────────────────
    data["yield_curve_label"] = _yield_curve_label(data.get("yield_curve"))
    data["vix_label"]         = _vix_label(data.get("vix"))
    data["cpi_label"]         = _cpi_label(data.get("cpi_yoy"))
    data["ue_label"]          = _ue_label(data.get("unemployment"))

    # ── Macro score + signal ──────────────────
    macro_score           = _compute_macro_score(data)
    data["macro_score"]   = macro_score
    data["macro_signal"]  = _macro_signal(macro_score)
    data["source"]        = source

    print(f"  ✅ Macro score: {macro_score}/100 — {data['macro_signal']} (via {source})")

    _save_cache(cache_path, data)
    return data
