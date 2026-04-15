"""
============================================================
  INVESTMENT INTELLIGENCE — PHASE 4b
  ETF Screener & Scorer
  Author: Claude (Vijay's Investment Project)
  Updated: 2026-04-13
============================================================

WHAT THIS DOES:
  Scores every ETF in ETF_WATCHLIST across three dimensions:

    Technical  (50 pts) — RSI, MACD, Moving Averages, Bollinger,
                           ADX trend strength, Volume surge
    Momentum   (30 pts) — 1m / 3m / 6m / 12m returns vs SPY benchmark
                           (relative strength, not raw return)
    Quality    (20 pts) — Expense ratio (lower = better),
                           AUM / liquidity (larger = better)

  Total score is out of 100.  Signal labels:
    ≥75 🟢 STRONG BUY  |  ≥58 🔵 BUY  |  ≥42 🟡 HOLD
    ≥28 🟠 CAUTION     |  <28 🔴 AVOID

OUTPUT:
  data/etf_screener_results.csv  — one row per ETF with all scores/metrics
  Terminal summary table

HOW TO RUN:
  python3 etf_screener.py
  python3 invest.py --etf          (via the main launcher)
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import os
import sys
from datetime import datetime, date

# ─────────────────────────────────────────────
#  CONFIGURATION  (from config.py)
# ─────────────────────────────────────────────
from config import ETF_WATCHLIST, ETF_MONITOR_CONFIG as CFG, DATA_DIR, get_etf_watchlist

OUTPUT_FILE = os.path.join(DATA_DIR, CFG["output_file"])
BENCHMARK   = CFG["benchmark"]   # "SPY"


# ─────────────────────────────────────────────
#  DATA FETCH
# ─────────────────────────────────────────────

def fetch_etf(ticker: str, years: int = 2) -> pd.DataFrame | None:
    """Download up to 2 years of daily price data for an ETF."""
    try:
        df = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)
        if df.empty or len(df) < 60:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None


def fetch_etf_info(ticker: str) -> dict:
    """Fetch ETF metadata (expense ratio, AUM, yield, etc.) from yfinance .info."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "expense_ratio": info.get("annualReportExpenseRatio")
                             or info.get("expenseRatio")
                             or info.get("totalExpenseRatio"),
            "aum_B":         (info.get("totalAssets") or 0) / 1e9,
            "yield_pct":     (info.get("yield") or info.get("dividendYield") or 0) * 100,
            "category":      info.get("category") or info.get("fundFamily") or "",
            "name":          info.get("longName") or info.get("shortName") or ticker,
            "beta":          info.get("beta3Year") or info.get("beta") or None,
        }
    except Exception:
        return {
            "expense_ratio": None, "aum_B": 0.0, "yield_pct": 0.0,
            "category": "", "name": ticker, "beta": None,
        }


# ─────────────────────────────────────────────
#  TECHNICAL SCORING  (50 pts)
# ─────────────────────────────────────────────

def score_technical(df: pd.DataFrame) -> dict:
    """
    Compute technical indicators and a 0-50 technical score.
    Returns a flat dict with all indicator values plus 'tech_score'.
    """
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()
    price = float(close.iloc[-1])

    # ── Day change ───────────────────────────
    prev_close  = float(close.iloc[-2])
    day_chg_pct = (price / prev_close - 1) * 100

    # ── RSI (14) ─────────────────────────────
    rsi_val = float(ta.momentum.RSIIndicator(close=close, window=14).rsi().iloc[-1])

    # ── MACD ─────────────────────────────────
    macd_ind    = ta.trend.MACD(close=close)
    macd_line   = macd_ind.macd()
    sig_line    = macd_ind.macd_signal()
    macd_hist   = macd_ind.macd_diff()
    macd_bull   = bool(macd_line.iloc[-1] > sig_line.iloc[-1])
    macd_cross  = bool(macd_line.iloc[-1] > sig_line.iloc[-1] and
                       macd_line.iloc[-3] < sig_line.iloc[-3])

    # ── Moving averages ──────────────────────
    sma50  = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
    sma200 = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
    above_50  = bool(price > float(sma50.iloc[-1]))
    above_200 = bool(price > float(sma200.iloc[-1]))
    golden_x  = False
    for i in range(-10, -1):
        if (float(sma50.iloc[i])   > float(sma200.iloc[i]) and
                float(sma50.iloc[i-1]) < float(sma200.iloc[i-1])):
            golden_x = True; break

    # ── Bollinger bands ──────────────────────
    bb      = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    bb_pct  = float(bb.bollinger_pband().iloc[-1]) * 100

    # ── ADX ──────────────────────────────────
    adx_val = float(ta.trend.ADXIndicator(high=high, low=low, close=close).adx().iloc[-1])

    # ── Volume ───────────────────────────────
    vol_avg20 = float(vol.rolling(20).mean().iloc[-1])
    vol_ratio = float(vol.iloc[-1]) / vol_avg20 if vol_avg20 > 0 else 1.0
    vol_surge = vol_ratio > CFG["volume_surge_ratio"]
    obv_up    = float(
        ta.volume.OnBalanceVolumeIndicator(close=close, volume=vol)
        .on_balance_volume().iloc[-1]
    ) > float(
        ta.volume.OnBalanceVolumeIndicator(close=close, volume=vol)
        .on_balance_volume().rolling(20).mean().iloc[-1]
    )

    # ── 52-week stats ─────────────────────────
    w52_high = float(close.tail(252).max())
    w52_low  = float(close.tail(252).min())
    pct_from_high = (price / w52_high - 1) * 100
    pct_from_low  = (price / w52_low  - 1) * 100
    at_52w_high   = pct_from_high > -2.0
    at_52w_low    = pct_from_low  < 5.0

    # ── Build score ──────────────────────────
    score = 0

    # RSI  (0–12)
    if 40 <= rsi_val <= 60:   score += 8
    elif 30 <= rsi_val < 40:  score += 12
    elif rsi_val < 30:        score += 10
    elif 60 < rsi_val <= 70:  score += 6
    else:                     score += 2

    # MACD  (0–12)
    if macd_cross:                                          score += 12
    elif macd_bull and float(macd_hist.iloc[-1]) > 0:       score += 8
    elif float(macd_hist.iloc[-1]) > float(macd_hist.iloc[-2]): score += 5
    else:                                                   score += 1

    # MAs  (0–14)
    if golden_x:    score += 7
    elif above_200: score += 5
    if above_200:   score += 3
    if above_50:    score += 4

    # Bollinger  (0–7)
    if 20 <= bb_pct <= 60:  score += 7
    elif bb_pct < 20:       score += 5
    elif bb_pct > 90:       score += 1
    else:                   score += 3

    # ADX  (0–5)
    if adx_val > 40:    score += 5
    elif adx_val > 25:  score += 3
    else:               score += 1

    # Volume / OBV  (0–5)
    if vol_surge and obv_up: score += 5
    elif obv_up:             score += 3
    elif vol_surge:          score += 2

    tech_score = min(int(score), 50)

    return {
        "price":        round(price, 2),
        "day_chg_pct":  round(day_chg_pct, 2),
        "rsi":          round(rsi_val, 1),
        "macd_bull":    macd_bull,
        "macd_cross":   macd_cross,
        "above_50ma":   above_50,
        "above_200ma":  above_200,
        "golden_cross": golden_x,
        "bb_pct":       round(bb_pct, 1),
        "adx":          round(adx_val, 1),
        "vol_ratio":    round(vol_ratio, 2),
        "vol_surge":    vol_surge,
        "52w_high":     round(w52_high, 2),
        "52w_low":      round(w52_low, 2),
        "pct_from_high": round(pct_from_high, 2),
        "pct_from_low":  round(pct_from_low, 2),
        "at_52w_high":  at_52w_high,
        "at_52w_low":   at_52w_low,
        "tech_score":   tech_score,
    }


# ─────────────────────────────────────────────
#  MOMENTUM SCORING  (30 pts)
# ─────────────────────────────────────────────

def score_momentum(df_etf: pd.DataFrame, df_spy: pd.DataFrame) -> dict:
    """
    Compute returns over 1m/3m/6m/12m and score relative to SPY.
    Returns individual returns, relative returns, and 'momentum_score' (0-30).
    """
    periods = CFG["momentum_periods"]  # {"1m":21, "3m":63, "6m":126, "12m":252}
    close_e = df_etf["Close"].squeeze()
    close_s = df_spy["Close"].squeeze()

    def pct_return(s: pd.Series, days: int) -> float | None:
        if len(s) < days + 1:
            return None
        return float((s.iloc[-1] / s.iloc[-days] - 1) * 100)

    returns = {}
    rel_returns = {}

    for label, days in periods.items():
        r_etf = pct_return(close_e, days)
        r_spy = pct_return(close_s, days)
        returns[f"ret_{label}"] = round(r_etf, 2) if r_etf is not None else None
        if r_etf is not None and r_spy is not None:
            rel_returns[f"rel_{label}"] = round(r_etf - r_spy, 2)
        else:
            rel_returns[f"rel_{label}"] = None

    # Score based on relative strength vs SPY
    # Weights: 1m=10pts, 3m=12pts, 6m=8pts (shorter-term momentum dominates)
    score = 15  # neutral baseline

    rel_1m  = rel_returns.get("rel_1m")
    rel_3m  = rel_returns.get("rel_3m")
    rel_6m  = rel_returns.get("rel_6m")
    rel_12m = rel_returns.get("rel_12m")

    # 1-month relative performance (±8 pts)
    if rel_1m is not None:
        if rel_1m > 3:    score += 8
        elif rel_1m > 1:  score += 5
        elif rel_1m > -1: score += 2
        elif rel_1m > -3: score -= 3
        else:             score -= 6

    # 3-month relative performance (±10 pts)
    if rel_3m is not None:
        if rel_3m > 5:    score += 10
        elif rel_3m > 2:  score += 6
        elif rel_3m > -2: score += 2
        elif rel_3m > -5: score -= 4
        else:             score -= 8

    # 6-month relative performance (±7 pts, less weight on stale signal)
    if rel_6m is not None:
        if rel_6m > 7:    score += 7
        elif rel_6m > 3:  score += 4
        elif rel_6m > -3: score += 1
        elif rel_6m > -7: score -= 3
        else:             score -= 6

    momentum_score = min(max(int(score), 0), 30)

    return {**returns, **rel_returns, "momentum_score": momentum_score}


# ─────────────────────────────────────────────
#  QUALITY SCORING  (20 pts)
# ─────────────────────────────────────────────

def score_quality(info: dict) -> dict:
    """
    Score expense ratio and AUM liquidity.  Returns 'quality_score' (0-20).
    Lower expense ratio and higher AUM are rewarded.
    """
    exp  = info.get("expense_ratio")   # decimal, e.g. 0.0003 = 0.03%
    aum  = info.get("aum_B", 0.0)

    # ── Expense ratio  (0–12 pts) ────────────
    exp_score = 0
    if exp is not None:
        exp_pct = exp * 100 if exp < 1 else exp   # normalise: 0.0003 → 0.03%
        tiers = CFG["expense_ratio_tiers"]         # [0.10, 0.20, 0.50, 1.00]
        if exp_pct <= tiers[0]:   exp_score = 12  # ≤0.10%  (Vanguard-class)
        elif exp_pct <= tiers[1]: exp_score = 10  # ≤0.20%
        elif exp_pct <= tiers[2]: exp_score = 7   # ≤0.50%
        elif exp_pct <= tiers[3]: exp_score = 4   # ≤1.00%
        else:                     exp_score = 1   # >1.00%  (expensive)
    else:
        exp_score = 5  # unknown — neutral

    # ── AUM / liquidity  (0–8 pts) ──────────
    aum_score = 0
    tiers_B = CFG["aum_tiers_B"]   # [1.0, 5.0, 10.0, 50.0]
    if aum >= tiers_B[3]:   aum_score = 8   # ≥$50B  (mega-liquid)
    elif aum >= tiers_B[2]: aum_score = 6   # ≥$10B
    elif aum >= tiers_B[1]: aum_score = 4   # ≥$5B
    elif aum >= tiers_B[0]: aum_score = 2   # ≥$1B
    else:                   aum_score = 0   # <$1B  (small/illiquid)

    quality_score = min(exp_score + aum_score, 20)
    return {"exp_score": exp_score, "aum_score": aum_score, "quality_score": quality_score}


# ─────────────────────────────────────────────
#  COMBINED SCORE + SIGNAL
# ─────────────────────────────────────────────

def compute_etf_signal(total_score: int) -> str:
    """Convert a 0-100 total score to a readable signal label."""
    cfg = CFG
    if total_score >= cfg["strong_buy_threshold"]:  return "🟢 STRONG BUY"
    if total_score >= cfg["buy_threshold"]:         return "🔵 BUY"
    if total_score >= cfg["hold_threshold"]:        return "🟡 HOLD"
    if total_score >= cfg["avoid_threshold"]:       return "🟠 CAUTION"
    return "🔴 AVOID"


def score_etf(ticker: str, etf_category: str, df_spy: pd.DataFrame) -> dict | None:
    """
    Full ETF scoring pipeline for one ticker.
    Returns a flat result dict ready to write to CSV, or None on failure.
    """
    df = fetch_etf(ticker)
    if df is None:
        return None

    info     = fetch_etf_info(ticker)
    tech     = score_technical(df)
    mom      = score_momentum(df, df_spy)
    qual     = score_quality(info)

    w = CFG["weights"]
    # Scale each component to its weight fraction
    tech_contrib = int(tech["tech_score"] * w["technical"] / 50)
    mom_contrib  = int(mom["momentum_score"] * w["momentum"] / 30)
    qual_contrib = int(qual["quality_score"] * w["quality"] / 20)
    total        = min(tech_contrib + mom_contrib + qual_contrib, 100)
    signal       = compute_etf_signal(total)

    # Expense ratio display
    exp_raw = info.get("expense_ratio")
    if exp_raw is not None:
        exp_pct = exp_raw * 100 if exp_raw < 1 else exp_raw
        exp_str = f"{exp_pct:.2f}%"
    else:
        exp_str = "N/A"

    aum_B = info.get("aum_B", 0.0)
    aum_str = f"${aum_B:.1f}B" if aum_B >= 1.0 else f"${aum_B*1000:.0f}M"

    return {
        "ticker":       ticker,
        "name":         info["name"],
        "category":     etf_category,
        "yf_category":  info["category"],

        # Price data
        "price":        tech["price"],
        "day_chg_pct":  tech["day_chg_pct"],
        "52w_high":     tech["52w_high"],
        "52w_low":      tech["52w_low"],
        "pct_from_high": tech["pct_from_high"],

        # Technical
        "tech_score":   tech["tech_score"],
        "rsi":          tech["rsi"],
        "macd_bull":    tech["macd_bull"],
        "macd_cross":   tech["macd_cross"],
        "above_50ma":   tech["above_50ma"],
        "above_200ma":  tech["above_200ma"],
        "golden_cross": tech["golden_cross"],
        "bb_pct":       tech["bb_pct"],
        "adx":          tech["adx"],
        "vol_ratio":    tech["vol_ratio"],
        "vol_surge":    tech["vol_surge"],
        "at_52w_high":  tech["at_52w_high"],
        "at_52w_low":   tech["at_52w_low"],

        # Momentum
        "ret_1m":       mom.get("ret_1m"),
        "ret_3m":       mom.get("ret_3m"),
        "ret_6m":       mom.get("ret_6m"),
        "ret_12m":      mom.get("ret_12m"),
        "rel_1m":       mom.get("rel_1m"),
        "rel_3m":       mom.get("rel_3m"),
        "rel_6m":       mom.get("rel_6m"),
        "rel_12m":      mom.get("rel_12m"),
        "momentum_score": mom["momentum_score"],

        # Quality
        "expense_ratio_pct": exp_str,
        "aum":          aum_str,
        "yield_pct":    round(info["yield_pct"], 2),
        "beta":         info.get("beta"),
        "quality_score": qual["quality_score"],

        # Composite
        "tech_contrib":  tech_contrib,
        "mom_contrib":   mom_contrib,
        "qual_contrib":  qual_contrib,
        "total_score":   total,
        "signal":        signal,

        # Metadata
        "as_of":  date.today().isoformat(),
    }


# ─────────────────────────────────────────────
#  MAIN RUNNER
# ─────────────────────────────────────────────

def run_etf_screener():
    print(f"\n{'='*60}")
    print(f"  INVESTMENT INTELLIGENCE — ETF Screener")
    print(f"  {datetime.now().strftime('%A, %B %d, %Y  %H:%M')}")
    print(f"{'='*60}\n")

    # ── Pre-fetch benchmark (SPY) ─────────────
    print(f"  📥 Fetching benchmark ({BENCHMARK})...")
    df_spy = fetch_etf(BENCHMARK)
    if df_spy is None:
        print(f"  ❌ Could not fetch {BENCHMARK}. Aborting.")
        return None

    # ── Resolve watchlist (static or live FMP) ─
    watchlist = get_etf_watchlist()

    # ── Build flat ticker list ────────────────
    all_etfs = []
    for cat, tickers in watchlist.items():
        for t in tickers:
            all_etfs.append((t, cat))

    n_total = len(all_etfs)
    print(f"  Scoring {n_total} ETFs across {len(ETF_WATCHLIST)} categories...\n")

    results = []
    for i, (ticker, category) in enumerate(all_etfs):
        sys.stdout.write(f"\r  [{i+1}/{n_total}] {ticker:<6} ({category})           ")
        sys.stdout.flush()
        row = score_etf(ticker, category, df_spy)
        if row:
            results.append(row)

    print(f"\n\n  ✅ Scored {len(results)}/{n_total} ETFs.\n")

    if not results:
        print("  ⚠️  No results — check network connection.")
        return None

    # ── Sort by total score ───────────────────
    results.sort(key=lambda r: -r["total_score"])

    # ── Terminal summary ──────────────────────
    print(f"  {'─'*78}")
    print(f"  {'TICKER':<6} {'CAT':<18} {'PRICE':>7} {'CHG':>6} "
          f"{'SCORE':>5} {'SIGNAL':<16} {'1M':>6} {'3M':>6} {'REL3M':>6}  EXPENSE  AUM")
    print(f"  {'─'*78}")
    for r in results:
        rel3m = r.get("rel_3m")
        rel3m_str = f"{rel3m:+.1f}%" if rel3m is not None else "  N/A"
        ret1m  = r.get("ret_1m")
        ret1m_str  = f"{ret1m:+.1f}%" if ret1m is not None else "  N/A"
        ret3m  = r.get("ret_3m")
        ret3m_str  = f"{ret3m:+.1f}%" if ret3m is not None else "  N/A"
        chg_str = f"{r['day_chg_pct']:+.1f}%"
        print(f"  {r['ticker']:<6} {r['category']:<18} ${r['price']:>6.2f} "
              f"{chg_str:>6}  {r['total_score']:>4}  "
              f"{r['signal']:<16} {ret1m_str:>6} {ret3m_str:>6} {rel3m_str:>6}  "
              f"{r['expense_ratio_pct']:>6}  {r['aum']}")

    # ── By-category summary ───────────────────
    df_r = pd.DataFrame(results)
    print(f"\n  {'─'*40}")
    print(f"  CATEGORY SUMMARY")
    print(f"  {'─'*40}")
    for cat in watchlist:
        sub = df_r[df_r["category"] == cat]
        if sub.empty:
            continue
        avg_score = sub["total_score"].mean()
        best = sub.sort_values("total_score", ascending=False).iloc[0]
        avg_rel3m = sub["rel_3m"].mean()
        rel3m_str = f"{avg_rel3m:+.1f}%" if not pd.isna(avg_rel3m) else "N/A"
        print(f"  {cat:<20}  avg={avg_score:.0f}/100  "
              f"best={best['ticker']} ({best['total_score']})  "
              f"avg 3m vs SPY: {rel3m_str}")

    # ── Save to CSV ───────────────────────────
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  💾 Saved → {OUTPUT_FILE}")
    print(f"\n{'='*60}\n")

    return df_out


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_etf_screener()
