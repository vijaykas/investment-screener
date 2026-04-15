"""
============================================================
  INVESTMENT INTELLIGENCE — PHASE 4
  Daily Morning Monitor
  Author: Claude (Vijay's Investment Project)
  Updated: 2026-04-12
============================================================

WHAT THIS DOES (runs every weekday morning before market open):
  1. Pulls fresh price data for your entire watchlist
  2. Re-scores every stock using the Phase 1 screener engine
  3. Detects SIGNAL CHANGES since yesterday (new buys, new exits, flips)
  4. Flags notable price events (52-week highs/lows, big moves, volume surges)
  5. Checks your PORTFOLIO positions for P&L and risk alerts
  6. Writes a clean HTML daily report → daily_reports/<YYYY-MM-DD>.html
  7. Appends signal history to signal_history.json (for trend tracking)

HOW TO RUN:
  python3 daily_monitor.py

  The script is also invoked automatically by the Claude scheduled task
  set up in Phase 4 of the Investment Intelligence stack.

CUSTOMISE:
  Edit MONITOR_CONFIG below — especially YOUR_POSITIONS to track
  actual holdings with entry prices and share counts.
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import json
import os
import sys
from datetime import datetime, date, timedelta

# ─────────────────────────────────────────────
#  CONFIGURATION  (loaded from config.py)
# ─────────────────────────────────────────────
from config import MONITOR_CONFIG, ETF_MONITOR_CONFIG, ETF_WATCHLIST, DATA_DIR, REPORTS_DIR, get_watchlist, NEWS_SENTIMENT_CONFIG, TOP20_CONFIG, CRYPTO_CONFIG, CRYPTO_WATCHLIST

OUTPUT_DIR   = DATA_DIR   # data files (signal_history.json) written to data/
HISTORY_FILE = os.path.join(DATA_DIR, MONITOR_CONFIG["history_file"])

TODAY    = date.today().strftime("%Y-%m-%d")
NOW_HOUR = datetime.now().hour                          # 0-23 local time
RUN_SLOT = "pm" if NOW_HOUR >= 12 else "am"            # am / pm label for report file


# ─────────────────────────────────────────────
#  DATA & SCORING (reuses Phase 1 logic)
# ─────────────────────────────────────────────

def fetch_stock(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
        if df.empty or len(df) < 60:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None


def compute_score(df: pd.DataFrame) -> dict:
    """Compute composite technical score + individual indicator values."""
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()
    price = float(close.iloc[-1])

    # Price change
    prev_close = float(close.iloc[-2])
    day_chg_pct = (price / prev_close - 1) * 100

    # RSI
    rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    rsi_val = float(rsi.iloc[-1])

    # MACD
    macd_ind   = ta.trend.MACD(close=close)
    macd_line  = macd_ind.macd()
    sig_line   = macd_ind.macd_signal()
    macd_hist  = macd_ind.macd_diff()
    macd_bullish = bool(macd_line.iloc[-1] > sig_line.iloc[-1])
    macd_cross   = bool(macd_line.iloc[-1] > sig_line.iloc[-1] and
                        macd_line.iloc[-3] < sig_line.iloc[-3])

    # MAs
    sma50  = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
    sma200 = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
    above_50  = bool(price > float(sma50.iloc[-1]))
    above_200 = bool(price > float(sma200.iloc[-1]))

    # Bollinger
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    bb_pct = float(bb.bollinger_pband().iloc[-1]) * 100

    # ADX
    adx = float(ta.trend.ADXIndicator(high=high, low=low, close=close).adx().iloc[-1])

    # Volume
    vol_avg20   = float(vol.rolling(20).mean().iloc[-1])
    vol_ratio   = float(vol.iloc[-1]) / vol_avg20 if vol_avg20 > 0 else 1.0
    vol_surge   = vol_ratio > MONITOR_CONFIG["volume_surge_ratio"]

    # 52-week
    w52_high = float(close.tail(252).max())
    w52_low  = float(close.tail(252).min())
    pct_from_high = (price / w52_high - 1) * 100
    pct_from_low  = (price / w52_low  - 1) * 100
    at_52w_high   = pct_from_high > -2.0   # Within 2% of 52w high
    at_52w_low    = pct_from_low  < 5.0    # Within 5% of 52w low

    # ── Composite technical score ─────────────
    score = 0

    # RSI (0-15)
    if 40 <= rsi_val <= 60:   score += 8
    elif 30 <= rsi_val < 40:  score += 15
    elif rsi_val < 30:        score += 12
    elif 60 < rsi_val <= 70:  score += 6
    else:                     score += 2

    # MACD (0-12)
    if macd_cross:            score += 12
    elif macd_bullish and float(macd_hist.iloc[-1]) > 0: score += 8
    elif float(macd_hist.iloc[-1]) > float(macd_hist.iloc[-2]): score += 5
    else:                     score += 1

    # MAs (0-15)
    if above_200: score += 5
    if above_50:  score += 5
    # check golden cross in last 10 bars
    for i in range(-10, -1):
        if (float(sma50.iloc[i]) > float(sma200.iloc[i]) and
                float(sma50.iloc[i-1]) < float(sma200.iloc[i-1])):
            score += 7; break
    else:
        score += 3 if price > float(sma50.iloc[-1]) else 0

    # Bollinger (0-8)
    if 20 <= bb_pct <= 60:   score += 8
    elif bb_pct < 20:        score += 6
    elif bb_pct > 90:        score += 1
    else:                    score += 4

    # ADX (0-5)
    if adx > 40:   score += 5
    elif adx > 25: score += 3
    else:          score += 1

    # Volume (0-5)
    obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=vol).on_balance_volume()
    obv_up = float(obv.iloc[-1]) > float(obv.rolling(20).mean().iloc[-1])
    if vol_surge and obv_up: score += 5
    elif obv_up:             score += 3
    elif vol_surge:          score += 2

    score = min(score, 75)  # cap at 75 (tech only — fundamentals not in daily run)

    # Signal label
    if score >= MONITOR_CONFIG["strong_buy_threshold"]:
        signal = "🟢 STRONG BUY"
    elif score >= MONITOR_CONFIG["buy_threshold"]:
        signal = "🔵 BUY"
    elif score >= 45:
        signal = "🟡 HOLD"
    elif score >= MONITOR_CONFIG["avoid_threshold"]:
        signal = "🟠 CAUTION"
    else:
        signal = "🔴 AVOID"

    return {
        "price":         round(price, 2),
        "day_chg_pct":   round(day_chg_pct, 2),
        "score":         score,
        "signal":        signal,
        "rsi":           round(rsi_val, 1),
        "macd_bullish":  macd_bullish,
        "macd_cross":    macd_cross,
        "above_50ma":    above_50,
        "above_200ma":   above_200,
        "bb_pct":        round(bb_pct, 1),
        "adx":           round(adx, 1),
        "vol_ratio":     round(vol_ratio, 2),
        "vol_surge":     vol_surge,
        "52w_high":      round(w52_high, 2),
        "52w_low":       round(w52_low, 2),
        "pct_from_high": round(pct_from_high, 2),
        "at_52w_high":   at_52w_high,
        "at_52w_low":    at_52w_low,
    }


# ─────────────────────────────────────────────
#  SIGNAL HISTORY
# ─────────────────────────────────────────────

def load_history() -> dict:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return {}


def save_history(history: dict):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def detect_signal_changes(today_scores: dict, history: dict) -> list[dict]:
    """Compare today's signals to yesterday's and return a list of notable changes."""
    changes = []
    dates = sorted(history.keys())
    if not dates:
        return changes
    yesterday = dates[-1]
    prev = history[yesterday]

    for ticker, today in today_scores.items():
        if ticker not in prev:
            continue
        prev_sig  = prev[ticker]["signal"]
        today_sig = today["signal"]
        prev_score = prev[ticker]["score"]
        today_score = today["score"]
        score_delta = today_score - prev_score

        # Signal level flip
        def sig_level(s):
            if "STRONG BUY" in s: return 5
            if "BUY" in s:        return 4
            if "HOLD" in s:       return 3
            if "CAUTION" in s:    return 2
            return 1

        pl = sig_level(prev_sig)
        tl = sig_level(today_sig)

        if tl > pl:
            changes.append({
                "ticker":      ticker,
                "type":        "UPGRADE",
                "from":        prev_sig,
                "to":          today_sig,
                "score_delta": score_delta,
                "price":       today["price"],
                "note":        f"Score: {prev_score} → {today_score} (+{score_delta})"
            })
        elif tl < pl:
            changes.append({
                "ticker":      ticker,
                "type":        "DOWNGRADE",
                "from":        prev_sig,
                "to":          today_sig,
                "score_delta": score_delta,
                "price":       today["price"],
                "note":        f"Score: {prev_score} → {today_score} ({score_delta})"
            })
        # MACD cross alert (new today)
        if today.get("macd_cross") and not prev.get("macd_cross"):
            changes.append({
                "ticker": ticker,
                "type":   "MACD_CROSS",
                "from":   "—",
                "to":     "📈 MACD Bullish Crossover",
                "score_delta": score_delta,
                "price":  today["price"],
                "note":   "Fresh MACD bullish crossover signal"
            })

    return sorted(changes, key=lambda x: abs(x["score_delta"]), reverse=True)


# ─────────────────────────────────────────────
#  PORTFOLIO P&L
# ─────────────────────────────────────────────

def compute_portfolio_pnl(positions: dict, today_scores: dict) -> list[dict]:
    rows = []
    for ticker, pos in positions.items():
        if ticker not in today_scores:
            continue
        price  = today_scores[ticker]["price"]
        cost   = pos["avg_cost"]
        shares = pos["shares"]
        pnl    = (price - cost) * shares
        pnl_pct = (price / cost - 1) * 100
        mkt_val = price * shares

        alert = None
        if pnl_pct < -MONITOR_CONFIG["drawdown_alert_pct"]:
            alert = f"⚠️ Down {pnl_pct:.1f}% — review stop-loss"
        elif pnl_pct > 30:
            alert = f"🎯 Up {pnl_pct:.1f}% — consider trimming"

        rows.append({
            "ticker":    ticker,
            "shares":    shares,
            "avg_cost":  cost,
            "price":     price,
            "mkt_val":   round(mkt_val, 2),
            "pnl":       round(pnl, 2),
            "pnl_pct":   round(pnl_pct, 2),
            "signal":    today_scores[ticker]["signal"],
            "score":     today_scores[ticker]["score"],
            "alert":     alert,
        })

    rows.sort(key=lambda x: x["pnl_pct"], reverse=True)
    return rows


# ─────────────────────────────────────────────
#  NOTABLE EVENTS
# ─────────────────────────────────────────────

def find_notable_events(today_scores: dict) -> list[dict]:
    events = []
    for ticker, s in today_scores.items():
        # Big daily move
        if abs(s["day_chg_pct"]) >= MONITOR_CONFIG["big_move_pct"]:
            events.append({
                "ticker": ticker, "price": s["price"],
                "type": "BIG MOVE",
                "detail": f"{s['day_chg_pct']:+.1f}% — {'🚀 surge' if s['day_chg_pct'] > 0 else '🔻 drop'}"
            })
        # 52-week high breakout
        if s["at_52w_high"]:
            events.append({
                "ticker": ticker, "price": s["price"],
                "type": "52W HIGH",
                "detail": f"Near 52-week high ${s['52w_high']} ({s['pct_from_high']:+.1f}%)"
            })
        # 52-week low (potential bottom)
        if s["at_52w_low"]:
            events.append({
                "ticker": ticker, "price": s["price"],
                "type": "52W LOW",
                "detail": f"Near 52-week low ${s['52w_low']} — monitor for reversal"
            })
        # RSI oversold
        if s["rsi"] < MONITOR_CONFIG["rsi_oversold"]:
            events.append({
                "ticker": ticker, "price": s["price"],
                "type": "RSI OVERSOLD",
                "detail": f"RSI {s['rsi']} — potential bounce zone"
            })
        # RSI overbought
        if s["rsi"] > MONITOR_CONFIG["rsi_overbought"]:
            events.append({
                "ticker": ticker, "price": s["price"],
                "type": "RSI OVERBOUGHT",
                "detail": f"RSI {s['rsi']} — overextended, caution"
            })
        # Volume surge
        if s["vol_surge"]:
            events.append({
                "ticker": ticker, "price": s["price"],
                "type": "VOLUME SURGE",
                "detail": f"{s['vol_ratio']:.1f}x average volume — unusual activity"
            })

    return events


# ─────────────────────────────────────────────
#  HTML REPORT GENERATOR
# ─────────────────────────────────────────────

SIGNAL_COLOR = {
    "🟢 STRONG BUY": "#00E676",
    "🔵 BUY":         "#40C4FF",
    "🟡 HOLD":        "#FFD740",
    "🟠 CAUTION":     "#FF6D00",
    "🔴 AVOID":       "#EF5350",
}

# ── Yahoo Finance clickable ticker link ───────
def ticker_link(ticker: str) -> str:
    """Return an HTML anchor that opens Yahoo Finance for the ticker in a new tab."""
    url = f"https://finance.yahoo.com/quote/{ticker}"
    return (
        f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
        f'style="color:#58a6ff;font-weight:bold;text-decoration:none;" '
        f'title="Open {ticker} on Yahoo Finance">'
        f'{ticker}</a>'
    )

# ── Load ML predictions from CSV ─────────────
def load_ml_predictions() -> dict:
    """
    Load Ensemble P(up)% for each monthly horizon (1m–12m) from ml_predictions.csv.
    Returns: {ticker: {"1m": float, "2m": float, ..., "12m": float}}
    Returns {} if the file doesn't exist or uses the old 5d/10d/20d format.
    """
    csv_path = os.path.join(DATA_DIR, "ml_predictions.csv")
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
        if "ticker" not in df.columns:
            return {}
        df = df.set_index("ticker")
        monthly = ["1m","2m","3m","4m","5m","6m","7m","8m","9m","10m","11m","12m"]
        result = {}
        for ticker in df.index:
            row = df.loc[ticker]
            preds = {}
            for h in monthly:
                col = f"prob_up_Ensemble_{h}"
                if col in df.columns:
                    val = row.get(col)
                    if pd.notna(val):
                        preds[h] = float(val)
            if preds:
                result[str(ticker)] = preds
        return result
    except Exception:
        return {}

# ── Load ETF screener results ─────────────────
def load_etf_scores() -> dict:
    """
    Load ETF scores from data/etf_screener_results.csv.
    Returns {ticker: row_dict} or {} if the file doesn't exist.
    """
    csv_path = os.path.join(DATA_DIR, ETF_MONITOR_CONFIG["output_file"])
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
        if "ticker" not in df.columns:
            return {}
        result = {}
        for _, row in df.iterrows():
            result[str(row["ticker"])] = row.to_dict()
        return result
    except Exception:
        return {}


# ── Load news sentiment data ─────────────────
def load_news_sentiment() -> dict:
    """
    Load per-ticker sentiment from data/news_sentiment.csv.
    Returns {ticker: {score, label, headline, themes}} or {} if missing.
    """
    csv_path = os.path.join(DATA_DIR, NEWS_SENTIMENT_CONFIG["output_file"])
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
        if "ticker" not in df.columns:
            return {}
        result = {}
        for _, row in df.iterrows():
            result[str(row["ticker"])] = {
                "score":    float(row.get("sentiment_score", 0)),
                "label":    str(row.get("sentiment_label", "⚪ Neutral")),
                "headline": str(row.get("top_headline", "")),
                "themes":   str(row.get("theme_tags", "")),
                "pct":      int(row.get("sentiment_pct", 50)),
                "source":   str(row.get("data_source", "")),
            }
        return result
    except Exception:
        return {}


# ── Load macro themes ─────────────────────────
def load_macro_themes() -> list:
    """
    Load macro market themes from data/macro_themes.json.
    Returns list of theme dicts or a default set if file missing.
    """
    json_path = os.path.join(DATA_DIR, NEWS_SENTIMENT_CONFIG["themes_file"])
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path) as f:
            data = json.load(f)
        return data.get("themes", [])
    except Exception:
        return []


# ── Load Top 20 predictions ───────────────────
def load_top20_predictions() -> list:
    """
    Load top20_predictions.csv as a list of row dicts.
    Returns [] if not found.
    """
    csv_path = os.path.join(DATA_DIR, TOP20_CONFIG["output_file"])
    if not os.path.exists(csv_path):
        return []
    try:
        df = pd.read_csv(csv_path)
        # Fill NaN to avoid rendering issues
        df = df.fillna("")
        return df.to_dict("records")
    except Exception:
        return []


# ── Load crypto screener results ─────────────
def load_crypto_scores() -> dict:
    """
    Load crypto screener results from data/crypto_screener_results.csv.
    Returns {ticker: row_dict} or {} if missing.
    """
    csv_path = os.path.join(DATA_DIR, CRYPTO_CONFIG["output_file"])
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
        if "ticker" not in df.columns:
            return {}
        result = {}
        for _, row in df.iterrows():
            result[str(row["ticker"])] = row.fillna("").to_dict()
        return result
    except Exception:
        return {}


# ── Load crypto market cycle context ─────────
def load_crypto_cycle() -> dict:
    """
    Load crypto market cycle context from data/crypto_cycle.json.
    Returns {} if missing.
    """
    json_path = os.path.join(DATA_DIR, CRYPTO_CONFIG["cycle_file"])
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path) as f:
            return json.load(f)
    except Exception:
        return {}


# ── Color-code a probability value ───────────
def _prob_style(prob: float) -> tuple[str, str]:
    """Return (background, text) color pair for a P(up)% value."""
    if prob >= 65:
        return "#1a3a1a", "#69f0ae"   # strong green
    elif prob >= 55:
        return "#1e3320", "#a5d6a7"   # mild green
    elif prob >= 45:
        return "#1e2430", "#8b949e"   # neutral grey
    elif prob >= 35:
        return "#3a1a1a", "#ef9a9a"   # mild red
    else:
        return "#4a0f0f", "#ff8a80"   # strong red

def _outlook_label(preds: dict) -> str:
    """Summarise the 12-month outlook trend as a short label + colour."""
    horizons = ["1m","2m","3m","4m","5m","6m","7m","8m","9m","10m","11m","12m"]
    vals = [preds[h] for h in horizons if h in preds]
    if len(vals) < 4:
        return "<span style='color:#8b949e'>—</span>"
    avg    = sum(vals) / len(vals)
    trend  = vals[-1] - vals[0]   # positive = improving over time
    if avg >= 60 and trend >= 0:
        return "<span style='color:#69f0ae;font-weight:bold'>📈 Bullish</span>"
    elif avg >= 55:
        return "<span style='color:#a5d6a7;font-weight:bold'>↗ Mildly Bullish</span>"
    elif avg <= 40 and trend <= 0:
        return "<span style='color:#ff8a80;font-weight:bold'>📉 Bearish</span>"
    elif avg <= 45:
        return "<span style='color:#ef9a9a;font-weight:bold'>↘ Mildly Bearish</span>"
    else:
        return "<span style='color:#8b949e;font-weight:bold'>➡ Neutral</span>"

def _build_etf_section(etf_scores: dict, sig_color_fn) -> str:
    """
    Build the full ETF Market Dashboard HTML section.

    Comprises three sub-panels:
      1. Category heatmap  — one coloured cell per ETF grouped by category,
                             coloured by 3-month return vs SPY
      2. Full ETF table    — all ETFs sorted by total_score with key metrics
      3. Sector spotlight  — XL* sector ETFs ranked by momentum
    """
    if not etf_scores:
        return """
<div class="section" style="color:#8b949e">
  <h2>📈 ETF Market Dashboard</h2>
  <p>No ETF data available. Run <code>python3 etf_screener.py</code> or
     <code>python3 invest.py --etf</code> to generate it.</p>
</div>"""

    # ── Helper: colour from rel_3m ────────────
    def rel_color(rel):
        """Background and text colour based on relative 3m return vs SPY."""
        if rel is None or (isinstance(rel, float) and np.isnan(rel)):
            return "#21262d", "#8b949e"
        if rel >= 5:    return "#0d3321", "#69f0ae"
        if rel >= 2:    return "#1e3320", "#a5d6a7"
        if rel >= -2:   return "#1e2430", "#8b949e"
        if rel >= -5:   return "#3a1a1a", "#ef9a9a"
        return "#4a0f0f", "#ff8a80"

    def ret_str(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        return f"{val:+.1f}%"

    def score_bg(score):
        """Row background tint based on ETF score."""
        if score >= 75: return "#0d3321"
        if score >= 58: return "#0d2a3a"
        if score >= 42: return "#1a1a1a"
        if score >= 28: return "#2a1a0a"
        return "#2a0d0d"

    # Gather all ETF rows as list of dicts
    rows = list(etf_scores.values())
    rows_sorted = sorted(rows, key=lambda r: -float(r.get("total_score", 0)))

    # ── 1. Category heatmap ───────────────────
    heatmap_html = ""
    for cat, tickers in ETF_WATCHLIST.items():
        cells = ""
        for t in tickers:
            if t not in etf_scores:
                continue
            r   = etf_scores[t]
            rel = r.get("rel_3m")
            try:
                rel = float(rel)
            except (TypeError, ValueError):
                rel = None
            bg, fg = rel_color(rel)
            ret3m  = ret_str(rel)
            day_c  = r.get("day_chg_pct", 0)
            try:
                day_c = float(day_c)
            except (TypeError, ValueError):
                day_c = 0
            day_fg = "#69f0ae" if day_c >= 0 else "#ff8a80"
            yf_url = f"https://finance.yahoo.com/quote/{t}"
            cells += f"""<div class="heatmap-cell" style="background:{bg};color:{fg}" title="{r.get('name', t)} | Score:{r.get('total_score','?')} | 3m vs SPY: {ret3m}">
  <div class="hm-ticker"><a href="{yf_url}" target="_blank" rel="noopener noreferrer" style="color:{fg};text-decoration:none">{t}</a></div>
  <div class="hm-ret" style="color:{day_fg}">{day_c:+.1f}% today</div>
  <div class="hm-ret">3m: {ret3m}</div>
</div>"""
        if cells:
            heatmap_html += f"""
<div style="margin-bottom:12px">
  <span style="font-size:11px;color:#8b949e;font-weight:700;text-transform:uppercase;
               letter-spacing:0.5px;display:block;margin-bottom:6px">{cat}</span>
  <div class="heatmap-grid">{cells}</div>
</div>"""

    # ── 2. Full ETF comparison table ──────────
    table_rows = ""
    for r in rows_sorted:
        ticker  = str(r.get("ticker", ""))
        cat     = str(r.get("category", ""))
        score   = float(r.get("total_score", 0))
        signal  = str(r.get("signal", ""))
        price   = float(r.get("price", 0))
        day_chg = float(r.get("day_chg_pct", 0))
        rsi     = r.get("rsi", "—")
        abv200  = r.get("above_200ma", False)
        rel_1m  = r.get("rel_1m")
        rel_3m  = r.get("rel_3m")
        rel_6m  = r.get("rel_6m")
        ret_12m = r.get("ret_12m")
        exp_r   = str(r.get("expense_ratio_pct", "N/A"))
        aum     = str(r.get("aum", "—"))
        yld     = r.get("yield_pct", 0)
        try:
            yld = float(yld)
        except (TypeError, ValueError):
            yld = 0.0

        sig_c   = sig_color_fn(signal)
        chg_c   = "#00E676" if day_chg >= 0 else "#EF5350"
        row_bg  = score_bg(score)

        def _rel_cell(val):
            try:
                v = float(val)
                c = "#69f0ae" if v >= 2 else ("#ef9a9a" if v <= -2 else "#8b949e")
                return f'<td style="color:{c};font-weight:600">{v:+.1f}%</td>'
            except (TypeError, ValueError):
                return '<td style="color:#484f58">—</td>'

        abv_html = "✅" if abv200 else "❌"
        yf_url = f"https://finance.yahoo.com/quote/{ticker}"
        link = (f'<a href="{yf_url}" target="_blank" rel="noopener noreferrer" '
                f'style="color:#58a6ff;font-weight:bold;text-decoration:none">{ticker}</a>')

        table_rows += f"""
<tr style="background:{row_bg}">
  <td>{link}</td>
  <td style="color:#8b949e;font-size:11px">{cat}</td>
  <td>${price:.2f}</td>
  <td style="color:{chg_c};font-weight:600">{day_chg:+.2f}%</td>
  <td style="font-weight:700;color:{'#69f0ae' if score>=58 else ('#ef9a9a' if score<42 else '#e6edf3')}">{score:.0f}</td>
  <td style="color:{sig_c};font-size:11px;font-weight:bold">{signal}</td>
  <td>{rsi}</td>
  <td>{abv_html}</td>
  {_rel_cell(rel_1m)}
  {_rel_cell(rel_3m)}
  {_rel_cell(rel_6m)}
  {_rel_cell(ret_12m)}
  <td style="color:#8b949e">{exp_r}</td>
  <td style="color:#8b949e">{aum}</td>
  <td style="color:#a5d6a7">{yld:.1f}%</td>
</tr>"""

    # ── 3. Sector spotlight (XL* ETFs only) ──
    sector_etfs = [r for r in rows if str(r.get("category", "")) == "Sector"]
    sector_etfs.sort(key=lambda r: -(r.get("rel_3m") or -99 if not (
        isinstance(r.get("rel_3m"), float) and np.isnan(r.get("rel_3m"))) else -99))

    sector_spotlight = ""
    if sector_etfs:
        for r in sector_etfs:
            t       = str(r.get("ticker", ""))
            name    = str(r.get("name", t))
            rel3m   = r.get("rel_3m")
            ret1m   = r.get("ret_1m")
            score   = float(r.get("total_score", 0))
            signal  = str(r.get("signal", ""))
            sig_c   = sig_color_fn(signal)
            try:
                rel3m_v = float(rel3m)
                bg, fg  = rel_color(rel3m_v)
                r3_html = f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">{rel3m_v:+.1f}%</span>'
            except (TypeError, ValueError):
                r3_html = '<span style="color:#484f58">—</span>'
            try:
                r1_v    = float(ret1m)
                r1_c    = "#69f0ae" if r1_v >= 0 else "#ff8a80"
                r1_html = f'<span style="color:{r1_c}">{r1_v:+.1f}%</span>'
            except (TypeError, ValueError):
                r1_html = "—"

            yf_url = f"https://finance.yahoo.com/quote/{t}"
            sector_spotlight += f"""
<tr>
  <td><a href="{yf_url}" target="_blank" rel="noopener noreferrer"
      style="color:#58a6ff;font-weight:bold;text-decoration:none">{t}</a></td>
  <td style="color:#8b949e;font-size:11px">{name}</td>
  <td style="font-weight:700">{score:.0f}/100</td>
  <td style="color:{sig_c};font-size:11px">{signal}</td>
  <td>{r1_html}</td>
  <td>{r3_html}</td>
</tr>"""

    sector_table = f"""
<h3>📊 Sector ETF Relative Strength vs SPY</h3>
<table>
  <tr><th>ETF</th><th>Sector</th><th>Score</th><th>Signal</th><th>1M Return</th><th>3M vs SPY</th></tr>
  {sector_spotlight}
</table>""" if sector_spotlight else ""

    # ── Category summary strip ────────────────
    cat_summary = ""
    for cat, tickers in ETF_WATCHLIST.items():
        cat_rows = [etf_scores[t] for t in tickers if t in etf_scores]
        if not cat_rows:
            continue
        avg_score = np.mean([float(r.get("total_score", 0)) for r in cat_rows])
        rel3m_vals = []
        for r in cat_rows:
            try:
                v = float(r.get("rel_3m", float("nan")))
                if not np.isnan(v):
                    rel3m_vals.append(v)
            except (TypeError, ValueError):
                pass
        avg_rel3m = np.mean(rel3m_vals) if rel3m_vals else None
        bg, fg = rel_color(avg_rel3m)
        rel_label = f"{avg_rel3m:+.1f}% vs SPY" if avg_rel3m is not None else "vs SPY N/A"
        cat_summary += (
            f'<span class="cat-chip" style="background:{bg};color:{fg}" '
            f'title="Avg score {avg_score:.0f}/100">'
            f'{cat}: {rel_label}</span>'
        )

    return f"""
<div class="section">
  <h2>📈 ETF Market Dashboard
    <span style="font-size:11px;color:#8b949e;font-weight:400;margin-left:10px">
      {len(etf_scores)} ETFs · Technical + Momentum vs SPY + Quality · Run <code>python3 invest.py --etf</code> to refresh
    </span>
  </h2>

  <!-- Category summary chips -->
  <div style="margin-bottom:16px">{cat_summary}</div>

  <!-- Heatmap by category -->
  <h3>🗺 Category Heatmap — 3-Month Return vs SPY (hover for details)</h3>
  {heatmap_html}

  <!-- Full comparison table -->
  <h3>📋 All ETFs — Sorted by Score</h3>
  <div style="overflow-x:auto">
  <table style="min-width:900px">
    <tr>
      <th>Ticker</th><th>Category</th><th>Price</th><th>Day Chg</th>
      <th>Score</th><th>Signal</th><th>RSI</th><th>200MA</th>
      <th>1M vs SPY</th><th>3M vs SPY</th><th>6M vs SPY</th><th>12M Ret</th>
      <th>Expense</th><th>AUM</th><th>Yield</th>
    </tr>
    {table_rows}
  </table>
  </div>

  {sector_table}

  <p style="color:#484f58;font-size:11px;margin-top:12px">
    Score = Technical (50) + Momentum vs SPY (30) + Quality/Expense (20).
    Colour: <span style="color:#69f0ae">green = outperforming SPY</span> /
    <span style="color:#ef9a9a">red = underperforming</span>.
    Refresh: <code>python3 invest.py --etf</code>
  </p>
</div>"""


def _build_crypto_section(crypto_scores: dict, cycle: dict) -> str:
    """
    Build the ₿ Crypto Market Dashboard HTML section.

    Comprises:
      1. Market cycle banner (BTC bull/bear indicator)
      2. Category heatmap  — coloured by 30d return vs BTC
      3. Full leaderboard  — all coins sorted by total_score
    """
    if not crypto_scores:
        return """
<div class="section" style="color:#8b949e">
  <h2>₿ Crypto Market Dashboard</h2>
  <p>No crypto data available. Run <code>python3 invest.py --crypto</code> to generate it.</p>
</div>"""

    # ── helpers ───────────────────────────────
    def _sig_color(sig: str) -> str:
        sig = sig.upper()
        if "STRONG BUY" in sig: return "#69f0ae"
        if "BUY"        in sig: return "#00E676"
        if "HOLD"       in sig: return "#FFC107"
        if "CAUTION"    in sig: return "#FF9800"
        return "#EF5350"

    def _rel_color(rel):
        """Background + text colour from relative 30d return vs BTC."""
        try: rel = float(rel)
        except: return "#21262d", "#8b949e"
        if rel >=  15: return "#0b2e1f", "#69f0ae"
        if rel >=   5: return "#1e3320", "#a5d6a7"
        if rel >=  -5: return "#1e2430", "#8b949e"
        if rel >= -15: return "#3a1a1a", "#ef9a9a"
        return "#4a0f0f", "#ff8a80"

    def _score_bg(score: float) -> str:
        if score >= 78: return "#0b2e1f"
        if score >= 62: return "#0d2a3a"
        if score >= 46: return "#1a1a1a"
        if score >= 30: return "#2a1a0a"
        return "#2a0d0d"

    def _fmt(val, fmt="+.1f", suffix="%", fallback="—"):
        try:
            v = float(val)
            return f"{v:{fmt}}{suffix}"
        except: return fallback

    rows  = list(crypto_scores.values())
    rows_sorted = sorted(rows, key=lambda r: -float(r.get("total_score", 0)))

    yf_link = lambda t: (
        f"<a href='https://finance.yahoo.com/quote/{t}' target='_blank' "
        f"style='color:#f7931a;text-decoration:none'>{t.replace('-USD','')}</a>"
    )

    # ── 1. Market cycle banner ────────────────
    cycle_col  = {"Bull Market": "#69f0ae", "Late Bull": "#FFC107",
                  "Caution Zone": "#FF9800", "Bear Market": "#ff8a80"}.get(
                  cycle.get("cycle",""), "#8b949e")
    cycle_emoji = cycle.get("emoji", "❓")
    cycle_name  = cycle.get("cycle", "Unknown")
    btc_price   = cycle.get("btc_price", 0)
    btc_vs_200  = cycle.get("btc_vs_200dma_pct", 0)
    btc_ath_pct = cycle.get("btc_pct_from_high", 0)

    cycle_html = f"""
<div style="display:flex;align-items:center;gap:16px;padding:12px 16px;
            background:#0d1117;border:1px solid #30363d;border-radius:8px;
            margin-bottom:18px;flex-wrap:wrap">
  <div>
    <span style="font-size:20px">{cycle_emoji}</span>
    <span style="font-size:16px;font-weight:bold;color:{cycle_col};margin-left:8px">{cycle_name}</span>
  </div>
  <div style="color:#8b949e;font-size:12px">
    BTC: <strong style="color:#f7931a">${btc_price:,.0f}</strong>
    &nbsp;|&nbsp;
    vs 200 DMA: <strong style="color:{cycle_col}">{btc_vs_200:+.1f}%</strong>
    &nbsp;|&nbsp;
    From ATH: <strong style="color:#8b949e">{btc_ath_pct:+.1f}%</strong>
  </div>
  <div style="color:#484f58;font-size:11px;margin-left:auto">
    Quality assets only · Meme coins excluded · Scored vs BTC benchmark
  </div>
</div>"""

    # ── 2. Category heatmap ───────────────────
    heatmap_html = ""
    for cat, tickers in CRYPTO_WATCHLIST.items():
        cells = ""
        for t in tickers:
            if t not in crypto_scores:
                continue
            r   = crypto_scores[t]
            rel = r.get("rel_30d", 0)
            bg, fg = _rel_color(rel)
            ret30  = _fmt(rel)
            score  = float(r.get("total_score", 0))
            signal = str(r.get("signal", ""))
            sc_col = _sig_color(signal)
            day_c  = float(r.get("day_chg_pct", 0) or 0)
            day_fg = "#69f0ae" if day_c >= 0 else "#ff8a80"
            name   = str(r.get("name", t.replace("-USD","")))[:18]
            price_str = f"${float(r.get('price',0)):,.2f}" if float(r.get('price',0)) >= 1 else f"${float(r.get('price',0)):.4f}"
            yf_url = f"https://finance.yahoo.com/quote/{t}"

            cells += f"""
<div class="heatmap-cell" style="background:{bg};border-color:{bg}"
     onclick="window.open('{yf_url}','_blank')" title="{name} | Score:{score:.0f} | 30d vs BTC:{ret30}">
  <div style="font-weight:bold;color:{fg}">{t.replace('-USD','')}</div>
  <div style="font-size:10px;color:{fg};opacity:0.8">{ret30}</div>
  <div style="font-size:10px;color:{sc_col}">{score:.0f}/100</div>
  <div style="font-size:9px;color:{day_fg}">{day_c:+.1f}%</div>
</div>"""

        if cells:
            heatmap_html += f"""
<div style="margin-bottom:14px">
  <span style="color:#8b949e;font-size:12px;font-weight:bold;display:block;margin-bottom:6px">
    {cat}
  </span>
  <div class="heatmap-grid">{cells}</div>
</div>"""

    # ── 3. Leaderboard table ──────────────────
    table_rows = ""
    for r in rows_sorted:
        ticker  = str(r.get("ticker",""))
        name    = str(r.get("name", ticker.replace("-USD","")))[:22]
        cat     = str(r.get("category",""))[:18]
        score   = float(r.get("total_score", 0))
        signal  = str(r.get("signal",""))
        price   = float(r.get("price", 0) or 0)
        day_c   = float(r.get("day_chg_pct", 0) or 0)
        rsi     = float(r.get("rsi", 50) or 50)
        a200    = r.get("above_200ma", False)
        rel_30d = float(r.get("rel_30d", 0) or 0)
        rel_90d = float(r.get("rel_90d", 0) or 0)
        ret_7d  = float(r.get("ret_7d", 0) or 0)
        ret_30d = float(r.get("ret_30d", 0) or 0)
        mc_B    = float(r.get("market_cap_B", 0) or 0)
        sent_l  = str(r.get("sentiment_label",""))
        tech_s  = float(r.get("tech_score", 0) or 0)
        mom_s   = float(r.get("momentum_score", 0) or 0)
        qual_s  = float(r.get("quality_score", 0) or 0)
        sent_s  = float(r.get("sentiment_score_pts", 0) or 0)
        headline = str(r.get("top_headline",""))[:70]

        bg      = _score_bg(score)
        sc_col  = _sig_color(signal)
        chg_col = "#69f0ae" if day_c >= 0 else "#ff8a80"
        rel30_bg, rel30_fg = _rel_color(rel_30d)
        rel90_bg, rel90_fg = _rel_color(rel_90d)

        price_fmt = f"${price:,.2f}" if price >= 1 else f"${price:.4f}"
        mc_fmt    = f"${mc_B:.0f}B" if mc_B >= 1 else f"${mc_B*1000:.0f}M" if mc_B > 0 else "—"

        table_rows += f"""
<tr style="background:{bg}">
  <td>{yf_link(ticker)}<div style="font-size:10px;color:#8b949e">{cat}</div></td>
  <td style="font-size:11px;color:#8b949e">{name}</td>
  <td style="color:#e6edf3">{price_fmt}</td>
  <td style="color:{chg_col}">{day_c:+.2f}%</td>
  <td style="font-size:16px;font-weight:bold;color:{sc_col}">{score:.1f}</td>
  <td style="color:{sc_col};font-size:11px">{signal}</td>
  <td style="color:#8b949e">{rsi:.0f}</td>
  <td>{'✅' if a200 else '❌'}</td>
  <td style="color:{chg_col if ret_7d>=0 else '#ff8a80'}">{ret_7d:+.1f}%</td>
  <td style="background:{rel30_bg};color:{rel30_fg};padding:2px 6px;border-radius:3px">{rel_30d:+.1f}%</td>
  <td style="background:{rel90_bg};color:{rel90_fg};padding:2px 6px;border-radius:3px">{rel_90d:+.1f}%</td>
  <td style="color:#8b949e;font-size:11px">{mc_fmt}</td>
  <td style="font-size:10px;color:#484f58" title="{headline}">{headline[:55]}{'…' if len(headline)>55 else ''}</td>
</tr>"""

    # Category summary chips
    cat_chips = ""
    for cat in CRYPTO_WATCHLIST:
        cat_tickers = [t for t in CRYPTO_WATCHLIST[cat] if t in crypto_scores]
        if not cat_tickers:
            continue
        cat_scores = [float(crypto_scores[t].get("total_score",0)) for t in cat_tickers]
        avg_sc = sum(cat_scores) / len(cat_scores)
        best   = max(cat_scores)
        col    = "#69f0ae" if avg_sc >= 62 else ("#FFC107" if avg_sc >= 46 else "#ff8a80")
        cat_chips += (f"<span style='background:#0d1117;border:1px solid {col};"
                      f"color:{col};padding:4px 10px;border-radius:12px;font-size:12px;"
                      f"margin:3px;display:inline-block'>"
                      f"{cat} · avg {avg_sc:.0f} · best {best:.0f}</span>")

    n_coins  = len(rows_sorted)
    n_sb     = sum(1 for r in rows if "STRONG BUY" in str(r.get("signal","")))
    n_buy    = sum(1 for r in rows if str(r.get("signal","")) == "🔵 BUY")
    avg_sc   = sum(float(r.get("total_score",0)) for r in rows) / max(n_coins,1)

    return f"""
<div class="section">
  <h2>₿ Crypto Market Dashboard
    <span style="font-size:11px;color:#8b949e;font-weight:400;margin-left:10px">
      {n_coins} quality assets · No meme coins · Scored vs BTC
    </span>
  </h2>

  {cycle_html}

  <!-- Category chips -->
  <div style="margin-bottom:16px">{cat_chips}</div>

  <!-- Category heatmap (30d return vs BTC) -->
  <h3>🗺 Category Heatmap — 30-Day Return vs BTC (hover for details)</h3>
  {heatmap_html}

  <!-- Leaderboard table -->
  <h3>📋 All Cryptos — Sorted by Score</h3>
  <div style="overflow-x:auto">
  <table style="min-width:1000px">
    <tr>
      <th>Ticker</th><th>Name</th><th>Price</th><th>Day</th>
      <th>Score</th><th>Signal</th><th>RSI</th><th>200MA</th>
      <th>7d Ret</th><th>30d vs BTC</th><th>90d vs BTC</th>
      <th>Mkt Cap</th><th>Top Headline</th>
    </tr>
    {table_rows}
  </table>
  </div>

  <p style="color:#484f58;font-size:11px;margin-top:12px">
    Score = Technical (40) + Momentum vs BTC (25) + Quality/On-chain (20) + Sentiment (15).
    Colour: <span style="color:#69f0ae">green = outperforming BTC</span> /
    <span style="color:#ef9a9a">red = underperforming</span>.
    Signal distribution: 🟢 {n_sb} · 🔵 {n_buy} · avg score {avg_sc:.1f}/100.
    Refresh: <code>python3 invest.py --crypto</code>
  </p>
</div>"""


def _build_top20_section(top20_rows: list, macro_themes: list) -> str:
    """
    Build the 🏆 Top 20 High-Yield Predictions HTML section.

    Layout:
      1. Macro Themes chips
      2. Podium cards — #1, #2, #3 (prominent highlight)
      3. Ranked table — #4 to #20 with full detail
    """
    if not top20_rows:
        return """
<div class="section" style="color:#8b949e">
  <h2>🏆 Top 20 High-Yield Predictions</h2>
  <p>No prediction data available. Run
     <code>python3 invest.py --news</code> then
     <code>python3 invest.py --top20</code> to generate it.</p>
</div>"""

    def _sig_color(sig: str) -> str:
        sig = sig.upper()
        if "STRONG BUY" in sig: return "#69f0ae"
        if "BUY"        in sig: return "#00E676"
        if "HOLD"       in sig: return "#FFC107"
        if "CAUTION"    in sig: return "#FF9800"
        return "#EF5350"

    def _score_bg(score: float) -> str:
        if score >= 85: return "#0b2e1f"
        if score >= 75: return "#0d2a3a"
        if score >= 65: return "#1a1f2a"
        if score >= 55: return "#1e1a0a"
        return "#1a1a1a"

    def _badge(asset_type: str) -> str:
        if asset_type == "ETF":
            return "<span style='background:#1a3a5c;color:#40C4FF;padding:2px 7px;border-radius:3px;font-size:11px;font-weight:bold'>ETF</span>"
        if asset_type == "CRYPTO":
            return "<span style='background:#2a1f00;color:#f7931a;padding:2px 7px;border-radius:3px;font-size:11px;font-weight:bold'>₿ CRYPTO</span>"
        return "<span style='background:#1a3a1a;color:#69f0ae;padding:2px 7px;border-radius:3px;font-size:11px;font-weight:bold'>STOCK</span>"

    def _sent_badge(label: str) -> str:
        label = str(label)
        if "Bullish" in label and "Somewhat" not in label:
            return f"<span style='background:#0d3321;color:#69f0ae;padding:2px 8px;border-radius:10px;font-size:11px'>{label}</span>"
        elif "Somewhat Bullish" in label:
            return f"<span style='background:#1e3320;color:#a5d6a7;padding:2px 8px;border-radius:10px;font-size:11px'>{label}</span>"
        elif "Bearish" in label and "Somewhat" not in label:
            return f"<span style='background:#3a0d0d;color:#ff8a80;padding:2px 8px;border-radius:10px;font-size:11px'>{label}</span>"
        elif "Somewhat Bearish" in label:
            return f"<span style='background:#2a1a0a;color:#ef9a9a;padding:2px 8px;border-radius:10px;font-size:11px'>{label}</span>"
        return f"<span style='background:#1e2430;color:#8b949e;padding:2px 8px;border-radius:10px;font-size:11px'>{label}</span>"

    def _yield_badge(label: str) -> str:
        label = str(label)
        if "15" in label or "25" in label:
            col = "#69f0ae"
        elif "10" in label or "18" in label:
            col = "#a5d6a7"
        elif "6" in label or "12" in label:
            col = "#FFC107"
        elif "3" in label or "8" in label:
            col = "#FF9800"
        else:
            col = "#8b949e"
        return f"<span style='color:{col};font-weight:bold'>{label}</span>"

    yf_link = lambda t: f"<a href='https://finance.yahoo.com/quote/{t}' target='_blank' style='color:#58a6ff;text-decoration:none'>{t}</a>"

    # ── Macro themes chips ────────────────────────────────────
    theme_chips = ""
    sentiment_colors = {
        "bullish": ("background:#0d3321;color:#69f0ae", "🟢"),
        "somewhat_bullish": ("background:#1e3320;color:#a5d6a7", "🔵"),
        "bearish": ("background:#3a0d0d;color:#ff8a80", "🔴"),
        "somewhat_bearish": ("background:#2a1a0a;color:#ef9a9a", "🟠"),
    }
    for t in macro_themes[:8]:
        emoji    = t.get("emoji", "")
        name     = t.get("theme", "")
        sent_key = t.get("sentiment", "bullish")
        style, _ = sentiment_colors.get(sent_key, ("background:#1e2430;color:#8b949e", "⚪"))
        detail   = t.get("detail", "")
        theme_chips += (
            f"<span style='{style};padding:4px 10px;border-radius:12px;font-size:12px;"
            f"margin:3px;display:inline-block;cursor:default' title='{detail}'>"
            f"{emoji} {name}</span>"
        )

    # ── Podium cards (#1, #2, #3) ─────────────────────────────
    podium_medals = ["🥇", "🥈", "🥉"]
    podium_html   = ""
    podium_rows   = top20_rows[:3]

    for i, row in enumerate(podium_rows):
        medal   = podium_medals[i]
        ticker  = str(row.get("ticker", ""))
        name    = str(row.get("name", ticker))[:35]
        atype   = str(row.get("asset_type", "STOCK"))
        signal  = str(row.get("signal", "HOLD"))
        score   = float(row.get("yield_potential_score", 0))
        sent_l  = str(row.get("sentiment_label", "⚪ Neutral"))
        yrange  = str(row.get("predicted_yield_range", ""))
        headline = str(row.get("top_headline", ""))[:90]
        themes  = str(row.get("theme_tags", ""))
        price   = float(row.get("price", 0))
        day_chg = float(row.get("day_chg_pct", 0) or 0)
        sector  = str(row.get("sector_or_category", ""))
        sc      = float(row.get("score_raw", 0))
        tech_c  = float(row.get("technical_component", 0))
        ml_c    = float(row.get("ml_or_momentum_component", 0))
        sent_c  = float(row.get("sentiment_component", 0))
        chg_col = "#69f0ae" if day_chg >= 0 else "#ff8a80"
        bg      = _score_bg(score)
        sc_col  = _sig_color(signal)
        theme_str = " · ".join(t for t in themes.split("|") if t)[:60] if themes else ""
        hl_esc  = headline.replace('"', '&quot;').replace("'", "&#39;")
        if atype == "STOCK":
            atype_label = "ml"
        elif atype == "CRYPTO":
            atype_label = "momentum vs BTC"
        else:
            atype_label = "momentum"

        podium_html += f"""
<div style="flex:1;min-width:220px;background:{bg};border:1px solid #30363d;
            border-radius:10px;padding:16px;position:relative;overflow:hidden">
  <div style="font-size:24px;line-height:1">{medal}</div>
  <div style="font-size:13px;color:#8b949e;margin:4px 0 2px">{_badge(atype)} {sector}</div>
  <div style="font-size:22px;font-weight:bold;color:#e6edf3">{yf_link(ticker)}</div>
  <div style="font-size:12px;color:#8b949e;margin:2px 0 8px">{name}</div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
    <span style="font-size:28px;font-weight:bold;color:{sc_col}">{score:.0f}</span>
    <span style="font-size:12px;color:#8b949e">/100<br>Yield Potential</span>
  </div>
  <div style="margin-bottom:6px">{_yield_badge(yrange)}</div>
  <div style="margin-bottom:6px">{_sent_badge(sent_l)}</div>
  <div style="font-size:11px;color:#484f58;margin:6px 0 4px">{hl_esc}</div>
  {f'<div style="font-size:10px;color:#484f58">🏷 {theme_str}</div>' if theme_str else ""}
  <div style="font-size:11px;color:#8b949e;margin-top:10px;border-top:1px solid #21262d;padding-top:8px">
    ${price:.2f} &nbsp;
    <span style="color:{chg_col}">{day_chg:+.2f}%</span> &nbsp;|&nbsp;
    Signal: <span style="color:{sc_col}">{signal}</span>
  </div>
  <div style="font-size:10px;color:#484f58;margin-top:4px">
    Tech {tech_c:.0f} + {atype_label.title()} {ml_c:.0f} + Sent {sent_c:.0f}
  </div>
</div>"""

    # ── Rank table (#4 onward) ────────────────────────────────
    table_rows = ""
    for row in top20_rows[3:]:
        rank    = int(row.get("rank", 0))
        ticker  = str(row.get("ticker", ""))
        name    = str(row.get("name", ticker))[:30]
        atype   = str(row.get("asset_type", "STOCK"))
        signal  = str(row.get("signal", "HOLD"))
        score   = float(row.get("yield_potential_score", 0))
        sent_l  = str(row.get("sentiment_label", "⚪ Neutral"))
        yrange  = str(row.get("predicted_yield_range", ""))
        headline = str(row.get("top_headline", ""))[:80]
        price   = float(row.get("price", 0))
        day_chg = float(row.get("day_chg_pct", 0) or 0)
        sector  = str(row.get("sector_or_category", ""))
        bg      = _score_bg(score)
        sc_col  = _sig_color(signal)
        chg_col = "#69f0ae" if day_chg >= 0 else "#ff8a80"
        hl_esc  = headline.replace('"', '&quot;').replace("'", "&#39;")

        table_rows += f"""
<tr style="background:{bg}">
  <td style="color:#8b949e;font-weight:bold;font-size:13px">{rank}</td>
  <td>{_badge(atype)}</td>
  <td>{yf_link(ticker)}<div style="font-size:10px;color:#8b949e">{sector[:22]}</div></td>
  <td style="font-size:11px;color:#8b949e">{name}</td>
  <td style="font-size:16px;font-weight:bold;color:{sc_col}">{score:.1f}</td>
  <td style="color:{sc_col};font-size:11px">{signal}</td>
  <td>{_sent_badge(sent_l)}</td>
  <td>{_yield_badge(yrange)}</td>
  <td style="font-size:10px;color:#484f58;max-width:200px" title="{hl_esc}">{hl_esc[:65]}{'…' if len(hl_esc)>65 else ''}</td>
  <td style="color:#8b949e;font-size:12px">${price:.2f} <span style="color:{chg_col}">{day_chg:+.1f}%</span></td>
</tr>"""

    # Count by type
    n_stock  = sum(1 for r in top20_rows if str(r.get("asset_type")) == "STOCK")
    n_etf    = sum(1 for r in top20_rows if str(r.get("asset_type")) == "ETF")
    n_crypto = sum(1 for r in top20_rows if str(r.get("asset_type")) == "CRYPTO")
    avg_sc   = sum(float(r.get("yield_potential_score", 0)) for r in top20_rows) / max(len(top20_rows), 1)

    return f"""
<div class="section">
  <h2>🏆 Top 20 High-Yield Predictions</h2>
  <p style="color:#8b949e;margin-top:-4px;margin-bottom:12px;font-size:13px">
    Ranked by Yield Potential Score (0–100) combining Technical strength,
    ML/Momentum signal, and News Sentiment &nbsp;|&nbsp;
    {n_stock} stocks · {n_etf} ETFs · {n_crypto} cryptos &nbsp;|&nbsp; Avg score: {avg_sc:.1f}
  </p>

  <!-- Macro Themes -->
  <div style="margin-bottom:18px">
    <span style="color:#8b949e;font-size:12px;margin-right:8px">📰 Market Themes:</span>
    {theme_chips if theme_chips else '<span style="color:#484f58;font-size:12px">Run Phase 7 to load live themes</span>'}
  </div>

  <!-- Podium cards -->
  <div style="display:flex;flex-wrap:wrap;gap:14px;margin-bottom:24px">
    {podium_html}
  </div>

  <!-- Rank table #4–#20 -->
  <h3 style="color:#8b949e;font-size:13px;margin-bottom:8px">Full Rankings — All 20 Picks</h3>
  <div style="overflow-x:auto">
  <table style="min-width:900px">
    <tr>
      <th>#</th><th>Type</th><th>Ticker</th><th>Name</th>
      <th>Score</th><th>Signal</th><th>Sentiment</th>
      <th>Yield Range</th><th>Top Headline</th><th>Price</th>
    </tr>
    {table_rows}
  </table>
  </div>

  <p style="color:#484f58;font-size:11px;margin-top:12px">
    Yield Potential Score: Stocks = Tech(50)+ML(30)+Sent(20) | ETFs = Tech(50)+Mom(30)+Sent(20) | Crypto = Tech(40)+Mom(25)+Quality(20)+Sent(15).
    Refresh: <code>python3 invest.py --crypto --news --top20</code>.
    Not financial advice.
  </p>
</div>"""


def generate_html_report(
        today_scores: dict,
        changes: list,
        pnl_rows: list,
        events: list,
        sector_map: dict,
        ml_preds: dict = None,
        etf_scores: dict = None,
        top20_rows: list = None,
        macro_themes: list = None,
        crypto_scores: dict = None,
        crypto_cycle: dict = None,
) -> str:
    """Build a self-contained HTML daily report."""

    if ml_preds is None:
        ml_preds = {}
    if etf_scores is None:
        etf_scores = {}
    if top20_rows is None:
        top20_rows = []
    if macro_themes is None:
        macro_themes = []
    if crypto_scores is None:
        crypto_scores = {}
    if crypto_cycle is None:
        crypto_cycle = {}

    def sig_color(sig):
        for k, v in SIGNAL_COLOR.items():
            if k in sig:
                return v
        return "#ffffff"

    # Top picks today (non-ETF, by score)
    top_picks = sorted(
        [(t, s) for t, s in today_scores.items() if "ETF" not in sector_map.get(t, "")],
        key=lambda x: -x[1]["score"]
    )[:10]

    # ── Signal Changes rows ───────────────────
    change_rows = ""
    for c in changes[:15]:
        color = "#00E676" if c["type"] == "UPGRADE" else (
                "#40C4FF" if c["type"] == "MACD_CROSS" else "#EF5350")
        change_rows += f"""
        <tr>
            <td>{ticker_link(c['ticker'])}</td>
            <td style="color:{color};font-weight:bold">{c['type']}</td>
            <td style="color:#aaa">{c['from']}</td>
            <td style="color:{color}">{c['to']}</td>
            <td style="color:#ccc">${c['price']}</td>
            <td style="color:#aaa">{c['note']}</td>
        </tr>"""

    # ── Top picks rows ────────────────────────
    picks_rows = ""
    for ticker, s in top_picks:
        color     = sig_color(s["signal"])
        chg_color = "#00E676" if s["day_chg_pct"] >= 0 else "#EF5350"
        picks_rows += f"""
        <tr>
            <td>{ticker_link(ticker)}</td>
            <td>${s['price']}</td>
            <td style="color:{chg_color}">{s['day_chg_pct']:+.2f}%</td>
            <td style="font-weight:bold">{s['score']}/75</td>
            <td style="color:{color};font-weight:bold">{s['signal']}</td>
            <td>{s['rsi']}</td>
            <td>{'✅' if s['above_200ma'] else '❌'}</td>
            <td>{s['bb_pct']:.0f}%</td>
            <td>{s['adx']:.0f}</td>
            <td>{'🔥' if s['vol_surge'] else f"{s['vol_ratio']}x"}</td>
        </tr>"""

    # ── Portfolio P&L rows ────────────────────
    pnl_html = ""
    if pnl_rows:
        total_mkt     = sum(r["mkt_val"] for r in pnl_rows)
        total_pnl     = sum(r["pnl"] for r in pnl_rows)
        pnl_rows_html = ""
        for r in pnl_rows:
            color  = "#00E676" if r["pnl"] >= 0 else "#EF5350"
            sig_c  = sig_color(r["signal"])
            alert_html = (
                f'<span style="color:#FFD740;font-size:11px">{r["alert"]}</span>'
                if r["alert"] else ""
            )
            pnl_rows_html += f"""
            <tr>
                <td>{ticker_link(r['ticker'])}</td>
                <td>{r['shares']}</td>
                <td>${r['avg_cost']:.2f}</td>
                <td>${r['price']:.2f}</td>
                <td>${r['mkt_val']:,.0f}</td>
                <td style="color:{color};font-weight:bold">${r['pnl']:+,.0f}</td>
                <td style="color:{color};font-weight:bold">{r['pnl_pct']:+.1f}%</td>
                <td style="color:{sig_c}">{r['signal']}</td>
                <td>{alert_html}</td>
            </tr>"""

        pnl_html = f"""
        <div class="section">
            <h2>💼 Portfolio P&amp;L</h2>
            <div class="summary-row">
                <div class="kpi">
                    <div class="kpi-val" style="color:{'#00E676' if total_pnl>=0 else '#EF5350'}">${total_pnl:+,.0f}</div>
                    <div class="kpi-label">Total Unrealised P&amp;L</div>
                </div>
                <div class="kpi">
                    <div class="kpi-val">${total_mkt:,.0f}</div>
                    <div class="kpi-label">Portfolio Market Value</div>
                </div>
            </div>
            <table>
                <tr>
                    <th>Ticker</th><th>Shares</th><th>Avg Cost</th><th>Price</th>
                    <th>Mkt Value</th><th>P&amp;L $</th><th>P&amp;L %</th><th>Signal</th><th>Alert</th>
                </tr>
                {pnl_rows_html}
            </table>
        </div>"""

    # ── Notable Events rows ───────────────────
    event_icons = {
        "BIG MOVE": "⚡", "52W HIGH": "🚀", "52W LOW": "🔻",
        "RSI OVERSOLD": "🟢", "RSI OVERBOUGHT": "🔴", "VOLUME SURGE": "🔥",
    }
    events_rows = ""
    for e in events[:20]:
        icon = event_icons.get(e["type"], "📌")
        events_rows += f"""
        <tr>
            <td>{ticker_link(e['ticker'])}</td>
            <td>${e['price']}</td>
            <td>{icon} <strong>{e['type']}</strong></td>
            <td style="color:#ccc">{e['detail']}</td>
        </tr>"""

    # ── ML 12-Month Predictions section ──────
    ml_horizons     = ["1m","2m","3m","4m","5m","6m","7m","8m","9m","10m","11m","12m"]
    key_horizons    = ["1m","3m","6m","12m"]   # shown in condensed summary strip
    ml_section_html = ""

    # Only show tickers that appear in today's top picks + user positions + events
    featured = set(t for t, _ in top_picks)
    featured |= set(r["ticker"] for r in pnl_rows)
    featured |= set(e["ticker"] for e in events[:20])
    featured_with_preds = [t for t in featured if t in ml_preds]

    if featured_with_preds:
        # Sort by 6m probability descending
        featured_with_preds.sort(
            key=lambda t: ml_preds[t].get("6m", 50), reverse=True
        )

        # ── Header row with all 12 months ──
        month_headers = "".join(f"<th>{h}</th>" for h in ml_horizons)

        # ── Data rows ──
        ml_rows = ""
        for ticker in featured_with_preds:
            preds  = ml_preds[ticker]
            score  = today_scores.get(ticker, {}).get("score", "—")
            signal = today_scores.get(ticker, {}).get("signal", "")

            # Mini sparkline: 12 coloured blocks, one per month
            spark_cells = ""
            for h in ml_horizons:
                prob = preds.get(h)
                if prob is None:
                    spark_cells += '<td style="color:#484f58;text-align:center">—</td>'
                    continue
                bg, fg = _prob_style(prob)
                label  = "🟢" if prob >= 55 else ("🔴" if prob <= 45 else "")
                spark_cells += (
                    f'<td style="background:{bg};color:{fg};text-align:center;'
                    f'font-size:12px;font-weight:600;padding:5px 4px;'
                    f'border-radius:3px" title="{h}: {prob:.1f}% P(Up)">'
                    f'{prob:.0f}%</td>'
                )

            # Condensed 4-point summary strip (1m / 3m / 6m / 12m)
            strip = ""
            for h in key_horizons:
                prob = preds.get(h)
                if prob is not None:
                    bg, fg = _prob_style(prob)
                    strip += (
                        f'<span style="background:{bg};color:{fg};padding:2px 7px;'
                        f'border-radius:10px;font-size:11px;font-weight:700;margin:0 2px">'
                        f'{h}&nbsp;{prob:.0f}%</span>'
                    )

            outlook = _outlook_label(preds)
            sig_c   = sig_color(signal)

            ml_rows += f"""
            <tr>
                <td style="white-space:nowrap">{ticker_link(ticker)}</td>
                <td style="color:{sig_c};font-size:11px">{signal}</td>
                {spark_cells}
                <td style="white-space:nowrap">{strip}</td>
                <td style="white-space:nowrap">{outlook}</td>
            </tr>"""

        # Build legend HTML
        legend_items = [
            ("#1a3a1a", "#69f0ae", "≥65% Bullish"),
            ("#1e3320", "#a5d6a7", "55–64% Mild Bull"),
            ("#1e2430", "#8b949e", "45–54% Neutral"),
            ("#3a1a1a", "#ef9a9a", "35–44% Mild Bear"),
            ("#4a0f0f", "#ff8a80", "≤34% Bearish"),
        ]
        legend_html = " &nbsp; ".join(
            f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:10px;font-size:11px">{label}</span>'
            for bg, fg, label in legend_items
        )

        ml_section_html = f"""
<div class="section">
  <h2>🤖 ML 12-Month Prediction Outlook
    <span style="font-size:11px;color:#8b949e;font-weight:400;margin-left:10px">
      Ensemble model · P(price higher) at each monthly horizon · Run Phase 5 to refresh
    </span>
  </h2>
  <div style="margin-bottom:12px">{legend_html}</div>
  <div style="overflow-x:auto">
  <table style="min-width:900px">
    <tr>
      <th style="min-width:70px">Ticker</th>
      <th style="min-width:110px">Signal</th>
      {month_headers}
      <th style="min-width:180px">Key Horizons</th>
      <th style="min-width:130px">Outlook</th>
    </tr>
    {ml_rows}
  </table>
  </div>
  <p style="color:#484f58;font-size:11px;margin-top:10px">
    ⚠️ Probabilities are model estimates, not guarantees. Refresh with: <code>python3 invest.py --ml</code>
  </p>
</div>"""

    elif ml_preds:
        # We have predictions but none for featured tickers — show a notice
        ml_section_html = """
<div class="section" style="color:#8b949e">
  <h2>🤖 ML 12-Month Predictions</h2>
  <p>Predictions loaded but no overlap with today's featured stocks.
     Run <code>python3 invest.py --ml</code> to generate predictions for the current watchlist.</p>
</div>"""
    else:
        # No predictions at all
        ml_section_html = """
<div class="section" style="color:#8b949e">
  <h2>🤖 ML 12-Month Predictions</h2>
  <p>No predictions available yet. Run <code>python3 invest.py --ml</code> to generate them.</p>
</div>"""

    # ── ETF Dashboard section ─────────────────
    etf_section_html  = _build_etf_section(etf_scores, sig_color)

    # ── Crypto Dashboard section ──────────────
    crypto_section_html = _build_crypto_section(crypto_scores, crypto_cycle)

    # ── Top 20 Predictions section ─────────────
    top20_section_html = _build_top20_section(top20_rows, macro_themes)

    # ── KPI summary counts ────────────────────
    n_strong_buy = sum(1 for s in today_scores.values() if "STRONG BUY" in s["signal"])
    n_buy        = sum(1 for s in today_scores.values() if s["signal"] == "🔵 BUY")
    n_avoid      = sum(1 for s in today_scores.values() if "AVOID" in s["signal"])
    avg_score    = np.mean([s["score"] for s in today_scores.values()])
    n_upgrades   = sum(1 for c in changes if c["type"] == "UPGRADE")
    n_downgrades = sum(1 for c in changes if c["type"] == "DOWNGRADE")
    ml_badge     = (f'<span style="background:#1e2d3d;color:#58a6ff;padding:2px 8px;'
                    f'border-radius:10px;font-size:11px">'
                    f'🤖 ML: {len(ml_preds)} tickers</span>'
                    if ml_preds else "")
    etf_badge    = (f'<span style="background:#1e2d1e;color:#69f0ae;padding:2px 8px;'
                    f'border-radius:10px;font-size:11px">'
                    f'📈 ETFs: {len(etf_scores)}</span>'
                    if etf_scores else "")

    now_str = datetime.now().strftime("%A, %B %d, %Y — %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Daily Monitor — {TODAY}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0d1117; color:#e6edf3; font-family:'Segoe UI',Helvetica,Arial,sans-serif; padding:24px; }}
  h1   {{ font-size:22px; color:#58a6ff; border-bottom:1px solid #30363d; padding-bottom:12px; margin-bottom:20px; }}
  h2   {{ font-size:16px; color:#79c0ff; margin:0 0 14px 0; }}
  h3   {{ font-size:13px; color:#8b949e; margin:16px 0 8px 0; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; }}
  .section {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:20px; margin-bottom:20px; }}
  .summary-row {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:18px; }}
  .kpi {{ background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:12px 20px; min-width:130px; }}
  .kpi-val {{ font-size:26px; font-weight:700; }}
  .kpi-label {{ font-size:11px; color:#8b949e; margin-top:4px; }}
  table {{ width:100%; border-collapse:separate; border-spacing:0 2px; font-size:13px; }}
  th {{ background:#21262d; color:#8b949e; padding:8px 10px; text-align:center; font-weight:600;
        border-bottom:1px solid #30363d; position:sticky; top:0; white-space:nowrap; }}
  th:first-child, td:first-child {{ text-align:left; }}
  td {{ padding:6px 8px; border-bottom:1px solid #21262d; vertical-align:middle; text-align:center; }}
  td:first-child {{ text-align:left; }}
  tr:hover td {{ background:#1c2128 !important; }}
  a {{ color:#58a6ff; text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
  code {{ background:#21262d; padding:1px 5px; border-radius:4px; font-size:12px; color:#e6edf3; }}
  .footer {{ color:#484f58; font-size:11px; text-align:center; margin-top:20px; }}
  .cat-chip {{ display:inline-block; padding:3px 10px; border-radius:12px; font-size:11px;
               font-weight:700; margin:0 3px 4px 0; white-space:nowrap; }}
  .heatmap-grid {{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:14px; }}
  .heatmap-cell {{ border-radius:6px; padding:6px 10px; min-width:70px; text-align:center; font-size:12px; }}
  .heatmap-cell .hm-ticker {{ font-weight:700; font-size:13px; }}
  .heatmap-cell .hm-ret {{ font-size:11px; margin-top:2px; }}
</style>
</head>
<body>

<h1>📊 Investment Intelligence — Daily Monitor &nbsp;|&nbsp; {now_str} &nbsp; {ml_badge} &nbsp; {etf_badge}</h1>

<div class="section">
  <h2>🔢 Market Overview</h2>
  <div class="summary-row">
    <div class="kpi"><div class="kpi-val" style="color:#00E676">{n_strong_buy}</div><div class="kpi-label">Strong Buy</div></div>
    <div class="kpi"><div class="kpi-val" style="color:#40C4FF">{n_buy}</div><div class="kpi-label">Buy</div></div>
    <div class="kpi"><div class="kpi-val" style="color:#EF5350">{n_avoid}</div><div class="kpi-label">Avoid</div></div>
    <div class="kpi"><div class="kpi-val">{avg_score:.0f}/75</div><div class="kpi-label">Avg Score</div></div>
    <div class="kpi"><div class="kpi-val" style="color:#00E676">▲{n_upgrades}</div><div class="kpi-label">Upgrades Today</div></div>
    <div class="kpi"><div class="kpi-val" style="color:#EF5350">▼{n_downgrades}</div><div class="kpi-label">Downgrades Today</div></div>
    <div class="kpi"><div class="kpi-val">{len(events)}</div><div class="kpi-label">Events Flagged</div></div>
  </div>
</div>

{"" if not changes else f'''
<div class="section">
  <h2>🔔 Signal Changes Since Yesterday</h2>
  <table>
    <tr><th>Ticker</th><th>Change</th><th>From</th><th>To</th><th>Price</th><th>Note</th></tr>
    {change_rows}
  </table>
</div>'''}

<div class="section">
  <h2>🏆 Top 10 Stocks Today</h2>
  <table>
    <tr>
      <th>Ticker</th><th>Price</th><th>Day Chg</th><th>Score</th><th>Signal</th>
      <th>RSI</th><th>200MA</th><th>BB%</th><th>ADX</th><th>Volume</th>
    </tr>
    {picks_rows}
  </table>
</div>

{pnl_html}

{ml_section_html}

{etf_section_html}

{crypto_section_html}

{top20_section_html}

{"" if not events else f'''
<div class="section">
  <h2>⚡ Notable Events &amp; Alerts</h2>
  <table>
    <tr><th>Ticker</th><th>Price</th><th>Event</th><th>Detail</th></tr>
    {events_rows}
  </table>
</div>'''}

<div class="footer">
  Generated by Investment Intelligence Stack — Phase 4c Daily Monitor &nbsp;|&nbsp;
  {len(today_scores)} stocks · {len(etf_scores)} ETFs · {len(crypto_scores) if crypto_scores else 0} cryptos scanned &nbsp;|&nbsp; {TODAY}<br>
  ⚠️ For research purposes only. Not financial advice.
</div>
</body></html>"""

    return html


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run_monitor():
    cfg = MONITOR_CONFIG
    print(f"\n{'='*60}")
    print(f"  INVESTMENT INTELLIGENCE — Daily Monitor")
    print(f"  {datetime.now().strftime('%A, %B %d, %Y  %H:%M')}")
    print(f"{'='*60}\n")

    # ── Build flat watchlist + sector map ─────
    all_tickers = []
    sector_map  = {}
    for sector, tickers in get_watchlist().items():
        for t in tickers:
            all_tickers.append(t)
            sector_map[t] = sector

    # ── Scan all stocks ───────────────────────
    print(f"  Scanning {len(all_tickers)} stocks...")
    today_scores = {}
    for i, ticker in enumerate(all_tickers):
        sys.stdout.write(f"\r  [{i+1}/{len(all_tickers)}] {ticker:<8}...")
        sys.stdout.flush()
        df = fetch_stock(ticker)
        if df is not None:
            today_scores[ticker] = compute_score(df)

    print(f"\n  ✅ Scored {len(today_scores)} stocks.\n")

    # ── Load history & detect changes ─────────
    history = load_history()
    changes = detect_signal_changes(today_scores, history)

    # ── Portfolio P&L ─────────────────────────
    pnl_rows = compute_portfolio_pnl(cfg["your_positions"], today_scores)

    # ── Notable events ────────────────────────
    events = find_notable_events(today_scores)

    # ── Print terminal summary ────────────────
    print(f"  {'─'*58}")
    print(f"  {'TICKER':<8} {'PRICE':>8} {'CHG%':>7} {'SCORE':>7} {'SIGNAL':<20} {'RSI':>5}")
    print(f"  {'─'*58}")
    sorted_stocks = sorted(today_scores.items(), key=lambda x: -x[1]["score"])
    for ticker, s in sorted_stocks[:15]:
        chg_sym = "▲" if s["day_chg_pct"] >= 0 else "▼"
        print(f"  {ticker:<8} ${s['price']:>7.2f}  {chg_sym}{abs(s['day_chg_pct']):>5.1f}%  "
              f"{s['score']:>5}/75  {s['signal']:<20}  {s['rsi']:>5.1f}")

    if changes:
        print(f"\n  🔔 SIGNAL CHANGES ({len(changes)}):")
        for c in changes[:8]:
            print(f"     {c['ticker']:<8} {c['type']:<12} {c['from']} → {c['to']}  ({c['note']})")

    if events:
        print(f"\n  ⚡ EVENTS ({len(events)}):")
        for e in events[:8]:
            print(f"     {e['ticker']:<8} {e['type']:<15}  {e['detail']}")

    if pnl_rows:
        total_pnl = sum(r["pnl"] for r in pnl_rows)
        print(f"\n  💼 PORTFOLIO P&L: ${total_pnl:+,.0f}")
        for r in pnl_rows:
            print(f"     {r['ticker']:<8} {r['pnl_pct']:>+6.1f}%  ${r['pnl']:>+8,.0f}  {r['signal']}")
            if r["alert"]:
                print(f"              {r['alert']}")

    # ── Save today's scores to history ────────
    history[TODAY] = {t: {k: v for k, v in s.items()} for t, s in today_scores.items()}
    # Keep only last 90 days of history
    cutoff = (date.today() - timedelta(days=90)).strftime("%Y-%m-%d")
    history = {d: v for d, v in history.items() if d >= cutoff}
    save_history(history)

    # ── Load ML predictions (Phase 5 output) ─
    ml_preds = load_ml_predictions()
    if ml_preds:
        print(f"  🤖 ML predictions loaded for {len(ml_preds)} tickers.")
    else:
        print(f"  ℹ️  No ML predictions found — run Phase 5 to generate them.")

    # ── Load ETF scores (Phase 4b output) ────
    etf_scores = load_etf_scores()
    if etf_scores:
        print(f"  📈 ETF scores loaded for {len(etf_scores)} ETFs.")
    else:
        print(f"  ℹ️  No ETF data found — run Phase 4b to generate it.")

    # ── Load Top 20 predictions (Phase 8 output) ──
    top20_rows   = load_top20_predictions()
    macro_themes = load_macro_themes()
    if top20_rows:
        print(f"  🏆 Top 20 predictions loaded ({len(top20_rows)} picks).")
    else:
        print(f"  ℹ️  No Top 20 data — run Phase 7+8 (--news --top20) to generate.")

    # ── Load Crypto scores (Phase 4c output) ──
    crypto_scores = load_crypto_scores()
    crypto_cycle  = load_crypto_cycle()
    if crypto_scores:
        print(f"  ₿  Crypto scores loaded for {len(crypto_scores)} coins.")
    else:
        print(f"  ℹ️  No crypto data found — run Phase 4c (--crypto) to generate.")

    # ── Generate HTML report ──────────────────
    html = generate_html_report(
        today_scores, changes, pnl_rows, events, sector_map,
        ml_preds, etf_scores, top20_rows, macro_themes,
        crypto_scores, crypto_cycle
    )
    report_path = os.path.join(REPORTS_DIR, f"{TODAY}-{RUN_SLOT}.html")
    with open(report_path, "w") as f:
        f.write(html)

    print(f"\n  📄 Report → {report_path}")
    print(f"  📦 History → {HISTORY_FILE}")

    # ── Email report (only if env vars are set) ───
    send_report_email(report_path, changes, events, today_scores)

    print(f"\n{'='*60}\n")

    return today_scores, changes, events


# ─────────────────────────────────────────────
#  EMAIL REPORT
#  Credentials are read from environment variables — never hardcoded.
#  Set these as GitHub Actions Secrets (or export locally for testing):
#    GMAIL_SENDER        — the Gmail address you send FROM
#    GMAIL_APP_PASSWORD  — 16-char App Password (not your login password)
#    REPORT_EMAIL        — destination address for the daily report
# ─────────────────────────────────────────────

def send_report_email(report_path: str, changes: list, events: list, today_scores: dict):
    """Send the HTML report by email. Silently skips if env vars are absent."""
    import smtplib
    import ssl
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    sender   = os.environ.get("GMAIL_SENDER", "").strip()
    password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
    to_addr  = os.environ.get("REPORT_EMAIL", "").strip()

    if not (sender and password and to_addr):
        print("  📧 Email skipped — GMAIL_SENDER / GMAIL_APP_PASSWORD / REPORT_EMAIL not set.")
        return

    slot_label = "Evening" if RUN_SLOT == "pm" else "Morning"
    subject = f"📊 Investment Monitor — {slot_label} Report {TODAY}"

    try:
        with open(report_path, "r") as f:
            html_body = f.read()

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = to_addr
        msg.attach(MIMEText(html_body, "html"))

        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
            server.login(sender, password)
            server.sendmail(sender, to_addr, msg.as_string())

        print(f"  📧 Report emailed → {to_addr}")

    except Exception as e:
        print(f"  ⚠️  Email failed: {e}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_monitor()
