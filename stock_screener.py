"""
============================================================
  INVESTMENT INTELLIGENCE — PHASE 1
  Stock Screener + Technical Analysis Engine
  Author: Claude (Vijay's Investment Project)
  Updated: 2026-04-12
============================================================

HOW TO RUN:
  python3 stock_screener.py

OUTPUT:
  - Terminal: ranked table of screened stocks with signals
  - stock_screener_results.csv   — full scored output
  - charts/<TICKER>_chart.html  — interactive chart per top pick

CUSTOMIZE:
  - Edit WATCHLIST to change stocks
  - Edit SCREENER_CONFIG to tighten/loosen filters
  - Edit WEIGHTS to shift scoring toward technical vs fundamental
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import sys
import json
import time

# ─────────────────────────────────────────────
#  CONFIGURATION  (loaded from config.py)
# ─────────────────────────────────────────────
from config import WATCHLIST, SCREENER_CONFIG, SCREENER_WEIGHTS as WEIGHTS, \
                   get_watchlist, DATA_DIR, CHARTS_DIR

OUTPUT_DIR = DATA_DIR   # CSVs written to data/


# ─────────────────────────────────────────────
#  DATA LAYER
# ─────────────────────────────────────────────

def fetch_price_history(ticker: str, period: str = "1y") -> pd.DataFrame | None:
    """Download OHLCV price history from Yahoo Finance."""
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty or len(df) < 60:
            return None
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        return None


def fetch_fundamentals(ticker: str) -> dict:
    """Fetch fundamental data: PE, growth, margins, market cap, debt."""
    defaults = {
        "pe_ratio": None, "forward_pe": None, "peg_ratio": None,
        "revenue_growth": None, "earnings_growth": None,
        "profit_margin": None, "operating_margin": None,
        "debt_to_equity": None, "market_cap": None,
        "dividend_yield": None, "sector": "Unknown",
        "short_name": ticker,
    }
    try:
        info = yf.Ticker(ticker).info
        return {
            "pe_ratio":        info.get("trailingPE"),
            "forward_pe":      info.get("forwardPE"),
            "peg_ratio":       info.get("pegRatio"),
            "revenue_growth":  info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "profit_margin":   info.get("profitMargins"),
            "operating_margin":info.get("operatingMargins"),
            "debt_to_equity":  info.get("debtToEquity"),
            "market_cap":      info.get("marketCap"),
            "dividend_yield":  info.get("dividendYield"),
            "sector":          info.get("sector", "Unknown"),
            "short_name":      info.get("shortName", ticker),
        }
    except Exception:
        return defaults


# ─────────────────────────────────────────────
#  TECHNICAL INDICATORS
# ─────────────────────────────────────────────

def compute_technicals(df: pd.DataFrame) -> dict:
    """
    Compute all technical indicators and return a flat dict of values/signals.
    Uses the closing prices from the last available bar.
    """
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    result = {}

    # ── RSI (14-period) ────────────────────────
    rsi_ind = ta.momentum.RSIIndicator(close=close, window=14)
    rsi = rsi_ind.rsi()
    result["rsi"] = round(rsi.iloc[-1], 2)
    result["rsi_prev"] = round(rsi.iloc[-2], 2)

    # ── MACD (12/26/9) ────────────────────────
    macd_ind = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    macd_line   = macd_ind.macd()
    signal_line = macd_ind.macd_signal()
    macd_hist   = macd_ind.macd_diff()
    result["macd"]        = round(macd_line.iloc[-1], 4)
    result["macd_signal"] = round(signal_line.iloc[-1], 4)
    result["macd_hist"]   = round(macd_hist.iloc[-1], 4)
    result["macd_hist_prev"] = round(macd_hist.iloc[-2], 4)
    # Bullish crossover: MACD crossed above signal in last 3 bars
    result["macd_bullish_cross"] = bool(
        (macd_line.iloc[-1] > signal_line.iloc[-1]) and
        (macd_line.iloc[-3] < signal_line.iloc[-3])
    )

    # ── Moving Averages ───────────────────────
    sma20  = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    sma50  = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
    sma200 = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
    ema12  = ta.trend.EMAIndicator(close=close, window=12).ema_indicator()
    ema26  = ta.trend.EMAIndicator(close=close, window=26).ema_indicator()

    price = close.iloc[-1]
    result["price"]   = round(price, 2)
    result["sma20"]   = round(sma20.iloc[-1], 2)
    result["sma50"]   = round(sma50.iloc[-1], 2)
    result["sma200"]  = round(sma200.iloc[-1], 2)

    result["above_sma20"]  = bool(price > sma20.iloc[-1])
    result["above_sma50"]  = bool(price > sma50.iloc[-1])
    result["above_sma200"] = bool(price > sma200.iloc[-1])

    # Golden cross: SMA50 crossed above SMA200 in last 10 days
    golden_cross = False
    for i in range(-10, -1):
        if (sma50.iloc[i] > sma200.iloc[i]) and (sma50.iloc[i-1] < sma200.iloc[i-1]):
            golden_cross = True
    result["golden_cross"] = golden_cross

    # Price distance from 52-week high/low
    result["52w_high"] = round(close.tail(252).max(), 2)
    result["52w_low"]  = round(close.tail(252).min(), 2)
    result["pct_from_52w_high"] = round((price / result["52w_high"] - 1) * 100, 2)
    result["pct_from_52w_low"]  = round((price / result["52w_low"]  - 1) * 100, 2)

    # ── Bollinger Bands (20/2) ─────────────────
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_mid   = bb.bollinger_mavg()
    bb_width = bb.bollinger_wband()

    result["bb_upper"] = round(bb_upper.iloc[-1], 2)
    result["bb_lower"] = round(bb_lower.iloc[-1], 2)
    result["bb_mid"]   = round(bb_mid.iloc[-1], 2)
    result["bb_pct"]   = round(bb.bollinger_pband().iloc[-1] * 100, 2)  # 0=lower, 100=upper
    result["bb_squeeze"] = bool(bb_width.iloc[-1] < bb_width.tail(50).mean() * 0.7)  # bandwidth compression

    # ── ADX — Trend Strength ──────────────────
    adx_ind = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
    result["adx"] = round(adx_ind.adx().iloc[-1], 2)
    # ADX > 25 = trending, > 40 = strong trend

    # ── Volume Analysis ───────────────────────
    vol_sma20 = vol.rolling(20).mean()
    result["volume"]         = int(vol.iloc[-1])
    result["volume_avg20"]   = int(vol_sma20.iloc[-1])
    result["volume_ratio"]   = round(vol.iloc[-1] / vol_sma20.iloc[-1], 2)
    result["volume_surge"]   = bool(result["volume_ratio"] > 1.5)

    # On-Balance Volume trend
    obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=vol).on_balance_volume()
    result["obv_trend_up"] = bool(obv.iloc[-1] > obv.rolling(20).mean().iloc[-1])

    # ── Stochastic RSI ─────────────────────────
    try:
        stoch = ta.momentum.StochRSIIndicator(close=close, window=14, smooth1=3, smooth2=3)
        result["stoch_rsi_k"] = round(stoch.stochrsi_k().iloc[-1] * 100, 2)
        result["stoch_rsi_d"] = round(stoch.stochrsi_d().iloc[-1] * 100, 2)
    except Exception:
        result["stoch_rsi_k"] = 50
        result["stoch_rsi_d"] = 50

    return result


# ─────────────────────────────────────────────
#  SCORING ENGINE
# ─────────────────────────────────────────────

def score_technicals(t: dict) -> tuple[float, list[str]]:
    """
    Technical score 0–60.
    Returns (score, list_of_signals).
    """
    score = 0.0
    signals = []

    # RSI scoring (0–15 pts)
    rsi = t["rsi"]
    if 40 <= rsi <= 60:
        score += 8   # Neutral momentum — healthy
        signals.append(f"RSI neutral ({rsi:.1f})")
    elif 30 <= rsi < 40:
        score += 15  # Oversold bounce zone — bullish
        signals.append(f"📈 RSI oversold bounce zone ({rsi:.1f})")
    elif rsi < 30:
        score += 12  # Deeply oversold — potentially bottoming
        signals.append(f"🟢 RSI deeply oversold ({rsi:.1f}) — watch for reversal")
    elif 60 < rsi <= 70:
        score += 6   # Bullish momentum but getting hot
        signals.append(f"RSI bullish momentum ({rsi:.1f})")
    else:
        score += 2   # Overbought
        signals.append(f"🔴 RSI overbought ({rsi:.1f}) — caution")

    # MACD scoring (0–12 pts)
    if t["macd_bullish_cross"]:
        score += 12
        signals.append("📈 MACD bullish crossover (strong buy signal)")
    elif t["macd"] > t["macd_signal"] and t["macd_hist"] > 0:
        score += 8
        signals.append("MACD above signal line (bullish)")
    elif t["macd_hist"] > t["macd_hist_prev"]:
        score += 5
        signals.append("MACD histogram expanding (momentum building)")
    else:
        score += 1
        signals.append("MACD bearish")

    # Moving Average scoring (0–15 pts)
    ma_score = 0
    if t["above_sma200"]: ma_score += 5
    if t["above_sma50"]:  ma_score += 5
    if t["above_sma20"]:  ma_score += 3
    if t["golden_cross"]: ma_score += 7; signals.append("✨ Golden Cross detected!")
    score += min(ma_score, 15)
    ma_count = sum([t["above_sma200"], t["above_sma50"], t["above_sma20"]])
    signals.append(f"Price above {ma_count}/3 key MAs (20/50/200)")

    # Bollinger Band scoring (0–8 pts)
    bb_pct = t["bb_pct"]
    if 20 <= bb_pct <= 60:
        score += 8   # In healthy middle-lower zone
        signals.append(f"BB position healthy ({bb_pct:.0f}%)")
    elif bb_pct < 20:
        score += 6   # Near lower band — oversold / mean reversion candidate
        signals.append(f"Near BB lower band ({bb_pct:.0f}%) — mean reversion watch")
    elif bb_pct > 90:
        score += 1
        signals.append(f"🔴 Near BB upper band ({bb_pct:.0f}%) — stretched")
    else:
        score += 4
    if t["bb_squeeze"]:
        score += 3
        signals.append("⚡ Bollinger Band squeeze — breakout potential")

    # ADX / Trend strength (0–5 pts)
    adx = t["adx"]
    if adx > 40:
        score += 5; signals.append(f"Very strong trend (ADX {adx:.0f})")
    elif adx > 25:
        score += 3; signals.append(f"Trending market (ADX {adx:.0f})")
    else:
        score += 1; signals.append(f"Weak/no trend (ADX {adx:.0f})")

    # Volume (0–5 pts)
    if t["volume_surge"] and t["obv_trend_up"]:
        score += 5; signals.append("📊 High volume + rising OBV (accumulation)")
    elif t["obv_trend_up"]:
        score += 3; signals.append("OBV rising (quiet accumulation)")
    elif t["volume_surge"]:
        score += 2; signals.append("Volume surge (confirm direction)")

    return round(min(score, 60), 2), signals


def score_fundamentals(f: dict) -> tuple[float, list[str]]:
    """
    Fundamental score 0–40.
    Returns (score, list_of_signals).
    """
    score = 0.0
    signals = []

    # Valuation — PE ratio (0–12 pts)
    pe = f.get("pe_ratio")
    fpe = f.get("forward_pe")
    if pe is not None and pe > 0:
        if pe < 15:
            score += 12; signals.append(f"💚 Very low PE ({pe:.1f}) — value territory")
        elif pe < 25:
            score += 10; signals.append(f"Reasonable PE ({pe:.1f})")
        elif pe < 40:
            score += 7;  signals.append(f"Moderate PE ({pe:.1f})")
        elif pe < 70:
            score += 4;  signals.append(f"High PE ({pe:.1f}) — priced for growth")
        else:
            score += 1;  signals.append(f"🔴 Very high PE ({pe:.1f})")
    elif fpe is not None and fpe > 0:
        if fpe < 25:
            score += 9; signals.append(f"Low forward PE ({fpe:.1f})")
        elif fpe < 40:
            score += 6; signals.append(f"Moderate forward PE ({fpe:.1f})")
        else:
            score += 2; signals.append(f"High forward PE ({fpe:.1f})")

    # Growth (0–12 pts)
    rev_growth = f.get("revenue_growth")
    earn_growth = f.get("earnings_growth")
    growth_pts = 0
    if rev_growth is not None:
        if rev_growth > 0.30:
            growth_pts += 6; signals.append(f"🚀 Revenue growth {rev_growth*100:.0f}% YoY")
        elif rev_growth > 0.15:
            growth_pts += 4; signals.append(f"Revenue growth {rev_growth*100:.0f}% YoY")
        elif rev_growth > 0.05:
            growth_pts += 2; signals.append(f"Revenue growth {rev_growth*100:.0f}% YoY")
        else:
            signals.append(f"Slow revenue growth ({rev_growth*100:.1f}%)")
    if earn_growth is not None:
        if earn_growth > 0.25:
            growth_pts += 6; signals.append(f"🚀 Earnings growth {earn_growth*100:.0f}% YoY")
        elif earn_growth > 0.10:
            growth_pts += 3; signals.append(f"Earnings growth {earn_growth*100:.0f}% YoY")
    score += min(growth_pts, 12)

    # Profitability (0–8 pts)
    margin = f.get("profit_margin")
    if margin is not None:
        if margin > 0.30:
            score += 8; signals.append(f"Excellent profit margin ({margin*100:.0f}%)")
        elif margin > 0.15:
            score += 5; signals.append(f"Good profit margin ({margin*100:.0f}%)")
        elif margin > 0.05:
            score += 3; signals.append(f"Thin profit margin ({margin*100:.0f}%)")
        else:
            score += 0; signals.append(f"⚠️ Negative/minimal margin ({margin*100:.1f}%)")

    # Balance sheet — Debt/Equity (0–8 pts)
    de = f.get("debt_to_equity")
    if de is not None:
        if de < 30:
            score += 8; signals.append(f"💚 Very low debt (D/E {de:.0f})")
        elif de < 80:
            score += 5; signals.append(f"Manageable debt (D/E {de:.0f})")
        elif de < 150:
            score += 2; signals.append(f"Moderate debt (D/E {de:.0f})")
        else:
            score += 0; signals.append(f"🔴 High debt (D/E {de:.0f})")

    return round(min(score, 40), 2), signals


def generate_signal(tech_score: float, fund_score: float, t: dict) -> str:
    total = tech_score + fund_score
    rsi = t["rsi"]
    if total >= 75 and rsi < 72:
        return "🟢 STRONG BUY"
    elif total >= 60 and rsi < 72:
        return "🔵 BUY"
    elif total >= 45:
        return "🟡 HOLD / WATCH"
    elif total >= 30:
        return "🟠 WEAK / CAUTION"
    else:
        return "🔴 AVOID"


# ─────────────────────────────────────────────
#  CHARTING
# ─────────────────────────────────────────────

def build_chart(ticker: str, df: pd.DataFrame, t: dict, f: dict, tech_score: float, fund_score: float):
    """
    Build a 4-panel interactive Plotly chart:
      1. Candlestick + MAs + Bollinger Bands
      2. Volume bars
      3. RSI
      4. MACD
    """
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    open_ = df["Open"].squeeze()
    vol   = df["Volume"].squeeze()

    # Recalculate indicators for full chart
    sma20  = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    sma50  = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
    sma200 = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
    bb     = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    rsi_s  = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd_i = ta.trend.MACD(close=close)
    macd_l = macd_i.macd()
    macd_s = macd_i.macd_signal()
    macd_h = macd_i.macd_diff()

    name = f.get("short_name", ticker)
    total_score = round(tech_score + fund_score, 1)
    signal = generate_signal(tech_score, fund_score, t)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.50, 0.15, 0.17, 0.18],
        subplot_titles=(
            f"{ticker} — {name}  |  Score: {total_score}/100  |  {signal}",
            "Volume",
            "RSI (14)",
            "MACD (12/26/9)"
        )
    )

    # Panel 1: Candlestick + MAs + BB
    fig.add_trace(go.Candlestick(
        x=df.index, open=open_, high=high, low=low, close=close,
        name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=sma20,  name="SMA 20",  line=dict(color="#FB8C00", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sma50,  name="SMA 50",  line=dict(color="#1E88E5", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sma200, name="SMA 200", line=dict(color="#E91E63", width=1.8, dash="dot")), row=1, col=1)
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index, y=bb.bollinger_hband(), name="BB Upper",
        line=dict(color="rgba(100,100,200,0.5)", width=1, dash="dash"), showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=bb.bollinger_lband(), name="BB Lower",
        line=dict(color="rgba(100,100,200,0.5)", width=1, dash="dash"),
        fill="tonexty", fillcolor="rgba(100,100,200,0.05)", showlegend=False
    ), row=1, col=1)

    # Panel 2: Volume
    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(close, open_)]
    fig.add_trace(go.Bar(x=df.index, y=vol, name="Volume", marker_color=colors, showlegend=False), row=2, col=1)
    vol_avg = vol.rolling(20).mean()
    fig.add_trace(go.Scatter(x=df.index, y=vol_avg, name="Vol MA20", line=dict(color="orange", width=1.2)), row=2, col=1)

    # Panel 3: RSI
    fig.add_trace(go.Scatter(x=df.index, y=rsi_s, name="RSI", line=dict(color="#7B1FA2", width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",   opacity=0.6, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.6, row=3, col=1)
    fig.add_hline(y=50, line_dash="dot",  line_color="gray",  opacity=0.4, row=3, col=1)

    # Panel 4: MACD
    fig.add_trace(go.Scatter(x=df.index, y=macd_l, name="MACD",   line=dict(color="#1565C0", width=1.5)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_s, name="Signal", line=dict(color="#E53935", width=1.5)), row=4, col=1)
    hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in macd_h]
    fig.add_trace(go.Bar(x=df.index, y=macd_h, name="Histogram", marker_color=hist_colors, showlegend=False), row=4, col=1)

    # Fundamental info annotation
    pe_str  = f"PE: {f['pe_ratio']:.1f}" if f.get("pe_ratio") else "PE: N/A"
    mcap    = f.get("market_cap")
    mc_str  = f"MCap: ${mcap/1e9:.0f}B" if mcap else ""
    mgn_str = f"Margin: {f['profit_margin']*100:.1f}%" if f.get("profit_margin") else ""
    sector  = f.get("sector", "")
    annotation_text = "  |  ".join(filter(None, [sector, pe_str, mc_str, mgn_str,
                                                   f"Tech: {tech_score:.0f}/60",
                                                   f"Fund: {fund_score:.0f}/40"]))

    fig.update_layout(
        height=950,
        template="plotly_dark",
        title=dict(
            text=f"<b>{ticker}</b> — {annotation_text}",
            font=dict(size=13)
        ),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=80, b=40),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume",   row=2, col=1)
    fig.update_yaxes(title_text="RSI",      row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD",     row=4, col=1)

    path = os.path.join(CHARTS_DIR, f"{ticker}_chart.html")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


# ─────────────────────────────────────────────
#  SCREENER RUNNER
# ─────────────────────────────────────────────

def run_screener():
    watchlist = get_watchlist()
    all_tickers = []
    ticker_sector = {}
    for sector, tickers in watchlist.items():
        for t in tickers:
            all_tickers.append(t)
            ticker_sector[t] = sector

    print(f"\n{'='*65}")
    print(f"  INVESTMENT INTELLIGENCE — Stock Screener + Technical Analysis")
    print(f"  Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Scanning {len(all_tickers)} stocks across {len(watchlist)} sectors...")
    print(f"{'='*65}\n")

    results = []

    for i, ticker in enumerate(all_tickers):
        sys.stdout.write(f"\r  [{i+1}/{len(all_tickers)}] Fetching {ticker:<8}...")
        sys.stdout.flush()

        df = fetch_price_history(ticker, period="1y")
        if df is None:
            continue

        # Basic filters
        price = df["Close"].squeeze().iloc[-1]
        avg_vol = df["Volume"].squeeze().rolling(20).mean().iloc[-1]
        if price < SCREENER_CONFIG["min_price"]:
            continue
        if avg_vol < SCREENER_CONFIG["min_avg_volume"]:
            continue

        fund = fetch_fundamentals(ticker)
        mcap = fund.get("market_cap") or 0
        if mcap > 0 and mcap < SCREENER_CONFIG["min_market_cap_B"] * 1e9:
            continue

        tech = compute_technicals(df)
        t_score, t_signals = score_technicals(tech)
        f_score, f_signals = score_fundamentals(fund)
        total = round(t_score + f_score, 1)
        signal = generate_signal(t_score, f_score, tech)

        results.append({
            "ticker":       ticker,
            "sector":       ticker_sector.get(ticker, fund.get("sector", "—")),
            "name":         fund.get("short_name", ticker),
            "price":        tech["price"],
            "52w_high":     tech["52w_high"],
            "pct_from_52w": tech["pct_from_52w_high"],
            "rsi":          tech["rsi"],
            "adx":          tech["adx"],
            "macd_cross":   tech["macd_bullish_cross"],
            "bb_pct":       tech["bb_pct"],
            "vol_ratio":    tech["volume_ratio"],
            "above_200ma":  tech["above_sma200"],
            "pe_ratio":     fund.get("pe_ratio"),
            "rev_growth":   round(fund["revenue_growth"] * 100, 1) if fund.get("revenue_growth") else None,
            "profit_margin":round(fund["profit_margin"] * 100, 1) if fund.get("profit_margin") else None,
            "market_cap_B": round(mcap / 1e9, 1) if mcap else None,
            "tech_score":   t_score,
            "fund_score":   f_score,
            "total_score":  total,
            "signal":       signal,
            "tech_signals": t_signals,
            "fund_signals": f_signals,
            "_df":          df,
            "_tech":        tech,
            "_fund":        fund,
        })

        time.sleep(0.3)  # be polite to Yahoo Finance

    print(f"\n\n  ✅ Screened {len(results)} stocks successfully.\n")

    # Sort by total score
    results.sort(key=lambda x: x["total_score"], reverse=True)

    # ── Print Results Table ───────────────────
    print(f"{'='*95}")
    print(f"  {'RANK':<5} {'TICKER':<8} {'NAME':<22} {'PRICE':>7} {'SCORE':>6} {'RSI':>6} {'ADX':>6} {'SIGNAL':<20} {'SECTOR'}")
    print(f"{'─'*95}")

    for rank, r in enumerate(results, 1):
        rev_g = f"{r['rev_growth']:+.0f}%" if r["rev_growth"] is not None else "  N/A"
        print(
            f"  {rank:<5} {r['ticker']:<8} {r['name'][:22]:<22} "
            f"${r['price']:>7.2f} {r['total_score']:>5.1f} {r['rsi']:>6.1f} {r['adx']:>5.1f}  "
            f"{r['signal']:<20} {r['sector']}"
        )

    print(f"{'='*95}")
    print(f"\n  📊 DETAILED SIGNALS — TOP 10 STOCKS\n")

    for r in results[:10]:
        print(f"\n  ┌── {r['ticker']} ({r['name']})  Score: {r['total_score']}/100  {r['signal']}")
        print(f"  │   Price: ${r['price']}  |  RSI: {r['rsi']}  |  ADX: {r['adx']}  |  BB%: {r['bb_pct']}%")
        pe_str = f"{r['pe_ratio']:.1f}" if r['pe_ratio'] else "N/A"
        mg_str = f"{r['profit_margin']}%" if r['profit_margin'] else "N/A"
        rv_str = f"{r['rev_growth']:+.0f}%" if r['rev_growth'] is not None else "N/A"
        mc_str = f"${r['market_cap_B']:.0f}B" if r['market_cap_B'] else "N/A"
        print(f"  │   PE: {pe_str}  |  Margin: {mg_str}  |  Rev Growth: {rv_str}  |  MCap: {mc_str}")
        print(f"  │   Above 200MA: {'✅' if r['above_200ma'] else '❌'}  "
              f"  MACD Cross: {'✅' if r['macd_cross'] else '—'}  "
              f"  Vol Ratio: {r['vol_ratio']}x")
        print(f"  │  Technical signals:")
        for s in r["tech_signals"][:5]:
            print(f"  │    · {s}")
        print(f"  │  Fundamental signals:")
        for s in r["fund_signals"][:4]:
            print(f"  │    · {s}")
        print(f"  └{'─'*70}")

    # ── Save CSV ──────────────────────────────
    csv_cols = [
        "rank", "ticker", "name", "sector", "price", "52w_high", "pct_from_52w",
        "rsi", "adx", "bb_pct", "vol_ratio", "above_200ma", "macd_cross",
        "pe_ratio", "rev_growth", "profit_margin", "market_cap_B",
        "tech_score", "fund_score", "total_score", "signal"
    ]
    df_out = pd.DataFrame(results)
    df_out.insert(0, "rank", range(1, len(df_out) + 1))
    csv_path = os.path.join(OUTPUT_DIR, "stock_screener_results.csv")
    df_out[csv_cols].to_csv(csv_path, index=False)
    print(f"\n  💾 Full results saved → {csv_path}")

    # ── Generate Charts for Top N ─────────────
    top_n = SCREENER_CONFIG["top_n_chart"]
    print(f"\n  📈 Generating interactive charts for top {top_n} stocks...\n")
    chart_paths = []
    for r in results[:top_n]:
        try:
            path = build_chart(r["ticker"], r["_df"], r["_tech"], r["_fund"],
                               r["tech_score"], r["fund_score"])
            chart_paths.append(path)
            print(f"    ✅ {r['ticker']} → {path}")
        except Exception as e:
            print(f"    ⚠️  {r['ticker']} chart failed: {e}")

    print(f"\n{'='*65}")
    print(f"  🏁 SCREENING COMPLETE")
    print(f"  Results: {csv_path}")
    print(f"  Charts:  {CHARTS_DIR}/")
    print(f"{'='*65}\n")

    return results


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    results = run_screener()

    print("\n  ⚠️  DISCLAIMER: This tool is for educational/research purposes only.")
    print("     It does not constitute financial advice. Always do your own due")
    print("     diligence and consult a licensed financial advisor before investing.\n")
