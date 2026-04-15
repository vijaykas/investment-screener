"""
============================================================
  INVESTMENT INTELLIGENCE — PHASE 4c
  Crypto Screener & Scorer
  Author: Claude (Vijay's Investment Project)
  Updated: 2026-04-13
============================================================

WHAT THIS DOES:
  Scores every cryptocurrency in CRYPTO_WATCHLIST across four dimensions.
  Only quality, utility-bearing assets are tracked — meme coins excluded
  by design through curated category lists and a hard exclusion filter.

  SCORING BREAKDOWN (0–100 total):
  ┌─────────────────────────────────────────────────────────┐
  │  Technical  (40 pts) — RSI, MACD, MA crossovers,       │
  │                         Bollinger, ADX, volume surge    │
  │  Momentum   (25 pts) — 7d / 30d / 90d returns vs BTC   │
  │                         (relative outperformance)       │
  │  Quality    (20 pts) — Market cap tier + network type   │
  │                         + volume-to-cap activity ratio  │
  │  Sentiment  (15 pts) — News sentiment from Phase 7      │
  │                         (or signal-derived if no data)  │
  └─────────────────────────────────────────────────────────┘

  SIGNAL LABELS:
    ≥78 🟢 STRONG BUY  |  ≥62 🔵 BUY  |  ≥46 🟡 HOLD
    ≥30 🟠 CAUTION     |   <30 🔴 AVOID

  MARKET CYCLE CONTEXT:
    Bull (BTC > 200 DMA) / Caution (BTC 0–5% above 200 DMA) /
    Bear (BTC < 200 DMA)

OUTPUT:
  data/crypto_screener_results.csv  — one row per coin with all metrics

HOW TO RUN:
  python3 crypto_screener.py
  python3 invest.py --crypto         (via the main launcher)
  python3 invest.py --quick          (included in the quick pipeline)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CRYPTO_WATCHLIST, CRYPTO_CONFIG as CFG, DATA_DIR

OUTPUT_FILE = os.path.join(DATA_DIR, CFG["output_file"])
BENCHMARK   = CFG["benchmark"]   # "BTC-USD"
TODAY_STR   = date.today().strftime("%Y-%m-%d")

# Hard exclusion list — any ticker matching these strings is skipped
# regardless of which watchlist it appears in
_MEME_EXCLUSIONS = {
    "DOGE-USD", "SHIB-USD", "PEPE-USD", "FLOKI-USD", "WIF-USD",
    "BONK-USD", "MEME-USD", "ELON-USD", "BABYDOGE-USD", "SAMO-USD",
    "SNEK-USD", "TURBO-USD", "WOJAK-USD", "LADYS-USD", "MOG-USD",
}


# ─────────────────────────────────────────────
#  DATA FETCH
# ─────────────────────────────────────────────

def fetch_crypto(ticker: str, years: int = 2) -> pd.DataFrame | None:
    """Download up to 2 years of daily price data for a crypto asset."""
    try:
        df = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)
        if df.empty or len(df) < 60:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None


def fetch_crypto_info(ticker: str) -> dict:
    """
    Fetch crypto metadata: market cap, 52w high/low, name.
    Returns a dict with defaults on failure.
    """
    defaults = {"market_cap_B": 0.0, "name": ticker.replace("-USD", ""),
                "52w_high": None, "52w_low": None}
    try:
        info = yf.Ticker(ticker).info
        mc = info.get("marketCap") or 0
        return {
            "market_cap_B": mc / 1e9,
            "name":         info.get("shortName") or info.get("name") or ticker.replace("-USD", ""),
            "52w_high":     info.get("fiftyTwoWeekHigh"),
            "52w_low":      info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        return defaults


# ─────────────────────────────────────────────
#  1. TECHNICAL SCORE  (0–40 pts)
# ─────────────────────────────────────────────

def score_technical(df: pd.DataFrame) -> dict:
    """
    Compute technical score (0–40) adapted for crypto volatility.
    Returns dict of score + all indicator values.
    """
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    price      = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    day_chg_pct = (price / prev_close - 1) * 100

    # ── RSI (14-period) ─────────────────────────────────────────────
    rsi_ind = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    rsi     = float(rsi_ind.iloc[-1])

    rsi_cfg = CFG["rsi"]
    if rsi <= rsi_cfg["strong_buy"]:     rsi_score = 10
    elif rsi <= rsi_cfg["buy"]:          rsi_score = 8
    elif rsi <= rsi_cfg["neutral_low"]:  rsi_score = 6
    elif rsi <= rsi_cfg["neutral_high"]: rsi_score = 4
    elif rsi <= rsi_cfg["caution"]:      rsi_score = 2
    else:                                rsi_score = 0

    # ── MACD ────────────────────────────────────────────────────────
    macd_ind    = ta.trend.MACD(close=close)
    macd_line   = macd_ind.macd()
    sig_line    = macd_ind.macd_signal()
    macd_hist   = macd_ind.macd_diff()
    macd_bull   = bool(macd_line.iloc[-1] > sig_line.iloc[-1])
    macd_cross  = bool(macd_line.iloc[-1] > sig_line.iloc[-1] and
                       macd_line.iloc[-3] < sig_line.iloc[-3])

    macd_score  = 8 if macd_cross else (5 if macd_bull else 0)

    # ── Moving Averages ─────────────────────────────────────────────
    sma50_s  = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
    sma200_s = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
    sma50    = float(sma50_s.iloc[-1])
    sma200   = float(sma200_s.iloc[-1])
    above_50   = bool(price > sma50)
    above_200  = bool(price > sma200)
    golden_cross = bool(sma50 > sma200)

    ma_score = (4 if above_200 else 0) + (2 if above_50 else 0) + (2 if golden_cross else 0)

    # ── Bollinger Bands ─────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    bb_pct = float(bb.bollinger_pband().iloc[-1]) * 100

    bb_score = 5 if bb_pct < 20 else (3 if bb_pct < 40 else (0 if bb_pct > 80 else 2))

    # ── ADX (trend strength) ─────────────────────────────────────────
    adx    = float(ta.trend.ADXIndicator(high=high, low=low, close=close).adx().iloc[-1])
    adx_score = 5 if adx > 40 else (3 if adx > 25 else (1 if adx > 15 else 0))

    # ── Volume surge ─────────────────────────────────────────────────
    vol_avg20 = float(vol.rolling(20).mean().iloc[-1])
    vol_ratio = float(vol.iloc[-1]) / vol_avg20 if vol_avg20 > 0 else 1.0
    vol_surge = vol_ratio > CFG["volume_surge_ratio"]
    vol_score = min(5, int(vol_ratio * 1.5)) if vol_surge else 0

    # ── 52-week extremes ─────────────────────────────────────────────
    w52_high      = float(close.tail(252).max())
    w52_low       = float(close.tail(252).min())
    pct_from_high = (price / w52_high - 1) * 100
    pct_from_low  = (price / w52_low  - 1) * 100
    at_52w_high   = pct_from_high > -5.0
    at_52w_low    = pct_from_low  < 10.0

    # Total technical score (cap at 40)
    raw = rsi_score + macd_score + ma_score + bb_score + adx_score + vol_score
    tech_score = min(40, raw)

    return {
        "tech_score":    tech_score,
        "rsi":           round(rsi, 1),
        "rsi_score":     rsi_score,
        "macd_bull":     macd_bull,
        "macd_cross":    macd_cross,
        "macd_score":    macd_score,
        "above_50ma":    above_50,
        "above_200ma":   above_200,
        "golden_cross":  golden_cross,
        "ma_score":      ma_score,
        "bb_pct":        round(bb_pct, 1),
        "bb_score":      bb_score,
        "adx":           round(adx, 1),
        "adx_score":     adx_score,
        "vol_ratio":     round(vol_ratio, 2),
        "vol_surge":     vol_surge,
        "vol_score":     vol_score,
        "price":         round(price, 4),
        "day_chg_pct":   round(day_chg_pct, 2),
        "sma50":         round(sma50, 4),
        "sma200":        round(sma200, 4),
        "pct_from_high": round(pct_from_high, 1),
        "pct_from_low":  round(pct_from_low, 1),
        "at_52w_high":   at_52w_high,
        "at_52w_low":    at_52w_low,
    }


# ─────────────────────────────────────────────
#  2. MOMENTUM SCORE  (0–25 pts)
# ─────────────────────────────────────────────

def score_momentum(df_crypto: pd.DataFrame, df_btc: pd.DataFrame) -> dict:
    """
    Score momentum (0–25) as returns relative to BTC benchmark.
    Positive relative return = outperforming Bitcoin.
    """
    close_c = df_crypto["Close"].squeeze()
    close_b = df_btc["Close"].squeeze() if df_btc is not None else None

    periods = CFG["momentum_periods"]

    results = {}
    total_mom = 0.0

    for label, days in periods.items():
        if len(close_c) >= days + 1:
            ret = (float(close_c.iloc[-1]) / float(close_c.iloc[-days]) - 1) * 100
        else:
            ret = 0.0

        if close_b is not None and len(close_b) >= days + 1:
            btc_ret = (float(close_b.iloc[-1]) / float(close_b.iloc[-days]) - 1) * 100
        else:
            btc_ret = 0.0

        rel = ret - btc_ret   # positive = beating BTC
        results[f"ret_{label}"]  = round(ret, 2)
        results[f"rel_{label}"]  = round(rel, 2)

    # Weight: 7d (15%), 30d (40%), 90d (45%)
    rel_7d  = results.get("rel_7d",  0)
    rel_30d = results.get("rel_30d", 0)
    rel_90d = results.get("rel_90d", 0)

    # Map relative performance to score
    def _rel_pts(rel, weight):
        """Convert relative return % to points."""
        if rel >= 20:   raw = 1.0
        elif rel >= 10: raw = 0.80
        elif rel >= 5:  raw = 0.65
        elif rel >= 2:  raw = 0.55
        elif rel >= 0:  raw = 0.45
        elif rel >= -5: raw = 0.35
        elif rel >= -10:raw = 0.20
        else:           raw = 0.0
        return raw * weight

    mom_score = (
        _rel_pts(rel_7d,  3.75)  +   # 15% of 25
        _rel_pts(rel_30d, 10.0)  +   # 40% of 25
        _rel_pts(rel_90d, 11.25)     # 45% of 25
    )

    return {
        "momentum_score": round(min(25.0, mom_score), 1),
        **results,
    }


# ─────────────────────────────────────────────
#  3. ON-CHAIN / QUALITY SCORE  (0–20 pts)
# ─────────────────────────────────────────────

# Network type bonuses — utility-bearing assets score higher
_NETWORK_BONUS = {
    "Layer 1":              4,   # Base chain infrastructure (BTC, ETH, SOL…)
    "AI / Infrastructure":  4,   # AI tokens + decentralised compute
    "DeFi":                 3,   # Decentralised finance protocols
    "Layer 2":              3,   # Scaling infrastructure
    "Payments / Interop":   2,   # Cross-chain / payment rails
    "Exchange / Infra":     2,   # Exchange utility tokens
}

def score_quality(ticker: str, category: str, market_cap_B: float,
                  vol_ratio: float) -> dict:
    """
    Score quality (0–20) using market cap tier + network type + activity.
    """
    # Market cap tier (0–12 pts)
    mc = market_cap_B
    if mc >= 200:   mc_score = 12   # Mega cap (BTC, ETH tier)
    elif mc >= 50:  mc_score = 10   # Large cap (SOL, BNB tier)
    elif mc >= 10:  mc_score = 8    # Mid-large (ADA, AVAX, XRP tier)
    elif mc >= 1:   mc_score = 5    # Mid cap
    elif mc >= 0.1: mc_score = 2    # Small cap
    else:           mc_score = 0    # Unknown / tiny

    # Network type bonus (0–4 pts)
    net_bonus = _NETWORK_BONUS.get(category, 1)

    # Activity score: healthy vol/mcap ratio (0–4 pts)
    # vol_ratio > 1 means above-average trading activity
    if vol_ratio >= 3.0:   act_score = 4
    elif vol_ratio >= 2.0: act_score = 3
    elif vol_ratio >= 1.0: act_score = 2
    elif vol_ratio >= 0.5: act_score = 1
    else:                  act_score = 0

    total = min(20, mc_score + net_bonus + act_score)

    return {
        "quality_score": total,
        "mc_score":      mc_score,
        "net_bonus":     net_bonus,
        "act_score":     act_score,
        "market_cap_B":  round(market_cap_B, 2),
    }


# ─────────────────────────────────────────────
#  4. SENTIMENT SCORE  (0–15 pts)
# ─────────────────────────────────────────────

def score_sentiment(ticker: str, signal: str, sentiment_data: dict) -> dict:
    """
    Score sentiment (0–15) from Phase 7 data or signal-derived proxy.
    """
    base_ticker = ticker.replace("-USD", "")
    sent_info   = sentiment_data.get(base_ticker, sentiment_data.get(ticker, {}))

    if sent_info:
        raw_score = float(sent_info.get("score", 0.0))
    else:
        # Derive from signal
        sig = signal.upper()
        if "STRONG BUY" in sig: raw_score =  0.55
        elif "BUY"      in sig: raw_score =  0.30
        elif "HOLD"     in sig: raw_score =  0.00
        elif "CAUTION"  in sig: raw_score = -0.25
        else:                   raw_score = -0.50

    sent_pts = ((raw_score + 1.0) / 2.0) * 15.0   # -1..+1 → 0..15

    return {
        "sentiment_score":     round(sent_pts, 1),
        "sentiment_raw":       round(raw_score, 3),
        "sentiment_label":     sent_info.get("label", "") if sent_info else "",
        "top_headline":        sent_info.get("headline", "") if sent_info else "",
        "theme_tags":          sent_info.get("themes", "") if sent_info else "",
    }


# ─────────────────────────────────────────────
#  5. COMPOSITE SIGNAL
# ─────────────────────────────────────────────

def compute_crypto_signal(total_score: float) -> str:
    thresholds = CFG["signal_thresholds"]
    if total_score >= thresholds["strong_buy"]: return "🟢 STRONG BUY"
    if total_score >= thresholds["buy"]:        return "🔵 BUY"
    if total_score >= thresholds["hold"]:       return "🟡 HOLD"
    if total_score >= thresholds["caution"]:    return "🟠 CAUTION"
    return "🔴 AVOID"


# ─────────────────────────────────────────────
#  6. FULL PIPELINE FOR ONE COIN
# ─────────────────────────────────────────────

def score_crypto(ticker: str, category: str, df_btc: pd.DataFrame,
                 sentiment_data: dict) -> dict | None:
    """
    Full scoring pipeline for a single crypto asset.
    Returns a flat dict of all metrics or None on failure.
    """
    if ticker in _MEME_EXCLUSIONS:
        print(f"  ⛔ {ticker} in meme exclusion list — skipped")
        return None

    df = fetch_crypto(ticker)
    if df is None or len(df) < 60:
        print(f"  ⚠️  {ticker}: insufficient data")
        return None

    # Fetch metadata
    info       = fetch_crypto_info(ticker)
    mc_B       = info.get("market_cap_B", 0.0)
    coin_name  = info.get("name", ticker.replace("-USD", ""))

    # Score each dimension
    tech    = score_technical(df)
    mom     = score_momentum(df, df_btc)
    qual    = score_quality(ticker, category, mc_B, tech.get("vol_ratio", 1.0))

    # Preliminary signal for sentiment fallback
    prelim_score = tech["tech_score"] + mom["momentum_score"] + qual["quality_score"]
    prelim_sig   = compute_crypto_signal(prelim_score / 0.85)  # rough scaling

    sent    = score_sentiment(ticker, prelim_sig, sentiment_data)

    total_score = (tech["tech_score"] + mom["momentum_score"]
                   + qual["quality_score"] + sent["sentiment_score"])
    signal      = compute_crypto_signal(total_score)

    return {
        "ticker":          ticker,
        "name":            coin_name,
        "category":        category,
        "price":           tech["price"],
        "day_chg_pct":     tech["day_chg_pct"],
        "total_score":     round(total_score, 1),
        "signal":          signal,
        # Components
        "tech_score":      tech["tech_score"],
        "momentum_score":  mom["momentum_score"],
        "quality_score":   qual["quality_score"],
        "sentiment_score_pts": sent["sentiment_score"],
        # Technical indicators
        "rsi":             tech["rsi"],
        "macd_bull":       tech["macd_bull"],
        "macd_cross":      tech["macd_cross"],
        "above_50ma":      tech["above_50ma"],
        "above_200ma":     tech["above_200ma"],
        "golden_cross":    tech["golden_cross"],
        "bb_pct":          tech["bb_pct"],
        "adx":             tech["adx"],
        "vol_ratio":       tech["vol_ratio"],
        "vol_surge":       tech["vol_surge"],
        "sma50":           tech["sma50"],
        "sma200":          tech["sma200"],
        "pct_from_high":   tech["pct_from_high"],
        "pct_from_low":    tech["pct_from_low"],
        "at_52w_high":     tech["at_52w_high"],
        "at_52w_low":      tech["at_52w_low"],
        # Momentum
        "ret_7d":          mom.get("ret_7d", 0),
        "ret_30d":         mom.get("ret_30d", 0),
        "ret_90d":         mom.get("ret_90d", 0),
        "rel_7d":          mom.get("rel_7d", 0),
        "rel_30d":         mom.get("rel_30d", 0),
        "rel_90d":         mom.get("rel_90d", 0),
        # Quality
        "market_cap_B":    qual["market_cap_B"],
        "mc_score":        qual["mc_score"],
        "net_bonus":       qual["net_bonus"],
        "act_score":       qual["act_score"],
        # Sentiment
        "sentiment_raw":       sent["sentiment_raw"],
        "sentiment_label":     sent["sentiment_label"],
        "top_headline":        sent["top_headline"],
        "theme_tags":          sent["theme_tags"],
        "run_date":        TODAY_STR,
    }


# ─────────────────────────────────────────────
#  MARKET CYCLE CONTEXT
# ─────────────────────────────────────────────

def get_market_cycle(df_btc: pd.DataFrame) -> dict:
    """
    Determine BTC market cycle context: bull / caution / bear.
    Based on BTC's position vs its 200-day moving average.
    """
    if df_btc is None or len(df_btc) < 201:
        return {"cycle": "Unknown", "btc_vs_200dma_pct": 0.0, "emoji": "❓"}

    close  = df_btc["Close"].squeeze()
    price  = float(close.iloc[-1])
    sma200 = float(ta.trend.SMAIndicator(close=close, window=200).sma_indicator().iloc[-1])
    pct    = (price / sma200 - 1) * 100

    if pct > 5:
        cycle, emoji = "Bull Market",   "🟢"
    elif pct > 0:
        cycle, emoji = "Late Bull",     "🟡"
    elif pct > -10:
        cycle, emoji = "Caution Zone",  "🟠"
    else:
        cycle, emoji = "Bear Market",   "🔴"

    # BTC dominance proxy (based on proximity to 52w high)
    w52_high      = float(close.tail(252).max())
    pct_from_high = (price / w52_high - 1) * 100

    return {
        "cycle":              cycle,
        "emoji":              emoji,
        "btc_vs_200dma_pct":  round(pct, 1),
        "btc_price":          round(price, 0),
        "btc_pct_from_high":  round(pct_from_high, 1),
    }


# ─────────────────────────────────────────────
#  MAIN RUNNER
# ─────────────────────────────────────────────

def run_crypto_screener():
    """
    Score all quality cryptocurrencies in CRYPTO_WATCHLIST.
    Saves data/crypto_screener_results.csv and prints a summary.
    """
    print(f"\n{'='*60}")
    print(f"  INVESTMENT INTELLIGENCE — Phase 4c: Crypto Screener")
    print(f"  {datetime.now().strftime('%A, %B %d, %Y  %H:%M')}")
    print(f"{'='*60}\n")

    # ── 0. Load pre-computed sentiment (optional) ────────────
    sentiment_data: dict = {}
    sent_csv = os.path.join(DATA_DIR, "news_sentiment.csv")
    if os.path.exists(sent_csv):
        try:
            df_s = pd.read_csv(sent_csv)
            for _, row in df_s.iterrows():
                sentiment_data[str(row.get("ticker", ""))] = {
                    "score":    float(row.get("sentiment_score", 0)),
                    "label":    str(row.get("sentiment_label", "")),
                    "headline": str(row.get("top_headline", "")),
                    "themes":   str(row.get("theme_tags", "")),
                }
            print(f"  📰 Loaded sentiment data for {len(sentiment_data)} tickers")
        except Exception:
            pass

    # ── 1. Download BTC benchmark ─────────────────────────────
    print(f"  📡 Downloading BTC benchmark...")
    df_btc = fetch_crypto(BENCHMARK)
    if df_btc is not None:
        cycle = get_market_cycle(df_btc)
        print(f"  {cycle['emoji']} Market cycle: {cycle['cycle']}  "
              f"(BTC ${cycle['btc_price']:,.0f}, "
              f"{cycle['btc_vs_200dma_pct']:+.1f}% vs 200 DMA, "
              f"{cycle['btc_pct_from_high']:+.1f}% from ATH)\n")
    else:
        print("  ⚠️  Could not download BTC data\n")
        cycle = {"cycle": "Unknown", "emoji": "❓"}

    # ── 2. Score each coin ────────────────────────────────────
    results = []
    total_coins = sum(len(v) for v in CRYPTO_WATCHLIST.items() if isinstance(v, list))
    done = 0

    for category, tickers in CRYPTO_WATCHLIST.items():
        print(f"  📂 {category} ({len(tickers)} coins)")
        for ticker in tickers:
            if ticker in _MEME_EXCLUSIONS:
                continue
            done += 1
            print(f"     [{done:>2}/{sum(len(v) for v in CRYPTO_WATCHLIST.values())}]"
                  f" {ticker:<12}", end="  ")
            row = score_crypto(ticker, category, df_btc, sentiment_data)
            if row:
                results.append(row)
                sig_emoji = row["signal"].split(" ")[0]
                print(f"score={row['total_score']:>5.1f}  {sig_emoji}  "
                      f"${row['price']:>12,.4f}  "
                      f"{row['day_chg_pct']:+.1f}%")
            else:
                print("— failed")

        print()

    if not results:
        print("  ❌ No results — check network / yfinance installation")
        return

    # ── 3. Save results ───────────────────────────────────────
    df = pd.DataFrame(results)
    df = df.sort_values("total_score", ascending=False).reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"  ✅ Saved: {OUTPUT_FILE}")

    # ── 4. Save cycle context ─────────────────────────────────
    cycle_file = os.path.join(DATA_DIR, "crypto_cycle.json")
    import json
    with open(cycle_file, "w") as f:
        json.dump(cycle, f, indent=2)

    # ── 5. Summary table ──────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  CRYPTO SCREENER RESULTS — {TODAY_STR}")
    print(f"{'═'*60}")
    print(f"  {'Ticker':<12} {'Name':<22} {'Score':>5}  {'Signal':<20} {'7d':>7} {'30d':>7} {'90d':>7}")
    print(f"  {'─'*12} {'─'*22} {'─'*5}  {'─'*20} {'─'*7} {'─'*7} {'─'*7}")
    for _, row in df.head(15).iterrows():
        print(f"  {row['ticker']:<12} {str(row['name'])[:22]:<22} {row['total_score']:>5.1f}  "
              f"{row['signal']:<20} {row['ret_7d']:>+6.1f}% {row['ret_30d']:>+6.1f}% "
              f"{row['ret_90d']:>+6.1f}%")

    n_sb = sum(1 for r in results if "STRONG BUY" in r["signal"])
    n_b  = sum(1 for r in results if r["signal"] == "🔵 BUY")
    n_h  = sum(1 for r in results if "HOLD"     in r["signal"])
    n_c  = sum(1 for r in results if "CAUTION"  in r["signal"])
    n_a  = sum(1 for r in results if "AVOID"    in r["signal"])

    print(f"\n  Signal distribution: 🟢 {n_sb} · 🔵 {n_b} · 🟡 {n_h} · 🟠 {n_c} · 🔴 {n_a}")
    print(f"  {cycle['emoji']} Market cycle: {cycle.get('cycle','—')}")
    print(f"\n{'═'*60}\n")

    return df


if __name__ == "__main__":
    run_crypto_screener()
