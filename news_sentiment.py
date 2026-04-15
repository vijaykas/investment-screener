"""
============================================================
  INVESTMENT INTELLIGENCE — PHASE 7
  News & Sentiment Engine
  Author: Claude (Vijay's Investment Project)
  Updated: 2026-04-13
============================================================

WHAT THIS DOES:
  1. Fetches recent news headlines for every stock + ETF in the watchlist
  2. Computes per-ticker sentiment scores (-1.0 → +1.0) via:
       a) Alpha Vantage NEWS_SENTIMENT (primary, 500 req/day free)
       b) Finnhub /company-news (secondary, 60 req/min free)
       c) FMP /stock_news (tertiary, 250 req/day free)
       d) Signal-derived synthetic sentiment (fallback — always available)
  3. Identifies macro market themes from the news cycle
  4. Saves data/news_sentiment.csv + data/macro_themes.json

OUTPUT COLUMNS (news_sentiment.csv):
  ticker, asset_type, sentiment_score, sentiment_label, sentiment_pct,
  news_count, top_headline, headline_source, headline_date,
  theme_tags, bullish_count, bearish_count, data_source

HOW TO RUN:
  python3 news_sentiment.py          # standalone
  python3 invest.py --news           # via unified runner
  python3 invest.py --quick          # included in quick run
============================================================
"""

import os
import sys
import json
import time
import random
import math
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    API_KEYS, DATA_DIR, WATCHLIST, ETF_WATCHLIST, CRYPTO_WATCHLIST,
    NEWS_SENTIMENT_CONFIG, get_watchlist, get_etf_watchlist,
)

OUTPUT_CSV   = os.path.join(DATA_DIR, NEWS_SENTIMENT_CONFIG["output_file"])
THEMES_JSON  = os.path.join(DATA_DIR, NEWS_SENTIMENT_CONFIG["themes_file"])
TODAY_STR    = date.today().strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────
#  SENTIMENT LABEL HELPERS
# ─────────────────────────────────────────────────────────────

SENTIMENT_THRESHOLDS = NEWS_SENTIMENT_CONFIG["sentiment_thresholds"]

def score_to_label(score: float) -> str:
    """Convert a -1..+1 sentiment score to a human label."""
    if score >= SENTIMENT_THRESHOLDS["bullish"]:
        return "🟢 Bullish"
    elif score >= SENTIMENT_THRESHOLDS["somewhat_bullish"]:
        return "🔵 Somewhat Bullish"
    elif score >= SENTIMENT_THRESHOLDS["neutral"]:
        return "⚪ Neutral"
    elif score >= SENTIMENT_THRESHOLDS["somewhat_bearish"]:
        return "🟠 Somewhat Bearish"
    else:
        return "🔴 Bearish"

def score_to_pct(score: float) -> int:
    """Convert -1..+1 score to 0..100 percentage."""
    return int((score + 1) / 2 * 100)


# ─────────────────────────────────────────────────────────────
#  1. ALPHA VANTAGE NEWS_SENTIMENT
# ─────────────────────────────────────────────────────────────

def fetch_av_sentiment(tickers: list, api_key: str) -> dict:
    """
    Fetch Alpha Vantage NEWS_SENTIMENT for batches of tickers.
    Returns {ticker: {"score": float, "count": int, "headline": str, "source": str, "date": str}}
    """
    import urllib.request, urllib.error
    results = {}
    batch_size = NEWS_SENTIMENT_CONFIG.get("av_batch_size", 5)
    sleep_s = NEWS_SENTIMENT_CONFIG.get("sleep_between_batches", 1.2)

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        ticker_str = ",".join(batch)
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=NEWS_SENTIMENT"
            f"&tickers={ticker_str}"
            f"&limit=50"
            f"&apikey={api_key}"
        )
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  ⚠️  AV batch {i//batch_size + 1} failed: {e}")
            continue

        feed = data.get("feed", [])
        if not feed:
            # Rate limit or error
            err = data.get("Information", data.get("Note", ""))
            if err:
                print(f"  ⚠️  Alpha Vantage: {err[:80]}")
                break
            continue

        # Aggregate per-ticker sentiment across articles
        ticker_data: dict[str, dict] = {t: {"scores": [], "headline": "", "source": "", "date": ""} for t in batch}

        for article in feed:
            art_ts   = article.get("time_published", "")[:8]   # YYYYMMDD
            art_title = article.get("title", "")
            art_src   = article.get("source", "")
            ticker_sentiments = article.get("ticker_sentiment", [])

            for ts in ticker_sentiments:
                sym = ts.get("ticker", "")
                if sym not in ticker_data:
                    continue
                try:
                    score = float(ts.get("ticker_sentiment_score", 0))
                except (TypeError, ValueError):
                    continue
                ticker_data[sym]["scores"].append(score)
                if not ticker_data[sym]["headline"]:
                    ticker_data[sym]["headline"] = art_title
                    ticker_data[sym]["source"]   = art_src
                    ticker_data[sym]["date"]      = art_ts

        for t, d in ticker_data.items():
            if d["scores"]:
                avg_score = sum(d["scores"]) / len(d["scores"])
                results[t] = {
                    "score":    avg_score,
                    "count":    len(d["scores"]),
                    "headline": d["headline"][:120],
                    "source":   d["source"],
                    "date":     d["date"],
                }

        time.sleep(sleep_s)
        print(f"  📰 AV: batch {i//batch_size + 1}/{math.ceil(len(tickers)/batch_size)} — "
              f"{len([t for t in batch if t in results])}/{len(batch)} tickers with data")

    return results


# ─────────────────────────────────────────────────────────────
#  2. FINNHUB NEWS
# ─────────────────────────────────────────────────────────────

_VADER_AVAILABLE = None

def _get_vader():
    global _VADER_AVAILABLE
    if _VADER_AVAILABLE is not None:
        return _VADER_AVAILABLE
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _VADER_AVAILABLE = SentimentIntensityAnalyzer()
    except ImportError:
        try:
            import nltk
            nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            _VADER_AVAILABLE = SentimentIntensityAnalyzer()
        except Exception:
            _VADER_AVAILABLE = False
    return _VADER_AVAILABLE


def _headline_sentiment(text: str) -> float:
    """Quick VADER sentiment on a headline. Returns -1..+1."""
    vader = _get_vader()
    if not vader:
        return 0.0
    scores = vader.polarity_scores(text)
    return float(scores["compound"])   # Already -1..+1


def fetch_finnhub_sentiment(tickers: list, api_key: str) -> dict:
    """
    Fetch Finnhub company news (7-day window) and score with VADER.
    Returns {ticker: {"score": float, "count": int, "headline": str, ...}}
    """
    import urllib.request, urllib.error
    results = {}
    from_date = (date.today() - timedelta(days=NEWS_SENTIMENT_CONFIG["lookback_days"])).strftime("%Y-%m-%d")
    to_date   = TODAY_STR

    for ticker in tickers:
        url = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}&from={from_date}&to={to_date}"
            f"&token={api_key}"
        )
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                articles = json.loads(resp.read().decode())
        except Exception:
            continue

        if not isinstance(articles, list) or not articles:
            continue

        # Limit to most recent N articles
        articles = articles[:NEWS_SENTIMENT_CONFIG["max_articles_per_ticker"]]
        scores   = []
        headline = ""

        for art in articles:
            title = art.get("headline", "") or art.get("summary", "")
            if not title:
                continue
            s = _headline_sentiment(title)
            scores.append(s)
            if not headline:
                headline = title
                headline_src  = art.get("source", "Finnhub")
                headline_date = str(art.get("datetime", ""))[:8]

        if scores:
            results[ticker] = {
                "score":    sum(scores) / len(scores),
                "count":    len(scores),
                "headline": headline[:120],
                "source":   headline_src if scores else "Finnhub",
                "date":     headline_date if scores else "",
            }
        time.sleep(0.12)  # 60 req/min → safe at 120ms

    return results


# ─────────────────────────────────────────────────────────────
#  3. FMP NEWS (fallback)
# ─────────────────────────────────────────────────────────────

def fetch_fmp_sentiment(tickers: list, api_key: str) -> dict:
    """
    Fetch FMP stock news for tickers (batch up to 10 at a time).
    Returns {ticker: {"score": float, "count": int, "headline": str, ...}}
    """
    import urllib.request
    results = {}
    batch_size = 10

    for i in range(0, len(tickers), batch_size):
        batch   = tickers[i:i + batch_size]
        symbols = ",".join(batch)
        url = (
            f"https://financialmodelingprep.com/api/v3/stock_news"
            f"?tickers={symbols}&limit=50&apikey={api_key}"
        )
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                articles = json.loads(resp.read().decode())
        except Exception:
            continue

        if not isinstance(articles, list):
            continue

        # Group by ticker
        ticker_articles: dict[str, list] = {t: [] for t in batch}
        for art in articles:
            sym = art.get("symbol", "")
            if sym in ticker_articles:
                ticker_articles[sym].append(art)

        for t, arts in ticker_articles.items():
            if not arts:
                continue
            scores   = []
            headline = ""
            for art in arts[:NEWS_SENTIMENT_CONFIG["max_articles_per_ticker"]]:
                title = art.get("title", "")
                if not title:
                    continue
                s = _headline_sentiment(title)
                scores.append(s)
                if not headline:
                    headline = title
                    headline_src  = art.get("site", "FMP")
                    headline_date = (art.get("publishedDate", "")[:10] or "").replace("-", "")
            if scores:
                results[t] = {
                    "score":    sum(scores) / len(scores),
                    "count":    len(scores),
                    "headline": headline[:120],
                    "source":   headline_src if scores else "FMP",
                    "date":     headline_date if scores else "",
                }
        time.sleep(0.3)

    return results


# ─────────────────────────────────────────────────────────────
#  4. SIGNAL-DERIVED SYNTHETIC SENTIMENT (fallback — no API needed)
# ─────────────────────────────────────────────────────────────

# Current macro themes as of April 2026 (used when no live news)
_CURRENT_MACRO_THEMES = [
    {
        "theme": "AI Infrastructure Buildout",
        "emoji": "🤖",
        "detail": "Hyperscalers continue massive capex; NVDA, MSFT, META, GOOGL leading",
        "sentiment": "bullish",
        "tickers": ["NVDA", "MSFT", "META", "GOOGL", "AVGO", "AMD", "QQQ", "XLK"],
    },
    {
        "theme": "Fed Rate Cut Watch",
        "emoji": "💰",
        "detail": "Markets pricing 2-3 cuts in 2026; bond yields falling, growth stocks benefiting",
        "sentiment": "bullish",
        "tickers": ["TLT", "IEF", "BND", "AGG", "PLTR", "CRM", "NOW", "NET"],
    },
    {
        "theme": "China / Trade Tariff Risk",
        "emoji": "🌏",
        "detail": "Tariff uncertainty weighing on supply-chain-exposed names; dollar strength",
        "sentiment": "bearish",
        "tickers": ["AAPL", "NKE", "AMZN", "XLE", "EEM", "VWO", "EFA", "VEA"],
    },
    {
        "theme": "GLP-1 Drug Revolution",
        "emoji": "💊",
        "detail": "Obesity/diabetes drugs reshaping healthcare; LLY and NVO outperforming",
        "sentiment": "bullish",
        "tickers": ["LLY", "UNH", "ABBV", "MRK", "JNJ", "XLV"],
    },
    {
        "theme": "Energy Transition",
        "emoji": "⚡",
        "detail": "IRA incentives + AI power demand driving clean energy and utilities",
        "sentiment": "bullish",
        "tickers": ["XLU", "NEE", "XLE", "CVX", "PDBC", "DJP"],
    },
    {
        "theme": "Gold / Safe-Haven Demand",
        "emoji": "🥇",
        "detail": "Central bank buying + geopolitical uncertainty driving gold to record highs",
        "sentiment": "bullish",
        "tickers": ["GLD", "IAU", "SLV"],
    },
    {
        "theme": "Q1 2026 Earnings Season",
        "emoji": "📊",
        "detail": "Broad earnings beats expected; tech + healthcare leading; energy mixed",
        "sentiment": "bullish",
        "tickers": ["SPY", "QQQ", "VTI", "XLF", "JPM", "V", "MA", "GS"],
    },
    {
        "theme": "Small-Cap Recovery",
        "emoji": "📈",
        "detail": "Rate-cut expectations benefiting domestically focused small caps",
        "sentiment": "somewhat_bullish",
        "tickers": ["IWM", "VTI"],
    },
]

# Headline templates per signal for synthetic mode
_SYNTHETIC_HEADLINES = {
    "STRONG BUY": [
        "{t}: Analysts raise price targets ahead of strong earnings",
        "{t}: Institutional buyers accumulate; momentum building",
        "{t}: Technical breakout confirmed — bullish setup intact",
        "{t}: Sector tailwinds accelerate; raised to Outperform",
    ],
    "BUY": [
        "{t}: Positive analyst coverage; above consensus estimates",
        "{t}: Recovery momentum continues; sentiment improving",
        "{t}: Upgraded by major bank; sees 15-20% upside",
        "{t}: Strong fundamentals underpin near-term rally",
    ],
    "HOLD": [
        "{t}: Stable outlook; awaiting next catalyst",
        "{t}: Mixed signals; investors on the sidelines",
        "{t}: Fairly valued; limited near-term upside expected",
    ],
    "CAUTION": [
        "{t}: Headwinds emerging; analyst cautious near-term",
        "{t}: Slowing growth raises valuation concerns",
        "{t}: Risk-off sentiment weighing on the sector",
    ],
    "SELL": [
        "{t}: Disappointing guidance; selloff accelerates",
        "{t}: Technical breakdown confirmed; avoid for now",
    ],
}

def _derive_signal_key(signal: str) -> str:
    for k in ("STRONG BUY", "BUY", "HOLD", "CAUTION", "SELL"):
        if k in signal.upper():
            return k
    return "HOLD"


def build_synthetic_sentiment(ticker: str, signal: str, score: float,
                               rsi: float = 50, day_chg: float = 0.0,
                               asset_type: str = "STOCK") -> dict:
    """
    Derive a plausible sentiment score from technical signals.
    Used as fallback when no news API is available.
    """
    sig_key = _derive_signal_key(signal)

    # Base score from signal
    base_map = {
        "STRONG BUY": 0.55,
        "BUY":        0.30,
        "HOLD":       0.00,
        "CAUTION":   -0.25,
        "SELL":      -0.50,
    }
    base = base_map.get(sig_key, 0.0)

    # Adjust for score (normalized to ±0.15)
    score_factor = 0 if asset_type == "ETF" else (score / 75 - 0.5) * 0.15
    if asset_type == "ETF":
        score_factor = (score / 100 - 0.5) * 0.15

    # RSI factor (±0.10)
    rsi_factor = ((rsi - 50) / 50) * 0.10

    # Recent momentum factor (±0.10)
    mom_factor = max(-0.10, min(0.10, day_chg / 10))

    # Small random noise ±0.05 (deterministic from ticker hash)
    noise = ((hash(ticker) % 100) / 100 - 0.5) * 0.05

    raw = base + score_factor + rsi_factor + mom_factor + noise
    final = max(-0.90, min(0.90, raw))

    # Headline
    headlines = _SYNTHETIC_HEADLINES.get(sig_key, _SYNTHETIC_HEADLINES["HOLD"])
    headline  = random.Random(ticker).choice(headlines).format(t=ticker)

    # Macro theme match
    theme_tags = []
    for theme in _CURRENT_MACRO_THEMES:
        if ticker in theme["tickers"]:
            theme_tags.append(theme["theme"])

    return {
        "score":    round(final, 3),
        "count":    0,
        "headline": headline,
        "source":   "Synthetic (signal-derived)",
        "date":     TODAY_STR.replace("-", ""),
        "themes":   theme_tags[:2],
    }


# ─────────────────────────────────────────────────────────────
#  5. THEME EXTRACTION FROM LIVE NEWS
# ─────────────────────────────────────────────────────────────

_THEME_KEYWORDS = {
    "AI Infrastructure Buildout": ["artificial intelligence", "ai capex", "data center", "gpu", "nvidia", "llm", "generative ai", "ai spending"],
    "Fed Rate Cut Watch":         ["federal reserve", "rate cut", "fomc", "interest rate", "fed pivot", "jerome powell", "basis points"],
    "China / Trade Tariff Risk":  ["tariff", "china", "trade war", "import duty", "supply chain", "beijing", "decoupling"],
    "GLP-1 Drug Revolution":      ["glp-1", "ozempic", "wegovy", "obesity drug", "eli lilly", "novo nordisk", "semaglutide"],
    "Energy Transition":          ["clean energy", "ira", "inflation reduction act", "solar", "wind energy", "ev charging", "grid"],
    "Gold / Safe-Haven Demand":   ["gold price", "safe haven", "central bank gold", "gold record", "bullion"],
    "Q1 2026 Earnings Season":    ["q1 earnings", "first quarter results", "earnings beat", "eps", "guidance raised"],
    "Small-Cap Recovery":         ["small cap", "russell 2000", "domestic growth", "rate sensitive"],
}

def extract_themes_from_headlines(news_results: dict) -> list:
    """
    Scan all collected headlines for macro theme keywords.
    Returns list of active theme dicts sorted by headline count.
    """
    theme_hits: dict[str, int] = {t: 0 for t in _THEME_KEYWORDS}
    all_headlines = " | ".join(
        v.get("headline", "") for v in news_results.values()
        if isinstance(v, dict)
    ).lower()

    for theme, keywords in _THEME_KEYWORDS.items():
        for kw in keywords:
            if kw in all_headlines:
                theme_hits[theme] += 1

    active = [t for t, c in theme_hits.items() if c > 0]
    active.sort(key=lambda t: -theme_hits[t])

    # Fill in from current macro themes lookup
    result = []
    seen = set()
    for theme_name in active:
        for t in _CURRENT_MACRO_THEMES:
            if t["theme"] == theme_name and theme_name not in seen:
                result.append({**t, "hit_count": theme_hits[theme_name]})
                seen.add(theme_name)

    # Always include top current themes even if not in headlines
    for t in _CURRENT_MACRO_THEMES:
        if t["theme"] not in seen and len(result) < 8:
            result.append({**t, "hit_count": 0})
            seen.add(t["theme"])

    return result


# ─────────────────────────────────────────────────────────────
#  6. MAIN RUNNER
# ─────────────────────────────────────────────────────────────

def _crypto_ticker_to_symbol(ticker: str) -> str:
    """Convert yfinance crypto ticker (BTC-USD) to news API symbol (BTC)."""
    return ticker.replace("-USD", "").replace("-USDT", "")


def run_news_sentiment():
    """
    Fetch news and sentiment for all stocks + ETFs + crypto.
    Saves data/news_sentiment.csv and data/macro_themes.json.
    """
    print("\n📰  Phase 7 — News & Sentiment Engine")
    print("─" * 50)

    # ── Build full ticker list ────────────────────────────────
    stock_watchlist  = get_watchlist()
    etf_watchlist    = get_etf_watchlist()
    crypto_watchlist = CRYPTO_WATCHLIST

    stock_tickers = list(dict.fromkeys(
        t for sect in stock_watchlist.values() for t in sect
        if t not in ("SPY", "QQQ", "VTI", "SCHD", "IWM", "DIA")
    ))
    etf_tickers = list(dict.fromkeys(
        t for cat in etf_watchlist.values() for t in cat
    ))
    # Store both the yfinance format and the news API format
    crypto_raw     = list(dict.fromkeys(
        t for cat in crypto_watchlist.values() for t in cat
    ))
    crypto_symbols = [_crypto_ticker_to_symbol(t) for t in crypto_raw]

    print(f"  📊 Stocks:  {len(stock_tickers)} tickers")
    print(f"  📈 ETFs:    {len(etf_tickers)} tickers")
    print(f"  ₿  Crypto:  {len(crypto_raw)} coins")

    # ── Load existing screener data for fallback ──────────────
    stock_scores:  dict[str, dict] = {}
    etf_scores:    dict[str, dict] = {}
    crypto_scores: dict[str, dict] = {}

    stock_csv = os.path.join(DATA_DIR, "stock_screener_results.csv")
    if os.path.exists(stock_csv):
        df_s = pd.read_csv(stock_csv)
        for _, row in df_s.iterrows():
            stock_scores[str(row.get("ticker", ""))] = {
                "signal": str(row.get("combined_signal_label", row.get("signal", "HOLD"))),
                "score":  float(row.get("combined_score", row.get("total_score", 50))),
                "rsi":    float(row.get("rsi", 50)) if pd.notna(row.get("rsi")) else 50.0,
                "day_chg": float(row.get("day_chg_pct", 0)) if pd.notna(row.get("day_chg_pct")) else 0.0,
            }

    etf_csv = os.path.join(DATA_DIR, "etf_screener_results.csv")
    if os.path.exists(etf_csv):
        df_e = pd.read_csv(etf_csv)
        for _, row in df_e.iterrows():
            etf_scores[str(row.get("ticker", ""))] = {
                "signal":  str(row.get("signal", "HOLD")),
                "score":   float(row.get("total_score", 50)),
                "rsi":     float(row.get("rsi", 50)) if pd.notna(row.get("rsi")) else 50.0,
                "day_chg": float(row.get("day_chg_pct", 0)) if pd.notna(row.get("day_chg_pct")) else 0.0,
            }

    crypto_csv = os.path.join(DATA_DIR, "crypto_screener_results.csv")
    if os.path.exists(crypto_csv):
        df_c = pd.read_csv(crypto_csv)
        for _, row in df_c.iterrows():
            t_raw = str(row.get("ticker", ""))
            sym   = _crypto_ticker_to_symbol(t_raw)   # BTC-USD → BTC
            crypto_scores[sym] = {
                "signal":  str(row.get("signal", "HOLD")),
                "score":   float(row.get("total_score", 50)),
                "rsi":     float(row.get("rsi", 50)) if pd.notna(row.get("rsi")) else 50.0,
                "day_chg": float(row.get("day_chg_pct", 0)) if pd.notna(row.get("day_chg_pct")) else 0.0,
            }

    # ── Try live API sources in preference order ──────────────
    all_tickers = stock_tickers + etf_tickers
    news_results: dict[str, dict] = {}
    data_source = "synthetic"

    av_key  = API_KEYS.get("alpha_vantage", "")
    fh_key  = API_KEYS.get("finnhub", "")
    fmp_key = API_KEYS.get("fmp", "")

    pref = NEWS_SENTIMENT_CONFIG.get("api_preference", ["alpha_vantage", "finnhub"])

    for api_name in pref:
        if api_name == "alpha_vantage" and av_key:
            print(f"\n  🌐 Trying Alpha Vantage NEWS_SENTIMENT...")
            try:
                results = fetch_av_sentiment(all_tickers[:50], av_key)   # free tier limit
                if results:
                    news_results.update(results)
                    data_source = "alpha_vantage"
                    print(f"  ✅ Alpha Vantage: {len(results)} tickers with sentiment data")
                    break
            except Exception as e:
                print(f"  ⚠️  Alpha Vantage failed: {e}")

        elif api_name == "finnhub" and fh_key:
            print(f"\n  🌐 Trying Finnhub news...")
            try:
                # Limit to top 60 stocks (rate limit: 60 req/min)
                results = fetch_finnhub_sentiment(stock_tickers[:60], fh_key)
                if results:
                    news_results.update(results)
                    # Also try FMP for ETFs if available
                    if fmp_key:
                        etf_results = fetch_fmp_sentiment(etf_tickers[:30], fmp_key)
                        news_results.update(etf_results)
                    data_source = "finnhub"
                    print(f"  ✅ Finnhub: {len(results)} tickers with sentiment data")
                    break
            except Exception as e:
                print(f"  ⚠️  Finnhub failed: {e}")

    if not news_results and fmp_key:
        print(f"\n  🌐 Trying FMP news (tertiary)...")
        try:
            results = fetch_fmp_sentiment(all_tickers[:40], fmp_key)
            if results:
                news_results.update(results)
                data_source = "fmp"
                print(f"  ✅ FMP: {len(results)} tickers with sentiment data")
        except Exception as e:
            print(f"  ⚠️  FMP news failed: {e}")

    if not news_results:
        print(f"\n  ℹ️  No live news API available — using signal-derived synthetic sentiment")
        print(f"       (Add API keys in config.py Section 1 to enable live news)")
        data_source = "synthetic"

    # ── Build final sentiment rows ────────────────────────────
    rows = []

    def _make_row(ticker, asset_type, score_data, live_data):
        if live_data:
            score     = float(live_data.get("score", 0))
            count     = int(live_data.get("count", 0))
            headline  = str(live_data.get("headline", ""))
            src       = str(live_data.get("source", data_source))
            hl_date   = str(live_data.get("date", ""))
            # Theme tags from live headlines
            theme_tags_live = []
            hl_lower = headline.lower()
            for theme, kws in _THEME_KEYWORDS.items():
                if any(kw in hl_lower for kw in kws):
                    theme_tags_live.append(theme)
        else:
            # Synthetic fallback
            sig   = score_data.get("signal", "HOLD")
            sc    = score_data.get("score", 50)
            rsi   = score_data.get("rsi", 50)
            day_c = score_data.get("day_chg", 0)
            synth = build_synthetic_sentiment(ticker, sig, sc, rsi, day_c, asset_type)
            score      = synth["score"]
            count      = synth["count"]
            headline   = synth["headline"]
            src        = synth["source"]
            hl_date    = synth["date"]
            theme_tags_live = synth["themes"]

        label = score_to_label(score)
        pct   = score_to_pct(score)

        # Directional counts (for display only)
        bullish_count  = max(0, round((score + 1) / 2 * count)) if count > 0 else (1 if score > 0 else 0)
        bearish_count  = max(0, count - bullish_count) if count > 0 else (1 if score < 0 else 0)

        return {
            "ticker":         ticker,
            "asset_type":     asset_type,
            "sentiment_score": round(score, 4),
            "sentiment_label": label,
            "sentiment_pct":  pct,
            "news_count":     count,
            "top_headline":   headline,
            "headline_source": src,
            "headline_date":  hl_date,
            "theme_tags":     "|".join(theme_tags_live),
            "bullish_count":  bullish_count,
            "bearish_count":  bearish_count,
            "data_source":    src,
            "run_date":       TODAY_STR,
        }

    # Process stocks
    print(f"\n  ⚙️  Processing sentiment for {len(stock_tickers)} stocks...")
    for ticker in stock_tickers:
        sd   = stock_scores.get(ticker, {"signal": "HOLD", "score": 50, "rsi": 50, "day_chg": 0})
        live = news_results.get(ticker)
        rows.append(_make_row(ticker, "STOCK", sd, live))

    # Process ETFs
    print(f"  ⚙️  Processing sentiment for {len(etf_tickers)} ETFs...")
    for ticker in etf_tickers:
        sd   = etf_scores.get(ticker, {"signal": "HOLD", "score": 50, "rsi": 50, "day_chg": 0})
        live = news_results.get(ticker)
        rows.append(_make_row(ticker, "ETF", sd, live))

    # Process Crypto — keyed by symbol (BTC) but stored as ticker (BTC-USD)
    print(f"  ⚙️  Processing sentiment for {len(crypto_raw)} crypto coins...")
    for raw_ticker, symbol in zip(crypto_raw, crypto_symbols):
        sd   = crypto_scores.get(symbol, {"signal": "HOLD", "score": 50, "rsi": 50, "day_chg": 0})
        live = news_results.get(symbol)
        rows.append(_make_row(symbol, "CRYPTO", sd, live))

    # ── Save sentiment CSV ────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  ✅ Saved: {OUTPUT_CSV}")
    print(f"     {len(df)} rows  |  data_source: {data_source}")

    # Print top sentiment
    top_bull = df.nlargest(5, "sentiment_score")[["ticker", "asset_type", "sentiment_score", "sentiment_label"]]
    top_bear = df.nsmallest(5, "sentiment_score")[["ticker", "asset_type", "sentiment_score", "sentiment_label"]]
    print(f"\n  🟢 Most Bullish:")
    for _, r in top_bull.iterrows():
        print(f"     {r['ticker']:<6} {r['asset_type']:<5} {r['sentiment_score']:+.3f}  {r['sentiment_label']}")
    print(f"\n  🔴 Most Bearish:")
    for _, r in top_bear.iterrows():
        print(f"     {r['ticker']:<6} {r['asset_type']:<5} {r['sentiment_score']:+.3f}  {r['sentiment_label']}")

    # ── Extract and save macro themes ─────────────────────────
    themes = extract_themes_from_headlines(news_results)
    with open(THEMES_JSON, "w") as f:
        json.dump({
            "run_date":   TODAY_STR,
            "data_source": data_source,
            "themes":     themes,
        }, f, indent=2)
    print(f"\n  ✅ Saved: {THEMES_JSON}")
    print(f"     {len(themes)} macro themes identified")
    for t in themes[:5]:
        emoji = t.get("emoji", "")
        print(f"     {emoji}  {t['theme']}  ({t['sentiment']})")

    print(f"\n  📰 News & Sentiment complete — {len(rows)} tickers processed")
    return df


if __name__ == "__main__":
    run_news_sentiment()
