"""
============================================================
  INVESTMENT INTELLIGENCE — Auto News Sentiment
  Phase 6b: Lightweight Daily Sentiment (no NLP required)
  Author: Claude (Vijay's Investment Project)
  Updated: 2026-04-15
============================================================

WHAT THIS DOES:
  Fetches pre-scored news sentiment for each stock automatically
  on every daily monitor run — no manual trigger needed.

  Unlike news_sentiment.py (which runs heavy VADER NLP), this module
  uses APIs that return sentiment scores directly, making it fast
  enough to run as part of the daily monitor loop.

DATA SOURCES (tried in priority order):
  1. Finnhub /news-sentiment — best quality, instant score (-1 to +1)
     Requires FINNHUB_KEY (free, 60 req/min)
  2. Alpha Vantage NEWS_SENTIMENT — good quality, batch-friendly
     Requires ALPHA_VANTAGE_KEY (free, 25 req/day)
  3. Existing news_sentiment.csv — use yesterday's file if available
     No API required — always works as fallback

OUTPUT:
  {ticker: {score, label, headline, source, age_days}}
  score: -1.0 (very bearish) → +1.0 (very bullish)

CACHING:
  Results cached for 6 hours. Morning run fetches fresh data;
  evening run reuses morning cache unless > 6 hours old.
============================================================
"""

import os
import json
import time
import requests
from datetime import datetime, date, timedelta

_CACHE_HOURS   = 6
_LOOKBACK_DAYS = 3     # headlines from last 3 days


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


def _score_to_label(score: float) -> str:
    if   score >=  0.20: return "🟢 Bullish"
    elif score >=  0.05: return "🔵 Somewhat Bullish"
    elif score >= -0.05: return "⚪ Neutral"
    elif score >= -0.20: return "🟠 Somewhat Bearish"
    else:                return "🔴 Bearish"


# ─────────────────────────────────────────────
#  SOURCE 1: FINNHUB
# ─────────────────────────────────────────────

def _fetch_finnhub_sentiment(tickers: list, api_key: str) -> dict:
    """
    Finnhub /news-sentiment returns a pre-computed buzz + sentiment score.
    Very fast — one call per ticker, no NLP needed on our side.
    """
    results = {}
    base = "https://finnhub.io/api/v1"

    for ticker in tickers:
        try:
            # News sentiment score
            r = requests.get(
                f"{base}/news-sentiment?symbol={ticker}&token={api_key}",
                timeout=8
            )
            if r.status_code != 200:
                continue
            d = r.json()

            sentiment = d.get("sentiment", {})
            score = sentiment.get("bullishPercent", None)
            if score is None:
                continue

            # Finnhub returns bullishPercent (0-1); convert to -1..+1 scale
            normalized = (float(score) - 0.5) * 2

            # Also get a recent headline
            r2 = requests.get(
                f"{base}/company-news?symbol={ticker}"
                f"&from={(date.today()-timedelta(days=_LOOKBACK_DAYS)).strftime('%Y-%m-%d')}"
                f"&to={date.today().strftime('%Y-%m-%d')}&token={api_key}",
                timeout=8
            )
            headline = ""
            source   = "Finnhub"
            if r2.status_code == 200:
                news = r2.json()
                if news:
                    headline = news[0].get("headline", "")[:100]
                    source   = news[0].get("source", "Finnhub")

            results[ticker] = {
                "score":    round(normalized, 3),
                "label":    _score_to_label(normalized),
                "headline": headline,
                "source":   source,
                "age_days": 0,
            }
            time.sleep(0.05)   # 60 req/min limit → 1 req/50ms is safe

        except Exception:
            continue

    return results


# ─────────────────────────────────────────────
#  SOURCE 2: ALPHA VANTAGE (BATCH)
# ─────────────────────────────────────────────

def _fetch_av_sentiment(tickers: list, api_key: str) -> dict:
    """
    Alpha Vantage NEWS_SENTIMENT: batch up to 5 tickers per call.
    Returns pre-scored sentiment per ticker — no NLP needed.
    """
    results  = {}
    from_dt  = (date.today() - timedelta(days=_LOOKBACK_DAYS)).strftime("%Y%m%dT0000")

    for i in range(0, len(tickers), 5):
        batch = tickers[i:i+5]
        try:
            url = (
                f"https://www.alphavantage.co/query"
                f"?function=NEWS_SENTIMENT"
                f"&tickers={','.join(batch)}"
                f"&time_from={from_dt}"
                f"&limit=50&apikey={api_key}"
            )
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                continue
            data = r.json()

            if "Information" in data or "Note" in data:
                # Rate limited
                print(f"      ⚠️  Alpha Vantage rate limit hit. Using available data.")
                break

            ticker_agg: dict[str, list] = {t: [] for t in batch}
            ticker_hl:  dict[str, str]  = {t: ""  for t in batch}
            ticker_src: dict[str, str]  = {t: ""  for t in batch}

            for article in data.get("feed", []):
                for ts in article.get("ticker_sentiment", []):
                    sym = ts.get("ticker", "")
                    if sym not in ticker_agg:
                        continue
                    try:
                        sc = float(ts.get("ticker_sentiment_score", 0))
                        ticker_agg[sym].append(sc)
                        if not ticker_hl[sym]:
                            ticker_hl[sym]  = article.get("title", "")[:100]
                            ticker_src[sym] = article.get("source", "Alpha Vantage")
                    except (TypeError, ValueError):
                        continue

            for t in batch:
                scores = ticker_agg[t]
                if scores:
                    avg = sum(scores) / len(scores)
                    results[t] = {
                        "score":    round(avg, 3),
                        "label":    _score_to_label(avg),
                        "headline": ticker_hl[t],
                        "source":   ticker_src[t] or "Alpha Vantage",
                        "age_days": 0,
                    }

        except Exception:
            continue

        time.sleep(1.2)   # AV free tier: ~5 req/min

    return results


# ─────────────────────────────────────────────
#  SOURCE 3: EXISTING CSV FALLBACK
# ─────────────────────────────────────────────

def _load_from_csv(data_dir: str, tickers: list) -> dict:
    """
    Fall back to the last news_sentiment.csv produced by news_sentiment.py.
    Only use rows that are less than 3 days old.
    """
    import pandas as pd
    csv_path = os.path.join(data_dir, "news_sentiment.csv")
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
        results = {}
        today = date.today()

        for _, row in df.iterrows():
            ticker = str(row.get("ticker", ""))
            if ticker not in tickers:
                continue
            score = float(row.get("sentiment_score", 0))
            # Estimate age from headline_date column if present
            age_days = 1
            try:
                hd = str(row.get("headline_date", ""))[:8]
                hdate = datetime.strptime(hd, "%Y%m%d").date()
                age_days = (today - hdate).days
            except Exception:
                pass
            if age_days > 3:
                continue

            results[ticker] = {
                "score":    round(score, 3),
                "label":    _score_to_label(score),
                "headline": str(row.get("top_headline", ""))[:100],
                "source":   f"{row.get('data_source','CSV')} (cached)",
                "age_days": age_days,
            }
        return results
    except Exception:
        return {}


# ─────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────

def fetch_auto_sentiment(
    tickers: list,
    data_dir: str,
    finnhub_key: str = "",
    av_key: str = "",
) -> dict:
    """
    Fetch news sentiment for all tickers automatically.

    Returns dict: {ticker: {score, label, headline, source, age_days}}

    Priority: Finnhub → Alpha Vantage → existing CSV → empty dict.
    Caches results for _CACHE_HOURS hours.
    """
    cache_path = os.path.join(data_dir, "auto_sentiment_cache.json")
    cached     = _load_cache(cache_path)
    today_str  = date.today().strftime("%Y-%m-%d")

    if cached.get("_date") == today_str and len(cached) > 5:
        print(f"  📰 Sentiment loaded from cache ({len(cached)-1} tickers).")
        return {k: v for k, v in cached.items() if k != "_date"}

    print(f"  📰 Fetching news sentiment for {len(tickers)} tickers...")
    results = {}

    # ── Try Finnhub first ────────────────────
    if finnhub_key:
        fh_results = _fetch_finnhub_sentiment(tickers, finnhub_key)
        results.update(fh_results)
        missing = [t for t in tickers if t not in results]
        print(f"      Finnhub: {len(fh_results)} tickers scored. {len(missing)} still needed.")

    # ── Alpha Vantage for missing tickers ────
    missing = [t for t in tickers if t not in results]
    if missing and av_key:
        av_results = _fetch_av_sentiment(missing, av_key)
        results.update(av_results)
        missing2 = [t for t in tickers if t not in results]
        print(f"      Alpha Vantage: {len(av_results)} more tickers scored. {len(missing2)} still needed.")

    # ── CSV fallback for anything remaining ──
    still_missing = [t for t in tickers if t not in results]
    if still_missing:
        csv_results = _load_from_csv(data_dir, still_missing)
        results.update(csv_results)
        if csv_results:
            print(f"      CSV fallback: {len(csv_results)} tickers from last news run.")

    print(f"  ✅ Sentiment complete: {len(results)}/{len(tickers)} tickers scored.")

    _save_cache(cache_path, {"_date": today_str, **results})
    return results
