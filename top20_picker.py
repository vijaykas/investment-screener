"""
============================================================
  INVESTMENT INTELLIGENCE — PHASE 8
  Top 20 High-Yield Predictions
  Author: Claude (Vijay's Investment Project)
  Updated: 2026-04-13
============================================================

WHAT THIS DOES:
  Combines every available signal — technical scores, ML predictions,
  news sentiment, and momentum — to rank the top 20 highest-potential
  investments across ALL asset classes (stocks + ETFs + crypto).

SCORING FORMULA:
  Stocks  (0–100):
    50 pts  Technical   →  combined_score / 75 × 50
    30 pts  ML Signal   →  avg(prob_up_3m..12m) × 30   [or signal-proxy if no ML]
    20 pts  Sentiment   →  (sentiment_score + 1) / 2 × 20

  ETFs  (0–100):
    50 pts  Technical   →  tech_score / 50 × 50
    30 pts  Momentum    →  momentum_score / 30 × 30
    20 pts  Sentiment   →  (sentiment_score + 1) / 2 × 20

  Crypto  (0–100):
    40 pts  Technical   →  tech_score / 40 × 40
    25 pts  Momentum    →  momentum_score / 25 × 25  (vs BTC)
    20 pts  Quality     →  quality_score / 20 × 20   (on-chain proxy)
    15 pts  Sentiment   →  (sentiment_score + 1) / 2 × 15

OUTPUT (data/top20_predictions.csv):
  rank, ticker, name, asset_type, sector_or_category,
  yield_potential_score, signal, sentiment_label, sentiment_score,
  predicted_yield_range, top_headline, theme_tags,
  technical_component, ml_or_momentum_component, sentiment_component,
  price, day_chg_pct, score_raw, run_date

HOW TO RUN:
  python3 top20_picker.py            # standalone
  python3 invest.py --top20          # via unified runner
  python3 invest.py --quick          # included in quick run
============================================================
"""

import os
import sys
import json
from datetime import date

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, TOP20_CONFIG

OUTPUT_CSV  = os.path.join(DATA_DIR, TOP20_CONFIG["output_file"])
TODAY_STR   = date.today().strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def _yield_range_label(score: float) -> str:
    """Map a yield potential score (0-100) to a return expectation string."""
    for threshold, label in TOP20_CONFIG["yield_ranges"]:
        if score >= threshold:
            return label
    return "Uncertain outlook"


def _safe_float(val, default=0.0) -> float:
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _signal_to_ml_proxy(signal: str, score: float, asset_type: str) -> float:
    """
    Convert signal label to a 0..1 ML-equivalent probability.
    Used when ml_predictions.csv is absent or ticker not covered.
    """
    sig = signal.upper()
    max_score = 75.0 if asset_type == "STOCK" else 100.0
    # Base from signal label
    if "STRONG BUY" in sig:
        base = 0.72
    elif "BUY" in sig:
        base = 0.62
    elif "HOLD" in sig:
        base = 0.50
    elif "CAUTION" in sig:
        base = 0.38
    else:
        base = 0.28

    # Blend with normalised score
    normalised = _safe_float(score) / max_score
    return base * 0.6 + normalised * 0.4


# ─────────────────────────────────────────────────────────────
#  LOAD DATA SOURCES
# ─────────────────────────────────────────────────────────────

def load_stock_data() -> pd.DataFrame:
    """Load stock screener results. Returns empty DataFrame if not found."""
    csv = os.path.join(DATA_DIR, "stock_screener_results.csv")
    if not os.path.exists(csv):
        return pd.DataFrame()
    df = pd.read_csv(csv)
    df["asset_type"] = "STOCK"
    return df


def load_etf_data() -> pd.DataFrame:
    """Load ETF screener results. Returns empty DataFrame if not found."""
    csv = os.path.join(DATA_DIR, "etf_screener_results.csv")
    if not os.path.exists(csv):
        return pd.DataFrame()
    df = pd.read_csv(csv)
    df["asset_type"] = "ETF"
    return df


def load_ml_data() -> dict:
    """
    Load ML predictions.
    Returns {ticker: {"avg_mid_prob": float}} or {} if unavailable.
    """
    csv = os.path.join(DATA_DIR, "ml_predictions.csv")
    if not os.path.exists(csv):
        return {}
    try:
        df = pd.read_csv(csv)
        if "ticker" not in df.columns:
            return {}
        result = {}
        mid_horizons = ["3m", "4m", "5m", "6m", "7m", "8m", "9m", "10m", "11m", "12m"]
        for _, row in df.iterrows():
            ticker = str(row["ticker"])
            probs = []
            for h in mid_horizons:
                col = f"prob_up_Ensemble_{h}"
                if col in df.columns and pd.notna(row.get(col)):
                    probs.append(float(row[col]) / 100.0)   # convert 0-100 → 0-1
            if probs:
                result[ticker] = {"avg_mid_prob": sum(probs) / len(probs)}
        return result
    except Exception:
        return {}


def load_crypto_data() -> pd.DataFrame:
    """Load crypto screener results. Returns empty DataFrame if not found."""
    csv = os.path.join(DATA_DIR, "crypto_screener_results.csv")
    if not os.path.exists(csv):
        return pd.DataFrame()
    df = pd.read_csv(csv)
    df["asset_type"] = "CRYPTO"
    return df


def load_sentiment_data() -> dict:
    """
    Load news sentiment scores.
    Returns {ticker: {"score": float, "label": str, "headline": str, "themes": str}} or {}.
    """
    csv = os.path.join(DATA_DIR, "news_sentiment.csv")
    if not os.path.exists(csv):
        return {}
    try:
        df = pd.read_csv(csv)
        if "ticker" not in df.columns:
            return {}
        result = {}
        for _, row in df.iterrows():
            result[str(row["ticker"])] = {
                "score":    _safe_float(row.get("sentiment_score", 0)),
                "label":    str(row.get("sentiment_label", "⚪ Neutral")),
                "headline": str(row.get("top_headline", "")),
                "themes":   str(row.get("theme_tags", "")),
                "pct":      int(row.get("sentiment_pct", 50)),
            }
        return result
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────
#  SCORING ENGINE
# ─────────────────────────────────────────────────────────────

def score_stock(row: pd.Series, ml_data: dict, sentiment_data: dict) -> dict:
    """
    Compute yield_potential_score for a single stock row.
    Returns a dict of components + final score.
    """
    ticker  = str(row.get("ticker", ""))
    signal  = str(row.get("combined_signal_label", row.get("signal", "HOLD")))
    raw_score = _safe_float(row.get("combined_score", row.get("total_score", 40)))

    # ── Technical component (0–50) ────────────────────────────
    tech_comp = min(50.0, (raw_score / 75.0) * 50.0)

    # ── ML component (0–30) ──────────────────────────────────
    ml_info = ml_data.get(ticker, {})
    if ml_info:
        prob = ml_info.get("avg_mid_prob", 0.50)
    else:
        prob = _signal_to_ml_proxy(signal, raw_score, "STOCK")
    ml_comp = prob * 30.0          # 0 → 30

    # ── Sentiment component (0–20) ───────────────────────────
    sent_info = sentiment_data.get(ticker, {})
    sent_score = sent_info.get("score", 0.0)
    sent_comp  = ((sent_score + 1.0) / 2.0) * 20.0   # -1..+1 → 0..20

    # ── Total ─────────────────────────────────────────────────
    total = tech_comp + ml_comp + sent_comp

    return {
        "yield_potential_score":    round(min(100.0, total), 1),
        "technical_component":      round(tech_comp, 1),
        "ml_or_momentum_component": round(ml_comp, 1),
        "sentiment_component":      round(sent_comp, 1),
        "sentiment_score":          round(sent_score, 3),
        "sentiment_label":          sent_info.get("label", "⚪ Neutral"),
        "top_headline":             sent_info.get("headline", ""),
        "theme_tags":               sent_info.get("themes", ""),
    }


def score_etf(row: pd.Series, sentiment_data: dict) -> dict:
    """
    Compute yield_potential_score for a single ETF row.
    Returns a dict of components + final score.
    """
    ticker    = str(row.get("ticker", ""))
    signal    = str(row.get("signal", "HOLD"))
    tech_raw  = _safe_float(row.get("tech_score", 0))    # 0–50
    mom_raw   = _safe_float(row.get("momentum_score", 0)) # 0–30
    total_raw = _safe_float(row.get("total_score", 0))    # 0–100

    # ── Technical component (0–50) ────────────────────────────
    tech_comp = min(50.0, (tech_raw / 50.0) * 50.0)

    # ── Momentum component (0–30) ─────────────────────────────
    if mom_raw > 0:
        mom_comp = min(30.0, (mom_raw / 30.0) * 30.0)
    else:
        # Derive from total_score if individual components missing
        mom_comp = min(30.0, (total_raw / 100.0) * 30.0)

    # ── Sentiment component (0–20) ───────────────────────────
    sent_info  = sentiment_data.get(ticker, {})
    sent_score = sent_info.get("score", 0.0)
    sent_comp  = ((sent_score + 1.0) / 2.0) * 20.0

    # ── Total ─────────────────────────────────────────────────
    total = tech_comp + mom_comp + sent_comp

    return {
        "yield_potential_score":    round(min(100.0, total), 1),
        "technical_component":      round(tech_comp, 1),
        "ml_or_momentum_component": round(mom_comp, 1),
        "sentiment_component":      round(sent_comp, 1),
        "sentiment_score":          round(sent_score, 3),
        "sentiment_label":          sent_info.get("label", "⚪ Neutral"),
        "top_headline":             sent_info.get("headline", ""),
        "theme_tags":               sent_info.get("themes", ""),
    }


def score_crypto(row: pd.Series, sentiment_data: dict) -> dict:
    """
    Compute yield_potential_score for a single crypto row.
    Weights: Technical 40 + Momentum 25 + Quality 20 + Sentiment 15 = 100
    """
    ticker      = str(row.get("ticker", "")).replace("-USD", "")
    signal      = str(row.get("signal", "HOLD"))
    tech_raw    = _safe_float(row.get("tech_score", 0))       # 0–40
    mom_raw     = _safe_float(row.get("momentum_score", 0))   # 0–25
    quality_raw = _safe_float(row.get("quality_score", 0))    # 0–20
    sent_raw    = _safe_float(row.get("sentiment_score", 0))  # 0–15
    total_raw   = _safe_float(row.get("total_score", 0))      # 0–100

    # ── Technical component (0–40) ────────────────────────────
    if tech_raw > 0:
        tech_comp = min(40.0, (tech_raw / 40.0) * 40.0)
    else:
        tech_comp = min(40.0, (total_raw / 100.0) * 40.0)

    # ── Momentum component (0–25) ─────────────────────────────
    if mom_raw > 0:
        mom_comp = min(25.0, (mom_raw / 25.0) * 25.0)
    else:
        mom_comp = min(25.0, (total_raw / 100.0) * 25.0)

    # ── Quality component (0–20) ──────────────────────────────
    if quality_raw > 0:
        qual_comp = min(20.0, (quality_raw / 20.0) * 20.0)
    else:
        qual_comp = min(20.0, (total_raw / 100.0) * 20.0)

    # ── Sentiment component (0–15) ───────────────────────────
    # Check sentiment CSV first (with -USD stripped key), then use crypto screener value
    sent_info = sentiment_data.get(ticker, sentiment_data.get(ticker + "-USD", {}))
    if sent_info:
        sent_score = sent_info.get("score", 0.0)
        sent_comp  = ((sent_score + 1.0) / 2.0) * 15.0
    else:
        # Use the screener's sentiment_score (already 0-15 pts)
        sent_score = (sent_raw / 15.0) * 2.0 - 1.0   # back-convert to -1..+1
        sent_comp  = min(15.0, sent_raw)

    sent_info_result = sent_info if sent_info else {}

    # ── Total ─────────────────────────────────────────────────
    total = tech_comp + mom_comp + qual_comp + sent_comp

    return {
        "yield_potential_score":    round(min(100.0, total), 1),
        "technical_component":      round(tech_comp, 1),
        "ml_or_momentum_component": round(mom_comp, 1),
        "quality_component":        round(qual_comp, 1),
        "sentiment_component":      round(sent_comp, 1),
        "sentiment_score":          round(sent_score if sent_info else _safe_float(sent_raw), 3),
        "sentiment_label":          sent_info_result.get("label", "⚪ Neutral"),
        "top_headline":             sent_info_result.get("headline", ""),
        "theme_tags":               sent_info_result.get("themes", ""),
    }


# ─────────────────────────────────────────────────────────────
#  MAIN RUNNER
# ─────────────────────────────────────────────────────────────

def run_top20():
    """
    Combine all signals to produce the top 20 high-yield predictions.
    Saves data/top20_predictions.csv.
    """
    print("\n🏆  Phase 8 — Top 20 High-Yield Predictions")
    print("─" * 50)

    # ── Load all data sources ─────────────────────────────────
    df_stocks = load_stock_data()
    df_etfs   = load_etf_data()
    df_crypto = load_crypto_data()
    ml_data   = load_ml_data()
    sent_data = load_sentiment_data()

    print(f"  📊 Stocks loaded:     {len(df_stocks)}")
    print(f"  📈 ETFs loaded:       {len(df_etfs)}")
    print(f"  ₿  Cryptos loaded:    {len(df_crypto)}")
    print(f"  🤖 ML tickers:        {len(ml_data)}")
    print(f"  📰 Sentiment tickers: {len(sent_data)}")

    if df_stocks.empty and df_etfs.empty and df_crypto.empty:
        print("  ❌ No screener data available — run Phase 1, 4b, and/or 4c first")
        return

    # ── Score all candidates ──────────────────────────────────
    all_rows = []

    # Stocks
    for _, row in df_stocks.iterrows():
        components = score_stock(row, ml_data, sent_data)
        if components["yield_potential_score"] < TOP20_CONFIG["min_score"]:
            continue

        ticker   = str(row.get("ticker", ""))
        sector   = str(row.get("sector", "Unknown"))
        name     = str(row.get("name", ticker))
        signal   = str(row.get("combined_signal_label", row.get("signal", "HOLD")))
        price    = _safe_float(row.get("price", 0))
        day_chg  = _safe_float(row.get("day_chg_pct", 0))
        raw_sc   = _safe_float(row.get("combined_score", row.get("total_score", 0)))
        rsi      = _safe_float(row.get("rsi", 50))
        above200 = bool(row.get("above_200ma", False))
        market_cap = _safe_float(row.get("market_cap_B", 0))

        all_rows.append({
            "ticker":               ticker,
            "name":                 name[:40],
            "asset_type":           "STOCK",
            "sector_or_category":   sector,
            "signal":               signal,
            "price":                round(price, 2),
            "day_chg_pct":          round(day_chg, 2),
            "score_raw":            round(raw_sc, 1),
            "rsi":                  round(rsi, 1),
            "above_200ma":          above200,
            "market_cap_B":         round(market_cap, 1),
            "run_date":             TODAY_STR,
            **components,
        })

    # ETFs
    for _, row in df_etfs.iterrows():
        components = score_etf(row, sent_data)
        if components["yield_potential_score"] < TOP20_CONFIG["min_score"]:
            continue

        ticker   = str(row.get("ticker", ""))
        category = str(row.get("category", "ETF"))
        name     = str(row.get("name", ticker))
        signal   = str(row.get("signal", "HOLD"))
        price    = _safe_float(row.get("price", 0))
        day_chg  = _safe_float(row.get("day_chg_pct", 0))
        raw_sc   = _safe_float(row.get("total_score", 0))
        rsi      = _safe_float(row.get("rsi", 50))
        above200 = bool(row.get("above_200ma", False))
        rel_3m   = _safe_float(row.get("rel_3m", 0))
        expense  = _safe_float(row.get("expense_ratio", 0))

        all_rows.append({
            "ticker":               ticker,
            "name":                 name[:40],
            "asset_type":           "ETF",
            "sector_or_category":   category,
            "signal":               signal,
            "price":                round(price, 2),
            "day_chg_pct":          round(day_chg, 2),
            "score_raw":            round(raw_sc, 1),
            "rsi":                  round(rsi, 1),
            "above_200ma":          above200,
            "rel_3m_vs_spy":        round(rel_3m, 2),
            "expense_ratio":        round(expense, 3),
            "run_date":             TODAY_STR,
            **components,
        })

    # Crypto
    for _, row in df_crypto.iterrows():
        components = score_crypto(row, sent_data)
        if components["yield_potential_score"] < TOP20_CONFIG["min_score"]:
            continue

        ticker   = str(row.get("ticker", ""))
        category = str(row.get("category", "Crypto"))
        name     = str(row.get("name", ticker))
        signal   = str(row.get("signal", "HOLD"))
        price    = _safe_float(row.get("price", 0))
        day_chg  = _safe_float(row.get("day_chg_pct", 0))
        raw_sc   = _safe_float(row.get("total_score", 0))
        rsi      = _safe_float(row.get("rsi", 50))
        above200 = bool(row.get("above_200ma", False))
        rel_30d  = _safe_float(row.get("rel_30d_vs_btc", 0))
        market_cap = _safe_float(row.get("market_cap_B", 0))

        all_rows.append({
            "ticker":               ticker,
            "name":                 name[:40],
            "asset_type":           "CRYPTO",
            "sector_or_category":   category,
            "signal":               signal,
            "price":                round(price, 2),
            "day_chg_pct":          round(day_chg, 2),
            "score_raw":            round(raw_sc, 1),
            "rsi":                  round(rsi, 1),
            "above_200ma":          above200,
            "rel_30d_vs_btc":       round(rel_30d, 2),
            "market_cap_B":         round(market_cap, 1),
            "run_date":             TODAY_STR,
            **components,
        })

    if not all_rows:
        print("  ❌ No candidates passed the minimum score threshold")
        return

    # ── Sort and apply caps ───────────────────────────────────
    df_all = pd.DataFrame(all_rows)
    df_all = df_all.sort_values("yield_potential_score", ascending=False)

    # Apply stock/ETF/crypto caps
    max_stocks = TOP20_CONFIG["max_stocks"]
    max_etfs   = TOP20_CONFIG["max_etfs"]
    max_crypto = TOP20_CONFIG.get("max_crypto", 6)
    n_picks    = TOP20_CONFIG["n_picks"]

    df_stocks_top = df_all[df_all["asset_type"] == "STOCK"].head(max_stocks)
    df_etfs_top   = df_all[df_all["asset_type"] == "ETF"].head(max_etfs)
    df_crypto_top = df_all[df_all["asset_type"] == "CRYPTO"].head(max_crypto)

    df_combined = pd.concat([df_stocks_top, df_etfs_top, df_crypto_top])
    df_combined = df_combined.sort_values("yield_potential_score", ascending=False)
    df_top20    = df_combined.head(n_picks).reset_index(drop=True)
    df_top20.insert(0, "rank", range(1, len(df_top20) + 1))

    # ── Add yield range labels ────────────────────────────────
    df_top20["predicted_yield_range"] = df_top20["yield_potential_score"].apply(_yield_range_label)

    # ── Save ──────────────────────────────────────────────────
    col_order = [
        "rank", "ticker", "name", "asset_type", "sector_or_category",
        "yield_potential_score", "signal", "sentiment_label", "sentiment_score",
        "predicted_yield_range", "top_headline", "theme_tags",
        "technical_component", "ml_or_momentum_component", "sentiment_component",
        "price", "day_chg_pct", "score_raw", "rsi", "above_200ma", "run_date",
    ]
    for col in ["rel_3m_vs_spy", "expense_ratio", "market_cap_B"]:
        if col in df_top20.columns:
            col_order.append(col)

    output_cols = [c for c in col_order if c in df_top20.columns]
    df_top20[output_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\n  ✅ Saved: {OUTPUT_CSV}")

    # ── Print summary ─────────────────────────────────────────
    print(f"\n  🏆 TOP 20 HIGH-YIELD PREDICTIONS — {TODAY_STR}")
    print(f"  {'─'*64}")
    print(f"  {'#':<3} {'Ticker':<7} {'Type':<6} {'Score':<7} {'Signal':<22} {'Yield Range':<28} {'Sentiment'}")
    print(f"  {'─'*64}")
    for _, r in df_top20.iterrows():
        print(
            f"  {int(r['rank']):<3} {r['ticker']:<7} {r['asset_type']:<6} "
            f"{r['yield_potential_score']:<7.1f} {r['signal']:<22} "
            f"{r['predicted_yield_range']:<28} {r['sentiment_label']}"
        )

    n_stock  = len(df_top20[df_top20["asset_type"] == "STOCK"])
    n_etf    = len(df_top20[df_top20["asset_type"] == "ETF"])
    n_crypto = len(df_top20[df_top20["asset_type"] == "CRYPTO"])
    avg_score = df_top20["yield_potential_score"].mean()
    print(f"\n  📊 Mix: {n_stock} stocks, {n_etf} ETFs, {n_crypto} cryptos  |  Avg score: {avg_score:.1f}/100")

    return df_top20


if __name__ == "__main__":
    run_top20()
