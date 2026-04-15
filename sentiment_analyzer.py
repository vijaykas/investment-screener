"""
============================================================
  INVESTMENT INTELLIGENCE — PHASE 6
  Sentiment & News Layer
  Author: Claude (Vijay's Investment Project)
  Updated: 2026-04-12
============================================================

WHAT THIS DOES:
  Enriches every stock in your watchlist with four independent
  sentiment signals, then combines them into a single Sentiment
  Score (0–100) that integrates with the Phase 1 technical score.

  SIGNAL SOURCES:
    1. News Sentiment      — VADER NLP on recent headlines (yfinance)
    2. Analyst Consensus   — buy/sell/hold counts + price target upside
    3. Earnings Surprise   — beat/miss history (last 4 quarters)
    4. Insider Activity    — net insider buying vs selling pressure

  COMPOSITE SCORING:
    News Sentiment     35 pts  (recency-weighted VADER compound scores)
    Analyst Consensus  30 pts  (strong buy → sell spectrum + PT upside)
    Earnings Surprise  20 pts  (beat magnitude + consistency)
    Insider Signal     15 pts  (net buy ratio over last 6 months)
    ─────────────────────────
    Total Sentiment    100 pts

  COMBINED SIGNAL (Technical + Sentiment):
    Phase 1 Technical Score (0–75)  × 0.55 weight
    Phase 6 Sentiment Score (0–100) × 0.45 weight
    → Normalised Combined Score (0–100) + upgraded signal label

  OUTPUTS:
    • Terminal: ranked sentiment table + combined signal leaderboard
    • sentiment_results.csv      — full scores per ticker
    • charts/sentiment_dashboard.html  — composite sentiment bar chart
    • charts/news_sentiment_timeline.html — sentiment over time per ticker
    • charts/analyst_heatmap.html  — analyst consensus heatmap
    • Appends sentiment column to daily_reports/<today>.html if it exists

HOW TO RUN:
  python3 sentiment_analyzer.py

  Integrates automatically with daily_monitor.py — run it after the
  screener to get a combined Technical + Sentiment ranking.

DEPENDENCIES:
  pip install vaderSentiment textblob yfinance pandas numpy plotly
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
import os, sys, json, time, re

# ─────────────────────────────────────────────
#  CONFIGURATION  (loaded from config.py)
# ─────────────────────────────────────────────
from config import SENTIMENT_CONFIG, DATA_DIR, CHARTS_DIR, get_watchlist

OUTPUT_DIR = DATA_DIR   # CSVs written to data/

# Initialise VADER once — reuse across all calls
VADER = SentimentIntensityAnalyzer()

# Financial keyword boosters for VADER (amplify domain-specific terms)
FINANCIAL_BOOSTERS = {
    # Very bullish
    "beat":          2.0, "beats":       2.0, "surge":       1.8,
    "record":        1.5, "breakthrough":1.5, "rally":       1.5,
    "soar":          1.8, "upgrade":     2.0, "outperform":  1.8,
    "buy":           1.5, "bullish":     1.8, "profitable":  1.5,
    "dividend":      1.2, "buyback":     1.3, "overweight":  1.5,
    # Very bearish
    "miss":         -2.0, "misses":     -2.0, "crash":      -2.0,
    "downgrade":    -2.0, "underperform":-1.8,"selloff":    -1.8,
    "decline":      -1.5, "cut":        -1.5, "loss":       -1.5,
    "bearish":      -1.8, "warning":    -1.5, "layoffs":    -1.5,
    "lawsuit":      -1.5, "fraud":      -2.5, "bankrupt":   -3.0,
    "recall":       -1.5, "investigation":-2.0,"sell":       -1.3,
}


# ─────────────────────────────────────────────
#  SIGNAL 1 — NEWS SENTIMENT
# ─────────────────────────────────────────────

def score_news(ticker: str, cfg: dict) -> tuple[float, list[dict]]:
    """
    Fetch recent news headlines via yfinance, score each with VADER
    (+ financial keyword boosters), weight by recency, return 0–35 score.
    """
    try:
        news_items = yf.Ticker(ticker).news or []
    except Exception:
        return 0.0, []

    cutoff_ts = (datetime.now() - timedelta(days=cfg["news_lookback_days"])).timestamp()
    now_ts    = datetime.now().timestamp()

    scored = []
    for item in news_items[:cfg["max_news_per_ticker"]]:
        pub_ts = item.get("providerPublishTime", 0)
        if pub_ts < cutoff_ts:
            continue

        # Pull text — headline + summary if available
        title   = item.get("title", "")
        summary = item.get("summary", "") or item.get("description", "")
        text    = f"{title}. {summary}".strip()
        if not text or len(text) < 10:
            continue

        # VADER base score
        vs = VADER.polarity_scores(text)
        compound = vs["compound"]   # [-1, +1]

        # Financial keyword boost
        text_lower = text.lower()
        boost = 0.0
        for word, weight in FINANCIAL_BOOSTERS.items():
            if re.search(r'\b' + word + r'\b', text_lower):
                boost += weight * 0.05  # Scale boost

        compound = max(-1.0, min(1.0, compound + boost))

        # Recency weight: articles published today get weight 1.0,
        # older articles decay linearly over the lookback window
        age_days = (now_ts - pub_ts) / 86400
        recency_weight = max(0.1, 1.0 - age_days / cfg["news_lookback_days"])

        scored.append({
            "title":          title[:90],
            "published":      datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d"),
            "compound":       round(compound, 3),
            "recency_weight": round(recency_weight, 2),
            "weighted":       round(compound * recency_weight, 3),
        })

    if not scored:
        return 0.0, []

    # Weighted average compound score → normalise to 0–35
    total_weight   = sum(a["recency_weight"] for a in scored)
    weighted_avg   = sum(a["weighted"] for a in scored) / total_weight
    # Map [-1, +1] → [0, 35]
    news_score = (weighted_avg + 1) / 2 * cfg["weights"]["news"]

    return round(news_score, 2), scored


# ─────────────────────────────────────────────
#  SIGNAL 2 — ANALYST CONSENSUS
# ─────────────────────────────────────────────

def score_analyst(ticker: str, cfg: dict) -> tuple[float, dict]:
    """
    Fetch analyst recommendations and price target.
    Counts Strong Buy / Buy / Hold / Sell / Strong Sell from most recent period.
    Returns 0–30 score.
    """
    info = {}
    try:
        t = yf.Ticker(ticker)
        tk_info = t.info

        # Current price target upside
        current_price  = tk_info.get("currentPrice") or tk_info.get("regularMarketPrice")
        target_mean    = tk_info.get("targetMeanPrice")
        target_high    = tk_info.get("targetHighPrice")
        target_low     = tk_info.get("targetLowPrice")
        n_analysts     = tk_info.get("numberOfAnalystOpinions", 0) or 0
        recommend_key  = tk_info.get("recommendationKey", "")   # e.g. "buy", "strong_buy"
        recommend_mean = tk_info.get("recommendationMean")       # 1=Strong Buy, 5=Strong Sell

        upside = None
        if current_price and target_mean and current_price > 0:
            upside = (target_mean / current_price - 1) * 100

        info = {
            "n_analysts":     n_analysts,
            "recommendation": recommend_key,
            "rec_mean":       recommend_mean,
            "target_mean":    target_mean,
            "target_high":    target_high,
            "target_low":     target_low,
            "current_price":  current_price,
            "upside_pct":     round(upside, 1) if upside is not None else None,
        }

        if n_analysts < cfg["min_analysts"] or recommend_mean is None:
            return 0.0, info

        # recommendationMean: 1=Strong Buy, 2=Buy, 3=Hold, 4=Sell, 5=Strong Sell
        # Invert and normalise to [0, 1]
        rec_score_norm = (5 - recommend_mean) / 4   # 0=Strong Sell, 1=Strong Buy

        # Upside score (0–0.5 of remaining weight)
        upside_score_norm = 0.0
        if upside is not None:
            if upside > 30:    upside_score_norm = 1.0
            elif upside > 15:  upside_score_norm = 0.7
            elif upside > 5:   upside_score_norm = 0.4
            elif upside > 0:   upside_score_norm = 0.2
            else:              upside_score_norm = 0.0   # Target below current price

        combined_norm = rec_score_norm * 0.65 + upside_score_norm * 0.35
        analyst_score = combined_norm * cfg["weights"]["analyst"]
        return round(analyst_score, 2), info

    except Exception:
        return 0.0, info


# ─────────────────────────────────────────────
#  SIGNAL 3 — EARNINGS SURPRISE
# ─────────────────────────────────────────────

def score_earnings(ticker: str, cfg: dict) -> tuple[float, list[dict]]:
    """
    Pull earnings history (actual EPS vs estimated EPS).
    Rewards consistent beats, especially large surprises.
    Returns 0–20 score.
    """
    try:
        earnings_hist = yf.Ticker(ticker).earnings_history
        if earnings_hist is None or earnings_hist.empty:
            return 0.0, []
    except Exception:
        return 0.0, []

    # Take the N most recent quarters
    n = cfg["earnings_quarters"]
    recent = earnings_hist.head(n).copy()

    rows = []
    beat_scores = []

    for _, row in recent.iterrows():
        eps_est    = row.get("epsEstimate")
        eps_actual = row.get("epsActual")
        surprise   = row.get("surprisePercent")
        qtr        = str(row.get("quarter", ""))

        if eps_est is None or eps_actual is None or eps_est == 0:
            continue

        if surprise is None:
            surprise = (eps_actual / eps_est - 1) * 100 if eps_est != 0 else 0

        # Score: beat → positive, miss → negative
        if surprise > 15:    beat = 1.0    # Large beat
        elif surprise > 5:   beat = 0.75
        elif surprise > 0:   beat = 0.5    # Small beat
        elif surprise > -5:  beat = 0.2    # Slight miss
        elif surprise > -15: beat = 0.05   # Miss
        else:                beat = 0.0    # Large miss

        beat_scores.append(beat)
        rows.append({
            "quarter":     qtr,
            "eps_est":     round(eps_est, 3),
            "eps_actual":  round(eps_actual, 3),
            "surprise_%":  round(surprise, 1),
            "beat_score":  round(beat, 2),
        })

    if not beat_scores:
        return 0.0, rows

    # Recency-weighted average (most recent quarter matters most)
    weights = [1.5, 1.2, 0.9, 0.6][:len(beat_scores)]
    w_avg = np.average(beat_scores, weights=weights)
    earnings_score = w_avg * cfg["weights"]["earnings"]

    return round(earnings_score, 2), rows


# ─────────────────────────────────────────────
#  SIGNAL 4 — INSIDER ACTIVITY
# ─────────────────────────────────────────────

def score_insider(ticker: str, cfg: dict) -> tuple[float, dict]:
    """
    Net insider buying vs selling over the lookback window.
    Insider buys are a strong bullish signal; large sells are bearish.
    Returns 0–15 score.
    """
    try:
        insider_df = yf.Ticker(ticker).insider_transactions
        if insider_df is None or insider_df.empty:
            return 7.5, {}   # Neutral if no data (midpoint of 0–15)
    except Exception:
        return 7.5, {}

    try:
        # Filter to lookback window
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=cfg["insider_lookback_days"])

        # Handle different column naming across yfinance versions
        date_col = None
        for col in ["startDate", "Date", "date", "transactionDate"]:
            if col in insider_df.columns:
                date_col = col
                break

        if date_col:
            insider_df[date_col] = pd.to_datetime(insider_df[date_col], errors="coerce")
            recent = insider_df[insider_df[date_col] >= cutoff].copy()
        else:
            recent = insider_df.copy()

        if recent.empty:
            return 7.5, {}

        # Find shares column
        shares_col = None
        for col in ["shares", "Shares", "sharesTraded", "value"]:
            if col in recent.columns:
                shares_col = col
                break

        # Find transaction type column
        type_col = None
        for col in ["transactionType", "transaction", "Transaction"]:
            if col in recent.columns:
                type_col = col
                break

        if type_col is None or shares_col is None:
            return 7.5, {}

        recent["_shares_num"] = pd.to_numeric(recent[shares_col], errors="coerce").fillna(0)

        buys  = recent[recent[type_col].str.contains("Buy|Purchase", case=False, na=False)]
        sells = recent[recent[type_col].str.contains("Sale|Sell", case=False, na=False)]

        buy_shares  = buys["_shares_num"].sum()
        sell_shares = sells["_shares_num"].sum()
        total       = buy_shares + sell_shares

        buy_ratio = buy_shares / total if total > 0 else 0.5  # Neutral if no trades

        # Score: pure buying = 15, pure selling = 0, neutral = 7.5
        insider_score = buy_ratio * cfg["weights"]["insider"]

        summary = {
            "buy_transactions":  len(buys),
            "sell_transactions": len(sells),
            "buy_shares":        int(buy_shares),
            "sell_shares":       int(sell_shares),
            "buy_ratio":         round(buy_ratio, 2),
        }
        return round(insider_score, 2), summary

    except Exception:
        return 7.5, {}


# ─────────────────────────────────────────────
#  COMPOSITE SENTIMENT SCORE
# ─────────────────────────────────────────────

def sentiment_signal(score: float) -> str:
    if score >= 75:   return "🟢 VERY BULLISH"
    elif score >= 60: return "🔵 BULLISH"
    elif score >= 45: return "🟡 NEUTRAL"
    elif score >= 30: return "🟠 BEARISH"
    else:             return "🔴 VERY BEARISH"


def combined_signal(tech_score: float, sent_score: float,
                    tech_w: float, sent_w: float) -> tuple[float, str]:
    """
    Blend Phase 1 technical score (0–75 normalised to 0–100)
    with Phase 6 sentiment score (0–100).
    """
    tech_norm = (tech_score / 75) * 100
    combined  = tech_norm * tech_w + sent_score * sent_w
    combined  = round(min(combined, 100), 1)

    if combined >= 78:   label = "🟢 STRONG BUY"
    elif combined >= 62: label = "🔵 BUY"
    elif combined >= 47: label = "🟡 HOLD"
    elif combined >= 33: label = "🟠 CAUTION"
    else:                label = "🔴 AVOID"
    return combined, label


# ─────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────

SIGNAL_COLORS = {
    "🟢 VERY BULLISH": "#00E676", "🔵 BULLISH":     "#40C4FF",
    "🟡 NEUTRAL":      "#FFD740", "🟠 BEARISH":     "#FF6D00",
    "🔴 VERY BEARISH": "#EF5350",
}

def chart_sentiment_dashboard(df: pd.DataFrame):
    """Stacked bar: news + analyst + earnings + insider scores per ticker."""
    df_sorted = df.sort_values("sentiment_total", ascending=False)

    fig = go.Figure()
    components = [
        ("news_score",     "News Sentiment",    "#00E676"),
        ("analyst_score",  "Analyst Consensus", "#40C4FF"),
        ("earnings_score", "Earnings Surprise", "#FFD740"),
        ("insider_score",  "Insider Activity",  "#FF6D00"),
    ]
    for col, name, color in components:
        fig.add_trace(go.Bar(
            x=df_sorted["ticker"], y=df_sorted[col],
            name=name, marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.1f}}<extra></extra>"
        ))

    # Overlay total score as a line
    fig.add_trace(go.Scatter(
        x=df_sorted["ticker"], y=df_sorted["sentiment_total"],
        name="Total Score", mode="lines+markers",
        line=dict(color="white", width=2, dash="dot"),
        marker=dict(size=7, color="white"),
        yaxis="y2"
    ))

    fig.update_layout(
        title="<b>Sentiment Dashboard</b> — Stacked Signal Scores per Stock",
        barmode="stack",
        xaxis_tickangle=-35,
        yaxis=dict(title="Component Score", range=[0, 105]),
        yaxis2=dict(title="Total (0–100)", overlaying="y", side="right",
                    range=[0, 105], showgrid=False),
        template="plotly_dark", height=580,
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=60, r=80, t=100, b=120),
    )
    path = os.path.join(CHARTS_DIR, "sentiment_dashboard.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"    ✅ Sentiment Dashboard → {path}")
    return path


def chart_combined_ranking(df: pd.DataFrame):
    """Combined Technical + Sentiment score ranking with signal labels."""
    df_sorted = df.sort_values("combined_score", ascending=True)
    colors = [
        SIGNAL_COLORS.get(sig, "#aaa")
        for sig in df_sorted["combined_signal_label"]
    ]

    fig = go.Figure(go.Bar(
        x=df_sorted["combined_score"],
        y=df_sorted["ticker"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}  {s}" for v, s in
              zip(df_sorted["combined_score"], df_sorted["combined_signal_label"])],
        textposition="outside",
        textfont=dict(size=11),
        customdata=df_sorted[["sentiment_total", "tech_score_raw"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>Combined: %{x:.1f}<br>"
            "Sentiment: %{customdata[0]:.1f}<br>"
            "Technical: %{customdata[1]:.1f}<extra></extra>"
        )
    ))

    fig.update_layout(
        title="<b>Combined Signal Ranking</b> — Technical (55%) + Sentiment (45%)",
        xaxis=dict(title="Combined Score (0–100)", range=[0, 115]),
        yaxis_title="",
        template="plotly_dark",
        height=max(400, len(df) * 30 + 160),
        margin=dict(l=90, r=250, t=80, b=60),
    )
    path = os.path.join(CHARTS_DIR, "combined_ranking.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"    ✅ Combined Ranking   → {path}")
    return path


def chart_analyst_heatmap(df: pd.DataFrame):
    """Heatmap: analyst upside % and recommendation per ticker."""
    df_valid = df[df["analyst_upside"].notna()].sort_values("analyst_upside", ascending=False)
    if df_valid.empty:
        return None

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Price Target Upside (%)", "Analyst Recommendation (1=Strong Buy)"],
        horizontal_spacing=0.12
    )

    # Upside bar
    upside_colors = ["#00E676" if u > 0 else "#EF5350" for u in df_valid["analyst_upside"]]
    fig.add_trace(go.Bar(
        x=df_valid["analyst_upside"], y=df_valid["ticker"],
        orientation="h", marker_color=upside_colors,
        text=[f"{u:+.1f}%" for u in df_valid["analyst_upside"]],
        textposition="outside", showlegend=False,
    ), row=1, col=1)

    # Recommendation mean (lower = more bullish)
    df_rec = df[df["analyst_rec_mean"].notna()].sort_values("analyst_upside", ascending=False)
    rec_colors = px.colors.sample_colorscale(
        "RdYlGn_r",
        [(r - 1) / 4 for r in df_rec["analyst_rec_mean"].clip(1, 5)]
    )
    fig.add_trace(go.Bar(
        x=df_rec["analyst_rec_mean"], y=df_rec["ticker"],
        orientation="h", marker_color=rec_colors,
        text=[f"{r:.2f}" for r in df_rec["analyst_rec_mean"]],
        textposition="outside", showlegend=False,
    ), row=1, col=2)

    fig.add_vline(x=3.0, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=2)

    fig.update_layout(
        title="<b>Analyst Signals</b> — Price Targets & Recommendation Ratings",
        template="plotly_dark",
        height=max(400, len(df_valid) * 26 + 160),
        margin=dict(l=90, r=120, t=80, b=60),
    )
    path = os.path.join(CHARTS_DIR, "analyst_heatmap.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"    ✅ Analyst Heatmap    → {path}")
    return path


def chart_news_headlines(all_news: dict):
    """Scatter of recent news items coloured by sentiment score."""
    rows = []
    for ticker, articles in all_news.items():
        for a in articles[:5]:
            rows.append({
                "ticker":    ticker,
                "title":     a["title"],
                "published": a["published"],
                "compound":  a["compound"],
            })

    if not rows:
        return None

    news_df = pd.DataFrame(rows)
    fig = px.scatter(
        news_df, x="published", y="ticker",
        color="compound",
        color_continuous_scale=["#B71C1C", "#E0E0E0", "#1B5E20"],
        range_color=[-1, 1],
        hover_data={"title": True, "compound": ":.2f"},
        title="<b>Recent News Sentiment</b> — Each dot = one headline (colour = sentiment)",
        template="plotly_dark",
        height=max(400, len(all_news) * 28 + 160),
    )
    fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=0.5, color="white")))
    fig.update_layout(
        coloraxis_colorbar=dict(title="Sentiment"),
        margin=dict(l=90, r=80, t=80, b=60),
    )
    path = os.path.join(CHARTS_DIR, "news_sentiment_scatter.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"    ✅ News Scatter       → {path}")
    return path


# ─────────────────────────────────────────────
#  INTEGRATION: load screener scores
# ─────────────────────────────────────────────

def load_screener_scores() -> dict:
    """Load Phase 1 technical scores from stock_screener_results.csv."""
    csv_path = os.path.join(OUTPUT_DIR, "stock_screener_results.csv")
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    return dict(zip(df["ticker"], df["total_score"]))


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run_sentiment_analysis():
    cfg = SENTIMENT_CONFIG
    # Derive ticker list from the live watchlist (respects WATCHLIST_MODE),
    # filtering out pure ETFs. Falls back to SENTIMENT_CONFIG["tickers"] if needed.
    _etfs = {"SPY", "QQQ", "VTI", "SCHD", "IWM", "DIA"}
    tickers = [
        t for sector_tickers in get_watchlist().values()
        for t in sector_tickers if t not in _etfs
    ] or cfg["tickers"]

    print(f"\n{'='*65}")
    print(f"  INVESTMENT INTELLIGENCE — Sentiment & News Layer (Phase 6)")
    print(f"  Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Scanning {len(tickers)} stocks across 4 signal sources")
    print(f"  Signals: News NLP | Analyst | Earnings | Insider Activity")
    print(f"{'='*65}\n")

    screener_scores = load_screener_scores()
    results = []
    all_news_articles = {}

    for i, ticker in enumerate(tickers):
        sys.stdout.write(f"\r  [{i+1}/{len(tickers)}] {ticker:<8}...")
        sys.stdout.flush()

        # Run all 4 signals
        news_score,     news_articles = score_news(ticker, cfg)
        analyst_score,  analyst_info  = score_analyst(ticker, cfg)
        earnings_score, earnings_info = score_earnings(ticker, cfg)
        insider_score,  insider_info  = score_insider(ticker, cfg)

        sentiment_total = round(news_score + analyst_score + earnings_score + insider_score, 1)
        sent_signal     = sentiment_signal(sentiment_total)

        # Combined with Phase 1 technical score
        tech_raw = screener_scores.get(ticker, 45.0)  # Default neutral if not run yet
        comb_score, comb_label = combined_signal(
            tech_raw, sentiment_total,
            cfg["combined_tech_weight"], cfg["combined_sentiment_weight"]
        )

        if news_articles:
            all_news_articles[ticker] = news_articles

        results.append({
            "ticker":               ticker,
            "news_score":           news_score,
            "analyst_score":        analyst_score,
            "earnings_score":       earnings_score,
            "insider_score":        insider_score,
            "sentiment_total":      sentiment_total,
            "sentiment_signal":     sent_signal,
            "tech_score_raw":       tech_raw,
            "combined_score":       comb_score,
            "combined_signal_label":comb_label,
            # Analyst details for heatmap
            "analyst_upside":       analyst_info.get("upside_pct"),
            "analyst_rec_mean":     analyst_info.get("rec_mean"),
            "analyst_recommendation": analyst_info.get("recommendation", ""),
            "n_analysts":           analyst_info.get("n_analysts", 0),
            # Top headline
            "top_headline":         news_articles[0]["title"] if news_articles else "",
            "top_headline_score":   news_articles[0]["compound"] if news_articles else 0,
            # Insider
            "insider_buy_ratio":    insider_info.get("buy_ratio", 0.5),
        })

        time.sleep(cfg["sleep_between_tickers"])

    print(f"\n  ✅ Sentiment scored for {len(results)} stocks.\n")

    # ── Sort & print ──────────────────────────
    results.sort(key=lambda x: -x["combined_score"])
    df_out = pd.DataFrame(results)

    print(f"{'='*85}")
    print(f"  COMBINED RANKING — Technical (55%) + Sentiment (45%)")
    print(f"{'='*85}")
    print(f"  {'#':<4} {'Ticker':<8} {'Combined':>9} {'Sentiment':>10} "
          f"{'Technical':>10} {'Signal':<22} {'Analyst'}")
    print(f"  {'─'*82}")

    for rank, row in enumerate(results, 1):
        upside_str = (f"{row['analyst_upside']:+.0f}% upside"
                      if row["analyst_upside"] is not None else "—")
        print(
            f"  {rank:<4} {row['ticker']:<8} "
            f"{row['combined_score']:>8.1f}  "
            f"{row['sentiment_total']:>9.1f}  "
            f"{row['tech_score_raw']:>9.1f}  "
            f"{row['combined_signal_label']:<22} {upside_str}"
        )

    # ── Sentiment breakdown for top 10 ────────
    print(f"\n  SENTIMENT BREAKDOWN — Top 10")
    print(f"  {'─'*70}")
    for row in results[:10]:
        print(f"\n  ┌── {row['ticker']}  |  Sentiment: {row['sentiment_total']:.0f}/100  "
              f"{row['sentiment_signal']}")
        print(f"  │   News:       {row['news_score']:>5.1f}/35  "
              f"  Analyst: {row['analyst_score']:>5.1f}/30  "
              f"  Earnings: {row['earnings_score']:>5.1f}/20  "
              f"  Insider: {row['insider_score']:>4.1f}/15")
        if row["analyst_recommendation"]:
            up_str = f"  |  {row['analyst_upside']:+.1f}% upside" if row["analyst_upside"] else ""
            print(f"  │   Analyst consensus: {row['analyst_recommendation'].upper()}{up_str}")
        if row["top_headline"]:
            cmp = row["top_headline_score"]
            sym = "📈" if cmp > 0.2 else ("📉" if cmp < -0.2 else "➡️")
            print(f"  │   {sym} \"{row['top_headline'][:75]}\"")
        print(f"  └{'─'*68}")

    # ── Save CSV ──────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "sentiment_results.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\n  💾 Results → {csv_path}")

    # ── Charts ────────────────────────────────
    print(f"\n  📈 Generating charts...")
    chart_sentiment_dashboard(df_out)
    chart_combined_ranking(df_out)
    chart_analyst_heatmap(df_out)
    if all_news_articles:
        chart_news_headlines(all_news_articles)

    # ── Merge back into screener CSV if exists ─
    sc_path = os.path.join(OUTPUT_DIR, "stock_screener_results.csv")
    if os.path.exists(sc_path):
        sc_df = pd.read_csv(sc_path)
        sent_cols = df_out[["ticker","sentiment_total","sentiment_signal",
                             "combined_score","combined_signal_label"]]
        merged = sc_df.merge(sent_cols, on="ticker", how="left")
        merged.to_csv(sc_path, index=False)
        print(f"  📊 Merged sentiment into screener CSV → {sc_path}")

    print(f"\n{'='*65}")
    print(f"  🏁 SENTIMENT ANALYSIS COMPLETE")
    print(f"  Run stock_screener.py first to get technical scores,")
    print(f"  then run this to get the full combined ranking.")
    print(f"\n  INTEGRATION: Add to daily_monitor.py — call run_sentiment_analysis()")
    print(f"  in run_monitor() to include sentiment in your morning report.")
    print(f"{'='*65}\n")

    return df_out


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_sentiment_analysis()
