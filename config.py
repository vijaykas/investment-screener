"""
============================================================
  INVESTMENT INTELLIGENCE — MASTER CONFIGURATION
  Edit THIS file to customise the entire stack.
  All phases read their settings from here.
============================================================

SECTIONS:
  0. Directory Paths   — where files are stored
  1. API Keys          — paste your free keys here
  2. Watchlist         — stocks to monitor (static or live S&P 500)
  3. Your Positions    — actual holdings for P&L tracking (Phase 4)
  4. Phase 1  — Stock Screener
  5. Phase 2  — Portfolio Optimizer
  6. Phase 3  — Backtester
  7. Phase 4  — Daily Monitor
  7b. Phase 4b — ETF Screener & Monitor
  7c. Phase 4c — Crypto Screener
  8. Phase 7  — News & Sentiment Engine
  9. Phase 8  — Top 20 High-Yield Predictions
  10. Phase 5  — ML Predictor
  11. Phase 6  — Old Sentiment Analyzer

FREE API KEYS (all optional — project works fine without them):
  Finnhub       https://finnhub.io              (60 req/min free tier)
  FRED          https://fred.stlouisfed.org/docs/api/api_key.html
  Alpha Vantage https://www.alphavantage.co/support/#api-key

  Set keys here OR as environment variables (safer for shared machines):
    export FINNHUB_KEY=your_key
    export FRED_KEY=your_key
    export ALPHA_VANTAGE_KEY=your_key
============================================================
"""

import os
import sys
from datetime import datetime

# ─────────────────────────────────────────────────────────────
#  0. DIRECTORY PATHS
#     All phase scripts import these instead of computing paths.
#     Change DATA_DIR if you want outputs stored elsewhere.
# ─────────────────────────────────────────────────────────────

# Root of the project (wherever config.py lives)
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

# Organised subdirectories
DATA_DIR    = os.path.join(BASE_DIR, "data")          # CSVs, JSON, cache files
CHARTS_DIR  = os.path.join(BASE_DIR, "charts")        # Interactive HTML charts
MODELS_DIR  = os.path.join(BASE_DIR, "models")        # Trained ML model .pkl files
REPORTS_DIR = os.path.join(BASE_DIR, "daily_reports") # Daily morning HTML reports
DOCS_DIR    = os.path.join(BASE_DIR, "docs")          # Reference docs

# Create all directories on first import (safe no-op if they already exist)
for _d in (DATA_DIR, CHARTS_DIR, MODELS_DIR, REPORTS_DIR, DOCS_DIR):
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  1. API KEYS
#     Paste your free keys below, OR leave "" to use env vars.
# ─────────────────────────────────────────────────────────────

API_KEYS = {
    # FINNHUB_KEY=d7e7jhpr01qkuebj88ogd7e7jhpr01qkuebj88p0
    # FRED_KEY=a7776e7caa34b0e4de03e935c6867de2
    # ALPHA_VANTAGE_KEY=KELWAWB5YY4VRDEW
    # FMP_KEY=WEvF1WI8UNXwmv6GOKydXTAKimjqgmdC

    # ── Finnhub (news, insider transactions, earnings) ─────────
    # Free tier: 60 req/min | Get key → https://finnhub.io
    "finnhub": os.getenv("FINNHUB_KEY", ""),

    # ── FRED — Federal Reserve Economic Data ───────────────────
    # Free, unlimited | Get key → https://fred.stlouisfed.org/docs/api/api_key.html
    "fred": os.getenv("FRED_KEY", ""),

    # ── Alpha Vantage (alternative price/fundamentals source) ──
    # Free tier: 25 req/day | Get key → https://www.alphavantage.co/support/#api-key
    "alpha_vantage": os.getenv("ALPHA_VANTAGE_KEY", ""),

    # ── Financial Modeling Prep — ETF list + market cap data ───
    # Free tier: 250 req/day | Get key → https://financialmodelingprep.com/developer/docs/
    # Used by ETF_WATCHLIST_MODE = "fmp" to fetch a live top-ETF universe
    "fmp": os.getenv("FMP_KEY", ""),
}


# ─────────────────────────────────────────────────────────────
#  2. WATCHLIST
#     Used by Phase 1 (Screener), Phase 4 (Monitor), Phase 6 (Sentiment).
#
#  WATCHLIST_MODE options:
#    "static" — use the WATCHLIST dict below (default, fast, predictable)
#    "sp500"  — live S&P 500 constituents fetched from Wikipedia on each run
#               (~500 stocks; screener will take longer but covers the full index)
# ─────────────────────────────────────────────────────────────

WATCHLIST_MODE = "sp500"   # Change to "sp500" for live S&P 500 coverage

# Static watchlist — edit freely; used when WATCHLIST_MODE = "static"
WATCHLIST = {
    "Tech":       ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD", "AVGO", "QCOM", "CRM", "NOW"],
    "AI/Growth":  ["PLTR", "CRWD", "SNOW", "NET", "DDOG", "TSLA", "MSTR", "ARM"],
    "Finance":    ["JPM", "V", "MA", "GS", "BAC", "BRK-B"],
    "Healthcare": ["LLY", "UNH", "ABBV", "JNJ", "MRK", "PFE"],
    "Consumer":   ["COST", "HD", "MCD", "AMZN", "NKE", "SBUX", "WMT"],
    "Energy":     ["XOM", "CVX", "COP"],
    "Industrial": ["CAT", "DE", "BA", "HON"],
    "Dividend":   ["PG", "KO", "T", "O", "VZ"],
    "ETF":        ["SPY", "QQQ", "VTI", "SCHD"],
}


def get_watchlist() -> dict[str, list[str]]:
    """
    Return the active watchlist based on WATCHLIST_MODE.

    "static" → returns WATCHLIST dict above (default).
    "sp500"  → fetches the current S&P 500 list from Wikipedia,
               grouped by GICS sector. Falls back to static on error.

    Call this in phase scripts instead of referencing WATCHLIST directly:
        from config import get_watchlist
        watchlist = get_watchlist()
    """
    if WATCHLIST_MODE == "static":
        return WATCHLIST

    if WATCHLIST_MODE == "sp500":
        try:
            import pandas as pd
            print("  🌐 Fetching live S&P 500 constituents from Wikipedia...")
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url, header=0)
            sp500_df = tables[0]

            # Normalise column names
            sp500_df.columns = [c.strip() for c in sp500_df.columns]
            ticker_col = next(c for c in sp500_df.columns if "Symbol" in c or "Ticker" in c)
            sector_col = next(c for c in sp500_df.columns if "Sector" in c or "GICS" in c)

            # Clean tickers (Wikipedia sometimes uses dots, e.g. BRK.B → BRK-B)
            sp500_df[ticker_col] = sp500_df[ticker_col].str.replace(".", "-", regex=False)

            # Group by sector
            grouped: dict[str, list[str]] = {}
            for sector, grp in sp500_df.groupby(sector_col):
                grouped[sector] = grp[ticker_col].tolist()

            print(f"  ✅ Loaded {sum(len(v) for v in grouped.values())} S&P 500 tickers "
                  f"across {len(grouped)} sectors.")
            return grouped

        except Exception as e:
            print(f"  ⚠️  Could not fetch S&P 500 ({e}). Falling back to static watchlist.")
            return WATCHLIST

    # Unknown mode — fall back to static
    print(f"  ⚠️  Unknown WATCHLIST_MODE '{WATCHLIST_MODE}'. Using static watchlist.")
    return WATCHLIST


# ─────────────────────────────────────────────────────────────
#  3. YOUR PORTFOLIO POSITIONS  (Phase 4 — Daily Monitor)
#     Fill in your actual holdings for P&L and risk tracking.
#     Format: "TICKER": {"shares": N, "avg_cost": price_per_share}
# ─────────────────────────────────────────────────────────────

YOUR_POSITIONS = {
    # Uncomment and edit with your real holdings:
    # "AAPL":  {"shares": 10,  "avg_cost": 175.00},
    # "NVDA":  {"shares": 5,   "avg_cost": 480.00},
    # "MSFT":  {"shares": 8,   "avg_cost": 350.00},
    # "PLTR":  {"shares": 50,  "avg_cost": 22.00},
    # "GOOGL": {"shares": 3,   "avg_cost": 140.00},
}


# ─────────────────────────────────────────────────────────────
#  4. PHASE 1 — STOCK SCREENER
# ─────────────────────────────────────────────────────────────

SCREENER_CONFIG = {
    "min_price":         5,        # Ignore penny stocks below this price ($)
    "min_avg_volume":    500_000,  # Minimum avg daily volume (liquidity filter)
    "max_pe":            120,      # Skip extremely overvalued (None = no limit)
    "rsi_oversold":      35,       # Below this = oversold / potential buy zone
    "rsi_overbought":    72,       # Above this = overbought / caution zone
    "min_market_cap_B":  2,        # Minimum market cap in billions ($)
    "top_n_chart":       5,        # Generate detailed charts for top N stocks
}

# Scoring weights — must sum to 100
SCREENER_WEIGHTS = {
    "technical":   60,   # RSI, MACD, MA crossovers, Bollinger, Volume
    "fundamental": 40,   # PE ratio, revenue growth, margins, debt
}


# ─────────────────────────────────────────────────────────────
#  5. PHASE 2 — PORTFOLIO OPTIMIZER
# ─────────────────────────────────────────────────────────────

PORTFOLIO_CONFIG = {
    # Set True to auto-pull top stocks from Phase 1 screener output
    "auto_read_screener": True,
    "screener_top_n":     15,      # How many top screener picks to include

    # Manual tickers — used when auto_read_screener = False
    "manual_tickers": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META",
        "AMZN", "CRM", "NOW", "PLTR",
        "JPM", "V",
        "LLY", "UNH",
        "COST", "PG",
    ],

    "history_years":  2,        # Years of price history for covariance estimation
    "risk_free_rate": 0.045,    # Current ~10yr Treasury yield (annualised)
    "n_simulations":  15_000,   # Monte Carlo portfolio iterations
    "min_weight":     0.01,     # Each stock must be ≥ 1% (no micro-slivers)
    "max_weight":     0.40,     # No single stock > 40% (concentration limit)
    "trading_days":   252,      # Standard annualisation factor
    "top_n_chart":    3,        # Plot weight charts for top N optimal portfolios
}


# ─────────────────────────────────────────────────────────────
#  6. PHASE 3 — BACKTESTER
# ─────────────────────────────────────────────────────────────

BACKTEST_CONFIG = {
    # Date window
    "start_date": "2022-01-01",
    "end_date":   datetime.today().strftime("%Y-%m-%d"),

    # Tickers — auto-read from screener or use manual list below
    "auto_read_screener": False,
    "screener_top_n":     12,
    "tickers": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META",
        "AMZN", "JPM", "LLY", "COST", "V",
        "PLTR", "CRM",
    ],

    # Execution model
    "commission":      0.001,    # 0.10% per trade (realistic broker cost)
    "slippage":        0.001,    # 0.10% slippage on fill
    "initial_capital": 100_000,  # Starting portfolio value ($)
    "position_size":   0.95,     # Fraction of capital deployed per trade
    "benchmark":       "SPY",    # Buy-and-hold benchmark ticker

    # Strategy parameters
    "rsi_period":  14,
    "rsi_buy":     35,    # RSI below this → buy signal
    "rsi_sell":    70,    # RSI above this → sell signal
    "macd_fast":   12,
    "macd_slow":   26,
    "macd_signal": 9,
    "sma_fast":    50,    # Golden Cross fast MA
    "sma_slow":    200,   # Golden Cross slow MA
    "bb_period":   20,
    "bb_std":      2.0,

    # Metrics
    "risk_free_rate": 0.045,   # For Sharpe ratio calculation
    "trading_days":   252,
}


# ─────────────────────────────────────────────────────────────
#  7. PHASE 4 — DAILY MONITOR
# ─────────────────────────────────────────────────────────────

MONITOR_CONFIG = {
    # Watchlist is shared from section 2 above
    "watchlist": WATCHLIST,

    # Your positions imported from section 3 above
    "your_positions": YOUR_POSITIONS,

    # Alert thresholds
    "big_move_pct":         3.0,   # Flag if stock moves > ±3% overnight
    "volume_surge_ratio":   2.0,   # Flag if volume > 2× its 20-day average
    "rsi_oversold":         35,
    "rsi_overbought":       72,
    "drawdown_alert_pct":   10.0,  # Alert if a position is down > 10% from cost

    # Signal scoring thresholds (mirrors screener)
    "strong_buy_threshold": 72,
    "buy_threshold":        58,
    "avoid_threshold":      35,

    # Output files
    "history_file": "signal_history.json",
    "reports_dir":  "daily_reports",
}


# ─────────────────────────────────────────────────────────────
#  7b. PHASE 4b — ETF SCREENER & MONITOR
#      All major ETF categories tracked in the daily report.
#
#  ETF_WATCHLIST_MODE options:
#    "static" — use the hand-curated ETF_WATCHLIST below (default, 42 ETFs)
#    "fmp"    — live universe fetched from Financial Modeling Prep API.
#               Pulls the top US-listed ETFs by AUM, auto-categorises them,
#               and uses ETF_FETCH_CONFIG to control size + category caps.
#               Requires API_KEYS["fmp"] — free key at:
#               https://financialmodelingprep.com/developer/docs/
#
#  Categories (static or auto-detected in fmp mode):
#    Broad Market  — SPY, QQQ, VTI, IWM, DIA, VUG, VTV
#    Sector        — All 11 GICS SPDR sector ETFs (XL*)
#    International — Developed + emerging market funds
#    Fixed Income  — Government, aggregate, high-yield bond ETFs
#    Commodity     — Gold, silver, oil, broad commodity ETFs
#    Real Estate   — REIT and real-estate-specific ETFs
#    Thematic      — (fmp mode only) ARK, clean energy, robotics, etc.
#    Dividend      — (fmp mode only) SCHD, VYM, NOBL, HDV
#    Factor        — (fmp mode only) MTUM, VLUE, QUAL, USMV
#    Leveraged     — (fmp mode only) TQQQ, SOXL, UPRO (if AUM qualifies)
# ─────────────────────────────────────────────────────────────

ETF_WATCHLIST_MODE = "static"   # Change to "fmp" once your FMP key is set

ETF_WATCHLIST = {
    "Broad Market": [
        "SPY",   # S&P 500 (State Street)
        "QQQ",   # Nasdaq-100 (Invesco)
        "VTI",   # Total US Market (Vanguard)
        "IWM",   # Russell 2000 Small-Cap (iShares)
        "DIA",   # Dow Jones 30 (State Street)
        "VUG",   # US Large-Cap Growth (Vanguard)
        "VTV",   # US Large-Cap Value (Vanguard)
    ],
    "Sector": [
        "XLK",   # Technology
        "XLF",   # Financials
        "XLE",   # Energy
        "XLV",   # Health Care
        "XLY",   # Consumer Discretionary
        "XLP",   # Consumer Staples
        "XLI",   # Industrials
        "XLB",   # Materials
        "XLU",   # Utilities
        "XLRE",  # Real Estate
        "XLC",   # Communication Services
    ],
    "International": [
        "VEA",   # Developed Markets ex-US (Vanguard)
        "VWO",   # Emerging Markets (Vanguard)
        "EFA",   # MSCI EAFE (iShares)
        "EEM",   # MSCI Emerging Markets (iShares)
        "VGK",   # Europe (Vanguard)
        "EWJ",   # Japan (iShares)
        "IEFA",  # Core MSCI EAFE (iShares)
    ],
    "Fixed Income": [
        "BND",   # Total US Bond Market (Vanguard)
        "AGG",   # US Aggregate Bond (iShares)
        "TLT",   # 20+ Year Treasury (iShares)
        "IEF",   # 7-10 Year Treasury (iShares)
        "SHY",   # 1-3 Year Treasury (iShares)
        "HYG",   # High Yield Corporate (iShares)
        "LQD",   # Investment Grade Corporate (iShares)
        "BNDX",  # Total International Bond (Vanguard)
    ],
    "Commodity": [
        "GLD",   # Gold (SPDR)
        "IAU",   # Gold (iShares, lower cost)
        "SLV",   # Silver (iShares)
        "USO",   # US Oil Fund
        "DJP",   # Bloomberg Commodity Index
        "PDBC",  # Diversified Commodity Strategy (Invesco)
    ],
    "Real Estate": [
        "VNQ",   # US Real Estate (Vanguard)
        "SCHH",  # US REIT (Schwab)
        "IYR",   # US Real Estate (iShares)
    ],
}


ETF_MONITOR_CONFIG = {
    # Benchmark for relative-return calculations
    "benchmark": "SPY",

    # Score thresholds (ETF scoring is out of 100)
    "strong_buy_threshold": 75,
    "buy_threshold":        58,
    "hold_threshold":       42,
    "avoid_threshold":      28,

    # Scoring component weights (must sum to 100)
    "weights": {
        "technical":  50,   # RSI, MACD, MAs, Bollinger, ADX, Volume
        "momentum":   30,   # 1m/3m/6m returns vs SPY benchmark
        "quality":    20,   # Expense ratio + AUM liquidity score
    },

    # Momentum lookback periods (trading days)
    "momentum_periods": {
        "1m": 21,
        "3m": 63,
        "6m": 126,
        "12m": 252,
    },

    # Quality scoring thresholds
    "expense_ratio_tiers": [0.10, 0.20, 0.50, 1.00],  # breakpoints in %
    "aum_tiers_B":         [1.0, 5.0, 10.0, 50.0],    # breakpoints in $B

    # Alerts
    "big_move_pct":       2.0,   # Flag if ETF moves > ±2% in a day
    "volume_surge_ratio": 2.0,

    # Output file
    "output_file": "etf_screener_results.csv",
}


# ── FMP fetch settings (used when ETF_WATCHLIST_MODE = "fmp") ──
ETF_FETCH_CONFIG = {
    # How many ETFs to return per category (caps each bucket)
    "top_n_per_category": 12,

    # Hard cap on the total ETF universe (keeps screener runtime reasonable)
    # At ~2s per ETF this is ~200s for 100 ETFs — adjust to taste
    "max_total": 120,

    # Only include ETFs with AUM above this threshold ($B)
    # Keeps out illiquid funds with poor bid/ask spreads
    "min_aum_B": 0.5,

    # US exchanges to include
    "exchanges": {"NASDAQ", "NYSE", "AMEX", "BATS", "NYSE ARCA", "NYSEArca"},

    # FMP batch size — their /quote endpoint accepts up to 50 symbols at once
    "batch_size": 50,

    # Max API requests to use for AUM lookup (free tier = 250/day)
    # Each batch call costs 1 request; 6 batches → 300 ETFs checked
    "max_aum_batches": 6,
}

# ── Keyword map for auto-categorisation of FMP ETF names ────
# Each category is tried in order; first match wins.
# Keywords are checked against the lowercased ETF long name.
_ETF_CATEGORY_KEYWORDS = {
    "Real Estate": [
        "real estate investment", "reit", "real estate etf",
        "mortgage reit", "property",
    ],
    "Fixed Income": [
        "bond", "treasury", "aggregate bond", "corporate bond",
        "high yield", "credit", "fixed income", "investment grade",
        "government bond", "municipal", "tips", "inflation-protected",
        "floating rate", "preferred", "ultrashort",
    ],
    "Commodity": [
        "gold", "silver", "oil", "commodity", "crude",
        "natural gas", "agriculture", "precious metal",
        "broad commodity", "energy commodity",
    ],
    "International": [
        "international", "foreign", "developed market", "emerging market",
        "europe", "asia pacific", "japan", "china", "india",
        "global ex-us", "ex-us", "eafe", "msci world", "world equity",
        "latin america", "africa", "korea", "taiwan",
    ],
    "Sector": [
        "technology sector", "financial sector", "energy sector",
        "health care sector", "consumer discretionary", "consumer staples",
        "industrials sector", "materials sector", "utilities sector",
        "communication services", "information technology select",
        "select sector", "sector spdr",
    ],
    "Leveraged": [
        "2x", "3x", "ultra", "leveraged", "daily bull", "daily bear",
        "inverse", "short", "bear 3x", "bull 3x",
    ],
    "Dividend": [
        "dividend", "income equity", "high dividend", "dividend aristocrat",
        "dividend growth", "dividend yield", "equity income",
    ],
    "Factor": [
        "momentum factor", "value factor", "quality factor",
        "minimum volatility", "min vol", "multifactor",
        "low volatility", "size factor", "growth factor",
    ],
    "Thematic": [
        "innovation", "clean energy", "renewable", "robotics", "artificial intelligence",
        "cybersecurity", "genomics", "space", "cloud", "fintech",
        "cannabis", "esg", "sustainable", "disruptive", "future", "ark",
        "semiconductor", "biotech", "healthcare innovation",
    ],
    "Broad Market": [
        "s&p 500", "total market", "total stock market", "total us market",
        "nasdaq-100", "nasdaq 100", "dow jones industrial", "russell 1000",
        "russell 2000", "large cap blend", "mid cap blend", "small cap blend",
        "total world", "all cap", "extended market",
    ],
}


def get_etf_watchlist() -> dict[str, list[str]]:
    """
    Return the active ETF watchlist based on ETF_WATCHLIST_MODE.

    "static" → returns the hard-coded ETF_WATCHLIST above (default, 42 ETFs).
    "fmp"    → fetches the top US-listed ETFs by AUM from Financial Modeling
               Prep, auto-categorises them by name keywords, caps each
               category at ETF_FETCH_CONFIG["top_n_per_category"], and falls
               back to the static list on any error.

    Usage in etf_screener.py:
        from config import get_etf_watchlist
        watchlist = get_etf_watchlist()
    """
    if ETF_WATCHLIST_MODE == "static":
        return ETF_WATCHLIST

    if ETF_WATCHLIST_MODE == "fmp":
        fmp_key = API_KEYS.get("fmp", "")
        if not fmp_key:
            print("  ⚠️  ETF_WATCHLIST_MODE='fmp' but API_KEYS['fmp'] is empty.")
            print("       Get a free key at https://financialmodelingprep.com/developer/docs/")
            print("       Then set it in config.py (Section 1) or as env var FMP_KEY.")
            print("       Falling back to static ETF watchlist.")
            return ETF_WATCHLIST

        try:
            import requests
            cfg = ETF_FETCH_CONFIG
            base = "https://financialmodelingprep.com/api/v3"

            # ── Step 1: Fetch full ETF list ───────────────────────
            print("  🌐 FMP: fetching ETF universe...")
            r = requests.get(f"{base}/etf/list?apikey={fmp_key}", timeout=15)
            r.raise_for_status()
            raw_list = r.json()

            # Keep only US-exchange ETFs with a clean symbol
            valid_exchanges = cfg["exchanges"]
            candidates = [
                e for e in raw_list
                if (isinstance(e, dict)
                    and e.get("exchangeShortName", "") in valid_exchanges
                    and e.get("symbol", "").isalpha()          # letters only — skip BRK-B style
                    and len(e.get("symbol", "")) <= 5)
            ]
            symbols = [e["symbol"] for e in candidates]
            name_map = {e["symbol"]: e.get("name", e["symbol"]) for e in candidates}
            print(f"       {len(symbols)} candidate ETFs on US exchanges.")

            # ── Step 2: Batch AUM (market cap proxy) lookup ───────
            print("  🌐 FMP: fetching AUM data in batches...")
            aum_map: dict[str, float] = {}
            bs = cfg["batch_size"]
            max_batches = cfg["max_aum_batches"]
            for i in range(0, min(len(symbols), bs * max_batches), bs):
                batch = symbols[i : i + bs]
                r2 = requests.get(
                    f"{base}/quote/{','.join(batch)}?apikey={fmp_key}", timeout=15
                )
                r2.raise_for_status()
                for item in r2.json():
                    sym = item.get("symbol", "")
                    aum_map[sym] = float(item.get("marketCap") or 0)

            # ── Step 3: Filter by min AUM and sort ────────────────
            min_aum = cfg["min_aum_B"] * 1e9
            qualified = [
                s for s in symbols
                if aum_map.get(s, 0) >= min_aum
            ]
            qualified.sort(key=lambda s: -aum_map.get(s, 0))
            qualified = qualified[: cfg["max_total"]]
            print(f"       {len(qualified)} ETFs pass AUM filter (≥${cfg['min_aum_B']}B).")

            # ── Step 4: Categorise by name keywords ───────────────
            grouped: dict[str, list[str]] = {cat: [] for cat in _ETF_CATEGORY_KEYWORDS}
            grouped["Other"] = []
            cap_per_cat = cfg["top_n_per_category"]

            for sym in qualified:
                name_lower = name_map.get(sym, "").lower()
                matched = False
                for cat, keywords in _ETF_CATEGORY_KEYWORDS.items():
                    if any(kw in name_lower for kw in keywords):
                        if len(grouped[cat]) < cap_per_cat:
                            grouped[cat].append(sym)
                        matched = True
                        break
                if not matched and len(grouped["Other"]) < cap_per_cat:
                    grouped["Other"].append(sym)

            # Drop empty buckets
            result = {cat: tickers for cat, tickers in grouped.items() if tickers}
            total = sum(len(v) for v in result.values())
            print(f"  ✅ FMP watchlist: {total} ETFs across {len(result)} categories.")
            for cat, tickers in result.items():
                print(f"       {cat:<20} {len(tickers):>3} ETFs  "
                      f"(top: {', '.join(tickers[:4])}{'…' if len(tickers) > 4 else ''})")
            return result

        except Exception as e:
            print(f"  ⚠️  FMP fetch failed ({e}). Falling back to static ETF watchlist.")
            return ETF_WATCHLIST

    print(f"  ⚠️  Unknown ETF_WATCHLIST_MODE '{ETF_WATCHLIST_MODE}'. Using static watchlist.")
    return ETF_WATCHLIST


# ─────────────────────────────────────────────────────────────
#  7c. PHASE 4c — CRYPTO SCREENER
#
#  PHILOSOPHY:
#    Only quality, utility-bearing assets are included.
#    Meme coins, joke coins, and tokens without clear network utility
#    are excluded by design. The hard exclusion list in crypto_screener.py
#    acts as a final safety net.
#
#  SCORING (0–100 total):
#    Technical   (40 pts) — RSI, MACD, MA crossovers, Bollinger, ADX, Volume
#    Momentum    (25 pts) — 7d/30d/90d returns vs BTC benchmark
#    Quality     (20 pts) — Market cap tier + Network type + Activity ratio
#    Sentiment   (15 pts) — From Phase 7 news, or signal-derived fallback
#
#  SIGNAL LABELS:
#    🟢 STRONG BUY (≥78) | 🔵 BUY (≥62) | 🟡 HOLD (≥46) | 🟠 CAUTION (≥30) | 🔴 AVOID
#
#  CATEGORIES (why these matter):
#    Layer 1           — Base chain infrastructure; the "operating systems" of crypto
#    Layer 2           — Scaling solutions built on top of L1s (ETH mainly)
#    DeFi              — Decentralised finance protocols with real revenue/TVL
#    AI / Infrastructure — Decentralised compute, AI/ML networks, data infrastructure
#    Payments / Interop — Cross-chain bridges, payment rails, interoperability
# ─────────────────────────────────────────────────────────────

CRYPTO_WATCHLIST = {
    "Layer 1": [
        "BTC-USD",   # Bitcoin — digital gold, store of value, network security
        "ETH-USD",   # Ethereum — smart contract platform, staking yield
        "SOL-USD",   # Solana — high-throughput L1, strong dev ecosystem
        "ADA-USD",   # Cardano — peer-reviewed academic approach, PoS
        "AVAX-USD",  # Avalanche — subnet architecture, institutional focus
        "DOT-USD",   # Polkadot — parachain interoperability hub
        "NEAR-USD",  # NEAR Protocol — sharding, UX-focused L1
        "APT-USD",   # Aptos — Move language, ex-Meta Diem engineers
        "SUI-USD",   # Sui — parallel transaction execution, gaming focus
    ],
    "Layer 2": [
        "MATIC-USD",  # Polygon — Ethereum scaling, enterprise adoption
        "OP-USD",     # Optimism — optimistic rollup, Superchain ecosystem
        "ARB-USD",    # Arbitrum — largest L2 by TVL, DeFi hub
    ],
    "DeFi": [
        "LINK-USD",  # Chainlink — oracle infrastructure; backbone of DeFi
        "UNI-USD",   # Uniswap — leading DEX by volume; protocol revenue
        "AAVE-USD",  # Aave — lending protocol; real yield to stakers
        "MKR-USD",   # Maker — DAI stablecoin engine; revenue-generating
        "LDO-USD",   # Lido — liquid staking derivative leader
        "ATOM-USD",  # Cosmos — IBC interoperability hub + staking yield
    ],
    "AI / Infrastructure": [
        "RNDR-USD",  # Render Network — decentralised GPU compute for AI/VFX
        "FET-USD",   # Fetch.ai (ASI) — autonomous AI agents network
        "GRT-USD",   # The Graph — decentralised blockchain indexing protocol
        "INJ-USD",   # Injective — DeFi L1 optimised for financial apps
    ],
    "Payments / Interop": [
        "XRP-USD",   # Ripple — cross-border payment rails; banking partnerships
        "XLM-USD",   # Stellar — low-cost remittance network
        "BNB-USD",   # Binance Coin — exchange utility + BSC ecosystem
        "ALGO-USD",  # Algorand — pure PoS, CBDCs & government adoption
    ],
}

CRYPTO_CONFIG = {
    # Benchmark for relative momentum (measuring outperformance vs BTC)
    "benchmark": "BTC-USD",

    # Signal score thresholds (out of 100)
    "signal_thresholds": {
        "strong_buy": 78,
        "buy":        62,
        "hold":       46,
        "caution":    30,
        # below caution → AVOID
    },

    # Scoring component weights (sum = 100)
    "weights": {
        "technical":  40,   # RSI, MACD, MAs, Bollinger, ADX
        "momentum":   25,   # 7d/30d/90d returns vs BTC
        "quality":    20,   # Market cap tier + network type + activity
        "sentiment":  15,   # News sentiment from Phase 7
    },

    # Momentum lookback periods (trading days — crypto trades 24/7)
    "momentum_periods": {
        "7d":   7,
        "30d":  30,
        "90d":  90,
        "180d": 180,
    },

    # RSI thresholds for crypto (wider bands than stocks due to volatility)
    "rsi": {
        "strong_buy":   30,   # Deeply oversold → high recovery potential
        "buy":          40,
        "neutral_low":  50,
        "neutral_high": 60,
        "caution":      75,   # Approaching overbought
        # above caution → overbought
    },

    # Volume surge threshold (ratio vs 20-day average)
    "volume_surge_ratio": 2.0,

    # Market cap tiers (USD billions)
    "mc_tiers_B": [0.1, 1.0, 10.0, 50.0, 200.0],

    # Output file
    "output_file":       "crypto_screener_results.csv",
    "cycle_file":        "crypto_cycle.json",
}


# ─────────────────────────────────────────────────────────────
#  8. PHASE 7 — NEWS & SENTIMENT ENGINE
#
#  DATA SOURCES (tried in api_preference order):
#    "alpha_vantage" → NEWS_SENTIMENT endpoint (500 req/day free)
#                      Returns per-article ticker sentiment scores (-1..+1)
#    "finnhub"       → company-news endpoint (60 req/min free)
#                      Headlines scored with VADER NLP
#    "fmp"           → stock_news endpoint (250 req/day free, same key as ETF)
#                      Headlines scored with VADER NLP
#    synthetic       → Always-available fallback; derives sentiment from
#                      technical signals + score + RSI + momentum
#
#  SENTIMENT LABELS:
#    🟢 Bullish          score ≥  0.20
#    🔵 Somewhat Bullish  score ≥  0.05
#    ⚪ Neutral          score ≥ -0.05
#    🟠 Somewhat Bearish  score ≥ -0.20
#    🔴 Bearish           score <  -0.20
# ─────────────────────────────────────────────────────────────

NEWS_SENTIMENT_CONFIG = {
    # API call order (first success wins)
    "api_preference": ["alpha_vantage", "finnhub"],

    # News lookback window
    "lookback_days":            7,
    "max_articles_per_ticker": 15,

    # Alpha Vantage batch size (5 tickers per call = conservative, avoids 403s)
    "av_batch_size": 5,

    # Pause between batches (seconds) — respects free tier rate limits
    "sleep_between_batches": 1.2,

    # Sentiment label breakpoints (-1..+1 scale)
    "sentiment_thresholds": {
        "bullish":          0.20,
        "somewhat_bullish": 0.05,
        "neutral":         -0.05,
        "somewhat_bearish":-0.20,
        # below somewhat_bearish = bearish
    },

    # Output files (written to DATA_DIR)
    "output_file": "news_sentiment.csv",
    "themes_file": "macro_themes.json",
}


# ─────────────────────────────────────────────────────────────
#  9. PHASE 8 — TOP 20 HIGH-YIELD PREDICTIONS
#
#  SCORING FORMULA (all assets scored 0–100):
#
#    Stocks:
#      50 pts  Technical  =  combined_score / 75 × 50
#      30 pts  ML Signal  =  avg(prob_up_3m..12m) × 30
#                            (signal-proxy if ML data unavailable)
#      20 pts  Sentiment  =  (sentiment_score + 1) / 2 × 20
#
#    ETFs:
#      50 pts  Technical  =  tech_score / 50 × 50
#      30 pts  Momentum   =  momentum_score / 30 × 30
#      20 pts  Sentiment  =  (sentiment_score + 1) / 2 × 20
#
#  YIELD RANGE LABELS (displayed in report, based on yield_potential_score):
#    ≥ 85 → "+15–25% potential (12m)"
#    ≥ 75 → "+10–18% potential (12m)"
#    ≥ 65 → "+6–12% potential (12m)"
#    ≥ 55 → "+3–8% potential (12m)"
#    ≥ 45 → "+1–5% potential (12m)"
#     < 45 → "Moderate/uncertain outlook"
# ─────────────────────────────────────────────────────────────

TOP20_CONFIG = {
    # How many picks to include in the final ranking
    "n_picks": 20,

    # Minimum yield_potential_score to be considered (filters noise)
    "min_score": 40,

    # Max assets per type (prevents the list from being dominated by one class)
    "max_stocks": 15,
    "max_etfs":   10,
    "max_crypto":  6,

    # Scoring weights — stocks use ML signal, ETFs use momentum
    "stock_weights": {
        "technical":  0.50,   # combined_score / 75
        "ml_signal":  0.30,   # avg ML probability 3m–12m
        "sentiment":  0.20,   # news sentiment score
    },
    "etf_weights": {
        "technical":  0.50,   # tech_score / 50
        "momentum":   0.30,   # momentum_score / 30
        "sentiment":  0.20,   # news sentiment score
    },

    # Yield expectation labels (score threshold → description)
    # Listed in descending order; first match wins
    "yield_ranges": [
        (85, "+15–25% potential (12m)"),
        (75, "+10–18% potential (12m)"),
        (65, "+6–12% potential (12m)"),
        (55, "+3–8% potential (12m)"),
        (45, "+1–5% potential (12m)"),
        (0,  "Moderate/uncertain outlook"),
    ],

    # Output file
    "output_file": "top20_predictions.csv",
}


# ─────────────────────────────────────────────────────────────
#  10. PHASE 5 — ML PREDICTOR
# ─────────────────────────────────────────────────────────────

ML_CONFIG = {
    # Tickers — auto-read from screener or use manual list below
    "auto_read_screener": False,
    "screener_top_n":     15,
    "tickers": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META",
        "AMZN", "JPM", "LLY", "COST", "V",
        "PLTR", "CRM", "AMD", "TSLA", "AVGO",
    ],

    # Prediction horizons (monthly buckets, in trading days)
    "horizons": {
        "1m":  21,   # ~1 month
        "2m":  42,   # ~2 months
        "3m":  63,   # ~3 months
        "4m":  84,   # ~4 months
        "5m":  105,  # ~5 months
        "6m":  126,  # ~6 months
        "7m":  147,  # ~7 months
        "8m":  168,  # ~8 months
        "9m":  189,  # ~9 months
        "10m": 210,  # ~10 months
        "11m": 231,  # ~11 months
        "12m": 252,  # ~1 year
    },

    # Training settings
    "history_years":   5,    # Years of data — needs 5y for 12m-out targets
    "cv_folds":        5,    # Walk-forward cross-validation folds
    "min_train_size":  400,  # Min rows per CV fold

    # Caching — avoids re-downloading on every run
    "cache_data": True,
    "cache_file": "ml_price_cache.pkl",  # Auto-invalidates after 12 hours

    # Random Forest hyperparameters
    "rf_params": {
        "n_estimators":      300,
        "max_depth":         6,
        "min_samples_split": 20,
        "min_samples_leaf":  10,
        "max_features":      "sqrt",
        "random_state":      42,
        "n_jobs":            -1,
        "class_weight":      "balanced",
    },

    # XGBoost hyperparameters
    "xgb_params": {
        "n_estimators":      300,
        "max_depth":         4,
        "learning_rate":     0.05,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "use_label_encoder": False,
        "eval_metric":       "logloss",
        "random_state":      42,
        "n_jobs":            -1,
    },

    # Logistic Regression hyperparameters
    "lr_params": {
        "C":            0.1,
        "max_iter":     1000,
        "random_state": 42,
        "class_weight": "balanced",
        "solver":       "lbfgs",
    },

    # Chart: how many top features to display
    "top_features_chart": 25,
}


# ─────────────────────────────────────────────────────────────
#  9. PHASE 6 — SENTIMENT ANALYZER
# ─────────────────────────────────────────────────────────────

# Build flat ticker list from WATCHLIST, excluding pure ETFs
_SENTIMENT_TICKERS = [
    t for sector, tickers in WATCHLIST.items()
    for t in tickers
    if t not in ("SPY", "QQQ", "VTI", "SCHD", "IWM", "DIA")
]

SENTIMENT_CONFIG = {
    "tickers": _SENTIMENT_TICKERS,

    # Scoring weights — must sum to 100
    "weights": {
        "news":     35,   # VADER NLP on recent headlines
        "analyst":  30,   # Analyst consensus + price target upside
        "earnings": 20,   # Earnings beat/miss history
        "insider":  15,   # Net insider buying vs selling
    },

    # Combined score blending (Phase 1 Technical + Phase 6 Sentiment)
    "combined_tech_weight":      0.55,   # Phase 1 technical score weight
    "combined_sentiment_weight": 0.45,   # Phase 6 sentiment score weight

    # News settings
    "news_lookback_days":   7,    # Only consider articles this many days old
    "max_news_per_ticker":  20,   # Max headlines to process per ticker

    # Analyst settings
    "min_analysts": 3,            # Minimum analysts needed to trust consensus

    # Earnings settings
    "earnings_quarters": 4,       # How many quarters to look back

    # Insider settings
    "insider_lookback_days": 180, # 6-month insider activity window

    # Rate limiting
    "sleep_between_tickers": 0.5, # Seconds between ticker API calls
}
