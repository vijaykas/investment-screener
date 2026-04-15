# 💹 Investment Intelligence Stack

A personal, end-to-end investment research and analysis system built in Python.
Screens stocks, optimises portfolios, backtests strategies, monitors daily signals,
predicts price direction with machine learning, and scores news & analyst sentiment —
all from a single command.

> **Disclaimer:** This project is for research and educational purposes only.
> It does not constitute financial advice. Always do your own due diligence and
> consult a licensed financial advisor before making any investment decisions.

---

## Quick Start

```bash
# Install dependencies (one time)
pip install yfinance ta plotly pandas numpy scipy scikit-learn xgboost \
            backtrader vaderSentiment textblob fredapi finnhub-python

# Launch the interactive menu
python3 invest.py

# Or run the full pipeline directly
python3 invest.py --all

# Or the recommended daily workflow
python3 invest.py --quick        # Screener → Sentiment → Optimizer
```

---

## Project Structure

```
investment/
│
├── invest.py                 ← 🚀 Single entry point — run everything from here
├── config.py                 ← ⚙️  MASTER CONFIG — the only file you need to edit
│
├── stock_screener.py         ← Phase 1:  Stock Screener
├── portfolio_optimizer.py    ← Phase 2:  Portfolio Optimizer
├── backtester.py             ← Phase 3:  Strategy Backtester
├── daily_monitor.py          ← Phase 4:  Daily Morning Monitor
├── etf_screener.py           ← Phase 4b: ETF Screener & Dashboard
├── crypto_screener.py        ← Phase 4c: Crypto Screener (26 quality coins, no meme coins)
├── ml_predictor.py           ← Phase 5:  ML Predictive Layer
├── sentiment_analyzer.py     ← Phase 6:  Sentiment & News Layer
├── news_sentiment.py         ← Phase 7:  News & Sentiment Engine (AV / Finnhub / FMP)
├── top20_picker.py           ← Phase 8:  Top 20 High-Yield Predictions
│
├── data/                     ← All CSV outputs, JSON history, ML cache
│   ├── stock_screener_results.csv
│   ├── etf_screener_results.csv  ← ETF scores (42 ETFs across 6 categories)
│   ├── crypto_screener_results.csv ← Crypto scores (26 quality coins, Phase 4c)
│   ├── crypto_cycle.json         ← BTC market cycle context (Phase 4c)
│   ├── portfolio_results.csv
│   ├── backtest_results.csv
│   ├── ml_predictions.csv
│   ├── ml_model_performance.csv
│   ├── sentiment_results.csv
│   ├── news_sentiment.csv    ← Per-ticker sentiment (Phase 7)
│   ├── macro_themes.json     ← Active market themes (Phase 7)
│   ├── top20_predictions.csv ← Top 20 ranked predictions (Phase 8)
│   ├── signal_history.json   ← 90-day rolling signal history
│   └── ml_price_cache.pkl    ← 12-hour price cache (auto-invalidates)
│
├── charts/                   ← Interactive HTML charts (open in any browser)
├── daily_reports/            ← Daily HTML reports (YYYY-MM-DD.html)
├── models/                   ← Trained ML model files (<TICKER>_rf.pkl)
│
├── docs/
│   └── investment_resources.md  ← Curated tools, repos & APIs reference
│
├── README.md
└── .invest_state.json        ← Run history (used by invest.py status panel)
```

---

## `invest.py` — The Unified Runner

All six phases are launched from a single script. You never need to run individual
phase files directly.

### Interactive Menu (no arguments)

```bash
python3 invest.py
```

Displays a live status dashboard showing the last run time and success/failure of
each phase. Use number keys `[1–6]` to run individual phases, `[A]` for the full
pipeline, `[Q]` for the quick run, `[S]` for the status view, `[X]` to exit.

### Command-Line Flags

| Command | What it does |
|---|---|
| `python3 invest.py --all` | Run all phases in order |
| `python3 invest.py --quick` | Run Phase 1 → 4b → 4c → 7 → 8 → 4 (recommended daily workflow) |
| `python3 invest.py --screener` | Phase 1 only |
| `python3 invest.py --optimizer` | Phase 2 only |
| `python3 invest.py --backtest` | Phase 3 only |
| `python3 invest.py --monitor` | Phase 4 only |
| `python3 invest.py --etf` | Phase 4b only (ETF Screener) |
| `python3 invest.py --crypto` | Phase 4c only (Crypto Screener) |
| `python3 invest.py --ml` | Phase 5 only |
| `python3 invest.py --sentiment` | Phase 6 only |
| `python3 invest.py --news` | Phase 7 only (News & Sentiment) |
| `python3 invest.py --top20` | Phase 8 only (Top 20 Predictions) |
| `python3 invest.py --phase 1 4b 4c 7 8 4` | Run any combination in order |
| `python3 invest.py --status` | Show run history + output files |
| `python3 invest.py --screener --debug` | Verbose error output |

---

## Phase 1 — Stock Screener (`stock_screener.py`)

Scans a universe of 55 stocks across 9 sectors and scores each one using technical
and fundamental analysis.

**What it scans:**
Tech, AI/Growth, Finance, Healthcare, Consumer, Energy, Industrial, Dividend, ETFs

**Technical scoring (60 pts):**
- RSI 14 — oversold/overbought momentum
- MACD — bullish crossover detection
- Moving Averages — price vs SMA 20/50/200, golden cross detection
- Bollinger Bands — position % and squeeze/breakout signal
- ADX — trend strength (above 25 = trending, above 40 = strong)
- Volume — surge detection + On-Balance Volume trend

**Fundamental scoring (40 pts):**
- PE ratio and Forward PE valuation
- Revenue and earnings growth (YoY)
- Profit margins
- Debt-to-equity ratio

**Outputs:**
- Terminal: ranked table of all stocks with signals
- `data/stock_screener_results.csv` — full scored dataset, sortable
- `charts/<TICKER>_chart.html` — 4-panel interactive chart (candlestick, volume, RSI, MACD) for the top 5 picks

**Signal labels:** 🟢 STRONG BUY / 🔵 BUY / 🟡 HOLD / 🟠 CAUTION / 🔴 AVOID

**Customise:** Edit `WATCHLIST`, `SCREENER_CONFIG`, and `SCREENER_WEIGHTS` in `config.py`. Set `WATCHLIST_MODE = "sp500"` to screen the entire S&P 500.

---

## Phase 2 — Portfolio Optimizer (`portfolio_optimizer.py`)

Takes the top stock picks and finds the mathematically optimal allocation using
Modern Portfolio Theory.

**What it does:**
1. Downloads 2 years of price history for all tickers
2. Computes annualised expected returns, volatility, and pairwise correlations
3. Runs a 15,000-portfolio Monte Carlo simulation to map the Efficient Frontier
4. Uses SciPy constrained optimisation to find three exact optimal portfolios

**Three optimised portfolios:**

| Portfolio | Goal | Best for |
|---|---|---|
| Max Sharpe | Best return per unit of risk | Most investors — the default choice |
| Min Volatility | Lowest possible risk | Conservative / capital preservation |
| Max Return | Highest return within weight limits | Aggressive / high-risk appetite |
| Equal Weight | Naïve 1/N benchmark | Comparison baseline |

**Weight constraints:** Each stock 1%–40% (configurable). No micro-positions,
no dangerous concentration.

**Outputs:**
- Terminal: allocation weights, CAGR, volatility, Sharpe per portfolio
- `portfolio_results.csv`
- `charts/efficient_frontier.html` — Monte Carlo cloud with optimal portfolio stars
- `charts/weights_max_sharpe.html` — recommended allocation breakdown
- `charts/correlation_heatmap.html` — diversification view
- `charts/risk_return.html` — each stock's individual risk vs return

**Configuration:** `AUTO_READ_SCREENER = True` (default) reads top picks from
Phase 1 automatically. Set `SCREENER_TOP_N` to control how many.

---

## Phase 3 — Strategy Backtester (`backtester.py`)

Tests 5 trading strategies on your stock universe over 3 years of historical data,
with realistic transaction costs and no lookahead bias.

**Strategies tested:**

| Strategy | Entry Signal | Exit Signal |
|---|---|---|
| RSI Mean Reversion | RSI crosses below 35 | RSI crosses above 70 |
| MACD Crossover | MACD crosses above signal line | MACD crosses below signal line |
| Golden Cross | SMA50 crosses above SMA200 | SMA50 crosses below SMA200 |
| Bollinger Bounce | Price near lower band (BB% < 10%) | Price near upper band (BB% > 90%) |
| Combined Signal | 3-of-4 indicators agree bullish | 3-of-4 indicators agree bearish |

**Execution model:**
- Next-bar open fill (no lookahead)
- 0.10% commission + 0.10% slippage per trade
- $100,000 starting capital, 95% deployed per trade
- Compared against SPY buy-and-hold benchmark

**Metrics per strategy:**
CAGR, Sharpe, Sortino, Max Drawdown, Calmar Ratio, Win Rate,
Profit Factor, Avg Win/Loss, Best/Worst Trade, Avg Holding Days

**Outputs:**
- Terminal: strategy comparison table + per-ticker breakdown for top strategy
- `backtest_results.csv`
- `charts/equity_curves.html` — all strategies vs SPY, normalised
- `charts/drawdown.html` — peak-to-trough drawdown timeline
- `charts/monthly_returns_<strategy>.html` — hedge-fund style monthly heatmap
- `charts/strategy_comparison.html` — bar chart of key metrics

---

## Phase 4 — Daily Monitor (`daily_monitor.py`)

Runs automatically every weekday morning at **8:00 AM** via a scheduled Claude task.
Re-scans the watchlist, detects signal changes since yesterday, and delivers a
morning briefing as a self-contained HTML report.

**What it checks daily:**
1. Scores every stock in the watchlist (same engine as Phase 1, technical only)
2. Compares signals to the previous session — detects upgrades, downgrades, fresh MACD crossovers
3. Flags notable events: price moves >±3%, 52-week highs/lows, RSI extremes, volume surges (>2× average)
4. Computes portfolio P&L for positions defined in `YOUR_POSITIONS` in `config.py`
5. Loads ML 12-month predictions from Phase 5 and embeds them in the report
6. Generates an HTML report and appends to the 90-day signal history log

**HTML report sections:**
- 🔢 **Market Overview** — KPI dashboard (Strong Buy / Buy / Avoid counts, upgrades/downgrades)
- 🔔 **Signal Changes** — upgrades and downgrades since yesterday
- 🏆 **Top 10 Stocks Today** — highest-scoring stocks with RSI, MACD, volume; each ticker is a clickable Yahoo Finance link
- 💼 **Portfolio P&L** — unrealised gains/losses per position with alerts
- 🤖 **ML 12-Month Outlook** — Ensemble model P(Up)% for every monthly horizon (1m → 12m), colour-coded from bearish red to bullish green
- ⚡ **Notable Events & Alerts** — big moves, 52-week extremes, volume surges

**To track your actual holdings:**
Edit `YOUR_POSITIONS` in `config.py` (Section 3):

```python
YOUR_POSITIONS = {
    "AAPL":  {"shares": 10,  "avg_cost": 175.00},
    "NVDA":  {"shares": 5,   "avg_cost": 480.00},
    "MSFT":  {"shares": 8,   "avg_cost": 350.00},
}
```

**Alert thresholds** (all configurable in `config.py` → `MONITOR_CONFIG`):
- Big move flag: ±3% daily change
- Volume surge: 2× 20-day average
- Drawdown alert: position down >10% from cost

**Outputs:**
- Terminal: ranked stock table + signal changes + events + P&L summary
- `daily_reports/YYYY-MM-DD.html` — self-contained HTML report, open in any browser
- `data/signal_history.json` — rolling 90-day history of all signals

**Schedule:** Weekdays 8:00 AM local time. Manage in the Scheduled section of the Claude desktop app sidebar.

---

## Phase 4b — ETF Screener (`etf_screener.py`)

Scores **42 ETFs across 6 asset-class categories** on a composite 0–100 scale and
embeds the results in the daily HTML report as a dedicated **ETF Market Dashboard** section.

**ETF Categories:**

| Category | ETFs |
|---|---|
| Broad Market | SPY, QQQ, VTI, IWM, DIA, VUG, VTV |
| Sector | XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU, XLRE, XLC |
| International | VEA, VWO, EFA, EEM, VGK, EWJ, IEFA |
| Fixed Income | BND, AGG, TLT, IEF, SHY, HYG, LQD, BNDX |
| Commodity | GLD, IAU, SLV, USO, DJP, PDBC |
| Real Estate | VNQ, SCHH, IYR |

**Scoring (0–100):**

| Component | Weight | What it measures |
|---|---|---|
| Technical | 50 pts | RSI, MACD crossover, 50/200MA position, Bollinger %, ADX, Volume surge |
| Momentum vs SPY | 30 pts | 1m / 3m / 6m return relative to S&P 500 benchmark |
| Quality | 20 pts | Expense ratio (lower = better) + AUM/liquidity (larger = better) |

**Signal labels:** 🟢 STRONG BUY (≥75) / 🔵 BUY (≥58) / 🟡 HOLD (≥42) / 🟠 CAUTION (≥28) / 🔴 AVOID (<28)

**Daily Report — ETF Dashboard panels:**
- **Category chips strip** — one badge per category showing average 3m return vs SPY
- **Category heatmap** — grid of ETF tiles colour-coded green/grey/red by 3m relative return (hover for details)
- **Full comparison table** — all 42 ETFs sorted by score with Price, Day Change, RSI, 200MA, 1m/3m/6m/12m returns vs SPY, Expense ratio, AUM, Yield
- **Sector spotlight** — GICS sector ETFs (XL\*) ranked by 3m momentum vs SPY

**Customise:** Edit `ETF_WATCHLIST` and `ETF_MONITOR_CONFIG` in `config.py` (Section 7b)
to add/remove ETFs or adjust scoring thresholds.

**Outputs:**
- Terminal: full ETF ranking table + category summary
- `data/etf_screener_results.csv` — full scored dataset
- ETF Dashboard section embedded in `daily_reports/YYYY-MM-DD.html`

---

## Phase 4c — Crypto Screener (`crypto_screener.py`)

Scores **26 quality cryptocurrency assets across 5 categories** on a composite 0–100 scale,
embeds a **₿ Crypto Market Dashboard** in the daily HTML report, and includes the top crypto
picks in the unified **Top 20 High-Yield Predictions**.

**No meme coins** — the screener explicitly excludes DOGE, SHIB, PEPE, FLOKI, WIF and other
speculative assets with no fundamental utility. Only projects with proven use cases are included.

**Crypto Categories:**

| Category | Coins |
|---|---|
| Layer 1 | BTC, ETH, SOL, ADA, AVAX, DOT, NEAR, APT, SUI |
| Layer 2 | MATIC, OP, ARB |
| DeFi | LINK, UNI, AAVE, MKR, LDO, ATOM |
| AI / Infrastructure | RNDR, FET, GRT, INJ |
| Payments / Interop | XRP, XLM, BNB, ALGO |

**Scoring (0–100):**

| Component | Weight | What it measures |
|---|---|---|
| Technical | 40 pts | RSI, MACD, 50/200 MA, Bollinger %, ADX (crypto-adapted thresholds) |
| Momentum vs BTC | 25 pts | 7d / 30d / 90d return relative to Bitcoin benchmark |
| On-chain Quality | 20 pts | Market cap tier + network type bonus (L1/L2/DeFi) + volume activity |
| Sentiment | 15 pts | News sentiment from Phase 7, or signal-derived synthetic fallback |

**Signal labels:** 🟢 STRONG BUY (≥78) / 🔵 BUY (≥62) / 🟡 HOLD (≥46) / 🟠 CAUTION (≥30) / 🔴 AVOID (<30)

**BTC Market Cycle** (based on BTC vs 200-day MA):
- 🟢 Bull Market: BTC > 200 DMA by +20%
- 🟡 Late Bull: +5% to +20% above 200 DMA
- 🟠 Caution Zone: within ±5% of 200 DMA
- 🔴 Bear Market: below 200 DMA

**Daily Report — Crypto Market Dashboard panels:**
- **Market cycle banner** — BTC price, % vs 200 DMA, % from ATH
- **Category heatmap** — grid of coin tiles colour-coded by 30d return vs BTC
- **Full leaderboard table** — all 26 coins sorted by score with Price, Day Change, RSI, Momentum, Quality score

**Customise:** Edit `CRYPTO_WATCHLIST` and `CRYPTO_CONFIG` in `config.py` (Section 7c/7d).

**Outputs:**
- Terminal: crypto leaderboard + BTC market cycle summary
- `data/crypto_screener_results.csv` — full scored dataset (26 coins)
- `data/crypto_cycle.json` — BTC market cycle context
- Crypto Dashboard section embedded in `daily_reports/YYYY-MM-DD.html`

---

## Phase 5 — ML Predictor (`ml_predictor.py`)

Trains machine learning models to predict whether each stock will be higher or
lower at each monthly milestone from **1 month to 12 months** out, using 5 years
of historical data.

**Models:**

| Model | Description |
|---|---|
| Random Forest | 300 trees, handles non-linearity, no scaling required |
| XGBoost | Gradient boosting, typically highest AUC |
| Logistic Regression | Fast linear baseline with L2 regularisation |
| Ensemble Voter | Soft-vote combination of all three |

**Prediction Horizons (12 monthly buckets):**

| Horizon | Trading Days | Description |
|---|---|---|
| 1m | 21 | ~1 month |
| 2m | 42 | ~2 months |
| 3m | 63 | ~3 months |
| … | … | … |
| 6m | 126 | ~6 months |
| … | … | … |
| 12m | 252 | ~1 year |

**45+ engineered features:**

| Category | Features |
|---|---|
| Price & Returns | 1/2/3/5/10/20/21/42/63/126/252-day lagged returns, daily range, close position |
| Moving Averages | Price/SMA ratios (10/20/50/100/200), EMA ratios, SMA50/200 + SMA100/200 crossovers |
| Long-Horizon Momentum | Quarterly (1m vs 3m), semi-annual (3m vs 6m) price acceleration |
| Momentum | RSI-7/14/21, Stochastic RSI K+D, RSI velocity, ROC (5/10/20d) |
| Trend | MACD line/histogram (normalised), ADX, +DI/−DI |
| Volatility | Bollinger Band %, bandwidth z-score, ATR ratio + delta, short/long vol regime |
| 52-Week Range | Price position within yearly high/low range |
| Volume | Volume ratio, OBV slope, volume trend, lagged volume |
| Calendar | Day-of-week + month sin/cos encoding |

**Validation:** 5-fold `TimeSeriesSplit` with a 5-day gap — strictly no lookahead.
Requires 400+ samples per fold (5 years of data ensures sufficient coverage even
for 12-month targets).

**Terminal output:** model accuracy table + multi-horizon prediction table showing
P(up)% at 1m, 3m, 6m, 12m for each ticker, sorted by 6-month confidence.

**Outputs:**
- `ml_predictions.csv` — P(up) for all tickers × models × 12 horizons
- `ml_model_performance.csv` — accuracy, precision, recall, F1, AUC per fold
- `models/<TICKER>_rf.pkl` — saved Random Forest models for reuse in Phase 4
- `charts/ml_prediction_timeline.html` — P(Up) curve across all 12 months per stock ⭐
- `charts/ml_feature_importance.html` — top features (RF vs XGB, 3m horizon)
- `charts/ml_roc_curves.html` — ROC curves per model (3m horizon)
- `charts/ml_confidence_6m.html` — 6-month confidence heatmap (tickers × models)
- `charts/ml_model_accuracy.html` — accuracy and AUC comparison

**Note:** First run takes 15–25 minutes (5y data + 12 horizons × 4 models). Subsequent
runs reuse `ml_price_cache.pkl` (auto-invalidates after 12 hours).

---

## Phase 6 — Sentiment Analyzer (`sentiment_analyzer.py`)

Scores every stock on four independent sentiment signals, then blends with the
Phase 1 technical score into a final Combined Score (0–100).

**Four signals:**

| Signal | Weight | Source | Method |
|---|---|---|---|
| News NLP | 35 pts | yfinance headlines | VADER + 30 financial keyword boosters, recency-weighted |
| Analyst Consensus | 30 pts | Analyst ratings | Recommendation mean (1=Strong Buy → 5=Sell) + price target upside % |
| Earnings Surprise | 20 pts | Earnings history | Beat/miss magnitude × recency weighting, last 4 quarters |
| Insider Activity | 15 pts | Insider transactions | Net buy-to-sell ratio over last 6 months |

**Combined Score formula:**
```
Combined Score = (Technical Score / 75 × 100) × 0.55
              + Sentiment Score × 0.45
```

**Combined signal labels:**
🟢 STRONG BUY (≥78) / 🔵 BUY (≥62) / 🟡 HOLD (≥47) / 🟠 CAUTION (≥33) / 🔴 AVOID

**Outputs:**
- Terminal: combined ranking table + per-stock sentiment breakdown for top 10
- `sentiment_results.csv` — all scores merged back into screener CSV
- `charts/sentiment_dashboard.html` — stacked bar per stock (all 4 components)
- `charts/combined_ranking.html` — final leaderboard with signal colours
- `charts/analyst_heatmap.html` — price target upside + recommendation ratings
- `charts/news_sentiment_scatter.html` — each headline as a dot, coloured by score

---

## Phase 7 — News & Sentiment Engine (`news_sentiment.py`)

Fetches recent news headlines and computes **per-ticker sentiment scores** (-1.0 → +1.0) for every stock and ETF in the watchlist. Identifies **macro market themes** from the news cycle that are currently driving prices.

**Data sources (tried in order, first success wins):**

| Source | API | Free Tier | Notes |
|---|---|---|---|
| Alpha Vantage NEWS_SENTIMENT | `alpha_vantage` key | 500 req/day | Pre-scored articles with ticker-specific sentiment |
| Finnhub company-news | `finnhub` key | 60 req/min | Headlines scored with VADER NLP |
| FMP stock_news | `fmp` key | 250 req/day | Headlines scored with VADER NLP |
| Synthetic (fallback) | None required | Always available | Derives sentiment from signal + RSI + momentum |

**Sentiment labels:**

| Score | Label |
|---|---|
| ≥ 0.20 | 🟢 Bullish |
| ≥ 0.05 | 🔵 Somewhat Bullish |
| ≥ -0.05 | ⚪ Neutral |
| ≥ -0.20 | 🟠 Somewhat Bearish |
| < -0.20 | 🔴 Bearish |

**Macro themes tracked:** AI Infrastructure Buildout, Fed Rate Cut Watch, GLP-1 Drug Revolution, Gold/Safe-Haven Demand, China/Trade Tariff Risk, Q1 Earnings Season, Energy Transition, Small-Cap Recovery — shown as coloured chips in the daily report.

**Outputs:**
- `data/news_sentiment.csv` — 51+ tickers with score, label, top headline, theme tags
- `data/macro_themes.json` — 8 active market themes with sentiment and affected tickers

---

## Phase 8 — Top 20 High-Yield Predictions (`top20_picker.py`)

Combines every available signal to **rank the top 20 highest-potential investments** across all asset classes (stocks + ETFs + crypto) into a single unified leaderboard.

**Scoring formula (all assets scored 0–100):**

| Component | Stocks (pts) | ETFs (pts) | Crypto (pts) | Source |
|---|---|---|---|---|
| Technical | 50 | 50 | 40 | Stock: combined_score/75 × 50 · ETF: tech_score/50 × 50 · Crypto: tech_score/40 × 40 |
| ML / Momentum | 30 | 30 | 25 | Stocks: avg ML prob_up 3m-12m · ETFs: momentum_score/30 × 30 · Crypto: momentum vs BTC |
| Quality | — | — | 20 | Crypto only: on-chain quality proxy (market cap tier + network type + volume) |
| Sentiment | 20 | 20 | 15 | (sentiment_score + 1) / 2 × weight from Phase 7 |

**Yield potential ranges:**

| Score | Expected Return (12m) |
|---|---|
| ≥ 85 | +15–25% potential |
| ≥ 75 | +10–18% potential |
| ≥ 65 | +6–12% potential |
| ≥ 55 | +3–8% potential |
| ≥ 45 | +1–5% potential |
| < 45 | Moderate/uncertain outlook |

**Report section — 🏆 Top 20 High-Yield Predictions:**
- Macro theme chips at the top (live news cycle context)
- Podium cards for #1, #2, #3 — score, signal, sentiment, yield range, news headline
- Full ranked table for #4–#20 with type badge (STOCK / ETF / ₿ CRYPTO), all metrics
- Mix capped at 15 stocks + 10 ETFs + 6 cryptos maximum

**Outputs:**
- `data/top20_predictions.csv` — ranked predictions with all components

---

## Recommended Run Order

```bash
# Best daily workflow — Screener → ETF → Crypto → News → Top20 → Monitor (fresh report)
python3 invest.py --quick
# Runs: Phase 1 → Phase 4b → Phase 4c → Phase 7 → Phase 8 → Phase 4

# News + Top 20 refresh only (fast, no live price fetch)
python3 invest.py --news --top20

# Full research session (run weekly or when building a new position)
python3 invest.py --all
# Runs: 1 → 2 → 3 → 4 → 4b → 4c → 5 → 6 → 7 → 8 in sequence

# Just check what happened overnight
python3 invest.py --monitor

# Refresh ETF scores only (fast — ~2 min for 42 ETFs)
python3 invest.py --etf

# Refresh Crypto scores only (~1 min for 26 coins)
python3 invest.py --crypto
```

**Phase dependencies:**

```
Phase 6 (Sentiment)   ──reads──► Phase 1 output (stock_screener_results.csv)
Phase 7 (News)        ──reads──► Phase 1 + 4b + 4c outputs (for fallback signal-derived sentiment)
Phase 8 (Top 20)      ──reads──► Phase 1 + 4b + 4c + 5 + 7 outputs
Phase 2 (Optimizer)   ──reads──► Phase 1 output when AUTO_READ_SCREENER=True
Phase 4 (Monitor)     ──reads──► Phase 4b + 4c + 7 + 8 outputs (ETF + Crypto + Top 20 in report)
invest.py warns you   if you run a dependent phase without its prerequisite.
```

---

## Configuration Reference

**All settings live in one file: `config.py`** — open it to customise the entire stack. Each phase script imports its config from there; you never need to edit individual phase scripts.

### API Keys (Section 1 of config.py)

```python
API_KEYS = {
    "finnhub":       "",   # https://finnhub.io  — free, 60 req/min
    "fred":          "",   # https://fred.stlouisfed.org/docs/api/api_key.html — free, unlimited
    "alpha_vantage": "",   # https://www.alphavantage.co  — free, 25 req/day
    "fmp":           "",   # https://financialmodelingprep.com — free, 250 req/day
}
```

Or set as environment variables (safer on shared machines):
```bash
export FINNHUB_KEY=your_key_here
export FRED_KEY=your_key_here
export ALPHA_VANTAGE_KEY=your_key_here
export FMP_KEY=your_key_here
```

### ETF Universe — Static or Live FMP Feed

`ETF_WATCHLIST_MODE` in `config.py` controls where the ETF list comes from:

```python
ETF_WATCHLIST_MODE = "static"   # 42 hand-curated ETFs across 6 categories (default, no API needed)
ETF_WATCHLIST_MODE = "fmp"      # Live top-ETFs by AUM from Financial Modeling Prep
```

**To enable `"fmp"` mode:**

1. Get a free API key at [financialmodelingprep.com](https://financialmodelingprep.com/developer/docs/) (250 req/day free)
2. Paste it into `config.py` Section 1:
   ```python
   "fmp": "your_fmp_key_here",
   ```
3. Set the mode:
   ```python
   ETF_WATCHLIST_MODE = "fmp"
   ```
4. Run `python3 invest.py --etf`

In `"fmp"` mode the screener fetches the full US ETF universe (~3,000 funds), filters to those with AUM ≥ `min_aum_B` (default $0.5B), takes the top `max_total` (default 120) by AUM, and auto-categorises them using ETF name keywords into up to 10 categories — including categories not available in static mode: **Thematic, Dividend, Factor, Leveraged**. The screener gracefully falls back to the static list if the API call fails.

| Config key | Default | Effect |
|---|---|---|
| `top_n_per_category` | 12 | Max ETFs per auto-detected category |
| `max_total` | 120 | Hard cap on total ETF universe (affects runtime) |
| `min_aum_B` | 0.5 | Exclude funds with AUM below $500M |

### Watchlist — Static or Live S&P 500

The watchlist controls which stocks all phases scan. Set `WATCHLIST_MODE` in `config.py`:

```python
WATCHLIST_MODE = "static"   # Use your hand-picked WATCHLIST dict (default, fast)
WATCHLIST_MODE = "sp500"    # Auto-fetch all ~500 S&P 500 stocks from Wikipedia each run
```

In `"static"` mode, edit the `WATCHLIST` dict in Section 2 of `config.py`. Changes there automatically propagate to Phase 1, 4, and 6. In `"sp500"` mode the list is fetched live and grouped by GICS sector — screener runs will take longer but cover the full index.

### Key Settings Quick Reference

| What to change | Location in config.py | Setting |
|---|---|---|
| Watchlist mode | Section 2 | `WATCHLIST_MODE = "static"` or `"sp500"` |
| Stocks to track | Section 2 `WATCHLIST` | Add/remove tickers per sector |
| Your holdings | Section 3 `YOUR_POSITIONS` | `"AAPL": {"shares": 10, "avg_cost": 175.00}` |
| Screener filters | Section 4 `SCREENER_CONFIG` | `min_price`, `max_pe`, `rsi_oversold` |
| Tech vs fundamental weight | Section 4 `SCREENER_WEIGHTS` | `{"technical": 60, "fundamental": 40}` |
| Risk-free rate | Section 5 `PORTFOLIO_CONFIG` | `"risk_free_rate": 0.045` |
| Max position size | Section 5 `PORTFOLIO_CONFIG` | `"max_weight": 0.40` |
| Backtest date range | Section 6 `BACKTEST_CONFIG` | `"start_date": "2022-01-01"` |
| Commission/slippage | Section 6 `BACKTEST_CONFIG` | `"commission": 0.001` |
| P&L alert threshold | Section 7 `MONITOR_CONFIG` | `"drawdown_alert_pct": 10.0` |
| ETF universe mode | Section 7b | `ETF_WATCHLIST_MODE = "static"` or `"fmp"` |
| Add/remove ETFs (static) | Section 7b `ETF_WATCHLIST` | Add tickers to existing categories or create new ones |
| FMP API key | Section 1 `API_KEYS` | `"fmp": "your_key"` — free at financialmodelingprep.com |
| ETFs per category (fmp) | Section 7b `ETF_FETCH_CONFIG` | `"top_n_per_category": 12` |
| Min AUM filter (fmp) | Section 7b `ETF_FETCH_CONFIG` | `"min_aum_B": 0.5` (excludes illiquid funds) |
| Total ETF cap (fmp) | Section 7b `ETF_FETCH_CONFIG` | `"max_total": 120` |
| ML prediction horizons | Section 10 `ML_CONFIG` | `"horizons": {"1m": 21, ..., "12m": 252}` |
| ML training history | Section 10 `ML_CONFIG` | `"history_years": 5` |
| Sentiment lookback (Phase 6) | Section 11 `SENTIMENT_CONFIG` | `"news_lookback_days": 7` |
| News API preference order | Section 8 `NEWS_SENTIMENT_CONFIG` | `"api_preference": ["alpha_vantage", "finnhub"]` |
| News lookback window | Section 8 `NEWS_SENTIMENT_CONFIG` | `"lookback_days": 7` |
| Sentiment thresholds | Section 8 `NEWS_SENTIMENT_CONFIG` | `"sentiment_thresholds": {"bullish": 0.20, ...}` |
| Top 20 pick count | Section 9 `TOP20_CONFIG` | `"n_picks": 20` |
| Top 20 stock/ETF/crypto caps | Section 9 `TOP20_CONFIG` | `"max_stocks": 15, "max_etfs": 10, "max_crypto": 6` |
| Top 20 min score | Section 9 `TOP20_CONFIG` | `"min_score": 40` (filters noise from bottom of rankings) |

---

## Extending the Stack

See `investment_resources.md` for the complete curated reference with
copy-paste integration code. Key next upgrades:

| Tool | What it adds | Phase to upgrade |
|---|---|---|
| `quantstats` | One-line hedge-fund tearsheet from equity curves | 3 |
| `PyPortfolioOpt` | Black-Litterman + Hierarchical Risk Parity | 2 |
| `vectorbt` | 100× faster parameter sweep backtesting | 3 |
| `FinBERT` | Transformer-based NLP, more accurate than VADER | 6 |
| `FinRL` | Deep reinforcement learning trading agents | 5 |
| `alpaca-trade-api` | Paper/live order execution from Phase 4 signals | 4 |

Install all upgrades at once:
```bash
pip install quantstats PyPortfolioOpt riskfolio-lib vectorbt \
            transformers torch alpaca-trade-api fredapi pandas-datareader
```

---

## Output Files Reference

| File | Updated by | Contents |
|---|---|---|
| `data/stock_screener_results.csv` | Phase 1, 6 | All tickers scored + sentiment merged |
| `data/etf_screener_results.csv` | Phase 4b | 42 ETFs scored — technical + momentum + quality |
| `data/portfolio_results.csv` | Phase 2 | Optimal weights per portfolio type |
| `data/backtest_results.csv` | Phase 3 | Strategy performance per ticker |
| `data/ml_predictions.csv` | Phase 5 | P(up) per ticker × model × 12 horizons |
| `data/ml_model_performance.csv` | Phase 5 | Accuracy, AUC per model per ticker |
| `data/sentiment_results.csv` | Phase 6 | All 4 sentiment scores + combined signal |
| `data/news_sentiment.csv` | Phase 7 | Per-ticker sentiment score, label, headline, themes |
| `data/macro_themes.json` | Phase 7 | Active market themes with sentiment + tickers |
| `data/crypto_screener_results.csv` | Phase 4c | 26 quality crypto coins scored 0–100 |
| `data/crypto_cycle.json` | Phase 4c | BTC market cycle (Bull/Late Bull/Caution/Bear) |
| `data/top20_predictions.csv` | Phase 8 | Top 20 ranked by Yield Potential Score (stocks + ETFs + crypto) |
| `data/signal_history.json` | Phase 4 | Rolling 90-day signal log |
| `data/ml_price_cache.pkl` | Phase 5 | 12-hour price cache (auto-invalidates) |
| `daily_reports/YYYY-MM-DD.html` | Phase 4 | Self-contained HTML report — stocks + ETFs + crypto + Top 20 |
| `charts/*.html` | All phases | Interactive Plotly charts |
| `models/<TICKER>_rf.pkl` | Phase 5 | Saved Random Forest models |
| `docs/investment_resources.md` | — | Curated tools, repos & API reference |
| `.invest_state.json` | invest.py | Run history for status panel |

---

## Changelog

| Date | Change |
|---|---|
| 2026-04-12 | Phase 1: Stock Screener — 55 stocks, 9 sectors, technical + fundamental |
| 2026-04-12 | Phase 2: Portfolio Optimizer — MPT, 15k Monte Carlo, Sharpe/MinVol/MaxReturn |
| 2026-04-12 | Phase 3: Backtester — 5 strategies, walk-forward, SPY benchmark |
| 2026-04-12 | Phase 4: Daily Monitor — 8 AM weekday auto-scan, HTML reports, P&L tracking |
| 2026-04-12 | Phase 5: ML Predictor — RF + XGBoost + LR ensemble, 45+ features, 12 monthly horizons (1m–12m) |
| 2026-04-12 | config.py — unified master config; all phases import from here (API keys, watchlist, all settings) |
| 2026-04-12 | File organisation — CSVs/cache moved to data/, docs to docs/; config.py exports all path constants |
| 2026-04-12 | Dynamic watchlist — WATCHLIST_MODE "static" or "sp500" (live S&P 500 from Wikipedia) |
| 2026-04-13 | Daily report — all tickers now clickable links to Yahoo Finance (open in new tab) |
| 2026-04-13 | Daily report — new 🤖 ML 12-Month Outlook section with colour-coded probability heatmap (1m→12m) |
| 2026-04-13 | Daily report — Portfolio P&L and Events sections also have clickable ticker links |
| 2026-04-13 | Phase 4b: ETF Screener — 42 ETFs across 6 categories; Technical + Momentum vs SPY + Quality (0–100) |
| 2026-04-13 | Daily report — new 📈 ETF Market Dashboard: category heatmap, full comparison table, sector spotlight |
| 2026-04-13 | config.py — added ETF_WATCHLIST (6 categories, 42 ETFs) and ETF_MONITOR_CONFIG |
| 2026-04-13 | invest.py — Phase 4b (--etf flag), --quick now includes ETF screener, --all covers all phases |
| 2026-04-14 | ETF dynamic mode — ETF_WATCHLIST_MODE="fmp" fetches top-120 ETFs by AUM from FMP API, auto-categorises into 10 buckets (incl. Thematic, Dividend, Factor, Leveraged); falls back to static on error |
| 2026-04-14 | config.py — added FMP API key, ETF_FETCH_CONFIG, ETF_WATCHLIST_MODE, get_etf_watchlist() |
| 2026-04-13 | Phase 7: News & Sentiment Engine (news_sentiment.py) — AV / Finnhub / FMP APIs with signal-derived synthetic fallback; identifies 8 macro market themes from news cycle |
| 2026-04-13 | Phase 8: Top 20 High-Yield Predictions (top20_picker.py) — unified stock + ETF ranking by Yield Potential Score (Technical 50 + ML/Momentum 30 + Sentiment 20) with yield range labels |
| 2026-04-13 | Daily report: new 🏆 Top 20 Predictions section — podium cards for #1–3, ranked table for #4–20, macro theme chips, sentiment badges, news headlines, yield ranges |
| 2026-04-13 | config.py — added NEWS_SENTIMENT_CONFIG (Section 8) and TOP20_CONFIG (Section 9) |
| 2026-04-13 | invest.py — Phase 7 (--news), Phase 8 (--top20); --quick now runs 1 → 4b → 7 → 8 → 4; --all includes all 8 phases |
| 2026-04-14 | Phase 4c: Crypto Screener (crypto_screener.py) — 26 quality coins across 5 categories; Technical(40)+Momentum vs BTC(25)+On-chain Quality(20)+Sentiment(15); meme coins excluded |
| 2026-04-14 | Daily report: new ₿ Crypto Market Dashboard — BTC market cycle banner, category heatmap, full leaderboard (26 coins) |
| 2026-04-14 | Phase 8 upgraded: Top 20 now unified across stocks + ETFs + crypto; max_crypto=6 cap; ₿ CRYPTO badge in rankings |
| 2026-04-14 | Phase 7 upgraded: news_sentiment.py now processes crypto tickers (symbol-mapped BTC-USD → BTC for news APIs) |
| 2026-04-14 | config.py — added CRYPTO_WATCHLIST (5 categories, 26 coins), CRYPTO_CONFIG (Section 7c/7d) |
| 2026-04-14 | invest.py — Phase 4c (--crypto flag); --quick now runs 1 → 4b → 4c → 7 → 8 → 4; --all covers all phases including 4c |
| 2026-04-12 | Phase 6: Sentiment Analyzer — VADER NLP, analyst, earnings, insider signals |
| 2026-04-12 | invest.py — unified CLI runner + interactive menu |
| 2026-04-12 | investment_resources.md — curated tools, repos & API reference |
| 2026-04-12 | README.md — this file |
