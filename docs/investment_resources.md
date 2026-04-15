# Investment Intelligence — Tools, Repos & Resources
*Curated reference for Vijay's Investment Stack | Updated 2026-04-12*

---

## How This Connects to Your Stack

Every tool below maps to a specific phase of your project. The Phase column tells you exactly where to plug it in.

| Phase | Script | Purpose |
|---|---|---|
| 1 | stock_screener.py | Stock screening & technical scoring |
| 2 | portfolio_optimizer.py | MPT portfolio allocation |
| 3 | backtester.py | Historical strategy validation |
| 4 | daily_monitor.py | Automated morning signals |
| 5 | ml_predictor.py | ML price direction predictions |
| 6 | sentiment_analyzer.py | News, analyst & insider sentiment |

---

## Data Sources

### Free (No API Key)

| Tool | What It Gives You | Phase | Install |
|---|---|---|---|
| **yfinance** | Prices, fundamentals, news, options, insider trades, analyst ratings | 1–6 | `pip install yfinance` |
| **pandas-datareader** | FRED macro data (rates, CPI, GDP), Fama-French factors | 2, 5 | `pip install pandas-datareader` |
| **FRED API** | 800k+ economic series (interest rates, inflation, employment) | 2, 5 | `pip install fredapi` |

### Free (API Key Required — Free Tier Available)

| Tool | Free Tier | What It Gives You | Phase |
|---|---|---|---|
| **[Alpha Vantage](https://www.alphavantage.co)** | 25 req/day | OHLCV, technicals, forex, crypto, news sentiment | 1, 6 |
| **[Finnhub](https://finnhub.io)** | 60 req/min | Real-time quotes, earnings calendar, news, SEC filings | 4, 6 |
| **[Financial Modeling Prep](https://financialmodelingprep.com)** | 250 req/day | Financial statements, DCF, ratios, insider trades | 1, 2 |
| **[Polygon.io](https://polygon.io)** | Delayed data | Tick data, options, forex, crypto, news | 3, 5 |
| **[Quandl / NASDAQ Data Link](https://data.nasdaq.com)** | Limited | Alternative data, futures, economic data | 5 |

### Paid (Worth Considering Long-Term)

| Tool | Cost | Advantage |
|---|---|---|
| **[Tiingo](https://tiingo.com)** | $10/mo | Clean end-of-day + real-time, intraday, crypto |
| **[IEX Cloud](https://iexcloud.io)** | $9/mo | Real-time, options, news, earnings — great API |
| **[Intrinio](https://intrinio.com)** | $25/mo | Institutional-grade fundamentals, XBRL filings |

---

## Data Integration Code Snippets

```python
# ── Alpha Vantage (news sentiment) ─────────────────────────────
import requests
AV_KEY = "YOUR_FREE_KEY"   # get at alphavantage.co
url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey={AV_KEY}"
data = requests.get(url).json()
# Returns sentiment scores per article — plug into sentiment_analyzer.py

# ── Finnhub (earnings calendar) ────────────────────────────────
import finnhub
client = finnhub.Client(api_key="YOUR_FREE_KEY")   # finnhub.io
earnings = client.earnings_calendar(_from="2026-04-12", to="2026-04-25",
                                     symbol="", international=False)
# Plug into daily_monitor.py for upcoming earnings alerts

# ── FRED (macro context for portfolio optimizer) ────────────────
from fredapi import Fred
fred = Fred(api_key="YOUR_FREE_KEY")   # fred.stlouisfed.org
fed_rate   = fred.get_series("FEDFUNDS")        # Fed funds rate
cpi        = fred.get_series("CPIAUCSL")        # CPI inflation
yield_10yr = fred.get_series("DGS10")           # 10-year Treasury
# Use these to adjust risk_free_rate in portfolio_optimizer.py dynamically

# ── Financial Modeling Prep (DCF / fundamentals) ───────────────
FMP_KEY = "YOUR_FREE_KEY"
url = f"https://financialmodelingprep.com/api/v3/discounted-cash-flow/AAPL?apikey={FMP_KEY}"
dcf = requests.get(url).json()
# Returns DCF intrinsic value — add as fundamental_score component in screener
```

---

## Backtesting & Strategy Engines

### Backtrader *(already in your Phase 3)*
- **Repo:** [github.com/mementum/backtrader](https://github.com/mementum/backtrader)
- **Stars:** 14k+ | **Why:** Pythonic, event-driven, handles live trading too
- **Best for:** Your current Phase 3 backtester — extending with custom indicators

```python
# Upgrade your backtester.py with a Backtrader strategy class:
import backtrader as bt

class GoldenCrossStrategy(bt.Strategy):
    def __init__(self):
        self.sma50  = bt.ind.SMA(period=50)
        self.sma200 = bt.ind.SMA(period=200)
        self.cross  = bt.ind.CrossOver(self.sma50, self.sma200)

    def next(self):
        if self.cross > 0:   self.buy()
        elif self.cross < 0: self.sell()

cerebro = bt.Cerebro()
cerebro.addstrategy(GoldenCrossStrategy)
cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.045)
# Feed yfinance data via bt.feeds.PandasData
```

### QuantConnect / LEAN
- **Repo:** [github.com/QuantConnect/Lean](https://github.com/QuantConnect/Lean)
- **Stars:** 10k+ | **Why:** Institutional grade, handles equities/options/futures/crypto
- **Best for:** Production-level backtesting with minute-bar data; upgrading Phase 3
- **Cloud:** quantconnect.com — free backtesting with 10 years of data included
- **Integration:** Export your Phase 5 ML signals as LEAN Alphas

### Zipline-Reloaded
- **Repo:** [github.com/stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded)
- **Stars:** 1.5k+ | **Why:** Powers factor investing "Pipeline API" — great for screener logic
- **Best for:** Replacing/augmenting Phase 1 screener with a factor-based approach

### VectorBT
- **Repo:** [github.com/polakowo/vectorbt](https://github.com/polaково/vectorbt)
- **Stars:** 4k+ | **Why:** Vectorized backtesting — 100× faster than event-driven
- **Best for:** Rapid iteration on hundreds of parameter combinations in Phase 3

```python
# Replace backtester.py loops with vectorbt for 100x speed:
import vectorbt as vbt
price = vbt.YFData.download("AAPL", period="3y").get("Close")
rsi   = vbt.RSI.run(price, window=14)
entries = rsi.rsi_below(35)
exits   = rsi.rsi_above(70)
pf = vbt.Portfolio.from_signals(price, entries, exits, fees=0.001)
print(pf.stats())   # Sharpe, max drawdown, win rate — instant
```

---

## Portfolio Optimisation

### PyPortfolioOpt *(extends your Phase 2)*
- **Repo:** [github.com/robertmartin8/PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt)
- **Stars:** 4.5k+ | **Why:** Implements everything — Efficient Frontier, Black-Litterman, HRP
- **Integration:** Drop-in replacement for your scipy optimiser in portfolio_optimizer.py

```python
# Add Black-Litterman (view-adjusted MPT) to portfolio_optimizer.py:
from pypfopt import BlackLittermanModel, risk_models, expected_returns

mu  = expected_returns.mean_historical_return(prices)
S   = risk_models.sample_cov(prices)

# Your ML predictions (Phase 5) become "views" in Black-Litterman
viewdict = {
    "NVDA": 0.25,   # ml_predictor says +25% probability-weighted return
    "JPM":  0.08,
}
bl = BlackLittermanModel(S, pi="equal", absolute_views=viewdict)
ret_bl, S_bl = bl.bl_returns(), bl.bl_cov()
# Feed into EfficientFrontier for ML-adjusted portfolio weights
```

### Riskfolio-Lib
- **Repo:** [github.com/dcajasn/Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib)
- **Stars:** 3k+ | **Why:** 20+ risk models including CVaR, CDaR, HRP, HERC
- **Best for:** Adding tail-risk-aware optimisation beyond Sharpe to Phase 2

---

## Performance Analytics & Reporting

### QuantStats
- **Repo:** [github.com/ranaroussi/quantstats](https://github.com/ranaroussi/quantstats)
- **Stars:** 4k+ | **Why:** One-line hedge-fund-style tearsheet generation
- **Integration:** Wrap your Phase 3 backtest equity curves for instant professional reports

```python
# Add to backtester.py after simulation:
import quantstats as qs
qs.extend_pandas()

# Generate full tearsheet comparing your strategy to SPY:
qs.reports.html(
    strategy_returns,         # from your simulate() function
    benchmark="SPY",
    output="tearsheet.html",
    title="Phase 3 — Combined Signal Strategy"
)
# Generates: Sharpe, Sortino, Calmar, drawdown charts, monthly heatmap,
# rolling beta, underwater plot — everything in one HTML file
```

### pyfolio
- **Repo:** [github.com/quantopian/pyfolio](https://github.com/quantopian/pyfolio)
- **Stars:** 5k+ | **Why:** Industry-standard performance attribution, created by Quantopian
- **Best for:** Adding sector attribution and position concentration analysis to Phase 3

---

## Machine Learning & AI for Finance

### FinRL *(extends your Phase 5)*
- **Repo:** [github.com/AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL)
- **Stars:** 10k+ | **Why:** Deep reinforcement learning agents that learn to trade
- **Best for:** Upgrading Phase 5 beyond classification to autonomous RL trading agents

```python
# Replace your XGBoost with a DRL agent:
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.envs.stock_trading_env import StockTradingEnv

env = StockTradingEnv(df=your_price_data, ...)
agent = DRLAgent(env=env)
model = agent.get_model("ppo")   # or "a2c", "ddpg", "sac", "td3"
trained_model = agent.train_model(model, tb_log_name="ppo_stock")
```

### OpenBB Terminal
- **Repo:** [github.com/OpenBB-finance/OpenBBTerminal](https://github.com/OpenBB-finance/OpenBBTerminal)
- **Stars:** 34k+ | **Why:** Open-source Bloomberg — fundamentals, technicals, ML, options
- **Integration:** Use as a research layer on top of your stack; export data to feed Phase 1

### mlfinlab (Hudson & Thames)
- **Repo:** [github.com/hudson-and-thames/mlfinlab](https://github.com/hudson-and-thames/mlfinlab)
- **Stars:** 4k+ | **Why:** Implements "Advances in Financial ML" (Marcos Lopez de Prado)
- **Best for:** Upgrading Phase 5 with fractional differentiation, triple-barrier labelling

```python
# Replace binary up/down labels in ml_predictor.py with triple-barrier:
from mlfinlab.labeling import triple_barrier_labeling
# Labels: +1 (hit profit target), -1 (hit stop loss), 0 (timeout)
# Far more realistic than simple next-N-day direction
```

### sklearn-compatible alternatives for Phase 5

```python
# Drop-in replacements / additions in ml_predictor.py:
from lightgbm import LGBMClassifier         # Often faster than XGBoost
from catboost import CatBoostClassifier     # Handles categoricals natively
from sklearn.calibration import CalibratedClassifierCV  # Better probability calibration
# Calibrate your existing models for more reliable P(up) scores:
cal_rf = CalibratedClassifierCV(your_rf_model, cv=5, method="isotonic")
```

---

## Sentiment & Alternative Data

### VADER + FinBERT
- **VADER Repo:** [github.com/cjhutto/vaderSentiment](https://github.com/cjhutto/vaderSentiment) *(already in Phase 6)*
- **FinBERT:** [github.com/ProsusAI/finbert](https://github.com/ProsusAI/finbert) — BERT fine-tuned on financial text, more accurate than VADER for long articles

```python
# Upgrade sentiment_analyzer.py: swap VADER for FinBERT on key articles
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
model     = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")

def finbert_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    # Returns: [negative, neutral, positive]
    return float(probs[2]) - float(probs[0])  # net sentiment [-1, +1]
```

### Stock News API + Reddit Sentiment
```python
# Reddit WallStreetBets sentiment (free, via praw):
import praw
reddit = praw.Reddit(client_id="...", client_secret="...", user_agent="...")
wsb = reddit.subreddit("wallstreetbets")
mentions = [post.title for post in wsb.hot(limit=100) if "NVDA" in post.title]
```

---

## Live Trading & Execution

> **Important:** Paper-trade for at least 3–6 months before using real capital.
> Your Phase 3 backtester must show consistent edge before going live.

### Alpaca *(best free option)*
- **Repo:** [github.com/alpacahq/alpaca-trade-api-python](https://github.com/alpacahq/alpaca-trade-api-python)
- **Why:** Commission-free, paper trading supported, REST + WebSocket API
- **Integration:** Wire Phase 4 daily_monitor.py signals directly to Alpaca orders

```python
# Add to daily_monitor.py: auto-execute screener signals
import alpaca_trade_api as tradeapi
api = tradeapi.REST("KEY_ID", "SECRET_KEY", base_url="https://paper-api.alpaca.markets")

def execute_signal(ticker, signal, capital_fraction=0.05):
    account = api.get_account()
    buying_power = float(account.buying_power)
    price = float(api.get_latest_trade(ticker).price)
    qty   = int((buying_power * capital_fraction) / price)

    if "BUY" in signal and qty > 0:
        api.submit_order(symbol=ticker, qty=qty, side="buy",
                         type="market", time_in_force="day")
    elif "AVOID" in signal:
        try:
            api.submit_order(symbol=ticker, qty=qty, side="sell",
                             type="market", time_in_force="day")
        except Exception:
            pass
```

### Interactive Brokers (IBKR)
- **Repo:** [github.com/InteractiveBrokers/tws-api](https://github.com/InteractiveBrokers/tws-api)
- **Python wrapper:** [github.com/erdewit/ib_insync](https://github.com/erdewit/ib_insync)
- **Why:** Best for international stocks, options, futures — institutional-grade execution

---

## Useful Curated Lists

| Resource | URL | What's Inside |
|---|---|---|
| Awesome Systematic Trading | [github.com/wangzhe3224/awesome-systematic-trading](https://github.com/wangzhe3224/awesome-systematic-trading) | 400+ libraries categorised by use case |
| Awesome Quant | [github.com/wilsonfreitas/awesome-quant](https://github.com/wilsonfreitas/awesome-quant) | Data, analysis, pricing, risk, ML |
| ML for Trading (book + code) | [github.com/stefan-jansen/machine-learning-for-algorithmic-trading-2nd-edition](https://github.com/stefan-jansen/machine-learning-for-algorithmic-trading-2nd-edition) | 900-page textbook with full Python code |

---

## Recommended Reading

- **"Advances in Financial Machine Learning"** — Marcos Lopez de Prado (essential for Phase 5)
- **"Quantitative Trading"** — Ernest Chan (strategy development fundamentals)
- **"Algorithmic Trading"** — Ernest Chan (execution and live trading)
- **"Active Portfolio Management"** — Grinold & Kahn (the MPT bible, relevant to Phase 2)

---

## Quick-Start Priority List

Run these in order to maximally upgrade your existing stack:

```bash
# 1. Add professional tearsheet reporting to Phase 3
pip install quantstats

# 2. Add Black-Litterman + HRP to Phase 2
pip install PyPortfolioOpt riskfolio-lib

# 3. Add FinBERT to Phase 6 (needs GPU for speed, CPU works too)
pip install transformers torch

# 4. Add vectorized backtesting to Phase 3 (100x faster parameter sweeps)
pip install vectorbt

# 5. Add live paper trading to Phase 4
pip install alpaca-trade-api

# 6. Add FRED macro data integration to Phases 2 & 5
pip install fredapi pandas-datareader
```

---

*This document is a living reference — update it as you discover new tools.*
*All repos are open-source. Always validate strategies with Phase 3 backtesting*
*before deploying real capital.*
