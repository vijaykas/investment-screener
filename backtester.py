"""
============================================================
  INVESTMENT INTELLIGENCE — PHASE 3
  Strategy Backtesting Engine
  Author: Claude (Vijay's Investment Project)
  Updated: 2026-04-12
============================================================

WHAT THIS DOES:
  Simulates 5 trading strategies on your stock universe over a
  configurable historical window, measures real performance, and
  compares every strategy against a SPY Buy-and-Hold benchmark.

  STRATEGIES TESTED:
    1. RSI Mean Reversion  — buy oversold, sell overbought
    2. MACD Crossover      — buy bullish cross, sell bearish cross
    3. Golden Cross        — SMA50 > SMA200 trend following
    4. Bollinger Bounce    — buy lower band, sell upper band
    5. Combined Signal     — composite of all 4 (like our screener)

  PERFORMANCE METRICS (per strategy, per ticker + rolled up):
    • Total Return & CAGR
    • Sharpe Ratio  (risk-adjusted return)
    • Sortino Ratio (downside-risk-adjusted)
    • Max Drawdown  (worst peak-to-trough loss)
    • Calmar Ratio  (CAGR / Max Drawdown)
    • Win Rate, Profit Factor
    • Avg Win / Avg Loss, Best / Worst trade
    • Number of trades

  OUTPUTS:
    • Terminal: ranked strategy comparison table
    • backtest_results.csv       — full results per ticker × strategy
    • charts/equity_curves.html  — portfolio value over time vs SPY
    • charts/drawdown.html        — drawdown timeline
    • charts/monthly_returns.html — monthly heatmap (like a fund report)
    • charts/strategy_comparison.html — bar chart of key metrics

HOW TO RUN:
  python3 backtester.py

CUSTOMISE:
  Edit BACKTEST_CONFIG below to change tickers, date range,
  commission, position sizing, or strategy parameters.
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os, sys

# ─────────────────────────────────────────────
#  CONFIGURATION  (loaded from config.py)
# ─────────────────────────────────────────────
from config import BACKTEST_CONFIG, DATA_DIR, CHARTS_DIR

OUTPUT_DIR = DATA_DIR   # CSVs written to data/

STRATEGY_COLORS = {
    "RSI Mean Reversion": "#00E676",
    "MACD Crossover":     "#40C4FF",
    "Golden Cross":       "#FFD740",
    "Bollinger Bounce":   "#FF6D00",
    "Combined Signal":    "#EA80FC",
    "Buy & Hold":         "#B0BEC5",
}


# ─────────────────────────────────────────────
#  DATA LAYER
# ─────────────────────────────────────────────

def load_tickers(cfg: dict) -> list[str]:
    if cfg["auto_read_screener"]:
        csv_path = os.path.join(OUTPUT_DIR, "stock_screener_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df[~df["ticker"].isin(["SPY","QQQ","VTI","SCHD","IWM","DIA"])]
            tickers = df.head(cfg["screener_top_n"])["ticker"].tolist()
            print(f"  📂 Loaded {len(tickers)} tickers from screener.")
            return tickers
    return cfg["tickers"]


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty or len(df) < 100:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df[["Open","High","Low","Close","Volume"]].copy()
    except Exception:
        return None


# ─────────────────────────────────────────────
#  SIGNAL GENERATORS
# ─────────────────────────────────────────────

def signals_rsi(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Buy when RSI crosses below buy threshold; sell when RSI crosses above sell threshold."""
    close = df["Close"].squeeze()
    rsi = ta.momentum.RSIIndicator(close=close, window=cfg["rsi_period"]).rsi()

    position = pd.Series(0, index=df.index, dtype=int)
    in_trade = False
    for i in range(1, len(df)):
        if not in_trade and rsi.iloc[i] < cfg["rsi_buy"] and rsi.iloc[i-1] >= cfg["rsi_buy"]:
            position.iloc[i] = 1    # Enter long
            in_trade = True
        elif in_trade and rsi.iloc[i] > cfg["rsi_sell"] and rsi.iloc[i-1] <= cfg["rsi_sell"]:
            position.iloc[i] = -1   # Exit long
            in_trade = False
    return position


def signals_macd(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Buy on MACD bullish crossover (MACD line crosses above signal); sell on bearish."""
    close = df["Close"].squeeze()
    macd_ind = ta.trend.MACD(close=close,
                              window_fast=cfg["macd_fast"],
                              window_slow=cfg["macd_slow"],
                              window_sign=cfg["macd_signal"])
    macd_line   = macd_ind.macd()
    signal_line = macd_ind.macd_signal()

    position = pd.Series(0, index=df.index, dtype=int)
    in_trade = False
    for i in range(1, len(df)):
        crossed_above = macd_line.iloc[i] > signal_line.iloc[i] and macd_line.iloc[i-1] <= signal_line.iloc[i-1]
        crossed_below = macd_line.iloc[i] < signal_line.iloc[i] and macd_line.iloc[i-1] >= signal_line.iloc[i-1]
        if not in_trade and crossed_above:
            position.iloc[i] = 1
            in_trade = True
        elif in_trade and crossed_below:
            position.iloc[i] = -1
            in_trade = False
    return position


def signals_golden_cross(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Buy when SMA_fast crosses above SMA_slow (golden cross); sell on death cross."""
    close = df["Close"].squeeze()
    sma_fast = ta.trend.SMAIndicator(close=close, window=cfg["sma_fast"]).sma_indicator()
    sma_slow = ta.trend.SMAIndicator(close=close, window=cfg["sma_slow"]).sma_indicator()

    position = pd.Series(0, index=df.index, dtype=int)
    in_trade = False
    for i in range(1, len(df)):
        if pd.isna(sma_slow.iloc[i]) or pd.isna(sma_slow.iloc[i-1]):
            continue
        golden = sma_fast.iloc[i] > sma_slow.iloc[i] and sma_fast.iloc[i-1] <= sma_slow.iloc[i-1]
        death  = sma_fast.iloc[i] < sma_slow.iloc[i] and sma_fast.iloc[i-1] >= sma_slow.iloc[i-1]
        if not in_trade and golden:
            position.iloc[i] = 1
            in_trade = True
        elif in_trade and death:
            position.iloc[i] = -1
            in_trade = False
    return position


def signals_bollinger(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Buy near lower Bollinger Band; sell near upper Bollinger Band."""
    close = df["Close"].squeeze()
    bb = ta.volatility.BollingerBands(close=close,
                                       window=cfg["bb_period"],
                                       window_dev=cfg["bb_std"])
    pband = bb.bollinger_pband()   # 0 = at lower band, 1 = at upper band

    position = pd.Series(0, index=df.index, dtype=int)
    in_trade = False
    for i in range(1, len(df)):
        if pd.isna(pband.iloc[i]):
            continue
        if not in_trade and pband.iloc[i] < 0.10:   # Near lower band
            position.iloc[i] = 1
            in_trade = True
        elif in_trade and pband.iloc[i] > 0.90:     # Near upper band
            position.iloc[i] = -1
            in_trade = False
    return position


def signals_combined(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Composite signal: requires agreement from at least 3 of 4 indicators.
    Buy when RSI oversold + MACD bullish + price above SMA200 + not BB overbought.
    """
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()

    rsi = ta.momentum.RSIIndicator(close=close, window=cfg["rsi_period"]).rsi()
    macd_ind = ta.trend.MACD(close=close, window_fast=cfg["macd_fast"],
                              window_slow=cfg["macd_slow"], window_sign=cfg["macd_signal"])
    macd_line   = macd_ind.macd()
    signal_line = macd_ind.macd_signal()
    sma200      = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
    bb          = ta.volatility.BollingerBands(close=close, window=cfg["bb_period"])
    pband       = bb.bollinger_pband()

    position = pd.Series(0, index=df.index, dtype=int)
    in_trade = False
    for i in range(1, len(df)):
        if pd.isna(sma200.iloc[i]) or pd.isna(pband.iloc[i]):
            continue

        # Score each condition: +1 if bullish
        score_buy = sum([
            rsi.iloc[i] < cfg["rsi_buy"],                                    # RSI oversold
            macd_line.iloc[i] > signal_line.iloc[i],                         # MACD bullish
            close.iloc[i] > sma200.iloc[i],                                  # Above 200MA
            pband.iloc[i] < 0.50,                                             # Lower half of BB
        ])
        score_sell = sum([
            rsi.iloc[i] > cfg["rsi_sell"],                                   # RSI overbought
            macd_line.iloc[i] < signal_line.iloc[i],                         # MACD bearish
            close.iloc[i] < sma200.iloc[i],                                  # Below 200MA
            pband.iloc[i] > 0.85,                                             # Near BB upper band
        ])

        if not in_trade and score_buy >= 3:
            position.iloc[i] = 1
            in_trade = True
        elif in_trade and score_sell >= 3:
            position.iloc[i] = -1
            in_trade = False

    return position


STRATEGIES = {
    "RSI Mean Reversion": signals_rsi,
    "MACD Crossover":     signals_macd,
    "Golden Cross":       signals_golden_cross,
    "Bollinger Bounce":   signals_bollinger,
    "Combined Signal":    signals_combined,
}


# ─────────────────────────────────────────────
#  SIMULATION ENGINE
# ─────────────────────────────────────────────

def simulate(df: pd.DataFrame, signals: pd.Series, cfg: dict) -> dict:
    """
    Event-driven simulation with next-bar execution, commission, and slippage.
    Returns dict of trade-level and time-series results.
    """
    capital     = cfg["initial_capital"]
    pos_size    = cfg["position_size"]
    commission  = cfg["commission"]
    slippage    = cfg["slippage"]

    portfolio_values = pd.Series(capital, index=df.index, dtype=float)
    shares_held  = 0
    entry_price  = 0.0
    entry_date   = None
    trades       = []   # List of (entry_date, exit_date, entry_px, exit_px, return_pct)
    cash         = capital

    for i in range(len(df) - 1):
        sig = signals.iloc[i]
        # Execute on NEXT bar open (avoid look-ahead)
        exec_price = df["Open"].iloc[i + 1]

        if sig == 1 and shares_held == 0:
            # BUY
            fill = exec_price * (1 + slippage)
            invest = cash * pos_size
            shares_held = invest / fill
            cost = shares_held * fill * commission
            cash = cash - invest - cost
            entry_price = fill
            entry_date  = df.index[i + 1]

        elif sig == -1 and shares_held > 0:
            # SELL
            fill = exec_price * (1 - slippage)
            proceeds = shares_held * fill
            cost = proceeds * commission
            trade_ret = (fill - entry_price) / entry_price
            trades.append({
                "entry_date": entry_date,
                "exit_date":  df.index[i + 1],
                "entry_px":   round(entry_price, 4),
                "exit_px":    round(fill, 4),
                "return_pct": round(trade_ret * 100, 3),
                "holding_days": (df.index[i + 1] - entry_date).days,
            })
            cash += proceeds - cost
            shares_held = 0
            entry_price = 0.0

        # Mark-to-market portfolio value
        mkt_value = cash + shares_held * df["Close"].iloc[i + 1]
        portfolio_values.iloc[i + 1] = mkt_value

    # Close any open position at last available price
    if shares_held > 0:
        last_price = df["Close"].iloc[-1]
        fill = last_price * (1 - slippage)
        proceeds = shares_held * fill
        cost = proceeds * commission
        trade_ret = (fill - entry_price) / entry_price
        trades.append({
            "entry_date": entry_date,
            "exit_date":  df.index[-1],
            "entry_px":   round(entry_price, 4),
            "exit_px":    round(fill, 4),
            "return_pct": round(trade_ret * 100, 3),
            "holding_days": (df.index[-1] - entry_date).days,
        })
        cash += proceeds - cost
        portfolio_values.iloc[-1] = cash

    return {"portfolio_values": portfolio_values, "trades": pd.DataFrame(trades)}


# ─────────────────────────────────────────────
#  PERFORMANCE METRICS
# ─────────────────────────────────────────────

def compute_metrics(pv: pd.Series, trades: pd.DataFrame, cfg: dict) -> dict:
    """Compute a full suite of performance metrics from a portfolio value series."""
    td  = cfg["trading_days"]
    rf  = cfg["risk_free_rate"]
    cap = cfg["initial_capital"]

    daily_returns = pv.pct_change().dropna()
    n_years = len(daily_returns) / td

    total_return = (pv.iloc[-1] / cap) - 1
    cagr = (pv.iloc[-1] / cap) ** (1 / max(n_years, 0.01)) - 1

    ann_vol = daily_returns.std() * np.sqrt(td)
    sharpe  = (cagr - rf) / ann_vol if ann_vol > 0 else 0

    downside = daily_returns[daily_returns < 0].std() * np.sqrt(td)
    sortino  = (cagr - rf) / downside if downside > 0 else 0

    # Max drawdown
    rolling_max = pv.cummax()
    drawdown = (pv - rolling_max) / rolling_max
    max_dd   = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Trade-level stats
    n_trades = len(trades)
    if n_trades > 0:
        wins     = trades[trades["return_pct"] > 0]
        losses   = trades[trades["return_pct"] <= 0]
        win_rate = len(wins) / n_trades
        avg_win  = wins["return_pct"].mean()   if len(wins)   > 0 else 0
        avg_loss = losses["return_pct"].mean() if len(losses) > 0 else 0
        best_trade  = trades["return_pct"].max()
        worst_trade = trades["return_pct"].min()
        gross_profit = wins["return_pct"].sum()
        gross_loss   = abs(losses["return_pct"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_holding  = trades["holding_days"].mean()
    else:
        win_rate = avg_win = avg_loss = best_trade = worst_trade = 0
        profit_factor = avg_holding = 0

    return {
        "total_return_%":    round(total_return * 100, 2),
        "cagr_%":            round(cagr * 100, 2),
        "ann_volatility_%":  round(ann_vol * 100, 2),
        "sharpe":            round(sharpe, 3),
        "sortino":           round(sortino, 3),
        "max_drawdown_%":    round(max_dd * 100, 2),
        "calmar":            round(calmar, 3),
        "n_trades":          n_trades,
        "win_rate_%":        round(win_rate * 100, 1),
        "avg_win_%":         round(avg_win, 2),
        "avg_loss_%":        round(avg_loss, 2),
        "profit_factor":     round(profit_factor, 2),
        "best_trade_%":      round(best_trade, 2),
        "worst_trade_%":     round(worst_trade, 2),
        "avg_holding_days":  round(avg_holding, 1),
    }


def buy_and_hold_metrics(df: pd.DataFrame, cfg: dict) -> tuple[pd.Series, dict]:
    """Simulate a simple buy-and-hold from day 1."""
    cap = cfg["initial_capital"]
    price_0 = df["Close"].iloc[0]
    shares = (cap * cfg["position_size"]) / price_0
    pv = df["Close"].squeeze() * shares + cap * (1 - cfg["position_size"])
    metrics = compute_metrics(pv, pd.DataFrame(), cfg)
    return pv, metrics


# ─────────────────────────────────────────────
#  PORTFOLIO AGGREGATION
# ─────────────────────────────────────────────

def equal_weight_portfolio(equity_curves: dict, cfg: dict) -> pd.Series:
    """Average all ticker equity curves into a single equal-weight portfolio curve."""
    n = len(equity_curves)
    if n == 0:
        return pd.Series(dtype=float)
    # Normalise each curve to start at 1.0, then weight equally
    normed = pd.DataFrame({t: c / c.iloc[0] for t, c in equity_curves.items()})
    aligned = normed.ffill().dropna()
    portfolio = aligned.mean(axis=1) * cfg["initial_capital"]
    return portfolio


# ─────────────────────────────────────────────
#  CHARTING
# ─────────────────────────────────────────────

def chart_equity_curves(strategy_curves: dict, benchmark_pv: pd.Series, cfg: dict):
    """Normalised equity curves: each strategy vs Buy & Hold SPY."""
    fig = go.Figure()
    cap = cfg["initial_capital"]

    # Benchmark
    bm_norm = benchmark_pv / benchmark_pv.iloc[0] * 100
    fig.add_trace(go.Scatter(
        x=bm_norm.index, y=bm_norm.values,
        name=f"{cfg['benchmark']} Buy & Hold",
        line=dict(color=STRATEGY_COLORS["Buy & Hold"], width=2, dash="dot"),
        hovertemplate="%{y:.1f}  (%{x|%Y-%m-%d})"
    ))

    for strat, pv in strategy_curves.items():
        norm = pv / pv.iloc[0] * 100
        final_ret = (pv.iloc[-1] / pv.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm.values,
            name=f"{strat} ({final_ret:+.1f}%)",
            line=dict(color=STRATEGY_COLORS.get(strat, "#ffffff"), width=2),
            hovertemplate=f"<b>{strat}</b><br>%{{y:.1f}}<br>%{{x|%Y-%m-%d}}"
        ))

    fig.update_layout(
        title=dict(text="<b>Strategy Equity Curves</b> — Normalised to 100 at Start",
                   font=dict(size=15)),
        xaxis_title="Date",
        yaxis_title="Portfolio Value (Start = 100)",
        template="plotly_dark",
        height=580,
        hovermode="x unified",
        legend=dict(orientation="v", x=1.01, y=1),
        margin=dict(l=70, r=220, t=80, b=60),
    )
    path = os.path.join(CHARTS_DIR, "equity_curves.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"    ✅ Equity Curves → {path}")


def chart_drawdowns(strategy_curves: dict, benchmark_pv: pd.Series, cfg: dict):
    """Drawdown chart — shows % below previous peak."""
    fig = go.Figure()

    bm_dd = (benchmark_pv / benchmark_pv.cummax() - 1) * 100
    fig.add_trace(go.Scatter(
        x=bm_dd.index, y=bm_dd.values,
        name=f"{cfg['benchmark']} Buy & Hold",
        line=dict(color=STRATEGY_COLORS["Buy & Hold"], width=1.5, dash="dot"),
        fill="tozeroy", fillcolor="rgba(176,190,197,0.10)"
    ))

    for strat, pv in strategy_curves.items():
        dd = (pv / pv.cummax() - 1) * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            name=strat,
            line=dict(color=STRATEGY_COLORS.get(strat, "#fff"), width=1.5),
            fill="tozeroy",
            fillcolor=f"rgba(0,0,0,0)"
        ))

    fig.update_layout(
        title="<b>Drawdown Timeline</b> — % Below Rolling Peak",
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        template="plotly_dark", height=450,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=70, r=60, t=80, b=100),
    )
    path = os.path.join(CHARTS_DIR, "drawdown.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"    ✅ Drawdown Chart → {path}")


def chart_monthly_returns(pv: pd.Series, strategy_name: str):
    """Calendar heatmap of monthly returns — like a hedge fund tear sheet."""
    monthly = pv.resample("ME").last().pct_change().dropna() * 100
    monthly.index = monthly.index.to_period("M")

    years  = sorted(monthly.index.year.unique())
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    z = np.full((len(years), 12), np.nan)
    for period, val in monthly.items():
        yi = years.index(period.year)
        mi = period.month - 1
        z[yi][mi] = round(val, 2)

    text = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=months, y=[str(y) for y in years],
        colorscale=[
            [0.0, "#B71C1C"],   # Deep red = big loss
            [0.4, "#EF9A9A"],
            [0.5, "#F5F5F5"],   # White = flat
            [0.6, "#A5D6A7"],
            [1.0, "#1B5E20"],   # Deep green = big gain
        ],
        zmid=0,
        text=text, texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="Return %", thickness=14),
        hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%"
    ))
    fig.update_layout(
        title=f"<b>Monthly Returns Heatmap</b> — {strategy_name}",
        template="plotly_dark", height=max(300, len(years) * 45 + 150),
        margin=dict(l=70, r=80, t=80, b=50),
    )
    slug = strategy_name.lower().replace(" ", "_")
    path = os.path.join(CHARTS_DIR, f"monthly_returns_{slug}.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"    ✅ Monthly Returns ({strategy_name}) → {path}")


def chart_strategy_comparison(summary_df: pd.DataFrame):
    """Side-by-side bar chart of key metrics across all strategies."""
    metrics_to_plot = {
        "cagr_%":           "CAGR (%)",
        "sharpe":           "Sharpe Ratio",
        "sortino":          "Sortino Ratio",
        "max_drawdown_%":   "Max Drawdown (%)",
        "win_rate_%":       "Win Rate (%)",
    }

    fig = make_subplots(
        rows=1, cols=len(metrics_to_plot),
        subplot_titles=list(metrics_to_plot.values()),
        horizontal_spacing=0.06
    )

    strategies = summary_df["strategy"].tolist()
    colors = [STRATEGY_COLORS.get(s, "#ffffff") for s in strategies]

    for col, (metric, label) in enumerate(metrics_to_plot.items(), 1):
        vals = summary_df[metric].tolist()
        # Max drawdown: flip to positive for visual comparison
        if metric == "max_drawdown_%":
            vals = [abs(v) for v in vals]

        fig.add_trace(go.Bar(
            x=strategies, y=vals,
            marker_color=colors,
            showlegend=False,
            text=[f"{v:.2f}" for v in vals],
            textposition="outside",
            textfont=dict(size=10),
        ), row=1, col=col)

    fig.update_layout(
        title="<b>Strategy Comparison</b> — Key Performance Metrics",
        template="plotly_dark",
        height=500,
        margin=dict(l=50, r=50, t=100, b=100),
    )
    for i in range(1, len(metrics_to_plot) + 1):
        fig.update_xaxes(tickangle=-30, row=1, col=i)

    path = os.path.join(CHARTS_DIR, "strategy_comparison.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"    ✅ Strategy Comparison → {path}")


# ─────────────────────────────────────────────
#  MAIN RUNNER
# ─────────────────────────────────────────────

def run_backtest():
    cfg = BACKTEST_CONFIG
    tickers = load_tickers(cfg)

    print(f"\n{'='*70}")
    print(f"  INVESTMENT INTELLIGENCE — Backtesting Engine")
    print(f"  Period: {cfg['start_date']}  →  {cfg['end_date']}")
    print(f"  Universe: {len(tickers)} stocks  |  Capital: ${cfg['initial_capital']:,}")
    print(f"  Commission: {cfg['commission']*100:.1f}%  |  Slippage: {cfg['slippage']*100:.1f}%")
    print(f"{'='*70}\n")

    # ── Download data ─────────────────────────
    print(f"  Downloading price data...")
    price_data = {}
    for i, ticker in enumerate(tickers):
        sys.stdout.write(f"\r  [{i+1}/{len(tickers)}] {ticker:<8}...")
        sys.stdout.flush()
        df = fetch_ohlcv(ticker, cfg["start_date"], cfg["end_date"])
        if df is not None:
            price_data[ticker] = df
    print(f"\n  ✅ Loaded {len(price_data)} tickers.\n")

    # Download benchmark
    bm_df = fetch_ohlcv(cfg["benchmark"], cfg["start_date"], cfg["end_date"])

    # ── Run all strategies ────────────────────
    all_results   = []   # Flat rows for CSV
    strategy_pv   = {}   # {strategy_name: portfolio_value_series}

    for strat_name, signal_fn in STRATEGIES.items():
        print(f"  ▶ Running: {strat_name}")
        ticker_curves = {}
        strat_trades  = []
        strat_metrics_list = []

        for ticker, df in price_data.items():
            try:
                signals = signal_fn(df, cfg)
                sim     = simulate(df, signals, cfg)
                pv      = sim["portfolio_values"]
                trades  = sim["trades"]
                metrics = compute_metrics(pv, trades, cfg)
                metrics["ticker"]   = ticker
                metrics["strategy"] = strat_name
                all_results.append(metrics)
                ticker_curves[ticker] = pv
                strat_trades.append(trades)
            except Exception as e:
                print(f"      ⚠️  {ticker}: {e}")

        # Equal-weight portfolio for this strategy
        port_pv = equal_weight_portfolio(ticker_curves, cfg)
        if not port_pv.empty:
            strategy_pv[strat_name] = port_pv
            all_trades = pd.concat(strat_trades, ignore_index=True) if strat_trades else pd.DataFrame()
            m = compute_metrics(port_pv, all_trades, cfg)
            print(f"       → CAGR: {m['cagr_%']:+.1f}%  "
                  f"Sharpe: {m['sharpe']:.2f}  "
                  f"MaxDD: {m['max_drawdown_%']:.1f}%  "
                  f"WinRate: {m['win_rate_%']:.0f}%  "
                  f"Trades: {m['n_trades']}")

    # ── Benchmark ─────────────────────────────
    print(f"\n  ▶ Running: {cfg['benchmark']} Buy & Hold")
    if bm_df is not None:
        bm_pv, bm_metrics = buy_and_hold_metrics(bm_df, cfg)
        bm_metrics["strategy"] = f"{cfg['benchmark']} Buy & Hold"
        print(f"       → CAGR: {bm_metrics['cagr_%']:+.1f}%  "
              f"Sharpe: {bm_metrics['sharpe']:.2f}  "
              f"MaxDD: {bm_metrics['max_drawdown_%']:.1f}%")
    else:
        bm_pv = None

    # ── Summary table ─────────────────────────
    print(f"\n{'='*80}")
    print(f"  STRATEGY SUMMARY (Equal-Weight Portfolio across all tickers)")
    print(f"{'='*80}")
    print(f"  {'Strategy':<24} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>8} {'WinRate':>8} {'Trades':>7}")
    print(f"  {'─'*75}")

    summary_rows = []
    for strat_name, pv in strategy_pv.items():
        all_t = pd.concat([r for r in [pd.DataFrame()] ], ignore_index=True)
        m = compute_metrics(pv, pd.DataFrame(), cfg)
        m["strategy"] = strat_name
        summary_rows.append(m)
        print(f"  {strat_name:<24} {m['cagr_%']:>+6.1f}%  {m['sharpe']:>6.2f}  "
              f"{m['sortino']:>7.2f}  {m['max_drawdown_%']:>7.1f}%  "
              f"{m['win_rate_%']:>6.1f}%  {m['n_trades']:>6}")

    if bm_pv is not None:
        print(f"  {bm_metrics['strategy']:<24} {bm_metrics['cagr_%']:>+6.1f}%  "
              f"{bm_metrics['sharpe']:>6.2f}  {bm_metrics['sortino']:>7.2f}  "
              f"{bm_metrics['max_drawdown_%']:>7.1f}%  {'—':>6}  {'—':>6}")
    print(f"{'='*80}")

    # Rank strategies by Sharpe ratio
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("sharpe", ascending=False).reset_index(drop=True)

    best = summary_df.iloc[0]
    print(f"\n  🏆 Best strategy by Sharpe: {best['strategy']}")
    print(f"     CAGR: {best['cagr_%']:+.1f}%  |  Sharpe: {best['sharpe']:.2f}  "
          f"|  MaxDD: {best['max_drawdown_%']:.1f}%")

    # ── Per-ticker breakdown for top strategy ─
    best_strat = best["strategy"]
    ticker_rows = [r for r in all_results if r["strategy"] == best_strat]
    if ticker_rows:
        tk_df = pd.DataFrame(ticker_rows).sort_values("sharpe", ascending=False)
        print(f"\n  📊 Top 5 Tickers under '{best_strat}':")
        print(f"  {'Ticker':<10} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'WinRate':>8} {'Trades':>7}")
        print(f"  {'─'*50}")
        for _, row in tk_df.head(5).iterrows():
            print(f"  {row['ticker']:<10} {row['cagr_%']:>+6.1f}%  {row['sharpe']:>6.2f}  "
                  f"{row['max_drawdown_%']:>7.1f}%  {row['win_rate_%']:>6.1f}%  "
                  f"{row['n_trades']:>6}")

    # ── Save CSV ──────────────────────────────
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, "backtest_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n  💾 Full results → {csv_path}")

    # ── Charts ────────────────────────────────
    print(f"\n  📈 Generating charts...")
    if bm_pv is not None:
        chart_equity_curves(strategy_pv, bm_pv, cfg)
        chart_drawdowns(strategy_pv, bm_pv, cfg)

    # Monthly returns for best strategy
    if best_strat in strategy_pv:
        chart_monthly_returns(strategy_pv[best_strat], best_strat)

    chart_strategy_comparison(summary_df)

    print(f"\n{'='*70}")
    print(f"  🏁 BACKTEST COMPLETE — {len(list(STRATEGIES))} strategies × {len(price_data)} tickers")
    print(f"  Charts: {CHARTS_DIR}/")
    print(f"\n  ⚠️  DISCLAIMER: Backtested results do not guarantee future performance.")
    print(f"     Overfitting, survivorship bias, and execution slippage affect real")
    print(f"     trading. This is for research purposes only.")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_backtest()
