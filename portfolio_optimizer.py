"""
============================================================
  INVESTMENT INTELLIGENCE — PHASE 2
  Portfolio Optimizer  (Modern Portfolio Theory)
  Author: Claude (Vijay's Investment Project)
  Updated: 2026-04-12
============================================================

WHAT THIS DOES:
  1. Downloads 2 years of historical price data for your chosen stocks
  2. Computes expected returns, volatility, and correlation
  3. Runs a 15,000-portfolio Monte Carlo simulation to map the Efficient Frontier
  4. Finds three mathematically optimal portfolios:
       • Max Sharpe  — best risk-adjusted return
       • Min Volatility — lowest risk for any given return
       • Max Return  — highest return (concentrated, higher risk)
  5. Generates interactive Plotly charts:
       • Efficient Frontier scatter (colour-coded by Sharpe ratio)
       • Optimal weights bar chart
       • Correlation heatmap
       • Individual stock risk/return scatter
  6. Saves results to:
       • portfolio_results.csv
       • charts/efficient_frontier.html
       • charts/weights_<portfolio>.html
       • charts/correlation_heatmap.html

HOW TO RUN:
  python3 portfolio_optimizer.py

  TIP: Edit PORTFOLIO_TICKERS to match the top stocks from your screener,
       or set AUTO_READ_SCREENER = True to pull them automatically.
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import os
import sys

# ─────────────────────────────────────────────
#  CONFIGURATION  (loaded from config.py)
# ─────────────────────────────────────────────
from config import PORTFOLIO_CONFIG, DATA_DIR, CHARTS_DIR

CONFIG             = PORTFOLIO_CONFIG
AUTO_READ_SCREENER = PORTFOLIO_CONFIG["auto_read_screener"]
SCREENER_TOP_N     = PORTFOLIO_CONFIG["screener_top_n"]
PORTFOLIO_TICKERS  = PORTFOLIO_CONFIG["manual_tickers"]

OUTPUT_DIR = DATA_DIR   # CSVs written to data/


# ─────────────────────────────────────────────
#  DATA LAYER
# ─────────────────────────────────────────────

def load_tickers() -> list[str]:
    """Load tickers from screener CSV or manual list."""
    if AUTO_READ_SCREENER:
        csv_path = os.path.join(OUTPUT_DIR, "stock_screener_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Filter out ETFs, keep stocks only
            df = df[~df["ticker"].isin(["SPY","QQQ","VTI","SCHD","IWM","DIA"])]
            tickers = df.head(SCREENER_TOP_N)["ticker"].tolist()
            print(f"  📂 Loaded {len(tickers)} tickers from screener: {tickers}")
            return tickers
        else:
            print(f"  ⚠️  screener CSV not found — using manual PORTFOLIO_TICKERS")
    return PORTFOLIO_TICKERS


def fetch_prices(tickers: list[str], years: int = 2) -> pd.DataFrame:
    """Download adjusted closing prices for all tickers."""
    period = f"{years}y"
    print(f"\n  Downloading {len(tickers)} tickers ({years} years of history)...")

    all_prices = {}
    failed = []

    for i, ticker in enumerate(tickers):
        sys.stdout.write(f"\r  [{i+1}/{len(tickers)}] Fetching {ticker:<8}...")
        sys.stdout.flush()
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if df.empty or len(df) < 60:
                failed.append(ticker)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            all_prices[ticker] = df["Close"].squeeze()
        except Exception:
            failed.append(ticker)

    if failed:
        print(f"\n  ⚠️  Could not fetch: {failed}")

    prices = pd.DataFrame(all_prices).dropna(how="all")
    # Drop columns with too many NaNs (< 80% data coverage)
    min_rows = int(len(prices) * 0.8)
    prices = prices.dropna(thresh=min_rows, axis=1)
    # Forward-fill remaining minor gaps, then drop leading NaNs
    prices = prices.ffill().dropna()

    print(f"\n  ✅ Price matrix: {prices.shape[0]} days × {prices.shape[1]} stocks\n")
    return prices


# ─────────────────────────────────────────────
#  RETURN & RISK METRICS
# ─────────────────────────────────────────────

def compute_stats(prices: pd.DataFrame, rf: float, td: int) -> tuple:
    """
    Returns:
      returns      — daily returns DataFrame
      mu           — annualised expected returns (Series)
      cov          — annualised covariance matrix
      corr         — correlation matrix
      vol          — annualised volatility per stock
    """
    returns = prices.pct_change().dropna()
    mu      = returns.mean() * td                         # Annualised expected return
    cov     = returns.cov()  * td                         # Annualised covariance
    corr    = returns.corr()
    vol     = returns.std()  * np.sqrt(td)                # Annualised volatility
    return returns, mu, cov, corr, vol


def portfolio_metrics(weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float) -> tuple:
    """Calculate portfolio annualised return, volatility, and Sharpe ratio."""
    ret  = float(np.dot(weights, mu))
    vol  = float(np.sqrt(weights @ cov.values @ weights))
    sharpe = (ret - rf) / vol if vol > 0 else 0
    return ret, vol, sharpe


# ─────────────────────────────────────────────
#  MONTE CARLO SIMULATION
# ─────────────────────────────────────────────

def monte_carlo(mu, cov, rf, n_sim, td, n_assets) -> pd.DataFrame:
    """
    Simulate n_sim random portfolios.
    Returns DataFrame with columns: return, volatility, sharpe, weights...
    """
    print(f"  Running {n_sim:,} Monte Carlo simulations...")

    results = np.zeros((n_sim, 3 + n_assets))

    for i in range(n_sim):
        w = np.random.dirichlet(np.ones(n_assets))   # Random weights that sum to 1
        ret, vol, sharpe = portfolio_metrics(w, mu, cov, rf)
        results[i, 0] = ret
        results[i, 1] = vol
        results[i, 2] = sharpe
        results[i, 3:] = w

    tickers = list(mu.index)
    cols = ["return", "volatility", "sharpe"] + [f"w_{t}" for t in tickers]
    df = pd.DataFrame(results, columns=cols)
    print(f"  ✅ Monte Carlo complete. Sharpe range: "
          f"{df['sharpe'].min():.2f} → {df['sharpe'].max():.2f}\n")
    return df


# ─────────────────────────────────────────────
#  SCIPY OPTIMISATION
# ─────────────────────────────────────────────

def optimise_max_sharpe(mu, cov, rf, cfg) -> dict:
    """Find the portfolio that maximises the Sharpe ratio."""
    n = len(mu)
    bounds = [(cfg["min_weight"], cfg["max_weight"])] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    init = np.array([1 / n] * n)

    def neg_sharpe(w):
        _, _, s = portfolio_metrics(w, mu, cov, rf)
        return -s

    res = minimize(neg_sharpe, init, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-12})

    w = res.x
    ret, vol, sharpe = portfolio_metrics(w, mu, cov, rf)
    return {"weights": w, "return": ret, "volatility": vol, "sharpe": sharpe,
            "label": "Max Sharpe", "color": "#00E676"}


def optimise_min_vol(mu, cov, rf, cfg) -> dict:
    """Find the minimum volatility portfolio."""
    n = len(mu)
    bounds = [(cfg["min_weight"], cfg["max_weight"])] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    init = np.array([1 / n] * n)

    def portfolio_vol(w):
        return float(np.sqrt(w @ cov.values @ w))

    res = minimize(portfolio_vol, init, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-12})

    w = res.x
    ret, vol, sharpe = portfolio_metrics(w, mu, cov, rf)
    return {"weights": w, "return": ret, "volatility": vol, "sharpe": sharpe,
            "label": "Min Volatility", "color": "#40C4FF"}


def optimise_max_return(mu, cov, rf, cfg) -> dict:
    """Find the maximum return portfolio (within weight constraints)."""
    n = len(mu)
    bounds = [(cfg["min_weight"], cfg["max_weight"])] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    init = np.array([1 / n] * n)

    def neg_return(w):
        return -float(np.dot(w, mu))

    res = minimize(neg_return, init, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-12})

    w = res.x
    ret, vol, sharpe = portfolio_metrics(w, mu, cov, rf)
    return {"weights": w, "return": ret, "volatility": vol, "sharpe": sharpe,
            "label": "Max Return", "color": "#FF6D00"}


def optimise_equal_weight(mu, cov, rf) -> dict:
    """Equal-weight benchmark portfolio."""
    n = len(mu)
    w = np.array([1 / n] * n)
    ret, vol, sharpe = portfolio_metrics(w, mu, cov, rf)
    return {"weights": w, "return": ret, "volatility": vol, "sharpe": sharpe,
            "label": "Equal Weight", "color": "#B0BEC5"}


# ─────────────────────────────────────────────
#  CHARTING
# ─────────────────────────────────────────────

def chart_efficient_frontier(mc_df, portfolios, tickers, mu, vol):
    """Interactive efficient frontier scatter + optimal portfolio markers."""
    fig = go.Figure()

    # Monte Carlo cloud — coloured by Sharpe
    fig.add_trace(go.Scatter(
        x=mc_df["volatility"] * 100,
        y=mc_df["return"] * 100,
        mode="markers",
        marker=dict(
            color=mc_df["sharpe"],
            colorscale="Viridis",
            size=3,
            opacity=0.6,
            colorbar=dict(title="Sharpe Ratio", thickness=14)
        ),
        name="Simulated Portfolios",
        text=[f"Sharpe: {s:.2f}" for s in mc_df["sharpe"]],
        hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<br>%{text}"
    ))

    # Individual stocks
    fig.add_trace(go.Scatter(
        x=vol.values * 100,
        y=mu.values * 100,
        mode="markers+text",
        marker=dict(size=11, color="#F06292", symbol="diamond",
                    line=dict(width=1.5, color="white")),
        text=tickers,
        textposition="top center",
        textfont=dict(size=10),
        name="Individual Stocks",
        hovertemplate="<b>%{text}</b><br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%"
    ))

    # Optimal portfolio markers
    symbols = {"Max Sharpe": "star", "Min Volatility": "pentagon",
               "Max Return": "triangle-up", "Equal Weight": "circle"}
    for p in portfolios:
        fig.add_trace(go.Scatter(
            x=[p["volatility"] * 100],
            y=[p["return"] * 100],
            mode="markers+text",
            marker=dict(size=20, color=p["color"], symbol=symbols.get(p["label"], "star"),
                        line=dict(width=2, color="white")),
            text=[f"  {p['label']}"],
            textposition="middle right",
            textfont=dict(size=11, color=p["color"]),
            name=f"{p['label']} (Sharpe {p['sharpe']:.2f})",
            hovertemplate=(
                f"<b>{p['label']}</b><br>"
                f"Return: {p['return']*100:.1f}%<br>"
                f"Volatility: {p['volatility']*100:.1f}%<br>"
                f"Sharpe: {p['sharpe']:.2f}"
            )
        ))

    fig.update_layout(
        title=dict(text="<b>Efficient Frontier</b> — Modern Portfolio Theory Optimisation",
                   font=dict(size=16)),
        xaxis_title="Annualised Volatility (Risk) %",
        yaxis_title="Annualised Expected Return %",
        template="plotly_dark",
        height=680,
        legend=dict(orientation="v", x=1.01, y=1),
        margin=dict(l=70, r=200, t=80, b=60),
    )

    path = os.path.join(CHARTS_DIR, "efficient_frontier.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"    ✅ Efficient Frontier → {path}")
    return path


def chart_weights(portfolio: dict, tickers: list[str]):
    """Horizontal bar chart of portfolio weights, sorted by weight."""
    weights = portfolio["weights"]
    df = pd.DataFrame({"ticker": tickers, "weight": weights})
    df = df[df["weight"] >= 0.005].sort_values("weight", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["weight"] * 100,
        y=df["ticker"],
        orientation="h",
        marker=dict(
            color=df["weight"],
            colorscale=[[0, "#1a1a2e"], [0.5, "#4a90d9"], [1, "#00E676"]],
            line=dict(color="white", width=0.5)
        ),
        text=[f"{w*100:.1f}%" for w in df["weight"]],
        textposition="outside",
        textfont=dict(size=12)
    ))

    label = portfolio["label"]
    ret   = portfolio["return"] * 100
    vol   = portfolio["volatility"] * 100
    sharpe = portfolio["sharpe"]

    fig.update_layout(
        title=dict(
            text=(f"<b>{label} Portfolio</b> — "
                  f"Return: {ret:.1f}%  |  Risk: {vol:.1f}%  |  Sharpe: {sharpe:.2f}"),
            font=dict(size=14)
        ),
        xaxis_title="Allocation (%)",
        yaxis_title="",
        template="plotly_dark",
        height=max(350, len(df) * 38 + 120),
        margin=dict(l=80, r=100, t=80, b=50),
        xaxis=dict(range=[0, max(df["weight"]) * 130])
    )

    slug = label.lower().replace(" ", "_")
    path = os.path.join(CHARTS_DIR, f"weights_{slug}.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"    ✅ {label} Weights → {path}")
    return path


def chart_correlation(corr: pd.DataFrame):
    """Interactive correlation heatmap with colour-coded cells."""
    tickers = list(corr.columns)
    z = corr.values

    fig = go.Figure(go.Heatmap(
        z=z,
        x=tickers,
        y=tickers,
        colorscale=[
            [0.0,  "#1565C0"],   # Strong negative = blue
            [0.5,  "#FAFAFA"],   # Zero corr = white
            [1.0,  "#B71C1C"],   # Strong positive = red
        ],
        zmin=-1, zmax=1,
        text=np.round(z, 2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="Correlation", thickness=14)
    ))

    fig.update_layout(
        title=dict(text="<b>Pairwise Correlation Heatmap</b> — Diversification View",
                   font=dict(size=15)),
        template="plotly_dark",
        height=max(500, len(tickers) * 38 + 120),
        margin=dict(l=80, r=80, t=80, b=80),
    )

    path = os.path.join(CHARTS_DIR, "correlation_heatmap.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"    ✅ Correlation Heatmap → {path}")
    return path


def chart_risk_return(mu, vol, tickers, portfolios):
    """Risk vs Return scatter for individual stocks + portfolios."""
    fig = go.Figure()

    # Colour by return
    colors = mu.values
    fig.add_trace(go.Scatter(
        x=vol.values * 100,
        y=mu.values * 100,
        mode="markers+text",
        marker=dict(
            size=14,
            color=colors,
            colorscale="RdYlGn",
            line=dict(width=1, color="white"),
            colorbar=dict(title="Return", thickness=12, x=1.02)
        ),
        text=tickers,
        textposition="top center",
        textfont=dict(size=9),
        name="Stocks",
        hovertemplate="<b>%{text}</b><br>Volatility: %{x:.1f}%<br>Return: %{y:.1f}%"
    ))

    # Add zero-return line (risk-free)
    fig.add_hline(y=CONFIG["risk_free_rate"] * 100, line_dash="dot",
                  line_color="yellow", opacity=0.5,
                  annotation_text=f"Risk-Free Rate ({CONFIG['risk_free_rate']*100:.1f}%)",
                  annotation_position="left")

    # Optimal portfolio stars
    for p in portfolios[:3]:
        fig.add_trace(go.Scatter(
            x=[p["volatility"] * 100],
            y=[p["return"] * 100],
            mode="markers",
            marker=dict(size=18, color=p["color"], symbol="star",
                        line=dict(width=2, color="white")),
            name=p["label"],
        ))

    fig.update_layout(
        title="<b>Individual Stock Risk vs Return</b> (Annualised)",
        xaxis_title="Annualised Volatility (Risk) %",
        yaxis_title="Annualised Expected Return %",
        template="plotly_dark",
        height=580,
        margin=dict(l=70, r=100, t=80, b=60),
    )

    path = os.path.join(CHARTS_DIR, "risk_return.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"    ✅ Risk/Return Scatter → {path}")
    return path


# ─────────────────────────────────────────────
#  PRINT RESULTS
# ─────────────────────────────────────────────

def print_portfolio(p: dict, tickers: list[str], label_width: int = 28):
    """Pretty-print a single optimal portfolio."""
    w_pairs = sorted(zip(tickers, p["weights"]), key=lambda x: -x[1])
    print(f"\n  ┌── {p['label']}")
    print(f"  │   Annual Return:    {p['return']*100:>6.2f}%")
    print(f"  │   Annual Volatility:{p['volatility']*100:>6.2f}%")
    print(f"  │   Sharpe Ratio:     {p['sharpe']:>6.3f}")
    print(f"  │")
    print(f"  │   Allocation:")
    for ticker, w in w_pairs:
        if w >= 0.005:
            bar = "█" * int(w * 40)
            print(f"  │     {ticker:<8} {w*100:>5.1f}%  {bar}")
    print(f"  └{'─'*55}")


def save_results(portfolios: list[dict], tickers: list[str], mu, vol):
    """Save all portfolio weights + stock stats to CSV."""
    rows = []

    # Stock-level rows
    for i, ticker in enumerate(tickers):
        rows.append({
            "type": "stock",
            "name": ticker,
            "ann_return_%": round(mu.iloc[i] * 100, 2),
            "ann_volatility_%": round(vol.iloc[i] * 100, 2),
            **{p["label"]: round(p["weights"][i] * 100, 2) for p in portfolios}
        })

    # Portfolio summary rows
    for p in portfolios:
        rows.append({
            "type": "portfolio_summary",
            "name": p["label"],
            "ann_return_%": round(p["return"] * 100, 2),
            "ann_volatility_%": round(p["volatility"] * 100, 2),
            "sharpe": round(p["sharpe"], 3),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "portfolio_results.csv")
    df.to_csv(path, index=False)
    print(f"\n  💾 Results saved → {path}")
    return path


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run_optimizer():
    print(f"\n{'='*65}")
    print(f"  INVESTMENT INTELLIGENCE — Portfolio Optimizer (MPT)")
    print(f"  Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Risk-free rate: {CONFIG['risk_free_rate']*100:.1f}%  "
          f"|  Simulations: {CONFIG['n_simulations']:,}")
    print(f"{'='*65}")

    # ── Step 1: Load tickers & prices ─────────
    tickers = load_tickers()
    prices  = fetch_prices(tickers, CONFIG["history_years"])
    tickers = list(prices.columns)   # update in case some failed
    n       = len(tickers)
    print(f"  Portfolio: {n} assets — {tickers}\n")

    # ── Step 2: Compute stats ─────────────────
    returns, mu, cov, corr, vol = compute_stats(
        prices, CONFIG["risk_free_rate"], CONFIG["trading_days"]
    )

    print("  Individual Stock Statistics:")
    print(f"  {'Ticker':<10} {'Ann.Return':>11} {'Ann.Volatility':>15} {'Sharpe':>8}")
    print(f"  {'─'*48}")
    for ticker in tickers:
        r = mu[ticker]
        v = vol[ticker]
        s = (r - CONFIG["risk_free_rate"]) / v
        print(f"  {ticker:<10} {r*100:>10.1f}%  {v*100:>13.1f}%  {s:>8.2f}")

    # ── Step 3: Monte Carlo ───────────────────
    print()
    mc_df = monte_carlo(mu, cov, CONFIG["risk_free_rate"],
                        CONFIG["n_simulations"], CONFIG["trading_days"], n)

    # ── Step 4: Optimise ──────────────────────
    print("  Running constrained optimisation...")
    p_sharpe  = optimise_max_sharpe(mu, cov, CONFIG["risk_free_rate"], CONFIG)
    p_minvol  = optimise_min_vol(mu, cov, CONFIG["risk_free_rate"], CONFIG)
    p_maxret  = optimise_max_return(mu, cov, CONFIG["risk_free_rate"], CONFIG)
    p_equal   = optimise_equal_weight(mu, cov, CONFIG["risk_free_rate"])
    portfolios = [p_sharpe, p_minvol, p_maxret, p_equal]
    print(f"  ✅ Optimisation complete.\n")

    # ── Step 5: Print results ─────────────────
    print(f"{'='*65}")
    print(f"  OPTIMAL PORTFOLIOS")
    print(f"{'='*65}")
    for p in portfolios:
        print_portfolio(p, tickers)

    # ── Step 6: Comparison table ──────────────
    print(f"\n  {'Portfolio':<22} {'Return':>8} {'Volatility':>12} {'Sharpe':>8}")
    print(f"  {'─'*55}")
    for p in portfolios:
        print(f"  {p['label']:<22} {p['return']*100:>7.1f}%  "
              f"{p['volatility']*100:>10.1f}%  {p['sharpe']:>8.3f}")

    # ── Step 7: Diversification alerts ───────
    print(f"\n  🔍 DIVERSIFICATION INSIGHTS")
    high_corr = []
    t_list = list(corr.columns)
    for i in range(len(t_list)):
        for j in range(i+1, len(t_list)):
            c = corr.iloc[i, j]
            if abs(c) > 0.80:
                high_corr.append((t_list[i], t_list[j], c))
    if high_corr:
        print(f"  ⚠️  Highly correlated pairs (>0.80) — limited diversification benefit:")
        for a, b, c in sorted(high_corr, key=lambda x: -abs(x[2]))[:8]:
            print(f"       {a} ↔ {b}: {c:.2f}")
    else:
        print(f"  ✅ No pair has correlation > 0.80 — good diversification.")

    # ── Step 8: Save & Chart ──────────────────
    print(f"\n  💾 Saving results...")
    save_results(portfolios, tickers, mu, vol)

    print(f"\n  📈 Generating charts...")
    chart_efficient_frontier(mc_df, portfolios, tickers, mu, vol)
    for p in portfolios[:3]:
        chart_weights(p, tickers)
    chart_correlation(corr)
    chart_risk_return(mu, vol, tickers, portfolios)

    print(f"\n{'='*65}")
    print(f"  🏁 OPTIMISATION COMPLETE")
    print(f"  Charts saved to: {CHARTS_DIR}/")
    print(f"\n  RECOMMENDED ACTION:")
    print(f"  → Max Sharpe portfolio offers the best risk-adjusted return.")
    print(f"    Return: {p_sharpe['return']*100:.1f}%  |  "
          f"Risk: {p_sharpe['volatility']*100:.1f}%  |  "
          f"Sharpe: {p_sharpe['sharpe']:.2f}")
    print(f"\n  ⚠️  DISCLAIMER: This is a quantitative model based on historical")
    print(f"     data. Past performance does not guarantee future results.")
    print(f"     Consult a licensed financial advisor before investing.")
    print(f"{'='*65}\n")

    return portfolios, mc_df, tickers, mu, vol, corr


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_optimizer()
