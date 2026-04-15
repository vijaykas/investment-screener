"""
============================================================
  INVESTMENT INTELLIGENCE — PHASE 5
  ML Predictive Layer
  Author: Claude (Vijay's Investment Project)
  Updated: 2026-04-12
============================================================

WHAT THIS DOES:
  Trains three machine learning models on 5 years of historical
  data to predict whether each stock will be HIGHER or LOWER
  at each monthly increment from 1 month to 12 months out.

  MODELS:
    • Random Forest     — robust, handles non-linearity, no scaling needed
    • XGBoost           — gradient boosting, typically best performer
    • Logistic Regression — fast linear baseline
    • Ensemble Voter    — combines all three (soft voting on probabilities)

  PREDICTION HORIZONS (monthly buckets):
    1m (21d) → 2m (42d) → 3m (63d) → ... → 12m (252d)
    Each bucket predicts: will the stock be higher in N trading days?

  FEATURES (45+):
    Price & Returns    — 1/2/3/5/10/20/21/42/63/126/252-day lagged returns
    Moving Averages    — price/SMA ratios (10/20/50/100/200), MA crossovers
    Long-Horizon MAs   — SMA100/200 ratio, quarterly trend, price acceleration
    Momentum           — RSI (7/14/21), Stochastic RSI, ROC (5/10/20d)
    Trend              — MACD, MACD histogram, ADX, +DI, -DI
    Volatility         — Bollinger Band %, ATR ratio, vol regime (short/long)
    52-Week Position   — price location in yearly high/low range
    Volume             — Volume ratio, OBV slope, volume momentum
    Calendar           — day-of-week (sin/cos), month (sin/cos)

  VALIDATION:
    Walk-Forward TimeSeriesSplit (5 folds) — strictly no lookahead.
    Each fold trains on the past and predicts only the future.

  OUTPUTS:
    • Terminal: per-ticker accuracy + current prediction table (all 12 months)
    • ml_predictions.csv           — all predictions + confidence scores
    • ml_model_performance.csv     — per-model accuracy/AUC per ticker × horizon
    • charts/ml_prediction_timeline.html — P(up) curve across all 12 months per stock
    • charts/ml_feature_importance.html  — top features driving predictions
    • charts/ml_roc_curves.html          — ROC curves per model
    • charts/ml_confidence_<Nm>.html     — confidence heatmap per horizon
    • Trained models saved to models/ for reuse in daily_monitor

HOW TO RUN:
  python3 ml_predictor.py

  First run takes ~15–25 min (5y data + 12 horizons × 4 models × all tickers).
  Subsequent runs reuse cached data if CACHE_DATA = True (cache auto-invalidates
  after 12 hours to ensure fresh data each session).

CUSTOMISE:
  Edit ML_CONFIG — tickers, prediction horizons, model parameters.
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import os, sys, json, pickle
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
import xgboost as xgb

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
#  CONFIGURATION  (loaded from config.py)
# ─────────────────────────────────────────────
from config import ML_CONFIG, DATA_DIR, CHARTS_DIR, MODELS_DIR

OUTPUT_DIR = DATA_DIR   # CSVs and cache written to data/


# ─────────────────────────────────────────────
#  DATA LAYER
# ─────────────────────────────────────────────

def load_tickers(cfg: dict) -> list[str]:
    if cfg["auto_read_screener"]:
        csv_path = os.path.join(OUTPUT_DIR, "stock_screener_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df[~df["ticker"].isin(["SPY","QQQ","VTI","SCHD","IWM","DIA"])]
            return df.head(cfg["screener_top_n"])["ticker"].tolist()
    return cfg["tickers"]


def fetch_or_load_prices(tickers: list[str], cfg: dict) -> dict[str, pd.DataFrame]:
    cache_path = os.path.join(OUTPUT_DIR, cfg["cache_file"])

    if cfg["cache_data"] and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        # Refresh if cache is older than 12 hours
        if (datetime.now() - cached["timestamp"]).seconds < 43200:
            print(f"  📦 Using cached price data ({cached['timestamp'].strftime('%H:%M')})")
            return cached["data"]

    print(f"  Downloading {len(tickers)} tickers ({cfg['history_years']}y)...")
    price_data = {}
    period = f"{cfg['history_years']}y"
    for i, ticker in enumerate(tickers):
        sys.stdout.write(f"\r  [{i+1}/{len(tickers)}] {ticker:<8}...")
        sys.stdout.flush()
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if df.empty or len(df) < 150:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            price_data[ticker] = df.copy()
        except Exception:
            pass

    if cfg["cache_data"]:
        with open(cache_path, "wb") as f:
            pickle.dump({"timestamp": datetime.now(), "data": price_data}, f)
    print(f"\n  ✅ Loaded {len(price_data)} tickers.\n")
    return price_data


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 30+ predictive features from OHLCV data.
    All features are computed using only past data — no lookahead.
    """
    feat = pd.DataFrame(index=df.index)
    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    vol    = df["Volume"].squeeze()
    open_  = df["Open"].squeeze()

    # ── Lagged Returns (momentum/mean-reversion features) ─────────
    for n in [1, 2, 3, 5, 10, 20]:
        feat[f"ret_{n}d"] = close.pct_change(n)

    # ── Price vs Moving Averages (trend) ──────────────────────────
    for w in [10, 20, 50, 100, 200]:
        sma = ta.trend.SMAIndicator(close=close, window=w).sma_indicator()
        feat[f"price_sma{w}_ratio"] = (close / sma) - 1

    # EMA ratios
    for w in [12, 26]:
        ema = ta.trend.EMAIndicator(close=close, window=w).ema_indicator()
        feat[f"price_ema{w}_ratio"] = (close / ema) - 1

    # MA crossover: SMA50 / SMA200 ratio (captures golden/death cross)
    sma50  = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
    sma200 = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
    feat["sma50_200_ratio"] = (sma50 / sma200) - 1

    # ── RSI & Stochastic RSI ──────────────────────────────────────
    rsi_14 = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    rsi_7  = ta.momentum.RSIIndicator(close=close, window=7).rsi()
    feat["rsi_14"]      = rsi_14 / 100
    feat["rsi_7"]       = rsi_7 / 100
    feat["rsi_14_lag1"] = rsi_14.shift(1) / 100
    feat["rsi_delta"]   = (rsi_14 - rsi_14.shift(1)) / 100   # RSI velocity

    stoch = ta.momentum.StochRSIIndicator(close=close, window=14)
    feat["stoch_rsi_k"] = stoch.stochrsi_k()
    feat["stoch_rsi_d"] = stoch.stochrsi_d()

    # ── MACD ─────────────────────────────────────────────────────
    macd_ind   = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    macd_line  = macd_ind.macd()
    macd_sig   = macd_ind.macd_signal()
    macd_hist  = macd_ind.macd_diff()
    feat["macd_norm"]      = macd_line / close             # Normalise by price
    feat["macd_signal_norm"] = macd_sig / close
    feat["macd_hist_norm"] = macd_hist / close
    feat["macd_hist_delta"]= macd_hist.diff() / close      # Histogram momentum
    feat["macd_above_sig"] = (macd_line > macd_sig).astype(int)

    # ── ADX & Directional Movement ────────────────────────────────
    adx_ind = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
    feat["adx"]       = adx_ind.adx() / 100
    feat["adx_pos_di"]= adx_ind.adx_pos() / 100
    feat["adx_neg_di"]= adx_ind.adx_neg() / 100
    feat["di_diff"]   = (adx_ind.adx_pos() - adx_ind.adx_neg()) / 100

    # ── Bollinger Bands ───────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    feat["bb_pband"]  = bb.bollinger_pband()               # 0=lower, 1=upper
    feat["bb_wband"]  = bb.bollinger_wband() / close       # Bandwidth (volatility)
    # BB width z-score (squeeze detection)
    bb_w = bb.bollinger_wband()
    feat["bb_squeeze"]= (bb_w - bb_w.rolling(50).mean()) / bb_w.rolling(50).std()

    # ── ATR — Volatility Regime ───────────────────────────────────
    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    feat["atr_ratio"]  = atr / close                       # ATR as % of price
    feat["atr_delta"]  = atr.pct_change()                  # Changing volatility

    # ── Volume Features ───────────────────────────────────────────
    vol_sma20       = vol.rolling(20).mean()
    feat["vol_ratio"]    = vol / vol_sma20                 # vs 20d avg
    feat["vol_ratio_lag"]= (vol / vol_sma20).shift(1)
    feat["vol_trend"]    = vol_sma20.pct_change(5)         # Volume trend

    # OBV normalised slope
    obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=vol).on_balance_volume()
    feat["obv_slope"] = obv.diff(5) / (close * vol_sma20 + 1e-8)

    # ── Rate of Change ────────────────────────────────────────────
    for n in [5, 10, 20]:
        feat[f"roc_{n}d"] = ta.momentum.ROCIndicator(close=close, window=n).roc() / 100

    # ── High/Low Range Features ───────────────────────────────────
    feat["hl_ratio"]  = (high - low) / close               # Daily range / price
    feat["close_pos"] = (close - low) / (high - low + 1e-8)# Where close sits in day range

    # ── Calendar Cyclical Encoding ────────────────────────────────
    feat["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 5)
    feat["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 5)
    feat["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    # ── Long-Horizon Features (essential for 1–12 month targets) ──
    # Extended lagged returns — monthly/quarterly/semi-annual/annual
    for n in [21, 42, 63, 126, 252]:
        feat[f"ret_{n}d"] = close.pct_change(n)

    # RSI-21 (smoother, less noisy than 14 for longer horizons)
    feat["rsi_21"] = ta.momentum.RSIIndicator(close=close, window=21).rsi() / 100

    # 52-week high/low range position — where is price in yearly range?
    roll_high_252 = close.rolling(252, min_periods=126).max()
    roll_low_252  = close.rolling(252, min_periods=126).min()
    feat["52w_range_pos"] = (close - roll_low_252) / (roll_high_252 - roll_low_252 + 1e-8)

    # SMA100 / SMA200 ratio — longer-term trend alignment
    sma100 = ta.trend.SMAIndicator(close=close, window=100).sma_indicator()
    feat["sma100_200_ratio"] = (sma100 / sma200) - 1          # sma200 already computed above

    # Quarterly momentum: 3-month return vs 6-month return
    ret_3m = close.pct_change(63)
    ret_6m = close.pct_change(126)
    feat["momentum_3m_vs_6m"] = ret_3m - ret_6m               # recent acceleration

    # Price acceleration: 1-month return vs 3-month return
    ret_1m = close.pct_change(21)
    feat["momentum_1m_vs_3m"] = ret_1m - ret_3m

    # Volatility regime: short-term ATR (14d) vs long-term ATR (63d)
    atr_63 = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=63).average_true_range()
    feat["vol_regime"] = (atr / close) / (atr_63 / close + 1e-8)  # >1 = elevated short-term vol

    return feat


def build_targets(df: pd.DataFrame, horizons: dict) -> pd.DataFrame:
    """
    Create binary targets: 1 if price is higher in N days, else 0.
    Uses future prices — these rows are dropped from training features
    to prevent lookahead.
    """
    close = df["Close"].squeeze()
    targets = pd.DataFrame(index=df.index)
    for name, n in horizons.items():
        future_ret = close.shift(-n) / close - 1
        targets[f"target_{name}"] = (future_ret > 0).astype(int)
    return targets


def prepare_dataset(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine features + targets, drop NaN rows.
    Returns (X, y) aligned DataFrames.
    """
    features = engineer_features(df)
    targets  = build_targets(df, cfg["horizons"])

    # Align and drop any rows with NaNs
    combined = pd.concat([features, targets], axis=1).dropna()
    feature_cols = features.columns.tolist()
    target_cols  = [f"target_{h}" for h in cfg["horizons"].keys()]

    X = combined[feature_cols]
    y = combined[target_cols]
    return X, y


# ─────────────────────────────────────────────
#  MODEL DEFINITIONS
# ─────────────────────────────────────────────

def build_models(cfg: dict) -> dict:
    """Return dict of named model pipelines."""
    rf = RandomForestClassifier(**cfg["rf_params"])
    xgb_clf = xgb.XGBClassifier(**cfg["xgb_params"])

    # Logistic Regression needs scaling
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(**cfg["lr_params"]))
    ])

    # Ensemble: RF + XGB vote, LR scaled separately inside pipeline
    # We wrap LR so all estimators in VotingClassifier get raw X
    lr_raw = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(**cfg["lr_params"]))
    ])
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb_clf), ("lr", lr_raw)],
        voting="soft",
        n_jobs=-1,
    )

    return {
        "Random Forest":        RandomForestClassifier(**cfg["rf_params"]),
        "XGBoost":              xgb.XGBClassifier(**cfg["xgb_params"]),
        "Logistic Regression":  lr,
        "Ensemble":             ensemble,
    }


# ─────────────────────────────────────────────
#  WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────

def walk_forward_cv(model, X: pd.DataFrame, y: pd.Series, cfg: dict) -> dict:
    """
    Strict temporal cross-validation — train on past, predict future.
    Returns aggregated metrics across all folds.
    """
    tscv = TimeSeriesSplit(n_splits=cfg["cv_folds"], gap=5)
    metrics = {"accuracy":[], "precision":[], "recall":[], "f1":[], "auc":[]}
    oof_probs = np.zeros(len(X))
    oof_true  = y.values

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < cfg["min_train_size"]:
            continue
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # Skip fold if only one class in training data
        if y_tr.nunique() < 2:
            continue

        try:
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            probs = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else preds

            metrics["accuracy"].append(accuracy_score(y_te, preds))
            metrics["precision"].append(precision_score(y_te, preds, zero_division=0))
            metrics["recall"].append(recall_score(y_te, preds, zero_division=0))
            metrics["f1"].append(f1_score(y_te, preds, zero_division=0))
            if y_te.nunique() > 1:
                metrics["auc"].append(roc_auc_score(y_te, probs))
            oof_probs[test_idx] = probs
        except Exception:
            continue

    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


# ─────────────────────────────────────────────
#  CURRENT PREDICTION
# ─────────────────────────────────────────────

def predict_current(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train on all available data, predict on the most recent row.
    Returns probability of price being higher for each horizon.
    """
    # Train on everything except the last row
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]

    if len(X_train) < 50 or y_train.nunique() < 2:
        return {}

    model.fit(X_train, y_train)
    X_latest = X.iloc[[-1]]

    if hasattr(model, "predict_proba"):
        prob_up = float(model.predict_proba(X_latest)[0, 1])
    else:
        prob_up = float(model.predict(X_latest)[0])

    direction = "UP 📈" if prob_up >= 0.55 else ("DOWN 📉" if prob_up <= 0.45 else "NEUTRAL ➡️")
    confidence = abs(prob_up - 0.5) * 200  # 0–100 scale

    return {
        "prob_up":    round(prob_up * 100, 1),
        "direction":  direction,
        "confidence": round(confidence, 1),
    }


# ─────────────────────────────────────────────
#  FEATURE IMPORTANCE (RF + XGB)
# ─────────────────────────────────────────────

def get_feature_importance(rf_model, xgb_model, feature_names: list) -> pd.DataFrame:
    """Average feature importance across RF and XGB."""
    try:
        rf_imp  = rf_model.feature_importances_
        xgb_imp = xgb_model.feature_importances_
        avg_imp = (rf_imp + xgb_imp) / 2
        return pd.DataFrame({
            "feature":    feature_names,
            "rf_imp":     rf_imp,
            "xgb_imp":    xgb_imp,
            "avg_imp":    avg_imp,
        }).sort_values("avg_imp", ascending=False)
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────

def chart_feature_importance(fi_df: pd.DataFrame, ticker: str, horizon: str):
    top = fi_df.head(ML_CONFIG["top_features_chart"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top["rf_imp"], y=top["feature"], orientation="h",
        name="Random Forest", marker_color="#00E676", opacity=0.75
    ))
    fig.add_trace(go.Bar(
        x=top["xgb_imp"], y=top["feature"], orientation="h",
        name="XGBoost", marker_color="#40C4FF", opacity=0.75
    ))

    fig.update_layout(
        title=f"<b>Feature Importance</b> — {ticker} | {horizon} horizon",
        xaxis_title="Importance", yaxis_title="",
        barmode="group", template="plotly_dark",
        height=max(500, len(top) * 24 + 150),
        margin=dict(l=200, r=60, t=80, b=60),
        legend=dict(orientation="h", y=1.05),
    )
    fig.update_yaxes(autorange="reversed")
    path = os.path.join(CHARTS_DIR, "ml_feature_importance.html")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def chart_roc_curves(roc_data: dict):
    """ROC curves for each model."""
    fig = go.Figure()
    colors = {"Random Forest": "#00E676", "XGBoost": "#40C4FF",
              "Logistic Regression": "#FFD740", "Ensemble": "#EA80FC"}

    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines",
        line=dict(color="gray", dash="dash", width=1),
        name="Random Chance", showlegend=True
    ))

    for model_name, (fpr, tpr, auc_val) in roc_data.items():
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{model_name}  (AUC = {auc_val:.3f})",
            line=dict(color=colors.get(model_name, "#fff"), width=2)
        ))

    fig.update_layout(
        title="<b>ROC Curves</b> — Model Discrimination Power",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark", height=520,
        legend=dict(x=0.6, y=0.1),
        margin=dict(l=70, r=60, t=80, b=60),
    )
    path = os.path.join(CHARTS_DIR, "ml_roc_curves.html")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def chart_confidence_heatmap(pred_df: pd.DataFrame, horizon: str):
    """Heatmap of prediction probabilities: tickers × model."""
    models = ["Random Forest", "XGBoost", "Logistic Regression", "Ensemble"]
    tickers = pred_df["ticker"].tolist()

    z = []
    for _, row in pred_df.iterrows():
        z.append([row.get(f"prob_up_{m.replace(' ','_')}_{horizon}", 50) for m in models])

    z = np.array(z, dtype=float)
    text = [[f"{v:.0f}%" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=models, y=tickers,
        colorscale=[
            [0.0, "#B71C1C"], [0.35, "#EF9A9A"],
            [0.5, "#37474F"],
            [0.65, "#A5D6A7"], [1.0, "#1B5E20"],
        ],
        zmid=50, zmin=20, zmax=80,
        text=text, texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="P(Up) %", thickness=14),
    ))
    fig.update_layout(
        title=f"<b>Prediction Confidence Heatmap</b> — P(Price Higher) in {horizon}",
        template="plotly_dark",
        height=max(300, len(tickers) * 30 + 160),
        margin=dict(l=90, r=80, t=80, b=80),
    )
    path = os.path.join(CHARTS_DIR, f"ml_confidence_{horizon}.html")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def chart_model_accuracy(perf_df: pd.DataFrame):
    """Bar chart of model accuracy across all tickers."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Accuracy by Model", "ROC-AUC by Model"],
        horizontal_spacing=0.12
    )
    model_names = perf_df["model"].unique().tolist()
    colors = {"Random Forest": "#00E676", "XGBoost": "#40C4FF",
              "Logistic Regression": "#FFD740", "Ensemble": "#EA80FC"}

    for metric, col in [("accuracy", 1), ("auc", 2)]:
        summary = perf_df.groupby("model")[metric].mean().reindex(model_names)
        fig.add_trace(go.Bar(
            x=model_names, y=summary.values,
            marker_color=[colors.get(m, "#fff") for m in model_names],
            text=[f"{v:.3f}" for v in summary.values],
            textposition="outside",
            showlegend=False,
        ), row=1, col=col)

    fig.add_hline(y=0.5, line_dash="dot", line_color="red",
                  annotation_text="Random baseline", row=1, col=1)

    fig.update_layout(
        title="<b>Model Performance Summary</b> (avg across all tickers & horizons)",
        template="plotly_dark", height=450,
        margin=dict(l=60, r=60, t=80, b=80),
    )
    path = os.path.join(CHARTS_DIR, "ml_model_accuracy.html")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def chart_prediction_timeline(pred_df: pd.DataFrame, cfg: dict):
    """
    Line chart showing P(Up) across all 12 monthly horizons for each ticker.
    One line per ticker, x-axis = horizon (1m–12m), y-axis = Ensemble P(Up)%.
    Makes the multi-month outlook visually scannable at a glance.
    """
    horizons = list(cfg["horizons"].keys())        # ["1m","2m",...,"12m"]
    months   = [int(h[:-1]) for h in horizons]    # [1,2,...,12]

    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
    fig = go.Figure()

    # Reference band: 45–55% = neutral zone
    fig.add_hrect(y0=45, y1=55, fillcolor="rgba(100,100,100,0.15)",
                  line_width=0, annotation_text="Neutral zone",
                  annotation_position="top right")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", line_width=1)

    for i, (_, row) in enumerate(pred_df.iterrows()):
        ticker = row["ticker"]
        y_vals = []
        for h in horizons:
            key = f"prob_up_Ensemble_{h}"
            y_vals.append(row.get(key, np.nan))

        # Only plot if we have at least half the horizons
        if sum(~np.isnan(v) for v in y_vals) < len(horizons) // 2:
            continue

        # Final direction arrow annotation
        last_valid = next((v for v in reversed(y_vals) if not np.isnan(v)), None)
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=months, y=y_vals,
            mode="lines+markers",
            name=ticker,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate=(
                f"<b>{ticker}</b><br>"
                "Horizon: %{x}m<br>"
                "P(Up): %{y:.1f}%<extra></extra>"
            ),
        ))

        # Annotate endpoint
        if last_valid is not None:
            fig.add_annotation(
                x=12, y=last_valid,
                text=f" {ticker}",
                showarrow=False, xanchor="left",
                font=dict(color=color, size=10),
            )

    fig.update_layout(
        title="<b>12-Month Prediction Timeline</b> — Ensemble P(Price Higher) per Stock",
        xaxis=dict(
            title="Months Ahead",
            tickmode="array",
            tickvals=months,
            ticktext=[f"{m}m" for m in months],
            gridcolor="#2a2a2a",
        ),
        yaxis=dict(
            title="P(Up) %", range=[20, 80],
            tickformat=".0f", ticksuffix="%",
            gridcolor="#2a2a2a",
        ),
        template="plotly_dark",
        height=550,
        margin=dict(l=70, r=120, t=80, b=70),
        legend=dict(
            orientation="v", x=1.01, y=1,
            bordercolor="#444", borderwidth=1,
            font=dict(size=10),
        ),
        hovermode="x unified",
    )

    path = os.path.join(CHARTS_DIR, "ml_prediction_timeline.html")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


# ─────────────────────────────────────────────
#  MAIN RUNNER
# ─────────────────────────────────────────────

def run_ml_predictor():
    cfg     = ML_CONFIG
    tickers = load_tickers(cfg)

    print(f"\n{'='*70}")
    print(f"  INVESTMENT INTELLIGENCE — ML Predictive Layer (Phase 5)")
    print(f"  Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Tickers: {len(tickers)}  |  Horizons: {list(cfg['horizons'].keys())}")
    print(f"  Models: Random Forest, XGBoost, Logistic Regression, Ensemble")
    print(f"  Validation: {cfg['cv_folds']}-fold Walk-Forward TimeSeriesSplit")
    print(f"{'='*70}\n")

    # ── Download prices ───────────────────────
    price_data = fetch_or_load_prices(tickers, cfg)
    tickers = [t for t in tickers if t in price_data]

    all_perf_rows   = []   # {ticker, model, horizon, accuracy, ...}
    all_pred_rows   = []   # {ticker, horizon, model, prob_up, direction, confidence}
    roc_data        = {}   # For ROC chart: {model_name: (fpr, tpr, auc)}
    fi_df_global    = None
    trained_rf      = None
    trained_xgb     = None

    # ── Train per ticker × horizon × model ───
    for t_idx, ticker in enumerate(tickers):
        print(f"\n  [{t_idx+1}/{len(tickers)}] Training: {ticker}")
        df = price_data[ticker]

        X, y_all = prepare_dataset(df, cfg)
        if len(X) < cfg["min_train_size"] + 50:
            print(f"      ⚠️  Insufficient data ({len(X)} rows), skipping.")
            continue

        pred_row = {"ticker": ticker}

        for horizon_name in cfg["horizons"].keys():
            target_col = f"target_{horizon_name}"
            if target_col not in y_all.columns:
                continue
            y = y_all[target_col]

            sys.stdout.write(f"\r      Horizon {horizon_name}...")
            sys.stdout.flush()

            models = build_models(cfg)

            for model_name, model in models.items():
                # Walk-forward CV
                cv_metrics = walk_forward_cv(model, X, y, cfg)

                all_perf_rows.append({
                    "ticker":    ticker,
                    "model":     model_name,
                    "horizon":   horizon_name,
                    **cv_metrics
                })

                # Current prediction (train on full data)
                model_fresh = build_models(cfg)[model_name]
                pred = predict_current(model_fresh, X, y)
                if pred:
                    key_suffix = f"{model_name.replace(' ','_')}_{horizon_name}"
                    pred_row[f"prob_up_{key_suffix}"]    = pred["prob_up"]
                    pred_row[f"direction_{key_suffix}"]  = pred["direction"]
                    pred_row[f"confidence_{key_suffix}"] = pred["confidence"]

                # Save trained RF & XGB on 3m horizon for feature importance
                if horizon_name == "3m" and model_name in ("Random Forest", "XGBoost"):
                    model_to_save = build_models(cfg)[model_name]
                    model_to_save.fit(X.iloc[:-1], y.iloc[:-1])
                    if model_name == "Random Forest":
                        trained_rf = model_to_save
                        model_path = os.path.join(MODELS_DIR, f"{ticker}_rf.pkl")
                        with open(model_path, "wb") as f:
                            pickle.dump(model_to_save, f)
                    else:
                        trained_xgb = model_to_save

            # ROC curve data (use Ensemble on 3m horizon for the last ticker)
            if horizon_name == "3m" and t_idx == len(tickers) - 1:
                for mname, model in build_models(cfg).items():
                    try:
                        tscv = TimeSeriesSplit(n_splits=3, gap=5)
                        all_probs, all_true = [], []
                        for train_idx, test_idx in tscv.split(X):
                            if len(train_idx) < 100:
                                continue
                            model.fit(X.iloc[train_idx], y.iloc[train_idx])
                            probs = model.predict_proba(X.iloc[test_idx])[:, 1]
                            all_probs.extend(probs)
                            all_true.extend(y.iloc[test_idx].values)
                        if len(set(all_true)) > 1:
                            fpr, tpr, _ = roc_curve(all_true, all_probs)
                            auc_val = roc_auc_score(all_true, all_probs)
                            roc_data[mname] = (fpr, tpr, auc_val)
                    except Exception:
                        pass

        all_pred_rows.append(pred_row)
        print(f"\r      ✅ {ticker} done ({len(X)} samples, {len(X.columns)} features)")

    # ── Feature importance (last trained RF + XGB) ────
    if trained_rf is not None and trained_xgb is not None:
        fi_df_global = get_feature_importance(trained_rf, trained_xgb, X.columns.tolist())

    # ── Performance summary ───────────────────
    perf_df = pd.DataFrame(all_perf_rows)
    pred_df = pd.DataFrame(all_pred_rows)

    print(f"\n\n{'='*75}")
    print(f"  MODEL PERFORMANCE (avg accuracy & AUC across all tickers)")
    print(f"{'='*75}")
    print(f"  {'Model':<22} {'Horizon':<8} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>6} {'AUC':>8}")
    print(f"  {'─'*72}")

    for horizon_name in cfg["horizons"].keys():
        h_df = perf_df[perf_df["horizon"] == horizon_name]
        for model_name in ["Random Forest", "XGBoost", "Logistic Regression", "Ensemble"]:
            m_df = h_df[h_df["model"] == model_name]
            if m_df.empty:
                continue
            acc  = m_df["accuracy"].mean()
            prec = m_df["precision"].mean()
            rec  = m_df["recall"].mean()
            f1   = m_df["f1"].mean()
            auc  = m_df["auc"].mean()
            print(f"  {model_name:<22} {horizon_name:<8} {acc:>8.3f}  {prec:>9.3f}  "
                  f"{rec:>7.3f}  {f1:>5.3f}  {auc:>7.3f}")
        print()

    # ── Current predictions — multi-horizon table ─────────────────
    key_horizons = ["1m", "3m", "6m", "12m"]
    col_w = 10
    print(f"\n{'='*75}")
    print(f"  CURRENT PREDICTIONS — Ensemble | P(Up)% across 1m → 12m horizons")
    print(f"{'='*75}")
    header = f"  {'Ticker':<10}" + "".join(f"{h:>{col_w}}" for h in key_horizons) + f"  {'6m Direction':<16} {'6m Conf':>8}"
    print(header)
    print(f"  {'─'*72}")

    pred_rows_sorted = sorted(
        all_pred_rows,
        key=lambda r: r.get("prob_up_Ensemble_6m", 50),
        reverse=True
    )
    for row in pred_rows_sorted:
        ticker = row["ticker"]
        probs  = [row.get(f"prob_up_Ensemble_{h}", None) for h in key_horizons]
        if all(p is None for p in probs):
            continue
        dirn = row.get("direction_Ensemble_6m", "—")
        conf = row.get("confidence_Ensemble_6m", 0)
        prob_strs = "".join(
            f"{p:>{col_w}.1f}%" if p is not None else f"{'—':>{col_w}}"
            for p in probs
        )
        print(f"  {ticker:<10}{prob_strs}  {dirn:<16}  {conf:>6.1f}%")

    # ── Top feature importance ────────────────
    if fi_df_global is not None and not fi_df_global.empty:
        print(f"\n  TOP 10 PREDICTIVE FEATURES (RF + XGB avg | 3m horizon):")
        print(f"  {'Feature':<30} {'Importance':>10}")
        print(f"  {'─'*44}")
        for _, row in fi_df_global.head(10).iterrows():
            bar = "█" * int(row["avg_imp"] * 400)
            print(f"  {row['feature']:<30} {row['avg_imp']:>9.4f}  {bar}")

    # ── Save CSVs ─────────────────────────────
    perf_path = os.path.join(OUTPUT_DIR, "ml_model_performance.csv")
    perf_df.to_csv(perf_path, index=False)
    print(f"\n  💾 Model performance → {perf_path}")

    pred_path = os.path.join(OUTPUT_DIR, "ml_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"  💾 Predictions       → {pred_path}")

    # ── Generate charts ───────────────────────
    print(f"\n  📈 Generating charts...")

    if fi_df_global is not None and not fi_df_global.empty:
        path = chart_feature_importance(fi_df_global, tickers[-1], "3m")
        print(f"    ✅ Feature Importance    → {path}")

    if roc_data:
        path = chart_roc_curves(roc_data)
        print(f"    ✅ ROC Curves            → {path}")

    if not pred_df.empty:
        path = chart_confidence_heatmap(pred_df, "6m")
        print(f"    ✅ Confidence Heatmap    → {path}")

    if not perf_df.empty:
        path = chart_model_accuracy(perf_df)
        print(f"    ✅ Model Accuracy        → {path}")

    if not pred_df.empty:
        path = chart_prediction_timeline(pred_df, cfg)
        print(f"    ✅ Prediction Timeline   → {path}")

    print(f"\n{'='*70}")
    print(f"  🏁 ML PREDICTIVE LAYER COMPLETE")
    print(f"  Trained on {len(tickers)} tickers × {len(cfg['horizons'])} horizons (1m–12m) × 4 models")
    print(f"  Models cached → {MODELS_DIR}/")
    print(f"\n  KEY CHARTS:")
    print(f"  • ml_prediction_timeline.html — P(Up) curves across all 12 months per stock")
    print(f"  • ml_confidence_6m.html       — 6-month confidence heatmap (all tickers × models)")
    print(f"  • ml_feature_importance.html  — top 25 features driving 3m predictions")
    print(f"\n  INTEGRATION TIP:")
    print(f"  Add 'import pickle' + load models/<ticker>_rf.pkl in daily_monitor.py")
    print(f"  to include ML predictions in your morning report automatically.")
    print(f"\n  ⚠️  ML predictions are probabilistic, not certainties. Always combine")
    print(f"     with fundamental research and risk management before trading.")
    print(f"{'='*70}\n")

    return perf_df, pred_df


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_ml_predictor()
