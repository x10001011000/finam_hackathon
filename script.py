#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import lightgbm as lgb
from lightgbm import LGBMRegressor

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except Exception:
    import nltk
    nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

from pykalman import KalmanFilter

load_dotenv()

NEWS_PATH = Path(os.getenv("NEWS_PATH", "ticker_agg_new_data.csv"))
CANDLES_PATH = Path(os.getenv("CANDLES_PATH", "candles.csv"))
OUT_DIR = Path(os.getenv("OUT_DIR", "unified_output"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

SBERT_MODEL_NAME = os.getenv("SBERT_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TEST_RATIO = float(os.getenv("TEST_RATIO", "0.27"))
MAX_HORIZON = int(os.getenv("MAX_HORIZON", "20"))

def parse_aggregated_news(news_df, text_col="all_news_text", ticker_col="ticker"):
    rows = []
    ts_pattern = re.compile(r"(\[\s*\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\s*\])")
    for r in tqdm(news_df.itertuples(index=False), total=len(news_df), desc="parsing news"):
        text = getattr(r, text_col, None)
        ticker = getattr(r, ticker_col, None)
        if not isinstance(text, str) or not text.strip():
            continue
        matches = list(ts_pattern.finditer(text))
        if matches:
            for i, m in enumerate(matches):
                start = matches[i-1].end() if i > 0 else 0
                end = m.start()
                snippet = text[start:end].strip(" \t\n\r;:-—")
                ts_str = m.group(0).strip("[]")
                try:
                    dt = pd.to_datetime(ts_str)
                except Exception:
                    dt = pd.NaT
                if not snippet:
                    next_start = m.end()
                    next_end = matches[i+1].start() if i+1 < len(matches) else len(text)
                    snippet = text[next_start:next_end].strip(" \t\n\r;:-—")
                if snippet:
                    rows.append({"ticker": ticker, "news_text": snippet, "publish_date": dt})
            continue
        # fallback: split by paragraphs
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(parts) > 1:
            dt_for_all = None
            rows.extend([{"ticker": ticker, "news_text": p, "publish_date": dt_for_all} for p in parts])
            continue
        rows.append({"ticker": ticker, "news_text": text.strip(), "publish_date": pd.NaT})
    exploded = pd.DataFrame(rows)
    if not exploded.empty:
        exploded["publish_date"] = pd.to_datetime(exploded["publish_date"], errors="coerce")
    return exploded

def encode_long_text_avg(text, model, chunk_sentences=20):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(model.get_sentence_embedding_dimension(), dtype=float)
    sents = sent_tokenize(text)
    if len(sents) == 0:
        return np.zeros(model.get_sentence_embedding_dimension(), dtype=float)
    chunks = [" ".join(sents[i:i+chunk_sentences]) for i in range(0, len(sents), chunk_sentences)]
    embs = model.encode(chunks, show_progress_bar=False)
    return np.mean(embs, axis=0)

def create_news_features(news_exploded_df, original_news_df, model_name=SBERT_MODEL_NAME):
    if news_exploded_df is None or news_exploded_df.empty:
        df = original_news_df.copy()
        if "all_news_text" not in df.columns:
            return pd.DataFrame(columns=["ticker", "publish_date"])
        df = df.rename(columns={"all_news_text": "news_text"})
        if "first_date" in df.columns:
            df["publish_date"] = pd.to_datetime(df["first_date"], errors="coerce")
        elif "last_date" in df.columns:
            df["publish_date"] = pd.to_datetime(df["last_date"], errors="coerce")
        else:
            df["publish_date"] = pd.NaT
        df = df[["ticker", "news_text", "publish_date"]].copy()
    else:
        df = news_exploded_df[["ticker", "news_text", "publish_date"]].copy()
    df["news_text"] = df["news_text"].astype(str)
    df = df[df["news_text"].str.strip() != ""].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "publish_date"])
    model = None
    if SentenceTransformer is not None:
        try:
            model = SentenceTransformer(model_name)
        except Exception:
            model = None
    if model is not None:
        embs = []
        for txt in tqdm(df["news_text"].tolist(), desc="encoding news"):
            if len(txt) > 2000:
                e = encode_long_text_avg(txt, model)
            else:
                e = model.encode(txt, show_progress_bar=False)
            embs.append(e)
        emb_arr = np.vstack(embs)
        emb_cols = [f"emb_{i}" for i in range(emb_arr.shape[1])]
        emb_df = pd.DataFrame(emb_arr, columns=emb_cols)
        news_emb = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
    else:
        vect = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
        X = vect.fit_transform(df["news_text"].tolist())
        svd = TruncatedSVD(n_components=128, random_state=42)
        S = svd.fit_transform(X)
        emb_cols = [f"emb_{i}" for i in range(S.shape[1])]
        emb_df = pd.DataFrame(S, columns=emb_cols)
        news_emb = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
    news_emb["publish_date"] = pd.to_datetime(news_emb["publish_date"], errors="coerce").dt.date
    emb_cols = [c for c in news_emb.columns if c.startswith("emb_")]
    agg_funcs = {c: ["mean", "std"] for c in emb_cols}
    agg_funcs["news_text"] = "count"
    agg = news_emb.groupby(["ticker", "publish_date"]).agg(agg_funcs).reset_index()
    agg.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) and col[1] != "" else col[0] for col in agg.columns]
    if "news_text_count" in agg.columns:
        agg = agg.rename(columns={"news_text_count": "daily_news_count"})
    return agg

def kalman_high_frequency(prices, Q_multiplier=0.001):
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=prices[0],
        initial_state_covariance=1,
        transition_covariance=0.01 * Q_multiplier,
        observation_covariance=0.1
    )
    state_means, _ = kf.filter(prices)
    return state_means.flatten()

def create_technical_features(candles_df):
    df = candles_df.copy().sort_values(['ticker', 'begin']).reset_index(drop=True)
    close_prices = df['close'].values
    df["close"] = kalman_high_frequency(close_prices, Q_multiplier=0.001)
    df['year'] = df["begin"].dt.year
    df['month'] = df["begin"].dt.month
    df['day'] = df["begin"].dt.day
    df['dayofweek'] = df["begin"].dt.dayofweek
    for window in (3,5,10):
        df[f'volatility_{window}'] = df.groupby('ticker')['close'].transform(lambda x: x.pct_change().rolling(window).std())
        df[f'momentum_{window}'] = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(window))
        df[f'ma_{window}'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window).mean())
        df[f'distance_from_ma_{window}'] = (df['close'] - df[f'ma_{window}']) / df[f'ma_{window}']
    for lag in (1,2,3):
        df[f'close_lag_{lag}'] = df.groupby('ticker')['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df.groupby('ticker')['volume'].shift(lag)
    df['volume_ema_5'] = df.groupby('ticker')['volume'].transform(lambda x: x.ewm(span=5).mean())
    df['price_range'] = df['high'] - df['low']
    df['price_range_pct'] = df['price_range'] / df['close']
    df['volume_vs_avg'] = df.groupby('ticker')['volume'].transform(lambda x: x / x.rolling(10).mean())
    df["return_1d"] = df.groupby("ticker")["close"].pct_change()
    df['body_size_rel'] = (df['close'] - df['open']) / df['open']
    df['wick_size_rel'] = (df['high'] - df['low']) / df['open']
    df["vol_log"] = np.log1p(df["volume"])
    df["vol_lag_1"] = df.groupby("ticker")["vol_log"].shift(1)
    for w in (3,7,14):
        g = df.groupby("ticker")
        df[f"vol_roll_mean_{w}"] = g["vol_log"].shift(1).rolling(w).mean().reset_index(level=0, drop=True)
        df[f"return_roll_std_{w}"] = g["return_1d"].shift(1).rolling(w).std().reset_index(level=0, drop=True)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    dow = pd.get_dummies(df['dayofweek'], prefix='dow')
    df = pd.concat([df, dow], axis=1)
    df['date'] = df['begin'].dt.date
    return df

def add_target_columns(df, max_horizon=MAX_HORIZON):
    df = df.sort_values(['ticker', 'begin']).copy().reset_index(drop=True)
    for k in range(1, max_horizon + 1):
        future_close = df.groupby('ticker')['close'].shift(-k)
        df[f'target_{k}'] = (future_close / df['close']) - 1
    df = df.dropna(subset=[f'target_{k}' for k in range(1, max_horizon + 1)])
    return df

def prepare_features_targets(merged_df, test_ratio=TEST_RATIO, max_horizon=MAX_HORIZON):
    exclude = ['begin', 'date', 'ticker', 'publish_date', 'first_date', 'last_date']
    feature_cols = [c for c in merged_df.columns if not any(e in c for e in exclude) and not c.startswith('target_')]
    rows_number = merged_df.shape[0]
    train_rows_number = round((1 - test_ratio) * rows_number)
    X_train = merged_df.iloc[:train_rows_number][feature_cols].copy()
    Y_train = merged_df.iloc[:train_rows_number][[f'target_{k}' for k in range(1, max_horizon + 1)]].copy()
    X_test = merged_df.iloc[train_rows_number:][feature_cols].copy()
    Y_test = merged_df.iloc[train_rows_number:][[f'target_{k}' for k in range(1, max_horizon + 1)]].copy()
    return X_train, Y_train, X_test, Y_test, feature_cols, train_rows_number

def train_regression(X_train, Y_train):
    model = MultiOutputRegressor(LGBMRegressor(
        n_estimators=200, learning_rate=0.01, max_depth=7, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbosity=-1))
    model.fit(X_train, Y_train)
    return model

def main():
    print("Loading data...")
    news_raw = pd.read_csv(NEWS_PATH, index_col=0)
    candles = pd.read_csv(CANDLES_PATH, parse_dates=["begin"])
    candles = candles.sort_values("begin").reset_index(drop=True)
    news_exploded = parse_aggregated_news(news_raw)
    news_features = create_news_features(news_exploded, news_raw)
    tech = create_technical_features(candles)
    merged = tech.merge(news_features, how="left", left_on=["ticker", "date"], right_on=["ticker", "publish_date"])
    merged = merged.drop(columns=["publish_date"], errors="ignore")
    merged = add_target_columns(merged, max_horizon=MAX_HORIZON)
    X_train, Y_train, X_test, Y_test, feature_cols, train_rows_number = prepare_features_targets(merged, test_ratio=TEST_RATIO, max_horizon=MAX_HORIZON)

    le = LabelEncoder(); le.fit(merged['ticker'].astype(str).values)
    scaler = MinMaxScaler()
    scale_cols = [c for c in ['open','close','high','low','volume'] if c in X_train.columns]
    if scale_cols:
        X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
        X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    print("Training regression...")
    reg_model = train_regression(X_train, Y_train)
    Y_test_pred = reg_model.predict(X_test)
    reg_mae = mean_absolute_error(Y_test, Y_test_pred)
    print("Test MAE:", reg_mae)

    joblib.dump(reg_model, OUT_DIR / "regression_model.pkl")
    joblib.dump(le, OUT_DIR / "label_encoder.pkl")
    joblib.dump(scaler, OUT_DIR / "scaler.pkl")

    preds_df = merged.iloc[train_rows_number:].reset_index(drop=True)
    for i in range(Y_test_pred.shape[1]):
        preds_df[f'pred_target_{i+1}'] = np.nan
        preds_df.loc[:len(Y_test_pred)-1, f'pred_target_{i+1}'] = Y_test_pred[:, i]
    preds_df.to_parquet(OUT_DIR / "predictions.parquet", index=False)

    if 'begin' in merged.columns:
        idx = merged.groupby('ticker')['begin'].idxmax()
        last_rows = merged.loc[idx].reset_index(drop=True)
    else:
        last_rows = merged.groupby('ticker').tail(1).reset_index(drop=True)
    use_features = feature_cols
    X_latest = last_rows[use_features].copy()
    X_latest['ticker'] = le.transform(last_rows['ticker'].astype(str))
    if scale_cols:
        X_latest[scale_cols] = scaler.transform(X_latest[scale_cols])
    preds = reg_model.predict(X_latest)
    preds = np.array(preds)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    n_h = preds.shape[1]
    if n_h < MAX_HORIZON:
        pad = np.full((preds.shape[0], MAX_HORIZON - n_h), np.nan)
        preds = np.hstack([preds, pad])
    elif n_h > MAX_HORIZON:
        preds = preds[:, :MAX_HORIZON]
    tickers_out = last_rows['ticker'].astype(str).values
    pred_cols = [f"p{i+1}" for i in range(preds.shape[1])]
    out_df = pd.DataFrame(preds, columns=pred_cols)
    out_df.insert(0, "ticker", tickers_out)
    out_csv_path = OUT_DIR / "per_ticker_predictions_p.csv"
    out_df.to_csv(out_csv_path, index=False)
    print("Saved per-ticker predictions ->", out_csv_path)

    # horizon MAE/RMSE report & plot
    horizon_maes = []
    horizon_rmses = []
    for k in range(1, MAX_HORIZON + 1):
        mae_k = mean_absolute_error(Y_test[f'target_{k}'], Y_test_pred[:, k-1])
        rmse_k = np.sqrt(np.mean((Y_test[f'target_{k}'] - Y_test_pred[:, k-1])**2))
        horizon_maes.append(mae_k)
        horizon_rmses.append(rmse_k)

    print("Overall Test MAE:", np.mean(horizon_maes))
    print("Saving MAE/RMSE plot...")
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, MAX_HORIZON + 1), horizon_maes, marker='o')
    plt.title('MAE by Prediction Horizon')
    plt.xlabel('Days'); plt.ylabel('MAE'); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, MAX_HORIZON + 1), horizon_rmses, marker='s')
    plt.title('RMSE by Prediction Horizon')
    plt.xlabel('Days'); plt.ylabel('RMSE'); plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'mae_rmse_by_horizon.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
