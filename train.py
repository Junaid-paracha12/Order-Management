"""Rebuild the model from the DataCo CSV (cleaned export of the EDA notebook, or the raw Kaggle file).

    python train.py --data data/DataCoSupplyChainDataset.csv

Same recipe as the notebook: one row per order, cancelled orders excluded, chronological 80/20 split with a
6-day embargo, lookup table on Shipping Mode x order hour, alert threshold from forward (expanding-window) out-of-fold
predictions on the training period, final evaluation on the later test period.
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss,
                             precision_score, recall_score, roc_auc_score)

import pipeline as P

SHRINK = 1.005012887             # Optuna-tuned smoothing strength from the notebook
RAW_RENAME = {"order date (DateOrders)": "Order Date"}


def load_orders(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1").rename(columns=RAW_RENAME)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    o = (df.groupby("Order Id").agg(**{c: (c, "first") for c in ["Shipping Mode", "Order Date", "Delivery Status", P.TARGET]})
           .reset_index())
    o = o[o["Delivery Status"] != "Shipping canceled"]           # estimate lateness for orders that eventually ship
    return o.sort_values("Order Date").reset_index(drop=True)


def embargo_days(path: str) -> int:
    df = pd.read_csv(path, encoding="latin-1", usecols=lambda c: c in ("Days for shipping (real)",))
    return int(df["Days for shipping (real)"].max()) if not df.empty else 6


def expanding_splits(frame, embargo, n_splits=5, min_train_fraction=.5):
    day = frame["Order Date"].dt.normalize()
    dates = pd.DatetimeIndex(sorted(day.unique()))
    first = max(1, int(len(dates) * min_train_fraction))
    out = []
    for pos in np.array_split(np.arange(first, len(dates)), n_splits):
        block = dates[pos]
        tr = np.flatnonzero(day.isin(dates[dates < block[0] - pd.Timedelta(days=embargo)]).to_numpy())
        va = np.flatnonzero(day.isin(block).to_numpy())
        if len(tr) and len(va):
            out.append((tr, va))
    return out


def pick_threshold(train, embargo):
    y, oof = train[P.TARGET].to_numpy(), np.full(len(train), np.nan)
    for tr, va in expanding_splits(train, embargo):
        m = P.build_pipeline(SHRINK).fit(train.iloc[tr], y[tr])
        oof[va] = m.predict_proba(train.iloc[va])[:, 1]
    ok = ~np.isnan(oof)
    grid = np.linspace(.05, .95, 91)
    s = np.array([balanced_accuracy_score(y[ok], oof[ok] >= g) for g in grid])
    return float(np.median(grid[s >= s.max() - 1e-9]))            # middle of the best plateau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(P.MODEL_PATH))
    a = ap.parse_args()

    orders, emb = load_orders(a.data), embargo_days(a.data)
    day = orders["Order Date"].dt.normalize()
    all_dates = pd.DatetimeIndex(sorted(day.unique()))
    cut = all_dates[int(np.floor(len(all_dates) * .8)) - 1]
    train, test = orders[day <= cut - pd.Timedelta(days=emb)].copy(), orders[day > cut].copy()
    print(f"orders {len(orders):,} | train {len(train):,} | test {len(test):,} | embargo {emb} days")

    thr = pick_threshold(train, emb)
    model = P.build_pipeline(SHRINK).fit(train, train[P.TARGET])
    p, y = model.predict_proba(test)[:, 1], test[P.TARGET].to_numpy()
    pred = p >= thr
    metrics = {"roc_auc": roc_auc_score(y, p), "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p),
               "accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
               "precision": precision_score(y, pred), "recall": recall_score(y, pred)}
    meta = {"model": "smoothed lookup table: Shipping Mode x order hour", "target": P.TARGET, "threshold": thr,
            "shrink": SHRINK, "embargo_days": emb, "train_orders": len(train), "test_orders": len(test),
            "train_end": str(train["Order Date"].max()), "test_start": str(test["Order Date"].min()),
            "test_metrics": {k: round(float(v), 4) for k, v in metrics.items()}}
    joblib.dump({"pipeline": model, "threshold": thr, "meta": meta}, a.out)
    json.dump(meta, open(P.ROOT / "model" / "model_card.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
