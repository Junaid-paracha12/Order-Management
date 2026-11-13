"""Late-delivery risk model: pipeline components, loading and prediction.

Everything the saved joblib file needs lives in this module, so it must stay importable
(same folder or installed) whenever `model/late_risk_model.joblib` is loaded.

    python pipeline.py --mode "Same Day" --time "2017-03-01 15:30"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "late_risk_model.joblib"
SHIPPING_MODES = ["First Class", "Same Day", "Second Class", "Standard Class"]
TARGET = "Late_delivery_risk"


class OrderTimeFeatures(BaseEstimator, TransformerMixin):
    """Raw order -> model features. Input columns: 'Shipping Mode', 'Order Date'. Output: 'Shipping Mode', 'order_hour'."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        when = pd.to_datetime(X["Order Date"])
        return pd.DataFrame({"Shipping Mode": X["Shipping Mode"].astype(str).to_numpy(),
                             "order_hour": when.dt.hour.to_numpy()}, index=X.index)


class LookupClassifier(ClassifierMixin, BaseEstimator):
    """Smoothed group-rate table. keys=('Shipping Mode', 'order_hour') gives a hierarchical estimate:
    prior -> mode rate -> mode x hour rate, each level shrunk toward its parent with strength `shrink`."""

    def __init__(self, keys=("Shipping Mode", "order_hour"), shrink=20.0):
        self.keys = keys
        self.shrink = shrink

    def _cols(self, X):
        return [X[k].astype(str).to_numpy() for k in self.keys]

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.prior_ = float(y.mean())
        cols = self._cols(X)
        d = pd.DataFrame({f"k{i}": c for i, c in enumerate(cols)})
        d["y"] = y
        self.tables_ = []
        for j in range(1, len(cols) + 1):
            names = [f"k{i}" for i in range(j)]
            g = d.groupby(names)["y"].agg(["sum", "count"])
            parent = np.full(len(g), self.prior_) if j == 1 else self.tables_[j - 2].reindex(g.index.droplevel(-1)).to_numpy()
            p = (g["sum"].to_numpy() + self.shrink * parent) / (g["count"].to_numpy() + self.shrink)
            self.tables_.append(pd.Series(p, index=g.index))
        return self

    def predict_proba(self, X):
        cols = self._cols(X)
        p = np.full(len(X), self.prior_)
        for j, table in enumerate(self.tables_, start=1):
            idx = pd.MultiIndex.from_arrays(cols[:j]) if j > 1 else pd.Index(cols[0])
            vals = table.reindex(idx).to_numpy(dtype=float)
            p = np.where(np.isnan(vals), p, vals)      # unseen cell -> keep the parent level's estimate
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)


def build_pipeline(shrink: float = 20.0) -> Pipeline:
    return Pipeline([("features", OrderTimeFeatures()), ("model", LookupClassifier(shrink=shrink))])


class LateRiskPredictor:
    """Loads the saved artifact and answers 'will this order be late?'."""

    def __init__(self, path: Path | str = MODEL_PATH):
        art = joblib.load(path)
        self.pipeline: Pipeline = art["pipeline"]
        self.threshold: float = float(art["threshold"])
        self.meta: dict = art.get("meta", {})

    def _frame(self, modes, times):
        return pd.DataFrame({"Shipping Mode": list(modes), "Order Date": pd.to_datetime(list(times))})

    def predict_proba(self, modes, times) -> np.ndarray:
        return self.pipeline.predict_proba(self._frame(modes, times))[:, 1]

    def predict(self, shipping_mode: str, order_time) -> dict:
        if shipping_mode not in SHIPPING_MODES:
            raise ValueError(f"shipping_mode must be one of {SHIPPING_MODES}")
        t = pd.Timestamp(order_time)
        p = float(self.predict_proba([shipping_mode], [t])[0])
        by_mode = self.predict_proba(SHIPPING_MODES, [t] * len(SHIPPING_MODES))
        return {"shipping_mode": shipping_mode, "order_time": t.isoformat(), "order_hour": int(t.hour),
                "late_probability": round(p, 4), "alert": p >= self.threshold, "threshold": self.threshold,
                "by_mode": [{"shipping_mode": m, "late_probability": round(float(q), 4), "alert": bool(q >= self.threshold)}
                            for m, q in zip(SHIPPING_MODES, by_mode)]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict late-delivery risk for one order")
    ap.add_argument("--mode", required=True, choices=SHIPPING_MODES)
    ap.add_argument("--time", required=True, help="order date-time, e.g. '2017-03-01 15:30'")
    a = ap.parse_args()
    print(json.dumps(LateRiskPredictor().predict(a.mode, a.time), indent=2))
