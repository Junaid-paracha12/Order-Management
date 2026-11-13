# Late Delivery Risk Predictor

**Problem.** The Order Management Hub (Shopify orders -> WMS inventory update -> tracking pushed to the customer portal)
does not know, when an order is placed, whether it will arrive late. This service predicts it at order time, so the hub can
warn the customer, escalate, or suggest a faster shipping mode.

**Model.** Binary classifier for `Late_delivery_risk` (1 = real shipping days > scheduled days). Final model: a smoothed
lookup table on **Shipping Mode x order hour**. All six families tuned with Optuna (lookup, logistic regression, random forest,
HistGradientBoosting, LightGBM, XGBoost; 50 trials each, +100 on the best) tied on cross-validated log loss, so the simplest won.

| Test set (later orders, cancelled excluded) | |
|---|---|
| ROC-AUC | 0.777 |
| Accuracy | 72.5 % |
| Precision / recall at threshold 0.595 | 90 % / 59 % |

Limits: only shipping mode and order hour carry signal at order time; Standard Class orders (~40 % late) cannot be separated
further. It estimates lateness for orders that eventually ship (cancellation is a separate problem).

## Layout
```
pipeline.py            LookupClassifier, OrderTimeFeatures, build_pipeline(), LateRiskPredictor (+ CLI)
train.py               rebuilds model/late_risk_model.joblib from the DataCo CSV
app.py                 FastAPI service: /api/predict, /api/model-info, /api/health, and the page at /
static/                index.html, style.css, app.js  (live-prediction frontend)
model/                 late_risk_model.joblib (sklearn Pipeline + threshold + metadata), model_card.json
tests/test_pipeline.py pytest suite
requirements.txt
```

## Run
```bash
pip install -r requirements.txt
uvicorn app:app --port 8000            # open http://localhost:8000
python pipeline.py --mode "Same Day" --time "2017-03-01 15:30"       # single prediction from the CLI
pytest -q                              # tests
python train.py --data data/DataCoSupplyChainDataset.csv             # retrain (put the CSV in data/)
```

API example:
```bash
curl -X POST localhost:8000/api/predict -H 'Content-Type: application/json' \
     -d '{"shipping_mode":"Same Day","order_time":"2017-03-01T15:30"}'
```

The joblib file stores classes defined in `pipeline.py`, so load it from this folder (or with `pipeline.py` on the path),
and with the pinned scikit-learn version.
