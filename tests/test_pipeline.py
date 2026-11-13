import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from app import app
from pipeline import SHIPPING_MODES, LateRiskPredictor

predictor = LateRiskPredictor()
client = TestClient(app)


def test_first_class_always_alert():
    assert predictor.predict("First Class", "2017-03-01 10:00")["alert"] is True


def test_same_day_noon_cutoff():
    before = predictor.predict("Same Day", "2017-03-01 09:00")["late_probability"]
    after = predictor.predict("Same Day", "2017-03-01 15:30")["late_probability"]
    assert before < 0.05 and after > 0.95


def test_standard_class_is_not_alerted():
    r = predictor.predict("Standard Class", "2017-03-01 09:00")
    assert 0.3 < r["late_probability"] < 0.5 and r["alert"] is False


def test_all_modes_in_response():
    assert [m["shipping_mode"] for m in predictor.predict("Same Day", "2017-03-01 15:30")["by_mode"]] == SHIPPING_MODES


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        predictor.predict("Drone", "2017-03-01 15:30")


def test_api_predict_and_validation():
    ok = client.post("/api/predict", json={"shipping_mode": "Second Class", "order_time": "2017-03-01T10:00"})
    assert ok.status_code == 200 and 0.7 < ok.json()["late_probability"] < 0.9
    assert client.post("/api/predict", json={"shipping_mode": "Drone", "order_time": "2017-03-01T10:00"}).status_code == 422
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/").status_code == 200
