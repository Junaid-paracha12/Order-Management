"""Web service: REST API + the live-prediction page.   uvicorn app:app --port 8000   ->   http://localhost:8000"""
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import ROOT, SHIPPING_MODES, LateRiskPredictor

app = FastAPI(title="Late Delivery Risk API", version="1.0")
predictor = LateRiskPredictor()


class OrderIn(BaseModel):
    shipping_mode: Literal["First Class", "Same Day", "Second Class", "Standard Class"]
    order_time: datetime          # ISO 8601, e.g. 2017-03-01T15:30


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/model-info")
def model_info():
    return {**predictor.meta, "shipping_modes": SHIPPING_MODES}


@app.post("/api/predict")
def predict(order: OrderIn):
    try:
        return predictor.predict(order.shipping_mode, order.order_time)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
