# Late Delivery Risk Predictor


```

## Run
```bash
pip install -r requirements.txt
uvicorn app:app --port 8000            # open http://localhost:8000
python pipeline.py --mode "Same Day" --time "2017-03-01 15:30"       # single prediction from the CLI
pytest -q                              # tests
python train.py --data data/DataCoSupplyChainDataset.csv             # retrain (put the CSV in data/)
``
