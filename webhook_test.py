# webhook_test.py — offline dry run of the webhook inference path
# Uses fastapi.testclient: no server needed. Seeds the buffer with real CSV bars,
# verifies warmup behavior, then streams bars and checks BUY/SELL/HOLD decisions.

import os
import pandas as pd
from fastapi.testclient import TestClient

import webhook_server
from webhook_server import app, load_model

MODEL = os.environ.get("GOLD_MODEL_PATH", "model_gold_best")
if MODEL not in ("",) and not (os.path.exists(MODEL + ".zip") or os.path.exists(MODEL)):
    raise SystemExit(f"Model {MODEL} not found — run test after training.")

from indicators import load_and_preprocess_data

df, fc = load_and_preprocess_data("data/gold_data_10y.csv")
test_df = df.tail(200)

client = TestClient(app)
load_model(MODEL)

assert client.get("/health").json()["status"] == "ok"
r = client.post("/webhook/reset")
assert r.json()["status"] == "reset"

rows = []
for idx, (ts, row) in enumerate(test_df.iterrows()):
    payload = {
        "timestamp": str(ts),
        "open": float(row["Open"]),
        "high": float(row["High"]),
        "low": float(row["Low"]),
        "close": float(row["Close"]),
        "volume": float(row["Volume"]),
    }
    r = client.post("/webhook/gold_bar", json=payload)
    assert r.status_code == 200, r.text
    j = r.json()
    rows.append(j)

warmup = sum(1 for r_ in rows if r_["status"] == "warming_up")
live = [r_ for r_ in rows if r_["status"] == "ok"]
print(f"bars posted: {len(rows)}  warming: {warmup}  live decisions: {len(live)}")
assert warmup > 0 and len(live) > 50, "did not reach live inference"
assert all("action" in r_ and r_["action"] in ("BUY", "SELL", "HOLD", "CLOSE") for r_ in live)

from collections import Counter
counts = Counter(r_["action"] for r_ in live)
conf = [r_["confidence"] for r_ in live]
print("action distribution:", dict(counts))
print(f"confidence: mean={sum(conf)/len(conf):.3f} min={min(conf):.3f} max={max(conf):.3f}")
print("last decision:", live[-1])

st = client.get("/state").json()
print("server state:", st)
print("WEBHOOK DRY RUN OK ✓")
