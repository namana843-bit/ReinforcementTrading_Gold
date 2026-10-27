# webhook_server.py
"""
Live Gold (XAU/USD) inference server.

POST /webhook/gold_bar   <- one OHLCV bar (dict, JSON)
POST /webhook/reset      <- clear the sliding buffer + position state
GET  /health             <- liveness + model info
GET  /state              <- current buffer size, position, last action

The server keeps a sliding window of recent bars. Once enough bars have
arrived to warm up the indicator features (RSI / ATR / SMA 20 & 50), every
new bar triggers: features -> observation (30-bar window) -> PPO policy ->
action BUY / SELL / HOLD / CLOSE, returned with confidence and the full
action probability distribution ("Q-value-like" scores for this PPO agent).
"""

from __future__ import annotations

import os
import threading
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from indicators import dataframe_from_records

# ----------------------------- config -----------------------------
MODEL_PATH = os.environ.get("GOLD_MODEL_PATH", "model_gold_best")
WINDOW_SIZE = int(os.environ.get("GOLD_WINDOW", "30"))
PREPARE_BARS = 120          # bars needed before first inference (indicator warmup)
BUILD_COLS = ["Open", "High", "Low", "Close", "Volume"]

from stable_baselines3 import PPO

from trading_env import ForexTradingEnv

app = FastAPI(title="Gold RL Webhook Inference", version="1.0")

_state = {
    "model": None,
    "feature_cols": None,
    "action_map": None,
    "buffer": deque(maxlen=1000),
    "position": 0,
    "entry_price": None,
    "time_in_trade": 0,
    "last_action": None,
    "last_conf": None,
}
_lock = threading.Lock()


class GoldBar(BaseModel):
    timestamp: str = Field(..., description="bar timestamp, e.g. 2024-05-01T00:00:00 or epoch")
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


def _build_action_map(sl_options, tp_options):
    action_map = [("HOLD", None, None, None), ("CLOSE", None, None, None)]
    for direction in [0, 1]:
        for sl in sl_options:
            for tp in tp_options:
                action_map.append(("OPEN", direction, float(sl), float(tp)))
    return action_map


def _must_match_training(env_kwargs_path: str):
    """SL/TP/feature config MUST match training config (see RUN.md)."""
    return {
        "sl_options": [1.0, 1.5, 2.0],      # ATR multiples
        "tp_options": [1.5, 2.0, 3.0],      # ATR multiples
    }


def load_model(model_path: str = MODEL_PATH):
    with _lock:
        _state["model"] = PPO.load(model_path)
        cfg = _must_match_training(model_path)
        _state["action_map"] = _build_action_map(cfg["sl_options"], cfg["tp_options"])
        # feature columns = the standard feature list from indicators.py
        from indicators import FEATURE_COLS
        _state["feature_cols"] = list(FEATURE_COLS)
    return True


def _obs_from_buffer() -> Optional[np.ndarray]:
    """Build the policy observation from the sliding window."""
    df_records = list(_state["buffer"])
    df_records = df_records[-PREPARE_BARS:]
    if len(df_records) < PREPARE_BARS:
        return None
    df, feature_cols = dataframe_from_records(df_records)
    if df is None or len(df) < WINDOW_SIZE:
        return None
    fc = _state["feature_cols"] or feature_cols
    rows = df[fc].tail(WINDOW_SIZE).astype(np.float32)

    position = float(_state["position"])
    t_norm = float(_state["time_in_trade"]) / 1000.0
    unreal_scaled = 0.0
    if _state["position"] != 0 and _state["entry_price"] is not None:
        last_close = float(df["Close"].iloc[-1])
        pnl = (last_close - _state["entry_price"]) if _state["position"] == 1 else (_state["entry_price"] - last_close)
        unreal_scaled = float(pnl / 100.0)   # XAUUSD: 1 pip=0.01 -> /100 keeps magnitudes tame
    state_feat = np.array([position, t_norm, unreal_scaled], dtype=np.float32)
    state_block = np.tile(state_feat, (WINDOW_SIZE, 1))
    return np.hstack([rows.values, state_block]).astype(np.float32)


def _prettify_action(action_id: int):
    amap = _state["action_map"]
    if amap is None or action_id >= len(amap):
        return "HOLD"
    act_type, direction, sl, tp = amap[action_id]
    if act_type == "HOLD":
        return "HOLD"
    if act_type == "CLOSE":
        return "CLOSE"
    return "BUY" if direction == 1 else "SELL"


def _update_server_position(action_id: int, last_close: float):
    amap = _state["action_map"]
    act_type, direction, sl, tp = amap[action_id]
    if act_type == "OPEN":
        _state["position"] = 1 if direction == 1 else -1
        _state["entry_price"] = last_close
        _state["time_in_trade"] = 0
    elif act_type == "CLOSE":
        _state["position"] = 0
        _state["entry_price"] = None
        _state["time_in_trade"] = 0
    else:
        if _state["position"] != 0:
            _state["time_in_trade"] += 1
    return act_type


@app.on_event("startup")
async def _startup():
    if not os.path.exists(MODEL_PATH + ".zip") and not os.path.exists(MODEL_PATH):
        print(f"[webhook] model '{MODEL_PATH}' not found yet; POST bars, then warm up.")
        return
    if _state["model"] is None:
        load_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "buffer_size": len(_state["buffer"]),
        "bars_required": PREPARE_BARS,
        "window_size": WINDOW_SIZE,
        "model_path": MODEL_PATH,
    }


@app.get("/state")
def get_state():
    return {
        "buffer_size": len(_state["buffer"]),
        "position": _state["position"],
        "last_action": _state["last_action"],
        "last_confidence": _state["last_conf"],
        "time_in_trade": _state["time_in_trade"],
    }


@app.post("/webhook/reset")
def reset_buffer():
    with _lock:
        _state["buffer"].clear()
        _state["position"] = 0
        _state["entry_price"] = None
        _state["time_in_trade"] = 0
        _state["last_action"] = None
        _state["last_conf"] = None
    return {"status": "reset", "buffer_size": 0}


@app.post("/webhook/gold_bar")
async def webhook_gold_bar(bar: GoldBar):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded; check /health")
    with _lock:
        _state["buffer"].append({
            "Date": bar.timestamp,
            "Open": bar.open,
            "High": bar.high,
            "Low": bar.low,
            "Close": bar.close,
            "Volume": bar.volume,
        })

    obs = _obs_from_buffer()
    if obs is None:
        return {
            "status": "warming_up",
            "buffer_size": len(_state["buffer"]),
            "bars_required": PREPARE_BARS,
            "action": None,
        }

    obs_batch = np.expand_dims(obs, axis=0)   # (1, window, features)
    model: PPO = _state["model"]

    with torch.no_grad():
        pred, _ = model.predict(obs_batch, deterministic=True)
        action_id = int(np.asarray(pred).ravel()[0])
        obs_t = torch.as_tensor(obs_batch)
        dist = model.policy.get_distribution(obs_t)
        probs = dist.distribution.probs[0].detach().cpu().numpy()
        value = float(model.policy.predict_values(obs_t).ravel()[0])

    confidence = float(probs[action_id])
    top_probs = np.argsort(probs)[::-1][:3]

    act_type = _update_server_position(action_id, last_close=float(bar.close))
    _update = _prettify_action(action_id)

    with _lock:
        _state["last_action"] = _update
        _state["last_conf"] = confidence

    return {
        "status": "ok",
        "action": _update,
        "action_id": action_id,
        "confidence": confidence,
        "value_estimate": value,
        "top_actions": [
            {"action": _prettify_action(int(a)), "prob": float(probs[a])} for a in top_probs
        ],
        "buffer_size": len(_state["buffer"]),
        "position": _state["position"],
        "act_type": act_type,
    }
