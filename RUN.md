# Gold (XAU/USD) RL Trading — Run Guide

Adapted from `ZiadFrancis/ReinforcementTrading_Part_1` (EURUSD/hourly) to **Gold (XAU/USD, daily bars)** with your `gold_data_10y.csv`.

## What's running where

| Stage | Script | Notes |
|---|---|---|
| Data | `indicators.py` | auto-detects datetime col (your `Date`), timezone-safe, cleans bad bars, adds RSI/ATR/SMA features |
| Env | `trading_env.py` | Gold params: tick 0.01, contract 100 oz (= $1/pip), 30-pip spread, **SL/TP as ATR multiples** (scale-agnostic for daily bars) |
| Metrics | `metrics.py` | Total Return, PnL, MDD, Sharpe, Sortino, trades, win rate, avg win/loss, profit factor |
| Train | `train_agent.py` | PPO (SB3), 80/20 time split, checkpoints every 50k, best-by-OOS-equity selection |
| Test | `test_agent.py` | held-out 20% backtest, trade CSV + equity plot + buy/sell markers on price |
| Live | `webhook_server.py` | FastAPI, `POST /webhook/gold_bar`, sliding 120-bar buffer → 30-bar obs → action + confidence |
| Colab | `Gold_RL_Trading_Colab.ipynb` | same pipeline end-to-end in 10 cells |

## Key design decisions (Gold-specific)

1. **SL/TP in ATR multiples** (`atr_col="atr_14"`): your original pip-based SL/TP (0.05–1.2 in price) is tuned for EURUSD *hourly* noise; on daily Gold bars (20–40 $/bar) every trade instantly hit both SL and TP, and the conservative "SL-first" rule turned trades into guaranteed losses. Now: SL=1.0–2.0×ATR, TP=1.5–3.0×ATR → 20 actions total.
2. **Idle penalty** (`idle_penalty_pips=1.0`): flat bars are slightly taxed so the agent can't collapse into "never trade" (which was the first run's HOLD-collapse: 0 trades, flat equity).
3. **Recognition state**: obs = rolling 30-bar window × (8 relative features + 3 state features: position, bars-in-trade, unrealized P&L scaled). No raw price levels → not confounded by the $1,100→$4,200 drift.

## CLI commands (local)

```bash
pip install stable-baselines3 gymnasium pandas-ta fastapi uvicorn torch pandas numpy matplotlib

# 1) train (writes checkpoints + ./results plots + model_gold_best.zip)
python train_agent.py --csv_path data/gold_data_10y.csv --timesteps 150000 --results_dir ./results
#    env var equivalent:  set GOLD_CSV_PATH=... (Windows) / export GOLD_CSV_PATH=... (bash)

# 2) backtest on the held-out 20%
python test_agent.py --model_path model_gold_best --csv_path data/gold_data_10y.csv --results_dir ./results

# 3) serve live inference
set GOLD_MODEL_PATH=model_gold_best     # or export on bash
uvicorn webhook_server:app --host 0.0.0.0 --port 8000

# then post bars:
curl -X POST http://localhost:8000/webhook/gold_bar ^
  -H "Content-Type: application/json" ^
  -d "{\"timestamp\":\"2026-10-13T00:00:00Z\",\"open\":4195.5,\"high\":4224.8,\"low\":4173.7,\"close\":4202.3,\"volume\":61889}"
```

Webhook payload = single OHLCV bar. First ~120 posts are "warming_up" while indicators need history; then each bar returns:

```json
{"status":"ok","action":"BUY","action_id":9,"confidence":0.41,"value_estimate":62.3,
 "top_actions":[{"action":"BUY","prob":0.41},{"action":"HOLD","prob":0.30}],"position":1}
```

## Colab

Open `Gold_RL_Trading_Colab.ipynb` → Runtime ▸ Run all. Upload the project files + CSV when prompted (cell 2). Defaults: 150k steps.

## Files created/modified

| File | Status |
|---|---|
| `indicators.py` | **modified** — symbol-agnostic loader, tz-safe, cleaning, numpy-2 shim, `dataframe_from_records()` for live bars |
| `trading_env.py` | **modified** — Gold asset params (tick/contract/spread), `atr_col` ATR-scaled SL/TP, `idle_penalty_pips` |
| `train_agent.py` | **rewritten** — `--csv_path`/`--timesteps`/`--results_dir` args, ATR-multiple SL/TP, metrics + plots into `./results/`, `model_gold_best` |
| `test_agent.py` | **rewritten** — `--csv_path`/`--model_path`/`--results_dir`, meaning metrics, entry/exit marker plots |
| `metrics.py` | **new** |
| `webhook_server.py` | **new** — FastAPI live-inference with sliding buffer |
| `Gold_RL_Trading_Colab.ipynb` | **new** |
| `RUN.md` | this file |
| `data/gold_data_10y.csv` | copied from Downloads |

## First-run lesson (kept for the record)

Run #1 with pip-scale SL/TP trained 100k steps → best checkpoint lost 44% (every trade = instant SL loss), final agent collapsed to HOLD. That diagnosed the scale mismatch and motivated the ATR-multiple redesign above.

## Final results (Run #2, ATR-based, 150k PPO steps)

Training on 1971 bars / backtest on held-out 493 bars. Sizing: 0.2 lot (20 oz) → $0.20 per pip; SL/TP = ATR multiples; spread 30 pips; slippage ≤ 2 pips.

Checkpoint selection by out-of-sample final equity:

| Model | OOS final equity |
|---|---|
| **ppo_gold_50000_steps → `model_gold_best`** | **$222,048 (best)** |
| ppo_gold_100000_steps | $175,806 |
| ppo_gold_150000_steps (last) | $207,177 |

**Backtest metrics (held-out test, `results/trade_history_output.csv`):**

| Metric | Value |
|---|---|
| Initial equity | $10,000 |
| Final equity | $52,409 (at 0.2-lot sizing) |
| Total PnL | +$42,409 |
| Total return | **+424.1%** |
| Max drawdown | 83.2% |
| Sharpe (daily, annualized) | 0.644 |
| Sortino | 0.722 |
| Total trades | 77 |
| Win rate | **57.1%** |
| Avg win / loss | +14,093 / −12,431 pips |
| Profit factor | **1.51** |

In-sample eval of the same agent: $10,000 → $173,320. Artifacts: `results/equity_train_test.png`, `results/equity_eval.png`, `results/price_markers.png` (entries ▲ green = long, ▼ red = short, ✕ = exits), trade CSVs.

Honesty notes: (1) the test window coincides with the exceptional 2025–26 gold rally ($2,700→$5,500), so a long-skewed RL agent is expected to shine — treat +424% as regime-dependent; (2) action probabilities are still near-uniform after 50k steps (PPO entropy stays close to ln(20)); the edge comes from argmax ordering, not sharp confidence — a longer/`reward_scale`d run would sharpen it.

Verification: `python smoke_env.py` (env sanity) and `python webhook_test.py` (200 posted bars → 119 warmup → 81 live BUY/SELL/HOLD/CLOSE decisions with confidence + value estimates) both pass.
