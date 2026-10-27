# Gold (XAU/USD) RL Trading Agent

A reinforcement-learning trading system for **Gold (XAU/USD, daily OHLCV bars)**, adapted from [`ZiadFrancis/ReinforcementTrading_Part_1`](https://github.com/ZiadFrancis/ReinforcementTrading_Part_1) (EUR/USD, hourly) and re-tuned for Gold's scale and volatility. A PPO agent (Stable-Baselines3) learns to open/hold/close trades with ATR-scaled stop-loss / take-profit levels, and is served for live inference over a FastAPI webhook.

## Pipeline

```
data/gold_data_10y.csv
      │
      ▼
indicators.py ──► cleaned df + features (RSI, ATR, SMA, relative price features)
      │
      ▼
trading_env.py ──► 20-action env: HOLD/SUBMIT/CLOSE × {SL 1.0/1.5/2.0×ATR} × {TP 1.5/2.0/3.0×ATR}
      │
      ├──► train_agent.py ──► PPO (SB3) ► checkpoints/ + model_gold_best.zip + results plots
      ├──► test_agent.py  ──► held-out 20% backtest ► trade CSV, equity curve, entry/exit markers
      └──► webhook_server.py ──► POST /webhook/gold_bar ► action + confidence (live inference)

metrics.py ──► total return, PnL, max drawdown, Sharpe, Sortino, win rate, profit factor
```

## Why ATR-multiple SL/TP?

The original EUR/USD-hourly pip-scale SL/TP is meaningless on daily Gold bars ($20–40/bar). With pip-scale stops, every trade instantly hit both SL and TP and the agent collapsed into "never trade" (HOLD-collapse: 0 trades, flat equity). Redesigns applied here:

1. **SL/TP as ATR multiples** — SL ∈ {1.0, 1.5, 2.0}×ATR(14), TP ∈ {1.5, 2.0, 3.0}×ATR(14) → 20 actions, scale-agnostic.
2. **Idle penalty** — flat bars are lightly taxed so the agent can't collapse into never trading.
3. **Recognition observation** — obs = rolling 30-bar window × (8 relative features + 3 state features: position, bars-in-trade, scaled unrealized P&L). No raw price levels, so the agent isn't confounded by the $1,100→$4,200 price drift.

## Project structure

| File | Purpose |
|---|---|
| `indicators.py` | Symbol-agnostic CSV loader (auto-detects datetime column), tz-safe cleaning, RSI/ATR/SMA features, `dataframe_from_records()` for live bars |
| `trading_env.py` | `ForexTradingEnv` / `GoldTradingEnv` — tick 0.01, contract 100 oz, 30-pip spread, ATR-multiple SL/TP, `idle_penalty_pips` |
| `train_agent.py` | PPO training, 80/20 chronological split, checkpoints every 50k steps, best-by-OOS-final-equity selection |
| `test_agent.py` | Backtest on the held-out 20%: trade CSV, equity plot, buy/sell markers on price |
| `metrics.py` | Trading performance metrics |
| `webhook_server.py` | FastAPI live inference: sliding 120-bar buffer → 30-bar observation → action + confidence |
| `webhook_test.py` | Posts 200 bars and verifies live decisions |
| `smoke_env.py` | Environment sanity check |
| `run_train.cmd` | Windows one-click training launcher |
| `model_gold_best.zip` | Saved best PPO model |
| `data/gold_data_10y.csv` | 10 years of Gold daily bars |
| `results/` | Equity curves, trade history CSVs, price-marker plots |
| `RUN.md` | Detailed run guide, design decisions, and run history |

## Setup

```bash
pip install -r Requirements.txt
# or just the essentials:
pip install stable-baselines3 gymnasium pandas-ta torch fastapi uvicorn pandas numpy matplotlib
```

Python 3.10+ recommended. Data goes in `data/gold_data_10y.csv` (or point at any OHLCV CSV with `--csv_path`).

## Usage

**Train** (writes checkpoints, `model_gold_best.zip`, plots into `./results`):

```bash
python train_agent.py --csv_path data/gold_data_10y.csv --timesteps 150000 --results_dir ./results
# or set GOLD_CSV_PATH instead of --csv_path
```

**Backtest on the held-out 20%:**

```bash
python test_agent.py --model_path model_gold_best --csv_path data/gold_data_10y.csv --results_dir ./results
```

**Live inference server:**

```bash
set GOLD_MODEL_PATH=model_gold_best        # or export on bash
uvicorn webhook_server:app --host 0.0.0.0 --port 8000

# post one OHLCV bar (first ~120 posts return "warming_up" while indicators build history):
curl -X POST http://localhost:8000/webhook/gold_bar ^
  -H "Content-Type: application/json" ^
  -d "{\"timestamp\":\"2026-10-13T00:00:00Z\",\"open\":4195.5,\"high\":4224.8,\"low\":4173.7,\"close\":4202.3,\"volume\":61889}"
```

Response:

```json
{"status":"ok","action":"BUY","action_id":9,"confidence":0.41,"value_estimate":62.3,
 "top_actions":[{"action":"BUY","prob":0.41},{"action":"HOLD","prob":0.30}],"position":1}
```

**Sanity checks:**

```bash
python smoke_env.py     # env reset/step/reward sanity
python webhook_test.py  # 200 posted bars → warmup → live BUY/SELL/HOLD/CLOSE decisions
```

## Results (Run #2 — ATR-based, 150k PPO steps)

Trained on 1,971 bars, backtested on 493 held-out daily bars. Sizing 0.2 lot (20 oz), spread 30 pips, slippage ≤ 2 pips. Best checkpoint selected by out-of-sample final equity (`ppo_gold_50000_steps` → `model_gold_best`, OOS $222,048).

| Metric (held-out test) | Value |
|---|---|
| Initial equity | $10,000 |
| Final equity | $52,409 |
| Total return | **+424.1%** |
| Max drawdown | 83.2% |
| Sharpe (daily, annualized) | 0.644 |
| Sortino | 0.722 |
| Trades | 77 |
| Win rate | 57.1% |
| Profit factor | 1.51 |

## Drawdown optimization (new)

The headline backtest above (fixed 0.2-lot sizing) reached +424% but with an **83.2% max drawdown**. Root cause: with daily Gold ATR ≈ $20–40, even a 1.0×ATR stop is ~2,000–4,000 pips ≈ **$400–800 per stop-loss (4–8% of equity per losing trade)** — a normal loss streak compounded into the 83% DD.

Fixes shipped in `trading_env.py` (all backward-compatible, off by default):

1. **Risk-% position sizing** (`risk_pct_per_trade`, e.g. `0.03`) — position size is chosen so that hitting the SL loses exactly N% of *current* equity (fireside: `units = equity·risk% / (SL_pips·pip_value)`). Losses are capped per trade and sizes automatically shrink after losses. Optional `min_lot_units`/`max_lot_units` bounds.
2. **Drawdown-aware reward** (`drawdown_penalty_weight`, optional `final_dd_penalty_weight`) — reward penalty proportional to intra-episode drawdown growth, so the agent is penalized for riding out deep equity dips.
3. **Max-drawdown circuit breaker** (`max_dd_pct`, e.g. `0.25`) — once equity falls below `peak·(1−x)`, any open position is force-closed (`MAX_DD_CIRCUIT_BREAKER`) and further OPENs are ignored for the rest of the episode.

Held-out test results (same 493 bars, same trades — the old PPO agent replayed under better money management wins this comparison):

| Config | Final equity | Return | **MDD** | Sharpe | Sortino | Win rate | PF |
|---|---|---|---|---|---|---|---|
| Original (fixed 0.2 lot) | $52,409 | +424.1% | **83.2%** | 0.644 | 0.72 | 57.1% | 1.51 |
| Old agent + 2% risk | $16,943 | +69.4% | **11.0%** | 1.264 | 2.86 | 57.1% | 1.51 |
| Old agent + 3% risk | $21,418 | +114.2% | **16.1%** | 1.251 | 2.82 | 57.1% | 1.51 |
| Old agent + 4% risk | $26,931 | +169.3% | **21.0%** | 1.251 | 2.82 | 57.1% | 1.51 |
| Retrained agent (3% risk + DD shaping + 25% breaker) | $20,753 | +107.5% | **26.3%** | 1.383 | — | 42.6% | 1.45 |

Return/max-DD ratio improved from 5.1 → 8.1. Risk sizing scales both return and DD roughly linearly, so pick your risk budget:

```bash
# backtest the old agent with risk sizing (recommended default: 3%)
python test_agent.py --model_path model_gold_best --risk_pct_per_trade 0.03 --max_dd_pct 0.25

# retrain with risk controls baked into the reward (produces model_gold_risk)
python train_agent.py --csv_path data/gold_data_10y.csv --timesteps 150000 \
  --risk_pct_per_trade 0.03 --drawdown_penalty_weight 10.0 --max_dd_pct 0.25 \
  --results_dir ./results_risk_train --model_name model_gold_risk
```

The retrained agent did not beat the risk-sized replay of the original agent on return/DD quality (its DD-penalty reward made it trade more cautiously but with a worse 26.3% DD). Both models are shipped (`model_gold_best` replay-config remains the recommended deployable one).

## Honest caveats

- The test window coincides with the exceptional 2025–26 gold rally ($2,700→$5,500); a long-skewed RL agent is expected to shine there. Treat +424% as **regime-dependent**, not a general edge.
- After 50k steps the PPO action probabilities are still near-uniform (entropy ≈ ln(20)); the current edge comes from argmax ordering, not sharp confidence. Longer / reward-scaled runs would sharpen it.
- Paper-trading model only — this is a research pipeline, not financial advice.

## See also

- [`RUN.md`](RUN.md) — full run guide, Gold-specific design decisions, and per-run history.
