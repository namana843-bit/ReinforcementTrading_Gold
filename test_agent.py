import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv
from metrics import compute_metrics, print_metrics


def run_one_episode(model, vec_env, deterministic=True):
    obs = vec_env.reset()
    equity_curve = []
    closed_trades = []
    entry_markers = []
    exit_markers = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = vec_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        info = infos[0] if isinstance(infos, (list, tuple)) else infos

        equity_curve.append(vec_env.get_attr("equity_usd")[0])

        trade_info = vec_env.get_attr("last_trade_info")[0]
        if isinstance(trade_info, dict):
            if trade_info.get("event") == "OPEN":
                if not entry_markers or entry_markers[-1][0] != trade_info.get("step"):
                    entry_markers.append((trade_info["step"], trade_info["entry_price"], trade_info["position"]))
            elif trade_info.get("event") == "CLOSE":
                if not exit_markers or exit_markers[-1][0] != trade_info.get("step"):
                    exit_markers.append((trade_info["step"], trade_info["exit_price"], trade_info["position"]))
                    closed_trades.append(trade_info)

        if done:
            # DummyVecEnv auto-resets the inner env after done (wiping its state),
            # so the terminal equity must come from THIS step's info dict.
            # Replace the (post-reset) point appended above with the true one.
            equity_curve[-1] = float(info.get("equity_usd", equity_curve[-1]))
            break
        equity_curve.append(vec_env.get_attr("equity_usd")[0])

    return equity_curve, closed_trades, entry_markers, exit_markers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default=os.environ.get("GOLD_CSV_PATH", "data/gold_data_10y.csv"))
    parser.add_argument("--model_path", type=str, default="model_gold_best")
    parser.add_argument("--results_dir", type=str, default="./results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        alt = r"C:\Users\Naman\Downloads\gold_data_10y.csv"
        if os.path.exists(alt):
            csv_path = alt

    df, feature_cols = load_and_preprocess_data(csv_path)

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    SL_OPTS = [1.0, 1.5, 2.0]   # ATR multiples (must match training)
    TP_OPTS = [1.5, 2.0, 3.0]
    WIN = 30

    test_env = ForexTradingEnv(
        df=test_df,
        window_size=WIN,
        sl_options=SL_OPTS,
        tp_options=TP_OPTS,
        spread_pips=30.0,
        commission_pips=0.0,
        max_slippage_pips=2.0,
        lot_size=20.0,
        random_start=False,
        episode_max_steps=None,
        feature_columns=feature_cols,
        hold_reward_weight=0.0,
        open_penalty_pips=0.0,
        time_penalty_pips=0.0,
        unrealized_delta_weight=0.0,
        asset_name="XAUUSD",
        atr_col="atr_14",
        idle_penalty_pips=1.0,
    )

    vec_test_env = DummyVecEnv([lambda: test_env])

    model = PPO.load(args.model_path, env=vec_test_env)

    equity_curve, closed_trades, entry_markers, exit_markers = run_one_episode(model, vec_test_env, deterministic=True)

    if closed_trades:
        pd.DataFrame(closed_trades).to_csv(os.path.join(args.results_dir, "trade_history_output.csv"), index=False)
        print(f"Closed trade history saved to {os.path.join(args.results_dir, 'trade_history_output.csv')}")
    else:
        print("No closed trades recorded.")

    m = compute_metrics(equity_curve, closed_trades)
    print_metrics(m)

    # equity plot
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label="Equity (Test)")
    plt.title("Equity Curve - Evaluation (Gold)")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "equity_eval.png"))
    plt.close()

    # price action with markers
    prices = test_df["Close"].values
    plt.figure(figsize=(14, 6))
    plt.plot(prices, label="Gold Close", color="black", linewidth=1)
    for step, price, pos in entry_markers:
        if 0 <= step < len(prices):
            color = "green" if pos == 1 else "red"
            marker = "^" if pos == 1 else "v"
            plt.scatter(step, price, color=color, marker=marker, s=60, zorder=5, label="Entry" if step == entry_markers[0][0] else "")
    for step, price, pos in exit_markers:
        if 0 <= step < len(prices):
            color = "blue" if pos == 1 else "orange"
            plt.scatter(step, price, color=color, marker="x", s=60, zorder=5)
    plt.title("Gold Price Action with Trade Markers")
    plt.xlabel("Bars")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "price_markers.png"))
    plt.close()

    print("Plots saved to", args.results_dir)


if __name__ == "__main__":
    main()
