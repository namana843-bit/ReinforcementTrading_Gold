import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv, GoldTradingEnv
from metrics import compute_metrics, print_metrics


def evaluate_model(model, eval_env, deterministic=True):
    obs = eval_env.reset()
    equity_curve = []
    closed_trades = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = eval_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        eq = info.get("equity_usd", eval_env.get_attr("equity_usd")[0])
        equity_curve.append(eq)

        trade_info = eval_env.get_attr("last_trade_info")[0]
        if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
            # avoid duplicates by checking last
            if not closed_trades or trade_info.get("step") != closed_trades[-1].get("step"):
                closed_trades.append(trade_info)

        if done:
            break

    final_equity = float(equity_curve[-1]) if equity_curve else 0.0
    return equity_curve, final_equity, closed_trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default=os.environ.get("GOLD_CSV_PATH", "data/gold_data_10y.csv"))
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--model_name", type=str, default="model_gold_best")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # ensure csv available locally
    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        # try the provided windows path
        alt = r"C:\Users\Naman\Downloads\gold_data_10y.csv"
        if os.path.exists(alt):
            csv_path = alt

    df, feature_cols = load_and_preprocess_data(csv_path)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print("Training bars:", len(train_df))
    print("Testing bars :", len(test_df))

    SL_OPTS = [1.0, 1.5, 2.0]   # ATR multiples (scale-agnostic for Gold daily bars)
    TP_OPTS = [1.5, 2.0, 3.0]   # ATR multiples
    WIN = 30

    def make_train_env():
        return ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=30.0,
            commission_pips=0.0,
            max_slippage_pips=2.0,
            lot_size=20.0,
            random_start=True,
            min_episode_steps=1000,
            episode_max_steps=2000,
            feature_columns=feature_cols,
            hold_reward_weight=0.0,
            open_penalty_pips=0.0,
            time_penalty_pips=0.0,
            unrealized_delta_weight=0.0,
            asset_name="XAUUSD",
            atr_col="atr_14",
            idle_penalty_pips=1.0,
        )

    def make_train_eval_env():
        return ForexTradingEnv(
            df=train_df,
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

    def make_test_eval_env():
        return ForexTradingEnv(
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

    train_vec_env = DummyVecEnv([make_train_env])
    train_eval_env = DummyVecEnv([make_train_eval_env])
    test_eval_env = DummyVecEnv([make_test_eval_env])

    model = PPO(
        policy="MlpPolicy",
        env=train_vec_env,
        verbose=1,
        tensorboard_log="./tensorboard_log/"
    )

    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=ckpt_dir,
        name_prefix="ppo_gold"
    )

    total_timesteps = args.timesteps
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    equity_curve_test_last, final_equity_test_last, _ = evaluate_model(model, test_eval_env)
    print(f"[OOS Eval] Last model final equity: {final_equity_test_last:.2f}")

    best_equity = -np.inf
    best_path = None

    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".zip") and f.startswith("ppo_gold")],
        key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x))
    )

    for ck in ckpts:
        ck_path = os.path.join(ckpt_dir, ck)
        try:
            m = PPO.load(ck_path, env=test_eval_env)
            _, final_eq, _ = evaluate_model(m, test_eval_env)
            print(f"[OOS Eval] {ck} -> final equity: {final_eq:.2f}")
            if final_eq > best_equity:
                best_equity = final_eq
                best_path = ck_path
        except Exception as e:
            print(f"[Skip] Could not evaluate checkpoint {ck}: {e}")

    if best_path is None or final_equity_test_last >= best_equity:
        print("Using last model as best (by OOS final equity).")
        best_model = model
    else:
        print(f"Using best checkpoint: {best_path} (OOS final equity: {best_equity:.2f})")
        best_model = PPO.load(best_path, env=train_vec_env)

    best_model.save(args.model_name)
    print("Best model saved:", args.model_name)

    equity_curve_train, final_equity_train, train_trades = evaluate_model(best_model, train_eval_env)
    equity_curve_test, final_equity_test, test_trades = evaluate_model(best_model, test_eval_env)

    print(f"[IS Eval]  Final equity (train): {final_equity_train:.2f}")
    print(f"[OOS Eval] Final equity (test) : {final_equity_test:.2f}")

    m = compute_metrics(equity_curve_test, test_trades)
    print_metrics(m)

    # plots
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_train, label="Train (in-sample) equity")
    plt.plot(equity_curve_test, label="Test (out-of-sample) equity")
    plt.title("Equity Curves: In-sample vs Out-of-sample (Best Model) - Gold")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "equity_train_test.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve_test, label="Equity (Test)")
    plt.title("Equity Curve - Evaluation (Gold)")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "equity_test.png"))
    plt.close()

    # save trades
    if test_trades:
        pd.DataFrame(test_trades).to_csv(os.path.join(args.results_dir, "trade_history_test.csv"), index=False)

    print("Plots saved to", args.results_dir)


if __name__ == "__main__":
    main()
