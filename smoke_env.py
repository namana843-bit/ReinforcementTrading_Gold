from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv

df, fc = load_and_preprocess_data("data/gold_data_10y.csv")
train_df = df.iloc[: int(len(df) * 0.8)].copy()
env = ForexTradingEnv(
    df=train_df,
    window_size=30,
    sl_options=[1.0, 1.5, 2.0],
    tp_options=[1.5, 2.0, 3.0],
    feature_columns=fc,
    spread_pips=30.0,
    max_slippage_pips=5.0,
    random_start=False,
    hold_reward_weight=0.02,
    idle_penalty_pips=1.0,
    atr_col="atr_14",
    asset_name="XAUUSD",
)
obs, _ = env.reset()
print("act space:", env.action_space.n, "obs:", obs.shape)
env.step(3)  # OPEN long sl=1.0 ATR tp=1.5 ATR
print("pos", env.position, "entry", env.entry_price, "sl", env.sl_price, "tp", env.tp_price)
env.step(0)
print("after next bar: pos", env.position, "equity", env.equity_usd, "last_trade", env.last_trade_info)
trade_count = 0
obs, _ = env.reset()
done = False
n = 0
while not done and n < 100000:
    a = int(env.action_space.sample())
    obs, r, term, trunc, info = env.step(a)
    done = term or trunc
    n += 1
    if env.last_trade_info and env.last_trade_info.get("event") == "CLOSE":
        trade_count += 1
print("random walk done. steps:", n, "closes:", trade_count, "final eq:", env.equity_usd)
