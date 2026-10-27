# trading_env.py

from __future__ import annotations

import numpy as np

# Prefer gymnasium if available (SB3 supports it), fallback to gym
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    _GYMNASIUM = False


class ForexTradingEnv(gym.Env):
    """
    RL Forex Trading Environment (Position-Persistent)

    Key properties:
      - Observation: rolling window of features + 3 state features (position, time_in_trade, unrealized_pnl_pips)
      - Actions:
          0: HOLD (do nothing)
          1: CLOSE (close position if any)
          2..: OPEN (direction + SL + TP), only effective when flat
      - Position persistence: once open, position remains until:
          - agent sends CLOSE, or
          - SL/TP hit intrabar
      - Friction: spread + commission + optional slippage
      - Reward:
          - realized PnL (pips) minus costs (pips) on closes
          - optional shaping via delta unrealized PnL (pips) while holding
      - Random episode start to reduce memorization / overfit
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        window_size: int = 30,
        sl_options=None,
        tp_options=None,
        feature_columns = None,
        asset_name: str = "XAUUSD",            # target instrument (Gold)
        tick_size: float = 0.01,               # Gold tick size in price units (1 pip = 0.01)
        pip_value: float | None = None,         # alias for tick_size (price per pip)
        contract_size: float = 100.0,          # Gold contract size (oz per lot); 1 lot = 100 oz
        lot_size: float | None = None,         # alias for contract_size
        spread_pips: float = 30.0,             # Gold spread in pips (round-trip approx)
        commission_pips: float = 0.0,          # cost in pips per round-trip
        max_slippage_pips: float = 2.0,        # random extra pips (0..max) applied on fills
        reward_scale: float = 1.0,             # optional scaling of rewards
        unrealized_delta_weight: float = 0.02, # shaping weight on delta-unrealized while holding
        random_start: bool = True,
        min_episode_steps: int = 300,          # minimum steps per episode (for random starts)
        episode_max_steps: int | None = None,  # optional cap (truncation)
        feature_mean: np.ndarray | None = None, # optional normalization (train-fitted)
        feature_std: np.ndarray | None = None,  # optional normalization (train-fitted)
        allow_flip: bool = False,               # if True, OPEN while in position flips (close+open). Default False.
        hold_reward_weight: float = 0.005,   # tuned below
        open_penalty_pips: float = 0.5,      # NEW: penalty per open
        time_penalty_pips: float = 0.02,     # NEW: cost per bar in a trade
        atr_col: str | None = None,          # if set, sl/tp options are ATR multiples (scale-agnostic)
        idle_penalty_pips: float = 0.0,      # OPTIONAL: penalty per bar spent flat (breaks HOLD-collapse)
        # ---- Drawdown-optimization knobs (all default to the legacy fixed-lot behavior) ----
        risk_pct_per_trade: float | None = None,  # e.g. 0.02 => risk 2% of current equity per SL distance
        max_lot_units: float | None = None,       # cap on dynamic position size (oz)
        min_lot_units: float = 1.0,               # floor on dynamic position size (oz)
        drawdown_penalty_weight: float = 0.0,     # reward penalty per 1% equity-dd growth (while in dd)
        final_dd_penalty_weight: float = 0.0,     # reward penalty per 1% of final total dd at episode end
        max_dd_pct: float | None = None,          # circuit breaker: stop OPENing once equity <= peak*(1-x)
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)

        if feature_columns is None:
            self.feature_columns = list(self.df.columns)  # fallback: everything
        else:
            self.feature_columns = list(feature_columns)

        if sl_options is None or tp_options is None:
            raise ValueError("sl_options and tp_options must be provided (e.g. [15,20,30]).")
        self.sl_options = list(sl_options)
        self.tp_options = list(tp_options)
        self.atr_col = None
        if atr_col is not None:
            if atr_col in self.df.columns:
                self.atr_col = atr_col
            else:
                print(f"[Env] WARNING: atr_col '{atr_col}' not in df columns; using raw-pip SL/TP.")

        if self.n_steps <= window_size + 2:
            raise ValueError("Dataframe is too short for the given window_size.")

        self.window_size = int(window_size)
        self.asset_name = str(asset_name)
        if pip_value is None:
            pip_value = tick_size
        if lot_size is None:
            lot_size = contract_size
        self.tick_size = float(tick_size)
        self.pip_value = float(pip_value)
        self.contract_size = float(contract_size)
        self.lot_size = float(lot_size)

        # Friction
        self.spread_pips = float(spread_pips)
        self.commission_pips = float(commission_pips)
        self.max_slippage_pips = float(max_slippage_pips)

        # Equity accounting (approx): for XAUUSD 1 pip (0.01) per 1 lot (100 oz) ≈ $1
        # pip_value (price) * lot_size (units) ≈ $ per 1.0 price move.
        # 1 pip = pip_value price move, so $/pip ≈ pip_value * lot_size.
        self.usd_per_pip = self.pip_value * self.lot_size

        # Reward handling
        self.reward_scale = float(reward_scale)
        self.unrealized_delta_weight = float(unrealized_delta_weight)
        self.hold_reward_weight = float(hold_reward_weight)
        self.hold_reward_weight = float(hold_reward_weight)
        self.open_penalty_pips = float(open_penalty_pips)
        self.time_penalty_pips = float(time_penalty_pips)
        self.idle_penalty_pips = float(idle_penalty_pips)

        # Drawdown-optimization knobs
        self.risk_pct_per_trade = risk_pct_per_trade
        self.max_lot_units = max_lot_units
        self.min_lot_units = float(min_lot_units)
        self.drawdown_penalty_weight = float(drawdown_penalty_weight)
        self.final_dd_penalty_weight = float(final_dd_penalty_weight)
        self.max_dd_pct = max_dd_pct

        # Episode handling
        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = episode_max_steps if episode_max_steps is None else int(episode_max_steps)

        # Optional normalization (fit on train only, pass arrays here)
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        self.allow_flip = bool(allow_flip)

        # --- Actions ---
        # 0: HOLD
        # 1: CLOSE
        # 2..: OPEN(direction, sl, tp)
        self.action_map = [("HOLD", None, None, None), ("CLOSE", None, None, None)]
        for direction in [0, 1]:  # 0=short, 1=long
            for sl in self.sl_options:
                for tp in self.tp_options:
                    self.action_map.append(("OPEN", direction, float(sl), float(tp)))

        self.action_space = spaces.Discrete(len(self.action_map))

        # Observation features: df columns + 3 state features
        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 3
        self.num_features = self.base_num_features + self.state_num_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32
        )

        # Internal state
        self._reset_state()

    # ----------------------------
    # Core Helpers
    # ----------------------------

    def _reset_state(self):
        self.current_step = 0
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        # Position state
        self.position = 0              # 0=flat, +1=long, -1=short
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0

        # Accounting
        self.initial_equity_usd = 10000.0
        self.equity_usd = self.initial_equity_usd
        self.peak_equity_usd = self.initial_equity_usd
        self.prev_drawdown_pct = 0.0
        self.drawdown_breached = False
        self.episode_max_dd_pct = 0.0

        # Per-position sizing (fixed-lot fallback = self.lot_size)
        self.pos_lot_units = float(self.lot_size)

        # Logging
        self.equity_curve = []
        self.last_trade_info = None

    def _get_state_features(self):
        # position in [-1,0,1], time normalized, unrealized in pips (scaled)
        pos = float(self.position)
        t_norm = float(self.time_in_trade) / 1000.0
        unreal_pips = float(self._compute_unrealized_pips()) if self.position != 0 else 0.0
        unreal_scaled = unreal_pips / 100.0  # prevent huge magnitudes
        return np.array([pos, t_norm, unreal_scaled], dtype=np.float32)

    def _compute_unrealized_pips(self):
        if self.position == 0 or self.entry_price is None:
            return 0.0
        close_price = float(self.df.loc[self.current_step, "Close"])
        if self.position == 1:
            pnl_price = close_price - self.entry_price
        else:
            pnl_price = self.entry_price - close_price
        return pnl_price / self.pip_value

    def _apply_optional_normalization(self, obs: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            return obs
        mean = self.feature_mean.reshape(1, 1, -1)
        std = self.feature_std.reshape(1, 1, -1)
        std = np.where(std == 0, 1.0, std)
        return (obs - mean) / std

    def _get_observation(self):
        start = self.current_step - self.window_size
        if start < 0:
            start = 0

        obs_df = self.df.iloc[start:self.current_step].copy()
        # use only selected feature columns for the agent
        obs_df = obs_df[self.feature_columns]

        # If empty (safety), use the first row repeated
        if len(obs_df) == 0:
            base = np.tile(self.df.iloc[0].values.astype(np.float32), (self.window_size, 1))
        else:
            base = obs_df.values.astype(np.float32)
            if base.shape[0] < self.window_size:
                pad_rows = self.window_size - base.shape[0]
                pad = np.tile(base[0], (pad_rows, 1))
                base = np.vstack([pad, base])

        # Append state features (same for each row)
        state_feat = self._get_state_features()
        state_block = np.tile(state_feat, (self.window_size, 1))
        obs = np.hstack([base, state_block]).astype(np.float32)

        # Optional normalization (only if user passes train-fitted mean/std matching obs dims)
        obs = self._apply_optional_normalization(obs)

        return obs

    def _sample_slippage_pips(self) -> float:
        if self.max_slippage_pips <= 0:
            return 0.0
        return float(np.random.uniform(0.0, self.max_slippage_pips))

    def _cost_pips_round_trip(self) -> float:
        # Simple friction model (round-trip)
        return self.spread_pips + self.commission_pips

    def _open_position(self, direction: int, sl_pips: float, tp_pips: float):
        # Entry on current close + slippage; costs applied on close (round-trip model)
        close_price = float(self.df.loc[self.current_step, "Close"])

        # If atr_col is set, sl/tp args are ATR multiples -> convert to pips
        if self.atr_col is not None:
            atr_now = float(self.df.loc[self.current_step, self.atr_col])
            sl_pips = float(sl_pips) * atr_now / self.pip_value
            tp_pips = float(tp_pips) * atr_now / self.pip_value

        # Risk-% sizing: choose lot so that hitting SL loses risk_pct of CURRENT equity.
        # Dollars at risk per oz = sl_pips * pip_value; units = equity*risk% / that.
        if self.risk_pct_per_trade is not None:
            dollars_at_risk_per_oz = max(sl_pips * self.pip_value, 1e-9)
            units = (self.equity_usd * self.risk_pct_per_trade) / dollars_at_risk_per_oz
            units = max(units, self.min_lot_units)
            if self.max_lot_units is not None:
                units = min(units, float(self.max_lot_units))
            self.pos_lot_units = float(units)
        else:
            self.pos_lot_units = float(self.lot_size)

        slip_pips = self._sample_slippage_pips()
        slip_price = slip_pips * self.pip_value

        if direction == 1:  # long
            entry = close_price + slip_price
            sl_price = entry - sl_pips * self.pip_value
            tp_price = entry + tp_pips * self.pip_value
            self.position = 1
        else:               # short
            entry = close_price - slip_price
            sl_price = entry + sl_pips * self.pip_value
            tp_price = entry - tp_pips * self.pip_value
            self.position = -1

        self.entry_price = entry
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0

        self.last_trade_info = {
            "event": "OPEN",
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "lot_units": float(self.pos_lot_units),
        }

    def _close_position(self, reason: str, exit_price: float):
        # Realized pips
        if self.position == 1:
            pnl_price = exit_price - self.entry_price
        else:
            pnl_price = self.entry_price - exit_price
        realized_pips = pnl_price / self.pip_value

        # Costs in pips (round-trip)
        cost_pips = self._cost_pips_round_trip()
        net_pips = realized_pips - cost_pips

        # Update equity in USD (per-position sizing)
        pos_usd_per_pip = self.pip_value * self.pos_lot_units
        pnl_usd = net_pips * pos_usd_per_pip
        self.equity_usd += pnl_usd

        trade_info = {
            "event": "CLOSE",
            "reason": reason,
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "realized_pips": float(realized_pips),
            "cost_pips": float(cost_pips),
            "net_pips": float(net_pips),
            "lot_units": float(self.pos_lot_units),
            "pnl_usd": float(pnl_usd),
            "equity_usd": float(self.equity_usd),
            "time_in_trade": int(self.time_in_trade),
        }

        # Reset position state
        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0

        self.last_trade_info = trade_info
        return net_pips

    def _check_sl_tp_intrabar_and_maybe_close(self) -> float:
        """
        Checks SL/TP on the *next bar* range [Low, High].
        Conservative rule if both touched: assume SL hits first (worst case).
        Returns realized net pips if closed; otherwise None.
        """
        if self.position == 0:
            return None

        # If last bar, close on close
        if self.current_step >= self.n_steps - 2:
            exit_price = float(self.df.loc[self.current_step, "Close"])
            net_pips = self._close_position("END_OF_DATA", exit_price)
            return net_pips

        next_high = float(self.df.loc[self.current_step + 1, "High"])
        next_low = float(self.df.loc[self.current_step + 1, "Low"])

        if self.position == 1:
            sl_hit = next_low <= self.sl_price
            tp_hit = next_high >= self.tp_price
            if sl_hit and tp_hit:
                # conservative: SL first
                return self._close_position("SL_AND_TP_SAME_BAR_SL_FIRST", self.sl_price)
            elif sl_hit:
                return self._close_position("SL_HIT", self.sl_price)
            elif tp_hit:
                return self._close_position("TP_HIT", self.tp_price)
        else:
            sl_hit = next_high >= self.sl_price
            tp_hit = next_low <= self.tp_price
            if sl_hit and tp_hit:
                return self._close_position("SL_AND_TP_SAME_BAR_SL_FIRST", self.sl_price)
            elif sl_hit:
                return self._close_position("SL_HIT", self.sl_price)
            elif tp_hit:
                return self._close_position("TP_HIT", self.tp_price)

        return None

    # ----------------------------
    # Gym API
    # ----------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._reset_state()

        # Choose start
        if self.random_start:
            max_start = self.n_steps - max(self.min_episode_steps, self.window_size) - 2
            if max_start <= self.window_size:
                self.current_step = self.window_size
            else:
                self.current_step = int(np.random.randint(self.window_size, max_start))
        else:
            self.current_step = self.window_size

        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        obs = self._get_observation()

        if _GYMNASIUM:
            return obs, {}
        return obs

    def step(self, action: int):
        if self.terminated or self.truncated:
            # If someone steps after done, just return current obs with 0 reward
            obs = self._get_observation()
            if _GYMNASIUM:
                return obs, 0.0, True, False, {}
            return obs, 0.0, True, {}

        self.steps_in_episode += 1

        # Reward components
        reward_pips = 0.0
        info = {}

        act_type, direction, sl_pips, tp_pips = self.action_map[int(action)]

        # 1) Apply action logic
        if act_type == "HOLD":
            pass
        elif act_type == "CLOSE":
            if self.position != 0:
                # Close at current close (with slippage)
                close_price = float(self.df.loc[self.current_step, "Close"])
                slip_pips = self._sample_slippage_pips()
                slip_price = slip_pips * self.pip_value
                exit_price = close_price - slip_price if self.position == 1 else close_price + slip_price
                reward_pips += self._close_position("MANUAL_CLOSE", exit_price)

        elif act_type == "OPEN":
            if self.position == 0 and not self.drawdown_breached:
                self._open_position(direction=direction, sl_pips=sl_pips, tp_pips=tp_pips)
                # penalty for opening a trade to discourage overtrading
                reward_pips -= self.open_penalty_pips
            else:
                if self.allow_flip:
                    close_price = float(self.df.loc[self.current_step, "Close"])
                    reward_pips += self._close_position("FLIP_CLOSE", close_price)
                    self._open_position(direction=direction, sl_pips=sl_pips, tp_pips=tp_pips)
                    reward_pips -= self.open_penalty_pips


        # 2) If position is open, check SL/TP on next bar intrabar
        realized_now = self._check_sl_tp_intrabar_and_maybe_close()
        if realized_now is not None:
            reward_pips += realized_now

        # 2c) Drawdown tracking, DD-penalty shaping, and circuit breaker
        if self.equity_usd > self.peak_equity_usd:
            self.peak_equity_usd = float(self.equity_usd)
        dd_now = max(1.0 - self.equity_usd / self.peak_equity_usd, 0.0) if self.peak_equity_usd > 0 else 0.0
        self.episode_max_dd_pct = max(self.episode_max_dd_pct, dd_now)
        if self.drawdown_penalty_weight != 0.0:
            dd_growth = max(dd_now - self.prev_drawdown_pct, 0.0)
            reward_pips -= self.drawdown_penalty_weight * 100.0 * dd_growth
        if self.max_dd_pct is not None and not self.drawdown_breached and dd_now > self.max_dd_pct:
            self.drawdown_breached = True
            if self.position != 0:
                close_price = float(self.df.loc[self.current_step, "Close"])
                slip_pips = self._sample_slippage_pips()
                slip_price = slip_pips * self.pip_value
                exit_price = close_price - slip_price if self.position == 1 else close_price + slip_price
                reward_pips += self._close_position("MAX_DD_CIRCUIT_BREAKER", exit_price)
        self.prev_drawdown_pct = dd_now

        # 2b) Idle penalty: bar spent flat (breaks the "never trade" collapse)
        if self.position == 0:
            reward_pips -= self.idle_penalty_pips

        # 3) If still open, apply reward shaping based on delta-unrealized pips
        # 3) If still open, apply reward shaping
        if self.position != 0:
            self.time_in_trade += 1

            unreal_now = self._compute_unrealized_pips()
            delta_unreal = unreal_now - self.prev_unrealized_pips

            # (a) small bonus for holding a winning trade
            #     proportional to current unrealized profit
            if unreal_now > 0:
                reward_pips += self.hold_reward_weight * unreal_now

            # (b) optional shaping on change in unrealized (can keep small or zero)
            if self.unrealized_delta_weight != 0.0:
                reward_pips += self.unrealized_delta_weight * delta_unreal

            # (c) small time cost per bar in a trade to avoid infinite holding
            reward_pips -= self.time_penalty_pips

            self.prev_unrealized_pips = unreal_now



        # 4) Advance time
        self.current_step += 1

        # 5) Termination / truncation
        if self.current_step >= self.n_steps - 1:
            self.terminated = True

        if self.episode_max_steps is not None and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True

        # 6) Log equity
        self.equity_curve.append(float(self.equity_usd))

        # 7) Build observation
        obs = self._get_observation()

        # 8) Final reward scaling
        if (self.terminated or self.truncated) and self.final_dd_penalty_weight != 0.0:
            reward_pips -= self.final_dd_penalty_weight * 100.0 * self.episode_max_dd_pct
        reward = float(reward_pips) * self.reward_scale

        # 9) Info
        info.update({
            "equity_usd": float(self.equity_usd),
            "position": int(self.position),
            "time_in_trade": int(self.time_in_trade),
            "reward_pips": float(reward_pips),
            "drawdown_pct": float(dd_now),
            "episode_max_dd_pct": float(self.episode_max_dd_pct),
            "drawdown_breached": bool(self.drawdown_breached),
            "peak_equity_usd": float(self.peak_equity_usd),
            "last_trade_info": self.last_trade_info
        })

        if _GYMNASIUM:
            return obs, reward, self.terminated, self.truncated, info
        else:
            done = bool(self.terminated or self.truncated)
            return obs, reward, done, info

    def render(self):
        print(
            f"Step={self.current_step} | Equity=${self.equity_usd:,.2f} | "
            f"Pos={self.position} | Entry={self.entry_price} | SL={self.sl_price} | TP={self.tp_price}"
        )

# Gold-focused alias (XAU/USD)
GoldTradingEnv = ForexTradingEnv

