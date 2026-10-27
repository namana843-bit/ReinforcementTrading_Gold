import numpy as np
import pandas as pd


def compute_metrics(equity_curve, trades=None, initial_equity=10000.0):
    eq = np.asarray(equity_curve, dtype=float)
    if eq.size == 0:
        return {}
    total_return_pct = (eq[-1] / eq[0] - 1.0) * 100.0 if eq[0] != 0 else 0.0
    pnl = float(eq[-1] - eq[0])
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak == 0, 1.0, peak)
    mdd_pct = float(dd.max() * 100.0) if dd.size else 0.0
    returns = np.diff(eq) / np.where(eq[:-1] == 0, 1.0, eq[:-1])
    returns = returns[np.isfinite(returns)]
    if returns.size > 1 and returns.std() > 0:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252.0))
    else:
        sharpe = 0.0
    downside = returns[returns < 0]
    if downside.size > 1 and downside.std() > 0:
        sortino = float(returns.mean() / downside.std() * np.sqrt(252.0))
    else:
        sortino = 0.0
    trades = trades or []
    n_trades = len(trades)
    wins = [t for t in trades if float(t.get("net_pips", 0.0)) > 0]
    win_rate = (len(wins) / n_trades * 100.0) if n_trades else 0.0
    avg_win = float(np.mean([t["net_pips"] for t in wins])) if wins else 0.0
    losses = [t for t in trades if float(t.get("net_pips", 0.0)) <= 0]
    avg_loss = float(np.mean([t["net_pips"] for t in losses])) if losses else 0.0
    profit_factor = (sum(t["net_pips"] for t in wins) / abs(sum(t["net_pips"] for t in losses))) if losses and sum(t["net_pips"] for t in losses) != 0 else float("inf")
    return {
        "initial_equity": float(eq[0]),
        "final_equity": float(eq[-1]),
        "total_pnl": pnl,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": mdd_pct,
        "sharpe": sharpe,
        "sortino": sortino,
        "total_trades": n_trades,
        "win_rate_pct": win_rate,
        "avg_win_pips": avg_win,
        "avg_loss_pips": avg_loss,
        "profit_factor": profit_factor,
    }


def print_metrics(m):
    print("=== Performance Metrics ===")
    for k, v in m.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print("===========================")
