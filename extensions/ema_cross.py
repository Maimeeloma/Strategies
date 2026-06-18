"""
EMA Cross Strategy Extension
─────────────────────────────
สัญญาณ: Fast EMA ตัดผ่าน Slow EMA
  - Fast EMA ตัดขึ้น → BUY
  - Fast EMA ตัดลง  → SELL

SL/TP คำนวณจาก ATR × Multiplier
Risk ต่อ trade = risk_percent % ของ Balance
"""

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# STRATEGY IDENTITY
# ──────────────────────────────────────────────
STRATEGY_NAME = "EMA Cross"

# ──────────────────────────────────────────────
# DEFAULT CONFIG  (แก้ได้จาก Dashboard)
# ──────────────────────────────────────────────
DEFAULT_CONFIG = {
    "fast_ema":      9,      # Fast EMA period
    "slow_ema":      21,     # Slow EMA period
    "atr_period":    14,     # ATR period
    "sl_multiplier": 1.5,    # SL = ATR × sl_multiplier
    "tp_multiplier": 2.5,    # TP = ATR × tp_multiplier
    "risk_percent":  1.0,    # % balance to risk per trade
}

# ══════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════
def process_strategy(data: dict, config: dict, add_log) -> tuple:
    candles     = data.get("candles", [])
    fast_p      = int(config.get("fast_ema",      DEFAULT_CONFIG["fast_ema"]))
    slow_p      = int(config.get("slow_ema",      DEFAULT_CONFIG["slow_ema"]))
    atr_period  = int(config.get("atr_period",    DEFAULT_CONFIG["atr_period"]))
    sl_mult     = float(config.get("sl_multiplier", DEFAULT_CONFIG["sl_multiplier"]))
    tp_mult     = float(config.get("tp_multiplier", DEFAULT_CONFIG["tp_multiplier"]))
    risk_pct    = float(config.get("risk_percent",  DEFAULT_CONFIG["risk_percent"]))

    min_bars = slow_p + 5
    if len(candles) < min_bars:
        return (
            {"action": "NONE", "signal_text": f"WAITING CANDLES ({len(candles)}/{min_bars})"},
            config, {}, None
        )

    # ── Build DataFrame ─────────────────────────────────────────────────
    df = pd.DataFrame(candles)
    df["time"]  = pd.to_numeric(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # ── Indicators ──────────────────────────────────────────────────────
    fast_ema = close.ewm(span=fast_p, adjust=False).mean()
    slow_ema = close.ewm(span=slow_p, adjust=False).mean()
    atr      = _atr(high, low, close, atr_period)

    current_atr  = float(atr.iloc[-1])   if not np.isnan(atr.iloc[-1])  else 0.001
    fast_cur     = float(fast_ema.iloc[-1])
    fast_prev    = float(fast_ema.iloc[-2])
    slow_cur     = float(slow_ema.iloc[-1])
    slow_prev    = float(slow_ema.iloc[-2])

    # ── Crossover Detection ─────────────────────────────────────────────
    golden_cross = (fast_prev <= slow_prev) and (fast_cur > slow_cur)   # bullish crossover
    death_cross  = (fast_prev >= slow_prev) and (fast_cur < slow_cur)   # bearish crossover

    # ── Market prices ───────────────────────────────────────────────────
    balance  = float(data.get("balance", 10000.0))
    bid      = float(data.get("bid", close.iloc[-1]))
    ask      = float(data.get("ask", close.iloc[-1]))

    tick_value = float(data.get("tick_value", 1.0))
    tick_size  = float(data.get("tick_size",  0.00001))
    min_lot    = float(data.get("min_lot",    0.01))
    max_lot    = float(data.get("max_lot",    100.0))
    lot_step   = float(data.get("lot_step",   0.01))

    # ── Build action ────────────────────────────────────────────────────
    res_dict = {"action": "NONE", "signal_text": "WAITING CROSS", "atr": current_atr}

    if golden_cross:
        lot = calculate_lot_size(balance, risk_pct, current_atr * sl_mult,
                                 tick_value, tick_size, min_lot, max_lot, lot_step)
        sl = round(ask - current_atr * sl_mult, 5)
        tp = round(ask + current_atr * tp_mult, 5)
        res_dict = {
            "action":      "BUY",
            "signal_text": "BUY SIGNAL",
            "lot":   round(lot, 2),
            "sl":    sl,
            "tp":    tp,
            "atr":   current_atr,
            "reason": f"EMA{fast_p} crossed above EMA{slow_p}"
        }
        add_log(f"GOLDEN CROSS — BUY | EMA{fast_p}={fast_cur:.5f} > EMA{slow_p}={slow_cur:.5f} | Lot={lot:.2f} SL={sl} TP={tp}")

    elif death_cross:
        lot = calculate_lot_size(balance, risk_pct, current_atr * sl_mult,
                                 tick_value, tick_size, min_lot, max_lot, lot_step)
        sl = round(bid + current_atr * sl_mult, 5)
        tp = round(bid - current_atr * tp_mult, 5)
        res_dict = {
            "action":      "SELL",
            "signal_text": "SELL SIGNAL",
            "lot":   round(lot, 2),
            "sl":    sl,
            "tp":    tp,
            "atr":   current_atr,
            "reason": f"EMA{fast_p} crossed below EMA{slow_p}"
        }
        add_log(f"DEATH CROSS — SELL | EMA{fast_p}={fast_cur:.5f} < EMA{slow_p}={slow_cur:.5f} | Lot={lot:.2f} SL={sl} TP={tp}")
    else:
        # No cross — just show direction
        direction = "ABOVE" if fast_cur > slow_cur else "BELOW"
        res_dict["signal_text"] = f"EMA{fast_p} {direction} EMA{slow_p}"

    # ── Live Metrics (แสดงใน Dashboard) ────────────────────────────────
    live_metrics = {
        f"EMA{fast_p}": round(fast_cur, 5),
        f"EMA{slow_p}": round(slow_cur, 5),
        "ATR":          round(current_atr, 5),
        "Trend":        "BULL" if fast_cur > slow_cur else "BEAR",
    }

    # ── Quick Backtest (walk-forward, last 200 bars) ────────────────────
    bt_result = _simple_backtest(df, fast_p, slow_p, atr_period,
                                 sl_mult, tp_mult, balance, risk_pct,
                                 tick_value, tick_size, min_lot, max_lot, lot_step)

    return res_dict, config, live_metrics, bt_result


# ══════════════════════════════════════════════
# REQUIRED — Lot size calculator
# ══════════════════════════════════════════════
def calculate_lot_size(balance, risk_pct, sl_distance,
                       tick_value, tick_size, min_lot, max_lot, lot_step):
    if sl_distance <= 0 or tick_size <= 0 or tick_value <= 0:
        return min_lot
    risk_amount = balance * (risk_pct / 100.0)
    pip_value   = tick_value / tick_size
    raw_lot     = risk_amount / (sl_distance * pip_value)
    lot = round(raw_lot / lot_step) * lot_step
    return max(min_lot, min(max_lot, lot))


# ══════════════════════════════════════════════
# PRIVATE HELPERS
# ══════════════════════════════════════════════
def _atr(high, low, close, period):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _simple_backtest(df, fast_p, slow_p, atr_period,
                     sl_mult, tp_mult, balance, risk_pct,
                     tick_value, tick_size, min_lot, max_lot, lot_step):
    """Walk-forward backtest on last 200 bars."""
    window = df.tail(200).reset_index(drop=True)
    if len(window) < slow_p + 5:
        return None

    close = window["close"]
    high  = window["high"]
    low   = window["low"]

    fast_ema = close.ewm(span=fast_p, adjust=False).mean()
    slow_ema = close.ewm(span=slow_p, adjust=False).mean()
    atr_s    = _atr(high, low, close, atr_period)

    trades      = []
    equity      = balance
    peak_equity = balance

    for i in range(slow_p + 1, len(window) - 1):
        f_cur  = fast_ema.iloc[i];   f_prv = fast_ema.iloc[i - 1]
        s_cur  = slow_ema.iloc[i];   s_prv = slow_ema.iloc[i - 1]
        atr_v  = atr_s.iloc[i]
        if np.isnan(atr_v) or atr_v <= 0:
            continue

        entry = close.iloc[i]
        direction = None
        if f_prv <= s_prv and f_cur > s_cur:
            direction = "BUY"
        elif f_prv >= s_prv and f_cur < s_cur:
            direction = "SELL"
        if direction is None:
            continue

        sl_dist = atr_v * sl_mult
        tp_dist = atr_v * tp_mult
        lot = calculate_lot_size(equity, risk_pct, sl_dist,
                                 tick_value, tick_size, min_lot, max_lot, lot_step)

        # Simulate next-bar outcome
        future = window.iloc[i + 1]
        pip_val = tick_value / tick_size if tick_size > 0 else 1

        if direction == "BUY":
            if future["low"] <= entry - sl_dist:
                pnl = -sl_dist * lot * pip_val
            elif future["high"] >= entry + tp_dist:
                pnl = tp_dist * lot * pip_val
            else:
                pnl = (future["close"] - entry) * lot * pip_val
        else:  # SELL
            if future["high"] >= entry + sl_dist:
                pnl = -sl_dist * lot * pip_val
            elif future["low"] <= entry - tp_dist:
                pnl = tp_dist * lot * pip_val
            else:
                pnl = (entry - future["close"]) * lot * pip_val

        equity += pnl
        peak_equity = max(peak_equity, equity)
        trades.append(pnl)

    if not trades:
        return None

    wins       = [p for p in trades if p > 0]
    win_rate   = len(wins) / len(trades) * 100
    total_pnl  = sum(trades)
    max_dd     = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0

    return {
        "trades_count": len(trades),
        "win_rate":     round(win_rate, 1),
        "best_profit":  round(total_pnl, 2),
        "max_drawdown": round(max_dd, 2),
    }
