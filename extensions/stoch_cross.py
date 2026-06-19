import pandas as pd
import numpy as np
from datetime import datetime

STRATEGY_NAME = "Stoch-Cross"

DEFAULT_CONFIG = {
    "atr_period": 9,
    "lookback": 80,
    "stoch_k_period": 14,
    "stoch_d_period": 3,
    "stoch_overbought": 80.0,
    "stoch_oversold": 20.0,
    "extreme_overbought": 90.0,
    "extreme_oversold": 10.0,
    "tp_multiplier": 2.0,
    "sl_multiplier": 1.5,
    "risk_percent": 10.0,
    "atr_threshold": 60.0,
    "use_atr_spike_exit": True,
    "use_extreme_stoch_exit": True,
}

# ==========================================
# Indicators
# ==========================================
def calculate_atr(high, low, close, period=14):
    high_low    = high - low
    high_close  = (high - close.shift(1)).abs()
    low_close   = (low  - close.shift(1)).abs()
    tr  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low    = low.rolling(window=k_period).min()
    highest_high  = high.rolling(window=k_period).max()
    range_diff    = highest_high - lowest_low
    raw_k = np.where(range_diff == 0, 50.0, ((close - lowest_low) / range_diff) * 100.0)
    stoch_k = pd.Series(raw_k, index=close.index)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d


def calculate_normalized_atr(atr_series, lookback=80):
    rolling_min = atr_series.rolling(window=lookback).min()
    rolling_max = atr_series.rolling(window=lookback).max()
    range_diff  = rolling_max - rolling_min
    norm_atr    = np.where(range_diff == 0, 0.0, ((atr_series - rolling_min) / range_diff) * 100.0)
    return pd.Series(norm_atr, index=atr_series.index)


def calculate_lot_size(balance, risk_percent, sl_dist, tick_value, tick_size, min_lot, max_lot, lot_step):
    if tick_value <= 0 or tick_size <= 0 or sl_dist <= 0:
        return min_lot
    risk_amount  = balance * (risk_percent / 100.0)
    loss_per_lot = (sl_dist / tick_size) * tick_value
    lot = risk_amount / loss_per_lot
    if lot_step > 0:
        lot = np.floor(lot / lot_step) * lot_step
    return float(max(min_lot, min(max_lot, lot)))


# ==========================================
# 1-Day Backtest Optimization
# Grid search: stoch_ob, stoch_os, tp_mult, sl_mult
# ==========================================
def run_stoch_1d_optimization(candles, atr_period, lookback, stoch_k_period, stoch_d_period,
                               balance, tick_value, tick_size, min_lot, max_lot, lot_step):
    if len(candles) < lookback + 5:
        return None

    df = pd.DataFrame(candles)
    df["time"] = pd.to_numeric(df["time"])
    df = df.sort_values(by="time").reset_index(drop=True)

    last_time  = df["time"].max()
    one_day_ago = last_time - 24 * 3600
    mask = df["time"] >= one_day_ago
    start_idx = df[mask].index[0] if mask.any() else 0

    # Compute indicators on full df (for warm-up), slice for simulation
    atr      = calculate_atr(df["high"], df["low"], df["close"], atr_period)
    norm_atr = calculate_normalized_atr(atr, lookback)
    stoch_k, stoch_d = calculate_stochastic(df["high"], df["low"], df["close"], stoch_k_period, stoch_d_period)

    k_vals        = stoch_k.values[start_idx:]
    norm_atr_vals = norm_atr.values[start_idx:]
    close_vals    = df["close"].values[start_idx:]
    high_vals     = df["high"].values[start_idx:]
    low_vals      = df["low"].values[start_idx:]
    atr_vals      = atr.values[start_idx:]

    n = len(k_vals)
    if n < 5:
        return None

    ob_list     = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0]
    os_list     = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    tp_list     = [1.0, 1.5, 2.0, 2.5, 3.0]
    sl_list     = [1.0, 1.5, 2.0]
    atr_thresh  = 60.0

    best_profit = -999999.0
    best_params = None

    for ob in ob_list:
        for os in os_list:
            if os >= ob:
                continue
            for tp_m in tp_list:
                for sl_m in sl_list:
                    sim_balance   = balance
                    peak_balance  = balance
                    max_dd        = 0.0
                    trades_count  = 0
                    winning_trades = 0

                    in_position = False
                    pos_type    = None
                    entry_price = 0.0
                    sl_level    = 0.0
                    tp_level    = 0.0
                    lot_size    = min_lot

                    for i in range(2, n):
                        k_curr     = k_vals[i - 1]
                        k_prev     = k_vals[i - 2]
                        cur_natr   = norm_atr_vals[i - 1]
                        close_p    = close_vals[i - 1]

                        if in_position:
                            hit           = False
                            profit_points = 0.0

                            if pos_type == "BUY":
                                if low_vals[i - 1] <= sl_level:
                                    profit_points = sl_level - entry_price
                                    hit = True
                                elif high_vals[i - 1] >= tp_level:
                                    profit_points = tp_level - entry_price
                                    hit = True
                            else:  # SELL
                                if high_vals[i - 1] >= sl_level:
                                    profit_points = entry_price - sl_level
                                    hit = True
                                elif low_vals[i - 1] <= tp_level:
                                    profit_points = entry_price - tp_level
                                    hit = True

                            if hit:
                                profit_money  = (profit_points / tick_size) * tick_value * lot_size
                                sim_balance  += profit_money
                                peak_balance  = max(peak_balance, sim_balance)
                                dd = ((peak_balance - sim_balance) / peak_balance) * 100.0 if peak_balance > 0 else 0.0
                                max_dd        = max(max_dd, dd)
                                trades_count += 1
                                if profit_money > 0:
                                    winning_trades += 1
                                in_position = False
                                pos_type    = None
                        else:
                            if cur_natr < atr_thresh:
                                cross_ob = (k_prev >= ob and k_curr < ob)
                                cross_os = (k_prev <= os and k_curr > os)

                                atr_val  = atr_vals[i - 1]
                                sl_dist  = atr_val * sl_m

                                if cross_ob:
                                    in_position = True
                                    pos_type    = "SELL"
                                    entry_price = close_p
                                    lot_size    = calculate_lot_size(sim_balance, 10.0, sl_dist, tick_value, tick_size, min_lot, max_lot, lot_step)
                                    sl_level    = entry_price + sl_dist
                                    tp_level    = entry_price - (atr_val * tp_m)
                                elif cross_os:
                                    in_position = True
                                    pos_type    = "BUY"
                                    entry_price = close_p
                                    lot_size    = calculate_lot_size(sim_balance, 10.0, sl_dist, tick_value, tick_size, min_lot, max_lot, lot_step)
                                    sl_level    = entry_price - sl_dist
                                    tp_level    = entry_price + (atr_val * tp_m)

                    net_profit = sim_balance - balance
                    win_rate   = (winning_trades / trades_count * 100) if trades_count > 0 else 0.0

                    if max_dd <= 20.0 and trades_count >= 1:
                        if net_profit > best_profit:
                            best_profit = net_profit
                            best_params = {
                                "stoch_overbought": ob,
                                "stoch_oversold":   os,
                                "tp_multiplier":    tp_m,
                                "sl_multiplier":    sl_m,
                                "win_rate":         win_rate,
                                "max_drawdown":     max_dd,
                                "trades_count":     trades_count,
                                "best_profit":      net_profit,
                            }

    return best_params


# ==========================================
# Core Strategy Logic
# ==========================================
def process_strategy(data, config, add_log_fn):
    candles          = data.get("candles", [])
    positions        = data.get("positions", [])
    symbol           = data.get("symbol", "UNKNOWN")
    timeframe        = data.get("timeframe", "UNKNOWN")
    is_new_bar       = bool(data.get("is_new_bar", False))
    trigger_backtest = bool(data.get("trigger_backtest", False))

    atr_period     = int(config.get("atr_period", DEFAULT_CONFIG["atr_period"]))
    lookback       = int(config.get("lookback", DEFAULT_CONFIG["lookback"]))
    stoch_k_period = int(config.get("stoch_k_period", DEFAULT_CONFIG["stoch_k_period"]))
    stoch_d_period = int(config.get("stoch_d_period", DEFAULT_CONFIG["stoch_d_period"]))
    atr_threshold  = float(config.get("atr_threshold", DEFAULT_CONFIG["atr_threshold"]))
    risk_percent   = float(config.get("risk_percent", DEFAULT_CONFIG["risk_percent"]))

    balance    = float(data.get("balance", 10000.0))
    tick_value = float(data.get("tick_value", 1.0))
    tick_size  = float(data.get("tick_size", 0.00001))
    min_lot    = float(data.get("min_lot", 0.01))
    max_lot    = float(data.get("max_lot", 100.0))
    lot_step   = float(data.get("lot_step", 0.01))

    if len(candles) < lookback + 5:
        return {"action": "NONE", "message": f"Insufficient bars: {len(candles)}/{lookback + 5}"}, config, {}, None

    # ── 1-Day Backtest Optimization (on every new bar or manual trigger) ──
    updated_config = config.copy()
    bt_res = None

    if is_new_bar or trigger_backtest:
        bt_res = run_stoch_1d_optimization(
            candles, atr_period, lookback, stoch_k_period, stoch_d_period,
            balance, tick_value, tick_size, min_lot, max_lot, lot_step
        )
        if bt_res:
            updated_config["stoch_overbought"] = bt_res["stoch_overbought"]
            updated_config["stoch_oversold"]   = bt_res["stoch_oversold"]
            updated_config["tp_multiplier"]    = bt_res["tp_multiplier"]
            updated_config["sl_multiplier"]    = bt_res["sl_multiplier"]

    # ── Build DataFrame & Indicators ──────────────────────────────────────
    df = pd.DataFrame(candles)
    df["time"] = pd.to_numeric(df["time"])
    df = df.sort_values(by="time").reset_index(drop=True)

    atr      = calculate_atr(df["high"], df["low"], df["close"], atr_period)
    norm_atr = calculate_normalized_atr(atr, lookback)
    stoch_k, stoch_d = calculate_stochastic(df["high"], df["low"], df["close"], stoch_k_period, stoch_d_period)

    k_curr        = stoch_k.iloc[-2]
    k_prev        = stoch_k.iloc[-3]
    d_curr        = stoch_d.iloc[-2]
    current_atr   = atr.iloc[-2]
    current_norm_atr = norm_atr.iloc[-2]

    cfg_ob     = float(updated_config.get("stoch_overbought", DEFAULT_CONFIG["stoch_overbought"]))
    cfg_os     = float(updated_config.get("stoch_oversold",   DEFAULT_CONFIG["stoch_oversold"]))
    cfg_ex_ob  = float(updated_config.get("extreme_overbought", DEFAULT_CONFIG["extreme_overbought"]))
    cfg_ex_os  = float(updated_config.get("extreme_oversold",   DEFAULT_CONFIG["extreme_oversold"]))
    cfg_tp     = float(updated_config.get("tp_multiplier", DEFAULT_CONFIG["tp_multiplier"]))
    cfg_sl     = float(updated_config.get("sl_multiplier", DEFAULT_CONFIG["sl_multiplier"]))

    cross_ob = (k_prev >= cfg_ob and k_curr < cfg_ob)
    cross_os = (k_prev <= cfg_os and k_curr > cfg_os)

    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if is_new_bar:
        add_log_fn(
            f"Stoch — [{symbol} {timeframe}] "
            f"%K: {k_curr:.2f} (prev: {k_prev:.2f}) | %D: {d_curr:.2f} | "
            f"NormATR: {current_norm_atr:.1f}%"
        )
        if bt_res:
            add_log_fn(
                f"1D Opt — OB: {cfg_ob:.0f} / OS: {cfg_os:.0f} | "
                f"TP: {cfg_tp:.1f} / SL: {cfg_sl:.1f} | "
                f"Trades: {bt_res['trades_count']} | WR: {bt_res['win_rate']:.1f}% | "
                f"Profit: {bt_res['best_profit']:.2f} | DD: {bt_res['max_drawdown']:.1f}%"
            )

    # ── Signal text ───────────────────────────────────────────────────────
    signal_text = "NONE"
    if current_norm_atr < atr_threshold:
        if cross_ob:
            signal_text = "SELL SIGNAL"
        elif cross_os:
            signal_text = "BUY SIGNAL"
        elif k_curr > cfg_ob:
            signal_text = "WAIT CROSSDOWN (OB)"
        elif k_curr < cfg_os:
            signal_text = "WAIT CROSSUP (OS)"
        else:
            signal_text = "WAIT STOCH"
    else:
        signal_text = "NO TRADE (ATR high)"

    display_line1 = f"Active Config — Stoch OB/OS: {cfg_ob:.0f} / {cfg_os:.0f}"
    display_line2 = f"Active Config — TP: {cfg_tp:.1f} | SL: {cfg_sl:.1f}"

    res_dict = {
        "norm_atr":         float(current_norm_atr),
        "stoch_k":          float(k_curr),
        "stoch_d":          float(d_curr),
        "signal_text":      signal_text,
        "stoch_overbought": cfg_ob,
        "stoch_oversold":   cfg_os,
        "tp_multiplier":    cfg_tp,
        "sl_multiplier":    cfg_sl,
        "update_time":      current_time_str,
        "display_line1":    display_line1,
        "display_line2":    display_line2,
    }

    if bt_res:
        res_dict["bt_total_trades"] = bt_res["trades_count"]
        res_dict["bt_win_rate"]     = bt_res["win_rate"]
        res_dict["bt_total_profit"] = bt_res["best_profit"]
        res_dict["bt_max_drawdown"] = bt_res["max_drawdown"]
        res_dict["opt_ob"]          = bt_res["stoch_overbought"]
        res_dict["opt_os"]          = bt_res["stoch_oversold"]
        res_dict["opt_tp"]          = bt_res["tp_multiplier"]
        res_dict["opt_sl"]          = bt_res["sl_multiplier"]

    action_dict = {"action": "NONE"}

    if is_new_bar:
        if len(positions) > 0:
            for pos in positions:
                ticket   = pos.get("ticket")
                pos_type = pos.get("type")
                profit   = pos.get("profit", 0)

                atr_spike = (
                    bool(updated_config.get("use_atr_spike_exit", True))
                    and (current_norm_atr >= atr_threshold * 1.5)
                )

                extreme_exit = False
                if bool(updated_config.get("use_extreme_stoch_exit", True)) and profit > 0:
                    if pos_type == "SELL":
                        if k_prev < cfg_ex_os and k_curr > cfg_ex_os:
                            extreme_exit = True
                    elif pos_type == "BUY":
                        if k_prev > cfg_ex_ob and k_curr < cfg_ex_ob:
                            extreme_exit = True

                if atr_spike or extreme_exit:
                    reason = "ATR Spike" if atr_spike else "Extreme Stoch Crossover"
                    action_dict = {
                        "action": "CLOSE",
                        "ticket": ticket,
                        "reason": reason,
                    }
                    break
        else:
            if current_norm_atr < atr_threshold:
                if cross_ob:
                    sl_dist  = current_atr * cfg_sl
                    lot_size = calculate_lot_size(balance, risk_percent, sl_dist, tick_value, tick_size, min_lot, max_lot, lot_step)
                    action_dict = {
                        "action":       "SELL",
                        "atr":          current_atr,
                        "tp_multiplier": cfg_tp,
                        "sl_multiplier": cfg_sl,
                        "lot":          round(lot_size, 2),
                        "reason":       "Stoch %K crossed down from Overbought",
                    }
                elif cross_os:
                    sl_dist  = current_atr * cfg_sl
                    lot_size = calculate_lot_size(balance, risk_percent, sl_dist, tick_value, tick_size, min_lot, max_lot, lot_step)
                    action_dict = {
                        "action":       "BUY",
                        "atr":          current_atr,
                        "tp_multiplier": cfg_tp,
                        "sl_multiplier": cfg_sl,
                        "lot":          round(lot_size, 2),
                        "reason":       "Stoch %K crossed up from Oversold",
                    }

    res_dict.update(action_dict)

    live_metrics = {
        "Stoch %K":  f"{k_curr:.2f}",
        "Stoch %D":  f"{d_curr:.2f}",
        "NormATR":   f"{current_norm_atr:.1f}%",
    }

    return res_dict, updated_config, live_metrics, bt_res
