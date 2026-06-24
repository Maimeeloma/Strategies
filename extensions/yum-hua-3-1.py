import pandas as pd
import numpy as np
from datetime import datetime

STRATEGY_NAME = "Yum Hua Trends Gold - 3.1"

DEFAULT_CONFIG = {
    "atr_period": 9,
    "lookback": 80,
    "regression_period": 50,
    "channel_dev_multiplier": 2.0,
    "tp_multiplier": 1.0,
    "sl_multiplier": 2.0,
    "risk_percent": 5.0,
    "atr_threshold": 80.0,
    "use_center_exit": True,
    "use_opposite_channel_exit": True,
    "use_atr_spike_exit": True,
    "auto_optimize_on_new_bar": True,
    "sideway_slope_threshold": 0.2
}

# ==========================================
# Indicators Calculation (Exact matching to server)
# ==========================================
def calculate_atr(high, low, close, period=14):
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/period, adjust=False).mean()
    return atr

def calculate_rsi(close, period=7):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_normalized_atr(atr_series, lookback=80):
    rolling_min = atr_series.rolling(window=lookback).min()
    rolling_max = atr_series.rolling(window=lookback).max()
    range_diff = rolling_max - rolling_min
    norm_atr = np.where(range_diff == 0, 0.0, ((atr_series - rolling_min) / range_diff) * 100.0)
    return pd.Series(norm_atr, index=atr_series.index)

def calculate_regression_channel(close_series, period=50, multiplier=2.0):
    close_vals = close_series.values
    size = len(close_vals)
    center = np.full(size, np.nan)
    upper = np.full(size, np.nan)
    lower = np.full(size, np.nan)
    slope = np.full(size, np.nan)
    
    if size < period:
        return (pd.Series(center, index=close_series.index), 
                pd.Series(upper, index=close_series.index), 
                pd.Series(lower, index=close_series.index), 
                pd.Series(slope, index=close_series.index))
        
    x = np.arange(period)
    x_sum = x.sum()
    x2_sum = (x**2).sum()
    denominator = period * x2_sum - x_sum**2
    
    shape = (size - period + 1, period)
    strides = (close_vals.strides[0], close_vals.strides[0])
    windows = np.lib.stride_tricks.as_strided(close_vals, shape=shape, strides=strides)
    
    sum_y = np.sum(windows, axis=1)
    sum_xy = np.sum(x * windows, axis=1)
    
    m = (period * sum_xy - x_sum * sum_y) / denominator
    c = (sum_y - m * x_sum) / period
    
    reg_val = m * (period - 1) + c
    
    expected = m[:, None] * x + c[:, None]
    errors = windows - expected
    std_dev = np.sqrt(np.mean(errors**2, axis=1))
    
    center[period-1:] = reg_val
    upper[period-1:] = reg_val + multiplier * std_dev
    lower[period-1:] = reg_val - multiplier * std_dev
    slope[period-1:] = m
    
    return (pd.Series(center, index=close_series.index), 
            pd.Series(upper, index=close_series.index), 
            pd.Series(lower, index=close_series.index), 
            pd.Series(slope, index=close_series.index))

def calculate_lot_size(balance, risk_percent, sl_dist, tick_value, tick_size, min_lot, max_lot, lot_step):
    if tick_value <= 0 or tick_size <= 0 or sl_dist <= 0:
        return min_lot
    risk_amount = balance * (risk_percent / 100.0)
    loss_per_lot = (sl_dist / tick_size) * tick_value
    lot = risk_amount / loss_per_lot
    if lot_step > 0:
        lot = np.floor(lot / lot_step) * lot_step
    return float(max(min_lot, min(max_lot, lot)))

# ==========================================
# Grid Search Backtest Optimization
# ==========================================
def run_backtest_optimization(candles, atr_period, lookback, balance, tick_value, tick_size, min_lot, max_lot, lot_step, config):
    if len(candles) < lookback + 10:
        return None
        
    df = pd.DataFrame(candles)
    df['time'] = pd.to_numeric(df['time'])
    df = df.sort_values(by='time').reset_index(drop=True)
    
    # Calculate indicators
    atr = calculate_atr(df['high'], df['low'], df['close'], atr_period)
    norm_atr = calculate_normalized_atr(atr, lookback)
    
    norm_atr_vals = norm_atr.values
    close_vals = df['close'].values
    high_vals = df['high'].values
    low_vals = df['low'].values
    atr_vals = atr.values
    
    risk_percent = float(config.get("risk_percent", 5.0))
    atr_threshold = float(config.get("atr_threshold", 80.0))
    use_center_exit = bool(config.get("use_center_exit", True))
    use_opp_exit = bool(config.get("use_opposite_channel_exit", True))
    use_atr_spike = bool(config.get("use_atr_spike_exit", True))
    sideway_threshold = float(config.get("sideway_slope_threshold", 0.2))
    
    period_list = [30, 40, 50, 60, 80]
    dev_mult_list = [1.5, 1.8, 2.0, 2.2, 2.5]
    tp_mult_list = [0.5, 1.0, 1.5, 2.0, 3.0]
    sl_mult_list = [1.0, 1.5, 2.0, 2.5]
    
    best_profit = -999999.0
    best_params = None
    
    # Precalculate channels to speed up optimization loop
    channels_cache = {}
    for period in period_list:
        for dev_mult in dev_mult_list:
            center, upper, lower, slope = calculate_regression_channel(df['close'], period, dev_mult)
            channels_cache[(period, dev_mult)] = (center.values, upper.values, lower.values, slope.values)
            
    for period in period_list:
        for dev_mult in dev_mult_list:
            center_vals, upper_vals, lower_vals, slope_vals = channels_cache[(period, dev_mult)]
            
            for tp_m in tp_mult_list:
                for sl_m in sl_mult_list:
                    sim_balance = balance
                    peak_balance = balance
                    max_dd = 0.0
                    trades_count = 0
                    winning_trades = 0
                    
                    in_position = False
                    pos_type = None
                    entry_price = 0.0
                    sl_level = 0.0
                    tp_level = 0.0
                    lot_size = min_lot
                    
                    for i in range(period + 1, len(df)):
                        current_norm_atr = norm_atr_vals[i-1]
                        close_price = close_vals[i-1]
                        high_price = high_vals[i-1]
                        low_price = low_vals[i-1]
                        
                        center_curr = center_vals[i-1]
                        upper_curr = upper_vals[i-1]
                        lower_curr = lower_vals[i-1]
                        slope_curr = slope_vals[i-1]
                        
                        is_sideway = abs(slope_curr) < sideway_threshold
                        
                        if in_position:
                            hit = False
                            profit_money = 0.0
                            
                            if pos_type == "BUY":
                                if low_price <= sl_level:
                                    profit_points = sl_level - entry_price
                                    profit_money = (profit_points / tick_size) * tick_value * lot_size
                                    hit = True
                                elif high_price >= tp_level:
                                    profit_points = tp_level - entry_price
                                    profit_money = (profit_points / tick_size) * tick_value * lot_size
                                    hit = True
                            elif pos_type == "SELL":
                                if high_price >= sl_level:
                                    profit_points = entry_price - sl_level
                                    profit_money = (profit_points / tick_size) * tick_value * lot_size
                                    hit = True
                                elif low_price <= tp_level:
                                    profit_points = entry_price - tp_level
                                    profit_money = (profit_points / tick_size) * tick_value * lot_size
                                    hit = True
                                    
                            if not hit:
                                center_exit = False
                                if use_center_exit and not is_sideway:
                                    if pos_type == "BUY" and close_price >= center_curr:
                                        center_exit = True
                                    elif pos_type == "SELL" and close_price <= center_curr:
                                        center_exit = True
                                        
                                opp_channel_exit = False
                                if use_opp_exit:
                                    if pos_type == "BUY" and close_price >= upper_curr:
                                        opp_channel_exit = True
                                    elif pos_type == "SELL" and close_price <= lower_curr:
                                        opp_channel_exit = True
                                        
                                atr_spike_exit = use_atr_spike and (current_norm_atr >= atr_threshold * 1.5)
                                
                                if center_exit or opp_channel_exit or atr_spike_exit:
                                    if pos_type == "BUY":
                                        profit_points = close_price - entry_price
                                    else:
                                        profit_points = entry_price - close_price
                                    profit_money = (profit_points / tick_size) * tick_value * lot_size
                                    hit = True
                                    
                            if hit:
                                sim_balance += profit_money
                                peak_balance = max(peak_balance, sim_balance)
                                dd = ((peak_balance - sim_balance) / peak_balance) * 100.0 if peak_balance > 0 else 0.0
                                max_dd = max(max_dd, dd)
                                trades_count += 1
                                if profit_money > 0:
                                    winning_trades += 1
                                in_position = False
                                pos_type = None
                        else:
                            if current_norm_atr < atr_threshold:
                                if is_sideway:
                                    buy_signal = (low_price <= lower_curr)
                                    sell_signal = (high_price >= upper_curr)
                                else:
                                    buy_signal = (slope_curr > 0) and (low_price <= lower_curr)
                                    sell_signal = (slope_curr < 0) and (high_price >= upper_curr)
                                
                                if buy_signal:
                                    in_position = True
                                    pos_type = "BUY"
                                    entry_price = close_price
                                    atr_val = atr_vals[i-1]
                                    sl_dist = atr_val * sl_m
                                    lot_size = calculate_lot_size(sim_balance, risk_percent, sl_dist, tick_value, tick_size, min_lot, max_lot, lot_step)
                                    sl_level = entry_price - sl_dist
                                    tp_level = entry_price + (atr_val * tp_m)
                                elif sell_signal:
                                    in_position = True
                                    pos_type = "SELL"
                                    entry_price = close_price
                                    atr_val = atr_vals[i-1]
                                    sl_dist = atr_val * sl_m
                                    lot_size = calculate_lot_size(sim_balance, risk_percent, sl_dist, tick_value, tick_size, min_lot, max_lot, lot_step)
                                    sl_level = entry_price + sl_dist
                                    tp_level = entry_price - (atr_val * tp_m)
                                    
                    net_profit = sim_balance - balance
                    win_rate = (winning_trades / trades_count * 100) if trades_count > 0 else 0.0
                    
                    if max_dd <= 15.0 and trades_count >= 3:
                        if net_profit > best_profit:
                            best_profit = net_profit
                            best_params = {
                                "regression_period": period,
                                "channel_dev_multiplier": dev_mult,
                                "tp_multiplier": tp_m,
                                "sl_multiplier": sl_m,
                                "win_rate": win_rate,
                                "max_drawdown": max_dd,
                                "trades_count": trades_count,
                                "best_profit": net_profit
                            }
                                    
    return best_params

# ==========================================
# Manual Scalp Backtest Engine (1-Day)
# ==========================================
def run_manual_scalp_backtest(candles, balance, tick_value, tick_size, min_lot, max_lot, lot_step, config):
    if len(candles) < 10:
        return None
        
    df = pd.DataFrame(candles)
    df['time'] = pd.to_numeric(df['time'])
    df = df.sort_values(by='time').reset_index(drop=True)
    
    last_time = df['time'].max()
    one_day_ago = last_time - 24 * 3600
    df_1d = df[df['time'] >= one_day_ago].reset_index(drop=True)
    start_idx = df[df['time'] >= one_day_ago].index[0] if len(df[df['time'] >= one_day_ago]) > 0 else 0
    
    if len(df_1d) < 10:
        df_1d = df
        start_idx = 0
        
    atr_period = int(config.get("atr_period", 9))
    lookback = int(config.get("lookback", 80))
    regression_period = int(config.get("regression_period", 50))
    dev_mult = float(config.get("channel_dev_multiplier", 2.0))
    atr_threshold = float(config.get("atr_threshold", 80.0))
    risk_percent = float(config.get("risk_percent", 5.0))
    
    use_center_exit = bool(config.get("use_center_exit", True))
    use_opp_exit = bool(config.get("use_opposite_channel_exit", True))
    use_atr_spike = bool(config.get("use_atr_spike_exit", True))
    sideway_threshold = float(config.get("sideway_slope_threshold", 0.2))
    
    atr = calculate_atr(df['high'], df['low'], df['close'], atr_period)
    norm_atr = calculate_normalized_atr(atr, lookback)
    center, upper, lower, slope = calculate_regression_channel(df['close'], regression_period, dev_mult)
    
    norm_atr_vals = norm_atr.values[start_idx:]
    close_vals = df['close'].values[start_idx:]
    high_vals = df['high'].values[start_idx:]
    low_vals = df['low'].values[start_idx:]
    atr_vals = atr.values[start_idx:]
    
    center_vals = center.values[start_idx:]
    upper_vals = upper.values[start_idx:]
    lower_vals = lower.values[start_idx:]
    slope_vals = slope.values[start_idx:]
    
    tp_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    sl_list = [1.0, 1.5, 2.0, 2.5]
    
    best_buy_res = None
    best_sell_res = None
    
    # Grid search for BUY
    best_buy_profit = -999999.0
    for tp_m in tp_list:
        for sl_m in sl_list:
            sim_balance = balance
            peak_balance = balance
            max_dd = 0.0
            trades_count = 0
            winning_trades = 0
            
            in_position = False
            entry_price = 0.0
            sl_level = 0.0
            tp_level = 0.0
            lot_size = min_lot
            
            for i in range(2, len(df_1d)):
                if np.isnan(center_vals[i-1]):
                    continue
                    
                current_norm_atr = norm_atr_vals[i-1]
                close_price = close_vals[i-1]
                high_price = high_vals[i-1]
                low_price = low_vals[i-1]
                
                center_curr = center_vals[i-1]
                upper_curr = upper_vals[i-1]
                lower_curr = lower_vals[i-1]
                slope_curr = slope_vals[i-1]
                
                is_sideway = abs(slope_curr) < sideway_threshold
                
                if in_position:
                    hit = False
                    profit_money = 0.0
                    
                    if low_price <= sl_level:
                        profit_points = sl_level - entry_price
                        profit_money = (profit_points / tick_size) * tick_value * lot_size
                        hit = True
                    elif high_price >= tp_level:
                        profit_points = tp_level - entry_price
                        profit_money = (profit_points / tick_size) * tick_value * lot_size
                        hit = True
                        
                    if not hit:
                        center_exit = use_center_exit and not is_sideway and (close_price >= center_curr)
                        opp_channel_exit = use_opp_exit and (close_price >= upper_curr)
                        atr_spike_exit = use_atr_spike and (current_norm_atr >= atr_threshold * 1.5)
                        
                        if center_exit or opp_channel_exit or atr_spike_exit:
                            profit_points = close_price - entry_price
                            profit_money = (profit_points / tick_size) * tick_value * lot_size
                            hit = True
                            
                    if hit:
                        sim_balance += profit_money
                        peak_balance = max(peak_balance, sim_balance)
                        dd = ((peak_balance - sim_balance) / peak_balance) * 100.0 if peak_balance > 0 else 0.0
                        max_dd = max(max_dd, dd)
                        trades_count += 1
                        if profit_money > 0:
                            winning_trades += 1
                        in_position = False
                else:
                    if current_norm_atr < atr_threshold:
                        buy_signal = (low_price <= lower_curr) if is_sideway else ((slope_curr > 0) and (low_price <= lower_curr))
                        if buy_signal:
                            in_position = True
                            entry_price = close_price
                            atr_val = atr_vals[i-1]
                            sl_dist = atr_val * sl_m
                            lot_size = calculate_lot_size(sim_balance, risk_percent, sl_dist, tick_value, tick_size, min_lot, max_lot, lot_step)
                            sl_level = entry_price - sl_dist
                            tp_level = entry_price + (atr_val * tp_m)
                            
            net_profit = sim_balance - balance
            win_rate = (winning_trades / trades_count * 100) if trades_count > 0 else 0.0
            
            if max_dd <= 20.0 and trades_count >= 1:
                if net_profit > best_buy_profit:
                    best_buy_profit = net_profit
                    best_buy_res = {
                        "direction": "BUY",
                        "tp_multiplier": tp_m,
                        "sl_multiplier": sl_m,
                        "profit": float(round(net_profit, 2)),
                        "win_rate": float(round(win_rate, 2)),
                        "max_drawdown": float(round(max_dd, 2)),
                        "trades": trades_count
                    }
                    
    # Grid search for SELL
    best_sell_profit = -999999.0
    for tp_m in tp_list:
        for sl_m in sl_list:
            sim_balance = balance
            peak_balance = balance
            max_dd = 0.0
            trades_count = 0
            winning_trades = 0
            
            in_position = False
            entry_price = 0.0
            sl_level = 0.0
            tp_level = 0.0
            lot_size = min_lot
            
            for i in range(2, len(df_1d)):
                if np.isnan(center_vals[i-1]):
                    continue
                    
                current_norm_atr = norm_atr_vals[i-1]
                close_price = close_vals[i-1]
                high_price = high_vals[i-1]
                low_price = low_vals[i-1]
                
                center_curr = center_vals[i-1]
                upper_curr = upper_vals[i-1]
                lower_curr = lower_vals[i-1]
                slope_curr = slope_vals[i-1]
                
                is_sideway = abs(slope_curr) < sideway_threshold
                
                if in_position:
                    hit = False
                    profit_money = 0.0
                    
                    if high_price >= sl_level:
                        profit_points = entry_price - sl_level
                        profit_money = (profit_points / tick_size) * tick_value * lot_size
                        hit = True
                    elif low_price <= tp_level:
                        profit_points = entry_price - tp_level
                        profit_money = (profit_points / tick_size) * tick_value * lot_size
                        hit = True
                        
                    if not hit:
                        center_exit = use_center_exit and not is_sideway and (close_price <= center_curr)
                        opp_channel_exit = use_opp_exit and (close_price <= lower_curr)
                        atr_spike_exit = use_atr_spike and (current_norm_atr >= atr_threshold * 1.5)
                        
                        if center_exit or opp_channel_exit or atr_spike_exit:
                            profit_points = entry_price - close_price
                            profit_money = (profit_points / tick_size) * tick_value * lot_size
                            hit = True
                            
                    if hit:
                        sim_balance += profit_money
                        peak_balance = max(peak_balance, sim_balance)
                        dd = ((peak_balance - sim_balance) / peak_balance) * 100.0 if peak_balance > 0 else 0.0
                        max_dd = max(max_dd, dd)
                        trades_count += 1
                        if profit_money > 0:
                            winning_trades += 1
                        in_position = False
                else:
                    if current_norm_atr < atr_threshold:
                        sell_signal = (high_price >= upper_curr) if is_sideway else ((slope_curr < 0) and (high_price >= upper_curr))
                        if sell_signal:
                            in_position = True
                            entry_price = close_price
                            atr_val = atr_vals[i-1]
                            sl_dist = atr_val * sl_m
                            lot_size = calculate_lot_size(sim_balance, risk_percent, sl_dist, tick_value, tick_size, min_lot, max_lot, lot_step)
                            sl_level = entry_price + sl_dist
                            tp_level = entry_price - (atr_val * tp_m)
                            
            net_profit = sim_balance - balance
            win_rate = (winning_trades / trades_count * 100) if trades_count > 0 else 0.0
            
            if max_dd <= 20.0 and trades_count >= 1:
                if net_profit > best_sell_profit:
                    best_sell_profit = net_profit
                    best_sell_res = {
                        "direction": "SELL",
                        "tp_multiplier": tp_m,
                        "sl_multiplier": sl_m,
                        "profit": float(round(net_profit, 2)),
                        "win_rate": float(round(win_rate, 2)),
                        "max_drawdown": float(round(max_dd, 2)),
                        "trades": trades_count
                    }
                    
    buy_score = 0.0
    if best_buy_res:
        buy_score = best_buy_res["win_rate"] * (best_buy_res["win_rate"]/100 * best_buy_res["trades"]) - best_buy_res["max_drawdown"]
    
    sell_score = 0.0
    if best_sell_res:
        sell_score = best_sell_res["win_rate"] * (best_sell_res["win_rate"]/100 * best_sell_res["trades"]) - best_sell_res["max_drawdown"]
        
    if best_buy_res is None and best_sell_res is None:
        return {
            "direction": "NONE",
            "tp_multiplier": 0.0,
            "sl_multiplier": 0.0,
            "profit": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "trades": 0
        }
    elif best_buy_res is not None and best_sell_res is None:
        return best_buy_res
    elif best_sell_res is not None and best_buy_res is None:
        return best_sell_res
    else:
        if buy_score >= sell_score:
            return best_buy_res
        else:
            return best_sell_res

# ==========================================
# Core Strategy Logic
# ==========================================
def process_strategy(data, config, add_log_fn):
    candles = data.get("candles", [])
    positions = data.get("positions", [])
    trigger_backtest = bool(data.get("trigger_backtest", False))
    symbol = data.get("symbol", "UNKNOWN")
    timeframe = data.get("timeframe", "UNKNOWN")
    is_new_bar = bool(data.get("is_new_bar", False))
    
    updated_config = config.copy()
    
    atr_period = int(updated_config.get("atr_period", 9))
    lookback = int(updated_config.get("lookback", 80))
    atr_threshold = float(updated_config.get("atr_threshold", 80.0))
    risk_percent = float(updated_config.get("risk_percent", 5.0))
    
    balance = float(data.get("balance", 10000.0))
    tick_value = float(data.get("tick_value", 1.0))
    tick_size = float(data.get("tick_size", 0.00001))
    min_lot = float(data.get("min_lot", 0.01))
    max_lot = float(data.get("max_lot", 100.0))
    lot_step = float(data.get("lot_step", 0.01))
    
    if len(candles) < lookback + 10:
        return {"action": "NONE", "message": f"Insufficient bars: {len(candles)}/{lookback + 10}"}, config, {}, None

    bt_res = None
    auto_opt = bool(updated_config.get("auto_optimize_on_new_bar", False))
    if trigger_backtest or (is_new_bar and auto_opt):
        bt_res = run_backtest_optimization(candles, atr_period, lookback, balance, tick_value, tick_size, min_lot, max_lot, lot_step, updated_config)
        if bt_res:
            updated_config["regression_period"] = bt_res["regression_period"]
            updated_config["channel_dev_multiplier"] = bt_res["channel_dev_multiplier"]
            updated_config["tp_multiplier"] = bt_res["tp_multiplier"]
            updated_config["sl_multiplier"] = bt_res["sl_multiplier"]
            
            # Recalculate parameters since config was optimized
            atr_threshold = float(updated_config.get("atr_threshold", 80.0))
            risk_percent = float(updated_config.get("risk_percent", 5.0))

    df = pd.DataFrame(candles)
    df['time'] = pd.to_numeric(df['time'])
    df = df.sort_values(by='time').reset_index(drop=True)
    
    atr = calculate_atr(df['high'], df['low'], df['close'], atr_period)
    norm_atr = calculate_normalized_atr(atr, lookback)
    
    reg_period = int(updated_config.get("regression_period", 50))
    dev_mult = float(updated_config.get("channel_dev_multiplier", 2.0))
    
    center, upper, lower, slope = calculate_regression_channel(df['close'], reg_period, dev_mult)
    
    center_curr = center.iloc[-2]
    upper_curr = upper.iloc[-2]
    lower_curr = lower.iloc[-2]
    slope_curr = slope.iloc[-2]
    
    current_atr = atr.iloc[-2]
    current_norm_atr = norm_atr.iloc[-2]
    
    cfg_tp_mult = float(updated_config.get("tp_multiplier", 1.0))
    cfg_sl_mult = float(updated_config.get("sl_multiplier", 2.0))
    use_center_exit = bool(updated_config.get("use_center_exit", True))
    use_opp_exit = bool(updated_config.get("use_opposite_channel_exit", True))
    
    close_curr = df['close'].iloc[-2]
    high_curr = df['high'].iloc[-2]
    low_curr = df['low'].iloc[-2]
    
    sideway_threshold = float(updated_config.get("sideway_slope_threshold", 0.2))
    is_sideway = abs(slope_curr) < sideway_threshold
    
    if is_sideway:
        buy_signal = (low_curr <= lower_curr)
        sell_signal = (high_curr >= upper_curr)
    else:
        buy_signal = (slope_curr > 0) and (low_curr <= lower_curr)
        sell_signal = (slope_curr < 0) and (high_curr >= upper_curr)
    
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if is_new_bar:
        add_log_fn(f"Calculations - [{symbol} {timeframe}] Slope: {slope_curr:.6f} | Center: {center_curr:.2f} | Upper: {upper_curr:.2f} | Lower: {lower_curr:.2f} | NormATR: {current_norm_atr:.1f}%")
 
    signal_text = "NONE"
    if current_norm_atr < atr_threshold:
        if sell_signal:
            signal_text = "SELL SIGNAL"
        elif buy_signal:
            signal_text = "BUY SIGNAL"
        elif is_sideway:
            signal_text = "SIDEWAY WAIT"
        elif slope_curr > 0:
            signal_text = "WAIT REBOUND BUY"
        elif slope_curr < 0:
            signal_text = "WAIT REBOUND SELL"
        else:
            signal_text = "WAIT SLOPE"
    else:
        signal_text = "NO TRADE (ATR high)"

    display_line1 = f"Active Config — Reg Period: {reg_period} | Dev Mult: {dev_mult:.2f}"
    display_line2 = f"Active Config — TP Multiplier: {cfg_tp_mult:.1f} | SL Multiplier: {cfg_sl_mult:.1f}"

    res_dict = {
        "norm_atr": float(current_norm_atr),
        "center_val": float(center_curr) if not np.isnan(center_curr) else 0.0,
        "upper_val": float(upper_curr) if not np.isnan(upper_curr) else 0.0,
        "lower_val": float(lower_curr) if not np.isnan(lower_curr) else 0.0,
        "slope": float(slope_curr) if not np.isnan(slope_curr) else 0.0,
        "signal_text": signal_text,
        "regression_period": reg_period,
        "channel_dev_multiplier": dev_mult,
        "tp_multiplier": cfg_tp_mult,
        "sl_multiplier": cfg_sl_mult,
        "update_time": current_time_str,
        "display_line1": display_line1,
        "display_line2": display_line2
    }
    
    if bt_res:
        res_dict["bt_total_trades"] = bt_res["trades_count"]
        res_dict["bt_win_rate"] = bt_res["win_rate"]
        res_dict["bt_total_profit"] = bt_res["best_profit"]
        res_dict["bt_max_drawdown"] = bt_res["max_drawdown"]
        res_dict["opt_period"] = bt_res["regression_period"]
        res_dict["opt_dev"] = bt_res["channel_dev_multiplier"]
        res_dict["opt_tp"] = bt_res["tp_multiplier"]
        res_dict["opt_sl"] = bt_res["sl_multiplier"]

    action_dict = {"action": "NONE"}
    
    # Filter active positions for the current symbol only
    symbol_positions = [pos for pos in positions if pos.get("symbol") == symbol]

    if len(symbol_positions) > 0:
        for pos in symbol_positions:
            ticket = pos.get("ticket")
            pos_type = pos.get("type")
            profit = float(pos.get("profit", 0.0))

            # 2. Standard/Indicator Exits (Only checked on new bar)
            if is_new_bar:
                center_exit = False
                if use_center_exit and not is_sideway:
                    if pos_type == "BUY" and close_curr >= center_curr:
                        center_exit = True
                    elif pos_type == "SELL" and close_curr <= center_curr:
                        center_exit = True

                opp_channel_exit = False
                if use_opp_exit:
                    if pos_type == "BUY" and close_curr >= upper_curr:
                        opp_channel_exit = True
                    elif pos_type == "SELL" and close_curr <= lower_curr:
                        opp_channel_exit = True

                atr_spike = bool(updated_config.get("use_atr_spike_exit", True)) and (current_norm_atr >= atr_threshold * 1.5)

                if center_exit or opp_channel_exit or atr_spike:
                    reason = "Center Line Exit" if center_exit else ("Opposite Channel Exit" if opp_channel_exit else "ATR Spike")
                    action_dict = {
                        "action": "CLOSE",
                        "ticket": ticket,
                        "reason": reason
                    }
                    break
    else:
        # 3. Entries: Checked ONLY on new bar
        if is_new_bar:
            if current_norm_atr < atr_threshold:
                if sell_signal:
                    sl_dist = current_atr * cfg_sl_mult
                    lot_size = calculate_lot_size(balance, risk_percent, sl_dist, tick_value, tick_size, min_lot, max_lot, lot_step)
                    reason_str = "Price touched Upper Channel (Flat Trend)" if is_sideway else "Price touched Upper Channel (Resistance) in Downtrend"
                    action_dict = {
                        "action": "SELL",
                        "atr": current_atr,
                        "tp_multiplier": cfg_tp_mult,
                        "sl_multiplier": cfg_sl_mult,
                        "lot": round(lot_size, 2),
                        "reason": reason_str
                    }
                elif buy_signal:
                    sl_dist = current_atr * cfg_sl_mult
                    lot_size = calculate_lot_size(balance, risk_percent, sl_dist, tick_value, tick_size, min_lot, max_lot, lot_step)
                    reason_str = "Price touched Lower Channel (Flat Trend)" if is_sideway else "Price touched Lower Channel (Support) in Uptrend"
                    action_dict = {
                        "action": "BUY",
                        "atr": current_atr,
                        "tp_multiplier": cfg_tp_mult,
                        "sl_multiplier": cfg_sl_mult,
                        "lot": round(lot_size, 2),
                        "reason": reason_str
                    }

    res_dict.update(action_dict)
    
    # Customize HUD lines for better feedback
    if res_dict.get("action") == "NONE":
        res_dict["display_line1"] = f"Slope: {slope_curr:.6f} | Center: {center_curr:.2f}"
        if len(symbol_positions) > 0:
            p = symbol_positions[0]
            res_dict["display_line2"] = f"Holding {p.get('type')} #{p.get('ticket')} | Profit: ${p.get('profit', 0.0):.2f}"
        else:
            res_dict["display_line2"] = f"Signal: {signal_text}"
    else:
        res_dict["display_line1"] = f"Action: {res_dict.get('action')}"
        res_dict["display_line2"] = f"Reason: {res_dict.get('reason')}"
    
    # UI live metrics list to render dynamically
    live_metrics = {
        "Slope": f"{slope_curr:.6f}",
        "Center": f"{center_curr:.2f}",
        "Upper": f"{upper_curr:.2f}",
        "Lower": f"{lower_curr:.2f}",
        "NormATR": f"{current_norm_atr:.1f}%"
    }
    
    return res_dict, updated_config, live_metrics, bt_res
