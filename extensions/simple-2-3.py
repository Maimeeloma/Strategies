import pandas as pd
import numpy as np
from datetime import datetime

STRATEGY_NAME = "RSI ATR SCALP"

DEFAULT_CONFIG = {
    "atr_period": 9,
    "lookback": 80,
    "rsi_period": 7,
    "rsi_overbought": 65.0,
    "rsi_oversold": 15.0,
    "extreme_overbought": 82.0,
    "extreme_oversold": 27.0,
    "tp_multiplier": 9.0,
    "sl_multiplier": 1.5,
    "risk_percent": 10.0,
    "atr_threshold": 60.0,
    "use_rsi_neutral_exit": False,
    "rsi_neutral": 50.0,
    "use_atr_spike_exit": True,
    "use_extreme_rsi_exit": True
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
def run_backtest_optimization(candles, atr_period, lookback, rsi_period, balance, tick_value, tick_size, min_lot, max_lot, lot_step):
    if len(candles) < lookback + 5:
        return None
        
    df = pd.DataFrame(candles)
    df['time'] = pd.to_numeric(df['time'])
    df = df.sort_values(by='time').reset_index(drop=True)
    
    # Calculate indicators
    atr = calculate_atr(df['high'], df['low'], df['close'], atr_period)
    norm_atr = calculate_normalized_atr(atr, lookback)
    rsi = calculate_rsi(df['close'], rsi_period)
    
    rsi_vals = rsi.values
    norm_atr_vals = norm_atr.values
    close_vals = df['close'].values
    high_vals = df['high'].values
    low_vals = df['low'].values
    atr_vals = atr.values
    
    rsi_ob_list = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0]
    rsi_os_list = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    tp_mult_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0]
    sl_mult_list = [1.0, 1.5, 2.0, 3.0, 5.0]
    extreme_os_list = [10.0, 20.0, 30.0, 40.0]
    extreme_ob_list = [60.0, 70.0, 80.0, 90.0]
    
    best_profit = -999999.0
    best_params = None
    
    for ob in rsi_ob_list:
        for os in rsi_os_list:
            for tp_m in tp_mult_list:
                for sl_m in sl_mult_list:
                    for ex_os in extreme_os_list:
                        for ex_ob in extreme_ob_list:
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
                            
                            for i in range(2, len(df)):
                                rsi_curr = rsi_vals[i-1]
                                rsi_prev = rsi_vals[i-2]
                                current_norm_atr = norm_atr_vals[i-1]
                                close_price = close_vals[i-1]
                                
                                if in_position:
                                    hit = False
                                    profit_points = 0.0
                                    
                                    if pos_type == "BUY":
                                        if low_vals[i-1] <= sl_level:
                                            profit_points = sl_level - entry_price
                                            hit = True
                                        elif high_vals[i-1] >= tp_level:
                                            profit_points = tp_level - entry_price
                                            hit = True
                                    elif pos_type == "SELL":
                                        if high_vals[i-1] >= sl_level:
                                            profit_points = entry_price - sl_level
                                            hit = True
                                        elif low_vals[i-1] <= tp_level:
                                            profit_points = entry_price - tp_level
                                            hit = True
                                            
                                    if hit:
                                        profit_money = (profit_points / tick_size) * tick_value * lot_size
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
                                    if current_norm_atr < 60.0:
                                        cross_ob = (rsi_prev >= ob and rsi_curr < ob)
                                        cross_os = (rsi_prev <= os and rsi_curr > os)
                                        
                                        if cross_ob:
                                            in_position = True
                                            pos_type = "SELL"
                                            entry_price = close_price
                                            atr_val = atr_vals[i-1]
                                            sl_dist = atr_val * sl_m
                                            lot_size = calculate_lot_size(sim_balance, 10.0, sl_dist, tick_value, tick_size, min_lot, max_lot, lot_step)
                                            sl_level = entry_price + sl_dist
                                            tp_level = entry_price - (atr_val * tp_m)
                                        elif cross_os:
                                            in_position = True
                                            pos_type = "BUY"
                                            entry_price = close_price
                                            atr_val = atr_vals[i-1]
                                            sl_dist = atr_val * sl_m
                                            lot_size = calculate_lot_size(sim_balance, 10.0, sl_dist, tick_value, tick_size, min_lot, max_lot, lot_step)
                                            sl_level = entry_price - sl_dist
                                            tp_level = entry_price + (atr_val * tp_m)
                                            
                            net_profit = sim_balance - balance
                            win_rate = (winning_trades / trades_count * 100) if trades_count > 0 else 0.0
                            
                            if max_dd <= 15.0 and trades_count >= 3:
                                if net_profit > best_profit:
                                    best_profit = net_profit
                                    best_params = {
                                        "rsi_overbought": ob,
                                        "rsi_oversold": os,
                                        "tp_multiplier": tp_m,
                                        "sl_multiplier": sl_m,
                                        "extreme_oversold": ex_os,
                                        "extreme_overbought": ex_ob,
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
    
    if len(df_1d) < 10:
        df_1d = df
        
    atr_period = int(config.get("atr_period", 9))
    lookback = int(config.get("lookback", 80))
    rsi_period = int(config.get("rsi_period", 7))
    atr_threshold = float(config.get("atr_threshold", 60.0))
    risk_percent = float(config.get("risk_percent", 10.0))
    rsi_os = float(config.get("rsi_oversold", 15.0))
    rsi_ob = float(config.get("rsi_overbought", 65.0))
    
    atr = calculate_atr(df['high'], df['low'], df['close'], atr_period)
    norm_atr = calculate_normalized_atr(atr, lookback)
    rsi = calculate_rsi(df['close'], rsi_period)
    
    start_idx = df[df['time'] >= one_day_ago].index[0] if len(df[df['time'] >= one_day_ago]) > 0 else 0
    
    rsi_vals = rsi.values[start_idx:]
    norm_atr_vals = norm_atr.values[start_idx:]
    close_vals = df['close'].values[start_idx:]
    high_vals = df['high'].values[start_idx:]
    low_vals = df['low'].values[start_idx:]
    atr_vals = atr.values[start_idx:]
    
    tp_list = [1.0, 1.5, 2.0, 2.5, 3.0]
    sl_list = [1.0, 1.5, 2.0]
    
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
                rsi_curr = rsi_vals[i-1]
                rsi_prev = rsi_vals[i-2]
                current_norm_atr = norm_atr_vals[i-1]
                close_price = close_vals[i-1]
                
                if in_position:
                    hit = False
                    profit_points = 0.0
                    
                    if low_vals[i-1] <= sl_level:
                        profit_points = sl_level - entry_price
                        hit = True
                    elif high_vals[i-1] >= tp_level:
                        profit_points = tp_level - entry_price
                        hit = True
                        
                    if hit:
                        profit_money = (profit_points / tick_size) * tick_value * lot_size
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
                        cross_os = (rsi_prev <= rsi_os and rsi_curr > rsi_os)
                        if cross_os:
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
                rsi_curr = rsi_vals[i-1]
                rsi_prev = rsi_vals[i-2]
                current_norm_atr = norm_atr_vals[i-1]
                close_price = close_vals[i-1]
                
                if in_position:
                    hit = False
                    profit_points = 0.0
                    
                    if high_vals[i-1] >= sl_level:
                        profit_points = entry_price - sl_level
                        hit = True
                    elif low_vals[i-1] <= tp_level:
                        profit_points = entry_price - tp_level
                        hit = True
                        
                    if hit:
                        profit_money = (profit_points / tick_size) * tick_value * lot_size
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
                        cross_ob = (rsi_prev >= rsi_ob and rsi_curr < rsi_ob)
                        if cross_ob:
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
    
    atr_period = int(config.get("atr_period", 9))
    lookback = int(config.get("lookback", 80))
    rsi_period = int(config.get("rsi_period", 7))
    atr_threshold = float(config.get("atr_threshold", 60.0))
    risk_percent = float(config.get("risk_percent", 10.0))
    
    balance = float(data.get("balance", 10000.0))
    tick_value = float(data.get("tick_value", 1.0))
    tick_size = float(data.get("tick_size", 0.00001))
    min_lot = float(data.get("min_lot", 0.01))
    max_lot = float(data.get("max_lot", 100.0))
    lot_step = float(data.get("lot_step", 0.01))
    
    if len(candles) < lookback + 5:
        return {"action": "NONE", "message": f"Insufficient bars: {len(candles)}/{lookback + 5}"}, config, {}, None

    updated_config = config.copy()
    bt_res = None
    if True:
        bt_res = run_backtest_optimization(candles, atr_period, lookback, rsi_period, balance, tick_value, tick_size, min_lot, max_lot, lot_step)
        if bt_res:
            updated_config["rsi_overbought"] = bt_res["rsi_overbought"]
            updated_config["rsi_oversold"] = bt_res["rsi_oversold"]
            updated_config["tp_multiplier"] = bt_res["tp_multiplier"]
            updated_config["sl_multiplier"] = bt_res["sl_multiplier"]
            updated_config["extreme_oversold"] = bt_res["extreme_oversold"]
            updated_config["extreme_overbought"] = bt_res["extreme_overbought"]

    df = pd.DataFrame(candles)
    df['time'] = pd.to_numeric(df['time'])
    df = df.sort_values(by='time').reset_index(drop=True)
    
    atr = calculate_atr(df['high'], df['low'], df['close'], atr_period)
    norm_atr = calculate_normalized_atr(atr, lookback)
    rsi = calculate_rsi(df['close'], rsi_period)
    
    rsi_curr = rsi.iloc[-2]
    rsi_prev = rsi.iloc[-3]
    current_atr = atr.iloc[-2]
    current_norm_atr = norm_atr.iloc[-2]
    
    cfg_ob = float(updated_config.get("rsi_overbought", 65.0))
    cfg_os = float(updated_config.get("rsi_oversold", 15.0))
    cfg_tp_mult = float(updated_config.get("tp_multiplier", 9.0))
    cfg_sl_mult = float(updated_config.get("sl_multiplier", 1.5))
    cfg_ex_ob = float(updated_config.get("extreme_overbought", 82.0))
    cfg_ex_os = float(updated_config.get("extreme_oversold", 27.0))
    
    cross_ob = (rsi_prev >= cfg_ob and rsi_curr < cfg_ob)
    cross_os = (rsi_prev <= cfg_os and rsi_curr > cfg_os)
    
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if is_new_bar:
        add_log_fn(f"Calculations - [{symbol} {timeframe}] RSI: {rsi_curr:.2f} (prev: {rsi_prev:.2f}) | NormATR: {current_norm_atr:.1f}% | ATR: {current_atr:.5f}")

    signal_text = "NONE"
    if current_norm_atr < atr_threshold:
        if cross_ob:
            signal_text = "SELL SIGNAL"
        elif cross_os:
            signal_text = "BUY SIGNAL"
        elif rsi_curr > cfg_ob:
            signal_text = "WAIT CROSSDOWN (OB)"
        elif rsi_curr < cfg_os:
            signal_text = "WAIT CROSSUP (OS)"
        else:
            signal_text = "WAIT RSI"
    else:
        signal_text = "NO TRADE (ATR high)"

    display_line1 = f"Active Config — RSI OB/OS: {cfg_ob:.1f} / {cfg_os:.1f} | Extreme: {cfg_ex_ob:.1f} / {cfg_ex_os:.1f}"
    display_line2 = f"Active Config — TP Multiplier: {cfg_tp_mult:.1f} | SL Multiplier: {cfg_sl_mult:.1f}"

    res_dict = {
        "norm_atr": float(current_norm_atr),
        "rsi": float(rsi_curr),
        "signal_text": signal_text,
        "rsi_overbought": cfg_ob,
        "rsi_oversold": cfg_os,
        "tp_multiplier": cfg_tp_mult,
        "sl_multiplier": cfg_sl_mult,
        "extreme_oversold": cfg_ex_os,
        "extreme_overbought": cfg_ex_ob,
        "update_time": current_time_str,
        "display_line1": display_line1,
        "display_line2": display_line2
    }
    
    if bt_res:
        res_dict["bt_total_trades"] = bt_res["trades_count"]
        res_dict["bt_win_rate"] = bt_res["win_rate"]
        res_dict["bt_total_profit"] = bt_res["best_profit"]
        res_dict["bt_max_drawdown"] = bt_res["max_drawdown"]
        res_dict["opt_ob"] = bt_res["rsi_overbought"]
        res_dict["opt_os"] = bt_res["rsi_oversold"]
        res_dict["opt_tp"] = bt_res["tp_multiplier"]
        res_dict["opt_sl"] = bt_res["sl_multiplier"]

    action_dict = {"action": "NONE"}

    if is_new_bar:
        if len(positions) > 0:
            for pos in positions:
                ticket = pos.get("ticket")
                pos_type = pos.get("type")
                profit = pos.get("profit", 0)

                rsi_neutral = False
                if bool(updated_config.get("use_rsi_neutral_exit", False)):
                    neutral_line = float(updated_config.get("rsi_neutral", 50.0))
                    if pos_type == "BUY" and rsi_curr > neutral_line:
                        rsi_neutral = True
                    elif pos_type == "SELL" and rsi_curr < neutral_line:
                        rsi_neutral = True

                atr_spike = bool(updated_config.get("use_atr_spike_exit", True)) and (current_norm_atr >= atr_threshold * 1.5)

                extreme_exit = False
                if bool(updated_config.get("use_extreme_rsi_exit", True)) and profit > 0:
                    if pos_type == "SELL":
                        cross_up_ex_os = (rsi_prev < cfg_ex_os and rsi_curr > cfg_ex_os)
                        if cross_up_ex_os:
                            extreme_exit = True
                    elif pos_type == "BUY":
                        cross_down_ex_ob = (rsi_prev > cfg_ex_ob and rsi_curr < cfg_ex_ob)
                        if cross_down_ex_ob:
                            extreme_exit = True

                if rsi_neutral or atr_spike or extreme_exit:
                    reason = "RSI Neutral" if rsi_neutral else ("ATR Spike" if atr_spike else "Extreme RSI Crossover")
                    action_dict = {
                        "action": "CLOSE",
                        "ticket": ticket,
                        "reason": reason
                    }
                    break
        else:
            if current_norm_atr < atr_threshold:
                if cross_ob:
                    sl_dist = current_atr * cfg_sl_mult
                    lot_size = calculate_lot_size(balance, risk_percent, sl_dist, tick_value, tick_size, min_lot, max_lot, lot_step)
                    action_dict = {
                        "action": "SELL",
                        "atr": current_atr,
                        "tp_multiplier": cfg_tp_mult,
                        "sl_multiplier": cfg_sl_mult,
                        "lot": round(lot_size, 2),
                        "reason": "RSI crossed down from Overbought"
                    }
                elif cross_os:
                    sl_dist = current_atr * cfg_sl_mult
                    lot_size = calculate_lot_size(balance, risk_percent, sl_dist, tick_value, tick_size, min_lot, max_lot, lot_step)
                    action_dict = {
                        "action": "BUY",
                        "atr": current_atr,
                        "tp_multiplier": cfg_tp_mult,
                        "sl_multiplier": cfg_sl_mult,
                        "lot": round(lot_size, 2),
                        "reason": "RSI crossed up from Oversold"
                    }

    res_dict.update(action_dict)
    
    # UI live metrics list to render dynamically
    live_metrics = {
        "RSI (7)": f"{rsi_curr:.2f}",
        "NormATR": f"{current_norm_atr:.1f}%"
    }
    
    return res_dict, updated_config, live_metrics, bt_res
