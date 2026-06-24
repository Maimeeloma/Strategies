"""
Rapid Fire RSI Grid Scalper
─────────────────────────────
กลยุทธ์ยิงรัวเก็บกำไรสั้นๆ
ถ้าผิดทาง กางกริดยิงไม้แก้ (Layering)
ถ้ารวมกำไรเป็นบวก และ RSI พลิกกลับ = ปิดรวบตึง (Close All)
"""
import pandas as pd
import numpy as np
from datetime import datetime

STRATEGY_NAME = "Rapid Fire Scalper"

DEFAULT_CONFIG = {
    "rsi_period": 7,         # Fast RSI
    "rsi_ob": 70.0,          # จุดขาย
    "rsi_os": 30.0,          # จุดซื้อ
    "tp_multiplier": 0.8,    # TP แคบๆ (0.8 ATR)
    "sl_multiplier": 3.0,    # SL เผื่อสวิง (3.0 ATR)
    "max_layers": 5,         # ยิงไม้แก้สูงสุด 5 ไม้
    "grid_step_atr": 1.0,    # ระยะห่างแต่ละไม้ (1 ATR)
    "risk_percent": 1.0      # ความเสี่ยง 1% ต่อไม้
}

def calculate_atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

def calculate_rsi(close, period=7):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_lot_size(balance, risk_pct, sl_distance, tick_value, tick_size, min_lot, max_lot, lot_step):
    if sl_distance <= 0 or tick_size <= 0 or tick_value <= 0:
        return min_lot
    risk_amount = balance * (risk_pct / 100.0)
    pip_value   = tick_value / tick_size
    raw_lot     = risk_amount / (sl_distance * pip_value)
    lot = round(raw_lot / lot_step) * lot_step
    return max(min_lot, min(max_lot, lot))

def process_strategy(data, config, add_log_fn):
    candles = data.get("candles", [])
    positions = data.get("positions", [])
    symbol = data.get("symbol", "UNKNOWN")
    timeframe = data.get("timeframe", "M5")
    is_new_bar = bool(data.get("is_new_bar", False))
    trigger_backtest = bool(data.get("trigger_backtest", False))
    
    rsi_period = int(config.get("rsi_period", 7))
    rsi_ob = float(config.get("rsi_ob", 70.0))
    rsi_os = float(config.get("rsi_os", 30.0))
    tp_mult = float(config.get("tp_multiplier", 0.8))
    sl_mult = float(config.get("sl_multiplier", 3.0))
    max_layers = int(config.get("max_layers", 5))
    grid_step = float(config.get("grid_step_atr", 1.0))
    risk_pct = float(config.get("risk_percent", 1.0))
    
    balance = float(data.get("balance", 10000.0))
    tick_value = float(data.get("tick_value", 1.0))
    tick_size = float(data.get("tick_size", 0.00001))
    min_lot = float(data.get("min_lot", 0.01))
    max_lot = float(data.get("max_lot", 100.0))
    lot_step = float(data.get("lot_step", 0.01))
    
    if len(candles) < rsi_period + 5:
        return {"action": "NONE", "display_line1": "Initializing..."}, config, {}, None
        
    df = pd.DataFrame(candles)
    df["close"] = pd.to_numeric(df["close"])
    df["high"] = pd.to_numeric(df["high"])
    df["low"] = pd.to_numeric(df["low"])
    
    atr = calculate_atr(df["high"], df["low"], df["close"], 14)
    rsi = calculate_rsi(df["close"], rsi_period)
    
    curr_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0.001
    curr_rsi = float(rsi.iloc[-1])
    prev_rsi = float(rsi.iloc[-2])
    curr_close = float(df["close"].iloc[-1])
    
    # Analyze open positions for this symbol
    my_positions = [p for p in positions if p.get("symbol") == symbol]
    
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    signal_text = "HOLDING" if len(my_positions) > 0 else "WAITING"

    res_dict = {
        "rsi": float(curr_rsi),
        "atr": float(curr_atr),
        "signal_text": signal_text,
        "rsi_ob": rsi_ob,
        "rsi_os": rsi_os,
        "tp_multiplier": tp_mult,
        "sl_multiplier": sl_mult,
        "update_time": current_time_str,
    }
    action_dict = {"action": "NONE"}
    
    updated_config = config.copy()
    bt_res = None
    if trigger_backtest:
        add_log_fn("Optimization requested. (To be implemented)")
        # bt_res = { ... }
        
    # ── Entry / Exit Logic ─────────────────────────────────────────────
    if len(my_positions) > 0:
        pos_type = my_positions[0].get("type")
        total_profit = sum(p.get("profit", 0.0) for p in my_positions)
        
        # 1. Check Close All condition
        # If total profit is positive AND RSI crossed back to neutral/opposite
        close_all = False
        if total_profit > 0:
            if pos_type == "BUY" and curr_rsi >= 50.0:
                close_all = True
            elif pos_type == "SELL" and curr_rsi <= 50.0:
                close_all = True
                
        if close_all:
            add_log_fn(f"Grid profitable! Total Profit: ${total_profit:.2f}. Closing all layers.")
            action_dict = {
                "action": "CLOSE_ALL",
                "reason": "Grid Profitable Exit"
            }
        else:
            # 2. Check Layer Addition (Grid)
            if len(my_positions) < max_layers:
                if pos_type == "BUY":
                    lowest_entry = min(p.get("price_open", curr_close) for p in my_positions)
                    dist = lowest_entry - curr_close
                    if dist >= grid_step * curr_atr and curr_rsi < rsi_os:
                        add_log_fn(f"Adding Layer {len(my_positions)+1} BUY. Dist: {dist:.5f}")
                        lot = calculate_lot_size(balance, risk_pct, curr_atr * sl_mult, tick_value, tick_size, min_lot, max_lot, lot_step)
                        action_dict = {
                            "action": "BUY",
                            "lot": lot,
                            "tp_multiplier": tp_mult,
                            "sl_multiplier": sl_mult,
                            "reason": f"Grid Layer {len(my_positions)+1}"
                        }
                elif pos_type == "SELL":
                    highest_entry = max(p.get("price_open", curr_close) for p in my_positions)
                    dist = curr_close - highest_entry
                    if dist >= grid_step * curr_atr and curr_rsi > rsi_ob:
                        add_log_fn(f"Adding Layer {len(my_positions)+1} SELL. Dist: {dist:.5f}")
                        lot = calculate_lot_size(balance, risk_pct, curr_atr * sl_mult, tick_value, tick_size, min_lot, max_lot, lot_step)
                        action_dict = {
                            "action": "SELL",
                            "lot": lot,
                            "tp_multiplier": tp_mult,
                            "sl_multiplier": sl_mult,
                            "reason": f"Grid Layer {len(my_positions)+1}"
                        }
                    
    else:
        # No positions, check initial entry
        cross_up_os = (prev_rsi <= rsi_os and curr_rsi > rsi_os)
        cross_down_ob = (prev_rsi >= rsi_ob and curr_rsi < rsi_ob)
        
        if cross_up_os:
            add_log_fn("Initial BUY Signal Triggered.")
            lot = calculate_lot_size(balance, risk_pct, curr_atr * sl_mult, tick_value, tick_size, min_lot, max_lot, lot_step)
            action_dict = {
                "action": "BUY",
                "lot": lot,
                "tp_multiplier": tp_mult,
                "sl_multiplier": sl_mult,
                "reason": "RSI Initial BUY"
            }
        elif cross_down_ob:
            add_log_fn("Initial SELL Signal Triggered.")
            lot = calculate_lot_size(balance, risk_pct, curr_atr * sl_mult, tick_value, tick_size, min_lot, max_lot, lot_step)
            action_dict = {
                "action": "SELL",
                "lot": lot,
                "tp_multiplier": tp_mult,
                "sl_multiplier": sl_mult,
                "reason": "RSI Initial SELL"
            }

    res_dict.update(action_dict)
    
    if res_dict.get("action") == "NONE":
        res_dict["display_line1"] = f"RSI({rsi_period}): {curr_rsi:.1f} | ATR: {curr_atr:.5f}"
        res_dict["display_line2"] = f"Layers: {len(my_positions)}/{max_layers}"
    else:
        res_dict["display_line1"] = f"Action: {res_dict.get('action')}"
        res_dict["display_line2"] = f"Reason: {res_dict.get('reason')}"

    # UI live metrics list to render dynamically
    live_metrics = {
        f"RSI ({rsi_period})": f"{curr_rsi:.1f}",
        "ATR": f"{curr_atr:.5f}",
        "Active Layers": str(len(my_positions)),
        "Total Profit": f"${sum(p.get('profit', 0) for p in my_positions):.2f}" if len(my_positions) > 0 else "$0.00"
    }

    if is_new_bar:
        profit_str = f"${sum(p.get('profit', 0) for p in my_positions):.2f}"
        add_log_fn(f"[{symbol} {timeframe}] RSI={curr_rsi:.1f} ATR={curr_atr:.5f} Layers={len(my_positions)}/{max_layers} Profit={profit_str}")

    return res_dict, updated_config, live_metrics, bt_res
