"""
Teaser Scout Strategy (กลยุทธ์หมาหยอกไก่)
─────────────────────────────
ไม้แรก (Scout): เข้าทำกำไรสั้นๆ ด้วย Lot ขนาดปกติ หยั่งเชิงตลาด
ไม้สอง (Sniper): 
  - ถ้าไม้แรกโดนลาก (ติดลบ) จะรอสัญญาณคอนเฟิร์มว่าแรงเริ่มหมด
  - แล้วยิงไม้ Sniper ด้วย Lot ที่ใหญ่กว่า (Martingale นิดๆ) เพื่อดึงต้นทุน
  - ปิดรวบทุกไม้เมื่อกำไรรวมกลับมาเป็นบวก (หลุดดอย)
"""
import pandas as pd
import numpy as np

STRATEGY_NAME = "Teaser Scout"

DEFAULT_CONFIG = {
    "rsi_period": 7,
    "rsi_ob": 70.0,
    "rsi_os": 30.0,
    "scout_tp_mult": 2.0,     # โหมดซิ่ง: ลาก TP กว้างเพื่อกำไรคำใหญ่
    "scout_sl_mult": 5.0,     # โหมดซิ่ง: ให้พื้นที่หายใจกว้างๆ ป้องกัน SL โดนเตะไวไป
    "sniper_dist_atr": 2.0,   # โหมดซิ่ง: รอให้กราฟลากลงลึกๆ ก่อนค่อยยิงไม้แก้
    "sniper_lot_mult": 2.0,   # โหมดซิ่ง: อัด Lot 2 เท่าในไม้แก้เพื่อกระชากต้นทุนลงมา
    "risk_percent": 20.0      # โหมดซิ่ง: ดันความเสี่ยงขึ้นเพื่อบังคับเพิ่ม Lot ตอนพอร์ตโต
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
    trigger_backtest = bool(data.get("trigger_backtest", False))
    
    rsi_period = int(config.get("rsi_period", 7))
    rsi_ob = float(config.get("rsi_ob", 70.0))
    rsi_os = float(config.get("rsi_os", 30.0))
    scout_tp = float(config.get("scout_tp_mult", 0.5))
    scout_sl = float(config.get("scout_sl_mult", 3.0))
    sniper_dist = float(config.get("sniper_dist_atr", 1.5))
    sniper_mult = float(config.get("sniper_lot_mult", 1.5))
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
    
    my_positions = [p for p in positions if p.get("symbol") == symbol]
    
    res_dict = {
        "action": "NONE",
        "display_line1": f"RSI: {curr_rsi:.1f} | ATR: {curr_atr:.5f}",
        "display_line2": f"Status: Waiting for signal..."
    }
    
    updated_config = config.copy()
    bt_res = None
    if trigger_backtest:
        add_log_fn("Optimization requested. (To be implemented)")
        # bt_res = { ... }
        
    total_profit = sum(p.get("profit", 0.0) for p in my_positions)
    
    # ── State 0: No Positions (Look for Scout Entry) ──────────────────────
    if len(my_positions) == 0:
        cross_up_os = (prev_rsi <= rsi_os and curr_rsi > rsi_os)
        cross_down_ob = (prev_rsi >= rsi_ob and curr_rsi < rsi_ob)
        
        if cross_up_os:
            add_log_fn("Deploying Scout BUY")
            lot = calculate_lot_size(balance, risk_pct, curr_atr * scout_sl, tick_value, tick_size, min_lot, max_lot, lot_step)
            res_dict = {
                "action": "BUY",
                "lot": lot,
                "tp_multiplier": scout_tp,
                "sl_multiplier": scout_sl,
                "reason": "Scout BUY",
                "display_line1": "Deploying Scout BUY",
                "display_line2": f"TP: {scout_tp} ATR"
            }
        elif cross_down_ob:
            add_log_fn("Deploying Scout SELL")
            lot = calculate_lot_size(balance, risk_pct, curr_atr * scout_sl, tick_value, tick_size, min_lot, max_lot, lot_step)
            res_dict = {
                "action": "SELL",
                "lot": lot,
                "tp_multiplier": scout_tp,
                "sl_multiplier": scout_sl,
                "reason": "Scout SELL",
                "display_line1": "Deploying Scout SELL",
                "display_line2": f"TP: {scout_tp} ATR"
            }

    # ── State 1: Scout is Active (Monitor & Learn) ───────────────────────
    elif len(my_positions) == 1:
        pos = my_positions[0]
        pos_type = pos.get("type")
        
        # ถ้ากำไร MT5 จะจัดการ Take Profit อัตโนมัติ (จบแบบ A: เน้นชัวร์)
        # เราจะจัดการเฉพาะ "กรณีโดนลากติดลบ"
        
        if pos_type == "BUY":
            dist = pos.get("price_open", curr_close) - curr_close
            res_dict["display_line2"] = f"Scout Dist: {dist:.5f}"
            
            # ถ้าราคาลากลงลึกกว่า sniper_dist
            if dist >= sniper_dist * curr_atr:
                res_dict["display_line2"] = f"Awaiting Sniper Confirm..."
                # รอให้กราฟเริ่มโค้งกลับ (RSI หักหัวขึ้น) เพื่อยิง Sniper
                if curr_rsi > prev_rsi and curr_rsi < 50.0:
                    add_log_fn(f"Scout dragged by {dist:.5f}. Deploying Sniper BUY!")
                    base_lot = pos.get("lot", min_lot)
                    # ใช้ Lot ใหญ่กว่าไม้แรก
                    raw_sniper = base_lot * sniper_mult
                    sniper_lot = max(min_lot, min(max_lot, round(raw_sniper / lot_step) * lot_step))
                    
                    res_dict = {
                        "action": "BUY",
                        "lot": sniper_lot,
                        "tp_multiplier": scout_tp * 2, # กว้างหน่อยเพราะเดี๋ยวเราจะ Close All ทิ้งเอง
                        "sl_multiplier": scout_sl,
                        "reason": "Sniper Recovery BUY",
                        "display_line1": "Deploying Sniper BUY!",
                        "display_line2": f"Lot: {sniper_lot}"
                    }
                    
        elif pos_type == "SELL":
            dist = curr_close - pos.get("price_open", curr_close)
            res_dict["display_line2"] = f"Scout Dist: {dist:.5f}"
            
            if dist >= sniper_dist * curr_atr:
                res_dict["display_line2"] = f"Awaiting Sniper Confirm..."
                if curr_rsi < prev_rsi and curr_rsi > 50.0:
                    add_log_fn(f"Scout dragged by {dist:.5f}. Deploying Sniper SELL!")
                    base_lot = pos.get("lot", min_lot)
                    raw_sniper = base_lot * sniper_mult
                    sniper_lot = max(min_lot, min(max_lot, round(raw_sniper / lot_step) * lot_step))
                    
                    res_dict = {
                        "action": "SELL",
                        "lot": sniper_lot,
                        "tp_multiplier": scout_tp * 2,
                        "sl_multiplier": scout_sl,
                        "reason": "Sniper Recovery SELL",
                        "display_line1": "Deploying Sniper SELL!",
                        "display_line2": f"Lot: {sniper_lot}"
                    }

    # ── State 2: Sniper Deployed (Wait for Recovery) ─────────────────────
    elif len(my_positions) >= 2:
        res_dict["display_line1"] = "Sniper Active"
        res_dict["display_line2"] = f"Total Profit: ${total_profit:.2f}"
        
        # ปิดรวบทุกไม้ (Scout + Sniper) ทันทีที่หลุดดอย
        if total_profit > 0:
            add_log_fn(f"Sniper Recovery Successful! Profit: ${total_profit:.2f}")
            res_dict = {
                "action": "CLOSE_ALL",
                "reason": "Sniper Recovery Exit",
                "display_line1": "Target Hit",
                "display_line2": f"Recovered: ${total_profit:.2f}"
            }

    live_metrics = {
        "Strategy State": "Sniper" if len(my_positions) >= 2 else ("Scout" if len(my_positions) == 1 else "Waiting"),
        f"RSI ({rsi_period})": f"{curr_rsi:.1f}",
        "Active Layers": str(len(my_positions)),
        "Total Profit": f"${total_profit:.2f}"
    }

    return res_dict, updated_config, live_metrics, bt_res
