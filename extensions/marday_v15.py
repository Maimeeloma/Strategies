"""
marday_v15.py — Python port ของ MarDay EA v15.0.0 (MQL5 → standalone Python module)

กลยุทธ์: BB Mean-Reversion + Stop Hunt Reversal + Lucky Element + D1 Swing + Volatility-Adaptive Risk
ยึดโครงสร้าง/สไตล์ตาม simple-2-3.py (pandas + numpy, indicator แยกฟังก์ชัน,
backtest engine แบบ bar-by-bar อ่านค่าจาก bar[i-1]=closed bar กัน repaint,
process_strategy() คืน signal dict สำหรับ live)

I/O contract: ตรงตาม README-create-python.md (MT5 Python Bridge Strategy Extension System)
- 3 elements: STRATEGY_NAME (str), DEFAULT_CONFIG (dict), process_strategy(data, config, add_log_fn)
- process_strategy คืน tuple 4 ตัว: (res_dict, updated_config, live_metrics, bt_res)
- res_dict ทุก path มี display_line1 / display_line2
- action: NONE | BUY | SELL | CLOSE (มี ticket) | CLOSE_ALL
- bt_res = None เมื่อ trigger_backtest=False; มิฉะนั้น dict ตาม section 4.4

หมายเหตุ: port logic จาก MarDay_v15.0.0.mq5 อย่างซื่อสัตย์ ทุกค่า indicator อ่านจาก bar[1]
(แท่งปิดล่าสุด) เหมือน MQL5 ที่ใช้ CopyBuffer(handle, buffer, 1, 1, buf).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

STRATEGY_NAME = "MarDay-v15.0.0"

# ==========================================
# DEFAULT_CONFIG — สะท้อน input parameters ของ MarDay_v15.0.0.mq5
# ==========================================
DEFAULT_CONFIG: dict[str, Any] = {
    # --- General ---
    "ea_comment": "MarDay",
    "magic_number": 20240101,

    # --- Bollinger Bands ---
    "bb_period": 20,
    "bb_deviation": 2.0,
    "bb_proximity_atr": 0.20,     # InpBBProximityATR
    "bb_max_width_pct": 1.5,      # InpBBMaxWidthPct

    # --- Stochastic (fast scalping) ---
    "stoch_k": 5,
    "stoch_d": 3,
    "stoch_slowing": 3,
    "stoch_oversold": 25,
    "stoch_overbought": 75,

    # --- ATR ---
    "atr_period": 14,
    "atr_mult_sl": 1.5,           # InpATRMultSL
    "atr_mult_tp": 2.5,           # InpATRMultTP

    # --- RSI ---
    "enable_rsi": True,
    "rsi_period": 14,

    # --- Candle ---
    "cps_min": 0.55,              # InpCPSMin

    # --- Risk ---
    "risk_mode": "RISK_FIXED_LOT",   # RISK_FIXED_LOT | RISK_PERCENT
    "fixed_lot": 0.01,
    "risk_percent": 0.5,
    "max_daily_drawdown": 25.0,
    "max_open_trades": 1,
    "consecutive_loss_limit": 3,

    # --- Spread ---
    "max_spread_points": 40,

    # --- Session ---
    "enable_session_filter": True,
    "session_start_hour": 8,
    "session_start_min": 0,
    "session_end_hour": 20,
    "session_end_min": 0,
    "close_friday_filter": True,

    # --- Trade Management ---
    "enable_breakeven": True,
    "breakeven_trigger_pts": 800.0,
    "breakeven_buffer_pts": 80.0,
    "enable_trailing": True,
    "trailing_start_pts": 1200.0,
    "trailing_step_pts": 200.0,
    "one_trade_per_bar": True,
    "cooldown_bars_after_loss": 3,
    "max_bars_open": 40,

    # --- BB Exit Target (v1.16.7) ---
    "bb_exit_ratio": 0.60,        # InpBBExitRatio

    # --- Stop Hunt Reversal ---
    "enable_shr": True,
    "shr_lookback_bars": 8,

    # --- ATR Volatility Filter ---
    "high_vol_atr_pct": 0.40,     # InpHighVolATRPct

    # --- Lucky (v1.16.9 + v1.16.10) ---
    "lucky_minute": 0,            # 0 = ปิด (disabled)
    "lucky_max_mult": 5,
    "lucky_element": "ELEM_TECHO",   # ธาตุ → กำหนด duration

    # --- v11.9.0: Wed-skip ---
    "skip_wednesday": True,

    # --- v15.0.0: Volatility-Adaptive Risk (VAR) ---
    "enable_var": True,
    "var_high_atr": 0.45,         # ATR% above = high-vol regime
    "var_low_atr": 0.15,          # ATR% below = low-vol regime
    "var_high_mult": 0.50,
    "var_low_mult": 1.25,
    "var_normal_mult": 1.00,
    "var_apply_d1": False,

    # --- v11.8.0: D1 Swing-Trend Module ---
    "enable_d1_swing": True,
    "d1_swing_magic": 20240111,
    "d1_atr_period": 14,
    "d1_mom_lookback": 5,
    "d1_swing_min_move_atr": 1.0,
    "d1_swing_atr_mult_sl": 1.5,
    "d1_swing_atr_mult_tp": 4.0,
    "d1_swing_skip_monday": True,
    "d1_swing_body_filter": True,
    "d1_swing_body_ratio": 0.55,
    "d1_swing_risk_percent": 0.5,
    "d1_swing_max_bars_open": 720,
}

# Lucky cycle length (นาที) — #define LUCKY_CYCLE_MIN ใน MQL5
LUCKY_CYCLE_MIN = 99

# ธาตุ 6 → ระยะเวลา lucky (นาที) — ตรงกับ GetElementDuration()
ELEMENT_DURATION = {
    "ELEM_NONE": 0,
    "ELEM_PATHAVI": 15,    # ปฐวี (ดิน)
    "ELEM_AAPO": 30,       # อาโป (น้ำ)
    "ELEM_VAYO": 45,       # วาโย (ลม)
    "ELEM_TECHO": 60,      # เตโช (ไฟ)
    "ELEM_AKASA": 75,      # อากาศ
    "ELEM_VINNANA": 90,    # วิญญาณ (จิต)
}


# ==========================================
# Indicators Calculation — ตรงกับ MT5 iBands / iStochastic / iATR / iRSI
# ==========================================
def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 14) -> pd.Series:
    """ATR แบบ Wilder smoothing (EMA alpha=1/period) เหมือน MT5 iATR."""
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI แบบ Wilder (EMA alpha=1/period) เหมือน MT5 iRSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_bollinger(close: pd.Series, period: int = 20,
                        deviation: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands (SMA + population std) เหมือน MT5 iBands.
    คืน (upper, middle, lower)."""
    middle = close.rolling(window=period).mean()
    # MT5 ใช้ population standard deviation (ddof=0)
    std = close.rolling(window=period).std(ddof=0)
    upper = middle + deviation * std
    lower = middle - deviation * std
    return upper, middle, lower


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                         k_period: int = 5, d_period: int = 3,
                         slowing: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator แบบ MODE_SMA + STO_LOWHIGH เหมือน MT5 iStochastic.
    K = SMA(slowing) ของ raw %K ; D = SMA(d_period) ของ K. คืน (k, d)."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    rng = highest_high - lowest_low
    raw_k = np.where(rng == 0, 0.0, (close - lowest_low) / rng * 100.0)
    raw_k = pd.Series(raw_k, index=close.index)
    k = raw_k.rolling(window=slowing).mean()       # slowing (MODE_SMA)
    d = k.rolling(window=d_period).mean()
    return k, d


def calculate_cps(high: float, low: float, close: float) -> float:
    """Close Position in range = (close-low)/(high-low). เหมือน GetCPS()."""
    if high <= low:
        return 0.5
    return (close - low) / (high - low)


def get_var_multiplier(atr_val: float, price: float, config: dict[str, Any]) -> float:
    """v15.0.0 Volatility-Adaptive Risk: ตัวคูณ lot ตาม regime ของ ATR%.
    ตรงกับ GetVARMultiplier() ใน MQL5."""
    if not bool(config.get("enable_var", True)):
        return 1.0
    if price <= 0 or atr_val <= 0:
        return 1.0
    atr_pct = (atr_val / price) * 100.0
    if atr_pct >= float(config.get("var_high_atr", 0.45)):
        return float(config.get("var_high_mult", 0.50))
    if atr_pct <= float(config.get("var_low_atr", 0.15)):
        return float(config.get("var_low_mult", 1.25))
    return float(config.get("var_normal_mult", 1.00))


def calculate_lot_size(balance: float, config: dict[str, Any], sl_points: float,
                       tick_value: float, tick_size: float, point: float,
                       min_lot: float, max_lot: float, lot_step: float,
                       lucky_active: bool = False, lucky_mult: int = 1,
                       atr_val: float = 0.0, price: float = 0.0) -> float:
    """คำนวณ lot — รองรับ RISK_FIXED_LOT / RISK_PERCENT + Lucky mult + VAR mult.
    ตรงกับ CalculateLotSize() ใน MQL5 (slMoney = (slPoints*point/ts)*tv)."""
    lot = float(config.get("fixed_lot", 0.01))

    if str(config.get("risk_mode", "RISK_FIXED_LOT")) == "RISK_PERCENT":
        risk = balance * float(config.get("risk_percent", 0.5)) / 100.0
        if tick_value > 0 and tick_size > 0 and sl_points > 0:
            sl_money = (sl_points * point / tick_size) * tick_value
            if sl_money > 0:
                lot = risk / sl_money

    # Lucky lot multiplier
    if lucky_active and lucky_mult > 1:
        lot *= lucky_mult

    # v15.0.0: Volatility-Adaptive Risk
    lot *= get_var_multiplier(atr_val, price, config)

    if lot_step <= 0:
        lot_step = 0.01
    lot = max(min_lot, min(max_lot, math.floor(lot / lot_step) * lot_step))
    return round(lot, 2)


# ==========================================
# Lucky Element cycle — port ของ RefreshLuckyCycle()
# ==========================================
class LuckyState:
    """สถานะ Lucky cycle (รอบ 99 นาที + ธาตุกำหนด duration).
    ใช้ track g_isLuckyActive / g_luckyMultiplier / g_cycleStartTime."""

    def __init__(self, config: dict[str, Any], start_time: float) -> None:
        self.cycle_start_time = float(start_time)
        self.is_active = False
        self.multiplier = 1
        self.cycle_count = 0
        elem = str(config.get("lucky_element", "ELEM_TECHO"))
        self.duration = ELEMENT_DURATION.get(elem, 60)
        if self.duration == 0:   # ELEM_NONE → default = ไฟ (เหมือน MQL5 default)
            self.duration = 60

    def refresh(self, now: float, config: dict[str, Any]) -> None:
        """อัปเดตสถานะ lucky ตามเวลาปัจจุบัน (epoch seconds)."""
        lucky_minute = int(config.get("lucky_minute", 0))
        if lucky_minute < 1 or lucky_minute > 99:
            return  # disabled
        if str(config.get("lucky_element", "ELEM_TECHO")) == "ELEM_NONE":
            return

        elapsed_min = int((now - self.cycle_start_time) / 60)

        # ครบรอบ → advance start time (รักษาจังหวะ)
        if elapsed_min >= LUCKY_CYCLE_MIN:
            self.cycle_start_time += LUCKY_CYCLE_MIN * 60
            elapsed_min = int((now - self.cycle_start_time) / 60)
            self.cycle_count += 1

        # lucky window = [lucky_minute, lucky_minute + duration); wrap ข้ามรอบได้
        lucky_end = lucky_minute + self.duration
        if lucky_end <= LUCKY_CYCLE_MIN:
            in_lucky = (lucky_minute <= elapsed_min < lucky_end)
        else:
            in_lucky = (elapsed_min >= lucky_minute or
                        elapsed_min < (lucky_end - LUCKY_CYCLE_MIN))

        if in_lucky and not self.is_active:
            self.is_active = True
            self.multiplier = 2     # เริ่มที่ x2
        elif not in_lucky and self.is_active:
            self.is_active = False
            self.multiplier = 1

    def on_win(self, config: dict[str, Any]) -> None:
        """ชนะ → multiplier +1 จนถึง cap (เหมือน OnTradeTransaction win branch)."""
        if self.is_active and self.multiplier < int(config.get("lucky_max_mult", 5)):
            self.multiplier += 1

    def on_loss(self) -> None:
        """แพ้ → กลับมา x2 (เหมือน OnTradeTransaction loss branch)."""
        if self.is_active and self.multiplier > 2:
            self.multiplier = 2


# ==========================================
# Signal helpers — port ของ CheckBBBounce* / CheckSHR*
# ==========================================
def _check_bb_bounce_buy(close1: float, low1: float, open1: float, high1: float,
                         bb_lower: float, atr_val: float, stoch_k: float,
                         rsi_val: float, config: dict[str, Any]) -> bool:
    """BB Bounce Buy — ตรงกับ CheckBBBounceBuy()."""
    prox = float(config.get("bb_proximity_atr", 0.20))
    at_bb_low = (low1 <= bb_lower + atr_val * prox)
    if not at_bb_low:
        return False
    if close1 < bb_lower - atr_val * 0.50:
        return False
    if stoch_k >= float(config.get("stoch_oversold", 25)):
        return False
    if bool(config.get("enable_rsi", True)) and rsi_val >= 55.0:
        return False
    cps = calculate_cps(high1, low1, close1)
    if cps < float(config.get("cps_min", 0.55)):
        return False
    if not (close1 > open1):   # IsBullishBody
        return False
    return True


def _check_bb_bounce_sell(close1: float, high1: float, open1: float, low1: float,
                          bb_upper: float, atr_val: float, stoch_k: float,
                          rsi_val: float, config: dict[str, Any]) -> bool:
    """BB Bounce Sell — ตรงกับ CheckBBBounceSell()."""
    prox = float(config.get("bb_proximity_atr", 0.20))
    at_bb_high = (high1 >= bb_upper - atr_val * prox)
    if not at_bb_high:
        return False
    if close1 > bb_upper + atr_val * 0.50:
        return False
    if stoch_k <= float(config.get("stoch_overbought", 75)):
        return False
    if bool(config.get("enable_rsi", True)) and rsi_val <= 45.0:
        return False
    cps = calculate_cps(high1, low1, close1)
    if cps > (1.0 - float(config.get("cps_min", 0.55))):
        return False
    if not (close1 < open1):   # IsBearishBody
        return False
    return True


def _check_shr_buy(close1: float, low1: float, open1: float, recent_low: float,
                   bb_middle: float, atr_val: float, stoch_k: float) -> bool:
    """Stop Hunt Reversal Buy — ตรงกับ CheckSHRBuy()."""
    if recent_low <= 0.0:
        return False
    if low1 >= recent_low:          # ต้อง sweep ต่ำกว่า swing low
        return False
    if close1 <= recent_low:        # แล้ว reclaim กลับขึ้นมา
        return False
    if not (close1 > open1):        # bullish body
        return False
    if stoch_k >= 30.0:
        return False
    if close1 > bb_middle + atr_val * 0.50:
        return False
    return True


def _check_shr_sell(close1: float, high1: float, open1: float, recent_high: float,
                    bb_middle: float, atr_val: float, stoch_k: float) -> bool:
    """Stop Hunt Reversal Sell — ตรงกับ CheckSHRSell()."""
    if recent_high <= 0.0:
        return False
    if high1 <= recent_high:
        return False
    if close1 >= recent_high:
        return False
    if not (close1 < open1):        # bearish body
        return False
    if stoch_k <= 70.0:
        return False
    if close1 < bb_middle - atr_val * 0.50:
        return False
    return True


def _is_trading_session(dt: datetime, config: dict[str, Any]) -> bool:
    """session filter 08:00-20:00 — ตรงกับ IsTradingSession()."""
    now = dt.hour * 60 + dt.minute
    start = int(config.get("session_start_hour", 8)) * 60 + int(config.get("session_start_min", 0))
    end = int(config.get("session_end_hour", 20)) * 60 + int(config.get("session_end_min", 0))
    return start <= now < end


def _is_friday_close(dt: datetime) -> bool:
    """Friday >= 20:00 — ตรงกับ IsFridayClose() (Mon=0..Sun=6, Fri=4)."""
    return dt.weekday() == 4 and dt.hour >= 20


def _is_wednesday(dt: datetime) -> bool:
    """Wed-skip — Python weekday() Wed=2 (เทียบ MQL day_of_week==3)."""
    return dt.weekday() == 2


# ==========================================
# Backtest Engine — bar-by-bar, อ่าน indicator จาก bar[i-1]=closed bar
# ==========================================
def run_backtest(candles: list[dict], config: dict[str, Any],
                 balance: float = 10000.0, tick_value: float = 1.0,
                 tick_size: float = 0.01, point: float = 0.01,
                 min_lot: float = 0.01, max_lot: float = 100.0,
                 lot_step: float = 0.01) -> Optional[dict[str, Any]]:
    """จำลอง entry/exit ตาม logic v15 (BB bounce + SHR + filters + lucky + VAR + D1).

    คืน dict สถิติ: net_profit, trades, win_rate, max_drawdown, profit_factor,
    direction (BUY/SELL/NONE = ทิศที่ทำกำไรมากกว่า).
    ทุก signal อ่านจาก bar ปิดล่าสุด (i-1) เพื่อกัน repaint เหมือน MQL5.
    """
    bb_period = int(config.get("bb_period", 20))
    atr_period = int(config.get("atr_period", 14))
    min_bars = bb_period + atr_period + 10
    if len(candles) < min_bars:
        return None

    df = pd.DataFrame(candles)
    df["time"] = pd.to_numeric(df["time"])
    df = df.sort_values(by="time").reset_index(drop=True)

    # --- indicators ---
    atr = calculate_atr(df["high"], df["low"], df["close"], atr_period)
    bb_upper, bb_middle, bb_lower = calculate_bollinger(
        df["close"], bb_period, float(config.get("bb_deviation", 2.0)))
    stoch_k, stoch_d = calculate_stochastic(
        df["high"], df["low"], df["close"],
        int(config.get("stoch_k", 5)), int(config.get("stoch_d", 3)),
        int(config.get("stoch_slowing", 3)))
    rsi = calculate_rsi(df["close"], int(config.get("rsi_period", 14)))

    open_v = df["open"].values
    high_v = df["high"].values
    low_v = df["low"].values
    close_v = df["close"].values
    time_v = df["time"].values
    atr_v = atr.values
    bbu_v = bb_upper.values
    bbm_v = bb_middle.values
    bbl_v = bb_lower.values
    sk_v = stoch_k.values
    rsi_v = rsi.values

    atr_mult_sl = float(config.get("atr_mult_sl", 1.5))
    atr_mult_tp = float(config.get("atr_mult_tp", 2.5))
    bb_exit_ratio = float(config.get("bb_exit_ratio", 0.60))
    high_vol_pct = float(config.get("high_vol_atr_pct", 0.40))
    bb_max_width = float(config.get("bb_max_width_pct", 1.5))
    shr_lookback = int(config.get("shr_lookback_bars", 8))
    enable_shr = bool(config.get("enable_shr", True))
    skip_wed = bool(config.get("skip_wednesday", True))
    max_bars_open = int(config.get("max_bars_open", 40))
    enable_session = bool(config.get("enable_session_filter", True))
    close_friday = bool(config.get("close_friday_filter", True))
    one_trade_per_bar = bool(config.get("one_trade_per_bar", True))
    cooldown_after_loss = int(config.get("cooldown_bars_after_loss", 3))
    consec_loss_limit = int(config.get("consecutive_loss_limit", 3))
    be_trigger = float(config.get("breakeven_trigger_pts", 800.0))
    be_buffer = float(config.get("breakeven_buffer_pts", 80.0))
    enable_be = bool(config.get("enable_breakeven", True))
    tr_start = float(config.get("trailing_start_pts", 1200.0))
    tr_step = float(config.get("trailing_step_pts", 200.0))
    enable_tr = bool(config.get("enable_trailing", True))

    sim_balance = balance
    peak_balance = balance
    max_dd = 0.0
    trades = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    # แยกกำไรตามทิศ เพื่อสรุป "direction" ที่ดีกว่า
    profit_buy = 0.0
    profit_sell = 0.0

    in_position = False
    pos_type = ""           # "BUY" | "SELL"
    entry_price = 0.0
    sl_level = 0.0
    tp_level = 0.0
    cur_sl = 0.0            # SL ที่ปรับด้วย BE/trailing
    lot_size = min_lot
    entry_bar = 0
    entry_tag = ""

    consec_losses = 0
    cooldown_left = 0

    lucky = LuckyState(config, float(time_v[0]))

    def _close_position(profit_points: float, exit_idx: int) -> None:
        nonlocal in_position, pos_type, sim_balance, peak_balance, max_dd
        nonlocal trades, wins, gross_profit, gross_loss, consec_losses, cooldown_left
        nonlocal profit_buy, profit_sell
        profit_money = (profit_points / tick_size) * tick_value * lot_size
        sim_balance += profit_money
        peak_balance = max(peak_balance, sim_balance)
        dd = ((peak_balance - sim_balance) / peak_balance) * 100.0 if peak_balance > 0 else 0.0
        max_dd = max(max_dd, dd)
        trades += 1
        if pos_type == "BUY":
            profit_buy += profit_money
        else:
            profit_sell += profit_money
        if profit_money > 0:
            wins += 1
            gross_profit += profit_money
            consec_losses = 0
            lucky.on_win(config)
        else:
            gross_loss += abs(profit_money)
            consec_losses += 1
            cooldown_left = cooldown_after_loss
            lucky.on_loss()
        in_position = False
        pos_type = ""

    for i in range(2, len(df)):
        # ค่า indicator/แท่งของ "bar ปิดล่าสุด" = index i-1
        c1 = close_v[i - 1]
        o1 = open_v[i - 1]
        h1 = high_v[i - 1]
        l1 = low_v[i - 1]
        atr_val = atr_v[i - 1]
        bbu = bbu_v[i - 1]
        bbm = bbm_v[i - 1]
        bbl = bbl_v[i - 1]
        sk = sk_v[i - 1]
        rv = rsi_v[i - 1]

        if (np.isnan(atr_val) or np.isnan(bbu) or np.isnan(bbm) or np.isnan(bbl)
                or np.isnan(sk) or np.isnan(rv)):
            continue

        bar_time = datetime.fromtimestamp(float(time_v[i]), tz=timezone.utc)
        lucky.refresh(float(time_v[i]), config)

        # ===== Manage open position (ใช้ high/low ของแท่งปัจจุบัน i เป็น proxy intrabar) =====
        if in_position:
            hi = high_v[i]
            lo = low_v[i]
            cur_price = close_v[i]
            profit_points = ((cur_price - entry_price) if pos_type == "BUY"
                             else (entry_price - cur_price))
            profit_pts_units = profit_points / point if point > 0 else 0.0

            # Breakeven (trigger 800pts, buffer 80pts)
            if enable_be and profit_pts_units >= be_trigger:
                if pos_type == "BUY":
                    new_sl = entry_price + be_buffer * point
                    if new_sl > cur_sl:
                        cur_sl = new_sl
                else:
                    new_sl = entry_price - be_buffer * point
                    if cur_sl <= 0 or new_sl < cur_sl:
                        cur_sl = new_sl

            # Trailing (start 1200pts, step 200pts)
            if enable_tr and profit_pts_units >= tr_start:
                if pos_type == "BUY":
                    new_sl = cur_price - tr_step * point
                    if new_sl > cur_sl:
                        cur_sl = new_sl
                else:
                    new_sl = cur_price + tr_step * point
                    if cur_sl <= 0 or new_sl < cur_sl:
                        cur_sl = new_sl

            # ตรวจ SL / TP hit (priority: SL ก่อน เหมือน simple-2-3.py)
            hit = False
            if pos_type == "BUY":
                if lo <= cur_sl:
                    _close_position(cur_sl - entry_price, i)
                    hit = True
                elif hi >= tp_level:
                    _close_position(tp_level - entry_price, i)
                    hit = True
            else:  # SELL
                if hi >= cur_sl:
                    _close_position(entry_price - cur_sl, i)
                    hit = True
                elif lo <= tp_level:
                    _close_position(entry_price - tp_level, i)
                    hit = True

            if hit:
                continue

            # time-exit เมื่อ barsOpen > max_bars_open
            if (i - entry_bar) > max_bars_open:
                exit_p = ((cur_price - entry_price) if pos_type == "BUY"
                          else (entry_price - cur_price))
                _close_position(exit_p, i)
                continue

            # BB mean-reversion exit (InpBBExitRatio)
            bb_exit_buy = bbl + bb_exit_ratio * (bbu - bbl)
            bb_exit_sell = bbu - bb_exit_ratio * (bbu - bbl)
            if pos_type == "BUY" and profit_points > 0 and cur_price >= bb_exit_buy:
                _close_position(cur_price - entry_price, i)
                continue
            if pos_type == "SELL" and profit_points > 0 and cur_price <= bb_exit_sell:
                _close_position(entry_price - cur_price, i)
                continue

            continue  # มี position แล้ว ไม่เปิดเพิ่ม (MaxOpenTrades=1)

        # ===== Entry filters =====
        if cooldown_left > 0:
            cooldown_left -= 1
            continue
        if consec_losses >= consec_loss_limit:
            continue
        if enable_session and not _is_trading_session(bar_time, config):
            continue
        if close_friday and _is_friday_close(bar_time):
            continue
        if skip_wed and _is_wednesday(bar_time):
            continue
        # OneTradePerBar — ใน backtest 1 แท่ง = 1 รอบ จึงผ่านโดยปริยาย
        _ = one_trade_per_bar

        # Volatility filter (เฉพาะ BB bounce)
        high_vol = (c1 > 0 and (atr_val / c1) * 100.0 > high_vol_pct)
        bb_wide = (bbm > 0 and (bbu - bbl) / bbm * 100.0 > bb_max_width)

        signal = ""   # "BUY" | "SELL"
        tag = ""

        if not high_vol and not bb_wide:
            if _check_bb_bounce_buy(c1, l1, o1, h1, bbl, atr_val, sk, rv, config):
                signal, tag = "BUY", "BBBuy"
            elif _check_bb_bounce_sell(c1, h1, o1, l1, bbu, atr_val, sk, rv, config):
                signal, tag = "SELL", "BBSell"

        # SHR (ไม่ติด volatility filter)
        if not signal and enable_shr:
            # recent swing high/low จาก lookback แท่ง เริ่ม shift 2 (iHighest/iLowest start=2)
            start = i - 1 - shr_lookback
            end = i - 1   # ไม่รวม bar ปิดล่าสุด (เริ่ม shift 2 = index i-2 ลงไป)
            if start >= 0:
                recent_high = float(np.max(high_v[start:end]))
                recent_low = float(np.min(low_v[start:end]))
                if _check_shr_buy(c1, l1, o1, recent_low, bbm, atr_val, sk):
                    signal, tag = "BUY", "SHRBuy"
                elif _check_shr_sell(c1, h1, o1, recent_high, bbm, atr_val, sk):
                    signal, tag = "SELL", "SHRSell"

        if not signal:
            continue

        # ===== Open position =====
        sl_dist = atr_val * atr_mult_sl
        tp_dist = atr_val * atr_mult_tp
        sl_points = sl_dist / point if point > 0 else 0.0
        lot_size = calculate_lot_size(
            sim_balance, config, sl_points, tick_value, tick_size, point,
            min_lot, max_lot, lot_step,
            lucky_active=lucky.is_active, lucky_mult=lucky.multiplier,
            atr_val=atr_val, price=c1)
        # entry ที่ราคา open ของแท่งปัจจุบัน (i) = แท่งที่เพิ่งเปิด
        entry_price = open_v[i]
        in_position = True
        pos_type = signal
        entry_tag = tag
        entry_bar = i
        if signal == "BUY":
            sl_level = entry_price - sl_dist
            tp_level = entry_price + tp_dist
        else:
            sl_level = entry_price + sl_dist
            tp_level = entry_price - tp_dist
        cur_sl = sl_level

    net_profit = sim_balance - balance
    win_rate = (wins / trades * 100.0) if trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0)

    if trades == 0:
        direction = "NONE"
    elif profit_buy >= profit_sell:
        direction = "BUY"
    else:
        direction = "SELL"

    return {
        "direction": direction,
        "net_profit": float(round(net_profit, 2)),
        "trades": trades,
        "win_rate": float(round(win_rate, 2)),
        "max_drawdown": float(round(max_dd, 2)),
        "profit_factor": (float(round(profit_factor, 3))
                          if profit_factor != float("inf") else float("inf")),
        "final_balance": float(round(sim_balance, 2)),
    }


def run_backtest_optimization(candles: list[dict], config: dict[str, Any],
                              balance: float, tick_value: float, tick_size: float,
                              point: float, min_lot: float, max_lot: float,
                              lot_step: float, add_log_fn: Callable[[str], None]
                              ) -> Optional[dict[str, Any]]:
    """Grid search หา (atr_mult_tp, atr_mult_sl) ที่ดีที่สุดด้วย run_backtest จริง.
    คืน dict ตาม README section 4.4: direction, tp_multiplier, sl_multiplier,
    win_rate, max_drawdown, trades, profit + optimized params (atr_mult_tp/atr_mult_sl)
    เพื่อ map กลับ updated_config. คืน None เมื่อ data ไม่พอ."""
    tp_list = [1.5, 2.0, 2.5, 3.0, 4.0]
    sl_list = [1.0, 1.5, 2.0]

    best_profit = -float("inf")
    best: Optional[dict[str, Any]] = None

    for tp_m in tp_list:
        for sl_m in sl_list:
            trial = config.copy()
            trial["atr_mult_tp"] = tp_m
            trial["atr_mult_sl"] = sl_m
            stats = run_backtest(candles, trial, balance, tick_value, tick_size,
                                 point, min_lot, max_lot, lot_step)
            if stats is None:
                return None
            if stats["trades"] < 1:
                continue
            if stats["net_profit"] > best_profit:
                best_profit = stats["net_profit"]
                best = {
                    "direction": stats["direction"],
                    "tp_multiplier": float(tp_m),
                    "sl_multiplier": float(sl_m),
                    "win_rate": float(stats["win_rate"]),
                    "max_drawdown": float(stats["max_drawdown"]),
                    "trades": int(stats["trades"]),
                    "profit": float(stats["net_profit"]),
                    # optimized params → map กลับ config
                    "atr_mult_tp": float(tp_m),
                    "atr_mult_sl": float(sl_m),
                }

    if best is None:
        # ไม่มี combo ไหนเทรดเลย → คืนผลกลาง ๆ (direction NONE) เพื่อให้ contract ครบ
        add_log_fn("Optimization: no trades across grid; returning NONE direction")
        base = run_backtest(candles, config, balance, tick_value, tick_size,
                            point, min_lot, max_lot, lot_step)
        if base is None:
            return None
        return {
            "direction": "NONE",
            "tp_multiplier": float(config.get("atr_mult_tp", 2.5)),
            "sl_multiplier": float(config.get("atr_mult_sl", 1.5)),
            "win_rate": 0.0,
            "max_drawdown": float(base["max_drawdown"]),
            "trades": 0,
            "profit": 0.0,
            "atr_mult_tp": float(config.get("atr_mult_tp", 2.5)),
            "atr_mult_sl": float(config.get("atr_mult_sl", 1.5)),
        }
    return best


# ==========================================
# D1 Swing module (optional) — port ของ TryD1SwingEntry()
# ==========================================
def get_d1_swing_direction(d1_close: list[float], d1_atr: float,
                           config: dict[str, Any]) -> int:
    """trend จาก close[1] vs close[1+lookback] เทียบ D1 ATR. คืน +1/-1/0.
    d1_close = list ของ D1 closes (index 0 = newest = bar 0), shift เหมือน MQL5."""
    lookback = int(config.get("d1_mom_lookback", 5))
    if len(d1_close) < (2 + lookback):
        return 0
    c1 = d1_close[1]
    cn = d1_close[1 + lookback]
    if c1 <= 0 or cn <= 0 or d1_atr <= 0:
        return 0
    move = c1 - cn
    thr = float(config.get("d1_swing_min_move_atr", 1.0)) * d1_atr
    if move >= thr:
        return +1
    if -move >= thr:
        return -1
    return 0


def evaluate_d1_swing(d1_close: list[float], d1_atr: float,
                      m5_bar1: dict, dt: datetime,
                      config: dict[str, Any]) -> dict[str, Any]:
    """ประเมิน D1 swing entry (optional module). คืน action dict.
    m5_bar1 = แท่ง M5 ปิดล่าสุด {open,high,low,close} สำหรับ body confirmation."""
    if not bool(config.get("enable_d1_swing", True)):
        return {"action": "NONE", "reason": "D1 disabled"}
    if bool(config.get("d1_swing_skip_monday", True)) and dt.weekday() == 0:
        return {"action": "NONE", "reason": "D1 skip Monday"}

    direction = get_d1_swing_direction(d1_close, d1_atr, config)
    if direction == 0:
        return {"action": "NONE", "reason": "D1 no trend"}

    o1 = float(m5_bar1["open"])
    c1 = float(m5_bar1["close"])
    h1 = float(m5_bar1["high"])
    l1 = float(m5_bar1["low"])
    bar_range = h1 - l1
    if bar_range <= 0:
        return {"action": "NONE", "reason": "D1 invalid bar"}
    body_ratio = abs(c1 - o1) / bar_range
    if (bool(config.get("d1_swing_body_filter", True)) and
            body_ratio < float(config.get("d1_swing_body_ratio", 0.55))):
        return {"action": "NONE", "reason": "D1 body too small"}

    sl_dist = d1_atr * float(config.get("d1_swing_atr_mult_sl", 1.5))
    tp_dist = d1_atr * float(config.get("d1_swing_atr_mult_tp", 4.0))

    if direction > 0 and c1 > o1:   # uptrend + bullish body
        return {"action": "BUY", "tag": "D1Long", "sl_dist": sl_dist,
                "tp_dist": tp_dist, "d1_atr": d1_atr}
    if direction < 0 and c1 < o1:   # downtrend + bearish body
        return {"action": "SELL", "tag": "D1Short", "sl_dist": sl_dist,
                "tp_dist": tp_dist, "d1_atr": d1_atr}
    return {"action": "NONE", "reason": "D1 body direction mismatch"}


# ==========================================
# Core Strategy Logic (live) — คืน signal dict ตาม MT5 Bridge contract
# ==========================================
def process_strategy(data: dict[str, Any], config: dict[str, Any],
                     add_log_fn: Callable[[str], None]
                     ) -> tuple[dict[str, Any], dict[str, Any],
                                dict[str, Any], Optional[dict[str, Any]]]:
    """ประมวลผล signal บนแท่งปิดล่าสุด (bar[1]) แล้วคืน
    (res_dict, updated_config, live_metrics, bt_res) ตาม README-create-python.md.

    res_dict ทุก path มี display_line1 / display_line2.
    action: NONE | BUY | SELL | CLOSE (มี ticket) | CLOSE_ALL.
    BUY/SELL ใช้ Option B (tp_multiplier/sl_multiplier ให้ client คูณ ATR เอง).

    data: {candles, positions, symbol, timeframe, is_new_bar, balance, equity, spread,
           tick_size, tick_value, min_lot, max_lot, lot_step, trigger_backtest,
           point(optional), server_time(optional), d1_close(optional), d1_atr(optional)}
    """
    candles: list[dict] = data.get("candles", [])
    positions: list[dict] = data.get("positions", [])
    symbol = data.get("symbol", "XAUUSD")
    timeframe = data.get("timeframe", "M5")
    is_new_bar = bool(data.get("is_new_bar", False))
    trigger_backtest = bool(data.get("trigger_backtest", False))

    balance = float(data.get("balance", 10000.0))
    equity = float(data.get("equity", balance))
    spread = int(data.get("spread", 0))
    tick_value = float(data.get("tick_value", 1.0))
    tick_size = float(data.get("tick_size", 0.01))
    # point: ถ้า client ไม่ส่ง ใช้ tick_size (ตรงกับทองทศนิยม 2 ตำแหน่ง)
    point = float(data.get("point", tick_size))
    if point <= 0:
        point = 0.01
    min_lot = float(data.get("min_lot", 0.01))
    max_lot = float(data.get("max_lot", 100.0))
    lot_step = float(data.get("lot_step", 0.01))
    server_time = float(data.get("server_time", 0.0))

    updated_config = config.copy()

    # กรอง positions เฉพาะ symbol ปัจจุบัน (ใช้จัดการ exit เท่านั้น)
    sym_positions = [p for p in positions if p.get("symbol") == symbol]

    bb_period = int(config.get("bb_period", 20))
    atr_period = int(config.get("atr_period", 14))
    min_bars = bb_period + atr_period + 10

    # ---- Path: ข้อมูลแท่งไม่พอ ----
    if len(candles) < min_bars:
        res_dict = {
            "action": "NONE",
            "display_line1": "Initializing...",
            "display_line2": f"Bars: {len(candles)}/{min_bars}",
        }
        return res_dict, updated_config, {"Status": "Syncing bars"}, None

    # ---- optional backtest / optimization ----
    bt_res = None
    if trigger_backtest:
        add_log_fn("Running parameter optimization (grid search TP/SL)...")
        bt_res = run_backtest_optimization(
            candles, config, balance, tick_value, tick_size, point,
            min_lot, max_lot, lot_step, add_log_fn)
        if bt_res:
            # map optimized params กลับเข้า updated_config
            if "atr_mult_tp" in bt_res:
                updated_config["atr_mult_tp"] = float(bt_res["atr_mult_tp"])
            if "atr_mult_sl" in bt_res:
                updated_config["atr_mult_sl"] = float(bt_res["atr_mult_sl"])
            add_log_fn(
                f"Optimization done: dir={bt_res['direction']} "
                f"TP={bt_res['tp_multiplier']} SL={bt_res['sl_multiplier']} "
                f"profit={bt_res['profit']} WR={bt_res['win_rate']}% "
                f"trades={bt_res['trades']} DD={bt_res['max_drawdown']}%")

    # ---- indicators ----
    df = pd.DataFrame(candles)
    df["time"] = pd.to_numeric(df["time"])
    df = df.sort_values(by="time").reset_index(drop=True)

    atr = calculate_atr(df["high"], df["low"], df["close"], atr_period)
    bb_upper, bb_middle, bb_lower = calculate_bollinger(
        df["close"], bb_period, float(config.get("bb_deviation", 2.0)))
    stoch_k, stoch_d = calculate_stochastic(
        df["high"], df["low"], df["close"],
        int(config.get("stoch_k", 5)), int(config.get("stoch_d", 3)),
        int(config.get("stoch_slowing", 3)))
    rsi = calculate_rsi(df["close"], int(config.get("rsi_period", 14)))

    # ค่าจาก bar[1] = แท่งปิดล่าสุด (iloc[-2])
    c1 = float(df["close"].iloc[-2])
    o1 = float(df["open"].iloc[-2])
    h1 = float(df["high"].iloc[-2])
    l1 = float(df["low"].iloc[-2])
    atr_val = float(atr.iloc[-2])
    bbu = float(bb_upper.iloc[-2])
    bbm = float(bb_middle.iloc[-2])
    bbl = float(bb_lower.iloc[-2])
    sk = float(stoch_k.iloc[-2])
    rv = float(rsi.iloc[-2])

    # ATR% และ BB width% (สำหรับ HUD / filter)
    atr_pct = (atr_val / c1 * 100.0) if c1 > 0 else 0.0
    bb_width_pct = ((bbu - bbl) / bbm * 100.0) if bbm > 0 else 0.0

    # Lucky cycle refresh
    now = server_time if server_time > 0 else float(df["time"].iloc[-1])
    lucky = LuckyState(config, float(df["time"].iloc[0]))
    lucky.refresh(now, config)
    lucky_str = (f"ON x{lucky.multiplier}" if lucky.is_active else "OFF")

    bar_dt = datetime.fromtimestamp(int(now), tz=timezone.utc)

    if is_new_bar:
        add_log_fn(f"[{symbol} {timeframe}] RSI={rv:.1f} StochK={sk:.1f} "
                   f"ATR={atr_val:.3f} ({atr_pct:.2f}%) "
                   f"BB[{bbl:.2f}/{bbm:.2f}/{bbu:.2f}] Lucky={lucky_str}")

    # display_line ค่าเริ่มต้น (สถานะ indicator) — ใช้กับ NONE path
    base_line1 = f"{symbol} {timeframe} | RSI {rv:.1f} | StochK {sk:.1f}"
    base_line2 = f"ATR {atr_pct:.2f}% | BBw {bb_width_pct:.2f}% | Lucky {lucky_str}"

    # ===== ตรวจ daily drawdown limit → CLOSE_ALL (priority สูงสุด) =====
    max_daily_dd = float(config.get("max_daily_drawdown", 25.0))
    if (sym_positions and balance > 0 and
            ((balance - equity) / balance * 100.0) >= max_daily_dd):
        dd_now = (balance - equity) / balance * 100.0
        add_log_fn(f"Daily drawdown limit hit: {dd_now:.2f}% >= {max_daily_dd}% → CLOSE_ALL")
        res_dict = {
            "action": "CLOSE_ALL",
            "reason": f"Daily drawdown limit {dd_now:.1f}% >= {max_daily_dd:.1f}%",
            "display_line1": "Emergency Exit",
            "display_line2": f"Daily DD {dd_now:.1f}% — closing all",
        }
        return res_dict, updated_config, _live_metrics(rv, sk, atr_pct, bb_width_pct, lucky_str), bt_res

    # ===== Friday close → CLOSE_ALL ถ้ามี position =====
    if (bool(config.get("close_friday_filter", True)) and _is_friday_close(bar_dt)
            and sym_positions):
        add_log_fn("Friday session close → CLOSE_ALL")
        res_dict = {
            "action": "CLOSE_ALL",
            "reason": "Friday session close (>=20:00)",
            "display_line1": "Friday Close",
            "display_line2": "Closing all positions before weekend",
        }
        return res_dict, updated_config, _live_metrics(rv, sk, atr_pct, bb_width_pct, lucky_str), bt_res

    # ===== Position management (live) → CLOSE signal (ไม้เดียว) =====
    action_dict: dict[str, Any] = {"action": "NONE"}
    if sym_positions:
        bb_exit_ratio = float(config.get("bb_exit_ratio", 0.60))
        bb_exit_buy = bbl + bb_exit_ratio * (bbu - bbl)
        bb_exit_sell = bbu - bb_exit_ratio * (bbu - bbl)
        cur_price = float(df["close"].iloc[-1])
        for pos in sym_positions:
            ptype = pos.get("type")
            profit = float(pos.get("profit", 0.0))
            bars_open = int(pos.get("bars_open", 0))
            close_reason = ""
            if bars_open > int(config.get("max_bars_open", 40)):
                close_reason = "Time-exit"
            elif ptype == "BUY" and profit > 0 and cur_price >= bb_exit_buy:
                close_reason = "BB mean-reversion exit (BUY)"
            elif ptype == "SELL" and profit > 0 and cur_price <= bb_exit_sell:
                close_reason = "BB mean-reversion exit (SELL)"
            if close_reason:
                ticket = pos.get("ticket")
                action_dict = {
                    "action": "CLOSE",
                    "ticket": ticket,
                    "reason": close_reason,
                    "display_line1": "Exit Triggered",
                    "display_line2": f"{close_reason} — Ticket #{ticket}",
                }
                break

    # ===== Entry signal (เฉพาะตอนไม่มี position และเป็นแท่งใหม่) =====
    if (action_dict["action"] == "NONE" and not sym_positions and is_new_bar
            and not math.isnan(atr_val) and not math.isnan(bbm)):
        blocked = False
        block_reason = ""
        if bool(config.get("enable_session_filter", True)) and not _is_trading_session(bar_dt, config):
            blocked, block_reason = True, "Outside session"
        if bool(config.get("close_friday_filter", True)) and _is_friday_close(bar_dt):
            blocked, block_reason = True, "Friday close window"
        if bool(config.get("skip_wednesday", True)) and _is_wednesday(bar_dt):
            blocked, block_reason = True, "Wed-skip"
        if spread > int(config.get("max_spread_points", 40)):
            blocked, block_reason = True, f"Spread {spread} too wide"

        if not blocked:
            cfg_tp_mult = float(config.get("atr_mult_tp", 2.5))
            cfg_sl_mult = float(config.get("atr_mult_sl", 1.5))
            high_vol = (c1 > 0 and atr_pct > float(config.get("high_vol_atr_pct", 0.40)))
            bb_wide = (bbm > 0 and bb_width_pct > float(config.get("bb_max_width_pct", 1.5)))

            signal = ""
            tag = ""
            if not high_vol and not bb_wide:
                if _check_bb_bounce_buy(c1, l1, o1, h1, bbl, atr_val, sk, rv, config):
                    signal, tag = "BUY", "BBBuy"
                elif _check_bb_bounce_sell(c1, h1, o1, l1, bbu, atr_val, sk, rv, config):
                    signal, tag = "SELL", "BBSell"

            if not signal and bool(config.get("enable_shr", True)):
                lookback = int(config.get("shr_lookback_bars", 8))
                # iHighest/iLowest จาก lookback แท่ง เริ่ม shift 2
                hi_arr = df["high"].values
                lo_arr = df["low"].values
                end = len(df) - 2          # ไม่รวม bar[1] (เริ่มจาก bar[2])
                start = end - lookback
                if start >= 0:
                    recent_high = float(np.max(hi_arr[start:end]))
                    recent_low = float(np.min(lo_arr[start:end]))
                    if _check_shr_buy(c1, l1, o1, recent_low, bbm, atr_val, sk):
                        signal, tag = "BUY", "SHRBuy"
                    elif _check_shr_sell(c1, h1, o1, recent_high, bbm, atr_val, sk):
                        signal, tag = "SELL", "SHRSell"

            if signal:
                sl_dist = atr_val * cfg_sl_mult
                sl_points = sl_dist / point if point > 0 else 0.0
                lot = calculate_lot_size(
                    balance, config, sl_points, tick_value, tick_size, point,
                    min_lot, max_lot, lot_step,
                    lucky_active=lucky.is_active, lucky_mult=lucky.multiplier,
                    atr_val=atr_val, price=c1)
                add_log_fn(f"{tag} signal → {signal} lot={lot} "
                           f"TPx{cfg_tp_mult} SLx{cfg_sl_mult} (ATR {atr_val:.3f})")
                action_dict = {
                    "action": signal,
                    "lot": round(lot, 2),
                    "tp_multiplier": cfg_tp_mult,   # Option B: client คูณ ATR เอง
                    "sl_multiplier": cfg_sl_mult,
                    "reason": f"{tag} signal",
                    "display_line1": f"Signal: {signal} ({tag})",
                    "display_line2": (f"lot {lot} | TPx{cfg_tp_mult:.1f} "
                                      f"SLx{cfg_sl_mult:.1f} | ATR {atr_val:.3f}"),
                }
            else:
                action_dict = {
                    "action": "NONE",
                    "display_line1": base_line1,
                    "display_line2": base_line2,
                }
        else:
            action_dict = {
                "action": "NONE",
                "display_line1": f"No trade — {block_reason}",
                "display_line2": base_line2,
            }

    # ถ้ายังไม่มี display lines (เช่นมี position แต่ไม่ปิด) → เติมสถานะ indicator
    if "display_line1" not in action_dict:
        action_dict["display_line1"] = base_line1
        action_dict["display_line2"] = base_line2

    res_dict: dict[str, Any] = dict(action_dict)

    live_metrics = _live_metrics(rv, sk, atr_pct, bb_width_pct, lucky_str)

    return res_dict, updated_config, live_metrics, bt_res


def _live_metrics(rv: float, sk: float, atr_pct: float,
                  bb_width_pct: float, lucky_str: str) -> dict[str, str]:
    """live indicators card (key→str ตาม README section 4.3)."""
    return {
        "RSI (14)": f"{rv:.1f}",
        "Stoch %K (5)": f"{sk:.1f}",
        "ATR %": f"{atr_pct:.2f}%",
        "BB Width %": f"{bb_width_pct:.2f}%",
        "Lucky": lucky_str,
    }


if __name__ == "__main__":
    # smoke test: สร้าง synthetic candles แล้วรัน backtest + process_strategy
    import random
    random.seed(42)
    base = 2000.0
    candles_demo: list[dict] = []
    t0 = 1_700_000_000
    price = base
    for i in range(500):
        o = price
        move = random.uniform(-3, 3)
        c = o + move
        h = max(o, c) + random.uniform(0, 2)
        lo = min(o, c) - random.uniform(0, 2)
        candles_demo.append({"time": t0 + i * 300, "open": o, "high": h,
                             "low": lo, "close": c, "tick_volume": 100})
        price = c

    stats = run_backtest(candles_demo, DEFAULT_CONFIG, balance=100.0,
                         tick_value=1.0, tick_size=0.01, point=0.01)
    print("Backtest:", stats)

    res, cfg, metrics, bt = process_strategy(
        {"candles": candles_demo, "positions": [], "symbol": "XAUUSD",
         "timeframe": "M5", "is_new_bar": True, "balance": 100.0,
         "equity": 100.0, "spread": 10, "trigger_backtest": True},
        DEFAULT_CONFIG, lambda m: print("LOG:", m))
    print("Action:", res.get("action"), "| line1:", res.get("display_line1"),
          "| line2:", res.get("display_line2"))
    print("bt_res:", bt)
