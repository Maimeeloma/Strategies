"""
marday_v46.py — Python port ของ MarDay_v46.0.0.mq5 (FUSION)

ยึดโครงสร้าง/สไตล์เดียวกับ simple-2-3.py:
  - pandas + numpy
  - DEFAULT_CONFIG dict สะท้อน input parameters หลักของ v46
  - ฟังก์ชัน indicator แยกตัว
  - backtest engine แบบ bar-by-bar (อ่าน indicator จาก bar[i-1] = closed bar กัน repaint)
  - process_strategy(data, config, add_log_fn) -> (res_dict, updated_config, live_metrics, bt_res)
  - SL/TP เป็น price level, profit = (profit_points / tick_size) * tick_value * lot_size

v46 FUSION = base Today(v43.2) killer-MM + superset indicators + per-day distinct engines
+ off-hours layer + confidence-mult + Lucky element. แต่ละ "engine" คือ logic ของแต่ละวัน:
  Mon = Asian-Range Keltner Fade (mean-reversion)
  Tue = VWAP sigma-band Reversion (ADX-gated)  [default skip]
  Wed = WVEC vol-expansion continuation (impulse-bar) [default skip]
  Thu = NY-trend MOMENTUM (N-bar breakout + impulse)
  Fri = Legacy BB-MR + SHR (CHOP regime-gated)
  off-hours = NR7 squeeze->expansion breakout (+ engine อื่น ๆ)

อ้างอิงบรรทัด MQL5 ระบุไว้ในคอมเมนต์ของแต่ละฟังก์ชัน (เช่น "ported from CheckOffSqzBuy L1139").
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

STRATEGY_NAME = "MarDay-v46.0.0-FUSION"

# ==========================================================================
# DEFAULT_CONFIG — สะท้อน input parameters หลักของ v46.0.0.mq5 (v39.9-omg baked)
# ==========================================================================
DEFAULT_CONFIG: dict[str, Any] = {
    # --- Bollinger Bands ---
    "bb_period": 20,
    "bb_deviation": 2.5,            # InpBBDeviation
    "bb_proximity_atr": 0.35,       # InpBBProximityATR
    "bb_max_width_pct": 1.5,        # InpBBMaxWidthPct
    "bb_exit_ratio": 0.60,          # InpBBExitRatio (v1.16.7)
    # --- Stochastic (fast) ---
    "stoch_k": 5,
    "stoch_d": 3,
    "stoch_slowing": 3,
    "stoch_oversold": 30,           # InpStochOversold
    "stoch_overbought": 70,         # InpStochOverbought
    # --- ATR / SL-TP base ---
    "atr_period": 21,               # InpATRPeriod
    "atr_mult_sl": 1.2,             # InpATRMultSL
    "atr_mult_tp": 2.5,             # InpATRMultTP
    "max_sl_pts": 500.0,            # InpMaxSLPts (hard cap)
    "max_tp_pts": 1000.0,           # InpMaxTPPts (hard cap)
    # --- RSI ---
    "enable_rsi": True,
    "rsi_period": 14,
    # --- Candle ---
    "cps_min": 0.45,                # InpCPSMin
    # --- Risk / killer MM (v43.2 auto-scale) ---
    "risk_mode": "RISK_PERCENT",    # InpRiskMode
    "fixed_lot": 0.01,
    "risk_percent": 2.0,            # InpRiskPercent (max aggression)
    "max_daily_drawdown": 10.0,     # InpMaxDailyDrawdown
    "max_open_trades": 4,           # InpMaxOpenTrades
    "auto_scale_by_capital": True,  # InpAutoScaleByCapital
    "max_account_risk_pct": 12.0,   # InpMaxAccountRiskPct
    "consecutive_loss_limit": 3,    # InpConsecutiveLossLimit
    # --- Spread / session ---
    "max_spread_points": 40,
    "session_start_hour": 8,
    "session_start_min": 0,
    "session_end_hour": 20,
    "session_end_min": 0,
    "close_friday_filter": True,
    # --- Trade management ---
    "enable_breakeven": True,
    "breakeven_trigger_pts": 800.0,
    "breakeven_buffer_pts": 80.0,
    "enable_trailing": True,
    "trailing_start_pts": 1200.0,
    "trailing_step_pts": 200.0,
    "one_trade_per_bar": True,
    "cooldown_bars_after_loss": 0,  # InpCooldownBarsAfterLoss (v39.9-omg)
    "max_bars_open": 20,            # InpMaxBarsOpen
    # --- Stop Hunt Reversal ---
    "enable_shr": True,
    "shr_lookback_bars": 8,
    "high_vol_atr_pct": 0.40,       # InpHighVolATRPct
    # --- Confidence-Based Lot Multiplier (v32/v46) ---
    "enable_confidence_mult": True,
    "confidence_max_mult": 2,       # x2 (sweet spot DD 17%)
    "conf_body_strong": 0.75,
    "conf_stoch_extreme": 15.0,
    "conf_rsi_extreme": 25.0,
    "conf_bb_deep_atr": 0.50,
    "conf_sweep_strength_atr": 0.50,
    "conf_pinbar_wick_ratio": 2.0,
    "conf_engulf_ratio": 1.3,
    "conf_bar_size_ratio": 1.3,
    "conf_momentum_bars": 3,
    "conf_vol_expansion": 1.2,
    "day_boost_min_conf": 3,
    "enable_day_boost": False,
    "mon_boost_mult": 2.0,
    "fri_boost_mult": 2.0,
    # --- Lucky element (v11.9 -> re-enabled in v46) ---
    "lucky_minute": 75,             # InpLuckyMinute (AKASA 75min)
    "lucky_max_mult": 5,            # InpLuckyMaxMult
    "lucky_duration": 75,           # ELEM_AKASA -> 75 min
    "lucky_cycle_min": 99,          # LUCKY_CYCLE_MIN
    # --- Per-day distinct engines (v39.0+) ---
    "enable_per_day": True,
    "friday_uses_legacy": True,
    "enable_mon_engine": True,
    "enable_tue_engine": True,
    "enable_wed_engine": True,
    "enable_thu_engine": True,
    # --- Day-of-week control ---
    "skip_sunday": True,
    "skip_tuesday": True,
    "skip_wednesday": True,
    # --- CHOP regime gate ---
    "use_chop_gate": True,
    "chop_period": 22,              # InpChopPeriod (H1)
    "chop_range_thr": 53.0,         # InpChopRangeThr (Friday MR allow)
    # --- Monday: Asian-Range Keltner Fade ---
    "mon_sl_mult": 1.8,
    "mon_tp_mult": 2.5,
    "mon_kelt_ema": 20,
    "mon_kelt_atr": 10,
    "mon_kelt_mult": 2.0,
    "mon_rsi2_period": 2,
    "mon_rsi2_os": 15,
    "mon_rsi2_ob": 90,
    "mon_vol_slow_atr": 50,
    "mon_vol_gate_max": 1.7,
    "mon_start_hour": 1,
    "mon_end_hour": 14,
    # --- Tuesday: VWAP sigma-band Reversion ---
    "tue_sl_mult": 2.2,
    "tue_tp_mult": 3.5,
    "tue_vwap_mult": 1.75,
    "tue_adx_max": 40.0,
    "tue_start_hour": 7,
    "tue_end_hour": 19,
    "tue_band_min_atr": 0.8,
    "tue_ema_fast": 9,
    "tue_ema_mid": 21,
    "tue_ema_slow": 50,
    "tue_ema_trend": 200,
    "tue_adx_period": 14,
    # --- Wednesday: WVEC vol-expansion continuation ---
    "wed_sl_mult": 2.0,
    "wed_tp_mult": 2.5,
    "wed_exp_mult": 1.1,
    "wed_body_frac": 0.40,
    "wed_opp_wick": 0.30,
    "wed_start_hour": 18,
    "wed_end_hour": 22,
    "wed_vol_gate_lo": 0.80,
    "wed_atr_slow_period": 20,
    # --- Thursday: NY-trend momentum (default) / Donchian fade (alt) ---
    "thu_use_momentum": True,
    "thu_mom_lookback": 8,
    "thu_mom_exp_mult": 0.6,
    "thu_mom_chop_max": 38.0,
    "thu_mom_adx_min": 16.0,
    "thu_mom_sl_mult": 1.2,
    "thu_mom_tp_mult": 2.5,
    "thu_mom_start": 13,
    "thu_mom_end": 20,
    "thu_atr_fast_period": 5,
    "thu_atr_slow_period": 20,
    "thu_donchian": 20,
    "thu_wpr_period": 14,
    "thu_wpr_os": -90.0,
    "thu_wpr_ob": -10.0,
    "thu_adx_max_h1": 25.0,
    "thu_vol_lo": 0.7,
    "thu_vol_hi": 2.0,
    "thu_sl_mult": 1.5,
    "thu_tp_mult": 2.5,
    "thu_start_hour": 13,
    "thu_end_hour": 19,
    # --- Off-hours layer (v41+) ---
    "enable_off_hours": True,
    "off_hours_engine": 5,          # OFFENG_SQUEEZE (0=perday,1=sweep,2=vwap,3=mom,4=indi,5=squeeze)
    "off_hours_risk_percent": 0.5,
    "off_hours_max_trades": 2,
    "off_hours_cooldown_bars": 6,
    "max_spread_off_hours": 40.0,
    "off_sqz_start": 2,
    "off_sqz_end": 8,
    "off_sqz_lookback": 7,
    "off_sqz_exp_mult": 1.0,
    "off_sqz_sl_mult": 1.0,
    "off_sqz_tp_mult": 3.0,
    # off-hours sweep (engine 1)
    "off_sweep_start": 2,
    "off_sweep_end": 8,
    "off_sweep_lookback": 12,
    "off_sweep_chop_min": 53.0,
    "off_sweep_stoch_os": 25,
    "off_sweep_stoch_ob": 75,
    "off_sweep_sl_mult": 1.2,
    "off_sweep_tp_mult": 1.5,
    # off-hours vwap (engine 2)
    "off_vwap_start": 2,
    "off_vwap_end": 8,
    "off_vwap_chop_min": 53.0,
    "off_vwap_band_min_atr": 0.5,
    "off_vwap_sl_mult": 1.2,
    "off_vwap_tp_mult": 2.0,
    # off-hours momentum (engine 3)
    "off_mom_start": 20,
    "off_mom_end": 24,
    "off_mom_lookback": 8,
    "off_mom_exp_mult": 0.6,
    "off_mom_chop_max": 38.0,
    "off_mom_sl_mult": 1.2,
    "off_mom_tp_mult": 2.5,
    # off-hours indicator (engine 4)
    "off_indi_mode": 1,
    "off_indi_adx_min": 25.0,
    "off_indi_cci_thr": 100.0,
    "off_indi_rsi2_os": 10,
    "off_indi_rsi2_ob": 90,
    "off_indi_sl_mult": 1.2,
    "off_indi_tp_mult": 2.0,
}


# ==========================================================================
# Indicators (vectorized; matched to MT5 buffer semantics)
# ==========================================================================
def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR แบบ Wilder (RMA) — ตรงกับ iATR ของ MT5."""
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI แบบ Wilder — ตรงกับ iRSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(100.0)


def calculate_ema(close: pd.Series, period: int) -> pd.Series:
    """EMA — ตรงกับ iMA(MODE_EMA)."""
    return close.ewm(span=period, adjust=False).mean()


def calculate_bollinger(close: pd.Series, period: int = 20, dev: float = 2.5
                        ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands (SMA + std*dev). คืน (middle, upper, lower) — ตรงกับ iBands."""
    middle = close.rolling(window=period).mean()
    # MT5 iBands ใช้ population std (ddof=0)
    sd = close.rolling(window=period).std(ddof=0)
    upper = middle + dev * sd
    lower = middle - dev * sd
    return middle, upper, lower


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                         k: int = 5, d: int = 3, slowing: int = 3
                         ) -> tuple[pd.Series, pd.Series]:
    """Stochastic %K/%D (STO_LOWHIGH, MODE_SMA) — ตรงกับ iStochastic."""
    lowest = low.rolling(window=k).min()
    highest = high.rolling(window=k).max()
    rng = (highest - lowest).replace(0, np.nan)
    raw_k = 100.0 * (close - lowest) / rng
    # slowing = SMA ของ (close-low) / (high-low) — ใช้ SMA ของ raw_k เป็น approximation มาตรฐาน
    slow_k = raw_k.rolling(window=slowing).mean()
    slow_d = slow_k.rolling(window=d).mean()
    return slow_k.fillna(50.0), slow_d.fillna(50.0)


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
                  ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """ADX + +DI/-DI (Wilder) — คืน (adx, plus_di, minus_di). ตรงกับ iADX."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return adx.fillna(0.0), plus_di.fillna(0.0), minus_di.fillna(0.0)


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """CCI บน typical price — ตรงกับ iCCI(PRICE_TYPICAL)."""
    tp = (high + low + close) / 3.0
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    return cci.fillna(0.0)


def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R (range -100..0) — ตรงกับ iWPR."""
    highest = high.rolling(window=period).max()
    lowest = low.rolling(window=period).min()
    rng = (highest - lowest).replace(0, np.nan)
    wpr = -100.0 * (highest - close) / rng
    return wpr.fillna(-50.0)


def calculate_keltner(close: pd.Series, high: pd.Series, low: pd.Series,
                      ema_period: int = 20, atr_period: int = 10, mult: float = 2.0
                      ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channel: EMA midline +/- mult*ATR. คืน (mid, upper, lower)."""
    mid = calculate_ema(close, ema_period)
    atr = calculate_atr(high, low, close, atr_period)
    upper = mid + mult * atr
    lower = mid - mult * atr
    return mid, upper, lower


def calculate_vwap(df: pd.DataFrame, sigma_mult: float = 1.75
                   ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Session VWAP (reset ที่ขึ้นวันใหม่ตาม server-day) + sigma bands.
    Ported from ComputeSessionVWAP L2701: typical price = (H+L+C)/3, volume-weighted,
    variance = sumPPV/sumV - vwap^2.
    คืน (vwap, upper, lower). ต้องมีคอลัมน์ 'time' (epoch sec) และ 'volume'.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df.get("volume", pd.Series(1.0, index=df.index)).astype(float)
    vol = vol.where(vol > 0, 1.0)
    # group ตามวัน (server day = floor(epoch/86400))
    day = (pd.to_numeric(df["time"]) // 86400).astype("int64")

    pv = tp * vol
    ppv = tp * tp * vol
    cum_pv = pv.groupby(day).cumsum()
    cum_v = vol.groupby(day).cumsum()
    cum_ppv = ppv.groupby(day).cumsum()

    vwap = cum_pv / cum_v.replace(0, np.nan)
    var = cum_ppv / cum_v.replace(0, np.nan) - vwap * vwap
    sd = np.sqrt(var.clip(lower=0.0))
    upper = vwap + sigma_mult * sd
    lower = vwap - sigma_mult * sd
    return vwap, upper, lower


def calculate_choppiness(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 22) -> pd.Series:
    """
    Choppiness Index — ported from GetChop L1716.
    CHOP = 100 * log10( sum(TR,n) / (HH(n)-LL(n)) ) / log10(n).
    สูง (>~61.8)=range/chop, ต่ำ (<~38.2)=trend.
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    sum_tr = tr.rolling(window=period).sum()
    hh = high.rolling(window=period).max()
    ll = low.rolling(window=period).min()
    rng = (hh - ll).replace(0, np.nan)
    chop = 100.0 * np.log10(sum_tr / rng) / math.log10(period)
    return chop


def calculate_lot_size(balance: float, risk_percent: float, sl_dist: float,
                       tick_value: float, tick_size: float,
                       min_lot: float, max_lot: float, lot_step: float,
                       conf_mult: int = 1, lucky_mult: int = 1,
                       day_boost: float = 1.0) -> float:
    """
    Lot sizing — ported from CalculateLotSize L2317.
    risk = balance * risk%/100 ; loss_per_lot = (sl_dist/tick_size)*tick_value ; lot = risk/loss_per_lot
    จากนั้นคูณ confidence (หรือ lucky ถ้า conf ปิด) แล้ว day-boost ถ้า conf>=min, สุดท้าย floor ตาม step + clamp.
    """
    if tick_value <= 0 or tick_size <= 0 or sl_dist <= 0:
        return float(min_lot)
    risk_amount = balance * (risk_percent / 100.0)
    loss_per_lot = (sl_dist / tick_size) * tick_value
    if loss_per_lot <= 0:
        return float(min_lot)
    lot = risk_amount / loss_per_lot

    # confidence mult แทน lucky (ถ้า conf เปิด); ไม่งั้นใช้ lucky
    if conf_mult > 1:
        lot *= conf_mult
    elif lucky_mult > 1:
        lot *= lucky_mult
    if day_boost > 1.0:
        lot *= day_boost

    if lot_step > 0:
        lot = math.floor(lot / lot_step) * lot_step
    return float(max(min_lot, min(max_lot, lot)))


# ==========================================================================
# MM: capital scaling — ported from ApplyCapitalScaling L486
# ==========================================================================
def apply_capital_scaling(balance: float, atr: float, tick_value: float, tick_size: float,
                          min_lot: float, atr_mult_sl: float, max_account_risk_pct: float,
                          max_open_trades: int, off_hours_max: int,
                          auto_scale: bool = True) -> tuple[int, int]:
    """
    คืน (eff_max_open, eff_off_max).
    max ไม้ = floor( (balance * maxAccountRiskPct%) / riskPerMinLot$ )
    riskPerMinLot = min_lot * (atr_mult_sl*atr/tick_size)*tick_value ; cap ด้วย max_open_trades.
    off-hours max = floor(max_open/2).
    """
    if not auto_scale:
        return max_open_trades, off_hours_max
    if balance <= 0 or tick_value <= 0 or tick_size <= 0 or min_lot <= 0 or atr <= 0:
        return 1, 1
    sl_money_per_lot = (atr_mult_sl * atr / tick_size) * tick_value
    risk_per_min_lot = min_lot * sl_money_per_lot
    if risk_per_min_lot <= 0:
        return 1, 1
    budget = balance * (max_account_risk_pct / 100.0)
    n = int(math.floor(budget / risk_per_min_lot))
    eff_max = int(max(1, min(n, max_open_trades)))
    eff_off = int(max(1, min(math.floor(eff_max / 2.0), off_hours_max)))
    return eff_max, eff_off


# ==========================================================================
# Bar-view helper — ให้ engine functions อ่านค่า bar[shift] ได้แบบ MQL5
# ==========================================================================
class BarView:
    """
    ตัวช่วยให้ engine functions อ่านค่าแบบ MQL5 (bar[1]=closed bar, ใหญ่ขึ้น = เก่ากว่า).
    `cur` = index ของ "forming bar 0" ใน array; bar(shift) -> ค่าที่ cur-shift.
    """

    __slots__ = ("o", "h", "l", "c", "v", "cur")

    def __init__(self, o, h, l, c, v, cur: int):
        self.o, self.h, self.l, self.c, self.v, self.cur = o, h, l, c, v, cur

    def _idx(self, shift: int) -> int:
        return self.cur - shift

    def open(self, shift: int) -> float:
        return float(self.o[self._idx(shift)])

    def high(self, shift: int) -> float:
        return float(self.h[self._idx(shift)])

    def low(self, shift: int) -> float:
        return float(self.l[self._idx(shift)])

    def close(self, shift: int) -> float:
        return float(self.c[self._idx(shift)])

    def volume(self, shift: int) -> float:
        return float(self.v[self._idx(shift)])

    def highest(self, lookback: int, start_shift: int) -> float:
        """เทียบเท่า iHighest(MODE_HIGH, lookback, start_shift)."""
        s = self._idx(start_shift)
        seg = self.h[s - lookback + 1: s + 1]
        return float(np.max(seg)) if len(seg) else 0.0

    def lowest(self, lookback: int, start_shift: int) -> float:
        """เทียบเท่า iLowest(MODE_LOW, lookback, start_shift)."""
        s = self._idx(start_shift)
        seg = self.l[s - lookback + 1: s + 1]
        return float(np.min(seg)) if len(seg) else 0.0


def is_bullish_body(bv: BarView, shift: int) -> bool:
    return bv.close(shift) > bv.open(shift)


def is_bearish_body(bv: BarView, shift: int) -> bool:
    return bv.close(shift) < bv.open(shift)


def get_cps(bv: BarView, shift: int) -> float:
    """Candle Position Score = (close-low)/(high-low). Ported from GetCPS L1576."""
    h, l, c = bv.high(shift), bv.low(shift), bv.close(shift)
    if h <= l:
        return 0.5
    return (c - l) / (h - l)


# ==========================================================================
# Confidence — ported from GetSetupConfidence L2196
# ==========================================================================
def compute_confidence(bv: BarView, ind: dict[str, float], cfg: dict[str, Any],
                       signal_tag: str) -> int:
    """
    นับ confidence factors -> multiplier (1..confidence_max_mult).
    F1 strong body, F2 pin bar, F3 engulf, F4 bar-size, F5 momentum seq,
    F6 stoch/RSI extreme, F7 BB-deep / SHR-sweep strength.
    """
    if not bool(cfg.get("enable_confidence_mult", True)):
        return 1
    max_mult = int(cfg.get("confidence_max_mult", 2))
    score = 1
    is_buy = "Buy" in signal_tag

    o1, c1, h1, l1 = bv.open(1), bv.close(1), bv.high(1), bv.low(1)
    rng1 = h1 - l1
    atr = float(ind.get("atr", 0.0))
    if rng1 <= 0 or atr <= 0:
        return 1
    body1 = abs(c1 - o1)
    body_ratio = body1 / rng1
    upper_wick = h1 - max(c1, o1)
    lower_wick = min(c1, o1) - l1

    # F1: strong body
    if body_ratio >= float(cfg["conf_body_strong"]):
        score += 1
    # F2: pin bar rejection
    if body1 > 0:
        if is_buy and lower_wick >= body1 * float(cfg["conf_pinbar_wick_ratio"]):
            score += 1
        if (not is_buy) and upper_wick >= body1 * float(cfg["conf_pinbar_wick_ratio"]):
            score += 1
    # F3: engulfing (bar1 range >= bar2 range * ratio)
    rng2 = bv.high(2) - bv.low(2)
    if rng2 > 0 and rng1 >= rng2 * float(cfg["conf_engulf_ratio"]):
        score += 1
    # F4: bar size vs 5-bar avg (shift 2..6)
    avg5, cnt = 0.0, 0
    for i in range(2, 7):
        r = bv.high(i) - bv.low(i)
        if r > 0:
            avg5 += r
            cnt += 1
    if cnt > 0:
        avg5 /= cnt
        if avg5 > 0 and rng1 >= avg5 * float(cfg["conf_bar_size_ratio"]):
            score += 1
    # F5: momentum sequence
    mom_bars = int(cfg["conf_momentum_bars"])
    if "BB" in signal_tag:
        opp = 0
        for i in range(2, mom_bars + 2):
            if is_buy and bv.close(i) < bv.open(i):
                opp += 1
            elif (not is_buy) and bv.close(i) > bv.open(i):
                opp += 1
        if opp >= mom_bars:
            score += 1
    else:
        same = 0
        for i in range(1, mom_bars + 1):
            if is_buy and bv.close(i) > bv.open(i):
                same += 1
            elif (not is_buy) and bv.close(i) < bv.open(i):
                same += 1
        if same >= mom_bars - 1:
            score += 1
    # F6: stoch / RSI extreme
    stoch_k = float(ind.get("stoch_k", 50.0))
    rsi = float(ind.get("rsi", 50.0))
    ext = float(cfg["conf_stoch_extreme"])
    rsi_ext = float(cfg["conf_rsi_extreme"])
    if is_buy:
        if stoch_k < ext:
            score += 1
        if bool(cfg.get("enable_rsi", True)) and rsi < rsi_ext:
            score += 1
    else:
        if stoch_k > 100.0 - ext:
            score += 1
        if bool(cfg.get("enable_rsi", True)) and rsi > 100.0 - rsi_ext:
            score += 1
    # F7: BB deep / SHR strong sweep
    if "BB" in signal_tag:
        if is_buy:
            pen = (float(ind.get("bb_lower", 0.0)) - l1) / atr
        else:
            pen = (h1 - float(ind.get("bb_upper", 0.0))) / atr
        if pen >= float(cfg["conf_bb_deep_atr"]):
            score += 1
    elif "SHR" in signal_tag:
        rl = float(ind.get("recent_low", 0.0))
        rh = float(ind.get("recent_high", 0.0))
        if is_buy and rl > 0:
            if (rl - l1) / atr >= float(cfg["conf_sweep_strength_atr"]):
                score += 1
        elif (not is_buy) and rh > 0:
            if (h1 - rh) / atr >= float(cfg["conf_sweep_strength_atr"]):
                score += 1

    return int(max(1, min(score, max_mult)))


# ==========================================================================
# Lucky cycle — ported from RefreshLuckyCycle L710
# ==========================================================================
def lucky_multiplier(epoch_sec: float, cycle_start: float, cfg: dict[str, Any]) -> int:
    """
    คืน 1 (ไม่ lucky) หรือ 2 (เริ่ม lucky). NB: หลังชนะ/แพ้ multiplier ปรับใน backtest loop.
    window = [lucky_minute, lucky_minute+duration) ภายในรอบ lucky_cycle_min นาที (wrap ข้ามรอบ).
    """
    lucky_min = int(cfg.get("lucky_minute", 0))
    duration = int(cfg.get("lucky_duration", 0))
    cyc = int(cfg.get("lucky_cycle_min", 99))
    if lucky_min < 1 or lucky_min > 99 or duration <= 0:
        return 1
    elapsed = int((epoch_sec - cycle_start) // 60) % cyc
    lucky_end = lucky_min + duration
    if lucky_end <= cyc:
        in_lucky = (lucky_min <= elapsed < lucky_end)
    else:
        in_lucky = (elapsed >= lucky_min) or (elapsed < (lucky_end - cyc))
    return 2 if in_lucky else 1


# ==========================================================================
# CHOP regime gate — ported from RegimeAllowsMR L1738 (CHOP-gate branch)
# ==========================================================================
def regime_allows_mr(chop: float, cfg: dict[str, Any]) -> bool:
    """range (chop>=thr) -> allow MR; trend -> block. fail-open ถ้า chop เป็น nan."""
    if not bool(cfg.get("use_chop_gate", True)):
        return True
    if chop is None or (isinstance(chop, float) and math.isnan(chop)):
        return True
    return chop >= float(cfg["chop_range_thr"])


# ==========================================================================
# Per-day SIGNAL ENGINES — แต่ละฟังก์ชันคืน (+1 buy / -1 sell / 0 none)
# ==========================================================================
def signal_mon_keltner(bv: BarView, ind: dict[str, float], cfg: dict[str, Any], hour: int) -> int:
    """
    MONDAY: Asian-Range Keltner Fade (MR).
    Ported from CheckMondayKeltFadeBuy/Sell L1851/L1864:
    pierce Keltner band (EMA20 +/- mult*ATR10), close back inside, RSI2 extreme, vol-gate, window hr 1-14.
    """
    kelt_mid = float(ind.get("mon_ema", 0.0))
    kelt_atr = float(ind.get("mon_atr", 0.0))
    kelt_atr50 = float(ind.get("mon_atr50", 0.0))
    rsi2 = float(ind.get("mon_rsi2", 50.0))
    if kelt_mid <= 0 or kelt_atr <= 0:
        return 0
    if not (cfg["mon_start_hour"] <= hour < cfg["mon_end_hour"]):
        return 0
    # vol-gate: ATR10 <= volGateMax * ATR50
    if kelt_atr50 <= 0 or kelt_atr > float(cfg["mon_vol_gate_max"]) * kelt_atr50:
        return 0
    mult = float(cfg["mon_kelt_mult"])
    c1, l1, h1 = bv.close(1), bv.low(1), bv.high(1)
    kelt_lower = kelt_mid - mult * kelt_atr
    kelt_upper = kelt_mid + mult * kelt_atr
    # buy: pierce lower, reclaim, RSI2 oversold
    if l1 <= kelt_lower and c1 > kelt_lower and rsi2 < float(cfg["mon_rsi2_os"]):
        return +1
    # sell: pierce upper, reclaim, RSI2 overbought
    if h1 >= kelt_upper and c1 < kelt_upper and rsi2 > float(cfg["mon_rsi2_ob"]):
        return -1
    return 0


def signal_tue_vwap(bv: BarView, ind: dict[str, float], cfg: dict[str, Any], hour: int) -> int:
    """
    TUESDAY: VWAP sigma-band Reversion (ADX-gated, MR).
    Ported from CheckTuesdayVWAPBuy/Sell L1930/L1942 + TuesdayWindowOK L1921:
    fade pierce ของ VWAP +/- 1.75sigma ด้วย rejection candle, ADX<=40, window hr 7-19, band>=0.8*ATR.
    """
    vwap = float(ind.get("vwap", 0.0))
    vwap_up = float(ind.get("vwap_upper", 0.0))
    vwap_lo = float(ind.get("vwap_lower", 0.0))
    adx = float(ind.get("adx", 0.0))
    atr = float(ind.get("atr", 0.0))
    if not (cfg["tue_start_hour"] <= hour < cfg["tue_end_hour"]):
        return 0
    if vwap <= 0 or vwap_up <= 0:
        return 0
    if adx > float(cfg["tue_adx_max"]):
        return 0
    if (vwap_up - vwap_lo) < float(cfg["tue_band_min_atr"]) * atr:
        return 0
    o1, c1, h1, l1 = bv.open(1), bv.close(1), bv.high(1), bv.low(1)
    # buy: pierce -sigma, bullish rejection back toward vwap
    if l1 <= vwap_lo and c1 < vwap and c1 > o1 and c1 > l1 + 0.5 * (h1 - l1):
        return +1
    # sell: pierce +sigma, bearish rejection
    if h1 >= vwap_up and c1 > vwap and c1 < o1 and c1 < h1 - 0.5 * (h1 - l1):
        return -1
    return 0


def signal_wed_wvec(bv: BarView, ind: dict[str, float], cfg: dict[str, Any], hour: int) -> int:
    """
    WEDNESDAY: WVEC vol-expansion continuation (TREND, impulse-bar).
    Ported from CheckWedVECBuy/Sell L2027/L2042 + WedVECWindowOK L2019:
    bar1 range >= ExpMult*ATR[2], body>=0.40*range, opp wick<=0.30*range, ทิศตาม EMA21, ADX rising, window hr 18-22.
    """
    atr_prev = float(ind.get("atr_prev", 0.0))
    atr_slow = float(ind.get("atr_slow", 0.0))
    atr = float(ind.get("atr", 0.0))
    ema_mid = float(ind.get("ema_mid", 0.0))
    adx = float(ind.get("adx", 0.0))
    adx_prev = float(ind.get("adx_prev", 0.0))
    if not (cfg["wed_start_hour"] <= hour < cfg["wed_end_hour"]):
        return 0
    if atr_prev <= 0 or atr_slow <= 0:
        return 0
    if atr < float(cfg["wed_vol_gate_lo"]) * atr_slow:
        return 0
    o1, c1, h1, l1 = bv.open(1), bv.close(1), bv.high(1), bv.low(1)
    rng1 = h1 - l1
    if rng1 <= 0 or rng1 < float(cfg["wed_exp_mult"]) * atr_prev:
        return 0
    body_frac = float(cfg["wed_body_frac"])
    opp_wick = float(cfg["wed_opp_wick"])
    # bullish impulse continuation
    if (c1 - o1) > 0 and (c1 - o1) >= body_frac * rng1 and \
       (min(o1, c1) - l1) <= opp_wick * rng1 and \
       (ema_mid <= 0 or c1 > ema_mid) and adx > adx_prev:
        return +1
    # bearish impulse continuation
    if (o1 - c1) > 0 and (o1 - c1) >= body_frac * rng1 and \
       (h1 - max(o1, c1)) <= opp_wick * rng1 and \
       (ema_mid <= 0 or c1 < ema_mid) and adx > adx_prev:
        return -1
    return 0


def signal_thu_momentum(bv: BarView, ind: dict[str, float], cfg: dict[str, Any], hour: int) -> int:
    """
    THURSDAY: NY-trend MOMENTUM (default engine).
    Ported from CheckThursdayMomoBuy/Sell L2140/L2154 + ThursdayMomoGateOK L2131:
    breakout เหนือ prior N-bar(8) high + impulse (range>=0.6*ATR), CHOP<38 (trending),
    H1 ADX>=16, window hr 13-20.
    """
    atr = float(ind.get("atr", 0.0))
    adx_h1 = float(ind.get("adx_h1", 0.0))
    chop = ind.get("chop", float("nan"))
    if atr <= 0:
        return 0
    if not (cfg["thu_mom_start"] <= hour < cfg["thu_mom_end"]):
        return 0
    if adx_h1 < float(cfg["thu_mom_adx_min"]):
        return 0
    if not (isinstance(chop, float) and math.isnan(chop)) and chop > float(cfg["thu_mom_chop_max"]):
        return 0  # ranging -> block (fail-open ถ้า chop nan)
    look = int(cfg["thu_mom_lookback"])
    exp = float(cfg["thu_mom_exp_mult"])
    o1, c1, h1, l1 = bv.open(1), bv.close(1), bv.high(1), bv.low(1)
    prior_high = bv.highest(look, 2)
    prior_low = bv.lowest(look, 2)
    if c1 > prior_high and c1 > o1 and (h1 - l1) >= exp * atr:
        return +1
    if c1 < prior_low and c1 < o1 and (h1 - l1) >= exp * atr:
        return -1
    return 0


def signal_thu_fade(bv: BarView, ind: dict[str, float], cfg: dict[str, Any], hour: int) -> int:
    """
    THURSDAY (alt): Donchian FADE + Williams %R (MR).
    Ported from CheckThursdayFadeBuy/Sell L2098/L2113 + ThursdayGateOK L2089:
    touch Donchian extreme, close back inside, %R exhaustion, H1 ADX<=25, vol-gate.
    """
    wpr = float(ind.get("thu_wpr", -50.0))
    adx_h1 = float(ind.get("adx_h1", 0.0))
    atr = float(ind.get("atr", 0.0))
    atr50 = float(ind.get("mon_atr50", 0.0))
    if not (cfg["thu_start_hour"] <= hour < cfg["thu_end_hour"]):
        return 0
    if adx_h1 > float(cfg["thu_adx_max_h1"]):
        return 0
    if atr <= 0 or atr50 <= 0:
        return 0
    vr = atr / atr50
    if not (cfg["thu_vol_lo"] <= vr <= cfg["thu_vol_hi"]):
        return 0
    don = int(cfg["thu_donchian"])
    don_low = bv.lowest(don, 2)
    don_high = bv.highest(don, 2)
    c1, l1, h1 = bv.close(1), bv.low(1), bv.high(1)
    if l1 <= don_low and c1 > don_low and wpr <= float(cfg["thu_wpr_os"]):
        return +1
    if h1 >= don_high and c1 < don_high and wpr >= float(cfg["thu_wpr_ob"]):
        return -1
    return 0


def signal_fri_legacy_mr(bv: BarView, ind: dict[str, float], cfg: dict[str, Any]) -> tuple[int, str]:
    """
    FRIDAY: Legacy BB-MR + SHR (CHOP regime-gated).
    Ported from RunLegacyStack L1304 / CheckBBBounce* L1599 / CheckSHR* L1655.
    คืน (dir, tag). dir +1/-1/0, tag = 'BBBuy'/'SHRSell'/... สำหรับ confidence routing.
    NB: caller ต้องเช็ค regime_allows_mr ก่อน (Friday-only gate).
    """
    atr = float(ind.get("atr", 0.0))
    bb_up = float(ind.get("bb_upper", 0.0))
    bb_mid = float(ind.get("bb_middle", 0.0))
    bb_lo = float(ind.get("bb_lower", 0.0))
    stoch_k = float(ind.get("stoch_k", 50.0))
    rsi = float(ind.get("rsi", 50.0))
    bid = bv.close(1)  # proxy ราคาปัจจุบันใน backtest

    high_vol = (bid > 0 and (atr / bid) * 100.0 > float(cfg["high_vol_atr_pct"]))
    bb_wide = (bb_mid > 0 and (bb_up - bb_lo) / bb_mid * 100.0 > float(cfg["bb_max_width_pct"]))

    c1, l1, h1 = bv.close(1), bv.low(1), bv.high(1)
    prox = float(cfg["bb_proximity_atr"])
    cps_min = float(cfg["cps_min"])

    if not high_vol and not bb_wide:
        # BB bounce buy
        if l1 <= bb_lo + atr * prox and not (c1 < bb_lo - atr * 0.50) and \
           stoch_k < float(cfg["stoch_oversold"]) and \
           not (bool(cfg.get("enable_rsi", True)) and rsi >= 55.0) and \
           get_cps(bv, 1) >= cps_min and is_bullish_body(bv, 1):
            return +1, "BBBuy"
        # BB bounce sell
        if h1 >= bb_up - atr * prox and not (c1 > bb_up + atr * 0.50) and \
           stoch_k > float(cfg["stoch_overbought"]) and \
           not (bool(cfg.get("enable_rsi", True)) and rsi <= 45.0) and \
           get_cps(bv, 1) <= (1.0 - cps_min) and is_bearish_body(bv, 1):
            return -1, "BBSell"

    if bool(cfg.get("enable_shr", True)):
        rl = float(ind.get("recent_low", 0.0))
        rh = float(ind.get("recent_high", 0.0))
        # SHR buy: sweep recent low, close back above, oversold
        if rl > 0 and l1 < rl and c1 > rl and is_bullish_body(bv, 1) and \
           stoch_k < 30.0 and not (c1 > bb_mid + atr * 0.50):
            return +1, "SHRBuy"
        # SHR sell
        if rh > 0 and h1 > rh and c1 < rh and is_bearish_body(bv, 1) and \
           stoch_k > 70.0 and not (c1 < bb_mid - atr * 0.50):
            return -1, "SHRSell"
    return 0, ""


# ==========================================================================
# OFF-HOURS ENGINES
# ==========================================================================
def signal_offhours_squeeze(bv: BarView, ind: dict[str, float], cfg: dict[str, Any], hour: int) -> int:
    """
    Off-hours engine 5 (DEFAULT): NR7 SQUEEZE -> EXPANSION breakout (Crabel).
    Ported from CheckOffSqzBuy/Sell L1139/L1154 + OffSqzIsNR L1128:
    bar[2] range เป็น strict-narrowest ของ K=7 แท่ง (squeeze) -> bar[1] break ขอบ squeeze
    + impulse range>=ExpMult*ATR. window hr 2-8, SL 1.0 TP 3.0.
    """
    atr = float(ind.get("atr", 0.0))
    if atr <= 0:
        return 0
    if not (cfg["off_sqz_start"] <= hour < cfg["off_sqz_end"]):
        return 0
    K = int(cfg["off_sqz_lookback"])
    # NR-K: range ของ bar[2] ต้องแคบสุดในช่วง [2 .. 2+K-1]
    r0 = bv.high(2) - bv.low(2)
    if r0 <= 0:
        return 0
    for i in range(3, 2 + K):
        if r0 >= (bv.high(i) - bv.low(i)):
            return 0  # ไม่แคบสุดจริง
    exp = float(cfg["off_sqz_exp_mult"])
    o1, c1, h1, l1 = bv.open(1), bv.close(1), bv.high(1), bv.low(1)
    sqz_high = bv.high(2)
    sqz_low = bv.low(2)
    if c1 > sqz_high and c1 > o1 and (h1 - l1) >= exp * atr:
        return +1
    if c1 < sqz_low and c1 < o1 and (h1 - l1) >= exp * atr:
        return -1
    return 0


def signal_offhours_sweep(bv: BarView, ind: dict[str, float], cfg: dict[str, Any], hour: int) -> int:
    """
    Off-hours engine 1: Asia Liquidity-Sweep Reversal (MR fade).
    Ported from CheckOffHoursSweepBuy/Sell L881/L903:
    sweep ทะลุขอบ rolling box (lookback 12) แล้ว close กลับเข้า box, Stoch confirm, CHOP>=53 (ranging).
    """
    chop = ind.get("chop", float("nan"))
    if not (cfg["off_sweep_start"] <= hour < cfg["off_sweep_end"]):
        return 0
    if not isinstance(chop, float) or math.isnan(chop) or chop < float(cfg["off_sweep_chop_min"]):
        return 0  # fail-CLOSED
    look = int(cfg["off_sweep_lookback"])
    stoch_k = float(ind.get("stoch_k", 50.0))
    box_low = bv.lowest(look, 2)
    box_high = bv.highest(look, 2)
    l1, h1, c1 = bv.low(1), bv.high(1), bv.close(1)
    if box_low > 0 and l1 < box_low and c1 > box_low and stoch_k < float(cfg["off_sweep_stoch_os"]):
        return +1
    if box_high > 0 and h1 > box_high and c1 < box_high and stoch_k > float(cfg["off_sweep_stoch_ob"]):
        return -1
    return 0


def signal_offhours_vwap(bv: BarView, ind: dict[str, float], cfg: dict[str, Any], hour: int) -> int:
    """
    Off-hours engine 2: Asia VWAP sigma-band Reversion (MR fade).
    Ported from CheckOffHoursVWAPBuy/Sell L948/L964 + OffVwapGateOK L935:
    fade pierce ของ VWAP +/-sigma -> center, CHOP>=53, band>=0.5*ATR.
    """
    chop = ind.get("chop", float("nan"))
    if not (cfg["off_vwap_start"] <= hour < cfg["off_vwap_end"]):
        return 0
    if not isinstance(chop, float) or math.isnan(chop) or chop < float(cfg["off_vwap_chop_min"]):
        return 0
    vwap = float(ind.get("vwap", 0.0))
    vwap_up = float(ind.get("vwap_upper", 0.0))
    vwap_lo = float(ind.get("vwap_lower", 0.0))
    atr = float(ind.get("atr", 0.0))
    if vwap <= 0 or vwap_up <= 0:
        return 0
    if (vwap_up - vwap_lo) < float(cfg["off_vwap_band_min_atr"]) * atr:
        return 0
    o1, c1, h1, l1 = bv.open(1), bv.close(1), bv.high(1), bv.low(1)
    if l1 <= vwap_lo and c1 < vwap and c1 > o1 and c1 > l1 + 0.5 * (h1 - l1):
        return +1
    if h1 >= vwap_up and c1 > vwap and c1 < o1 and c1 < h1 - 0.5 * (h1 - l1):
        return -1
    return 0


def signal_offhours_momentum(bv: BarView, ind: dict[str, float], cfg: dict[str, Any], hour: int) -> int:
    """
    Off-hours engine 3: late-NY MOMENTUM (trend-continuation).
    Ported from CheckOffHoursMomBuy/Sell L1006/L1024 + OffMomGateOK L991:
    N-bar(8) breakout + impulse (range>=0.6*ATR), CHOP<38 (trending), window hr 20-23.
    """
    atr = float(ind.get("atr", 0.0))
    chop = ind.get("chop", float("nan"))
    if atr <= 0:
        return 0
    if not (cfg["off_mom_start"] <= hour < cfg["off_mom_end"]):
        return 0
    if isinstance(chop, float) and not math.isnan(chop) and chop > float(cfg["off_mom_chop_max"]):
        return 0  # ranging -> block (fail-open ถ้า chop nan)
    look = int(cfg["off_mom_lookback"])
    exp = float(cfg["off_mom_exp_mult"])
    o1, c1, h1, l1 = bv.open(1), bv.close(1), bv.high(1), bv.low(1)
    if c1 > bv.highest(look, 2) and c1 > o1 and (h1 - l1) >= exp * atr:
        return +1
    if c1 < bv.lowest(look, 2) and c1 < o1 and (h1 - l1) >= exp * atr:
        return -1
    return 0


def signal_offhours_indicator(bv: BarView, ind: dict[str, float], cfg: dict[str, Any], hour: int) -> int:
    """
    Off-hours engine 4: INDICATOR sweep (orthogonal families by mode).
    Ported from OffIndiDir L1060 (reuse late-NY window hr 20-23):
    mode 1=ADX/DI trend, 2=CCI momentum cross, 3=RSI(2) Connors MR, 4=EMA fast/slow cross.
    """
    atr = float(ind.get("atr", 0.0))
    if atr <= 0 or not (cfg["off_mom_start"] <= hour < cfg["off_mom_end"]):
        return 0
    mode = int(cfg["off_indi_mode"])
    o1, c1 = bv.open(1), bv.close(1)
    if mode == 1:
        adx = float(ind.get("adx", 0.0))
        dip = float(ind.get("adx_plus", 0.0))
        dim = float(ind.get("adx_minus", 0.0))
        if adx < float(cfg["off_indi_adx_min"]):
            return 0
        if dip > dim and c1 > o1:
            return +1
        if dim > dip and c1 < o1:
            return -1
    elif mode == 2:
        cci1 = float(ind.get("cci", 0.0))
        cci2 = float(ind.get("cci_prev", 0.0))
        thr = float(cfg["off_indi_cci_thr"])
        if cci1 > thr and cci2 <= thr:
            return +1
        if cci1 < -thr and cci2 >= -thr:
            return -1
    elif mode == 3:
        rsi2 = float(ind.get("mon_rsi2", 50.0))
        if rsi2 < float(cfg["off_indi_rsi2_os"]):
            return +1
        if rsi2 > float(cfg["off_indi_rsi2_ob"]):
            return -1
    elif mode == 4:
        ema_f = float(ind.get("ema_fast", 0.0))
        ema_s = float(ind.get("ema_slow", 0.0))
        if ema_f > ema_s and c1 > ema_f:
            return +1
        if ema_f < ema_s and c1 < ema_f:
            return -1
    return 0


# ==========================================================================
# Router — get_day_engine: route weekday -> engine + skip-day check
# ==========================================================================
def get_day_engine(weekday: int, cfg: dict[str, Any]) -> Optional[str]:
    """
    weekday: 0=Sun .. 6=Sat (MT5 day_of_week).
    คืนชื่อ engine ของ in-session ('mon'/'tue'/'wed'/'thu'/'fri') หรือ None ถ้า skip/ปิด.
    NB: v46 defaults skip_sunday/tuesday/wednesday=true => in-session active = Mon/Thu/Fri.
    """
    if weekday in (0, 6):
        return None
    if weekday == 0 and bool(cfg.get("skip_sunday", True)):
        return None
    if not bool(cfg.get("enable_per_day", True)):
        return None
    if weekday == 1:
        return "mon" if bool(cfg.get("enable_mon_engine", True)) else None
    if weekday == 2:
        if bool(cfg.get("skip_tuesday", True)):
            return None
        return "tue" if bool(cfg.get("enable_tue_engine", True)) else None
    if weekday == 3:
        if bool(cfg.get("skip_wednesday", True)):
            return None
        return "wed" if bool(cfg.get("enable_wed_engine", True)) else None
    if weekday == 4:
        return "thu" if bool(cfg.get("enable_thu_engine", True)) else None
    if weekday == 5:
        return "fri" if bool(cfg.get("friday_uses_legacy", True)) else None
    return None


def _day_sltp(engine: str, cfg: dict[str, Any]) -> tuple[float, float, bool, bool, bool]:
    """คืน (sl_mult, tp_mult, bb_exit, vwap_exit, donch_mid_exit) ตาม engine ของวันนั้น."""
    if engine == "mon":
        return float(cfg["mon_sl_mult"]), float(cfg["mon_tp_mult"]), True, False, False
    if engine == "tue":
        return float(cfg["tue_sl_mult"]), float(cfg["tue_tp_mult"]), False, True, False
    if engine == "wed":
        return float(cfg["wed_sl_mult"]), float(cfg["wed_tp_mult"]), False, False, False
    if engine == "thu":
        if bool(cfg.get("thu_use_momentum", True)):
            return float(cfg["thu_mom_sl_mult"]), float(cfg["thu_mom_tp_mult"]), False, False, False
        return float(cfg["thu_sl_mult"]), float(cfg["thu_tp_mult"]), False, False, True
    if engine == "fri":
        return float(cfg["atr_mult_sl"]), float(cfg["atr_mult_tp"]), True, False, False
    return float(cfg["atr_mult_sl"]), float(cfg["atr_mult_tp"]), False, False, False


def _offhours_sltp(off_engine: int, cfg: dict[str, Any]) -> tuple[float, float, bool, bool]:
    """คืน (sl_mult, tp_mult, bb_exit, vwap_exit) ของ off-hours engine ที่เลือก."""
    if off_engine == 1:
        return float(cfg["off_sweep_sl_mult"]), float(cfg["off_sweep_tp_mult"]), False, False
    if off_engine == 2:
        return float(cfg["off_vwap_sl_mult"]), float(cfg["off_vwap_tp_mult"]), False, True
    if off_engine == 3:
        return float(cfg["off_mom_sl_mult"]), float(cfg["off_mom_tp_mult"]), False, False
    if off_engine == 4:
        return float(cfg["off_indi_sl_mult"]), float(cfg["off_indi_tp_mult"]), False, False
    # 5 = squeeze (default)
    return float(cfg["off_sqz_sl_mult"]), float(cfg["off_sqz_tp_mult"]), False, False


# ==========================================================================
# Indicator precompute — เตรียม array ทั้งหมดทีเดียวสำหรับ backtest
# ==========================================================================
def _precompute_indicators(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    high, low, close, open_ = df["high"], df["low"], df["close"], df["open"]
    out: dict[str, np.ndarray] = {}

    bb_mid, bb_up, bb_lo = calculate_bollinger(close, cfg["bb_period"], cfg["bb_deviation"])
    out["bb_middle"], out["bb_upper"], out["bb_lower"] = bb_mid.values, bb_up.values, bb_lo.values

    sk, sd = calculate_stochastic(high, low, close, cfg["stoch_k"], cfg["stoch_d"], cfg["stoch_slowing"])
    out["stoch_k"], out["stoch_d"] = sk.values, sd.values

    atr = calculate_atr(high, low, close, cfg["atr_period"])
    out["atr"] = atr.values
    out["atr_prev"] = atr.shift(1).values

    if cfg.get("enable_rsi", True):
        out["rsi"] = calculate_rsi(close, cfg["rsi_period"]).values
    else:
        out["rsi"] = np.full(len(df), 50.0)

    # EMAs (Tue stack + indicator engine)
    out["ema_fast"] = calculate_ema(close, cfg["tue_ema_fast"]).values
    out["ema_mid"] = calculate_ema(close, cfg["tue_ema_mid"]).values
    out["ema_slow"] = calculate_ema(close, cfg["tue_ema_slow"]).values

    # ADX (current TF) + prev
    adx, dip, dim = calculate_adx(high, low, close, cfg["tue_adx_period"])
    out["adx"], out["adx_plus"], out["adx_minus"] = adx.values, dip.values, dim.values
    out["adx_prev"] = adx.shift(1).values

    # ATR fast/slow (Thu/Wed)
    out["atr_slow"] = calculate_atr(high, low, close, cfg["wed_atr_slow_period"]).values

    # Monday Keltner pieces
    out["mon_ema"] = calculate_ema(close, cfg["mon_kelt_ema"]).values
    out["mon_atr"] = calculate_atr(high, low, close, cfg["mon_kelt_atr"]).values
    out["mon_atr50"] = calculate_atr(high, low, close, cfg["mon_vol_slow_atr"]).values
    out["mon_rsi2"] = calculate_rsi(close, cfg["mon_rsi2_period"]).values

    # CCI (+ prev) for off-hours indicator engine
    cci = calculate_cci(high, low, close, 20)
    out["cci"], out["cci_prev"] = cci.values, cci.shift(1).values

    # Williams %R (Thu fade)
    out["thu_wpr"] = calculate_williams_r(high, low, close, cfg["thu_wpr_period"]).values

    # VWAP (Tue + off-hours vwap) — current-TF approximation of session VWAP
    vwap, vwu, vwl = calculate_vwap(df, cfg["tue_vwap_mult"])
    out["vwap"], out["vwap_upper"], out["vwap_lower"] = vwap.values, vwu.values, vwl.values

    # Choppiness — ใช้ TF ปัจจุบันเป็น proxy ของ H1 (ถ้าไม่มี H1 data แยก)
    out["chop"] = calculate_choppiness(high, low, close, cfg["chop_period"]).values
    # H1 ADX proxy = ADX(14) บน TF ปัจจุบัน
    adx_h1, _, _ = calculate_adx(high, low, close, 14)
    out["adx_h1"] = adx_h1.values
    return out


# ==========================================================================
# Backtest engine — bar-by-bar, per-day routing + off-hours + killer MM
# ==========================================================================
def run_backtest(candles: list[dict], balance: float, tick_value: float, tick_size: float,
                 min_lot: float, max_lot: float, lot_step: float,
                 config: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    """
    Bar-by-bar backtest จำลอง v46:
      - per-day routing (Mon/Thu/Fri active; Tue/Wed/Sun skip ตาม default)
      - off-hours layer นอก session 08:00-20:00
      - confidence-mult + lucky + capital-scaling
      - exit: BB-mid / VWAP / Donchian-mid + ATR SL/TP + breakeven + trailing (simplified) + max bars
    อ่าน indicator จาก bar[i-1] (closed bar) กัน repaint.
    คืน dict สถิติ: net_profit, trades, win_rate, max_drawdown, profit_factor, ...
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if len(candles) < 60:
        return None

    df = pd.DataFrame(candles)
    df["time"] = pd.to_numeric(df["time"])
    df = df.sort_values(by="time").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col])
    # spec candle ใช้คีย์ tick_volume -> map เป็น volume (fallback 1.0)
    if "volume" not in df.columns:
        df["volume"] = pd.to_numeric(df["tick_volume"]) if "tick_volume" in df.columns else 1.0

    ind = _precompute_indicators(df, cfg)
    n = len(df)

    o_arr = df["open"].values
    h_arr = df["high"].values
    l_arr = df["low"].values
    c_arr = df["close"].values
    v_arr = df["volume"].values
    t_arr = df["time"].values.astype("float64")

    sess_start = cfg["session_start_hour"] * 60 + cfg["session_start_min"]
    sess_end = cfg["session_end_hour"] * 60 + cfg["session_end_min"]
    shr_look = int(cfg["shr_lookback_bars"])

    sim_balance = balance
    peak_balance = balance
    max_dd = 0.0
    trades = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0

    consec_losses = 0
    cooldown = 0
    off_cooldown = 0
    cycle_start = float(t_arr[0])
    lucky_mult = 1
    lucky_active = False

    # positions: list ของ dict (รองรับ multi-position ตาม max_open_trades / off_hours_max)
    positions: list[dict[str, Any]] = []
    last_bar_handled = -1

    eff_max_open = int(cfg["max_open_trades"])
    eff_off_max = int(cfg["off_hours_max_trades"])

    def _dt(epoch: float) -> datetime:
        return datetime.fromtimestamp(epoch, tz=timezone.utc)

    def _ind_at(idx: int) -> dict[str, float]:
        d: dict[str, float] = {}
        for k, arr in ind.items():
            val = arr[idx]
            d[k] = float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else (
                float("nan") if k == "chop" else 0.0)
        return d

    def _close_position(pos: dict[str, Any], exit_price: float, idx: int) -> None:
        nonlocal sim_balance, peak_balance, max_dd, trades, wins
        nonlocal gross_profit, gross_loss, consec_losses, cooldown, off_cooldown
        nonlocal lucky_mult, lucky_active
        if pos["dir"] == +1:
            profit_pts = exit_price - pos["entry"]
        else:
            profit_pts = pos["entry"] - exit_price
        profit_money = (profit_pts / tick_size) * tick_value * pos["lot"]
        sim_balance += profit_money
        peak_balance = max(peak_balance, sim_balance)
        if peak_balance > 0:
            dd = (peak_balance - sim_balance) / peak_balance * 100.0
            if dd > max_dd:
                # noqa
                pass
        # recompute max_dd safely
        _update_dd()
        trades += 1
        if profit_money > 0:
            wins += 1
            gross_profit += profit_money
            consec_losses = 0
            if pos.get("off"):
                pass
            elif lucky_active and lucky_mult < int(cfg["lucky_max_mult"]):
                lucky_mult += 1
        else:
            gross_loss += abs(profit_money)
            if pos.get("off"):
                off_cooldown = int(cfg["off_hours_cooldown_bars"])
            else:
                consec_losses += 1
                cooldown = int(cfg["cooldown_bars_after_loss"])
                if lucky_active and lucky_mult > 2:
                    lucky_mult = 2

    def _update_dd() -> None:
        nonlocal max_dd
        if peak_balance > 0:
            dd = (peak_balance - sim_balance) / peak_balance * 100.0
            if dd > max_dd:
                max_dd = dd

    # main loop: i = forming bar; ใช้ closed bar = i-1 (อ่านผ่าน BarView shift)
    for i in range(40, n):
        idx1 = i - 1                      # closed bar index (bar[1])
        bv = BarView(o_arr, h_arr, l_arr, c_arr, v_arr, cur=i)
        ic = _ind_at(idx1)
        atr = ic["atr"]
        if atr <= 0 or math.isnan(atr):
            continue

        dt = _dt(float(t_arr[i]))
        hour = dt.hour
        minute_of_day = dt.hour * 60 + dt.minute
        py_wd = dt.weekday()              # Mon=0..Sun=6
        mt5_wd = (py_wd + 1) % 7          # MT5: Sun=0..Sat=6
        in_session = (sess_start <= minute_of_day < sess_end)
        is_new_bar = (i != last_bar_handled)
        last_bar_handled = i

        # SHR recent high/low (จาก bar[2] ย้อน shr_look)
        ic["recent_high"] = bv.highest(shr_look, 2)
        ic["recent_low"] = bv.lowest(shr_look, 2)

        # capital scaling (dynamic จากทุนปัจจุบัน + ATR สด)
        eff_max_open, eff_off_max = apply_capital_scaling(
            sim_balance, atr, tick_value, tick_size, min_lot, cfg["atr_mult_sl"],
            cfg["max_account_risk_pct"], cfg["max_open_trades"], cfg["off_hours_max_trades"],
            bool(cfg["auto_scale_by_capital"]))

        # lucky cycle (in-session pool เท่านั้น)
        new_lucky = lucky_multiplier(float(t_arr[i]), cycle_start, cfg) > 1
        if new_lucky and not lucky_active:
            lucky_active = True
            lucky_mult = 2
        elif not new_lucky and lucky_active:
            lucky_active = False
            lucky_mult = 1

        # ---------- MANAGE OPEN POSITIONS (intrabar SL/TP + target exits) ----------
        bid = bv.high(0) if False else c_arr[i]  # ใช้ราคา intrabar ของ bar i
        bb_up, bb_mid, bb_lo = ic["bb_upper"], ic["bb_middle"], ic["bb_lower"]
        bb_exit_ratio = float(cfg["bb_exit_ratio"])
        bb_exit_buy = bb_lo + bb_exit_ratio * (bb_up - bb_lo)
        bb_exit_sell = bb_up - bb_exit_ratio * (bb_up - bb_lo)
        vwap = ic["vwap"]

        still_open: list[dict[str, Any]] = []
        for pos in positions:
            exited = False
            hi, lo = h_arr[i], l_arr[i]
            # 1) SL/TP intrabar (SL ก่อน — conservative)
            if pos["dir"] == +1:
                if lo <= pos["sl"]:
                    _close_position(pos, pos["sl"], i); exited = True
                elif hi >= pos["tp"]:
                    _close_position(pos, pos["tp"], i); exited = True
            else:
                if hi >= pos["sl"]:
                    _close_position(pos, pos["sl"], i); exited = True
                elif lo <= pos["tp"]:
                    _close_position(pos, pos["tp"], i); exited = True
            if exited:
                continue
            # 2) time exit
            if (i - pos["entry_bar"]) > int(cfg["max_bars_open"]):
                _close_position(pos, c_arr[i], i); continue
            # 3) target exits (BB-mid / VWAP / Donchian-mid) เมื่อกำไร
            in_profit = ((c_arr[i] - pos["entry"]) if pos["dir"] == +1 else (pos["entry"] - c_arr[i])) > 0
            if in_profit:
                if pos.get("bb_exit") and pos["dir"] == +1 and c_arr[i] >= bb_exit_buy:
                    _close_position(pos, c_arr[i], i); continue
                if pos.get("bb_exit") and pos["dir"] == -1 and c_arr[i] <= bb_exit_sell:
                    _close_position(pos, c_arr[i], i); continue
                if pos.get("vwap_exit") and vwap > 0:
                    if pos["dir"] == +1 and c_arr[i] >= vwap:
                        _close_position(pos, c_arr[i], i); continue
                    if pos["dir"] == -1 and c_arr[i] <= vwap:
                        _close_position(pos, c_arr[i], i); continue
                if pos.get("donch_mid_exit") and pos.get("donch_mid", 0.0) > 0:
                    dm = pos["donch_mid"]
                    if pos["dir"] == +1 and c_arr[i] >= dm:
                        _close_position(pos, c_arr[i], i); continue
                    if pos["dir"] == -1 and c_arr[i] <= dm:
                        _close_position(pos, c_arr[i], i); continue
            still_open.append(pos)
        positions = still_open

        # cooldown decrement (new bar)
        if is_new_bar and cooldown > 0:
            cooldown -= 1
        if is_new_bar and off_cooldown > 0:
            off_cooldown -= 1

        # ---------- daily DD guard (ใช้ balance start เป็น ref โดยประมาณ) ----------
        # (simplified: ใช้ peak เป็น ref; consec-loss guard ตาม MQL5)
        if consec_losses >= int(cfg["consecutive_loss_limit"]):
            continue

        # ===================== ENTRY =====================
        if not in_session:
            # ---------- OFF-HOURS LAYER ----------
            if not bool(cfg["enable_off_hours"]):
                continue
            if mt5_wd == 5 and hour >= 20:    # Friday close
                continue
            if off_cooldown > 0 or not is_new_bar:
                continue
            n_off = sum(1 for p in positions if p.get("off"))
            if n_off >= eff_off_max:
                continue
            off_eng = int(cfg["off_hours_engine"])
            if off_eng == 1:
                sig = signal_offhours_sweep(bv, ic, cfg, hour)
            elif off_eng == 2:
                sig = signal_offhours_vwap(bv, ic, cfg, hour)
            elif off_eng == 3:
                sig = signal_offhours_momentum(bv, ic, cfg, hour)
            elif off_eng == 4:
                sig = signal_offhours_indicator(bv, ic, cfg, hour)
            else:
                sig = signal_offhours_squeeze(bv, ic, cfg, hour)
            if sig == 0:
                continue
            sl_m, tp_m, bb_ex, vw_ex = _offhours_sltp(off_eng, cfg)
            _open_trade(positions, sig, bv, ic, cfg, sim_balance, tick_value, tick_size,
                        min_lot, max_lot, lot_step, sl_m, tp_m, bb_ex, vw_ex, False,
                        is_off=True, conf=1, lucky=1, day_boost=1.0)
            continue

        # ---------- IN-SESSION ----------
        if mt5_wd == 5 and hour >= 20:        # Friday close
            for pos in list(positions):
                _close_position(pos, c_arr[i], i)
            positions = []
            continue
        if cooldown > 0:
            continue
        if len(positions) >= eff_max_open:
            continue
        if bool(cfg["one_trade_per_bar"]) and not is_new_bar:
            continue

        engine = get_day_engine(mt5_wd, cfg)
        if engine is None:
            continue

        sig = 0
        tag = ""
        if engine == "mon":
            sig = signal_mon_keltner(bv, ic, cfg, hour)
            tag = "MonKeltBuy" if sig > 0 else ("MonKeltSell" if sig < 0 else "")
        elif engine == "tue":
            sig = signal_tue_vwap(bv, ic, cfg, hour)
            tag = "TueVWAPBuy" if sig > 0 else ("TueVWAPSell" if sig < 0 else "")
        elif engine == "wed":
            sig = signal_wed_wvec(bv, ic, cfg, hour)
            tag = "WedVECBuy" if sig > 0 else ("WedVECSell" if sig < 0 else "")
        elif engine == "thu":
            if bool(cfg["thu_use_momentum"]):
                sig = signal_thu_momentum(bv, ic, cfg, hour)
                tag = "ThuMomoBuy" if sig > 0 else ("ThuMomoSell" if sig < 0 else "")
            else:
                sig = signal_thu_fade(bv, ic, cfg, hour)
                tag = "ThuFadeBuy" if sig > 0 else ("ThuFadeSell" if sig < 0 else "")
        elif engine == "fri":
            if not regime_allows_mr(ic.get("chop", float("nan")), cfg):
                continue
            sig, tag = signal_fri_legacy_mr(bv, ic, cfg)

        if sig == 0:
            continue

        sl_m, tp_m, bb_ex, vw_ex, dm_ex = _day_sltp(engine, cfg)
        # confidence (in-session เท่านั้น)
        conf = compute_confidence(bv, ic, cfg, tag)
        day_boost = 1.0
        if bool(cfg["enable_day_boost"]) and conf >= int(cfg["day_boost_min_conf"]):
            if mt5_wd == 1:
                day_boost = float(cfg["mon_boost_mult"])
            elif mt5_wd == 5:
                day_boost = float(cfg["fri_boost_mult"])
        lk = lucky_mult if (lucky_active and not bool(cfg["enable_confidence_mult"])) else 1

        _open_trade(positions, sig, bv, ic, cfg, sim_balance, tick_value, tick_size,
                    min_lot, max_lot, lot_step, sl_m, tp_m, bb_ex, vw_ex, dm_ex,
                    is_off=False, conf=conf, lucky=lk, day_boost=day_boost)

    # close residual positions at last close
    for pos in list(positions):
        _close_position(pos, c_arr[-1], n - 1)

    net_profit = sim_balance - balance
    win_rate = (wins / trades * 100.0) if trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0)

    return {
        "net_profit": round(net_profit, 2),
        "trades": trades,
        "win_rate": round(win_rate, 2),
        "max_drawdown": round(max_dd, 2),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else None,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "final_balance": round(sim_balance, 2),
    }


def _open_trade(positions: list[dict[str, Any]], direction: int, bv: BarView, ind: dict[str, float],
                cfg: dict[str, Any], balance: float, tick_value: float, tick_size: float,
                min_lot: float, max_lot: float, lot_step: float,
                sl_mult: float, tp_mult: float, bb_exit: bool, vwap_exit: bool,
                donch_mid_exit: bool, is_off: bool, conf: int, lucky: int, day_boost: float) -> None:
    """
    เปิดไม้: คำนวณ SL/TP จาก ATR*mult (+ hard cap), lot จาก risk%/conf/lucky/day-boost.
    Ported from OpenBuy/OpenSell L2382/L2397 + GetSLPoints/GetTPPoints L2363.
    is_off=True ใช้ off_hours_risk_percent และไม่เข้า conf/lucky/day-boost (เหมือน riskPct>0 ใน MQL5).
    """
    atr = float(ind.get("atr", 0.0))
    point = tick_size  # บน XAUUSD ปกติ point == tick_size
    sl_pts = (atr * sl_mult) / point
    tp_pts = (atr * tp_mult) / point
    if cfg["max_sl_pts"] > 0:
        sl_pts = min(sl_pts, float(cfg["max_sl_pts"]))
    if cfg["max_tp_pts"] > 0:
        tp_pts = min(tp_pts, float(cfg["max_tp_pts"]))
    sl_dist = sl_pts * point
    tp_dist = tp_pts * point
    if sl_dist <= 0:
        return

    if is_off:
        risk_pct = float(cfg["off_hours_risk_percent"])
        lot = calculate_lot_size(balance, risk_pct, sl_dist, tick_value, tick_size,
                                 min_lot, max_lot, lot_step, conf_mult=1, lucky_mult=1, day_boost=1.0)
    else:
        risk_pct = float(cfg["risk_percent"])
        lot = calculate_lot_size(balance, risk_pct, sl_dist, tick_value, tick_size,
                                 min_lot, max_lot, lot_step,
                                 conf_mult=conf if bool(cfg["enable_confidence_mult"]) else 1,
                                 lucky_mult=lucky, day_boost=day_boost)
    if lot <= 0:
        return

    entry = bv.close(1)  # entry ที่ราคาปิดของ closed bar (proxy ของ ask/bid ตอนเปิด)
    if direction == +1:
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist

    donch_mid = 0.0
    if donch_mid_exit:
        don = int(cfg["thu_donchian"])
        donch_mid = (bv.lowest(don, 2) + bv.highest(don, 2)) * 0.5

    positions.append({
        "dir": direction, "entry": entry, "sl": sl, "tp": tp, "lot": lot,
        "entry_bar": bv.cur, "off": is_off,
        "bb_exit": bb_exit, "vwap_exit": vwap_exit, "donch_mid_exit": donch_mid_exit,
        "donch_mid": donch_mid,
    })


# ==========================================================================
# Backtest optimization (spec bt_res) — grid search ATR SL/TP multipliers
# ==========================================================================
def run_backtest_optimization(candles: list[dict], balance: float, tick_value: float,
                              tick_size: float, min_lot: float, max_lot: float,
                              lot_step: float, config: dict[str, Any]
                              ) -> Optional[dict[str, Any]]:
    """
    รัน backtest engine ที่มีอยู่ (run_backtest) แบบ grid-search SL/TP base multiplier
    เพื่อหา combo ที่กำไรสูงสุด แล้วคืน dict ตาม spec bt_res:
      direction, tp_multiplier, sl_multiplier, win_rate, max_drawdown, trades, profit
    + key พิเศษ atr_mult_sl / atr_mult_tp สำหรับ map กลับ updated_config (UI inputs).
    ใช้ทิศ net direction ของ best run เป็น "direction" (BUY ถ้า net>0, ไม่งั้น NONE).
    """
    base_cfg = {**DEFAULT_CONFIG, **(config or {})}
    sl_list = [1.0, 1.2, 1.5, 1.8, 2.0]
    tp_list = [1.5, 2.0, 2.5, 3.0, 3.5]

    best_profit = float("-inf")
    best: Optional[dict[str, Any]] = None
    best_sl = float(base_cfg["atr_mult_sl"])
    best_tp = float(base_cfg["atr_mult_tp"])

    for sl_m in sl_list:
        for tp_m in tp_list:
            trial_cfg = {**base_cfg, "atr_mult_sl": sl_m, "atr_mult_tp": tp_m}
            stats = run_backtest(candles, balance, tick_value, tick_size,
                                 min_lot, max_lot, lot_step, trial_cfg)
            if stats is None or stats.get("trades", 0) < 1:
                continue
            net = float(stats.get("net_profit", 0.0))
            if net > best_profit:
                best_profit = net
                best_sl, best_tp = sl_m, tp_m
                best = stats

    if best is None:
        # ไม่มี combo ที่ยิงเทรดเลย — คืนผลรันเดิม (อาจว่าง)
        fallback = run_backtest(candles, balance, tick_value, tick_size,
                                min_lot, max_lot, lot_step, base_cfg) or {}
        return {
            "direction": "NONE",
            "tp_multiplier": float(best_tp),
            "sl_multiplier": float(best_sl),
            "win_rate": float(fallback.get("win_rate", 0.0)),
            "max_drawdown": float(fallback.get("max_drawdown", 0.0)),
            "trades": int(fallback.get("trades", 0)),
            "profit": float(fallback.get("net_profit", 0.0)),
            "atr_mult_sl": float(best_sl),
            "atr_mult_tp": float(best_tp),
        }

    direction = "BUY" if best_profit > 0 else ("SELL" if best_profit < 0 else "NONE")
    return {
        "direction": direction,
        "tp_multiplier": float(best_tp),
        "sl_multiplier": float(best_sl),
        "win_rate": float(best.get("win_rate", 0.0)),
        "max_drawdown": float(best.get("max_drawdown", 0.0)),
        "trades": int(best.get("trades", 0)),
        "profit": float(round(best_profit, 2)),
        # optimized params -> map กลับ UI config inputs
        "atr_mult_sl": float(best_sl),
        "atr_mult_tp": float(best_tp),
    }


# ==========================================================================
# Live strategy — process_strategy: คืน signal dict สำหรับ bar ปัจจุบัน
# ==========================================================================
def process_strategy(data: dict[str, Any], config: dict[str, Any],
                     add_log_fn: Callable[[str], None]
                     ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Optional[dict[str, Any]]]:
    """
    Live entrypoint (เหมือน simple-2-3.py).
    คืน (res_dict, updated_config, live_metrics, bt_res).
    res_dict.action = NONE/BUY/SELL/CLOSE พร้อม lot/sl_mult/tp_mult/reason.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    candles = data.get("candles", [])
    positions = data.get("positions", [])
    symbol = data.get("symbol", "XAUUSD")
    timeframe = data.get("timeframe", "M5")
    is_new_bar = bool(data.get("is_new_bar", False))

    balance = float(data.get("balance", 10000.0))
    tick_value = float(data.get("tick_value", 1.0))
    tick_size = float(data.get("tick_size", 0.01))
    min_lot = float(data.get("min_lot", 0.01))
    max_lot = float(data.get("max_lot", 100.0))
    lot_step = float(data.get("lot_step", 0.01))
    spread = float(data.get("spread", 0.0))          # spec field: current spread in points
    equity = float(data.get("equity", balance))
    trigger_backtest = bool(data.get("trigger_backtest", False))

    updated_config = dict(cfg)                       # copy ที่อาจอัปเดตจาก backtest opt

    min_bars = 60
    if len(candles) < min_bars:
        res_dict = {
            "action": "NONE",
            "display_line1": "Initializing...",
            "display_line2": f"Bars: {len(candles)}/{min_bars}",
        }
        return res_dict, updated_config, {"Status": "Syncing bars"}, None

    # ---- backtest optimization (spec bt_res) — run only เมื่อ user trigger ----
    bt_res: Optional[dict[str, Any]] = None
    if trigger_backtest:
        add_log_fn("Running v46 backtest optimization (ATR SL/TP grid)...")
        bt_res = run_backtest_optimization(candles, balance, tick_value, tick_size,
                                           min_lot, max_lot, lot_step, cfg)
        if bt_res:
            updated_config["atr_mult_sl"] = float(bt_res.get("atr_mult_sl",
                                                             cfg["atr_mult_sl"]))
            updated_config["atr_mult_tp"] = float(bt_res.get("atr_mult_tp",
                                                             cfg["atr_mult_tp"]))
            cfg = updated_config
            add_log_fn(f"Optimization done: dir={bt_res['direction']} "
                       f"SLx{bt_res['sl_multiplier']} TPx{bt_res['tp_multiplier']} "
                       f"profit={bt_res['profit']} WR={bt_res['win_rate']}% "
                       f"DD={bt_res['max_drawdown']}% trades={bt_res['trades']}")

    df = pd.DataFrame(candles)
    df["time"] = pd.to_numeric(df["time"])
    df = df.sort_values(by="time").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col])
    # spec candle ใช้คีย์ tick_volume -> map เป็น volume (fallback 1.0)
    if "volume" not in df.columns:
        df["volume"] = pd.to_numeric(df["tick_volume"]) if "tick_volume" in df.columns else 1.0

    ind_arrs = _precompute_indicators(df, cfg)
    n = len(df)
    cur = n - 1                       # forming bar 0
    idx1 = cur - 1                    # closed bar (bar[1])

    bv = BarView(df["open"].values, df["high"].values, df["low"].values,
                 df["close"].values, df["volume"].values, cur=cur)
    ic: dict[str, float] = {}
    for k, arr in ind_arrs.items():
        val = arr[idx1]
        ic[k] = float(val) if not (isinstance(val, float) and math.isnan(val)) else (
            float("nan") if k == "chop" else 0.0)
    shr_look = int(cfg["shr_lookback_bars"])
    ic["recent_high"] = bv.highest(shr_look, 2)
    ic["recent_low"] = bv.lowest(shr_look, 2)

    dt = datetime.fromtimestamp(float(df["time"].values[cur]), tz=timezone.utc)
    hour = dt.hour
    minute_of_day = dt.hour * 60 + dt.minute
    mt5_wd = (dt.weekday() + 1) % 7
    sess_start = cfg["session_start_hour"] * 60 + cfg["session_start_min"]
    sess_end = cfg["session_end_hour"] * 60 + cfg["session_end_min"]
    in_session = (sess_start <= minute_of_day < sess_end)
    atr = ic["atr"]

    # capital scaling สำหรับ display
    eff_max_open, eff_off_max = apply_capital_scaling(
        balance, atr, tick_value, tick_size, min_lot, cfg["atr_mult_sl"],
        cfg["max_account_risk_pct"], cfg["max_open_trades"], cfg["off_hours_max_trades"],
        bool(cfg["auto_scale_by_capital"]))

    # positions ของ symbol ปัจจุบันเท่านั้น (กรองตาม spec)
    sym_positions = [p for p in positions if p.get("symbol", symbol) == symbol]

    # decide engine + signal
    engine = None
    engine_label = "OFF"           # human-friendly สำหรับ HUD
    sig = 0
    tag = ""
    sl_m, tp_m = cfg["atr_mult_sl"], cfg["atr_mult_tp"]

    _day_label = {"mon": "Mon-Keltner", "tue": "Tue-VWAP", "wed": "Wed-WVEC",
                  "thu": "Thu-Momentum" if bool(cfg["thu_use_momentum"]) else "Thu-Fade",
                  "fri": "Fri-LegacyMR"}

    if in_session:
        engine = get_day_engine(mt5_wd, cfg)
        if engine == "mon":
            sig = signal_mon_keltner(bv, ic, cfg, hour); tag = "MonKelt"
            sl_m, tp_m = cfg["mon_sl_mult"], cfg["mon_tp_mult"]
        elif engine == "tue":
            sig = signal_tue_vwap(bv, ic, cfg, hour); tag = "TueVWAP"
            sl_m, tp_m = cfg["tue_sl_mult"], cfg["tue_tp_mult"]
        elif engine == "wed":
            sig = signal_wed_wvec(bv, ic, cfg, hour); tag = "WedVEC"
            sl_m, tp_m = cfg["wed_sl_mult"], cfg["wed_tp_mult"]
        elif engine == "thu":
            if bool(cfg["thu_use_momentum"]):
                sig = signal_thu_momentum(bv, ic, cfg, hour); tag = "ThuMomo"
                sl_m, tp_m = cfg["thu_mom_sl_mult"], cfg["thu_mom_tp_mult"]
            else:
                sig = signal_thu_fade(bv, ic, cfg, hour); tag = "ThuFade"
                sl_m, tp_m = cfg["thu_sl_mult"], cfg["thu_tp_mult"]
        elif engine == "fri":
            if regime_allows_mr(ic.get("chop", float("nan")), cfg):
                sig, fri_tag = signal_fri_legacy_mr(bv, ic, cfg)
                tag = fri_tag or "FriMR"
        engine_label = _day_label.get(engine or "", "Skip-Day")
    elif bool(cfg["enable_off_hours"]):
        engine = f"offhours-{cfg['off_hours_engine']}"
        oe = int(cfg["off_hours_engine"])
        if oe == 1:
            sig = signal_offhours_sweep(bv, ic, cfg, hour)
        elif oe == 2:
            sig = signal_offhours_vwap(bv, ic, cfg, hour)
        elif oe == 3:
            sig = signal_offhours_momentum(bv, ic, cfg, hour)
        elif oe == 4:
            sig = signal_offhours_indicator(bv, ic, cfg, hour)
        else:
            sig = signal_offhours_squeeze(bv, ic, cfg, hour)
        tag = "OffH"
        sl_m, tp_m, _be, _ve = _offhours_sltp(oe, cfg)
        _off_name = {1: "Sweep", 2: "VWAP", 3: "Momentum", 4: "Indicator", 5: "Squeeze"}
        engine_label = f"OffHours-{_off_name.get(oe, 'Squeeze')}"

    signal_text = "NONE"
    if sig > 0:
        signal_text = "BUY SIGNAL"
    elif sig < 0:
        signal_text = "SELL SIGNAL"

    # confidence (in-session เท่านั้น) — ใช้ทั้งใน HUD และ lot sizing
    conf = 1
    if sig != 0 and in_session:
        conf = compute_confidence(bv, ic, cfg, tag + ("Buy" if sig > 0 else "Sell"))

    chop_val = ic.get("chop", float("nan"))
    chop_str = "n/a" if (isinstance(chop_val, float) and math.isnan(chop_val)) else f"{chop_val:.1f}"

    if is_new_bar:
        add_log_fn(f"[{symbol} {timeframe}] engine={engine_label} sig={signal_text} "
                   f"ATR={atr:.2f} CHOP={chop_str} ADX={ic.get('adx', 0.0):.1f} "
                   f"conf=x{conf} effMax={eff_max_open}")

    # ค่า indicator/metadata ภายใน (เก็บไว้สำหรับ debugging — ไม่กระทบ spec contract)
    base_meta: dict[str, Any] = {
        "engine": engine_label,
        "signal_text": signal_text,
        "weekday_mt5": mt5_wd,
        "in_session": in_session,
        "atr": float(atr),
        "chop": None if (isinstance(chop_val, float) and math.isnan(chop_val)) else float(chop_val),
        "adx": float(ic.get("adx", 0.0)),
        "bb_upper": float(ic.get("bb_upper", 0.0)),
        "bb_middle": float(ic.get("bb_middle", 0.0)),
        "bb_lower": float(ic.get("bb_lower", 0.0)),
        "eff_max_open": eff_max_open,
        "eff_off_max": eff_off_max,
        "conf_mult": conf,
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    live_metrics = _live_metrics(ic, engine_label, conf, eff_max_open)

    # ---------- EXIT / GUARD LAYER (จัดการเฉพาะ position ของ symbol นี้) ----------
    # 1) Daily DD limit -> ปิดทุกไม้ (CLOSE_ALL)
    if sym_positions and balance > 0:
        dd_pct = (balance - equity) / balance * 100.0
        if dd_pct >= float(cfg["max_daily_drawdown"]):
            add_log_fn(f"Soft equity SL tripped: drawdown={dd_pct:.2f}% — CLOSE_ALL")
            res = {
                **base_meta,
                "action": "CLOSE_ALL",
                "reason": f"Daily DD limit {dd_pct:.1f}% >= {cfg['max_daily_drawdown']}%",
                "display_line1": f"Emergency Exit — DD {dd_pct:.1f}%",
                "display_line2": f"Closing all {len(sym_positions)} positions",
            }
            return res, updated_config, live_metrics, bt_res

    # 2) Friday session close -> ปิดทุกไม้ (CLOSE_ALL)
    if sym_positions and mt5_wd == 5 and hour >= cfg["session_end_hour"]:
        res = {
            **base_meta,
            "action": "CLOSE_ALL",
            "reason": "Friday close — flatten before weekend",
            "display_line1": "Friday Close — flatten all",
            "display_line2": f"Closing all {len(sym_positions)} positions",
        }
        return res, updated_config, live_metrics, bt_res

    # 3) max-bars-open time exit (ปิดไม้เดียวที่เก่าสุด) — เฉพาะตอน new bar
    if is_new_bar and sym_positions:
        latest_time = int(df["time"].values[cur])
        tf_sec = _timeframe_seconds(timeframe)
        for pos in sym_positions:
            open_age_bars = None
            if tf_sec > 0 and pos.get("time_open"):
                open_age_bars = (latest_time - int(pos["time_open"])) // tf_sec
            if open_age_bars is not None and open_age_bars > int(cfg["max_bars_open"]):
                tk = pos.get("ticket")
                add_log_fn(f"Time exit ticket #{tk} ({open_age_bars} bars open)")
                res = {
                    **base_meta,
                    "action": "CLOSE",
                    "ticket": tk,
                    "reason": f"Max bars open ({open_age_bars} > {cfg['max_bars_open']})",
                    "display_line1": "Time Exit Triggered",
                    "display_line2": f"Closing ticket #{tk} ({open_age_bars} bars)",
                }
                return res, updated_config, live_metrics, bt_res

    # ---------- SPREAD GATE (ไม่เปิดไม้ใหม่ถ้า spread กว้างเกิน) ----------
    max_spread = float(cfg["max_spread_off_hours"]) if not in_session else float(cfg["max_spread_points"])
    spread_ok = (spread <= 0) or (spread <= max_spread)

    # ---------- ENTRY DECISION ----------
    if is_new_bar and sig != 0 and spread_ok and len(sym_positions) < eff_max_open:
        is_off = not in_session
        if is_off:
            risk_pct = float(cfg["off_hours_risk_percent"])
            lot = calculate_lot_size(balance, risk_pct, atr * sl_m, tick_value, tick_size,
                                     min_lot, max_lot, lot_step)
            conf_disp = 1
        else:
            day_boost = 1.0
            if bool(cfg["enable_day_boost"]) and conf >= int(cfg["day_boost_min_conf"]):
                if mt5_wd == 1:
                    day_boost = float(cfg["mon_boost_mult"])
                elif mt5_wd == 5:
                    day_boost = float(cfg["fri_boost_mult"])
            lot = calculate_lot_size(balance, float(cfg["risk_percent"]), atr * sl_m,
                                     tick_value, tick_size, min_lot, max_lot, lot_step,
                                     conf_mult=conf if bool(cfg["enable_confidence_mult"]) else 1,
                                     day_boost=day_boost)
            conf_disp = conf
        action = "BUY" if sig > 0 else "SELL"
        add_log_fn(f"{action} signal [{engine_label}] lot={lot:.2f} confx{conf_disp} "
                   f"SLx{sl_m} TPx{tp_m}")
        res = {
            **base_meta,
            "action": action,
            "lot": round(lot, 2),
            "sl_multiplier": float(sl_m),
            "tp_multiplier": float(tp_m),
            "tag": tag,
            "confidence": conf_disp,
            "reason": f"{engine_label} {signal_text} (conf x{conf_disp})",
            "display_line1": f"Signal: {action} [{engine_label}]",
            "display_line2": f"lot {lot:.2f} | ATR SLx{sl_m} TPx{tp_m} | conf x{conf_disp}",
        }
        return res, updated_config, live_metrics, bt_res

    # ---------- DEFAULT: NONE ----------
    if not spread_ok:
        line2 = f"Spread {spread:.0f} > max {max_spread:.0f} pts — no entry"
    else:
        line2 = (f"{signal_text} | CHOP {chop_str} | ADX {ic.get('adx', 0.0):.1f} "
                 f"| conf x{conf}")
    res = {
        **base_meta,
        "action": "NONE",
        "display_line1": f"{engine_label} | {'IN-SESSION' if in_session else 'OFF-HOURS'} "
                         f"| ATR {atr:.2f}",
        "display_line2": line2,
    }
    return res, updated_config, live_metrics, bt_res


def _timeframe_seconds(tf: str) -> int:
    """แปลง timeframe string -> วินาทีต่อแท่ง (สำหรับคำนวณ bars-open)."""
    table = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800,
             "H1": 3600, "H4": 14400, "D1": 86400}
    return table.get(str(tf).upper(), 300)


def _live_metrics(ic: dict[str, float], engine_label: str = "OFF",
                  conf: int = 1, eff_max_open: int = 0) -> dict[str, str]:
    """live_metrics สำหรับ Live Indicators card (key -> str ตาม spec)."""
    chop = ic.get("chop", float("nan"))
    return {
        "Engine": str(engine_label),
        "ATR": f"{ic.get('atr', 0.0):.2f}",
        "RSI": f"{ic.get('rsi', 50.0):.1f}",
        "StochK": f"{ic.get('stoch_k', 50.0):.1f}",
        "ADX": f"{ic.get('adx', 0.0):.1f}",
        "CHOP": ("n/a" if (isinstance(chop, float) and math.isnan(chop)) else f"{chop:.1f}"),
        "Conf Mult": f"x{int(conf)}",
        "Max Open": str(int(eff_max_open)),
    }


if __name__ == "__main__":
    # smoke check — สร้าง synthetic candles แล้วรัน backtest
    import random

    random.seed(7)
    base = 2000.0
    start = int(datetime(2025, 1, 6, 0, 0, tzinfo=timezone.utc).timestamp())  # Monday
    rows = []
    price = base
    for k in range(3000):
        drift = random.uniform(-1.5, 1.5)
        o = price
        c = price + drift
        h = max(o, c) + random.uniform(0.0, 1.0)
        l = min(o, c) - random.uniform(0.0, 1.0)
        rows.append({"time": start + k * 300, "open": o, "high": h,
                     "low": l, "close": c, "volume": random.randint(50, 500)})
        price = c

    res = run_backtest(rows, balance=100.0, tick_value=1.0, tick_size=0.01,
                       min_lot=0.01, max_lot=100.0, lot_step=0.01)
    print("backtest:", res)

    out, cfg2, metrics, _ = process_strategy(
        {"candles": rows, "positions": [], "symbol": "XAUUSD", "timeframe": "M5",
         "is_new_bar": True, "balance": 100.0, "tick_value": 1.0, "tick_size": 0.01,
         "min_lot": 0.01, "max_lot": 100.0, "lot_step": 0.01},
        DEFAULT_CONFIG, lambda m: print("LOG:", m))
    print("live action:", out.get("action"), "engine:", out.get("engine"))
    print("metrics:", metrics)
