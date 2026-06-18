# AI Spec: MT5 Bridge Strategy Extension System

This specification document guides AI coding assistants (e.g., ChatGPT, Claude) and developers in generating fully compatible strategy extensions (plugins) for the MT5 Python Bridge Server.

---

## 1. File Specification
- **Location:** All strategies must be saved inside the `./strategies/` folder.
- **Filename:** Use lowercase snake_case (e.g., `bollinger_bands_scalp.py`). The filename (excluding `.py`) acts as the `strategy_id`.
- **Runtime Reload:** The server automatically scans and hot-reloads these files every 2 seconds. No compilation or server restarts are required.

---

## 2. Core Requirements
Every strategy module must define exactly three elements:
1. **`STRATEGY_NAME`** (str): The display name shown in the dashboard's active strategy selector.
2. **`DEFAULT_CONFIG`** (dict): The parameter keys and default values editable in the dashboard.
3. **`process_strategy(data, config, add_log_fn)`** (function): The main execution engine.

---

## 3. Data Schemas

### A. Input Parameter: `data` (dict)
This dictionary contains live account metrics, market specs, and historical data fed from the MT5 client:
```python
{
    "symbol": "EURUSD",                     # str: Current symbol name
    "timeframe": "M1",                      # str: timeframe string (e.g. M1, M5, M15, H1)
    "is_new_bar": True,                     # bool: True only on the first tick of a new bar
    "balance": 10000.00,                    # float: Account balance
    "equity": 10050.00,                     # float: Account equity
    "spread": 12,                           # int: Current spread in points
    
    # Broker symbol specifications:
    "tick_size": 0.00001,                   # float: Minimum price change step (e.g., 0.00001 for 5-digit forex)
    "tick_value": 1.0,                      # float: Value of 1 tick change per 1 standard lot (USD)
    "min_lot": 0.01,                        # float: Minimum lot size allowed by broker
    "max_lot": 100.0,                       # float: Maximum lot size allowed by broker
    "lot_step": 0.01,                       # float: Lot step increment
    
    # Historical Candles (stored oldest to newest):
    "candles": [
        {
            "time": 1781795000,             # int: Epoch unix timestamp
            "open": 1.08500,                # float: Open price
            "high": 1.08550,                # float: High price
            "low": 1.08480,                 # float: Low price
            "close": 1.08520,               # float: Close price
            "tick_volume": 120              # int: Tick volume
        },
        ...
    ],
    
    # Open positions for this account:
    "positions": [
        {
            "ticket": 98765432,             # int: Unique position ID
            "symbol": "EURUSD",             # str: Symbol
            "type": "BUY",                  # str: Position type ("BUY" or "SELL")
            "volume": 0.10,                 # float: Lot size
            "price_open": 1.08450,          # float: Entry price
            "sl": 1.08300,                  # float: Stop Loss price (0.0 if none)
            "tp": 1.08750,                  # float: Take Profit price (0.0 if none)
            "profit": 7.00                  # float: Current profit/loss in USD
        }
    ],
    
    # Flag triggered manually by user via UI:
    "trigger_backtest": False               # bool: True if the user requested a parameters optimization
}
```

### B. Input Parameter: `config` (dict)
Contains the current values of the strategy settings edited via the UI.
Example: `{"indicator_period": 14, "risk_percent": 2.0}`

### C. Input Parameter: `add_log_fn` (function)
A logging callback. Call it like this: `add_log_fn("Signal triggered: EMA Crossover")` to print logs directly to the live dashboard console.

---

## 4. Expected Return Value
The `process_strategy` function **must** return a tuple of 4 elements:
`return res_dict, updated_config, live_metrics, bt_res`

### 1. `res_dict` (dict)
Specifies the action for the MT5 client. Supported structures:

#### Option A: Do Nothing
```python
{
    "action": "NONE",
    "display_line1": "No setup found",      # str: Primary HUD text on dashboard
    "display_line2": "RSI: 50.0"            # str: Secondary HUD text on dashboard
}
```

#### Option B: Open BUY or SELL Order
```python
{
    "action": "BUY",                        # str: "BUY" or "SELL"
    "lot": 0.10,                            # float: Calculated lot size
    "tp_multiplier": 2.0,                   # float: Take Profit multiplier (multiplied to ATR or pip size by client)
    "sl_multiplier": 1.5,                   # float: Stop Loss multiplier
    "reason": "RSI Crossed Oversold",       # str: Reason logged to file and MT5 terminal
    "display_line1": "Signal: BUY",
    "display_line2": "ATR SL/TP set"
}
```

#### Option C: Close a Specific Position
```python
{
    "action": "CLOSE",
    "ticket": 98765432,                     # int: Position ticket number to close
    "reason": "Crossover reverse exit",
    "display_line1": "Exit Triggered",
    "display_line2": "Closing Ticket #98765432"
}
```

#### Option D: Close All Positions
```python
{
    "action": "CLOSE_ALL",
    "reason": "Emergency exit trigger",
    "display_line1": "Emergency Exit",
    "display_line2": "Closing all active positions"
}
```

### 2. `updated_config` (dict)
A copy of `config` with optionally updated parameters (e.g., if a backtest optimization was executed and new values are set).

### 3. `live_metrics` (dict)
Key-value pairs displayed dynamically in the "Live Indicators" card on the UI:
```python
{
    "RSI (14)": "45.2",
    "SMA (50)": "1.08450",
    "Market State": "Ranging"
}
```

### 4. `bt_res` (dict or None)
Statistics from a parameter optimization run. Return `None` if `trigger_backtest` is False. If populated, it must follow this structure:
```python
{
    "direction": "BUY",                     # str: Suggested direction ("BUY", "SELL", or "NONE")
    "tp_multiplier": 2.2,                   # float: Best TP multiplier
    "sl_multiplier": 1.4,                   # float: Best SL multiplier
    "win_rate": 65.5,                       # float: Backtest win rate percent
    "max_drawdown": 3.20,                   # float: Maximum drawdown percent
    "trades": 45,                           # int: Total backtest trades executed
    "profit": 420.50,                       # float: Net profit in USD
    
    # Parameters that got optimized (updates UI config inputs):
    "rsi_overbought": 70.0,
    "rsi_oversold": 30.0
}
```

---

## 5. Coding Template (Copy & Paste for AI Prompting)
Here is a complete, minimal skeletal implementation template for generating new extensions:

```python
# Save as strategies/dummy_ema.py
import pandas as pd
import numpy as np

STRATEGY_NAME = "EMA Crossover Template"

DEFAULT_CONFIG = {
    "ema_fast": 9,
    "ema_slow": 21,
    "tp_multiplier": 2.0,
    "sl_multiplier": 1.5,
    "risk_percent": 1.0
}

def process_strategy(data, config, add_log_fn):
    candles = data.get("candles", [])
    positions = data.get("positions", [])
    trigger_backtest = bool(data.get("trigger_backtest", False))
    symbol = data.get("symbol", "UNKNOWN")
    
    fast_period = int(config.get("ema_fast", 9))
    slow_period = int(config.get("ema_slow", 21))
    risk_pct = float(config.get("risk_percent", 1.0))
    
    # 1. Require minimum candles for indicators
    min_required = max(fast_period, slow_period) + 5
    if len(candles) < min_required:
        res_dict = {
            "action": "NONE",
            "display_line1": "Initializing...",
            "display_line2": f"Bars: {len(candles)}/{min_required}"
        }
        return res_dict, config, {"Status": "Syncing bars"}, None

    # 2. Backtest optimization stub
    updated_config = config.copy()
    bt_res = None
    if trigger_backtest:
        add_log_fn("Running optimization...")
        # (Insert grid search optimization code here)
        # bt_res = { ... }

    # 3. Calculate Indicators
    df = pd.DataFrame(candles)
    df['close'] = pd.to_numeric(df['close'])
    df['ema_f'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['ema_s'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    
    curr_f = df['ema_f'].iloc[-1]
    curr_s = df['ema_s'].iloc[-1]
    prev_f = df['ema_f'].iloc[-2]
    prev_s = df['ema_s'].iloc[-2]
    curr_close = df['close'].iloc[-1]
    
    # 4. Generate Signal Actions
    res_dict = {
        "action": "NONE",
        "display_line1": f"Fast: {curr_f:.5f} | Slow: {curr_s:.5f}",
        "display_line2": f"Last Price: {curr_close:.5f}"
    }
    
    # Check if there is an active position for this symbol
    active_position = next((pos for pos in positions if pos.get("symbol") == symbol), None)
    
    # Signal Crossovers
    gold_cross = (prev_f <= prev_s) and (curr_f > curr_s)
    death_cross = (prev_f >= prev_s) and (curr_f < curr_s)
    
    if active_position:
        # Exit rules
        pos_type = active_position.get("type")
        ticket = active_position.get("ticket")
        if (pos_type == "BUY" and death_cross) or (pos_type == "SELL" and gold_cross):
            add_log_fn(f"Exit signal triggered. Closing position #{ticket}.")
            res_dict = {
                "action": "CLOSE",
                "ticket": ticket,
                "reason": "EMA Trend Reversal"
            }
    else:
        # Entry rules
        if gold_cross:
            add_log_fn("Gold Cross Crossover! Triggering BUY.")
            res_dict = {
                "action": "BUY",
                "lot": 0.1,  # Calculated or static lot size
                "tp_multiplier": config["tp_multiplier"],
                "sl_multiplier": config["sl_multiplier"],
                "reason": "EMA Gold Cross"
            }
        elif death_cross:
            add_log_fn("Death Cross Crossover! Triggering SELL.")
            res_dict = {
                "action": "SELL",
                "lot": 0.1,
                "tp_multiplier": config["tp_multiplier"],
                "sl_multiplier": config["sl_multiplier"],
                "reason": "EMA Death Cross"
            }
            
    # 5. Live metrics displayed on UI card
    live_metrics = {
        "Fast EMA": f"{curr_f:.5f}",
        "Slow EMA": f"{curr_s:.5f}",
        "Trend": "Bullish" if curr_f > curr_s else "Bearish"
    }
    
    return res_dict, updated_config, live_metrics, bt_res
```
