import pandas as pd
import numpy as np
import time
from datetime import datetime
from coinbase_advanced.client import CoinbaseAdvancedTradeAPIClient

# --- Coinbase API setup ---
API_KEY = "YOUR_COINBASE_API_KEY"
API_SECRET = "YOUR_COINBASE_API_SECRET"
client = CoinbaseAdvancedTradeAPIClient.from_cloud_api_keys(API_KEY, API_SECRET)

# --- Futures products ---
pairs = ["BTC-PERP", "ETH-PERP"]  # Add more regulated futures as needed

# --- Risk parameters ---
account_balance = 1000          # USD
risk_per_trade_pct = 0.01       # 1% per trade
max_daily_loss = 0.03           # 3% daily loss limit
max_trades_per_day = 5
trades_today = 0
daily_loss = 0

# --- Trade log ---
log_file = "coinbase_futures_trades.csv"
columns = ["timestamp","pair","side","units","entry_price","stop_loss","atr","exit_price","exit_time","profit"]
try:
    trade_log = pd.read_csv(log_file)
except FileNotFoundError:
    trade_log = pd.DataFrame(columns=columns)
    trade_log.to_csv(log_file, index=False)

# --- Functions ---
def fetch_5m_candles(product_id, limit=500):
    """Fetch 5-min futures candles"""
    raw = client.get_futures_candles(product_id=product_id, granularity=300, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp","low","high","open","close","volume"])
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df.set_index("timestamp", inplace=True)
    return df

def compute_ema_trend(df, short=20, long=50):
    df["ema_short"] = df["close"].ewm(span=short).mean()
    df["ema_long"] = df["close"].ewm(span=long).mean()
    df["trend"] = np.where(df["ema_short"] > df["ema_long"], 1, -1)
    return df

def compute_atr(df, window=14):
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low","high_close","low_close"]].max(axis=1)
    df["atr"] = df["tr"].rolling(window).mean()
    return df

def calculate_units(account_balance, risk_per_trade, stop_distance):
    return risk_per_trade / stop_distance

def log_trade_open(pair, side, units, entry_price, stop_loss, atr):
    global trade_log
    trade_log = trade_log.append({
        "timestamp": datetime.utcnow(),
        "pair": pair,
        "side": side,
        "units": units,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "atr": atr,
        "exit_price": None,
        "exit_time": None,
        "profit": None
    }, ignore_index=True)
    trade_log.to_csv(log_file, index=False)
    print(f"[OPEN] {pair} {side} units={units} entry={entry_price:.2f} stop={stop_loss:.2f}")

def update_trade_close(pair, side, exit_price):
    global trade_log, daily_loss
    mask = (trade_log["pair"]==pair) & (trade_log["side"]==side) & (trade_log["profit"].isnull())
    if not mask.any():
        return
    idx = trade_log[mask].index[-1]
    units = trade_log.at[idx,"units"]
    entry_price = trade_log.at[idx,"entry_price"]
    profit = (exit_price - entry_price)*units if side=="buy" else (entry_price - exit_price)*units

    trade_log.at[idx,"exit_price"] = exit_price
    trade_log.at[idx,"exit_time"] = datetime.utcnow()
    trade_log.at[idx,"profit"] = profit
    daily_loss += max(0, -profit)
    trade_log.to_csv(log_file, index=False)
    print(f"[CLOSE] {pair} {side} exit={exit_price:.2f} profit={profit:.2f}")

def place_order(product_id, side, units):
    """Place market order on Coinbase Futures"""
    if side=="buy":
        order = client.market_order_buy(client_order_id=str(time.time()), product_id=product_id, quote_size=str(units))
    else:
        order = client.market_order_sell(client_order_id=str(time.time()), product_id=product_id, quote_size=str(units))
    return float(order["price"])

def update_trailing_stop(pair, side, atr_distance):
    df = fetch_5m_candles(pair, limit=1)
    price = df["close"].iloc[-1]
    open_trades = trade_log[(trade_log["pair"]==pair) & (trade_log["side"]==side) & (trade_log["profit"].isnull())]
    if open_trades.empty:
        return
    idx = open_trades.index[-1]
    stop_loss = trade_log.at[idx,"stop_loss"]

    new_stop = max(stop_loss, price-atr_distance) if side=="buy" else min(stop_loss, price+atr_distance)
    trade_log.at[idx,"stop_loss"] = new_stop

    if (side=="buy" and price <= new_stop) or (side=="sell" and price >= new_stop):
        update_trade_close(pair, side, price)

# --- Main Loop ---
while True:
    now = datetime.utcnow()
    if now.hour==0 and now.minute<5:
        trades_today=0
        daily_loss=0

    if daily_loss >= max_daily_loss*account_balance:
        print("Daily loss limit reached. Waiting 5 min...")
        time.sleep(300)
        continue

    signals=[]
    for pair in pairs:
        df = fetch_5m_candles(pair)
        df = compute_atr(df)
        df = compute_ema_trend(df)
        slope = (df["close"].iloc[-1]-df["close"].iloc[-2])

        # Entry conditions
        long_cond = (df["trend"].iloc[-1]==1) and (slope>0) and (df["atr"].iloc[-1]>df["atr"].rolling(50).mean().iloc[-1])
        short_cond = (df["trend"].iloc[-1]==-1) and (slope<0) and (df["atr"].iloc[-1]>df["atr"].rolling(50).mean().iloc[-1])

        if long_cond:
            signals.append((pair,1,df["atr"].iloc[-1]))
        elif short_cond:
            signals.append((pair,-1,df["atr"].iloc[-1]))

    # Sort by ATR for strongest signals
    signals.sort(key=lambda x:x[2], reverse=True)

    for s in signals[:max_trades_per_day]:
        if trades_today>=max_trades_per_day:
            break
        pair,direction,atr_val = s
        stop_distance = atr_val*2
        risk_per_trade = risk_per_trade_pct*account_balance
        units = calculate_units(account_balance,risk_per_trade,stop_distance)
        side = "buy" if direction==1 else "sell"
        entry_price = place_order(pair,side,units)
        log_trade_open(pair,side,units,entry_price,stop_distance,atr_val)
        trades_today+=1

    # Update trailing stops
    for pair,direction,atr_val in signals:
        side = "buy" if direction==1 else "sell"
        update_trailing_stop(pair,side,atr_val*2)

    # Sleep until next 5-min candle
    sleep_seconds = 300 - (datetime.utcnow().minute % 5)*60 - datetime.utcnow().second
    time.sleep(sleep_seconds)
