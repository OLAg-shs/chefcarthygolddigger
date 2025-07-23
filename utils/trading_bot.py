# utils/trading_bot.py

import requests
import pandas as pd
import numpy as np
from ta import trend, momentum, volatility, volume
import datetime as dt
import os

# --- CONFIGURATION ---
API_KEY = os.getenv("TWELVE_API_KEY")  # ✅ FIXED: matches .env key on Render
SYMBOL = "XAU/USD"
INTERVAL = "15min"
LIMIT = 200

def fetch_data():
    url = f"https://api.twelvedata.com/time_series?symbol={SYMBOL}&interval={INTERVAL}&outputsize={LIMIT}&apikey={API_KEY}&format=JSON"
    response = requests.get(url)
    data = response.json()
    if "values" not in data:
        raise Exception("Failed to fetch data: " + str(data))

    df = pd.DataFrame(data["values"])
    df.rename(columns={"datetime": "time", "close": "close", "open": "open", "high": "high", "low": "low", "volume": "volume"}, inplace=True)
    df = df.astype(float, errors='ignore')
    df["time"] = pd.to_datetime(df["time"])
    df.sort_values("time", inplace=True)
    return df

def compute_indicators(df):
    try:
        df["EMA_12"] = trend.ema_indicator(df["close"], window=12)
        df["EMA_26"] = trend.ema_indicator(df["close"], window=26)
        df["EMA_50"] = trend.ema_indicator(df["close"], window=50)
        df["SMA_20"] = trend.sma_indicator(df["close"], window=20)
        df["SMA_50"] = trend.sma_indicator(df["close"], window=50)

        df["MACD"] = trend.macd(df["close"])
        df["MACD_Signal"] = trend.macd_signal(df["close"])

        df["RSI"] = momentum.rsi(df["close"], window=14)
        df["Stochastic_K"] = momentum.stoch(df["high"], df["low"], df["close"])
        df["Stochastic_D"] = df["Stochastic_K"].rolling(window=3).mean()
        df["CCI"] = trend.cci(df["high"], df["low"], df["close"], window=20)

        df["ADX"] = trend.adx(df["high"], df["low"], df["close"], window=14)

        bb = volatility.BollingerBands(df["close"], window=20, window_dev=2)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()

        df["ATR"] = volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

        df["OBV"] = volume.on_balance_volume(df["close"], df["volume"])
        df["CMF"] = volume.chaikin_money_flow(df["high"], df["low"], df["close"], df["volume"], window=20)

        return df
    except Exception as e:
        raise Exception("Failed to compute indicators: " + str(e))

def explain_signals(row):
    messages = []
    price = row["close"]
    ema12, ema26 = row["EMA_12"], row["EMA_26"]
    rsi = row["RSI"]
    stochastic_k, stochastic_d = row["Stochastic_K"], row["Stochastic_D"]
    macd, macd_signal = row["MACD"], row["MACD_Signal"]
    bb_upper, bb_lower = row["BB_upper"], row["BB_lower"]
    atr = row["ATR"]
    adx = row["ADX"]
    cci = row["CCI"]
    obv = row["OBV"]
    cmf = row["CMF"]

    trend_bias = "Neutral"
    if price < ema12 and ema12 < ema26:
        trend_bias = "Bearish"
        messages.append("📉 Price is below EMA12 and EMA26 → Bearish trend")
    elif price > ema12 and ema12 > ema26:
        trend_bias = "Bullish"
        messages.append("📈 Price is above EMA12 and EMA26 → Bullish trend")
    else:
        messages.append("🔁 Price is between EMAs → Sideways market")

    messages.append(f"💠 RSI = {rsi:.2f} → {'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'}")
    messages.append(f"💠 MACD = {macd:.2f}, Signal = {macd_signal:.2f} → {'Bullish crossover' if macd > macd_signal else 'Bearish crossover'}")
    messages.append(f"💠 Stochastic K/D = {stochastic_k:.2f}/{stochastic_d:.2f} → {'Oversold' if stochastic_k < 20 else 'Overbought' if stochastic_k > 80 else 'Neutral'}")
    messages.append(f"💠 CCI = {cci:.2f} → {'Bullish' if cci > 100 else 'Bearish' if cci < -100 else 'Neutral'}")
    messages.append(f"💠 ADX = {adx:.2f} → {'Strong trend' if adx > 25 else 'Weak trend'}")
    messages.append(f"💠 ATR = {atr:.2f} → Volatility measure")
    messages.append(f"💠 OBV = {obv:.2f}, CMF = {cmf:.2f} → {'Positive flow' if cmf > 0 else 'Negative flow'}")
    messages.append(f"💠 Bollinger Bands: Price is {'below' if price < bb_lower else 'above' if price > bb_upper else 'within'} the bands")

    bias = "BUY" if trend_bias == "Bullish" and rsi < 70 and macd > macd_signal else \
           "SELL" if trend_bias == "Bearish" and rsi > 30 and macd < macd_signal else "HOLD"

    sl = price + atr if bias == "SELL" else price - atr if bias == "BUY" else price
    tp1 = price - 2 * atr if bias == "SELL" else price + 2 * atr if bias == "BUY" else price
    tp2 = price - 3 * atr if bias == "SELL" else price + 3 * atr if bias == "BUY" else price

    summary = f"""

📊 **Market Summary**
Trend: {trend_bias}
Momentum: {"Weak" if adx < 20 else "Strong"}
Volatility (ATR): {atr:.2f}
Volume Flow: {"Positive" if cmf > 0 else "Negative"}

🎯 **Bias**: {bias}

🔹 Entry Price: {price:.2f}
🔻 Stop Loss: {sl:.2f}
🔺 Take Profit 1: {tp1:.2f}
🔺 Take Profit 2: {tp2:.2f}
"""

    return "\n".join(messages) + summary

# ✅ Main function to use in app.py
def run_analysis():
    df = fetch_data()
    df = compute_indicators(df)
    latest = df.iloc[-1]
    report = explain_signals(latest)
    return report
