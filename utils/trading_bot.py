import requests
import pandas as pd
import numpy as np
from ta import trend, momentum, volatility, volume
import datetime as dt
import os

API_KEY = os.getenv("TWELVE_API_KEY")
INTERVAL = "15min"
LIMIT = 200

def fetch_data(symbol):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={INTERVAL}&outputsize={LIMIT}&apikey={API_KEY}&format=JSON"
    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        raise Exception(f"Failed to fetch data for {symbol}: {data}")

    df = pd.DataFrame(data["values"])
    df.rename(columns={"datetime": "time", "close": "close", "open": "open", "high": "high", "low": "low", "volume": "volume"}, inplace=True)

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    # Convert numeric columns
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    else:
        df["volume"] = np.nan

    return df.dropna()

def compute_indicators(df):
    df["EMA_12"] = trend.ema_indicator(df["close"], window=12)
    df["EMA_26"] = trend.ema_indicator(df["close"], window=26)
    df["EMA_50"] = trend.ema_indicator(df["close"], window=50)
    df["SMA_20"] = trend.sma_indicator(df["close"], window=20)

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

    # Volume indicators only if volume data exists
    if not df["volume"].isna().all():
        df["OBV"] = volume.on_balance_volume(df["close"], df["volume"])
        df["CMF"] = volume.chaikin_money_flow(df["high"], df["low"], df["close"], df["volume"], window=20)
    else:
        df["OBV"] = np.nan
        df["CMF"] = np.nan

    return df

def explain_signals(row, symbol):
    messages = [f"\n📈 **Symbol**: {symbol}"]
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

    # Trend Analysis
    trend_bias = "Neutral"
    if price < ema12 and ema12 < ema26:
        trend_bias = "Bearish"
        messages.append("📉 Price < EMA12 < EMA26 → **Bearish trend**")
    elif price > ema12 and ema12 > ema26:
        trend_bias = "Bullish"
        messages.append("📈 Price > EMA12 > EMA26 → **Bullish trend**")
    else:
        messages.append("🔁 Price is between EMAs → **Sideways market**")

    # RSI
    if rsi < 30:
        messages.append(f"💠 RSI = {rsi:.2f} → **Oversold** (Buy signal)")
    elif rsi > 70:
        messages.append(f"💠 RSI = {rsi:.2f} → **Overbought** (Sell signal)")
    else:
        messages.append(f"💠 RSI = {rsi:.2f} → Neutral")

    # MACD
    if macd > macd_signal:
        messages.append(f"💠 MACD = {macd:.2f}, Signal = {macd_signal:.2f} → **Bullish crossover**")
    else:
        messages.append(f"💠 MACD = {macd:.2f}, Signal = {macd_signal:.2f} → **Bearish crossover**")

    # Stochastic
    if stochastic_k < 20:
        messages.append(f"💠 Stochastic K/D = {stochastic_k:.2f}/{stochastic_d:.2f} → **Oversold**")
    elif stochastic_k > 80:
        messages.append(f"💠 Stochastic K/D = {stochastic_k:.2f}/{stochastic_d:.2f} → **Overbought**")
    else:
        messages.append(f"💠 Stochastic K/D = {stochastic_k:.2f}/{stochastic_d:.2f} → Neutral")

    # CCI
    if cci > 100:
        messages.append(f"💠 CCI = {cci:.2f} → **Strong Buy signal**")
    elif cci < -100:
        messages.append(f"💠 CCI = {cci:.2f} → **Strong Sell signal**")
    else:
        messages.append(f"💠 CCI = {cci:.2f} → Neutral")

    # ADX
    messages.append(f"💠 ADX = {adx:.2f} → {'Strong' if adx > 25 else 'Weak'} trend strength")

    # Volatility
    messages.append(f"💠 ATR = {atr:.2f} → Volatility measure")

    # Volume indicators
    if not np.isnan(obv) and not np.isnan(cmf):
        messages.append(f"💠 OBV = {obv:.2f}, CMF = {cmf:.2f} → {'Positive flow' if cmf > 0 else 'Negative flow'}")
    else:
        messages.append("💠 Volume Flow: N/A")

    # Bollinger Bands
    if price > bb_upper:
        bb_status = "above"
    elif price < bb_lower:
        bb_status = "below"
    else:
        bb_status = "within"
    messages.append(f"💠 Bollinger Bands: Price is **{bb_status}** the bands")

    # Final decision logic
    bias = "BUY" if trend_bias == "Bullish" and rsi < 70 and macd > macd_signal else \
           "SELL" if trend_bias == "Bearish" and rsi > 30 and macd < macd_signal else "HOLD"

    sl = price + atr if bias == "SELL" else price - atr if bias == "BUY" else price
    tp1 = price - 2 * atr if bias == "SELL" else price + 2 * atr if bias == "BUY" else price
    tp2 = price - 3 * atr if bias == "SELL" else price + 3 * atr if bias == "BUY" else price

    summary = f"""

📊 **Market Summary**
Trend: {trend_bias}
Momentum: {"Strong" if adx > 25 else "Weak"}
Volatility (ATR): {atr:.2f}
Volume Flow: {"Positive" if cmf > 0 else "Negative" if cmf < 0 else "N/A"}

🎯 **Bias**: {bias}
🔹 Entry Price: {price:.2f}
🔻 Stop Loss: {sl:.2f}
🔺 Take Profit 1: {tp1:.2f}
🔺 Take Profit 2: {tp2:.2f}
"""
    return "\n".join(messages) + summary

def run_analysis():
    symbols = ["XAU/USD", "BTC/USD", "AAPL"]
    reports = []
    for symbol in symbols:
        try:
            df = fetch_data(symbol)
            df = compute_indicators(df)
            latest = df.iloc[-1]
            report = explain_signals(latest, symbol)
            reports.append(report)
        except Exception as e:
            reports.append(f"❌ Error analyzing {symbol}: {str(e)}")
    return "\n\n".join(reports)
