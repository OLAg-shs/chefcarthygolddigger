import os
import requests
import pandas as pd
import ta
from datetime import datetime

# Get TwelveData API key from environment variable
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")

def fetch_data(symbol):
    interval = "15min"
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=100&apikey={TWELVE_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        raise ValueError(f"Failed to fetch data for {symbol}: {data.get('message', 'Unknown error')}")

    df = pd.DataFrame(data["values"])
    df = df.astype({"open": float, "high": float, "low": float, "close": float})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    return df

def compute_indicators(df):
    # Trend indicators
    df["EMA12"] = ta.trend.ema_indicator(df["close"], window=12)
    df["EMA26"] = ta.trend.ema_indicator(df["close"], window=26)
    
    # Momentum
    df["RSI"] = ta.momentum.rsi(df["close"])
    macd = ta.trend.macd(df["close"])
    df["MACD"] = macd.macd_diff()
    df["MACD_line"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["STOCH_K"] = ta.momentum.stoch(df["high"], df["low"], df["close"])
    df["STOCH_D"] = ta.momentum.stoch_signal(df["high"], df["low"], df["close"])
    df["CCI"] = ta.trend.cci(df["high"], df["low"], df["close"])
    df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"])
    
    # Volatility
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])
    df["BB_upper"] = ta.volatility.bollinger_hband(df["close"])
    df["BB_lower"] = ta.volatility.bollinger_lband(df["close"])

    # Volume indicators (some assets don’t provide volume)
    try:
        df["OBV"] = ta.volume.on_balance_volume(df["close"], df["volume"])
        df["CMF"] = ta.volume.chaikin_money_flow(df["high"], df["low"], df["close"], df["volume"])
        has_volume = True
    except:
        df["OBV"] = df["CMF"] = None
        has_volume = False

    return df, has_volume

def interpret_indicators(df, has_volume):
    latest = df.iloc[-1]
    price = round(latest["close"], 2)

    explanations = []
    confidence = 0

    # Trend
    if latest["EMA12"] > latest["EMA26"]:
        trend = "Bullish"
        explanations.append("📈 Price > EMA12 > EMA26 → **Bullish trend**")
        confidence += 1
    else:
        trend = "Bearish"
        explanations.append("📉 Price < EMA12 < EMA26 → **Bearish trend**")

    # RSI
    rsi = latest["RSI"]
    if rsi > 70:
        explanations.append(f"💠 RSI = {rsi:.2f} → Overbought (SELL signal)")
    elif rsi < 30:
        explanations.append(f"💠 RSI = {rsi:.2f} → Oversold (BUY signal)")
        confidence += 1
    else:
        explanations.append(f"💠 RSI = {rsi:.2f} → Neutral")

    # MACD
    macd_line = latest["MACD_line"]
    macd_signal = latest["MACD_signal"]
    macd_diff = latest["MACD"]
    if macd_diff > 0:
        explanations.append(f"💠 MACD = {macd_line:.2f}, Signal = {macd_signal:.2f} → **Bullish crossover**")
        confidence += 1
    else:
        explanations.append(f"💠 MACD = {macd_line:.2f}, Signal = {macd_signal:.2f} → Bearish crossover")

    # Stochastic
    stoch_k = latest["STOCH_K"]
    stoch_d = latest["STOCH_D"]
    explanations.append(f"💠 Stochastic K/D = {stoch_k:.2f}/{stoch_d:.2f} → Neutral")

    # CCI
    cci = latest["CCI"]
    if cci > 100:
        explanations.append(f"💠 CCI = {cci:.2f} → Overbought")
    elif cci < -100:
        explanations.append(f"💠 CCI = {cci:.2f} → Oversold")
        confidence += 1
    else:
        explanations.append(f"💠 CCI = {cci:.2f} → Neutral")

    # ADX
    adx = latest["ADX"]
    if adx > 25:
        explanations.append(f"💠 ADX = {adx:.2f} → Strong trend")
        confidence += 1
    else:
        explanations.append(f"💠 ADX = {adx:.2f} → Weak trend")

    # ATR
    atr = round(latest["ATR"], 2)
    explanations.append(f"💠 ATR = {atr:.2f} → Measures market volatility")

    # Bollinger Bands
    bb_upper = latest["BB_upper"]
    bb_lower = latest["BB_lower"]
    if bb_lower < price < bb_upper:
        explanations.append("💠 Bollinger Bands: Price is within the bands (normal)")
    else:
        explanations.append("💠 Bollinger Bands: Price is outside the bands (volatility)")

    # Volume
    if has_volume:
        obv = latest["OBV"]
        cmf = latest["CMF"]
        explanations.append(f"💠 OBV = {obv:.2f}, CMF = {cmf:.2f} → {'Positive' if cmf > 0 else 'Negative'} volume flow")
        if cmf > 0: confidence += 1
    else:
        explanations.append("💠 Volume Flow: ❌ Not available for this symbol")

    # Bias decision
    bias = "BUY" if confidence >= 4 else "SELL"
    stop_loss = round(price * (0.998 if bias == "BUY" else 1.002), 2)
    take_profit_1 = round(price * (1.01 if bias == "BUY" else 0.99), 2)
    take_profit_2 = round(price * (1.02 if bias == "BUY" else 0.98), 2)

    return {
        "price": price,
        "bias": bias,
        "stop_loss": stop_loss,
        "take_profit_1": take_profit_1,
        "take_profit_2": take_profit_2,
        "confidence": confidence,
        "trend": trend,
        "atr": atr,
        "explanations": explanations,
    }

def run_analysis_for_symbol(symbol):
    try:
        df = fetch_data(symbol)
        df, has_volume = compute_indicators(df)
        insights = interpret_indicators(df, has_volume)
        return insights
    except Exception as e:
        return {"error": f"❌ Error analyzing {symbol}: {str(e)}"}
