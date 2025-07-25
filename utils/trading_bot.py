# utils/trading_bot.py

import os
import pandas as pd
import requests
import ta
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TWELVE_API_KEY")
SYMBOL = "XAU/USD"
INTERVAL = "1h"
LIMIT = 200
CONFIDENCE_THRESHOLD = 8  # Minimum confidence required

def fetch_data(symbol=SYMBOL):
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={symbol}&interval={INTERVAL}&outputsize={LIMIT}"
        f"&apikey={API_KEY}"
    )
    resp = requests.get(url)
    data = resp.json()
    if "values" not in data:
        raise Exception(f"Failed fetch: {data}")
    df = pd.DataFrame(data["values"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["time"] = pd.to_datetime(df["datetime"])
    df = df.set_index("time").sort_index()
    return df.dropna()

def compute_indicators(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"] if "volume" in df.columns else None

    indicators = {
        "price": close.iloc[-1],
        "ema12": ta.trend.EMAIndicator(close, window=12).ema_indicator().iloc[-1],
        "ema26": ta.trend.EMAIndicator(close, window=26).ema_indicator().iloc[-1],
        "rsi": ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1],
        "macd": ta.trend.MACD(close).macd().iloc[-1],
        "macd_signal": ta.trend.MACD(close).macd_signal().iloc[-1],
        "stoch_k": ta.momentum.StochasticOscillator(high, low, close).stoch().iloc[-1],
        "stoch_d": ta.momentum.StochasticOscillator(high, low, close).stoch_signal().iloc[-1],
        "cci": ta.trend.CCIIndicator(high, low, close).cci().iloc[-1],
        "adx": ta.trend.ADXIndicator(high, low, close).adx().iloc[-1],
        "atr": ta.volatility.AverageTrueRange(high, low, close).average_true_range().iloc[-1],
        "bb_high": ta.volatility.BollingerBands(close).bollinger_hband().iloc[-1],
        "bb_low": ta.volatility.BollingerBands(close).bollinger_lband().iloc[-1],
    }

    if volume is not None:
        indicators["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume().iloc[-1]
        indicators["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(high, low, close, volume).chaikin_money_flow().iloc[-1]
    else:
        indicators["obv"] = None
        indicators["cmf"] = None

    return indicators

def analyze_and_signal(ind):
    msgs = []
    price = ind["price"]

    # --- TREND CHECK (EMA) ---
    if price > ind["ema12"] > ind["ema26"]:
        trend = "Bullish"
        msgs.append("📈 Price > EMA12 > EMA26 → Bullish trend")
    elif price < ind["ema12"] < ind["ema26"]:
        trend = "Bearish"
        msgs.append("📉 Price < EMA12 < EMA26 → Bearish trend")
    else:
        trend = "Neutral"
        msgs.append("🔁 Price/EMAs misaligned → Sideways/uncertain")

    # --- INDICATORS LOG ---
    msgs.append(f"💠 RSI = {ind['rsi']:.2f}")
    msgs.append(f"💠 MACD = {ind['macd']:.2f}, Signal = {ind['macd_signal']:.2f}")
    msgs.append(f"💠 Stochastic K/D = {ind['stoch_k']:.2f}/{ind['stoch_d']:.2f}")
    msgs.append(f"💠 CCI = {ind['cci']:.2f}")
    msgs.append(f"💠 ADX = {ind['adx']:.2f}")
    msgs.append(f"💠 ATR = {ind['atr']:.2f}")
    msgs.append(f"💠 Bollinger Bands = {ind['bb_low']:.2f} - {ind['bb_high']:.2f}")

    if ind["cmf"] is not None:
        msgs.append(f"💠 CMF = {ind['cmf']:.2f}")

    # --- CONFIRMATION SCORING ---
    confirmations = 0
    if trend == "Bullish" and ind["adx"] >= 25: confirmations += 1
    if trend == "Bearish" and ind["adx"] >= 25: confirmations += 1
    if ind["macd"] > ind["macd_signal"]: confirmations += 1
    if ind["rsi"] < 70 and ind["rsi"] > 30: confirmations += 1
    if ind["stoch_k"] > ind["stoch_d"]: confirmations += 1
    if ind["cci"] > 0: confirmations += 1

    confidence = int((confirmations / 6) * 10)

    if confidence < CONFIDENCE_THRESHOLD:
        return f"⚠️ No confirmed signal (Confidence: {confidence}/10)\n" + "\n".join(msgs)

    bias = "BUY" if trend == "Bullish" else "SELL"

    # --- ENTRY, SL, TP ---
    atr = ind["atr"]
    sl = price - atr if bias == "BUY" else price + atr
    tp1 = price + 2 * atr if bias == "BUY" else price - 2 * atr
    tp2 = price + 3 * atr if bias == "BUY" else price - 3 * atr

    # --- SHORT-TERM FORECASTS ---
    msg = f"""
📊 Symbol: {SYMBOL}
💵 Current Price: {price:.2f}
🎯 Bias: {bias}
🔥 Confidence: {confidence}/10

🔹 Entry: {price:.2f}
🔻 SL: {sl:.2f}
🔺 TP1 (1–2h): {tp1:.2f}
🔺 TP2 (3–4h): {tp2:.2f}
🕒 1D Forecast: {'Uptrend' if bias=='BUY' else 'Downtrend'}

{chr(10).join(msgs)}
    """
    return msg.strip()

def run_analysis():
    try:
        df = fetch_data()
        ind = compute_indicators(df)
        signal = analyze_and_signal(ind)
        return "🧠 AI Trading Insight\n\n" + signal
    except Exception as e:
        return f"❌ Error: {e}"
