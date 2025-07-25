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
CONFIDENCE_THRESHOLD = 8  # only trades ≥8/10

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

    indicators = {}
    indicators["price"] = close.iloc[-1]
    indicators["ema12"] = ta.trend.EMAIndicator(close, window=12).ema_indicator().iloc[-1]
    indicators["ema26"] = ta.trend.EMAIndicator(close, window=26).ema_indicator().iloc[-1]
    indicators["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    indicators["macd"] = macd.macd().iloc[-1]
    indicators["macd_signal"] = macd.macd_signal().iloc[-1]
    indicators["stoch_k"] = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3).stoch().iloc[-1]
    indicators["stoch_d"] = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3).stoch_signal().iloc[-1]
    indicators["cci"] = ta.trend.CCIIndicator(high, low, close, window=20).cci().iloc[-1]
    indicators["adx"] = ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]
    indicators["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    indicators["bb_high"] = bb.bollinger_hband().iloc[-1]
    indicators["bb_low"] = bb.bollinger_lband().iloc[-1]

    if volume is not None:
        indicators["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume().iloc[-1]
        indicators["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(high, low, close, volume, window=20).chaikin_money_flow().iloc[-1]
    else:
        indicators["obv"] = None
        indicators["cmf"] = None

    return indicators

def analyze_and_signal(ind):
    msgs = []
    price = ind["price"]

    # Trend via EMAs
    if price > ind["ema12"] > ind["ema26"]:
        trend = "Bullish"
        msgs.append("📈 Price > EMA12 > EMA26 → Bullish trend")
    elif price < ind["ema12"] < ind["ema26"]:
        trend = "Bearish"
        msgs.append("📉 Price < EMA12 < EMA26 → Bearish trend")
    else:
        trend = "Neutral"
        msgs.append("🔁 Price/EMAs misaligned → Sideways/uncertain market")

    # Indicator checks
    msgs.append(f"💠 RSI = {ind['rsi']:.2f} → {'Oversold' if ind['rsi']<30 else 'Overbought' if ind['rsi']>70 else 'Neutral'}")
    msgs.append(f"💠 MACD = {ind['macd']:.2f}, Signal = {ind['macd_signal']:.2f} → {'Bullish crossover' if ind['macd']>ind['macd_signal'] else 'Bearish crossover'}")
    msgs.append(f"💠 Stoch K/D = {ind['stoch_k']:.2f}/{ind['stoch_d']:.2f} → {'Oversold' if ind['stoch_k']<20 else 'Overbought' if ind['stoch_k']>80 else 'Neutral'}")
    msgs.append(f"💠 CCI = {ind['cci']:.2f} → {'Bullish' if ind['cci']>100 else 'Bearish' if ind['cci']<-100 else 'Neutral'}")
    msgs.append(f"💠 ADX = {ind['adx']:.2f} → {'Strong trend' if ind['adx']>=25 else 'Weak trend'}")
    msgs.append(f"💠 ATR = {ind['atr']:.2f} (volatility)")
    volflow = "Positive" if ind["cmf"] and ind["cmf"] > 0 else "Negative" if ind["cmf"] else "N/A"
    msgs.append(f"💠 CMF = {ind.get('cmf', 'N/A')} → {volflow}")

    # Count confirmations
    conf = 0
    if trend == "Bullish" and ind["adx"] >=25: conf +=1
    if ind["rsi"]<70 and ind["rsi"]>30: conf+=1
    if ind["macd"] > ind["macd_signal"]: conf+=1
    if ind["stoch_k"] > ind["stoch_d"] and ind["stoch_k"]<80: conf+=1
    if ind["cci"]>0: conf+=1

    # Determine bias only if enough confirmations
    if conf >=4:
        bias = "BUY" if trend=="Bullish" else "SELL"
    else:
        bias = "HOLD"

    # Confidence rating scaled
    confidence = int(10 * (conf / 5))

    if bias=="HOLD" or confidence < CONFIDENCE_THRESHOLD:
        return "\n".join(msgs) + f"\n⚠️ No confirmed trade signal — confidence {confidence}/10"

    sl = price - ind["atr"] if bias=="BUY" else price + ind["atr"]
    tp1 = price + 2*ind["atr"] if bias=="BUY" else price - 2*ind["atr"]
    tp2 = price + 3*ind["atr"] if bias=="BUY" else price - 3*ind["atr"]

    msgs.append(f"🎯 Bias = {bias} (Confidence: {confidence}/10)")
    msgs.append(f"🔹 Entry Price: {price:.2f}")
    msgs.append(f"🔻 Stop Loss: {sl:.2f}")
    msgs.append(f"🔺 Take Profit 1: {tp1:.2f}")
    msgs.append(f"🔺 Take Profit 2: {tp2:.2f}")

    return "\n".join(msgs)

def run_analysis():
    try:
        df = fetch_data()
        ind = compute_indicators(df)
        signal = analyze_and_signal(ind)
        return f"🧠 AI Trading Insight:\n\n📈 **Symbol**: {SYMBOL}\n\n" + signal
    except Exception as e:
        return f"❌ Error: {e}"
