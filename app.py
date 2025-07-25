# utils/trading_bot.py

import os
import pandas as pd
import requests
import ta
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TWELVE_API_KEY")
INTERVAL = "1h"
LIMIT = 200
CONFIDENCE_THRESHOLD = 8

MARKETS = [
    "XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY", "BTC/USD", "ETH/USD",
    "US30", "NAS100", "SPX500", "WTI/USD", "NG/USD"
]

def fetch_data(symbol):
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={symbol}&interval={INTERVAL}&outputsize={LIMIT}&apikey={API_KEY}"
    )
    resp = requests.get(url)
    data = resp.json()
    if "values" not in data:
        raise Exception(f"Fetch failed for {symbol}: {data.get('message', 'No data returned')}")
    df = pd.DataFrame(data["values"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["time"] = pd.to_datetime(df["datetime"])
    df = df.set_index("time").sort_index()
    return df.dropna()

def compute_indicators(df):
    close, high, low = df["close"], df["high"], df["low"]
    indicators = {
        "price": close.iloc[-1],
        "ema12": ta.trend.EMAIndicator(close, 12).ema_indicator().iloc[-1],
        "ema26": ta.trend.EMAIndicator(close, 26).ema_indicator().iloc[-1],
        "rsi": ta.momentum.RSIIndicator(close).rsi().iloc[-1],
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
    return indicators

def analyze_market(symbol, ind):
    msgs = []
    price = ind["price"]
    trend = "Neutral"
    if price > ind["ema12"] > ind["ema26"]:
        trend = "Bullish"
        msgs.append("📈 Price > EMA12 > EMA26 → Bullish trend")
    elif price < ind["ema12"] < ind["ema26"]:
        trend = "Bearish"
        msgs.append("📉 Price < EMA12 < EMA26 → Bearish trend")
    else:
        msgs.append("🔁 Price/EMAs misaligned → Sideways trend")

    msgs += [
        f"💠 RSI = {ind['rsi']:.2f}",
        f"💠 MACD = {ind['macd']:.2f}, Signal = {ind['macd_signal']:.2f}",
        f"💠 Stoch K/D = {ind['stoch_k']:.2f}/{ind['stoch_d']:.2f}",
        f"💠 CCI = {ind['cci']:.2f}",
        f"💠 ADX = {ind['adx']:.2f}",
        f"💠 ATR = {ind['atr']:.2f}",
        f"💠 Bollinger Bands = {ind['bb_low']:.2f} - {ind['bb_high']:.2f}",
    ]

    conf = 0
    if trend == "Bullish" and ind["adx"] >= 25: conf += 1
    if 30 < ind["rsi"] < 70: conf += 1
    if ind["macd"] > ind["macd_signal"]: conf += 1
    if ind["stoch_k"] > ind["stoch_d"] and ind["stoch_k"] < 80: conf += 1
    if ind["cci"] > 0: conf += 1

    confidence = int(conf * 2)
    bias = "HOLD"
    if conf >= 4:
        bias = "BUY" if trend == "Bullish" else "SELL"

    if bias == "HOLD" or confidence < CONFIDENCE_THRESHOLD:
        return f"\n📊 {symbol} — Confidence {confidence}/10\n" + "\n".join(msgs)

    sl = price - ind["atr"] if bias == "BUY" else price + ind["atr"]
    tp1 = price + 2 * ind["atr"] if bias == "BUY" else price - 2 * ind["atr"]
    tp2 = price + 3 * ind["atr"] if bias == "BUY" else price - 3 * ind["atr"]

    return f"""
📊 {symbol}
Price: {price:.2f}
🎯 Bias: {bias} (Confidence: {confidence}/10)
🔻 SL: {sl:.2f} | 🔺 TP1: {tp1:.2f}, TP2: {tp2:.2f}
""" + "\n".join(msgs)

def run_analysis():
    insights = []
    for market in MARKETS:
        try:
            df = fetch_data(market)
            ind = compute_indicators(df)
            result = analyze_market(market, ind)
            insights.append(result)
        except Exception as e:
            insights.append(f"⚠️ {market} → Error: {e}")
    return "\n\n".join(insights)
