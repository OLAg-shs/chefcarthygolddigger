import os
import requests
import pandas as pd
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from dotenv import load_dotenv
from utils.twelve_data import get_data
# custom module to fetch OHLCV
import datetime

load_dotenv()

TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SYMBOLS = ["XAU/USD", "BTC/USD", "AAPL", "EUR/USD"]

TRADINGVIEW_SYMBOLS = {
    "XAU/USD": "FX_IDC:XAUUSD",
    "BTC/USD": "BINANCE:BTCUSDT",
    "AAPL": "NASDAQ:AAPL",
    "EUR/USD": "FX_IDC:EURUSD"
}


def analyze_market(df):
    insight = ""

    close = df['close']
    high = df['high']
    low = df['low']

    rsi = RSIIndicator(close).rsi()
    macd = MACD(close)
    bb = BollingerBands(close)

    last_rsi = round(rsi.iloc[-1], 2)
    last_macd = round(macd.macd_diff().iloc[-1], 4)
    last_upper = round(bb.bollinger_hband().iloc[-1], 2)
    last_lower = round(bb.bollinger_lband().iloc[-1], 2)
    last_close = round(close.iloc[-1], 2)

    rsi_signal = "Neutral"
    if last_rsi > 70:
        rsi_signal = "Overbought"
    elif last_rsi < 30:
        rsi_signal = "Oversold"

    macd_signal = "Bullish" if last_macd > 0 else "Bearish"

    bb_signal = "Near Resistance" if last_close >= last_upper else (
        "Near Support" if last_close <= last_lower else "Mid Band")

    insight += f"📉 RSI: {last_rsi} ({rsi_signal})\n"
    insight += f"📈 MACD: {last_macd} ({macd_signal})\n"
    insight += f"📊 Bollinger: {last_close} is {bb_signal} [{last_lower} - {last_upper}]\n"

    return insight, {
        "rsi": last_rsi,
        "macd": last_macd,
        "bb_signal": bb_signal
    }


def generate_prompt(symbol, indicator_text, indicators):
    prompt = f"""You are a professional trading assistant.
Market: {symbol}
Timeframe: 1H
Current Technical Analysis:
{indicator_text}

Based on the data above:
1. What is the overall bias (BUY, SELL, HOLD)?
2. What is your confidence level from 1–10?
3. Suggest Entry, Stop Loss, TP1, TP2.
4. Explain how each indicator contributes to your decision.
Respond in a clear format.
"""
    return prompt


def query_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }

    response = requests.post(url, json=data, headers=headers)
    result = response.json()
    return result["choices"][0]["message"]["content"]


def run_analysis():
    results = []
    for symbol in SYMBOLS:
        try:
            df = get_data(symbol, interval="1h", outputsize=100)
            if df is None or len(df) < 20:
                continue

            indicator_text, indicators = analyze_market(df)
            prompt = generate_prompt(symbol, indicator_text, indicators)
            ai_response = query_groq(prompt)

            # Only keep high confidence results
            confidence = 0
            for line in ai_response.splitlines():
                if "confidence" in line.lower():
                    digits = ''.join(filter(str.isdigit, line))
                    if digits:
                        confidence = int(digits[:2]) if len(digits) >= 2 else int(digits[0])
                    break

            if confidence >= 8:
                results.append({
                    "symbol": TRADINGVIEW_SYMBOLS.get(symbol, symbol.replace("/", "")),
                    "current_price": df['close'].iloc[-1],
                    "signal": ai_response.strip()
                })

        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
            continue

    return results
