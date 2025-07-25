import os
import requests
import pandas as pd
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator
from ta.trend import CCIIndicator
from dotenv import load_dotenv
from utils.twelve_data import get_data
from utils.indicator_utils import detect_price_action, detect_break_retest

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
    close = df['close']
    high = df['high']
    low = df['low']

    # === Indicators ===
    rsi = RSIIndicator(close).rsi()
    macd = MACD(close)
    bb = BollingerBands(close)
    ema = EMAIndicator(close, window=50).ema_indicator()
    stoch = StochasticOscillator(high, low, close).stoch()
    cci = CCIIndicator(high, low, close).cci()

    # === Last values ===
    last_close = round(close.iloc[-1], 2)
    last_rsi = round(rsi.iloc[-1], 2)
    last_macd = round(macd.macd_diff().iloc[-1], 4)
    last_upper = round(bb.bollinger_hband().iloc[-1], 2)
    last_lower = round(bb.bollinger_lband().iloc[-1], 2)
    last_ema = round(ema.iloc[-1], 2)
    last_stoch = round(stoch.iloc[-1], 2)
    last_cci = round(cci.iloc[-1], 2)

    # === Logic Flags ===
    rsi_signal = "Oversold" if last_rsi < 30 else "Overbought" if last_rsi > 70 else "Neutral"
    macd_signal = "Bullish" if last_macd > 0 else "Bearish"
    bb_signal = "Near Support" if last_close <= last_lower else "Near Resistance" if last_close >= last_upper else "Mid Band"
    ema_trend = "Bullish" if last_close > last_ema else "Bearish"
    stoch_signal = "Bullish" if last_stoch < 20 else "Bearish" if last_stoch > 80 else "Neutral"
    cci_signal = "Bullish" if last_cci > 100 else "Bearish" if last_cci < -100 else "Neutral"

    # === Price Action Patterns ===
    pattern = detect_price_action(df)
    structure = detect_break_retest(df)

    # === Summary text ===
    insight = f"""📉 RSI: {last_rsi} ({rsi_signal})
📈 MACD: {last_macd} ({macd_signal})
📊 Bollinger Bands: {last_close} is {bb_signal} [{last_lower} - {last_upper}]
📐 EMA(50): {last_ema} ({ema_trend})
🔁 Stochastic RSI: {last_stoch} ({stoch_signal})
🔃 CCI: {last_cci} ({cci_signal})
📌 Price Action: {pattern}
📌 Structure: {structure}
"""

    # === Agreement Check ===
    agree_count = sum([
        macd_signal == "Bullish" and rsi_signal in ["Oversold", "Neutral"],
        bb_signal == "Near Support" and macd_signal == "Bullish",
        ema_trend == "Bullish",
        stoch_signal == "Bullish",
        cci_signal == "Bullish"
    ])

    indicators = {
        "rsi": last_rsi,
        "macd": last_macd,
        "bb_signal": bb_signal,
        "ema": last_ema,
        "stoch": last_stoch,
        "cci": last_cci,
        "pattern": pattern,
        "structure": structure,
        "agreement_score": agree_count
    }

    return insight, indicators


def generate_prompt(symbol, indicator_text, indicators):
    prompt = f"""
You are an expert trading assistant.
Only give CONFIRMED trade signals that are backed by aligned indicators.
Market: {symbol}
Timeframe: 1H
Indicators & Technicals:
{indicator_text}

Based on the analysis:
1. What is the CONFIRMED trade signal? (BUY, SELL, HOLD)
2. Confidence Level (1-10)?
3. Entry, SL, TP1, TP2?
4. Brief justification using the indicator readings above.

If there is no strong agreement, say: "Hold – Not Confirmed Yet".
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
            if df is None or len(df) < 50:
                continue

            indicator_text, indicators = analyze_market(df)

            # Filter out if RSI and MACD conflict strongly
            if indicators['rsi'] > 70 and indicators['macd'] < 0:
                continue
            if indicators['rsi'] < 30 and indicators['macd'] > 0:
                continue

            # Require minimum 3 agreeing indicators
            if indicators["agreement_score"] < 3:
                continue

            prompt = generate_prompt(symbol, indicator_text, indicators)
            ai_response = query_groq(prompt)

            # Extract confidence
            confidence = 0
            for line in ai_response.splitlines():
                if "confidence" in line.lower():
                    digits = ''.join(filter(str.isdigit, line))
                    if digits:
                        confidence = int(digits[:2]) if len(digits) >= 2 else int(digits[0])
                    break

            if confidence >= 8 and "hold" not in ai_response.lower():
                results.append({
                    "symbol": TRADINGVIEW_SYMBOLS.get(symbol, symbol.replace("/", "")),
                    "current_price": df['close'].iloc[-1],
                    "signal": ai_response.strip()
                })

        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
            continue

    return results
