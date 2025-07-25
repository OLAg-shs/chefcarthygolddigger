# utils/trading_bot.py

import os
from twelve_data import get_data  # your own wrapper
import openai
from dotenv import load_dotenv
import pandas as pd
import ta
import datetime

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SYMBOLS = ["XAU/USD", "BTC/USD", "AAPL", "EUR/USD"]

def calculate_indicators(df):
    df['EMA12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['EMA26'] = ta.trend.ema_indicator(df['close'], window=26)
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.macd(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_SIGNAL'] = macd.macd_signal()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14)
    df['STOCH_K'] = stoch.stoch()
    df['STOCH_D'] = stoch.stoch_signal()
    df['CCI'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
    df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    bb = ta.volatility.BollingerBands(df['close'])
    df['BB_HIGH'] = bb.bollinger_hband()
    df['BB_LOW'] = bb.bollinger_lband()
    return df

def summarize_indicators(df):
    last = df.iloc[-1]
    summary = {
        "price": round(last["close"], 2),
        "EMA12": round(last["EMA12"], 2),
        "EMA26": round(last["EMA26"], 2),
        "RSI": round(last["RSI"], 2),
        "MACD": round(last["MACD"], 2),
        "MACD_SIGNAL": round(last["MACD_SIGNAL"], 2),
        "STOCH_K": round(last["STOCH_K"], 2),
        "STOCH_D": round(last["STOCH_D"], 2),
        "CCI": round(last["CCI"], 2),
        "ADX": round(last["ADX"], 2),
        "ATR": round(last["ATR"], 2),
        "BB_HIGH": round(last["BB_HIGH"], 2),
        "BB_LOW": round(last["BB_LOW"], 2),
    }
    return summary

def ask_groq(summary, symbol):
    openai.api_key = GROQ_API_KEY
    indicators = "\n".join([f"{k} = {v}" for k, v in summary.items() if k != "price"])
    prompt = f"""
You are a trading expert. Analyze the following indicators for {symbol} on the 1H chart and predict:

- 1–2 hour trend
- 3–4 hour trend
- 1 day trend
- Whether it's a confirmed trade signal or not
- Confidence score out of 10
- Explain WHY based on indicators

Current Price = {summary['price']}
Indicators:
{indicators}
    """.strip()

    response = openai.ChatCompletion.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

def run_analysis():
    results = []
    for symbol in SYMBOLS:
        try:
            df = get_data(symbol, interval="1h", outputsize=100)
            df = calculate_indicators(df)
            summary = summarize_indicators(df)
            insight = ask_groq(summary, symbol)
            results.append({
                "symbol": symbol,
                "price": summary["price"],
                "insight": insight
            })
        except Exception as e:
            results.append({
                "symbol": symbol,
                "price": "N/A",
                "insight": f"⚠️ Error fetching or analyzing data: {str(e)}"
            })
    return results
