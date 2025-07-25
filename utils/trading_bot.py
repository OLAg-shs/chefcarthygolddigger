import os
import pandas as pd
from dotenv import load_dotenv
from twelve_data import get_data  # Ensure this is your custom data fetcher
from groq_ai import ask_groq  # Ensure this is your wrapper to call Groq
from utils.indicator_utils import analyze_indicators  # All indicator logic here

load_dotenv()
SYMBOLS = ["XAU/USD", "BTC/USD", "AAPL", "EUR/USD"]  # Add more as needed

def run_analysis():
    results = []

    for symbol in SYMBOLS:
        print(f"\n🔍 Analyzing {symbol}...")
        df = get_data(symbol=symbol, interval="1h", outputsize=200)

        if df is None or df.empty:
            print(f"❌ No data for {symbol}")
            continue

        indicators = analyze_indicators(df)
        if not indicators:
            print(f"⚠️ Indicator analysis failed for {symbol}")
            continue

        prompt = f"""
        Analyze the following market data for {symbol} and provide only high-confidence signals (≥ 8/10):

        📊 Indicators:
        - Trend: {'Bullish' if df['close'].iloc[-1] > indicators['EMA12'] > indicators['EMA26'] else 'Bearish'}
        - RSI: {indicators['RSI']}
        - MACD: {indicators['MACD']} vs Signal: {indicators['MACD_Signal']}
        - Stochastic: {indicators['Stoch_K']} / {indicators['Stoch_D']}
        - CCI: {indicators['CCI']}
        - ADX: {indicators['ADX']}
        - ATR: {indicators['ATR']}
        - Bollinger Bands: {indicators['Bollinger_Lower']} - {indicators['Bollinger_Upper']}

        Current Price: {df['close'].iloc[-1]}

        Give precise and reliable predictions with:
        - Confidence score (1–10)
        - 1–2h, 3–4h, and 1-day direction
        - Entry, SL, TP1, TP2 levels
        """

        ai_response = ask_groq(prompt)
        results.append({
            "symbol": symbol,
            "current_price": df['close'].iloc[-1],
            "signal": ai_response
        })

    return results
