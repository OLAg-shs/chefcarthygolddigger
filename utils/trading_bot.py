import os
import pandas as pd
import requests
import datetime as dt
import numpy as np
import ta

from dotenv import load_dotenv

load_dotenv()

TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ========== Fetch Live Data ==========
def fetch_data(symbol):
    try:
        interval = "15min" if symbol != "AAPL" else "1min"  # Apple is a stock
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=100&apikey={TWELVE_API_KEY}"
        response = requests.get(url)
        data = response.json()

        if "values" not in data:
            raise ValueError(f"Failed to fetch data: {data}")

        df = pd.DataFrame(data["values"])
        df = df.rename(columns={
            "datetime": "time",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        })

        df = df.astype({
            "open": float, "high": float, "low": float, "close": float
        })

        if "volume" in df.columns:
            df["volume"] = df["volume"].astype(float)

        df = df.sort_values("time")
        df.reset_index(drop=True, inplace=True)

        return df
    except Exception as e:
        return str(e)

# ========== Compute Indicators ==========
def compute_indicators(df, symbol):
    try:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"] if "volume" in df.columns else None

        indicators = {}

        # EMAs
        ema12 = ta.trend.EMAIndicator(close, window=12).ema_indicator()
        ema26 = ta.trend.EMAIndicator(close, window=26).ema_indicator()
        indicators["EMA12"] = ema12.iloc[-1]
        indicators["EMA26"] = ema26.iloc[-1]

        # RSI
        rsi = ta.momentum.RSIIndicator(close).rsi()
        indicators["RSI"] = rsi.iloc[-1]

        # MACD
        macd = ta.trend.MACD(close)
        indicators["MACD"] = macd.macd().iloc[-1]
        indicators["MACD_signal"] = macd.macd_signal().iloc[-1]

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(high, low, close)
        indicators["Stoch_K"] = stoch.stoch().iloc[-1]
        indicators["Stoch_D"] = stoch.stoch_signal().iloc[-1]

        # CCI
        cci = ta.trend.CCIIndicator(high, low, close)
        indicators["CCI"] = cci.cci().iloc[-1]

        # ADX
        adx = ta.trend.ADXIndicator(high, low, close)
        indicators["ADX"] = adx.adx().iloc[-1]

        # ATR (Volatility)
        atr = ta.volatility.AverageTrueRange(high, low, close)
        indicators["ATR"] = atr.average_true_range().iloc[-1]

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close)
        indicators["BB_high"] = bb.bollinger_hband().iloc[-1]
        indicators["BB_low"] = bb.bollinger_lband().iloc[-1]

        # Volume Indicators
        if volume is not None:
            obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
            cmf = ta.volume.ChaikinMoneyFlowIndicator(high, low, close, volume).chaikin_money_flow()
            indicators["OBV"] = obv.iloc[-1]
            indicators["CMF"] = cmf.iloc[-1]
        else:
            indicators["OBV"] = None
            indicators["CMF"] = None

        # Last Price
        indicators["Current Price"] = close.iloc[-1]

        return indicators
    except Exception as e:
        return {"error": f"Failed to compute indicators: {e}"}

# ========== Analyze with Groq / GPT ==========
def analyze_with_ai(symbol, indicators):
    prompt = f"""
You are a professional trading assistant. Based on the following indicators for {symbol}, explain in simple terms:

1. What each indicator means.
2. What the values currently suggest.
3. Whether to BUY or SELL.
4. Suggested Entry, Stop Loss, and Take Profit (TP1 and TP2).
5. Confidence Level (1 to 10).
6. Do NOT recommend a trade if data is unclear.

Indicators:
- Current Price = {indicators['Current Price']}
- EMA12 = {indicators['EMA12']}
- EMA26 = {indicators['EMA26']}
- RSI = {indicators['RSI']}
- MACD = {indicators['MACD']} / Signal = {indicators['MACD_signal']}
- Stochastic %K = {indicators['Stoch_K']} / %D = {indicators['Stoch_D']}
- CCI = {indicators['CCI']}
- ADX = {indicators['ADX']}
- ATR = {indicators['ATR']}
- Bollinger High = {indicators['BB_high']}, Low = {indicators['BB_low']}
- OBV = {indicators['OBV']}
- CMF = {indicators['CMF']}
    """

    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        json_data = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }

        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=json_data)
        result = response.json()

        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ AI analysis failed: {e}"

# ========== Run Main Analysis ==========
def run_analysis():
    symbols = ["XAU/USD", "BTC/USD", "AAPL"]
    results = []

    for symbol in symbols:
        try:
            df = fetch_data(symbol)
            if isinstance(df, str):
                results.append(f"❌ Error analyzing {symbol}: {df}")
                continue

            indicators = compute_indicators(df, symbol)
            if "error" in indicators:
                results.append(f"❌ Error analyzing {symbol}: {indicators['error']}")
                continue

            analysis = analyze_with_ai(symbol, indicators)
            results.append(f"📈 **Symbol**: {symbol}\n{analysis}")

        except Exception as e:
            results.append(f"❌ Unexpected error for {symbol}: {e}")

    return "\n\n".join(results)
