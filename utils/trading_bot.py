# utils/trading_bot.py

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
import ta
from dotenv import load_dotenv
from groq import Groq
import json

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def fetch_price():
    if not TWELVE_API_KEY:
        print("Error: TWELVE_API_KEY is not set in environment variables.")
        return None

    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&apikey={TWELVE_API_KEY}&outputsize=500"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        if 'values' not in data or not data['values']:
            print(f"Error: Twelve Data API returned no values. Response: {json.dumps(data, indent=2)}")
            return None

        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"Warning: Column '{col}' not found in fetched data.")

        df = df.dropna().sort_index()
        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def add_indicators(df):
    if df is None or df.empty or 'close' not in df.columns:
        return pd.DataFrame()

    try:
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        df['stoch_k'] = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3).stoch()
        df['stoch_d'] = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3).stoch_signal()
        df['mom'] = ta.momentum.ROCIndicator(close=df['close'], window=10).roc()  # Using ROC instead of MomentumIndicator
        df['cci'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20).cci()
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        df['ema12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
        df['ema26'] = ta.trend.EMAIndicator(close=df['close'], window=26).ema_indicator()
        df['ema50'] = ta.trend.EMAIndicator(close=df['close'], window=50).ema_indicator()
        df['sma20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
        df['adx'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).chaikin_money_flow()
        df = df.dropna()
        return df
    except Exception as e:
        print(f"❌ Indicator error: {e}")
        return pd.DataFrame()

def generate_chart(df):
    if df.empty:
        return
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Close Price', color='blue')
    plt.plot(df.index, df['ema12'], label='EMA 12', color='orange', linestyle='--')
    plt.plot(df.index, df['ema26'], label='EMA 26', color='purple', linestyle='--')
    plt.title("XAU/USD Price & EMAs")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/market_chart.png")
    plt.close()

def generate_prompt(df):
    if df.empty:
        return "No data available."

    latest = df.iloc[-1]
    def val(k): return f"{latest[k]:.2f}" if k in latest and pd.notna(latest[k]) else "N/A"

    details = f"""
    - Price: {val('close')}
    - RSI: {val('rsi')}
    - Stochastic K/D: {val('stoch_k')}, {val('stoch_d')}
    - Momentum (ROC): {val('mom')}
    - CCI: {val('cci')}
    - MACD: {val('macd')} | Signal: {val('macd_signal')} | Histogram: {val('macd_diff')}
    - EMA (12/26/50): {val('ema12')}, {val('ema26')}, {val('ema50')}
    - SMA (20/50): {val('sma20')}, {val('sma50')}
    - ADX: {val('adx')}
    - Bollinger Bands (High/Mid/Low): {val('bb_high')}, {val('bb_mid')}, {val('bb_low')}
    - ATR: {val('atr')}
    - OBV: {val('obv')}
    - CMF: {val('cmf')}
    """

    return f"""
    You are Chef Carthy, a veteran institutional forex analyst. Analyze XAU/USD on the 1-hour timeframe using **all technical indicators** above. Justify your prediction based on confluence between indicators. Be precise and professional.

    Format:
    - 📊 Market Analysis: Explain trend, volatility, momentum, and volume
    - 📈 Bias: BUY / SELL / HOLD (with reasons)
    - 📍 Entry:
    - 🛑 Stop Loss:
    - 🎯 Take Profit 1:
    - 🎯 Take Profit 2 (optional):
    - ✅ Summary & Confidence:
    
    Market Data:
    {details}
    """

def run_analysis():
    print("Starting analysis...")
    df = fetch_price()
    if df is None or df.empty:
        return "❌ Error: Could not fetch data."
    df = add_indicators(df)
    if df.empty:
        return "❌ Error: Failed to compute indicators."
    generate_chart(df)
    prompt = generate_prompt(df)

    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are Chef Carthy, a professional forex analyst giving detailed, high-accuracy advice using every indicator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error generating analysis: {e}"
