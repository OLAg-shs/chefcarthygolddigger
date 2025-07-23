# ai_trading_bot.py (Upgraded with Pro-Level Indicator Suite)

import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import requests
import ta
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# === 1. Fetch Market Data ===
def fetch_price():
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&apikey={os.getenv('TWELVE_API_KEY')}"
    data = requests.get(url).json()
    df = pd.DataFrame(data['values'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df.astype(float).sort_index()
    return df

# === 2. Add All Professional Indicators ===
def add_indicators(df):
    # Trend indicators
    df['ema12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema26'] = ta.trend.ema_indicator(df['close'], window=26)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
    df['sma50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['sma200'] = ta.trend.sma_indicator(df['close'], window=200)

    # Volatility
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()

    # Momentum
    df['rsi'] = ta.momentum.rsi(df['close'])
    df['macd'] = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])

    # Volume
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['accdist'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])

    # Trend strength
    adx = ta.trend.adx(df['high'], df['low'], df['close'])
    df['adx'] = adx
    df['adx_pos'] = ta.trend.adx_pos(df['high'], df['low'], df['close'])
    df['adx_neg'] = ta.trend.adx_neg(df['high'], df['low'], df['close'])

    # Aroon
    df['aroon_up'] = ta.trend.aroon_up(df['close'])
    df['aroon_down'] = ta.trend.aroon_down(df['close'])

    return df

# === 3. Generate Market Chart ===
def generate_chart(df):
    plt.figure(figsize=(10, 5))
    df[['close', 'ema12', 'ema26', 'ema50', 'ema200']].plot(ax=plt.gca())
    plt.title("XAU/USD Price with EMAs")
    plt.savefig("static/market_chart.png")
    plt.close()

# === 4. Generate Prompt for Groq AI ===
def generate_prompt(df):
    latest = df.iloc[-1]
    return f"""
You are a professional financial analyst. Here is the latest gold (XAU/USD) data:

Price: {latest['close']:.2f}

Trend:
- EMA12: {latest['ema12']:.2f}, EMA26: {latest['ema26']:.2f}, EMA50: {latest['ema50']:.2f}, EMA200: {latest['ema200']:.2f}
- SMA50: {latest['sma50']:.2f}, SMA200: {latest['sma200']:.2f}
- Aroon Up: {latest['aroon_up']:.2f}, Aroon Down: {latest['aroon_down']:.2f}
- ADX: {latest['adx']:.2f} (DI+: {latest['adx_pos']:.2f}, DI-: {latest['adx_neg']:.2f})

Momentum:
- RSI: {latest['rsi']:.2f}
- MACD: {latest['macd']:.2f} | Signal: {latest['macd_signal']:.2f}
- Stochastic: {latest['stoch']:.2f}

Volume:
- OBV: {latest['obv']:.2f}
- Accumulation/Distribution: {latest['accdist']:.2f}

Volatility:
- Bollinger Bands: High={latest['bb_high']:.2f}, Low={latest['bb_low']:.2f}, Mid={latest['bb_middle']:.2f}

Using all the above, give a crystal-clear market analysis:
- Identify the current trend (bullish/bearish/consolidation)
- Entry decision: Buy/Sell/Hold
- Suggest an ideal stop loss and take profit
- Justify clearly using indicator alignments
- Explain in beginner-friendly language
"""

# === 5. Run Full AI-Powered Analysis ===
def run_analysis():
    df = fetch_price()
    df = add_indicators(df)
    generate_chart(df)
    prompt = generate_prompt(df)
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content
