import requests
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import os

# === Configuration ===
API_KEY = os.getenv("TWELVE_API_KEY")
INTERVAL = "15min"
LIMIT = 100
SYMBOLS = ["XAU/USD", "BTC/USD", "AAPL"]  # Add more as needed


def fetch_data(symbol):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={INTERVAL}&outputsize={LIMIT}&apikey={API_KEY}&format=JSON"
    response = requests.get(url)
    data = response.json()
    if "values" not in data:
        raise Exception(f"Failed to fetch data for {symbol}: {data}")

    df = pd.DataFrame(data["values"])
    df.rename(columns={
        "datetime": "time",
        "close": "close",
        "open": "open",
        "high": "high",
        "low": "low",
        "volume": "volume"
    }, inplace=True)
    df["time"] = pd.to_datetime(df["time"])
    df.sort_values("time", inplace=True)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_indicators(df):
    try:
        df["EMA_12"] = trend.ema_indicator(df["close"], window=12)
        df["EMA_26"] = trend.ema_indicator(df["close"], window=26)
        df["EMA_50"] = trend.ema_indicator(df["close"], window=50)
        df["SMA_20"] = trend.sma_indicator(df["close"], window=20)
        df["SMA_50"] = trend.sma_indicator(df["close"], window=50)
        df["MACD"] = trend.macd(df["close"])
        df["MACD_Signal"] = trend.macd_signal(df["close"])
        df["RSI"] = momentum.rsi(df["close"], window=14)
        df["Stochastic_K"] = momentum.stoch(df["high"], df["low"], df["close"])
        df["Stochastic_D"] = df["Stochastic_K"].rolling(window=3).mean()
        df["CCI"] = trend.cci(df["high"], df["low"], df["close"], window=20)
        df["ADX"] = trend.adx(df["high"], df["low"], df["close"], window=14)

        bb = volatility.BollingerBands(df["close"], window=20, window_dev=2)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()
        df["ATR"] = volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

        # Volume indicators if available
        if "volume" in df.columns and df["volume"].notnull().all():
            df["OBV"] = volume.on_balance_volume(df["close"], df["volume"])
            df["CMF"] = volume.chaikin_money_flow(df["high"], df["low"], df["close"], df["volume"], window=20)
        else:
            df["OBV"] = np.nan
            df["CMF"] = np.nan

        return df

    except Exception as e:
        raise Exception("Failed to compute indicators: " + str(e))


def explain_signals(row, symbol):
    messages = []
    price = row["close"]
    ema_fast = row["EMA_12"]
    ema_slow = row["EMA_26"]
    macd = row["MACD"]
    macd_signal = row["MACD_Signal"]
    rsi = row["RSI"]
    adx = row["ADX"]
    atr = row["ATR"]
    bb_upper = row["BB_upper"]
    bb_lower = row["BB_lower"]
    cmf = row["CMF"]

    # Trend bias
    if ema_fast > ema_slow:
        trend_bias = "Bullish"
    elif ema_fast < ema_slow:
        trend_bias = "Bearish"
    else:
        trend_bias = "Neutral"

    # Signal interpretation
    if macd > macd_signal and rsi < 70:
        bias = "BUY"
    elif macd < macd_signal and rsi > 30:
        bias = "SELL"
    else:
        bias = "HOLD"

    # Price target estimation
    tp1 = price + 1.5 * atr if bias == "BUY" else price - 1.5 * atr
    tp2 = price + 2.5 * atr if bias == "BUY" else price - 2.5 * atr
    sl = price - 1.2 * atr if bias == "BUY" else price + 1.2 * atr

    messages.append(f"\n📈 **Symbol**: {symbol}")
    messages.append(f"📊 **Market Summary**")
    messages.append(f"Trend: {trend_bias}")
    messages.append(f"Momentum: {'Weak' if adx < 20 else 'Strong'}")
    messages.append(f"Volatility (ATR): {atr:.2f}")
    messages.append(f"Volume Flow: {'Positive' if cmf > 0 else 'Negative' if not pd.isna(cmf) else 'N/A'}")
    messages.append(f"\n🎯 **Bias**: {bias}")
    messages.append(f"🔹 Entry Price: {price:.2f}")
    messages.append(f"🔻 Stop Loss: {sl:.2f}")
    messages.append(f"🔺 Take Profit 1: {tp1:.2f}")
    messages.append(f"🔺 Take Profit 2: {tp2:.2f}")
    return "\n".join(messages)


def run_analysis():
    reports = []
    for symbol in SYMBOLS:
        try:
            df = fetch_data(symbol)
            df = compute_indicators(df)
            latest = df.iloc[-1]
            report = explain_signals(latest, symbol)
            reports.append(report)
        except Exception as e:
            reports.append(f"❌ Failed for {symbol}: {str(e)}")

    return "\n\n".join(reports)
