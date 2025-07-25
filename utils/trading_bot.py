import pandas as pd
from ta import trend, momentum, volatility, volume
from twelve_data import get_data  # Your custom wrapper
from ai_model import get_ai_prediction  # Your AI logic (Groq/OpenAI/Local)
import numpy as np

def run_analysis(symbol: str):
    df = get_data(symbol, interval="1h", outputsize=100)
    if df is None or df.empty:
        return {"symbol": symbol, "ai_insight": "⚠️ No data available"}

    df.dropna(inplace=True)
    last = df.iloc[-1]
    close_price = last['close']

    # === INDICATORS ===
    df["EMA12"] = trend.ema_indicator(df["close"], window=12)
    df["EMA26"] = trend.ema_indicator(df["close"], window=26)
    df["RSI"] = momentum.rsi(df["close"], window=14)
    macd = trend.macd_diff(df["close"])
    df["MACD"] = macd
    stoch_k = momentum.stoch(df["high"], df["low"], df["close"])
    stoch_d = stoch_k.rolling(window=3).mean()
    df["Stoch_K"] = stoch_k
    df["Stoch_D"] = stoch_d
    df["CCI"] = momentum.cci(df["high"], df["low"], df["close"], window=20)
    df["ADX"] = trend.adx(df["high"], df["low"], df["close"])
    df["ATR"] = volatility.average_true_range(df["high"], df["low"], df["close"])
    try:
        df["CMF"] = volume.chaikin_money_flow(df["high"], df["low"], df["close"], df["volume"])
    except:
        df["CMF"] = None
    df["BB_upper"], df["BB_middle"], df["BB_lower"] = volatility.bollinger_hband(df["close"]), trend.sma_indicator(df["close"]), volatility.bollinger_lband(df["close"])

    # === READ LATEST VALUES ===
    latest = df.iloc[-1]
    trend_bias = "Bullish" if latest["EMA12"] > latest["EMA26"] else "Bearish"
    signal_strength = 0
    total_signals = 0

    # === SIGNAL SCORING ===
    if trend_bias == "Bullish" and latest["RSI"] > 50:
        signal_strength += 1
    if latest["MACD"] > 0:
        signal_strength += 1
    if latest["RSI"] < 30 or latest["RSI"] > 70:
        signal_strength += 1
    if latest["Stoch_K"] < 20 or latest["Stoch_K"] > 80:
        signal_strength += 1
    if latest["CCI"] < -100 or latest["CCI"] > 100:
        signal_strength += 1
    if latest["ADX"] > 25:
        signal_strength += 1
    if latest["ATR"] > np.mean(df["ATR"][-10:]):
        signal_strength += 1

    total_signals = 7
    confidence = round((signal_strength / total_signals) * 10, 1)

    # === AI Prediction (optional) ===
    prediction = get_ai_prediction(df[-50:]) if confidence >= 5 else "No action"

    # === Entry, SL, TP (example logic) ===
    entry = round(close_price, 2)
    if trend_bias == "Bullish":
        sl = round(entry - 2 * latest["ATR"], 2)
        tp1 = round(entry + 2 * latest["ATR"], 2)
        tp2 = round(entry + 4 * latest["ATR"], 2)
    else:
        sl = round(entry + 2 * latest["ATR"], 2)
        tp1 = round(entry - 2 * latest["ATR"], 2)
        tp2 = round(entry - 4 * latest["ATR"], 2)

    # === Final Signal Output ===
    if confidence < 8:
        insight = f"⚠️ No confirmed signal (Confidence: {confidence}/10)"
    else:
        insight = f"✅ Confirmed {trend_bias} signal — {prediction} (Confidence: {confidence}/10)"

    return {
        "symbol": symbol,
        "current_price": f"${entry}",
        "trend": trend_bias,
        "confidence": confidence,
        "prediction_1_2h": prediction,
        "prediction_3_4h": prediction,
        "prediction_1d": prediction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "ai_insight": insight,
        "indicators": {
            "EMA": f"{round(latest['EMA12'],2)} / {round(latest['EMA26'],2)}",
            "RSI": round(latest["RSI"], 2),
            "MACD": round(latest["MACD"], 2),
            "Stoch K/D": f"{round(latest['Stoch_K'],2)} / {round(latest['Stoch_D'],2)}",
            "CCI": round(latest["CCI"], 2),
            "ADX": round(latest["ADX"], 2),
            "ATR": round(latest["ATR"], 2),
            "CMF": None if pd.isna(latest["CMF"]) else round(latest["CMF"], 2),
            "Bollinger Bands": f"{round(latest['BB_lower'],2)} - {round(latest['BB_upper'],2)}"
        }
    }
