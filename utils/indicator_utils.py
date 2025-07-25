import ta

def analyze_indicators(df):
    try:
        df = df.copy()
        df["EMA12"] = ta.trend.ema_indicator(df["close"], window=12)
        df["EMA26"] = ta.trend.ema_indicator(df["close"], window=26)
        macd = ta.trend.macd(df["close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["RSI"] = ta.momentum.rsi(df["close"], window=14)
        stoch = ta.momentum.stoch(df["high"], df["low"], df["close"])
        df["Stoch_K"] = stoch.stoch()
        df["Stoch_D"] = stoch.stoch_signal()
        df["CCI"] = ta.trend.cci(df["high"], df["low"], df["close"], window=20)
        df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"])
        df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])
        bb = ta.volatility.BollingerBands(df["close"])
        df["Bollinger_Upper"] = bb.bollinger_hband()
        df["Bollinger_Lower"] = bb.bollinger_lband()

        latest = df.iloc[-1]
        return {
            "EMA12": latest["EMA12"],
            "EMA26": latest["EMA26"],
            "MACD": latest["MACD"],
            "MACD_Signal": latest["MACD_Signal"],
            "RSI": latest["RSI"],
            "Stoch_K": latest["Stoch_K"],
            "Stoch_D": latest["Stoch_D"],
            "CCI": latest["CCI"],
            "ADX": latest["ADX"],
            "ATR": latest["ATR"],
            "Bollinger_Upper": latest["Bollinger_Upper"],
            "Bollinger_Lower": latest["Bollinger_Lower"]
        }
    except Exception as e:
        print(f"❌ Indicator error: {e}")
        return None
