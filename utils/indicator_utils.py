import pandas as pd
import numpy as np
import ta

def compute_indicators(df):
    try:
        # Ensure proper column names
        df.columns = [col.lower() for col in df.columns]

        # Technical Indicators
        df['ema12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema26'] = ta.trend.ema_indicator(df['close'], window=26)
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['stoch_rsi'] = ta.momentum.stochrsi_k(df['close'])
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        boll = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['boll_upper'] = boll.bollinger_hband()
        df['boll_lower'] = boll.bollinger_lband()
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)

        return df.dropna()
    except Exception as e:
        print(f"❌ Error computing indicators: {e}")
        return df


def detect_price_action(df):
    try:
        latest = df.iloc[-2:]  # Use last two candles for pattern detection
        signals = {
            "pin_bar": False,
            "doji": False,
            "break_retest": False,
            "trend_support": False,
        }

        candle = latest.iloc[-1]
        body = abs(candle["close"] - candle["open"])
        upper_wick = candle["high"] - max(candle["close"], candle["open"])
        lower_wick = min(candle["close"], candle["open"]) - candle["low"]

        # ✅ Pin Bar (long wick, small body)
        if body < (upper_wick + lower_wick) and (upper_wick > body * 2 or lower_wick > body * 2):
            signals["pin_bar"] = True

        # ✅ Doji: very small body
        if body < 0.1 * (candle["high"] - candle["low"]):
            signals["doji"] = True

        # ✅ Break & Retest (price near recent high or low)
        if len(df) >= 10:
            prev_highs = df["high"].iloc[-10:-2]
            prev_lows = df["low"].iloc[-10:-2]
            recent_high = max(prev_highs)
            recent_low = min(prev_lows)
            current_price = candle["close"]

            if abs(current_price - recent_high) / recent_high < 0.01:
                signals["break_retest"] = True
            elif abs(current_price - recent_low) / recent_low < 0.01:
                signals["break_retest"] = True

        # ✅ Trend Structure: Higher high & higher low
        if df["high"].iloc[-1] > df["high"].iloc[-5] and df["low"].iloc[-1] > df["low"].iloc[-5]:
            signals["trend_support"] = True

        return signals
    except Exception as e:
        print(f"❌ Price Action Detection error: {e}")
        return {
            "pin_bar": False,
            "doji": False,
            "break_retest": False,
            "trend_support": False,
        }
