import pandas as pd
import numpy as np
import ta

def calculate_macd_rsi_bbands(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    return df

def detect_price_action(df):
    candles = []
    for i in range(2, len(df)):
        open_, close, high, low = df.loc[i, ['open', 'close', 'high', 'low']]
        body = abs(close - open_)
        range_ = high - low
        upper_shadow = high - max(open_, close)
        lower_shadow = min(open_, close) - low

        # Pin bar
        if lower_shadow > body * 2 and body < range_ * 0.3:
            candles.append('bullish_pin_bar')
        elif upper_shadow > body * 2 and body < range_ * 0.3:
            candles.append('bearish_pin_bar')
        # Doji
        elif body < range_ * 0.1:
            candles.append('doji')
        else:
            candles.append('normal')
    candles = ['normal', 'normal'] + candles
    df['candle_type'] = candles
    return df

def detect_break_retest(df, lookback=10):
    """
    Basic break-and-retest detection:
    - Breaks a resistance (recent high), then closes above it
    - Then returns near that level within next few candles (retest)
    """
    df['break_retest'] = 'none'
    for i in range(lookback, len(df) - 2):
        recent_high = df['high'][i - lookback:i].max()
        recent_low = df['low'][i - lookback:i].min()
        close = df.loc[i, 'close']
        next_close = df.loc[i + 1, 'close']

        # Bullish break and retest
        if close > recent_high and next_close < close and next_close >= recent_high:
            df.loc[i + 1, 'break_retest'] = 'bullish_retest'

        # Bearish break and retest
        elif close < recent_low and next_close > close and next_close <= recent_low:
            df.loc[i + 1, 'break_retest'] = 'bearish_retest'
    return df

def detect_trend_structure(df):
    """
    Trend approximation using EMA 20 vs EMA 50
    """
    df['trend'] = np.where(df['ema_20'] > df['ema_50'], 'uptrend', 'downtrend')
    return df

def analyze_indicators(df):
    df = calculate_macd_rsi_bbands(df)
    df = detect_price_action(df)
    df = detect_break_retest(df)
    df = detect_trend_structure(df)
    return df
