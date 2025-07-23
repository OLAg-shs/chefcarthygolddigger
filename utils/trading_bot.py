# enhanced_ai_trading_bot.py

import requests
import pandas as pd
import numpy as np
import ta
import datetime
from ta.trend import MACD, SMAIndicator, EMAIndicator, CCIIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, StochRSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from ta.others import DailyReturnIndicator
from typing import Dict


class EnhancedTradingBot:
    def __init__(self, symbol: str, timeframe: str = '1h', limit: int = 500):
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.limit = limit
        self.df = pd.DataFrame()

    def fetch_data(self):
        url = f"https://api.twelvedata.com/time_series?symbol={self.symbol}&interval={self.timeframe}&outputsize={self.limit}&apikey=demo"
        r = requests.get(url)
        if 'values' not in r.json():
            raise ValueError("Error fetching data from TwelveData")
        df = pd.DataFrame(r.json()['values'])
        df = df.rename(columns={"datetime": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
        df = df.sort_values("date")
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        self.df = df.reset_index(drop=True)

    def add_indicators(self):
        df = self.df.copy()
        df['rsi'] = RSIIndicator(df['close']).rsi()
        df['macd'] = MACD(df['close']).macd_diff()
        df['ema_12'] = EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = EMAIndicator(df['close'], window=26).ema_indicator()
        df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
        df['cci'] = CCIIndicator(df['high'], df['low'], df['close']).cci()
        df['adx'] = ADXIndicator(df['high'], df['low'], df['close']).adx()
        df['stochastic_k'] = StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        df['roc'] = ROCIndicator(df['close']).roc()
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        df['bollinger_h'] = BollingerBands(df['close']).bollinger_hband()
        df['bollinger_l'] = BollingerBands(df['close']).bollinger_lband()
        df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['cmf'] = ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()

        self.df = df

    def analyze(self) -> Dict:
        row = self.df.iloc[-1]
        result = {
            "symbol": self.symbol,
            "current_price": row['close'],
            "trend": "Bullish" if row['ema_12'] > row['ema_26'] else "Bearish",
            "momentum": "Strong" if row['macd'] > 0 else "Weak",
            "volatility": row['atr'],
            "volume_trend": "High" if row['obv'] > 0 else "Low",
            "rsi": row['rsi'],
            "macd": row['macd'],
            "cci": row['cci'],
            "adx": row['adx'],
            "stochastic": row['stochastic_k'],
            "roc": row['roc'],
            "bollinger_signal": "Overbought" if row['close'] > row['bollinger_h'] else ("Oversold" if row['close'] < row['bollinger_l'] else "Normal"),
            "bias": "BUY" if row['ema_12'] > row['ema_26'] and row['rsi'] > 50 and row['macd'] > 0 else "SELL",
            "confidence": self.calculate_confidence(row)
        }
        return result

    def calculate_confidence(self, row) -> int:
        score = 0
        if row['ema_12'] > row['ema_26']: score += 1
        if row['macd'] > 0: score += 1
        if row['rsi'] > 50: score += 1
        if row['adx'] > 20: score += 1
        if row['cci'] > 0: score += 1
        if row['stochastic_k'] > 50: score += 1
        if row['roc'] > 0: score += 1
        if row['cmf'] > 0: score += 1
        return round((score / 8) * 10)

    def run(self):
        try:
            self.fetch_data()
            self.add_indicators()
            return self.analyze()
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    bot = EnhancedTradingBot("XAU/USD", "1h")
    analysis = bot.run()
    print("\nAI Trading Insight:\n")
    if 'error' in analysis:
        print(f"❌ Error: {analysis['error']}")
    else:
        print(f"📍 Symbol: {analysis['symbol']}")
        print(f"💰 Current Price: {analysis['current_price']:.2f}")
        print(f"📈 Trend: {analysis['trend']}, Momentum: {analysis['momentum']}, Volatility (ATR): {analysis['volatility']:.2f}, Volume: {analysis['volume_trend']}")
        print(f"📊 RSI: {analysis['rsi']:.2f}, MACD: {analysis['macd']:.2f}, CCI: {analysis['cci']:.2f}, ADX: {analysis['adx']:.2f}, Stochastic %K: {analysis['stochastic']:.2f}, ROC: {analysis['roc']:.2f}")
        print(f"📉 Bollinger Signal: {analysis['bollinger_signal']}")
        print(f"📍 Bias: {analysis['bias']}, Confidence Level: {analysis['confidence']}/10")
