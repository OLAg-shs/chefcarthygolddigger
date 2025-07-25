import sys
import os
import time

# ✅ Add root and utils/ folder to Python path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "utils"))

# ✅ Import functions directly from utils/ modules
from twelve_data import get_data
from indicator_utils import (
    analyze_indicators,
    detect_price_action,
    detect_break_retest
)
from groq_api import get_groq_prediction

# ✅ Symbols to analyze
SYMBOLS = ["XAU/USD", "BTC/USD", "AAPL", "EUR/USD"]

def run_analysis():
    insights = []

    for symbol in SYMBOLS:
        try:
            # 📊 Fetch latest OHLCV data (1H timeframe)
            df = get_data(symbol, interval="1h", outputsize=100)
            if df is None or df.empty:
                insights.append({
                    "symbol": symbol,
                    "text": f"❌ No data for {symbol}",
                })
                continue

            # 📈 Analyze technical indicators
            indicators = analyze_indicators(df)

            # 🔍 Detect price action signals
            price_action = detect_price_action(df)

            # 🧱 Detect break & retest pattern
            retest = detect_break_retest(df)

            # 🤖 Get AI-based prediction from Groq
            ai_result = get_groq_prediction(symbol, indicators, price_action, retest)

            # ✅ Store insight
            insights.append({
                "symbol": symbol,
                "text": ai_result.get("explanation", "No explanation"),
                "bias": ai_result.get("bias", "Unknown"),
                "confidence": ai_result.get("confidence", 0),
                "entry": ai_result.get("entry", "N/A"),
                "sl": ai_result.get("sl", "N/A"),
                "tp1": ai_result.get("tp1", "N/A"),
                "tp2": ai_result.get("tp2", "N/A"),
                "current_price": df["close"].iloc[-1]
            })

            # ⏳ Sleep to avoid API rate-limiting
            time.sleep(8)

        except Exception as e:
            insights.append({
                "symbol": symbol,
                "text": f"⚠️ Error with {symbol}: {e}",
            })

    return insights
