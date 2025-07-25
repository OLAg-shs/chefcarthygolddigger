import time
from utils.twelve_data import get_data
from utils.indicator_utils import (
    analyze_indicators,
    detect_price_action,
    detect_break_retest
)
from utils.groq_api import get_groq_prediction  # ✅ This works if file exists in utils/

# Symbols to analyze
SYMBOLS = ["XAU/USD", "BTC/USD", "AAPL", "EUR/USD"]

def run_analysis():
    insights = []

    for symbol in SYMBOLS:
        try:
            # Fetch latest OHLCV data (1H)
            df = get_data(symbol, interval="1h", outputsize=100)
            if df is None or df.empty:
                insights.append({
                    "symbol": symbol,
                    "text": f"❌ No data for {symbol}",
                })
                continue

            # Analyze indicators
            indicators = analyze_indicators(df)

            # Detect price action patterns
            price_action = detect_price_action(df)

            # Detect break & retest logic
            retest = detect_break_retest(df)

            # Get AI-based signal prediction
            ai_result = get_groq_prediction(symbol, indicators, price_action, retest)

            # Append AI insight
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

            # Sleep to avoid rate-limiting (TwelveData free tier)
            time.sleep(8)

        except Exception as e:
            insights.append({
                "symbol": symbol,
                "text": f"⚠️ Error with {symbol}: {e}",
            })

    return insights
