import os
import openai  # Used to call Groq's LLaMA3 endpoint (mocked as OpenAI-compatible)
from dotenv import load_dotenv

# Load .env variables (for local dev)
load_dotenv()

# Groq or OpenAI-compatible API key
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"
openai.api_type = "openai"
openai.api_version = None

# Groq Model
MODEL = "llama3-8b-8192"

def get_groq_prediction(symbol, indicators, price_action, retest):
    """
    Sends technical analysis data to Groq's AI to get a final prediction.
    """

    prompt = f"""
You are an elite trading assistant. Analyze the following market data for {symbol}.

### Technical Indicators:
{indicators}

### Price Action:
{price_action}

### Break and Retest Analysis:
{retest}

1. Based on this analysis, is it a good time to BUY, SELL, or HOLD?
2. What is the confidence level (scale of 1–10)?
3. What entry price should I look for?
4. Suggest a Stop Loss (SL).
5. Suggest 2 Take Profit levels (TP1 and TP2).
6. Explain the rationale clearly using the indicator results.

Only return your answer as a JSON dictionary like this:

{{
  "bias": "Buy" or "Sell" or "Hold",
  "confidence": 1–10,
  "entry": float,
  "sl": float,
  "tp1": float,
  "tp2": float,
  "explanation": "Your explanation here"
}}
"""

    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a precise and experienced trading AI that gives only high-confidence signals."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        ai_reply = response['choices'][0]['message']['content']

        # Try to safely evaluate the AI response as dictionary
        import json
        result = json.loads(ai_reply)
        return result

    except Exception as e:
        print(f"[⚠️ Groq API Error]: {e}")
        # Fallback dummy data
        return {
            "bias": "Hold",
            "confidence": 5,
            "entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "explanation": "⚠️ Could not fetch prediction due to error."
        }
