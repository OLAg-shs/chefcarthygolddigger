# utils/trading_bot.py

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib

import matplotlib.pyplot as plt
import requests
import ta
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY") # Ensure this is also loaded

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

def fetch_price():
    """Fetches XAU/USD (Gold) 1-hour price data from Twelve Data API."""
    if not TWELVE_API_KEY:
        print("Warning: TWELVE_API_KEY is not set in environment variables.")
        return pd.DataFrame() # Return empty DataFrame if key is missing

    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&apikey={TWELVE_API_KEY}&outputsize=100" # Request more data points for indicators
    
    try:
        response = requests.get(url, timeout=10) # Add a timeout
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if 'values' not in data or not data['values']:
            print(f"Error fetching data or no values returned: {data.get('message', 'No values key in response')}")
            return pd.DataFrame() # Return empty DataFrame on error

        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        # Convert numeric columns to float, coercing errors to NaN
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().sort_index() # Drop rows with NaN values that resulted from coercion
        
        if df.empty:
            print("Fetched an empty DataFrame after processing.")
        return df
    except requests.exceptions.Timeout:
        print("Error: The request to Twelve Data API timed out.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"Network or API request error: {e}")
        return pd.DataFrame()
    except ValueError as e:
        print(f"JSON decoding error or data processing error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred during data fetching: {e}")
        return pd.DataFrame()


def add_indicators(df):
    """Adds various professional technical indicators to the DataFrame."""
    if df.empty or 'close' not in df.columns or len(df) < 50: # Ensure enough data for indicators
        print(f"DataFrame is empty or has insufficient data ({len(df)} rows) for reliable indicator calculation.")
        return df

    # Ensure float type before calculation to prevent errors
    df['close'] = df['close'].astype(float) 
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)

    # 1. Momentum Indicators
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi() # Standard 14-period RSI
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # 2. Trend Indicators
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9) # Standard MACD settings
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff() # Histogram

    df['ema12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    df['ema26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator() # Longer-term trend

    # 3. Volatility Indicators
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2) # Standard BB settings
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range() # Standard 14-period ATR

    # Drop any rows that now have NaN due to indicator calculation (e.g., first few rows)
    df = df.dropna() 
    return df

def generate_chart(df):
    """Generates and saves a market chart (Price & EMAs) to the static folder."""
    if df.empty or 'close' not in df.columns or 'ema12' not in df.columns or 'ema26' not in df.columns:
        print("Not enough data or required columns for chart generation.")
        return # Exit if chart can't be generated

    plt.figure(figsize=(12, 6)) # Slightly larger for better readability
    plt.plot(df.index, df['close'], label='Close Price', color='blue', linewidth=1.5)
    plt.plot(df.index, df['ema12'], label='EMA 12', color='orange', linestyle='--', linewidth=1)
    plt.plot(df.index, df['ema26'], label='EMA 26', color='purple', linestyle='--', linewidth=1)
    
    plt.title("XAU/USD (Gold) Price & EMAs (1-Hour Chart)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price ($)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right') # Rotate dates for better display
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    
    # Ensure the 'static' directory exists
    os.makedirs("static", exist_ok=True)
    chart_path = os.path.join("static", "market_chart.png")
    plt.savefig(chart_path)
    plt.close() # Close the plot to free up memory

def generate_prompt(df):
    """Generates a detailed, professional prompt for the AI based on the latest market data and indicators."""
    if df.empty:
        return "No market data available for analysis. Cannot generate a professional prompt."

    latest = df.iloc[-1]
    
    # Helper to safely get indicator values
    def get_indicator_val(key):
        return f"{latest[key]:.2f}" if key in latest and pd.notna(latest[key]) else "N/A"

    prompt_details = f"""
    - **Current Price**: {get_indicator_val('close')}
    - **RSI (14)**: {get_indicator_val('rsi')} (Overbought >70, Oversold <30)
    - **Stochastic Oscillator (%K, %D)**: {get_indicator_val('stoch_k')}, {get_indicator_val('stoch_d')} (Overbought >80, Oversold <20)
    - **MACD Line**: {get_indicator_val('macd')}
    - **MACD Signal Line**: {get_indicator_val('macd_signal')}
    - **MACD Histogram**: {get_indicator_val('macd_diff')}
    - **EMA 12**: {get_indicator_val('ema12')}
    - **EMA 26**: {get_indicator_val('ema26')}
    - **EMA 50**: {get_indicator_val('ema50')}
    - **Bollinger Bands (Upper, Middle, Lower)**: {get_indicator_val('bb_high')}, {get_indicator_val('bb_mid')}, {get_indicator_val('bb_low')}
    - **Average True Range (ATR 14)**: {get_indicator_val('atr')} (Measure of volatility)
    - **Volume**: {get_indicator_val('volume')}
    """

    return f"""
    You are Chef Carthy, an **elite, highly experienced, and conservative institutional forex trading analyst** specializing in XAU/USD (Gold) using technical analysis. Your primary goal is to provide **actionable, clear, and profitable trading recommendations** for a 1-hour time frame based on the provided data, considering a risk-averse approach.

    **Current 1-Hour Chart Data for XAU/USD:**
    {prompt_details}

    **Based on this data, perform a professional analysis using price action, indicator confluence, and prudent risk management. Provide your output in the following structured format, ensuring all recommendations are specific price levels and explanations are beginner-friendly:**

    ---
    ### 📈 Market Analysis & Trend Prediction
    * **Current Market Outlook**: Describe the current market sentiment (e.g., strong bullish, bearish correction, sideways consolidation) and its driving factors based on the indicators.
    * **Short-Term Trend (next 1-4 hours)**: State the predicted trend (e.g., Uptrend, Downtrend, Sideways) and the primary reasons (e.g., EMA crossovers, RSI position, MACD momentum).
    * **Mid-Term Trend (next 1-3 days)**: State the predicted trend, considering broader movements and EMA50.

    ---
    ### 📊 Trading Strategy & Actionable Plan
    * **Recommended Bias**: (BUY / SELL / HOLD - choose one)
    * **Entry Price (Target)**: [Specific Price e.g., $2350.50] - State the exact price where a trade should ideally be entered. Explain *why* this is a good entry point (e.g., retest of support, breakout confirmation, indicator signal).
    * **Stop Loss (Mandatory)**: [Specific Price e.g., $2345.00] - State the exact price where the trade should be closed to limit losses. Explain *why* this level is chosen (e.g., below key support, outside volatility range).
    * **Take Profit Target 1**: [Specific Price e.g., $2365.00] - State the exact price for the first profit target. Explain *why* this level is chosen (e.g., previous resistance, 1:2 risk-reward).
    * **Take Profit Target 2 (Optional)**: [Specific Price e.g., $2380.00] - State the exact price for a second, more ambitious profit target if conditions allow.

    ---
    ### 🎓 Key Insights & Beginner-Friendly Explanation
    * **Why this Trade?**: Break down the core reasons for the recommended bias and entry point in simple, clear terms, referencing the indicators you used. For example, "The RSI is oversold, suggesting a potential bounce," or "Price is testing a strong support level, making it a good area for buyers."
    * **Understanding Risk Management**: Emphasize the importance of the Stop Loss and never trading without one. Explain how the Stop Loss helps protect capital.
    * **Final Precaution**: A brief disclaimer about market volatility and the importance of continuous learning.

    ---
    """

# THIS IS THE ORCHESTRATION FUNCTION your app.py imports
def run_analysis():
    """
    Orchestrates the data fetching, indicator calculation, chart generation,
    and AI insight generation for the trading bot.
    Returns the AI's generated trading insight.
    """
    print("Starting professional trading analysis...")
    df = fetch_price()
    
    if df.empty:
        return "❌ Error: Could not fetch sufficient market data for analysis. Please check API keys and network connection."
    
    df = add_indicators(df)
    
    # Check again after adding indicators, as dropna might make it empty
    if df.empty or 'close' not in df.columns or len(df) < 30: # Ensure enough data for a meaningful latest row
        return "❌ Error: Not enough valid data points after indicator calculation for meaningful analysis."

    generate_chart(df) # Generate the chart after indicators are added
    
    prompt = generate_prompt(df)
    
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192", # This model is suitable for detailed text generation
            messages=[
                {"role": "system", "content": "You are Chef Carthy, an expert institutional forex trading analyst providing precise, actionable, and beginner-friendly advice on XAU/USD. Your responses must be highly structured as requested by the user, using specific price points for entry, stop loss, and take profit, and clear explanations based on technical indicators and price action."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, # A bit higher temperature for more detailed analysis, but not too wild
            max_tokens=800, # Increased max_tokens to allow for detailed analysis
            top_p=0.9 # Adds a bit more diversity without going off-topic
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error communicating with AI: {e}")
        return f"❌ Error: Could not generate AI analysis. Reason: {str(e)}. Please check your Groq API key and network connection."