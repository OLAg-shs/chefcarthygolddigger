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
import json # Import json for detailed error printing

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY") # Ensure this is also loaded

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

def fetch_price():
    """Fetches XAU/USD (Gold) 1-hour price data from Twelve Data API."""
    if not TWELVE_API_KEY:
        print("Error: TWELVE_API_KEY is not set in environment variables.")
        # Returning a clear error message that the higher-level function can display
        return None # Indicate a critical failure to fetch data

    # Changed outputsize to a more reasonable number for free tier and indicator calculation
    # 200 is generally sufficient for common indicators like EMA50/200, RSI, MACD
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&apikey={TWELVE_API_KEY}&outputsize=200"
    
    try:
        # Increased timeout slightly for potentially larger data fetch
        response = requests.get(url, timeout=15)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # More robust check for 'values' and potential API error messages
        if 'values' not in data or not data['values']:
            error_message = data.get('message', 'No specific error message provided by API.')
            print(f"Error: Twelve Data API returned no values or an error. Response: {json.dumps(data, indent=2)}")
            print(f"API Message: {error_message}")
            return None # Indicate failure to fetch valid data

        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        # Define the columns we expect and want to convert to numeric
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Iterate only over columns that actually exist in the DataFrame
        for col in numeric_cols:
            if col in df.columns: # Check if the column exists before trying to convert
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"Warning: Column '{col}' not found in fetched data. It will be skipped.")
                # If 'volume' is missing, it will now simply be skipped and not cause a crash.
                # The prompt generator might show 'N/A' for volume if it's missing.

        # Drop rows with NaN values (from coercion or incomplete data) and sort
        df = df.dropna().sort_index() 
        
        if df.empty:
            print("Fetched an empty DataFrame after processing (e.g., all data was NaN).")
            return None # Indicate failure to get usable data

        # Check if enough data is available after all cleaning for indicator calculation
        # Most indicators need at least 26 (for EMA26/MACD) or 50 (for EMA50) periods.
        # Ensure sufficient data for reliable indicator calculation and analysis
        if len(df) < 50: 
            print(f"Warning: Fetched only {len(df)} data points, which might be insufficient for all indicators.")
            # Still return df, but the calling function might handle this if it's too few

        return df
    
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error fetching market data ({e.response.status_code}): {e.response.text}")
        if e.response.status_code == 429:
            print("This is a Rate Limit error. You've made too many requests. Please wait a minute or check your Twelve Data plan limits.")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"Network Connection Error fetching market data: {e}. Check your internet connection.")
        return None
    except requests.exceptions.Timeout:
        print("Error: The request to Twelve Data API timed out (took too long).")
        return None
    except requests.exceptions.RequestException as e:
        print(f"An unexpected Request Error occurred: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: Could not parse API response. {e}")
        # Print the raw response content to help debug malformed JSON
        print(f"Problematic response content: {response.text[:500]}...") # Print first 500 chars
        return None
    except Exception as e:
        print(f"An unknown error occurred during data fetching: {e}")
        return None


def add_indicators(df):
    """Adds various professional technical indicators to the DataFrame."""
    # Adjusted minimum length check to be consistent with fetch_price's warning
    if df is None or df.empty or 'close' not in df.columns or len(df) < 50: 
        print(f"DataFrame is empty or has insufficient data ({len(df) if df is not None else 0} rows) for reliable indicator calculation.")
        return pd.DataFrame() # Return empty DataFrame if cannot calculate indicators

    # Ensure float type before calculation to prevent errors
    # Check if column exists before trying to convert
    for col in ['open', 'high', 'low', 'close']: # 'volume' is handled in fetch_price more robustly if it's missing entirely
        if col in df.columns:
            df[col] = df[col].astype(float) 

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
    # Ensure at least one row remains after dropna for 'latest' access in generate_prompt
    df = df.dropna() 
    
    if df.empty:
        print("DataFrame became empty after dropping NaNs from indicator calculation.")
        return pd.DataFrame()
        
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
    
    # Helper to safely get indicator values, ensuring formatting and handling of missing data
    def get_indicator_val(key):
        if key in latest and pd.notna(latest[key]):
            # Format volume as integer if it's available and numeric, otherwise float
            if key == 'volume':
                return f"{int(latest[key]):,}" if latest[key] == int(latest[key]) else f"{latest[key]:,.0f}"
            return f"{latest[key]:.2f}"
        return "N/A"

    prompt_details = f"""
    - **Current Price**: {get_indicator_val('close')}
    - **RSI (14)**: {get_indicator_val('rsi')} (Overbought >70, Oversold <30, Mid-range 30-70)
    - **Stochastic Oscillator (%K, %D)**: {get_indicator_val('stoch_k')}, {get_indicator_val('stoch_d')} (Overbought >80, Oversold <20, Crossover signals)
    - **MACD Line**: {get_indicator_val('macd')}
    - **MACD Signal Line**: {get_indicator_val('macd_signal')}
    - **MACD Histogram**: {get_indicator_val('macd_diff')} (Momentum: above zero and rising = bullish, below zero and falling = bearish)
    - **EMA 12**: {get_indicator_val('ema12')} (Short-term trend)
    - **EMA 26**: {get_indicator_val('ema26')} (Medium-term trend)
    - **EMA 50**: {get_indicator_val('ema50')} (Longer-term trend)
    - **Bollinger Bands (Upper, Middle, Lower)**: U:{get_indicator_val('bb_high')}, M:{get_indicator_val('bb_mid')}, L:{get_indicator_val('bb_low')} (Volatility & potential reversal points)
    - **Average True Range (ATR 14)**: {get_indicator_val('atr')} (Measure of volatility, useful for stop loss placement)
    - **Volume**: {get_indicator_val('volume')} (Confirmation of price moves)
    """

    return f"""
    You are Chef Carthy, an **elite, highly experienced, and conservative institutional forex trading analyst** specializing in XAU/USD (Gold) using technical analysis. Your primary goal is to provide **actionable, clear, and profitable trading recommendations** for a 1-hour time frame based on the provided data, considering a risk-averse approach.

    **Current 1-Hour Chart Data for XAU/USD (Gold):**
    {prompt_details}

    **Based on this comprehensive data, perform a professional analysis. Your output must be highly detailed, explain your reasoning by explicitly referencing the NUMERICAL VALUES of the indicators, their significance, and how they show confluence. Provide your output in the following structured format, ensuring all recommendations are specific price levels and explanations are beginner-friendly yet technically sound.**

    ---
    ### 📈 Market Analysis & Trend Prediction

    * **Current Market Outlook**: Describe the current market sentiment and its driving factors. **Explicitly analyze the confluence of at least three key indicators by citing their current numerical values and what they individually imply (e.g., "RSI is {get_indicator_val('rsi')}, indicating [overbought/oversold/neutral] conditions; concurrently, MACD Line is at {get_indicator_val('macd')} crossing [above/below] its Signal Line at {get_indicator_val('macd_signal')}, suggesting [momentum shift/continuation]; price at {get_indicator_val('close')} is positioned [above/below/at] the EMA 26 at {get_indicator_val('ema26')}, confirming [bullish/bearish/sideways] sentiment.").**
    * **Short-Term Trend (next 1-4 hours)**: State the predicted trend (Uptrend, Downtrend, Sideways). **Justify your prediction by detailing the interaction of short-term EMAs (EMA 12 at {get_indicator_val('ema12')}, EMA 26 at {get_indicator_val('ema26')}), price action relative to these EMAs, and momentum indicators like MACD ({get_indicator_val('macd_diff')} for histogram) and Stochastic ({get_indicator_val('stoch_k')}, {get_indicator_val('stoch_d')}).**
    * **Mid-Term Trend (next 1-3 days)**: State the predicted trend. **Analyze the broader market context using EMA 50 ({get_indicator_val('ema50')}) and Bollinger Bands (Middle Band at {get_indicator_val('bb_mid')}), explaining how current price action and indicator values support this mid-term view.**

    ---
    ### 📊 Trading Strategy & Actionable Plan

    * **Recommended Bias**: (BUY / SELL / HOLD - **choose one and justify based on your analysis**).
    * **Entry Price (Target)**: [Specific Price e.g., $2350.50] - State the exact price for ideal entry. **Explain *why* this specific price is chosen, linking it to direct technical levels (e.g., "Entry at [Price] near the EMA 12 at {get_indicator_val('ema12')}," or "retest of the lower Bollinger Band at {get_indicator_val('bb_low')}").**
    * **Stop Loss (Mandatory)**: [Specific Price e.g., $2345.00] - State the exact price for limiting losses. **Explain *why* this level is chosen by calculating it relative to the current ATR ({get_indicator_val('atr')}) or placing it logically beyond a key support/resistance level (e.g., "Placed below the EMA 50 at {get_indicator_val('ema50')} to protect against a trend reversal," or "a multiple of ATR away from entry").**
    * **Take Profit Target 1**: [Specific Price e.g., $2365.00] - State the exact price for the first profit target. **Explain *why* this level is chosen, linking it to previous resistance, a Bollinger Band ({get_indicator_val('bb_high')}), or a favorable risk-reward ratio (e.g., "Targeting the upper Bollinger Band at {get_indicator_val('bb_high')}").**
    * **Take Profit Target 2 (Optional)**: [Specific Price e.g., $2380.00] - State the exact price for a second, more ambitious target. **Justify this by potential trend extension or next major resistance.**

    ---
    ### 📈 Summary & Final Prediction

    * **Overall Market Summary**: Provide a concise summary of the current market state and your primary conclusion. Reiterate the key indicator signals that drive your recommendation.
    * **Final Prediction & Confidence**: Based on all the analysis, state your final prediction for XAU/USD in the near term (next 1-4 hours) and mid-term (next 1-3 days). Express a level of confidence (e.g., "High Confidence," "Moderate Confidence").
    * **Important Considerations for Traders**: Reiterate the importance of risk management (stop loss), emotional discipline, and adapting to new market data.

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
    
    if df is None or df.empty: # Check for None or empty DataFrame from fetch_price
        return "❌ Error: Could not fetch sufficient market data for analysis. Please check API keys, network connection, or Twelve Data limits."
    
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