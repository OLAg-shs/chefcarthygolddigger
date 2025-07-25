import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")

# Base API endpoint
BASE_URL = "https://api.twelvedata.com/time_series"

def get_data(symbol, interval="1h", outputsize=100):
    symbol_map = {
        "XAU/USD": "XAU/USD",
        "BTC/USD": "BTC/USD",
        "AAPL": "AAPL",
        "EUR/USD": "EUR/USD"
    }

    mapped_symbol = symbol_map.get(symbol, symbol)

    params = {
        "symbol": mapped_symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_API_KEY,
        "format": "JSON"
    }

    try:
        response = requests.get(BASE_URL, params=params)
        if response.status_code != 200:
            print(f"[ERROR] Failed to fetch data for {symbol}: HTTP {response.status_code}")
            return None

        data = response.json()

        if "values" not in data:
            print(f"[ERROR] Invalid data received for {symbol}: {data}")
            return None

        # Build DataFrame
        df = pd.DataFrame(data["values"])
        df = df.rename(columns={
            "datetime": "time",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        })

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")

        # Convert numeric columns safely
        float_cols = ["open", "high", "low", "close"]
        if "volume" in df.columns:
            float_cols.append("volume")
        else:
            print(f"[INFO] Volume not available for {symbol} — skipping volume.")

        df[float_cols] = df[float_cols].astype(float)

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"[ERROR] Exception occurred while fetching data for {symbol}: {e}")
        return None
